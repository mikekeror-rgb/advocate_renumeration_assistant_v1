"""
Day 5: QLoRA fine-tune Qwen2.5-7B-Instruct on the Day 4 SFT data.

- 4-bit (NF4) base model + LoRA adapters on every linear layer (QLoRA).
- Loss is computed ONLY on the assistant's answer. The system prompt and the ~4k tokens
  of retrieved context are masked out, so the model learns how to answer, not to
  reproduce the context.
- Plain transformers Trainer + a custom collator (no TRL), so the script doesn't break
  when TRL renames its config options.

Run on the GPU pod (see Day 5 instructions):
  MAX_STEPS=5 python training/train_lora.py      # smoke test, a few minutes
  python training/train_lora.py                  # full run

Env (defaults in brackets):
  BASE_MODEL [Qwen/Qwen2.5-7B-Instruct]   EPOCHS [2]   LR [2e-4]   GRAD_ACCUM [8]
  LORA_R [16]   LORA_ALPHA [32]   MAX_LEN [6144]   MAX_STEPS [-1 = full run]
  WORK_DIR [/root/lora-run]   checkpoints, on the pod's local disk
  ADAPTER_OUT [adapters/qwen7b-advocate-lora]   final adapter, on the persistent volume
"""

import json
import math
import os
import shutil
import statistics
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

HERE = Path(__file__).resolve().parent
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
TRAIN_PATH = Path(os.environ.get("TRAIN_PATH", HERE / "sft_train.jsonl"))
VAL_PATH = Path(os.environ.get("VAL_PATH", HERE / "sft_val.jsonl"))
WORK_DIR = Path(os.environ.get("WORK_DIR", "/root/lora-run"))
ADAPTER_OUT = Path(os.environ.get("ADAPTER_OUT", HERE.parent / "adapters" / "qwen7b-advocate-lora"))

EPOCHS = float(os.environ.get("EPOCHS", "2"))
LR = float(os.environ.get("LR", "2e-4"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
MAX_LEN = int(os.environ.get("MAX_LEN", "6144"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))
SEED = 42
END_TOKEN = "<|im_end|>"  # Qwen chat format: ends every turn
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def encode(example: dict, tokenizer) -> tuple[list[int], list[int]]:
    """input_ids = prompt + answer; labels = -100 over the prompt, answer ids over the answer."""
    messages = example["messages"]
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    completion_text = messages[-1]["content"] + END_TOKEN
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    return prompt_ids + completion_ids, [-100] * len(prompt_ids) + completion_ids


class SFTDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, name: str):
        self.items, lengths, dropped = [], [], 0
        for example in examples:
            input_ids, labels = encode(example, tokenizer)
            if len(input_ids) > MAX_LEN:
                dropped += 1
                continue
            self.items.append({"input_ids": input_ids, "labels": labels})
            lengths.append(len(input_ids))
        print(f"{name}: {len(self.items)} examples | tokens median {statistics.median(lengths):.0f}, "
              f"max {max(lengths)} | dropped {dropped} over MAX_LEN={MAX_LEN}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def make_collator(pad_id: int):
    def collate(batch: list[dict]) -> dict:
        longest = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention = [], [], []
        for item in batch:
            pad = longest - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention.append([1] * len(item["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention),
        }
    return collate


# ---------------------------------------------------------------------------
# Saving (the RunPod global volume rejects chmod, so copy file contents only)
# ---------------------------------------------------------------------------

def copy_flat_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_file():
            shutil.copyfile(path, dst / path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU visible. Run this on the RunPod GPU pod.")
    torch.manual_seed(SEED)
    started = time.time()

    print(f"Base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if END_TOKEN not in tokenizer.get_vocab():
        raise RuntimeError(f"{END_TOKEN} not in the tokenizer: this script expects a Qwen chat model.")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.convert_tokens_to_ids(END_TOKEN)

    train_set = SFTDataset(load_jsonl(TRAIN_PATH), tokenizer, "train")
    val_set = SFTDataset(load_jsonl(VAL_PATH), tokenizer, "val")

    # Sanity check: show exactly which text the loss is computed on.
    sample = train_set[0]
    trained_ids = [t for t, l in zip(sample["input_ids"], sample["labels"]) if l != -100]
    print(f"\nLoss is computed only on this text (example 0):\n  {tokenizer.decode(trained_ids)!r}\n")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGETS,
    ))
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(train_set) / GRAD_ACCUM)
    total_steps = MAX_STEPS if MAX_STEPS > 0 else math.ceil(steps_per_epoch * EPOCHS)
    full_run = MAX_STEPS <= 0
    print(f"Optimizer steps: {total_steps} ({steps_per_epoch} per epoch, effective batch {GRAD_ACCUM})")

    args = TrainingArguments(
        output_dir=str(WORK_DIR),
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * total_steps)),
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5 if full_run else 1,
        eval_strategy="epoch" if full_run else "no",
        save_strategy="epoch" if full_run else "no",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=make_collator(pad_id),
    )
    trainer.train()
    final_eval = trainer.evaluate()
    print(f"\nFinal validation loss: {final_eval['eval_loss']:.4f}")

    # Save to local disk first, then copy to the persistent volume.
    local_adapter = WORK_DIR / "final_adapter"
    model.save_pretrained(local_adapter)
    tokenizer.save_pretrained(local_adapter)
    summary = {
        "base_model": BASE_MODEL,
        "train_examples": len(train_set),
        "val_examples": len(val_set),
        "epochs": EPOCHS, "max_steps": MAX_STEPS, "optimizer_steps": total_steps,
        "learning_rate": LR, "effective_batch": GRAD_ACCUM,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_targets": LORA_TARGETS,
        "max_len": MAX_LEN, "final_val_loss": final_eval["eval_loss"],
        "minutes": round((time.time() - started) / 60, 1),
        "log_history": trainer.state.log_history,
    }
    (local_adapter / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    copy_flat_dir(local_adapter, ADAPTER_OUT)
    print(f"Adapter saved to {ADAPTER_OUT}")

    # Quick look: generate answers for two validation examples.
    model.eval()
    model.config.use_cache = True
    for example in load_jsonl(VAL_PATH)[:2]:
        prompt = tokenizer.apply_chat_template(example["messages"][:-1], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        question = example["meta"].get("question", example["meta"]["kind"])
        print(f"\nQ: {question}\nMODEL:  {answer.strip()}\nTARGET: {example['messages'][-1]['content']}")

    print(f"\nDone in {summary['minutes']} min.")


if __name__ == "__main__":
    main()
