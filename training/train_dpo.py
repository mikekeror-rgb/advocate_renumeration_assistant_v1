"""
Day 7b: DPO on top of the SFT adapter (QLoRA), with a memory-light DPO implementation.

- Policy = the SFT adapter (advocate-ep1), trainable, on the 4-bit base model.
- Reference = the same SFT adapter BEFORE training. Its log-probs are computed once, up
  front, so no second model copy sits in GPU memory.
- Loss = DPO (beta) + a small NLL term on the chosen answer (NLL_ALPHA), which keeps the
  model from drifting away from good answers when the pair count is small.
- Memory trick: the DPO gradient splits exactly into a weighted chosen term and a weighted
  rejected term, so each ~4k-token sequence is back-propagated on its own. Peak memory is
  about the same as SFT, not double.
- Dropout is switched off, as is standard for DPO.
- Plain PyTorch loop (no TRL), so library renames can't break it.

Run on the GPU pod, with vLLM STOPPED (both need the GPU):
  python training/train_dpo.py
Env (defaults): BASE_MODEL [Qwen/Qwen2.5-7B-Instruct]  SFT_ADAPTER [adapters/qwen7b-advocate-lora-ep1]
  PAIRS [training/dpo_pairs.jsonl]  OUT [adapters/qwen7b-advocate-dpo]
  BETA [0.1]  LR [2e-5]  EPOCHS [2]  GRAD_ACCUM [4]  NLL_ALPHA [0.2]  MAX_LEN [6144]
"""

import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SFT_ADAPTER = Path(os.environ.get("SFT_ADAPTER", ROOT / "adapters" / "qwen7b-advocate-lora-ep1"))
PAIRS_PATH = Path(os.environ.get("PAIRS", HERE / "dpo_pairs.jsonl"))
OUT = Path(os.environ.get("OUT", ROOT / "adapters" / "qwen7b-advocate-dpo"))
WORK_DIR = Path(os.environ.get("WORK_DIR", "/root/dpo-run"))

BETA = float(os.environ.get("BETA", "0.1"))
LR = float(os.environ.get("LR", "2e-5"))
EPOCHS = int(os.environ.get("EPOCHS", "2"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
NLL_ALPHA = float(os.environ.get("NLL_ALPHA", "0.2"))
MAX_LEN = int(os.environ.get("MAX_LEN", "6144"))
VAL_SHARE = 0.15
SEED = 42
END_TOKEN = "<|im_end|>"


def encode(tokenizer, prompt: list[dict], answer: str) -> tuple[list[int], list[int]]:
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer + END_TOKEN, add_special_tokens=False)["input_ids"]
    return prompt_ids + answer_ids, [-100] * len(prompt_ids) + answer_ids


def answer_logp(causal_lm, input_ids: list[int], labels: list[int]) -> tuple[torch.Tensor, int]:
    """Sum of log-probs of the answer tokens. The vocab projection is applied only at
    answer positions, avoiding a ~2.5 GB logits tensor for a 4k-token prompt."""
    ids = torch.tensor([input_ids], device="cuda")
    target = torch.tensor([labels], device="cuda")[:, 1:]
    mask = target != -100
    hidden = causal_lm.model(input_ids=ids).last_hidden_state[:, :-1]
    logits = causal_lm.lm_head(hidden[mask]).float()
    logp = torch.log_softmax(logits, dim=-1).gather(-1, target[mask].unsqueeze(-1)).squeeze(-1)
    return logp.sum(), int(mask.sum())


def copy_flat_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_file():
            shutil.copyfile(path, dst / path.name)  # the RunPod volume rejects chmod/copystat


@torch.no_grad()
def preference_metrics(causal_lm, pairs: list[dict]) -> dict:
    margins = []
    for p in pairs:
        pc, _ = answer_logp(causal_lm, p["c_ids"], p["c_lab"])
        pr, _ = answer_logp(causal_lm, p["r_ids"], p["r_lab"])
        margins.append(((pc.item() - p["ref_c"]) - (pr.item() - p["ref_r"])))
    if not margins:
        return {}
    return {"accuracy": round(sum(m > 0 for m in margins) / len(margins), 3),
            "mean_margin": round(sum(margins) / len(margins), 3)}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU visible. Run this on the RunPod GPU pod.")
    random.seed(SEED)
    torch.manual_seed(SEED)
    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    raw_pairs = [json.loads(l) for l in open(PAIRS_PATH, encoding="utf-8") if l.strip()]
    pairs, dropped = [], 0
    for raw in raw_pairs:
        c_ids, c_lab = encode(tokenizer, raw["prompt"], raw["chosen"])
        r_ids, r_lab = encode(tokenizer, raw["prompt"], raw["rejected"])
        if max(len(c_ids), len(r_ids)) > MAX_LEN:
            dropped += 1
            continue
        pairs.append({"c_ids": c_ids, "c_lab": c_lab, "r_ids": r_ids, "r_lab": r_lab, "meta": raw["meta"]})
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_SHARE)) if len(pairs) >= 10 else 0
    val, train = pairs[:n_val], pairs[n_val:]
    print(f"Pairs: {len(train)} train, {len(val)} val, {dropped} dropped over MAX_LEN")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_use_double_quant=True),
        dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(model, str(SFT_ADAPTER), is_trainable=True)
    for module in model.modules():  # DPO standard: no dropout, so policy and reference agree at step 0
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.train()  # keeps gradient checkpointing active
    causal_lm = model.get_base_model()
    model.print_trainable_parameters()

    print("Reference log-probs (the SFT adapter before DPO)...")
    with torch.no_grad():
        for p in pairs:
            p["ref_c"] = answer_logp(causal_lm, p["c_ids"], p["c_lab"])[0].item()
            p["ref_r"] = answer_logp(causal_lm, p["r_ids"], p["r_lab"])[0].item()
    val_before = preference_metrics(causal_lm, val)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    total_steps = max(1, math.ceil(len(train) / GRAD_ACCUM) * EPOCHS)
    warmup = max(1, int(0.1 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    print(f"DPO: beta={BETA} lr={LR} epochs={EPOCHS} grad_accum={GRAD_ACCUM} nll_alpha={NLL_ALPHA} "
          f"-> {total_steps} optimizer steps")

    history, step = [], 0
    window = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}
    for epoch in range(EPOCHS):
        random.shuffle(train)
        for i, p in enumerate(train):
            with torch.no_grad():
                pr_value = answer_logp(causal_lm, p["r_ids"], p["r_lab"])[0].item()
            pc, n_c = answer_logp(causal_lm, p["c_ids"], p["c_lab"])
            z = BETA * ((pc.item() - p["ref_c"]) - (pr_value - p["ref_r"]))
            w = BETA * (1 - 1 / (1 + math.exp(-z)))  # = -dLoss/d(logp_chosen) = dLoss/d(logp_rejected)
            ((-w * pc - NLL_ALPHA * pc / n_c) / GRAD_ACCUM).backward()
            pr, _ = answer_logp(causal_lm, p["r_ids"], p["r_lab"])
            ((w * pr) / GRAD_ACCUM).backward()

            window["loss"] += -F.logsigmoid(torch.tensor(z)).item()
            window["acc"] += float(z > 0)
            window["margin"] += z / BETA
            window["n"] += 1
            if (i + 1) % GRAD_ACCUM == 0 or i + 1 == len(train):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                record = {"step": step, "epoch": epoch + 1,
                          "loss": round(window["loss"] / window["n"], 4),
                          "pref_acc": round(window["acc"] / window["n"], 3),
                          "margin": round(window["margin"] / window["n"], 3),
                          "lr": round(scheduler.get_last_lr()[0], 8)}
                history.append(record)
                print(record, flush=True)
                window = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}

    val_after = preference_metrics(causal_lm, val)
    print(f"\nValidation preference accuracy: before {val_before} -> after {val_after}")

    local = WORK_DIR / "final_adapter"
    model.save_pretrained(local)
    summary = {"base_model": BASE_MODEL, "sft_adapter": str(SFT_ADAPTER), "train_pairs": len(train),
               "val_pairs": len(val), "beta": BETA, "lr": LR, "epochs": EPOCHS, "grad_accum": GRAD_ACCUM,
               "nll_alpha": NLL_ALPHA, "val_before": val_before, "val_after": val_after,
               "minutes": round((time.time() - started) / 60, 1), "history": history}
    (local / "dpo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    copy_flat_dir(local, OUT)
    print(f"DPO adapter saved to {OUT} ({summary['minutes']} min)")


if __name__ == "__main__":
    main()
