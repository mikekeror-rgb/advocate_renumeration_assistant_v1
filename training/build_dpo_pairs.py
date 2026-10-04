"""
Day 7b (DPO, deferred from Day 5): build ON-POLICY preference pairs.

The fine-tuned model (advocate-ep1) answers every TRAINING question with the exact prompt
it was trained on. Where it gets something wrong, its own answer becomes "rejected" and
the reference answer becomes "chosen". DPO then teaches it to avoid the mistakes it
actually makes (v2 eval: over-compressed answers that drop or swap legal details, and
answers sourced from the wrong case), instead of mistakes we guess at.

Checked per example kind:
  grounded         reference-based judge (same prompt as eval.py) + must cite the source chunk
  refusal/offscope must give the exact refusal (catches answering from a different case)
Calculator examples are skipped: the calculator already makes those deterministic.
Eval questions are never used: this only reads training/sft_train.jsonl.

Run from the project root ON YOUR MAC while the pod's vLLM serves advocate-ep1:
  export LLM_BASE_URL="https://<pod-id>-8000.proxy.runpod.net/v1"  LLM_API_KEY=<vllm key>
  export JUDGE_MODEL=gpt-5.4-mini            (needs OPENAI_API_KEY)
  python training/build_dpo_pairs.py
Optional env: POLICY_MODEL (advocate-ep1), WORKERS (6), LIMIT (0 = all; e.g. 20 for a test)
Replies are cached in training/dpo_policy_cache.jsonl, so an interrupted run resumes.

Outputs (training/): dpo_pairs.jsonl  {"prompt": [system, user], "chosen": str, "rejected": str, "meta": {...}}
                     dpo_stats.json
"""

import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "training" / "sft_train.jsonl"
CACHE_PATH = ROOT / "training" / "dpo_policy_cache.jsonl"
PAIRS_PATH = ROOT / "training" / "dpo_pairs.jsonl"
STATS_PATH = ROOT / "training" / "dpo_stats.json"

POLICY_MODEL = os.environ.get("POLICY_MODEL", "advocate-ep1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.4-mini")
WORKERS = int(os.environ.get("WORKERS", "6"))
LIMIT = int(os.environ.get("LIMIT", "0"))
REFUSAL = "Not covered in the provided documents."
CITATION = re.compile(r"\s*\[chunk_id:[^\]]*\]")
CHECKED_KINDS = {"grounded", "refusal", "offscope"}

if not os.environ.get("LLM_BASE_URL"):
    sys.exit("Set LLM_BASE_URL (and LLM_API_KEY) to the pod's vLLM endpoint first.")
policy_client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ.get("LLM_API_KEY", "none"),
                       timeout=300, max_retries=2)
judge_client = OpenAI(timeout=120, max_retries=3)  # uses OPENAI_API_KEY
cache_lock = Lock()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_json(raw: str) -> dict | None:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
    for candidate in (raw, *re.findall(r"\{.*\}", raw, re.DOTALL)):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def judge_correct(question: str, reference: str, answer: str) -> tuple[bool | None, str]:
    """Same reference-based judgement as eval/eval.py, so 'wrong' means the same thing."""
    prompt = f"""You are grading whether a legal research assistant's answer is CORRECT, \
compared with a reference answer written by an expert.

Question: {question}
Reference answer: {reference}
System's answer: {answer}

The system's answer is CORRECT if it states the same key facts or conclusion as the reference. \
Different wording is fine, and extra accurate detail is fine.
It is INCORRECT if it contradicts the reference, misses the key fact or conclusion, or says the \
question is not covered when the reference gives an answer.

Respond with ONLY a JSON object, no other text: {{"correct": true or false, "reasoning": "<one sentence>"}}"""
    kwargs = {"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}]}
    if JUDGE_MODEL.startswith(("gpt-5", "o1", "o3", "o4")):
        kwargs["reasoning_effort"] = "low"
    raw = judge_client.chat.completions.create(**kwargs).choices[0].message.content
    parsed = parse_json(raw)
    if not parsed or "correct" not in parsed:
        return None, f"[judge unparseable: {(raw or '')[:120]}]"
    correct = parsed["correct"]
    if isinstance(correct, str):
        correct = correct.strip().lower() == "true"
    return bool(correct), parsed.get("reasoning", "")


def check_example(idx: int, example: dict) -> dict:
    meta, kind = example["meta"], example["meta"]["kind"]
    response = policy_client.chat.completions.create(
        model=POLICY_MODEL, messages=example["messages"][:-1], temperature=0.1, max_tokens=1024,
    )
    answer = (response.choices[0].message.content or "").strip()
    result = {"key": f"{POLICY_MODEL}:{idx}", "idx": idx, "kind": kind, "answer": answer}

    if kind in ("refusal", "offscope"):
        ok = REFUSAL.lower() in answer.lower()
        result.update(ok=ok, reasons=[] if ok else ["answered instead of refusing"])
        return result

    reference = example["messages"][-1]["content"]
    correct, why = judge_correct(meta["question"], CITATION.sub("", reference).strip(),
                                 CITATION.sub("", answer).strip())
    reasons = []
    if correct is None:
        result.update(ok=None, reasons=[why])  # can't tell: no pair, and not counted
        return result
    if not correct:
        reasons.append(f"incorrect: {why}")
    if meta["gold_chunk_id"] not in answer:
        reasons.append("did not cite the source chunk")
    result.update(ok=not reasons, reasons=reasons)
    return result


def main() -> None:
    examples = [e for e in load_jsonl(TRAIN_PATH) if e["meta"]["kind"] in CHECKED_KINDS]
    if LIMIT:
        examples = examples[:LIMIT]
    cache = {row["key"]: row for row in load_jsonl(CACHE_PATH)}
    todo = [(i, e) for i, e in enumerate(examples) if f"{POLICY_MODEL}:{i}" not in cache]
    print(f"Policy: {POLICY_MODEL} | judge: {JUDGE_MODEL} | {len(examples)} training examples "
          f"({len(examples) - len(todo)} cached, {len(todo)} to run)")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(check_example, i, e): i for i, e in todo}
        for n, future in enumerate(as_completed(futures), start=1):
            try:
                row = future.result()
            except Exception as error:  # one failed call shouldn't stop the run; rerun retries it
                print(f"  ⚠ example {futures[future]}: {type(error).__name__}: {str(error)[:120]}")
                continue
            with cache_lock:
                cache[row["key"]] = row
                with open(CACHE_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if n % 25 == 0:
                print(f"  {n}/{len(todo)} checked")

    pairs, checked, failing, reasons = [], Counter(), Counter(), Counter()
    for i, example in enumerate(examples):
        row = cache.get(f"{POLICY_MODEL}:{i}")
        if not row or row["ok"] is None:
            continue
        checked[row["kind"]] += 1
        if row["ok"]:
            continue
        failing[row["kind"]] += 1
        for reason in row["reasons"]:
            reasons[reason.split(":")[0]] += 1
        chosen, rejected = example["messages"][-1]["content"], row["answer"]
        if chosen.strip() == rejected.strip():
            continue
        pairs.append({
            "prompt": example["messages"][:-1],
            "chosen": chosen,
            "rejected": rejected,
            "meta": {"kind": row["kind"], "reasons": row["reasons"],
                     "doc_title": example["meta"].get("doc_title"), "question": example["meta"].get("question")},
        })

    with open(PAIRS_PATH, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    stats = {"policy": POLICY_MODEL, "judge": JUDGE_MODEL, "pairs": len(pairs),
             "checked": dict(checked), "failing": dict(failing), "reasons": dict(reasons)}
    STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\n=== On-policy check of the SFT model on its own training prompts ===")
    for kind in sorted(checked):
        print(f"  {kind:<9} {failing[kind]:>3} failing of {checked[kind]}")
    for reason, count in reasons.most_common():
        print(f"  [{reason}] {count}")
    print(f"\nWrote {len(pairs)} DPO pairs to {PAIRS_PATH.name}")
    if len(pairs) < 20:
        print("⚠ Fewer than 20 pairs: DPO would have very little to learn from. Tell Claude before training.")
    for pair in pairs[:2]:
        print(f"\nQ: {pair['meta']['question']}\nCHOSEN:   {pair['chosen'][:300]}\nREJECTED: {pair['rejected'][:300]}"
              f"\nWHY: {pair['meta']['reasons']}")


if __name__ == "__main__":
    main()
