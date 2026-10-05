"""
build_eval_v2.py: turn the held-out validation split into extra eval questions.

Why: on the original 24 questions every model (base, epoch 1, epoch 2) scores 100%
correct, so the eval can no longer tell them apart. The validation split comes from
rulings the model never trained on (the split is grouped by case), so it is a fair,
harder test.

What it writes (eval/):
  qa_pairs_v2_candidates.jsonl   new questions, each marked "needs_review": true
  v2_review.txt                  question + reference answer + source passage, for checking
It also prints rulings that share a name (e.g. the two Cheruiyot v Ngeno cases), so you
can hand-write extra "which case?" questions, the hardest attribution test.

Question types created:
  hruling_*    held-out ruling questions   (factual; checks correctness + citing the right case)
  hstatute_*   held-out statute questions  (factual)
  hcalc_*      held-out calculator questions (numeric)
  hnoanswer_*  held-out out-of-scope questions (exact refusal)
Refusal examples are skipped on purpose: in training their context had the case removed,
but in the real pipeline that case IS retrievable, so the correct live answer is not a refusal.

Run from the project root (no LLM calls, no GPU):  python eval/build_eval_v2.py
Then review, delete bad lines, and merge:
  cat eval/qa_pairs.jsonl eval/qa_pairs_v2_candidates.jsonl > eval/qa_pairs_v2.jsonl
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chromadb  # noqa: E402

import case_matcher  # noqa: E402
import fee_router  # noqa: E402

VAL_PATH = ROOT / "training" / "sft_val.jsonl"
EXISTING_QA = ROOT / "eval" / "qa_pairs.jsonl"
OUT_CANDIDATES = ROOT / "eval" / "qa_pairs_v2_candidates.jsonl"
OUT_REVIEW = ROOT / "eval" / "v2_review.txt"
REFUSAL = "Not covered in the provided documents."
CITATION = re.compile(r"\s*\[chunk_id:[^\]]*\]")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def case_number(doc_title: str) -> str | None:
    """A short identifier that appears in every chunk_id of that ruling,
    e.g. 'E368of2024' or, for older titles, '2021KEHC6864'."""
    match = re.search(r"\b[A-Z]*\d+of\d{4}\b", doc_title) or re.search(r"\d{4}KE[A-Z]+\d+", doc_title)
    return match.group(0) if match else None


def source_passage(example: dict) -> str:
    """The gold chunk's text, cut from the user message (for the review sheet)."""
    gold = example["meta"].get("gold_chunk_id")
    user = example["messages"][1]["content"]
    start = user.find(gold) if gold else -1
    return user[start:start + 1800].replace("\n", " ") if start >= 0 else "(source passage not found)"


def main() -> None:
    val = load_jsonl(VAL_PATH)
    existing_questions = {q["question"].strip().lower() for q in load_jsonl(EXISTING_QA)}

    candidates, review_lines = [], []
    counters = defaultdict(int)
    skipped = defaultdict(int)

    for example in val:
        meta = example["meta"]
        kind = meta["kind"]
        question = meta.get("question") or example["messages"][1]["content"].rsplit("Question: ", 1)[-1].strip()
        if question.strip().lower() in existing_questions:
            skipped["already in eval"] += 1
            continue
        answer = example["messages"][2]["content"]

        if kind == "grounded":
            prefix = "hruling" if meta["doc_type"] == "ruling" else "hstatute"
            counters[prefix] += 1
            item = {
                "id": f"{prefix}_{counters[prefix]:02d}",
                "question": question,
                "expected_type": "factual",
                "expected_value": CITATION.sub("", answer).strip(),
                "expected_source": meta["doc_type"],
                "evidence_chunk_id": meta["gold_chunk_id"],
                "notes": "Held-out validation split; reference answer written by the teacher model.",
                "needs_review": True,
            }
            if meta["doc_type"] == "ruling":
                hint = case_number(meta["doc_title"])
                if hint:
                    item["doc_hint"] = hint
            review_lines.append(
                f"[{item['id']}]\nQ: {question}\nREFERENCE: {item['expected_value']}\n"
                f"PASSAGE: {source_passage(example)}\n{'=' * 70}\n"
            )
        elif kind == "calc":
            route = fee_router.route(question)
            if route is None or route.fee is None:
                skipped["calc with two possible figures"] += 1
                continue
            counters["hcalc"] += 1
            item = {
                "id": f"hcalc_{counters['hcalc']:02d}",
                "question": question,
                "expected_type": "numeric",
                "expected_value": route.fee,
                "expected_source": "statute",
                "notes": f"Held-out calculator question ({route.scenario}).",
            }
        elif kind == "offscope":
            counters["hnoanswer"] += 1
            item = {
                "id": f"hnoanswer_{counters['hnoanswer']:02d}",
                "question": question,
                "expected_type": "no_answer",
                "expected_value": REFUSAL,
                "expected_source": "none",
                "notes": "Held-out out-of-scope question.",
            }
        else:
            skipped[f"{kind} (not a fair live test)"] += 1
            continue
        candidates.append(item)

    with open(OUT_CANDIDATES, "w", encoding="utf-8") as f:
        for item in candidates:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    OUT_REVIEW.write_text("".join(review_lines), encoding="utf-8")

    print(f"Wrote {len(candidates)} candidate questions to {OUT_CANDIDATES.name}")
    for prefix, count in sorted(counters.items()):
        print(f"  {prefix:<10} {count}")
    for reason, count in skipped.items():
        print(f"  skipped ({reason}): {count}")
    print(f"Review sheet: {OUT_REVIEW.name} ({len(review_lines)} questions with source passages)")

    # Rulings that share a name: the hardest attribution test.
    metadatas = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection("policy_docs").get(
        include=["metadatas"])["metadatas"]
    titles = {m["doc_title"] for m in metadatas if m.get("doc_type") == "ruling"}
    by_keys = defaultdict(list)
    for title, keys in case_matcher.build_case_index(titles).items():
        by_keys[tuple(keys)].append(title)
    collisions = [group for group in by_keys.values() if len(group) > 1]
    print(f"\nRulings sharing a name ({len(collisions)} groups). Good material for hand-written questions:")
    for group in sorted(collisions)[:15]:
        print("  - " + "\n    ".join(sorted(group)))


if __name__ == "__main__":
    main()
