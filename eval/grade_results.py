"""
grade_results.py: grade a results file that eval_v2.py saved with JUDGE_MODEL=none.

Answer generation needs the GPU; judging doesn't. So a run can generate answers on
Kaggle without a judge, and be graded here later with the same judge, prompts and
context as an inline run (the context is stored per row in judge_context).

Only rows still missing a grade are sent to the judge, and the file is saved after
every row, so if the judge stops (credits, network) just run the command again.

Run from the project root (OPENAI_API_KEY in .env or the environment):
  python eval/grade_results.py eval/results_ev3_t4_rr_on.csv
  JUDGE_MODEL=gpt-5.4-mini python eval/grade_results.py <file> --qa eval/qa_pairs_v3.jsonl
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

# The judge used for every v3 run. Must be set before eval_v2 is imported.
os.environ["JUDGE_MODEL"] = os.environ.get("JUDGE_MODEL") or "gpt-5.4-mini"
if os.environ["JUDGE_MODEL"].lower() == "none":
    sys.exit("Set JUDGE_MODEL to a real judge, e.g. gpt-5.4-mini.")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_v2  # noqa: E402


def needs_correctness(row: dict) -> bool:
    return row["expected_type"] == "factual" and row["answer_correct"] == ""


def needs_faithfulness(row: dict) -> bool:
    return row["expected_type"] != "guard" and row["faithfulness_score"] in ("", "0")


def typed(row: dict) -> dict:
    """CSV strings back to the values print_summary() expects."""
    out = {}
    for key, value in row.items():
        if value == "True":
            out[key] = True
        elif value == "False":
            out[key] = False
        elif value == "":
            out[key] = None
        elif key == "faithfulness_score":
            out[key] = int(value)
        else:
            out[key] = value
    return out


def save(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)  # never leaves a half-written results file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="results CSV written with JUDGE_MODEL=none")
    ap.add_argument("--qa", default="eval/qa_pairs_v3.jsonl")
    a = ap.parse_args()

    path = Path(a.results)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if "judge_context" not in rows[0]:
        sys.exit(f"{path} has no judge_context column: it wasn't written with JUDGE_MODEL=none.")
    questions = {q["id"]: q for q in map(json.loads, open(a.qa))}
    todo = [r for r in rows if needs_correctness(r) or needs_faithfulness(r)]
    print(f"{path.name}: {len(todo)} of {len(rows)} rows need grading (judge: {eval_v2.JUDGE_MODEL_NAME})")

    for n, row in enumerate(todo, 1):
        q = questions[row["id"]]
        try:
            if needs_correctness(row):
                ok, why = eval_v2.judge_correctness(q["question"], row["answer"], str(q["expected_value"]))
                correct = ("" if ok is None else str(ok), why)
            else:
                correct = (row["answer_correct"], row["correctness_reasoning"])
            if needs_faithfulness(row):
                score, why = eval_v2.judge_faithfulness(q["question"], row["answer"], row["judge_context"])
                faithful = (str(score), why)
            else:
                faithful = (row["faithfulness_score"], row["faithfulness_reasoning"])
        except Exception as e:  # out of credits, network, rate limit: keep what's done
            sys.exit(f"\nJudge stopped at {row['id']} ({type(e).__name__}: {e}).\n"
                     f"{n - 1} rows graded and saved. Run the same command again to continue.")
        row["answer_correct"], row["correctness_reasoning"] = correct
        row["faithfulness_score"], row["faithfulness_reasoning"] = faithful
        save(path, rows)
        print(f"  [{n}/{len(todo)}] {row['id']}: correct={row['answer_correct'] or '?'} faithfulness={row['faithfulness_score']}")

    left = [r["id"] for r in rows if needs_correctness(r) or needs_faithfulness(r)]
    if left:
        print(f"\n{len(left)} rows still ungraded (judge reply unparseable or failed); run again to retry: {left}")
    eval_v2.print_summary([typed(r) for r in rows])


if __name__ == "__main__":
    main()
