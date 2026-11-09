"""
retrieval_eval.py: score retrieval alone, without calling the LLM or the judge.

Retrieval is deterministic and runs on CPU, so a retrieval change can be measured in a
couple of minutes on a laptop instead of a GPU eval run. Use it to try one change at a
time, then run the full eval (eval_v2.py) only on the change worth keeping.

For every question with a gold evidence chunk (evidence_chunk_id in the question file):
  - hit@k : the gold chunk is among the chunks passed to the model (k = what
            RagPipeline.retrieve() returns, normally top_k + statute_boost = 7)
  - rank  : its position in that list (for MRR)
  - pool@N: the gold chunk is in the top N of a deeper retrieve(top_k=N). This is the
            ceiling for a re-ranker that reorders N candidates into the final k.

Run from the project root:
  python eval/retrieval_eval.py                                   # v3 set, current pipeline
  python eval/retrieval_eval.py --tag baseline --compare eval/results_ev3_base.csv

--compare checks each hit against that run's logged evidence_retrieved (they should
agree when retrieval code and index are unchanged) and reports hits on the questions
that run answered "Not covered" although they're answerable.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# RagPipeline insists on an LLM backend at construction. Retrieval never calls it, so
# a placeholder is enough when none is configured.
if not (os.environ.get("GROQ_API_KEY") or os.environ.get("LLM_BASE_URL")):
    os.environ.update(LLM_BASE_URL="http://127.0.0.1:9/v1", LLM_API_KEY="none", LLM_MODEL="none")

from rag_pipeline_v2 import RagPipeline  # noqa: E402

REFUSAL = "not covered in the provided documents"


def stratum_of(q: dict) -> str:
    s = q.get("stratum", "")
    if s.startswith("ruling"):
        return "ruling_" + q.get("case_status", "")
    return s


def pct(hits: list[bool]) -> str:
    return f"{100 * sum(hits) / len(hits):5.1f}% ({sum(hits)}/{len(hits)})" if hits else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=str(ROOT / "eval" / "qa_pairs_v3.jsonl"))
    ap.add_argument("--tag", default="current", help="label; per-question rows go to eval/retrieval_<tag>.csv")
    ap.add_argument("--depths", default="20,50", help="candidate-pool depths to check")
    ap.add_argument("--compare", default=None, help="a results_*.csv from eval_v2.py to check against")
    a = ap.parse_args()

    questions = [json.loads(line) for line in open(a.qa)]
    questions = [q for q in questions if q.get("evidence_chunk_id")]
    depths = [int(d) for d in a.depths.split(",") if d]
    pipeline = RagPipeline()

    rows = []
    for q in questions:
        gold = q["evidence_chunk_id"]
        ids = [c["chunk_id"] for c in pipeline.retrieve(q["question"])]
        row = {"id": q["id"], "stratum": stratum_of(q), "k": len(ids),
               "hit": gold in ids, "rank": ids.index(gold) + 1 if gold in ids else None}
        for n in depths:
            row[f"pool@{n}"] = gold in [c["chunk_id"] for c in pipeline.retrieve(q["question"], top_k=n)]
        rows.append(row)

    out = ROOT / "eval" / f"retrieval_{a.tag}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ks = sorted({r["k"] for r in rows})
    print(f"\n=== Retrieval only: {a.tag} ({len(rows)} questions with a gold chunk; k = {ks}) ===")
    strata = ["ALL"] + sorted({r["stratum"] for r in rows})
    cols = ["hit@k"] + [f"pool@{n}" for n in depths]
    print(f"{'subset':16s}" + "".join(f"{c:>22s}" for c in cols))
    for s in strata:
        sub = [r for r in rows if s == "ALL" or r["stratum"] == s]
        cells = [pct([r["hit"] for r in sub])] + [pct([r[f"pool@{n}"] for r in sub]) for n in depths]
        print(f"{s:16s}" + "".join(f"{c:>22s}" for c in cells))
    mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / len(rows)
    print(f"MRR (within k): {mrr:.3f}")

    if a.compare:
        logged = {r["id"]: r for r in csv.DictReader(open(a.compare))}
        both = [r for r in rows if r["id"] in logged and logged[r["id"]]["evidence_retrieved"]]
        disagree = [r["id"] for r in both if r["hit"] != (logged[r["id"]]["evidence_retrieved"] in ("visible", "truncated"))]
        print(f"\nAgreement with {Path(a.compare).name}: {len(both) - len(disagree)}/{len(both)}"
              + (f"  disagree: {disagree}" if disagree else ""))
        refused = [r for r in rows if r["id"] in logged and REFUSAL in logged[r["id"]]["answer"].lower()]
        print(f"False refusals in that run with a gold chunk: {len(refused)}")
        for r in refused:
            pools = " ".join(f"pool@{n}={'Y' if r[f'pool@{n}'] else 'n'}" for n in depths)
            print(f"  {r['id']:26s} hit@k={'Y' if r['hit'] else 'n'}  {pools}")
    print(f"\nPer-question rows: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
