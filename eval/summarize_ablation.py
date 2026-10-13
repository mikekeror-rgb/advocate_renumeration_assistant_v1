"""Summarise ablation result CSVs with bootstrap confidence intervals.

Usage (from the repo root):
    python eval/summarize_ablation.py eval/results_*.csv
    python eval/summarize_ablation.py --pair eval/results_A.csv eval/results_B.csv

Per file: n, correctness with 95% CI, unverifiable-citation count.
--pair: paired bootstrap of (B - A) correctness on the questions both runs share.
Needs only the standard library.
"""
import argparse
import csv
import random

TRUE = {"true", "1", "yes", "y"}
FALSE = {"false", "0", "no", "n"}


def to_bool(v):
    s = str(v).strip().lower()
    if s in TRUE:
        return True
    if s in FALSE:
        return False
    return None  # blank / not applicable


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def correct_by_id(rows):
    if rows and "answer_correct" not in rows[0]:
        raise SystemExit("No 'answer_correct' column: re-run with the current eval/eval.py.")
    return {r["id"]: to_bool(r["answer_correct"]) is True for r in rows}


def ci(values, n_boot=10000, seed=0):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def summarize(path):
    rows = load(path)
    c = list(correct_by_id(rows).values())
    lo, hi = ci([int(x) for x in c])
    fabricated = sum(1 for r in rows if to_bool(r.get("citation_verifiable")) is False)
    print(f"{path}\n  n={len(c)}  correct={sum(c)/len(c):.1%}  95% CI [{lo:.1%}, {hi:.1%}]"
          f"  unverifiable-citation rows={fabricated}")


def pair(a_path, b_path, n_boot=10000, seed=0):
    a, b = correct_by_id(load(a_path)), correct_by_id(load(b_path))
    ids = sorted(set(a) & set(b))
    diffs = [int(b[i]) - int(a[i]) for i in ids]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
    better = sum(d == 1 for d in diffs)
    worse = sum(d == -1 for d in diffs)
    print(f"B - A over {len(ids)} shared questions: {sum(diffs)/len(diffs):+.1%}  "
          f"95% CI [{lo:+.1%}, {hi:+.1%}]   (B fixes {better}, B breaks {worse})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--pair", action="store_true", help="compare exactly two files")
    args = ap.parse_args()
    if args.pair:
        if len(args.files) != 2:
            raise SystemExit("--pair needs exactly two CSVs: A then B")
        pair(*args.files)
    else:
        for f in args.files:
            summarize(f)
