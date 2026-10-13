"""Per-category correctness table across result CSVs (paste-ready Markdown).

Category = question id with its trailing number removed (calc_01 -> calc,
hruling_07 -> hruling). Run from the repo root:

    python eval/category_breakdown.py                      # default ablation set
    python eval/category_breakdown.py eval/a.csv eval/b.csv ...   # your own files

Needs only the standard library. Requires an 'answer_correct' column.
"""
import csv
import re
import sys
from collections import OrderedDict

DEFAULT = [
    ("full (base)", "eval/results_v3_base.csv"),
    ("snip 1000", "eval/results_abl_snip1000.csv"),
    ("no case-scope", "eval/results_abl_nocase.csv"),
    ("no calculator", "eval/results_abl_nocalc.csv"),
    ("all three off", "eval/results_abl_stripped.csv"),
    ("SFT", "eval/results_v3_advocate-ep1.csv"),
    ("SFT+DPO", "eval/results_v3_advocate-dpo.csv"),
]
TRUE = {"true", "1", "yes", "y"}


def category(qid):
    return re.sub(r"_\d+$", "", qid)


def load(path):
    out = OrderedDict()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["id"]] = str(r["answer_correct"]).strip().lower() in TRUE
    return out


def main(argv):
    runs = [(p.split("results_")[-1].removesuffix(".csv"), p) for p in argv] or DEFAULT
    data = []
    for name, path in runs:
        try:
            data.append((name, load(path)))
        except FileNotFoundError:
            print(f"(skipping {path}: not found)", file=sys.stderr)
    if not data:
        raise SystemExit("no result files found")

    cats = OrderedDict()
    for qid in data[0][1]:
        cats.setdefault(category(qid), []).append(qid)

    header = "| Category (n) | " + " | ".join(n for n, _ in data) + " |"
    print(header)
    print("|" + "---|" * (len(data) + 1))
    for cat, ids in cats.items():
        cells = [f"{sum(d.get(i, False) for i in ids)}/{len(ids)}" for _, d in data]
        print(f"| {cat} ({len(ids)}) | " + " | ".join(cells) + " |")
    total = [f"{sum(d.values())}/{len(d)}" for _, d in data]
    print("| **all** | " + " | ".join(f"**{t}**" for t in total) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
