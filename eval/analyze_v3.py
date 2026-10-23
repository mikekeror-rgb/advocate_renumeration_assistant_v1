"""
analyze_v3.py: pre-registered analysis for the v3 held-out set (eval/PREREG.md).

Paired differences between systems, 95% percentile intervals from a cluster
bootstrap (clusters = case for ruling questions; every other question is its
own cluster), 10,000 resamples.

Run from the project root:
  python eval/analyze_v3.py                       # uses eval/qa_pairs_v3.jsonl
  python eval/analyze_v3.py --review path/to/v3_review_edit_2.csv   # fallback
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

SEED = 20261017
N_BOOT = 10000
SYSTEMS = ["base", "base_rerun", "ep1", "dpo", "nocase", "snip1000", "nocalc", "stripped"]


def strat_of(qid):
    p = qid.split("_")
    if p[1] == "unseen":
        return "ruling_unseen"
    if p[1] == "seen":
        return "ruling_seen"
    return p[1]  # statute / noanswer / calc


def load_clusters(qa_path, review_path):
    """id -> cluster key. Ruling questions cluster by case; the rest by id."""
    cl = {}
    if review_path:
        for r in csv.DictReader(open(review_path, encoding="utf-8-sig")):
            if r["verdict"].strip().lower() == "drop":
                continue
            s = strat_of(r["id"])
            cl[r["id"]] = r["doc_title"] if s.startswith("ruling") else r["id"]
    else:
        for line in open(qa_path):
            q = json.loads(line)
            s = strat_of(q["id"])
            cl[q["id"]] = q["case"] if s.startswith("ruling") else q["id"]
    return cl


def load_results(res_dir, tag):
    f = Path(res_dir) / f"results_ev3_{tag}.csv"
    out = {}
    for r in csv.DictReader(open(f, encoding="utf-8-sig")):
        if r["answer_correct"] == "":
            sys.exit(f"{f}: ungraded row {r['id']}")
        out[r["id"]] = 1 if r["answer_correct"] == "True" else 0
    return out


def boot_ci(ids, values, clusters, rng):
    """Mean of values over ids with a cluster bootstrap. Returns (mean, lo, hi)."""
    groups = {}
    for i in ids:
        g = groups.setdefault(clusters[i], [0.0, 0])
        g[0] += values[i]
        g[1] += 1
    sums = [g[0] for g in groups.values()]
    cnts = [g[1] for g in groups.values()]
    n = len(sums)
    mean = sum(sums) / sum(cnts)
    stats = []
    for _ in range(N_BOOT):
        s = c = 0
        for _ in range(n):
            k = rng.randrange(n)
            s += sums[k]
            c += cnts[k]
        stats.append(s / c)
    stats.sort()
    return mean, stats[int(0.025 * N_BOOT)], stats[int(0.975 * N_BOOT) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="eval/qa_pairs_v3.jsonl")
    ap.add_argument("--review", default=None)
    ap.add_argument("--results", default="eval")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    clusters = load_clusters(a.qa, a.review)
    res = {t: load_results(a.results, t) for t in SYSTEMS}
    ids = sorted(res["base"])
    missing = [i for i in ids if i not in clusters]
    if missing:
        sys.exit(f"no case/cluster for: {missing[:5]}")
    clusters = {i: clusters[i] for i in ids}  # eval set = the questions that were run
    for t, d in res.items():
        if set(d) != set(ids):
            sys.exit(f"{t}: question ids differ from base")
    strata = ["ruling_unseen", "ruling_seen", "statute", "noanswer", "calc"]
    subsets = {"ALL": ids}
    for s in strata:
        subsets[s] = [i for i in ids if strat_of(i) == s]
    rng = random.Random(SEED)
    lines = []

    def p(x=""):
        print(x)
        lines.append(x)

    p("# v3 results (pre-registered analysis)")
    p(f"{len(ids)} questions; cluster bootstrap by case, {N_BOOT} resamples, seed {SEED}.")
    n_cl = {k: len({clusters[i] for i in v}) for k, v in subsets.items()}
    p("Questions / clusters: " + ", ".join(f"{k} {len(v)}/{n_cl[k]}" for k, v in subsets.items()))

    p("\n## Accuracy by system (%, with 95% interval)")
    p("| system | " + " | ".join(subsets) + " |")
    p("|---|" + "---|" * len(subsets))
    for t in SYSTEMS:
        cells = []
        for k, v in subsets.items():
            m, lo, hi = boot_ci(v, res[t], clusters, rng)
            cells.append(f"{100*m:.1f} [{100*lo:.1f}, {100*hi:.1f}]")
        p(f"| {t} | " + " | ".join(cells) + " |")

    def compare(label, a_tag, b_tag, keys):
        p(f"\n**{label}** ({b_tag} minus {a_tag}, points)")
        p("| subset | diff | 95% interval | b better / a better |")
        p("|---|---|---|---|")
        for k in keys:
            v = subsets[k]
            d = {i: res[b_tag][i] - res[a_tag][i] for i in v}
            m, lo, hi = boot_ci(v, d, clusters, rng)
            wins = sum(1 for x in d.values() if x > 0)
            loss = sum(1 for x in d.values() if x < 0)
            p(f"| {k} | {100*m:+.1f} | [{100*lo:+.1f}, {100*hi:+.1f}] | {wins} / {loss} |")

    p("\n## H1: each pipeline component (ablation on base, full pipeline = base)")
    for t, name in [("nocase", "case scoping off"), ("snip1000", "context snippets cut to 1000 chars"),
                    ("nocalc", "calculator off"), ("stripped", "all three removed")]:
        compare(name, "base", t, ["ALL"] + strata)
    p("\n## Noise: identical configuration rerun")
    compare("base rerun", "base", "base_rerun", ["ALL"] + strata)
    p("\n## H2: SFT (ep1) vs base")
    compare("SFT effect", "base", "ep1", ["ruling_unseen", "ALL"] + [s for s in strata if s != "ruling_unseen"])
    p("\n## H3: DPO vs SFT (ep1)")
    compare("DPO effect", "ep1", "dpo", ["ruling_unseen", "ALL"] + [s for s in strata if s != "ruling_unseen"])
    p("\nH2 and H3 are the confirmatory tests (unseen-ruling row). All other rows are exploratory.")

    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()