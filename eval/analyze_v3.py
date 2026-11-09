"""
analyze_v3.py: pre-registered analysis for the v3 held-out set (eval/PREREG.md).

Paired differences between systems, 95% percentile intervals from a cluster
bootstrap (clusters = case for ruling questions; every other question is its
own cluster), 10,000 resamples.

The last sections (false refusals, evidence chunk recall, invented citations) are exploratory and not
part of the pre-registration. It runs after the pre-registered sections, so
their bootstrap draws and numbers are unchanged.

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import citations  # noqa: E402

SEED = 20261017
N_BOOT = 10000
SYSTEMS = ["base", "base_rerun", "ep1", "dpo", "nocase", "snip1000", "nocalc", "stripped"]
REFUSAL = "not covered in the provided documents"  # same check as eval_v2.EXPECTED_REFUSAL


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
    """id -> full result row."""
    f = Path(res_dir) / f"results_ev3_{tag}.csv"
    out = {}
    for r in csv.DictReader(open(f, encoding="utf-8-sig")):
        if r["answer_correct"] == "":
            sys.exit(f"{f}: ungraded row {r['id']}")
        out[r["id"]] = r
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
    rows = {t: load_results(a.results, t) for t in SYSTEMS}
    res = {t: {i: 1 if r["answer_correct"] == "True" else 0 for i, r in d.items()} for t, d in rows.items()}
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

    # Exploratory diagnostics, not pre-registered. Must stay after the sections
    # above: it draws from the same rng, and moving it would change their intervals.
    answerable = {k: [i for i in v if strat_of(i) != "noanswer"] for k, v in subsets.items() if k != "noanswer"}
    p("\n## Exploratory: false refusals (not pre-registered)")
    p('Share of answerable questions (every stratum except noanswer) where the answer is the '
      'refusal "Not covered in the provided documents". Lower is better. Correctness scoring '
      "can't separate a refusal from a wrong answer; this can.")
    p("| system | " + " | ".join(answerable) + " |")
    p("|---|" + "---|" * len(answerable))
    for t in SYSTEMS:
        refused = {i: 1 if REFUSAL in r["answer"].lower() else 0 for i, r in rows[t].items()}
        cells = []
        for k, v in answerable.items():
            m, lo, hi = boot_ci(v, refused, clusters, rng)
            cells.append(f"{100*m:.1f} [{100*lo:.1f}, {100*hi:.1f}] ({sum(refused[i] for i in v)}/{len(v)})")
        p(f"| {t} | " + " | ".join(cells) + " |")

    labelled = [i for i in ids if rows["base"][i]["evidence_retrieved"] in ("visible", "truncated", "missing")]
    with_ev = {k: [i for i in v if i in set(labelled)] for k, v in subsets.items()}
    with_ev = {k: v for k, v in with_ev.items() if v}
    p("\n## Exploratory: evidence chunk recall (not pre-registered)")
    p("Share of questions with a gold evidence chunk where that chunk was among the retrieved "
      "chunks. Higher is better. Measured on the chunk ID, so it does not see the context snippet "
      "cut-off: the snip1000 ablation keeps the same recall while losing accuracy.")
    p("retrieval_hit is not reported: passes by construction on 171/236 questions; replaced by evidence recall.")
    p("| system | " + " | ".join(with_ev) + " |")
    p("|---|" + "---|" * len(with_ev))
    for t in SYSTEMS:
        hit = {i: 1 if rows[t][i]["evidence_retrieved"] in ("visible", "truncated") else 0 for i in labelled}
        cells = []
        for k, v in with_ev.items():
            m, lo, hi = boot_ci(v, hit, clusters, rng)
            cells.append(f"{100*m:.1f} [{100*lo:.1f}, {100*hi:.1f}] ({sum(hit[i] for i in v)}/{len(v)})")
        p(f"| {t} | " + " | ".join(cells) + " |")

    corpus = citations.load_corpus()
    p("\n## Exploratory: invented citations (not pre-registered)")
    p("Answers citing a chunk ID that names a real document but a section it doesn't have "
      "(e.g. __s10__c1 for a ruling whose chunks end at s2). Counted from the answer text "
      "against data/chunks.jsonl, so it needs no retrieval log. Lower is better.")
    p("| system | answers with an invented citation |")
    p("|---|---|")
    for t in SYSTEMS:
        n = sum(bool(citations.citations_invented(r["answer"], corpus)) for r in rows[t].values())
        p(f"| {t} | {n} |")

    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()