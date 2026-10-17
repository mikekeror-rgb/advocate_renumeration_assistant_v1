"""
v3_precheck.py: add a 'flag' column to eval/v3_review.csv marking rows that are
likely bad, so you can triage them first. It never changes your verdicts or any
other column, and it never drops rows. Safe to rerun.

Run from the project root:  python eval/v3_precheck.py
"""
import csv
import re
from pathlib import Path

PATH = Path(__file__).resolve().parent / "v3_review.csv"
STOP = {
    "the", "and", "others", "other", "county", "kenya", "limited", "ltd", "petition",
    "republic", "ex", "parte", "of", "in", "re", "estate", "civil", "case", "no",
    "high", "court", "appeal", "application", "miscellaneous", "cause", "suit",
    "advocate", "advocates", "bill", "costs", "taxation",
}
PASSAGE_REFS = (
    "the passage", "the text", "this passage", "the excerpt", "the above",
    "according to the passage", "as stated above", "the following",
)
DOMAIN_WORDS = (
    "advocate", "remuneration", "taxation", "bill of costs", "instruction fee",
    "taxing master", "schedule", "disbursement", "retainer", "legal fees",
)


def words(s):
    return {w.lower() for w in re.findall(r"[A-Za-z]{4,}", s)} - STOP


def numbers(s):
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]{2,}", s)}


def main():
    with open(PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames)
    if "flag" not in fields:
        fields.append("flag")

    seen = {}
    for r in rows:
        flags = []
        q = (r.get("fixed_question") or r["question"]).strip()
        ref = (r.get("fixed_reference") or r["reference"]).strip()
        typ, title, passage = r["type"], r.get("doc_title", ""), r.get("passage", "")
        ql = q.lower()

        if len(q) < 25:
            flags.append("short_q")
        if any(p in ql for p in PASSAGE_REFS):
            flags.append("refers_to_passage")
        if typ != "calc":  # calc questions share templates by design
            key = re.sub(r"\W+", " ", ql)[:110]
            if key in seen:
                flags.append(f"dup_of_{seen[key]}")
            else:
                seen[key] = r["id"]

        if typ == "ruling_unnamed":
            leak = words(title) & words(q)
            if leak or re.search(r"\d+\s*of\s*\d{4}", ql):
                flags.append("names_case:" + ",".join(sorted(leak)[:3]))
        if typ == "ruling_named":
            first = sorted(words(title.split(" v ")[0]))
            if first and not any(w in ql for w in first):
                flags.append("named_but_missing_party")

        if typ in ("ruling_named", "ruling_unnamed", "statute"):
            missing = numbers(ref) - numbers(passage)
            if missing:
                flags.append("ref_number_not_in_passage:" + ",".join(sorted(missing)[:3]))
            if len(ref) > 600:
                flags.append("long_ref")
            if len(ref) < 20:
                flags.append("short_ref")

        if typ == "noanswer" and any(w in ql for w in DOMAIN_WORDS):
            flags.append("check_answerable_from_corpus")

        r["flag"] = "; ".join(flags)

    with open(PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    flagged = [r for r in rows if r["flag"]]
    print(f"{len(flagged)} of {len(rows)} rows flagged.")
    by = {}
    for r in flagged:
        for fl in r["flag"].split("; "):
            k = fl.split(":")[0].split("_of_")[0]
            by[k] = by.get(k, 0) + 1
    for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    print("Filter the 'flag' column for non-empty cells and review those first.")


if __name__ == "__main__":
    main()
