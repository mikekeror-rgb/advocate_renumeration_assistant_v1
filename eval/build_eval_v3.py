"""
build_eval_v3.py: build a FRESH, frozen-pipeline held-out eval set (v3).

Why: the 62-question set was used while the pipeline was being tuned, so the
pipeline gains are in-sample. v3 is built AFTER the pipeline is frozen
(tag pipeline-frozen-v1) from material no earlier step has touched.

Two stages (run from the project root):

  python eval/build_eval_v3.py generate     # LLM drafts questions, writes review files
  ... you review eval/v3_review.csv by hand ...
  python eval/build_eval_v3.py finalize     # merges your verdicts -> eval/qa_pairs_v3.jsonl

Generation env (use a DIFFERENT model from the judge, gpt-5.4-mini, to limit circularity):
  GEN_BASE_URL, GEN_API_KEY, GEN_MODEL
Optional: V3_SEED (default 20261014), N_RULING_UNSEEN, N_RULING_SEEN, MAX_PER_CASE,
          N_STATUTE, N_NOANSWER, N_CALC

What is excluded:
  * rulings (doc_title) that appear in training/sft_train.jsonl or any eval/qa_pairs*.jsonl
  * chunk ids used as gold anywhere in training or eval
  * any question text already used, and any Kshs amount already used in a calc question
Statute passages cannot be unseen (the Order is in training), so statute rows
are reported as their own stratum, never pooled into the 'unseen' claim.

Calculator gold is HAND-COMPUTED by you (column hand_value), not taken from
fee_router. fee_router's value is kept in a separate file and only compared at
finalize time, to avoid circularity.
"""
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

EVAL_DIR = ROOT / "eval"
TRAIN_FILES = [ROOT / "training" / "sft_train.jsonl", ROOT / "training" / "sft_val.jsonl"]
REVIEW_CSV = EVAL_DIR / "v3_review.csv"
CALC_PROPOSED = EVAL_DIR / "v3_calc_proposed.csv"
OUT_QA = EVAL_DIR / "qa_pairs_v3.jsonl"
REFUSAL = "Not covered in the provided documents."
AMOUNT = re.compile(r"Kshs\.?\s?([\d,]{4,})")
CITATION = re.compile(r"\s*\[chunk_id:[^\]]*\]")

SEED = int(os.environ.get("V3_SEED", "20261014"))
MAX_PER_CASE = int(os.environ.get("MAX_PER_CASE", "7"))
N = {
    "ruling_unseen": int(os.environ.get("N_RULING_UNSEEN", "90")),  # cap, spread over unseen cases
    "ruling_seen": int(os.environ.get("N_RULING_SEEN", "60")),      # 1 per trained-on case
    "statute": int(os.environ.get("N_STATUTE", "40")),
    "noanswer": int(os.environ.get("N_NOANSWER", "40")),
    "calc": int(os.environ.get("N_CALC", "40")),
}


def load_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def case_number(doc_title):
    m = re.search(r"\b[A-Z]*\d+of\d{4}\b", doc_title) or re.search(
        r"\d{4}KE[A-Z]+\d+", doc_title
    )
    return m.group(0) if m else None


def amount_int(s):
    return int(s.replace(",", ""))


# ---------- exclusions ----------
def collect_exclusions():
    used_titles, used_chunks, used_q, used_amounts = set(), set(), set(), set()
    for path in TRAIN_FILES:
        for ex in load_jsonl(path):
            meta = ex.get("meta", {})
            if meta.get("doc_title"):
                used_titles.add(meta["doc_title"])
            if meta.get("gold_chunk_id"):
                used_chunks.add(meta["gold_chunk_id"])
            q = meta.get("question") or ex["messages"][1]["content"].rsplit("Question: ", 1)[-1]
            used_q.add(norm(q))
            used_amounts.update(amount_int(a) for a in AMOUNT.findall(q))
    for path in sorted(EVAL_DIR.glob("qa_pairs*.jsonl")):
        for row in load_jsonl(path):
            used_q.add(norm(row["question"]))
            used_amounts.update(amount_int(a) for a in AMOUNT.findall(row["question"]))
            if row.get("evidence_chunk_id"):
                used_chunks.add(row["evidence_chunk_id"])
    return used_titles, used_chunks, used_q, used_amounts


# ---------- LLM drafting ----------
def gen_client():
    from openai import OpenAI

    client = OpenAI(
        base_url=os.environ.get("GEN_BASE_URL"),
        api_key=os.environ["GEN_API_KEY"],
        timeout=60.0,
        max_retries=1,
    )
    return client, os.environ["GEN_MODEL"]


CACHE_PATH = EVAL_DIR / ".v3_gen_cache.json"


def _cache_load():
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _cache_save(cache):
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _retry_wait(msg):
    """Seconds to wait from a Groq 'try again in 2h13m5.5s' style message."""
    m = re.search(r"try again in\s+((?:\d+(?:\.\d+)?(?:ms|h|m|s)\s*)+)", msg)
    if not m:
        return 30.0
    total = 0.0
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", m.group(1)):
        total += float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total


def ask_json(client, model, prompt, retries=5):
    """Cached, rate-limit-aware JSON call. Finished calls survive a crash or daily cap."""
    import time

    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    cache = _cache_load()
    key = hashlib.sha1((model + "\n" + prompt).encode()).hexdigest()
    if key in cache:
        return cache[key]
    bad, waits = 0, 0
    while bad < retries:
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
            )
        except (APITimeoutError, APIConnectionError, InternalServerError) as e:
            bad += 1
            print(f"  network/server error ({type(e).__name__}), retry {bad}/{retries} in 10s",
                  flush=True)
            time.sleep(10)
            continue
        except RateLimitError as e:
            msg = str(e)
            wait = _retry_wait(msg)
            waits += 1
            if wait > 180 or waits > 8 or "per day" in msg.lower():
                sys.exit(
                    f"\nDaily/long rate limit hit ({msg[:160]}...).\n"
                    f"{len(cache)} calls are cached in {CACHE_PATH.name}. "
                    "Rerun the same command later and it resumes."
                )
            print(f"  rate limited, waiting {wait + 2:.0f}s ...", flush=True)
            time.sleep(wait + 2)
            continue
        text = (r.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                out = json.loads(m.group(0))
                cache[key] = out
                _cache_save(cache)
                return out
            except json.JSONDecodeError:
                pass
        bad += 1
    return None


RULING_PROMPT = """You are writing one exam question for a Kenyan legal assistant,
based ONLY on the passage below, from the ruling "{title}".

{mode_rule}
The question must be fully answerable from the passage alone and have a single
clear answer (a rule, a holding, a figure, a reason). Do not ask about anything
the passage does not state. Then write the reference answer in 1-3 sentences,
using only facts in the passage.

PASSAGE:
{passage}

Reply with JSON only: {{"question": "...", "reference": "..."}}"""

NAMED_RULE = 'The question MUST name the case (e.g. "In {title}, ...").'
UNNAMED_RULE = (
    "The question must NOT mention the case name, any party name or case number; "
    "ask about the legal point in general terms, as a practitioner who does not "
    "know which ruling decided it would."
)

STATUTE_PROMPT = """Write one exam question about the Advocates (Remuneration) Order
based ONLY on the passage below. It must be answerable from the passage alone with
a single clear answer, and must not require any other provision. Then write a
1-3 sentence reference answer using only facts in the passage.

PASSAGE:
{passage}

Reply with JSON only: {{"question": "...", "reference": "..."}}"""

NOANSWER_KINDS = [
    "clearly unrelated to law (cooking, sport, general tech, travel)",
    (
        "near-domain but out of scope: other countries' fee rules (Uganda, Tanzania, "
        "England), other Kenyan statutes (employment, tax, land procedure), or "
        "advocates' discipline rather than fees"
    ),
    (
        "legal-sounding but needing facts the documents cannot contain (a specific "
        "named client's bill, today's court backlog, a judge's private view, a "
        "future ruling)"
    ),
]
NOANSWER_PROMPT = """Write {k} distinct questions that a user of a Kenyan advocates'
remuneration assistant might ask, that are NOT answerable from the Advocates
(Remuneration) Order or Kenyan rulings on taxation of costs. Every question must be
of this kind: {kind}.
This is batch {b}; make the questions different from the obvious ones.
Reply with JSON only: {{"questions": ["...", "..."]}}"""


def pick_chunks(rng, items, k):
    items = items[:]
    rng.shuffle(items)
    return items[:k]


def generate():
    import chromadb

    rng = random.Random(SEED)
    used_titles, used_chunks, used_q, used_amounts = collect_exclusions()
    print(
        f"Excluding {len(used_titles)} titles, {len(used_chunks)} chunks, "
        f"{len(used_q)} questions, {len(used_amounts)} amounts."
    )

    col = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection("policy_docs")
    got = col.get(include=["documents", "metadatas"])
    chunks = [
        {"id": i, "text": d, "meta": m}
        for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
    ]
    print("Sample chunk id/meta (check these match your chunk_id format):")
    print(" ", chunks[0]["id"], chunks[0]["meta"])

    def ruling_chunks(seen):
        return [
            c for c in chunks
            if c["meta"].get("doc_type") == "ruling"
            and (c["meta"].get("doc_title") in used_titles) == seen
            and c["id"] not in used_chunks
            and len(c["text"]) > 600
        ]

    def spread(items, per_case, cap):
        """Up to per_case chunks per case, round-robin across cases, capped at cap."""
        groups = {}
        for c in items:
            groups.setdefault(c["meta"]["doc_title"], []).append(c)
        for g in groups.values():
            rng.shuffle(g)
        titles = list(groups)
        rng.shuffle(titles)
        out = []
        for rnd in range(per_case):
            for t in titles:
                if rnd < len(groups[t]):
                    out.append(groups[t][rnd])
        return out[:cap], len(groups)

    unseen_pick, n_unseen_cases = spread(ruling_chunks(False), MAX_PER_CASE, N["ruling_unseen"])
    seen_pick, n_seen_cases = spread(ruling_chunks(True), 1, N["ruling_seen"])
    statutes = [
        c for c in chunks
        if c["meta"].get("doc_type") != "ruling"
        and c["id"] not in used_chunks
        and len(c["text"]) > 400
    ]
    print(f"Unseen cases: {n_unseen_cases} -> {len(unseen_pick)} passages "
          f"(max {MAX_PER_CASE}/case); trained-on cases with unused passages: "
          f"{n_seen_cases} -> {len(seen_pick)}; unused statute chunks: {len(statutes)}")

    client, model = gen_client()
    rows = []

    def add(rid, typ, question, reference, chunk, extra=None):
        if norm(question) in used_q:
            return
        used_q.add(norm(question))
        row = {
            "id": rid, "type": typ, "question": question, "reference": reference,
            "chunk_id": chunk["id"] if chunk else "",
            "doc_title": chunk["meta"].get("doc_title", "") if chunk else "",
            "passage": (chunk["text"][:1800].replace("\n", " ") if chunk else ""),
            "hand_value": "", "verdict": "", "fixed_question": "", "fixed_reference": "",
        }
        row.update(extra or {})
        rows.append(row)

    # rulings: alternate named / unnamed; status = unseen case vs trained-on case, new passage
    def draft_rulings(picks, status):
        counts = {"named": 0, "unnamed": 0}
        for idx, c in enumerate(picks):
            title = c["meta"]["doc_title"]
            mode = "named" if idx % 2 == 0 else "unnamed"
            rule = NAMED_RULE.format(title=title) if mode == "named" else UNNAMED_RULE
            out = ask_json(client, model, RULING_PROMPT.format(
                title=title, mode_rule=rule, passage=c["text"][:1800]))
            if not out or "question" not in out:
                continue
            counts[mode] += 1
            extra = {"case_status": status}
            if mode == "named":
                extra["doc_hint"] = case_number(title) or ""
            add(f"v3_{status}_{mode}_{counts[mode]:03d}", f"ruling_{mode}",
                out["question"], out["reference"], c, extra)

    draft_rulings(unseen_pick, "unseen")
    draft_rulings(seen_pick, "seen")

    # statute
    for i, c in enumerate(pick_chunks(rng, statutes, int(N["statute"] * 1.4)), 1):
        out = ask_json(client, model, STATUTE_PROMPT.format(passage=c["text"][:1800]))
        if out and "question" in out:
            add(f"v3_statute_{i:03d}", "statute", out["question"], out["reference"], c)

    # no-answer (LLM-drafted, you review; add handwritten guard rows in the CSV too)
    # Small batches: the free tier allows only ~1,000 output tokens/minute.
    per_batch = 8
    n_batches = -(-int(N["noanswer"] * 1.3) // per_batch)
    k = 0
    for b in range(1, n_batches + 1):
        kind = NOANSWER_KINDS[(b - 1) % len(NOANSWER_KINDS)]
        out = ask_json(client, model, NOANSWER_PROMPT.format(k=per_batch, kind=kind, b=b))
        for q in (out or {}).get("questions", []):
            k += 1
            add(f"v3_noanswer_{k:03d}", "noanswer", q, REFUSAL, None)

    # calculator: reuse phrasing of existing calc questions with FRESH amounts
    import fee_router

    templates = [
        r for r in load_jsonl(EVAL_DIR / "qa_pairs.jsonl")
        if r.get("expected_type") == "numeric" and AMOUNT.search(r["question"])
    ]
    proposed = []
    tries = 0
    ci = 0
    while ci < N["calc"] and templates and tries < N["calc"] * 20:
        tries += 1
        t = rng.choice(templates)
        # fresh amount: log-uniform 100k..200M, rounded to 2 significant figures
        raw = 10 ** rng.uniform(5, 8.3)
        mag = 10 ** (int(math.log10(raw)) - 1)
        amt = int(round(raw / mag) * mag)
        if amt in used_amounts:
            continue
        q = AMOUNT.sub(f"Kshs {amt:,}", t["question"], count=1)
        if norm(q) in used_q:
            continue
        route = fee_router.route(q)
        if route is None or getattr(route, "fee", None) is None:
            continue
        used_amounts.add(amt)
        ci += 1
        add(f"v3_calc_{ci:03d}", "calc", q, "", None, {"doc_title": f"template:{t['id']}"})
        proposed.append({"id": f"v3_calc_{ci:03d}", "router_fee": route.fee,
                         "scenario": getattr(route, "scenario", "")})

    fields = ["id", "type", "question", "reference", "hand_value", "verdict",
              "fixed_question", "fixed_reference", "doc_hint", "case_status", "chunk_id",
              "doc_title", "passage"]
    for r in rows:
        r.setdefault("doc_hint", "")
        r.setdefault("case_status", "")
    with open(REVIEW_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(CALC_PROPOSED, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "router_fee", "scenario"])
        w.writeheader()
        w.writerows(proposed)

    counts = {}
    for r in rows:
        counts[r["type"]] = counts.get(r["type"], 0) + 1
    print("Candidates:", counts)
    print(f"Review file: {REVIEW_CSV}")
    print("Fill 'verdict' with keep / fix / drop on every row.")
    print("  fix  -> put corrected text in fixed_question / fixed_reference")
    print("  calc -> compute the fee BY HAND from the Order and put it in hand_value")
    print("DO NOT open v3_calc_proposed.csv until every hand_value is filled.")


# ---------- finalize ----------
def finalize():
    if not REVIEW_CSV.exists():
        sys.exit("Run 'generate' first.")
    with open(REVIEW_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    router = {}
    if CALC_PROPOSED.exists():
        with open(CALC_PROPOSED, encoding="utf-8-sig") as f:
            router = {r["id"]: r for r in csv.DictReader(f)}

    out, dropped, unreviewed, disagreements = [], 0, 0, []
    for r in rows:
        v = r["verdict"].strip().lower()
        if v not in {"keep", "fix", "drop"}:
            unreviewed += 1
            continue
        if v == "drop":
            dropped += 1
            continue
        q = r["fixed_question"].strip() if v == "fix" and r["fixed_question"].strip() else r["question"]
        ref = r["fixed_reference"].strip() if v == "fix" and r["fixed_reference"].strip() else r["reference"]
        ref = CITATION.sub("", ref).strip()
        t = r["type"]
        base = {"id": r["id"], "question": q, "stratum": t, "needs_review": False,
                "evidence_chunk_id": r["chunk_id"], "case": r["doc_title"],
                "case_status": r.get("case_status", ""), "notes": "v3 fresh set; hand-verified."}
        if t == "calc":
            if not r["hand_value"].strip():
                unreviewed += 1
                continue
            hv = float(r["hand_value"].replace(",", ""))
            rf = float(router[r["id"]]["router_fee"]) if r["id"] in router else None
            if rf is not None and abs(rf - hv) > 0.5:
                disagreements.append((r["id"], hv, rf))
            base.update(expected_type="numeric", expected_value=int(hv) if hv.is_integer() else hv,
                        expected_source="statute")
        elif t == "noanswer":
            base.update(expected_type="no_answer", expected_value=REFUSAL, expected_source="none")
        else:
            base.update(expected_type="factual", expected_value=ref,
                        expected_source="ruling" if t.startswith("ruling") else "statute")
            if t == "ruling_named" and r["doc_hint"]:
                base["doc_hint"] = r["doc_hint"]
        out.append(base)

    if unreviewed:
        sys.exit(f"{unreviewed} rows have no verdict (or calc rows lack hand_value). Finish review first.")
    with open(OUT_QA, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in out)
    digest = hashlib.sha256(OUT_QA.read_bytes()).hexdigest()[:16]
    counts = {}
    for r in out:
        counts[r["stratum"]] = counts.get(r["stratum"], 0) + 1
    print(f"Wrote {len(out)} questions to {OUT_QA.name} (dropped {dropped}). sha256[:16]={digest}")
    print("By stratum:", counts)
    if disagreements:
        print("\nHAND vs fee_router DISAGREEMENTS (check each: calculator bug or your arithmetic?):")
        for i, hv, rf in disagreements:
            print(f"  {i}: hand={hv:,.0f} router={rf:,.0f}")
    print("\nNext: write eval/PREREG.md with this hash BEFORE running any model.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "generate":
        generate()
    elif cmd == "finalize":
        finalize()
    else:
        sys.exit(__doc__)
