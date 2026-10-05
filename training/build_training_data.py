"""
Day 4: build SFT training data for LoRA fine-tuning Qwen2.5-7B-Instruct.

The fine-tune teaches BEHAVIOUR. Facts come from retrieval and the fee calculator,
so the model is never asked to memorise the Order or the rulings. Four example kinds:

  grounded  A question that names a ruling (or asks about the Order), answered from the
            passage that holds the answer, sitting among REAL retrieved distractors.
            Teaches: answer from the named case, attribute correctly, name every
            authority relied on, cite properly. (Eval failures: ruling_03, ruling_04.)
  refusal   The same kind of question with every chunk of that ruling removed from the
            context. Target is the exact refusal. Teaches: don't guess outcomes that
            aren't in the context. (Eval: ruling_02's lucky guess.)
  calc      A VERIFIED CALCULATION is present. Target states the figure and its source
            and nothing else. Teaches: never recompute or copy figures from rulings.
  offscope  Out-of-scope question. Target is the exact refusal.

Every model input is built by the REAL pipeline (retrieve() and the same prompt
builder generate() uses), so training data matches inference exactly.

Never trains on eval material: rulings named in eval/qa_pairs.jsonl, statute chunks
containing an eval content_hint, eval questions, and eval calculator amounts are all
excluded. The validation split is grouped by ruling, so no case appears in both.

Teacher LLM (writes grounded questions + answers), any OpenAI-compatible endpoint:
  TEACHER_BASE_URL  default https://api.groq.com/openai/v1
  TEACHER_API_KEY   default $GROQ_API_KEY
  TEACHER_MODEL     default openai/gpt-oss-20b (Apache-2.0, so its outputs can be used for training)
Self-hosted teacher (no daily cap): run `ollama pull gpt-oss:20b` on the GPU pod, then set
  TEACHER_BASE_URL=https://<pod-id>-11434.proxy.runpod.net/v1 TEACHER_API_KEY=ollama TEACHER_MODEL=gpt-oss:20b
Teacher replies are cached in training/teacher_cache.jsonl. If a rate limit stops the
run, just run it again later: it resumes where it stopped.

Sizes: N_RULING (300), N_STATUTE (100), N_CALC (60), REFUSAL_SHARE (0.2), SEED (42)

Outputs (training/):
  sft_train.jsonl, sft_val.jsonl   {"messages": [system, user, assistant], "meta": {...}}
  rejected_log.jsonl               teacher outputs that failed validation, with the reason
  build_stats.json                 counts per kind and rejection reasons

Run from the project root: python training/build_training_data.py
"""

import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI, RateLimitError  # noqa: E402

import case_matcher  # noqa: E402
import fee_router  # noqa: E402
import rag_pipeline_v2 as rp  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = OUT_DIR / "sft_train.jsonl"
VAL_PATH = OUT_DIR / "sft_val.jsonl"
CACHE_PATH = OUT_DIR / "teacher_cache.jsonl"
REJECTS_PATH = OUT_DIR / "rejected_log.jsonl"
STATS_PATH = OUT_DIR / "build_stats.json"
EVAL_QA_PATH = ROOT / "eval" / "qa_pairs.jsonl"

SEED = int(os.environ.get("SEED", "42"))
N_RULING = int(os.environ.get("N_RULING", "300"))
N_STATUTE = int(os.environ.get("N_STATUTE", "100"))
N_CALC = int(os.environ.get("N_CALC", "60"))
REFUSAL_SHARE = float(os.environ.get("REFUSAL_SHARE", "0.2"))
OVERSAMPLE = 1.3       # ask the teacher for extra, since some outputs fail validation
MAX_PER_CASE = 3       # spread examples across many rulings
MIN_CHUNK_CHARS = 400  # skip near-empty chunks
VAL_SHARE = 0.1

TEACHER_BASE_URL = os.environ.get("TEACHER_BASE_URL", "https://api.groq.com/openai/v1")
TEACHER_API_KEY = os.environ.get("TEACHER_API_KEY") or os.environ.get("GROQ_API_KEY")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "openai/gpt-oss-20b")

REFUSAL = "Not covered in the provided documents."
CITE_TOKEN = "[CITE]"
CONTEXT_SIZE = rp.TOP_K + 2  # retrieve() returns top_k + statute_boost chunks


class TeacherQuotaExhausted(Exception):
    """Raised when the teacher's rate limit needs a long wait (e.g. a daily token cap)."""


# ---------------------------------------------------------------------------
# Teacher prompts
# ---------------------------------------------------------------------------

RULING_PROMPT = """You are creating training data for a legal research assistant that answers \
questions about Kenyan court rulings on advocates' costs.

Below is ONE passage from the ruling "{case_name}" (section: {section}).

PASSAGE:
\"\"\"{passage}\"\"\"

Write ONE question a lawyer might realistically ask that this passage answers, and the ideal answer.

Question rules:
- Name the case in the question exactly as "{case_name}".
- Ask about something the passage actually states: what the court held, found or ordered, its \
reasoning, the authorities it relied on, or what a party argued.
- Do not ask about anything the passage does not state.

Answer rules:
- Use ONLY the passage. No outside knowledge.
- Be concise: 1-4 sentences.
- If the passage reports a party's submission rather than the court's finding, say it was that \
party's argument.
- If the passage relies on more than one authority or case for the point, name every one of them.
- End each sentence that relies on the passage with the marker [CITE]. It will be replaced with \
the citation, so do not write any citation yourself.

Respond with ONLY a JSON object: {{"question": "...", "answer": "..."}}"""

STATUTE_PROMPT = """You are creating training data for a legal research assistant that answers \
questions about Kenya's Advocates Remuneration Order.

Below is ONE passage from THE ADVOCATES REMUNERATION ORDER.

PASSAGE:
\"\"\"{passage}\"\"\"

Write ONE question a lawyer might realistically ask that this passage answers, and the ideal answer.

Question rules:
- Ask about a rule, time limit, rate, figure or procedure the passage actually states.
- Do not ask for a fee calculation on a specific amount.
- Do not ask about anything the passage does not state.

Answer rules:
- Use ONLY the passage. No outside knowledge.
- Be concise: 1-3 sentences. State any figure exactly as the passage gives it.
- End each sentence that relies on the passage with the marker [CITE]. It will be replaced with \
the citation, so do not write any citation yourself.

Respond with ONLY a JSON object: {{"question": "...", "answer": "..."}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def short_name(doc_title: str) -> str:
    """'Kariithi v GN Thiongo Associates (Misc...) ...' -> 'Kariithi v GN Thiongo Associates'"""
    name = re.sub(r"\s+", " ", doc_title.split("(")[0]).strip()
    return re.sub(r"\s+\d{4}\s*[A-Z]{2,}\S*$", "", name)  # drop a trailing neutral citation, e.g. 2021KEHC6864


def citation_for(chunk: dict) -> str:
    label = "Advocates Remuneration Order" if chunk["doc_type"] == "statute" else short_name(chunk["doc_title"])
    return f"[chunk_id: {chunk['chunk_id']}, {label}]"


def chunk_dict(pipeline, idx: int) -> dict:
    """Same shape as the chunks retrieve() returns."""
    meta = pipeline._bm25_metadatas[idx]
    return {
        "chunk_id": pipeline._bm25_ids[idx],
        "text": pipeline._bm25_documents[idx],
        "doc_title": meta["doc_title"],
        "section": meta["section"],
        "source_url": meta["source_url"],
        "doc_type": meta.get("doc_type", "Unknown"),
        "case_number": meta.get("case_number", "Unknown"),
        "court": meta.get("court", "Unknown"),
        "ruling_date": meta.get("ruling_date", "Unknown"),
        "case_citation": meta.get("case_citation", "Unknown"),
    }


def parse_json_reply(raw: str) -> dict | None:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
    for candidate in (raw, *re.findall(r"\{.*\}", raw, re.DOTALL)):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def retry_after_seconds(message: str) -> float | None:
    """Groq phrases waits as 'Please try again in 3m17.856s' or 'in 12.5s'."""
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", message)
    if not match:
        return None
    return int(match.group(1) or 0) * 60 + float(match.group(2))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Eval exclusions
# ---------------------------------------------------------------------------

def load_eval_exclusions(case_index: dict) -> dict:
    qa_pairs = load_jsonl(EVAL_QA_PATH)
    titles, content_hints, questions, amounts = set(), [], set(), set()
    for qa in qa_pairs:
        questions.add(qa["question"].strip().lower())
        text = f"{qa['question']} {qa.get('expected_value', '')}"
        titles.update(case_matcher.match_cases(text, case_index))  # cases asked about or named as authorities
        doc_hint = qa.get("doc_hint")
        if doc_hint:
            titles.update(t for t in case_index if doc_hint.lower() in t.lower())
        if qa.get("content_hint"):
            content_hints.append(qa["content_hint"].lower())
        if qa["expected_type"] == "numeric":
            amount = fee_router.extract_amount(qa["question"])
            if amount:
                amounts.add(round(amount))
    return {"titles": titles, "content_hints": content_hints, "questions": questions, "amounts": amounts}


# ---------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------

def make_teacher() -> OpenAI:
    if not TEACHER_API_KEY:
        raise RuntimeError("No teacher API key: set TEACHER_API_KEY (or GROQ_API_KEY for the default Groq teacher).")
    return OpenAI(base_url=TEACHER_BASE_URL, api_key=TEACHER_API_KEY, timeout=120, max_retries=0)


def call_teacher(client: OpenAI, prompt: str) -> str:
    kwargs = {"model": TEACHER_MODEL, "messages": [{"role": "user", "content": prompt}]}
    if "gpt-oss" in TEACHER_MODEL:
        # gpt-oss is a reasoning model: hidden reasoning tokens share the output budget,
        # so give it room, and use low effort so replies come back quickly.
        kwargs["extra_body"] = {"reasoning_effort": "low"}
        kwargs["temperature"] = 0.7
        if "groq.com" in TEACHER_BASE_URL:
            kwargs["max_completion_tokens"] = 2048
        else:
            kwargs["max_tokens"] = 4096  # e.g. self-hosted via Ollama's OpenAI-compatible endpoint
    elif TEACHER_MODEL.startswith(("gpt-5", "o1", "o3", "o4")):
        kwargs["max_completion_tokens"] = 2048  # reasoning models: no temperature
    else:
        kwargs["max_tokens"] = 1024
        kwargs["temperature"] = 0.7

    for attempt in range(6):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except RateLimitError as error:
            wait = retry_after_seconds(str(error)) or 20 * (attempt + 1)
            if wait > 300 or "per day" in str(error).lower():
                raise TeacherQuotaExhausted(str(error)) from error
            print(f"    rate limited, waiting {wait:.0f}s")
            time.sleep(wait + 1)
    raise TeacherQuotaExhausted("Rate limit persisted after 6 retries.")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_grounded(item: dict | None, gold: dict, exclusions: dict, case_index: dict) -> str | None:
    """Returns a rejection reason, or None if the teacher output is usable."""
    if not item:
        return "unparseable JSON"
    question, answer = item.get("question"), item.get("answer")
    if not isinstance(question, str) or not isinstance(answer, str):
        return "missing question or answer"
    question, answer = question.strip(), answer.strip()
    if not 15 <= len(question) <= 400:
        return "question length"
    if not 30 <= len(answer) <= 1500:
        return "answer length"
    if CITE_TOKEN not in answer:
        return "no [CITE] marker"
    if "chunk_id" in answer.lower():
        return "teacher wrote its own citation"
    if "not covered" in answer.lower():
        return "teacher refused"
    if question.lower() in exclusions["questions"]:
        return "duplicates an eval question"
    if gold["doc_type"] == "ruling":
        if gold["doc_title"] not in case_matcher.match_cases(question, case_index):
            return "question doesn't name the case"
    elif any(hint in question.lower() for hint in exclusions["content_hints"]):
        return "overlaps an eval statute question"
    return None


# ---------------------------------------------------------------------------
# Context builders (the real retriever, so distractors are realistic)
# ---------------------------------------------------------------------------

def context_with_gold(pipeline, question: str, gold: dict, rng: random.Random) -> tuple[list[dict], bool]:
    chunks = pipeline.retrieve(question)
    if any(c["chunk_id"] == gold["chunk_id"] for c in chunks):
        return chunks, True
    chunks = chunks[:-1]  # retrieval missed it: insert it so the target answer is grounded
    chunks.insert(rng.randrange(len(chunks) + 1), gold)
    return chunks, False


def context_without_case(pipeline, question: str, doc_title: str) -> list[dict]:
    """Remove EVERY ruling the question names, not just the exact source title: the corpus
    can hold the same ruling under slightly different titles, and a refusal example whose
    context still contains the case teaches the model to refuse answerable questions."""
    excluded = set(case_matcher.match_cases(question, pipeline._case_index)) | {doc_title}
    chunks = pipeline.retrieve(question, top_k=15)
    return [c for c in chunks if c["doc_title"] not in excluded][:CONTEXT_SIZE]


def make_example(pipeline, question: str, chunks: list[dict], answer: str, kind: str, meta: dict,
                 calculator_result=None) -> dict:
    return {
        "messages": [
            {"role": "system", "content": rp.SYSTEM_PROMPT},
            {"role": "user", "content": pipeline.build_user_message(question, chunks, calculator_result)},
            {"role": "assistant", "content": answer},
        ],
        "meta": {"kind": kind, **meta},
    }


# ---------------------------------------------------------------------------
# Calculator examples (deterministic, no teacher)
# ---------------------------------------------------------------------------

CALC_TEMPLATES = [
    ("What is the instruction fee for a {d} High Court suit worth {amt}?", {"d": ["defended", "undefended"]}),
    ("Calculate the instruction fee in a {d} suit in the High Court where the subject matter is {amt}",
     {"d": ["defended", "undefended"]}),
    ("What is the subordinate court instruction fee for a {amt} claim on the {s} scale?", {"s": ["lower", "higher"]}),
    ("What is the instruction fee in the magistrate's court for a claim of {amt} on the {s} scale?",
     {"s": ["lower", "higher"]}),
    ("What is the instruction fee for the sale of land worth {amt}?", {}),
    ("What is the conveyancing fee for a property purchase of {amt}?", {}),
    ("What is the fee for the {p} advocate for creating a charge over a loan of {amt}?",
     {"p": ["grantee's", "grantor's"]}),
    ("What is the discharge fee for a charge over a {amt} loan{u}?",
     {"u": [", where an undertaking is required for redemption", ", with no undertaking required"]}),
]


def format_amount(value: int, rng: random.Random) -> str:
    """Vary the format so the model sees every style fee_router accepts."""
    style = rng.choice(["kshs", "comma", "bare", "million"])
    if style == "comma":
        return f"{value:,}"
    if style == "bare":
        return str(value)
    if style == "million" and value >= 1_000_000 and value % 100_000 == 0:
        return f"{value / 1_000_000:g} million"
    return f"Kshs {value:,}"


def build_calc_examples(pipeline, rng: random.Random, exclusions: dict, stats: Counter) -> list[dict]:
    examples, attempts = [], 0
    while len(examples) < N_CALC and attempts < N_CALC * 10:
        attempts += 1
        template, options = rng.choice(CALC_TEMPLATES)
        amount = rng.randrange(1, 400) * 50_000  # 50,000 to 19,950,000
        if amount in exclusions["amounts"]:
            continue
        fills = {key: rng.choice(values) for key, values in options.items()}
        question = template.format(amt=format_amount(amount, rng), **fills)
        try:
            route = fee_router.route(question)
        except Exception as error:  # calculator may reject out-of-range amounts
            stats[f"calc skipped: {type(error).__name__}"] += 1
            continue
        if route is None or route.scenario == "schedule6_full_bill":
            stats["calc skipped: not routed"] += 1
            continue
        answer = f"{route.explanation} [source: {route.schedule_citation}]"
        chunks = pipeline.retrieve(question)
        examples.append(make_example(pipeline, question, chunks, answer, "calc",
                                     {"scenario": route.scenario}, calculator_result=route))
    return examples


# ---------------------------------------------------------------------------
# Out-of-scope examples (deterministic, no teacher)
# ---------------------------------------------------------------------------

OFFSCOPE_QUESTIONS = [
    "What is the hourly rate for a corporate lawyer in New York?",
    "How much does it cost to register a company in Uganda?",
    "What are the court filing fees in Tanzania's High Court?",
    "What is the VAT rate in South Africa?",
    "How do I apply for a Kenyan passport?",
    "What is the penalty for late filing of income tax returns in Kenya?",
    "What are the requirements for a valid will in England?",
    "How long does an uncontested divorce take in Nigeria?",
    "What is the current minimum wage in Kenya?",
    "What is the limitation period for contract claims in California?",
    "How are barristers' fees assessed in Australia?",
    "What is the fee for filing a patent application in the United States?",
    "What is the stamp duty rate for residential property purchases in England?",
    "Who is the current Chief Justice of Kenya?",
    "What are the rules on advertising by advocates in Uganda?",
    "How much does a notary public charge in Germany?",
    "What is the Central Bank of Kenya's current policy rate?",
    "What qualifications are needed to be admitted as an advocate in India?",
    "How do I register a trademark in the European Union?",
    "What is the court fee for a small claims case in Ontario?",
    "What is the capital of Rwanda?",
    "What are the conveyancing fees for buying property in Scotland?",
    "What is the fee for filing a divorce application in the UK family court?",
    "How do I appeal a traffic fine in Kenya?",
    "How do retainer fees work for lawyers in Canada?",
    "How much is a single business permit in Mombasa?",
    "What are the arbitration fees at the ICC International Court of Arbitration?",
    "What is the legal drinking age in Kenya?",
    "What are the advocates' remuneration rules in Tanzania?",
    "How is legal aid funded in Ghana?",
]


def build_offscope_examples(pipeline, exclusions: dict) -> list[dict]:
    examples = []
    for question in OFFSCOPE_QUESTIONS:
        if question.strip().lower() in exclusions["questions"]:
            continue
        chunks = pipeline.retrieve(question)
        examples.append(make_example(pipeline, question, chunks, REFUSAL, "offscope", {}))
    return examples


# ---------------------------------------------------------------------------
# Grounded + refusal examples (teacher)
# ---------------------------------------------------------------------------

def pick_gold_chunks(pipeline, exclusions: dict, rng: random.Random) -> tuple[list[int], list[int]]:
    metas, docs = pipeline._bm25_metadatas, pipeline._bm25_documents

    by_case = defaultdict(list)
    statute = []
    for idx, meta in enumerate(metas):
        if len(docs[idx]) < MIN_CHUNK_CHARS:
            continue
        if meta.get("doc_type") == "ruling" and meta["doc_title"] not in exclusions["titles"]:
            by_case[meta["doc_title"]].append(idx)
        elif meta.get("doc_type") == "statute":
            if not any(hint in docs[idx].lower() for hint in exclusions["content_hints"]):
                statute.append(idx)

    titles = sorted(by_case)
    rng.shuffle(titles)
    for title in titles:
        rng.shuffle(by_case[title])
    ruling_target = int(N_RULING * OVERSAMPLE)
    ruling_picks = []
    for round_number in range(MAX_PER_CASE):  # round-robin: one chunk per case per round
        for title in titles:
            if len(ruling_picks) >= ruling_target:
                break
            if round_number < len(by_case[title]):
                ruling_picks.append(by_case[title][round_number])

    rng.shuffle(statute)
    return ruling_picks, statute[: int(N_STATUTE * OVERSAMPLE)]


def build_grounded_examples(pipeline, client, picks: list[int], exclusions: dict, rng: random.Random,
                            stats: Counter, cache: dict, target: int) -> list[dict]:
    examples = []
    for n, idx in enumerate(picks, start=1):
        if len(examples) >= target:
            break
        gold = chunk_dict(pipeline, idx)
        if gold["chunk_id"] in cache:
            raw = cache[gold["chunk_id"]]
        else:
            prompt = (
                RULING_PROMPT.format(case_name=short_name(gold["doc_title"]), section=gold["section"],
                                     passage=gold["text"][: rp.CONTEXT_SNIPPET_CHARS])
                if gold["doc_type"] == "ruling"
                else STATUTE_PROMPT.format(passage=gold["text"][: rp.CONTEXT_SNIPPET_CHARS])
            )
            raw = call_teacher(client, prompt)  # may raise TeacherQuotaExhausted
            if not raw.strip():
                stats["rejected: empty teacher reply"] += 1
                continue  # not cached, so a rerun tries this chunk again
            cache[gold["chunk_id"]] = raw
            append_jsonl(CACHE_PATH, {"chunk_id": gold["chunk_id"], "raw": raw})

        item = parse_json_reply(raw)
        if item and isinstance(item.get("answer"), str):
            answer_text = item["answer"].strip()
            if CITE_TOKEN not in answer_text and "not covered" not in answer_text.lower():
                # Teacher forgot the marker. Answers are 1-4 sentences from ONE passage,
                # so a single citation at the end is accurate.
                item["answer"] = (answer_text[:-1] + f" {CITE_TOKEN}." if answer_text.endswith(".")
                                  else f"{answer_text} {CITE_TOKEN}")
                stats["salvaged: [CITE] appended"] += 1
        reason = validate_grounded(item, gold, exclusions, pipeline._case_index)
        if reason:
            stats[f"rejected: {reason}"] += 1
            append_jsonl(REJECTS_PATH, {"chunk_id": gold["chunk_id"], "reason": reason, "raw": (raw or "")[:600]})
            continue

        question = item["question"].strip()
        answer = item["answer"].strip().replace(CITE_TOKEN, citation_for(gold))
        chunks, retrieved = context_with_gold(pipeline, question, gold, rng)
        stats["gold retrieved by pipeline" if retrieved else "gold inserted (retrieval missed it)"] += 1
        meta = {"doc_title": gold["doc_title"], "gold_chunk_id": gold["chunk_id"], "doc_type": gold["doc_type"],
                "question": question}
        examples.append(make_example(pipeline, question, chunks, answer, "grounded", meta))
        if n % 10 == 0:
            print(f"  {gold['doc_type']}: {len(examples)} accepted from {n} tried")
    return examples


def build_refusal_examples(pipeline, grounded: list[dict], rng: random.Random) -> list[dict]:
    ruling_examples = [e for e in grounded if e["meta"]["doc_type"] == "ruling"]
    count = int(len(ruling_examples) * REFUSAL_SHARE)
    examples = []
    for source in rng.sample(ruling_examples, min(count, len(ruling_examples))):
        question = source["meta"]["question"]
        chunks = context_without_case(pipeline, question, source["meta"]["doc_title"])
        examples.append(make_example(pipeline, question, chunks, REFUSAL, "refusal",
                                     {"doc_title": source["meta"]["doc_title"], "question": question}))
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def split_train_val(examples: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Ruling examples are split by case, so no ruling appears in both sets."""
    titles = sorted({e["meta"]["doc_title"] for e in examples if e["meta"].get("doc_title")})
    val_titles = set(rng.sample(titles, max(1, int(len(titles) * VAL_SHARE)))) if titles else set()
    train, val = [], []
    for example in examples:
        title = example["meta"].get("doc_title")
        is_val = (title in val_titles) if title else (rng.random() < VAL_SHARE)
        (val if is_val else train).append(example)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main() -> None:
    rng = random.Random(SEED)
    stats = Counter()

    print("Loading pipeline (retrieval only, no generation calls)...")
    pipeline = rp.RagPipeline()
    if not hasattr(pipeline, "build_user_message"):
        raise RuntimeError("rag_pipeline_v2.RagPipeline needs build_user_message(); see the Day 4 instructions.")

    exclusions = load_eval_exclusions(pipeline._case_index)
    print(f"Excluding {len(exclusions['titles'])} eval rulings, {len(exclusions['content_hints'])} eval statute "
          f"passages and {len(exclusions['amounts'])} eval amounts.")

    cache = {row["chunk_id"]: row["raw"] for row in load_jsonl(CACHE_PATH)}
    print(f"Teacher: {TEACHER_MODEL} at {TEACHER_BASE_URL} ({len(cache)} cached replies)")
    client = make_teacher()

    ruling_picks, statute_picks = pick_gold_chunks(pipeline, exclusions, rng)
    print(f"Candidates: {len(ruling_picks)} ruling chunks, {len(statute_picks)} statute chunks\n")

    grounded, stopped_early = [], None
    try:
        print("Grounded ruling examples...")
        grounded += build_grounded_examples(pipeline, client, ruling_picks, exclusions, rng, stats, cache, N_RULING)
        print("Grounded statute examples...")
        grounded += build_grounded_examples(pipeline, client, statute_picks, exclusions, rng, stats, cache, N_STATUTE)
    except TeacherQuotaExhausted as error:
        stopped_early = str(error)
        print(f"\n⚠ Teacher quota exhausted, stopping early. Progress is cached; rerun later to continue.\n  {str(error)[:300]}")

    print("Refusal examples...")
    refusals = build_refusal_examples(pipeline, grounded, rng)
    print("Calculator examples...")
    calcs = build_calc_examples(pipeline, rng, exclusions, stats)
    print("Out-of-scope examples...")
    offscope = build_offscope_examples(pipeline, exclusions)

    examples = grounded + refusals + calcs + offscope
    train, val = split_train_val(examples, rng)
    write_jsonl(TRAIN_PATH, train)
    write_jsonl(VAL_PATH, val)

    kinds = Counter(e["meta"]["kind"] + (f"/{e['meta']['doc_type']}" if e["meta"]["kind"] == "grounded" else "")
                    for e in examples)
    summary = {
        "teacher": TEACHER_MODEL,
        "stopped_early": stopped_early,
        "train": len(train),
        "val": len(val),
        "by_kind": dict(kinds),
        "events": dict(stats),
    }
    STATS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Training data ===")
    print(f"train: {len(train)}  val: {len(val)}")
    for kind, count in sorted(kinds.items()):
        print(f"  {kind:<18} {count}")
    for event, count in sorted(stats.items()):
        print(f"  [{event}] {count}")
    if stopped_early:
        print("\nIncomplete: the teacher hit its rate limit. Run the same command again later to finish.")
    print(f"\nWrote {TRAIN_PATH.name}, {VAL_PATH.name}, {STATS_PATH.name}")


if __name__ == "__main__":
    main()
