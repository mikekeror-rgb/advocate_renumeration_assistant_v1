"""
Step 5-6: Eval harness.

Reads:  eval/qa_pairs.jsonl
Runs:   RagPipeline (rag_pipeline_v2 by default; override with PIPELINE_MODULE) against
        each question. The LLM backend follows the pipeline's own env vars
        (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL), so the SAME eval runs against
        Groq, vLLM, Ollama or TGI.

Scores (per question):
  - retrieval_hit        : did retrieval surface the expected source (case-level for rulings)?
  - evidence_retrieved   : optional (needs 'evidence_hint'): was the answer-bearing passage
                           retrieved, and inside the part of the chunk the generator sees?
  - numeric_exact_match  : calculator figure matches expected (router correctness)
  - answer_states_figure : the visible answer states the expected figure
  - citation_verifiable  : EVERY cited chunk_id was actually retrieved
  - calculator_routing_correct
  - guard_correct        : guard rows got the deterministic guard message, not an LLM answer
  - answer_correct       : does the answer match the expected answer? Deterministic for
                           numeric / no_answer / guard rows; reference-based LLM judge for
                           factual rows.
  - faithfulness_score   : 1-5 LLM judge: is the answer supported by its own context?

Why both correctness AND faithfulness: they fail differently. In the Groq baseline,
ruling_02 answered "Not covered" although the answer exists in the corpus. That is
faithful (score 5) but wrong. Only a reference-based check catches it.

Writes: eval/results.csv, or eval/results_<EVAL_TAG>.csv, and prints a summary.

Run from the project root: `python eval/eval.py`
Env:
  PIPELINE_MODULE  rag_pipeline_v2 (default) | rag_pipeline
  JUDGE_MODEL      qwen2.5:7b (default, local Ollama) | e.g. gpt-5.4-mini
  EVAL_TAG         label for the results file, e.g. groq_baseline, qwen_vllm, qwen_lora
  EVAL_ONLY        comma-separated ids for a quick subset run, e.g. ruling_02,ruling_05
  QA_FILE          question file (default eval/qa_pairs.jsonl), e.g. eval/qa_pairs_v2.jsonl
"""

import csv
import importlib
from dotenv import load_dotenv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so the pipeline imports when run as eval/eval.py

PIPELINE_MODULE = os.environ.get("PIPELINE_MODULE", "rag_pipeline_v2")
_pipeline_module = importlib.import_module(PIPELINE_MODULE)
RagPipeline = _pipeline_module.RagPipeline

# The judge must see what the generator saw. The generator truncates every chunk to
# CONTEXT_SNIPPET_CHARS; judging with shorter snippets (previously 300) marked correct
# answers as unsupported (ruling_03 and ruling_06 in the Groq baseline).
GENERATOR_SNIPPET_CHARS = getattr(_pipeline_module, "CONTEXT_SNIPPET_CHARS", 1000)

QA_PATH = Path(os.environ.get("QA_FILE", Path(__file__).parent / "qa_pairs.jsonl"))
EVAL_TAG = os.environ.get("EVAL_TAG", "").strip()
RESULTS_PATH = Path(__file__).parent / (f"results_{EVAL_TAG}.csv" if EVAL_TAG else "results.csv")
EVAL_ONLY = {i.strip() for i in os.environ.get("EVAL_ONLY", "").split(",") if i.strip()}

# Default is the free, local, zero-config judge, so `git clone && python eval.py`
# works for anyone with no API key. Set the JUDGE_MODEL env var (e.g. in your own
# untracked .env, never committed) to opt into a paid frontier judge instead,
# e.g. JUDGE_MODEL=gpt-5.4-mini. See README for why this is worth doing: the two
# judges disagreed noticeably on several questions in this project's own eval runs.
JUDGE_MODEL_NAME = os.environ.get("JUDGE_MODEL", "qwen2.5:7b")

# Every 'chunk_id: X' citation, whether the model used [ ] or 【 】 brackets.
CITED_CHUNK_ID_PATTERN = re.compile(r"chunk_id:\s*([^\]】]+)")
EXPECTED_REFUSAL = "not covered in the provided documents"


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------

def check_citation_verifiable(answer_text: str, retrieved_chunks: list[dict]) -> bool | None:
    """Does EVERY 'chunk_id: X' cited in the answer correspond to a chunk that was
    actually retrieved for this query? Catches fabricated citations: a model stating
    a plausible-looking chunk_id that was never in its context (ruling_05 cited
    '...__s5__c3', which doesn't exist in the corpus at all).

    Previously only the FIRST citation was checked, so a fabricated second citation
    passed silently.

    Returns None if the answer makes no chunk_id citation at all (e.g. a pure
    calculator answer citing only '[source: ...]', or a correct 'not covered').
    """
    citations = CITED_CHUNK_ID_PATTERN.findall(answer_text)
    if not citations:
        return None
    retrieved_ids = {c["chunk_id"] for c in retrieved_chunks}
    # Verbatim substring check: a real citation contains an actually-retrieved
    # chunk_id somewhere in the cited text (models sometimes append a case name
    # after the real chunk_id inside the same brackets).
    return all(any(cid in citation for cid in retrieved_ids) for citation in citations)


def load_qa_pairs(path: Path) -> list[dict]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def check_retrieval_hit(qa: dict, retrieved_chunks: list[dict]) -> bool:
    """Did retrieval surface the expected source? For statute/none this is coarse
    (source type only). For rulings, if qa_pairs.jsonl provides a 'doc_hint'
    (a substring of the expected case's doc_title), require a chunk from that
    SPECIFIC case. Note this is case-level only: the right case can be retrieved
    without the passage holding the answer. Use 'evidence_hint' for that."""
    expected = qa["expected_source"]
    if expected == "none":
        return True  # nothing to retrieve for out-of-corpus questions
    if expected == "statute":
        content_hint = qa.get("content_hint")
        if content_hint:
            # Any retrieved chunk containing the exact phrase counts: rulings
            # legitimately quote the Order verbatim (confirmed by statute_01).
            return any(content_hint.lower() in c["text"].lower() for c in retrieved_chunks)
        return any(c["doc_type"] == "statute" for c in retrieved_chunks)
    if expected == "ruling":
        doc_hint = qa.get("doc_hint")
        if doc_hint:
            return any(
                c["doc_type"] == "ruling" and doc_hint.lower() in c["doc_title"].lower()
                for c in retrieved_chunks
            )
        return any(c["doc_type"] == "ruling" for c in retrieved_chunks)
    return False


def check_evidence_retrieved(qa: dict, retrieved_chunks: list[dict]) -> str | None:
    """Separates retrieval failures from generation failures. Needs an optional
    'evidence_hint' in qa_pairs.jsonl: a short phrase that appears verbatim in the
    passage stating the answer (find it with eval/find_evidence.py). Returns:
      'visible'   - a retrieved chunk contains it within the text the generator sees
      'truncated' - retrieved, but only past the generator's snippet cut-off
      'missing'   - no retrieved chunk contains it: a retrieval failure
      None        - no evidence_hint for this question
    """
    evidence_chunk_id = qa.get("evidence_chunk_id")
    if evidence_chunk_id:
        return "visible" if any(c["chunk_id"] == evidence_chunk_id for c in retrieved_chunks) else "missing"
    hint = qa.get("evidence_hint")
    if not hint:
        return None
    hint = hint.lower()
    status = "missing"
    for c in retrieved_chunks:
        text = c["text"].lower()
        if hint in text[:GENERATOR_SNIPPET_CHARS]:
            return "visible"
        if hint in text:
            status = "truncated"
    return status


def check_numeric_exact_match(qa: dict, answer_text: str, calculator_result) -> bool | None:
    if qa["expected_type"] != "numeric":
        return None

    # Prefer the calculator's own computed figure: exact and unambiguous, unlike
    # parsing the LLM's phrasing, where the loan amount often appears before the fee.
    if calculator_result is not None and calculator_result.fee is not None:
        return abs(calculator_result.fee - qa["expected_value"]) < 1.0

    # Fallback when the calculator didn't fire: take the LAST Kshs figure in the
    # answer, since answers state amounts-in-question before the computed result.
    matches = re.findall(r"Kshs\.?\s*([\d,]+(?:\.\d+)?)", answer_text, re.IGNORECASE)
    if not matches:
        matches = re.findall(r"\b([\d]{1,3}(?:,\d{3})+)\b", answer_text)
    if not matches:
        return False
    extracted = float(matches[-1].replace(",", ""))
    return abs(extracted - qa["expected_value"]) < 1.0


def check_answer_states_figure(qa: dict, answer_text: str) -> bool | None:
    """Separate from numeric_exact_match on purpose: that check can pass purely on
    the router's internal math even if the VISIBLE answer never mentions the figure
    (the calc_01/06/07 context-truncation regression). This checks what the user reads."""
    if qa["expected_type"] != "numeric":
        return None
    expected_str = f"{qa['expected_value']:,.0f}"
    pattern = re.escape(expected_str).replace(r"\,", ",?")
    return bool(re.search(pattern, answer_text)) or str(int(qa["expected_value"])) in answer_text.replace(",", "")


def check_guard(qa: dict, answer_text: str, calculator_result) -> bool | None:
    """Guard questions must get the deterministic guard message, never an LLM answer."""
    if qa["expected_type"] != "guard":
        return None
    return calculator_result is None and qa["expected_value"].lower() in answer_text.lower()


def check_cites_named_case(qa: dict, answer_text: str) -> bool | None:
    """For ruling questions: does the answer cite at least one chunk from the case
    the question asks about? Catches cross-case attribution (ruling_03 cited a
    Brookshill v County Government of Kwale chunk to describe the Kariithi court's reasoning)."""
    doc_hint = qa.get("doc_hint")
    if qa["expected_source"] != "ruling" or not doc_hint or qa["expected_type"] != "factual":
        return None
    citations = CITED_CHUNK_ID_PATTERN.findall(answer_text)
    if not citations:
        return False
    return any(doc_hint.lower() in c.lower() for c in citations)


def category_of(qa: dict) -> str:
    """calc_01 -> calc, ruling_05 -> ruling, hruling_07 -> hruling (held-out), guard_01 -> guard."""
    return qa["id"].split("_")[0]


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def build_judge_context(
    retrieved_chunks: list[dict],
    calculator_result=None,
    max_chunks: int = 12,
    snippet_chars: int = GENERATOR_SNIPPET_CHARS,
) -> str:
    """What the judge sees must match what the generator saw: every retrieved chunk
    (retrieve() returns at most top_k + statute_boost), each truncated to the same
    length the generator used."""
    parts = []
    if calculator_result is not None:
        parts.append(
            f"VERIFIED CALCULATION: {calculator_result.explanation} "
            f"[source: {calculator_result.schedule_citation}]"
        )
    for c in retrieved_chunks[:max_chunks]:
        snippet = c["text"][:snippet_chars].replace("\n", " ")
        parts.append(f"[{c['doc_type']} / {c['section']}]: {snippet}")
    return "\n\n".join(parts) if parts else "(no context retrieved)"


def _call_judge(prompt: str) -> str:
    """Ollama for local models (the free default), or ChatOpenAI for gpt-* models.
    Each library is imported only when that judge is actually used."""
    if JUDGE_MODEL_NAME.startswith("gpt-"):
        from langchain_openai import ChatOpenAI
        judge = ChatOpenAI(model=JUDGE_MODEL_NAME, reasoning_effort="low")
        return judge.invoke(prompt).content
    import ollama
    response = ollama.chat(model=JUDGE_MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


def _parse_judge_json(raw: str) -> dict | None:
    """Strip markdown fences; fall back to the first {...} block if the judge added prose."""
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def judge_faithfulness(question: str, answer: str, context_summary: str) -> tuple[int, str]:
    """Is the answer fully supported by the context it was given? Returns
    (score 1-5, reasoning), or (0, error) if the judge's reply isn't parseable."""
    judge_prompt = f"""You are grading a RAG system's answer for faithfulness to its own context.

Question: {question}
Context available to the system: {context_summary}
System's answer: {answer}

Rate 1-5 how well the answer is supported ONLY by the context provided (5 = fully supported and \
accurate, 1 = unsupported or hallucinated). A correct "Not covered in the provided documents" \
response to a question with no relevant context should score 5, not 1.

Respond with ONLY a JSON object, no other text: {{"score": <int 1-5>, "reasoning": "<one sentence>"}}"""

    raw = _call_judge(judge_prompt)
    parsed = _parse_judge_json(raw)
    try:
        return int(parsed["score"]), parsed.get("reasoning", "")
    except (TypeError, KeyError, ValueError):
        return 0, f"[judge response unparseable: {raw[:200]}]"


def judge_correctness(question: str, answer: str, reference: str) -> tuple[bool | None, str]:
    """Reference-based: does the answer state the same key facts as the expected
    answer? Returns (True/False, reasoning), or (None, error) if unparseable."""
    judge_prompt = f"""You are grading whether a legal research assistant's answer is CORRECT, \
compared with a reference answer written by an expert.

Question: {question}
Reference answer: {reference}
System's answer: {answer}

The system's answer is CORRECT if it states the same key facts or conclusion as the reference. \
Different wording is fine, and extra accurate detail is fine.
It is INCORRECT if it contradicts the reference, misses the key fact or conclusion, or says the \
question is not covered when the reference gives an answer.

Respond with ONLY a JSON object, no other text: {{"correct": true or false, "reasoning": "<one sentence>"}}"""

    raw = _call_judge(judge_prompt)
    parsed = _parse_judge_json(raw)
    if not parsed or "correct" not in parsed:
        return None, f"[judge response unparseable: {raw[:200]}]"
    correct = parsed["correct"]
    if isinstance(correct, str):
        correct = correct.strip().lower() == "true"
    return bool(correct), parsed.get("reasoning", "")


def determine_correctness(qa: dict, answer_text: str, answer_states_figure, guard_correct) -> tuple[bool | None, str]:
    """Deterministic wherever Python can decide; the LLM judge only for factual rows."""
    expected_type = qa["expected_type"]
    if expected_type == "numeric":
        return bool(answer_states_figure), "Deterministic: expected figure stated in the answer." if answer_states_figure \
            else "Deterministic: expected figure NOT stated in the answer."
    if expected_type == "no_answer":
        refused = EXPECTED_REFUSAL in answer_text.lower()
        return refused, "Deterministic: correct refusal." if refused else "Deterministic: answered an out-of-scope question."
    if expected_type == "guard":
        return bool(guard_correct), "Deterministic: guard message returned." if guard_correct \
            else "Deterministic: guard expected, but the LLM answered."
    if expected_type == "factual":
        return judge_correctness(qa["question"], answer_text, str(qa["expected_value"]))
    return None, f"Unknown expected_type: {expected_type}"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_eval():
    pipeline = RagPipeline()
    backend = getattr(pipeline, "backend", "groq (legacy pipeline)")
    model = getattr(pipeline, "generation_model_name", "unknown")
    print(f"Pipeline: {PIPELINE_MODULE} | backend: {backend} | model: {model} | judge: {JUDGE_MODEL_NAME}\n")

    qa_pairs = load_qa_pairs(QA_PATH)
    if EVAL_ONLY:
        qa_pairs = [qa for qa in qa_pairs if qa["id"] in EVAL_ONLY]
        print(f"EVAL_ONLY: running {len(qa_pairs)} of the questions\n")
    if not qa_pairs:
        print("No questions to run.")
        return

    rows = []
    for qa in qa_pairs:
        print(f"Running {qa['id']}: {qa['question'][:70]}...")
        result = pipeline.answer(qa["question"])
        answer_text = result["answer"]
        retrieved_chunks = result["retrieved_chunks"]
        calculator_result = result["calculator_result"]

        retrieval_hit = check_retrieval_hit(qa, retrieved_chunks)
        evidence_retrieved = check_evidence_retrieved(qa, retrieved_chunks)
        numeric_match = check_numeric_exact_match(qa, answer_text, calculator_result)
        answer_states_figure = check_answer_states_figure(qa, answer_text)
        citation_verifiable = check_citation_verifiable(answer_text, retrieved_chunks)
        guard_correct = check_guard(qa, answer_text, calculator_result)

        expects_calculator = qa["expected_type"] == "numeric"
        calculator_fired = calculator_result is not None
        calculator_routing_correct = (calculator_fired == expects_calculator)

        answer_correct, correctness_reasoning = determine_correctness(
            qa, answer_text, answer_states_figure, guard_correct
        )

        context_summary = build_judge_context(retrieved_chunks, calculator_result)
        if qa["expected_type"] == "no_answer" and EXPECTED_REFUSAL in answer_text.lower():
            # The judge was unreliable on correct refusals; Python can check this exactly.
            faithfulness_score, faithfulness_reasoning = 5, "Deterministic: correct refusal on a no_answer question (judge bypassed)."
        elif qa["expected_type"] == "guard":
            faithfulness_score, faithfulness_reasoning = 0, "Not judged: guard row, see guard_correct."
        else:
            faithfulness_score, faithfulness_reasoning = judge_faithfulness(qa["question"], answer_text, context_summary)

        rows.append({
            "id": qa["id"],
            "category": category_of(qa),
            "backend": backend,
            "model": model,
            "expected_type": qa["expected_type"],
            "expected_value": qa["expected_value"],
            "answer": answer_text,
            "answer_correct": answer_correct,
            "correctness_reasoning": correctness_reasoning,
            "retrieval_hit": retrieval_hit,
            "evidence_retrieved": evidence_retrieved,
            "numeric_exact_match": numeric_match,
            "answer_states_figure": answer_states_figure,
            "citation_verifiable": citation_verifiable,
            "calculator_fired": calculator_fired,
            "calculator_routing_correct": calculator_routing_correct,
            "guard_correct": guard_correct,
            "cites_named_case": check_cites_named_case(qa, answer_text),
            "faithfulness_score": faithfulness_score,
            "faithfulness_reasoning": faithfulness_reasoning,
        })

    write_results(rows)
    print_summary(rows)


def write_results(rows: list[dict]) -> None:
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} results to {RESULTS_PATH}")


def print_summary(rows: list[dict]) -> None:
    total = len(rows)
    print("\n=== Eval Summary ===")
    print(f"Backend / model:                 {rows[0]['backend']} / {rows[0]['model']}")
    print(f"Total questions:                 {total}")

    # Correctness first: it's the headline number.
    by_category = defaultdict(list)
    for r in rows:
        if r["answer_correct"] is not None:
            by_category[r["category"]].append(r["answer_correct"])
    graded = [v for values in by_category.values() for v in values]
    if graded:
        print(f"Answer correctness (overall):    {sum(graded) / len(graded):.1%}  (n={len(graded)})")
        for category in sorted(by_category):
            values = by_category[category]
            print(f"  - {category:<10}                   {sum(values)}/{len(values)}")
    wrong = [r["id"] for r in rows if r["answer_correct"] is False]
    if wrong:
        print(f"  ✗ Incorrect: {', '.join(wrong)}")
    ungraded = [r["id"] for r in rows if r["answer_correct"] is None]
    if ungraded:
        print(f"  ? Not graded (judge unparseable): {', '.join(ungraded)}")

    print(f"Retrieval hit rate:              {sum(r['retrieval_hit'] for r in rows) / total:.1%}")

    evidence_rows = [r for r in rows if r["evidence_retrieved"] is not None]
    if evidence_rows:
        print(f"Evidence retrieval (n={len(evidence_rows)} with evidence_hint):")
        for status in ("visible", "truncated", "missing"):
            ids = [r["id"] for r in evidence_rows if r["evidence_retrieved"] == status]
            if ids:
                print(f"  - {status:<10} {', '.join(ids)}")

    numeric_rows = [r for r in rows if r["expected_type"] == "numeric"]
    if numeric_rows:
        numeric_accuracy = sum(r["numeric_exact_match"] for r in numeric_rows) / len(numeric_rows)
        figure_stated_rate = sum(r["answer_states_figure"] for r in numeric_rows) / len(numeric_rows)
        print(f"Numeric exact-match accuracy:    {numeric_accuracy:.1%}  (n={len(numeric_rows)}) [router/calculator correctness]")
        print(f"Answer actually states figure:   {figure_stated_rate:.1%}  (n={len(numeric_rows)}) [what the user actually sees]")

    print(f"Calculator routing accuracy:     {sum(r['calculator_routing_correct'] for r in rows) / total:.1%}")

    guard_rows = [r for r in rows if r["guard_correct"] is not None]
    if guard_rows:
        print(f"Calculation guard accuracy:      {sum(r['guard_correct'] for r in guard_rows) / len(guard_rows):.1%}  (n={len(guard_rows)})")

    cited_rows = [r for r in rows if r["citation_verifiable"] is not None]
    if cited_rows:
        fabricated = [r for r in cited_rows if not r["citation_verifiable"]]
        print(f"Citation accuracy:               {(len(cited_rows) - len(fabricated)) / len(cited_rows):.1%}  (n={len(cited_rows)} answers with a chunk_id citation)")
        if fabricated:
            print(f"  ⚠ UNVERIFIABLE CITATIONS ({len(fabricated)}):")
            for r in fabricated:
                print(f"    - {r['id']}: cited a chunk_id not present in retrieval")

    named_rows = [r for r in rows if r["cites_named_case"] is not None]
    if named_rows:
        wrong_case = [r["id"] for r in named_rows if not r["cites_named_case"]]
        print(f"Cites the case asked about:      {(len(named_rows) - len(wrong_case)) / len(named_rows):.1%}  (n={len(named_rows)})")
        if wrong_case:
            print(f"  ⚠ Cited another case instead: {', '.join(wrong_case)}")

    judged_rows = [r for r in rows if r["faithfulness_score"] > 0]
    if judged_rows:
        avg_faithfulness = sum(r["faithfulness_score"] for r in judged_rows) / len(judged_rows)
        print(f"Avg faithfulness score (1-5):    {avg_faithfulness:.2f}  (n={len(judged_rows)} judged)")


if __name__ == "__main__":
    run_eval()
