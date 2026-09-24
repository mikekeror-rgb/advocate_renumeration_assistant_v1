"""
Step 5-6: Eval harness.

Reads:  eval/qa_pairs.jsonl
Runs:   rag_pipeline.RagPipeline against each question
Scores:
  - retrieval_hit      : did retrieval surface the expected source type (statute/ruling)?
  - numeric_exact_match: for calculator questions, does the answer contain the exact expected figure?
  - calculator_fired   : did fee_router actually route this question, when it was supposed to?
  - faithfulness_score : 1-5, LLM-as-judge — is the answer fully supported by its own context?
Writes: eval/results.csv (per-question) + prints an aggregate summary.

Run from the project root: `python eval/eval.py`
"""

import csv
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
import ollama

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `import rag_pipeline` works when run as eval/eval.py
from rag_pipeline import RagPipeline

QA_PATH = Path(__file__).parent / "qa_pairs.jsonl"
RESULTS_PATH = Path(__file__).parent / "results.csv"

# Default is the free, local, zero-config judge — `git clone && python eval.py`
# works for anyone with no API key. Set the JUDGE_MODEL env var (e.g. in your own
# untracked .env, never committed) to opt into a paid frontier judge instead —
# e.g. JUDGE_MODEL=gpt-5.4-mini. See README for why this is worth doing: the two
# judges disagreed noticeably on several questions in this project's own eval runs.
JUDGE_MODEL_NAME = os.environ.get("JUDGE_MODEL", "qwen2.5:7b")

CITED_CHUNK_ID_PATTERN = re.compile(r"chunk_id:\s*([^\]]+)")


def check_citation_verifiable(answer_text: str, retrieved_chunks: list[dict]) -> bool | None:
    """Does every 'chunk_id: X' cited in the answer correspond to a chunk that was
    ACTUALLY retrieved for this query? Catches fabricated citations — a model
    stating a plausible-looking chunk_id (or even real-looking content) that was
    never in its context at all. This is the check that would have caught the
    ruling_05 case: the answer cited '...__s5__c3', a chunk_id that doesn't exist
    anywhere in the corpus (only '...__s5__c0' does) — a fabricated citation
    attached to fabricated content, invisible to every other check in this file.

    Returns None if the answer makes no chunk_id citation at all (e.g. a pure
    calculator answer citing only '[source: ...]', or a correct 'not covered').
    """
    match = CITED_CHUNK_ID_PATTERN.search(answer_text)
    if not match:
        return None

    citation_text = match.group(1)
    retrieved_ids = {c["chunk_id"] for c in retrieved_chunks}
    # verbatim substring check: a real citation contains an actually-retrieved
    # chunk_id somewhere in the cited text (models sometimes append extra text
    # like a case name after the real chunk_id within the same brackets)
    return any(cid in citation_text for cid in retrieved_ids)


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
    SPECIFIC case — with a large multi-ruling corpus, 'any ruling retrieved' is
    too weak a signal and was masking real misses (see ruling_02/05/06)."""
    expected = qa["expected_source"]
    if expected == "none":
        return True  # nothing to retrieve for out-of-corpus questions — vacuously fine
    if expected == "statute":
        content_hint = qa.get("content_hint")
        if content_hint:
            # Any retrieved chunk containing the exact phrase counts — not just
            # doc_type=="statute" ones. Confirmed necessary by statute_01: rulings
            # legitimately quote the Order verbatim, so a correct answer can
            # legitimately cite a ruling chunk that happens to contain the same text.
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


def check_numeric_exact_match(qa: dict, answer_text: str, calculator_result) -> bool | None:
    if qa["expected_type"] != "numeric":
        return None  # not applicable

    # Prefer comparing the calculator's own computed figure — exact and unambiguous,
    # unlike parsing the LLM's phrasing where the loan amount often appears before
    # the fee itself in the sentence (e.g. "...on a Kshs 2,800,000 security is Kshs
    # 15,000" — naively grabbing the first Kshs-number picks the loan, not the fee).
    if calculator_result is not None and calculator_result.fee is not None:
        return abs(calculator_result.fee - qa["expected_value"]) < 1.0

    # Fallback when the calculator didn't fire: take the LAST Kshs-figure in the
    # answer, since answers consistently state amounts-in-question before the
    # computed result.
    matches = re.findall(r"Kshs\.?\s*([\d,]+(?:\.\d+)?)", answer_text, re.IGNORECASE)
    if not matches:
        matches = re.findall(r"\b([\d]{1,3}(?:,\d{3})+)\b", answer_text)
    if not matches:
        return False
    extracted = float(matches[-1].replace(",", ""))
    return abs(extracted - qa["expected_value"]) < 1.0


def check_answer_states_figure(qa: dict, answer_text: str) -> bool | None:
    """Separate from numeric_exact_match on purpose: that check can pass purely on
    the router's internal math even if the VISIBLE answer text never mentions the
    figure at all (exactly what happened in the calc_01/06/07 context-truncation
    regression — router correct, generated answer completely unrelated). This
    check looks only at what the user would actually read."""
    if qa["expected_type"] != "numeric":
        return None
    expected_str = f"{qa['expected_value']:,.0f}"
    # accept with or without comma-grouping, with or without decimals
    pattern = re.escape(expected_str).replace(r"\,", ",?") 
    return bool(re.search(pattern, answer_text)) or str(int(qa["expected_value"])) in answer_text.replace(",", "")


def build_judge_context(retrieved_chunks: list[dict], calculator_result=None, max_chunks: int = 12, snippet_chars: int = 300) -> str:
    """What the judge sees must resemble what the generator saw. Previously capped
    at 4 chunks by raw distance — but the statute-boost in rag_pipeline.retrieve()
    can rank a highly relevant chunk 5th+ on pure semantic distance despite it being
    exactly the useful one (see statute_03: correct answer, judge scored it 1
    because the boosted chunk wasn't in its first-4 window). Raised the cap to
    cover everything retrieve() actually returns (TOP_K=8 + statute_boost=3 = 11
    max) and shrunk the per-chunk snippet to keep the judge prompt reasonable."""
    parts = []
    if calculator_result is not None:
        parts.append(f"VERIFIED CALCULATION: {calculator_result.explanation} [source: {calculator_result.schedule_citation}]")
    for c in retrieved_chunks[:max_chunks]:
        snippet = c["text"][:snippet_chars].replace("\n", " ")
        parts.append(f"[{c['doc_type']} / {c['section']}]: {snippet}")
    return "\n\n".join(parts) if parts else "(no context retrieved)"


def _call_judge(prompt: str) -> str:
    """Route to Ollama for local models (the free default), or ChatOpenAI for
    anything else (e.g. gpt-5.4-mini) — only imports/needs an API key when a
    non-local judge is actually requested via the JUDGE_MODEL env var."""
    if JUDGE_MODEL_NAME.startswith("gpt-"):
        from langchain_openai import ChatOpenAI
        judge = ChatOpenAI(model=JUDGE_MODEL_NAME, reasoning_effort="low")
        response = judge.invoke(prompt)
        return response.content
    else:
        response = ollama.chat(model=JUDGE_MODEL_NAME, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]


def judge_faithfulness(question: str, answer: str, context_summary: str) -> tuple[int, str]:
    """LLM-as-judge: is the answer fully supported by the context it was given?
    Returns (score 1-5, reasoning). Falls back to (0, error) if the judge's
    response isn't parseable JSON, so a judge hiccup doesn't crash the whole run."""
    judge_prompt = f"""You are grading a RAG system's answer for faithfulness to its own context.

Question: {question}
Context available to the system: {context_summary}
System's answer: {answer}

Rate 1-5 how well the answer is supported ONLY by the context provided (5 = fully supported and \
accurate, 1 = unsupported or hallucinated). A correct "Not covered in the provided documents" \
response to a question with no relevant context should score 5, not 1.

Respond with ONLY a JSON object, no other text: {{"score": <int 1-5>, "reasoning": "<one sentence>"}}"""

    raw = _call_judge(judge_prompt).strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip())  # strip markdown fences if the judge adds them

    try:
        parsed = json.loads(raw)
        return int(parsed["score"]), parsed.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0, f"[judge response unparseable: {raw[:200]}]"


def run_eval():
    pipeline = RagPipeline()
    qa_pairs = load_qa_pairs(QA_PATH)

    rows = []
    for qa in qa_pairs:
        print(f"Running {qa['id']}: {qa['question'][:70]}...")
        result = pipeline.answer(qa["question"])
        answer_text = result["answer"]
        retrieved_chunks = result["retrieved_chunks"]
        calculator_result = result["calculator_result"]

        retrieval_hit = check_retrieval_hit(qa, retrieved_chunks)
        numeric_match = check_numeric_exact_match(qa, answer_text, calculator_result)
        answer_states_figure = check_answer_states_figure(qa, answer_text)
        citation_verifiable = check_citation_verifiable(answer_text, retrieved_chunks)

        expects_calculator = qa["id"].startswith("calc_")
        calculator_fired = calculator_result is not None
        calculator_routing_correct = (calculator_fired == expects_calculator)

        context_summary = build_judge_context(retrieved_chunks, calculator_result)

        # The judge was unreliable specifically on correct "Not covered" responses to
        # no_answer questions — scoring them 1 despite an explicit instruction that
        # they should score 5 (confirmed twice: noanswer_01/02 in the prior run).
        # Rather than keep tuning a flaky judge prompt for something Python can check
        # deterministically, bypass the judge entirely for this one verifiable case.
        expected_refusal = "not covered in the provided documents"
        if qa["expected_type"] == "no_answer" and expected_refusal in answer_text.lower():
            faithfulness_score, faithfulness_reasoning = 5, "Deterministic: correct refusal on a no_answer question (judge bypassed)."
        else:
            faithfulness_score, faithfulness_reasoning = judge_faithfulness(qa["question"], answer_text, context_summary)

        rows.append({
            "id": qa["id"],
            "expected_type": qa["expected_type"],
            "expected_value": qa["expected_value"],
            "answer": answer_text,
            "retrieval_hit": retrieval_hit,
            "numeric_exact_match": numeric_match,
            "answer_states_figure": answer_states_figure,
            "citation_verifiable": citation_verifiable,
            "calculator_fired": calculator_fired,
            "calculator_routing_correct": calculator_routing_correct,
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
    retrieval_hit_rate = sum(r["retrieval_hit"] for r in rows) / total

    numeric_rows = [r for r in rows if r["expected_type"] == "numeric"]
    numeric_accuracy = (
        sum(r["numeric_exact_match"] for r in numeric_rows) / len(numeric_rows) if numeric_rows else None
    )
    figure_stated_rate = (
        sum(r["answer_states_figure"] for r in numeric_rows) / len(numeric_rows) if numeric_rows else None
    )

    routing_accuracy = sum(r["calculator_routing_correct"] for r in rows) / total

    cited_rows = [r for r in rows if r["citation_verifiable"] is not None]
    fabricated_citations = [r for r in cited_rows if not r["citation_verifiable"]]
    citation_accuracy = (
        (len(cited_rows) - len(fabricated_citations)) / len(cited_rows) if cited_rows else None
    )

    judged_rows = [r for r in rows if r["faithfulness_score"] > 0]
    avg_faithfulness = sum(r["faithfulness_score"] for r in judged_rows) / len(judged_rows) if judged_rows else 0

    print("\n=== Eval Summary ===")
    print(f"Total questions:                {total}")
    print(f"Retrieval hit rate:              {retrieval_hit_rate:.1%}")
    if numeric_accuracy is not None:
        print(f"Numeric exact-match accuracy:    {numeric_accuracy:.1%}  (n={len(numeric_rows)}) [router/calculator correctness]")
        print(f"Answer actually states figure:   {figure_stated_rate:.1%}  (n={len(numeric_rows)}) [what the user actually sees]")
    print(f"Calculator routing accuracy:     {routing_accuracy:.1%}")
    if citation_accuracy is not None:
        print(f"Citation accuracy:               {citation_accuracy:.1%}  (n={len(cited_rows)} answers with a chunk_id citation)")
        if fabricated_citations:
            print(f"  ⚠ FABRICATED CITATIONS ({len(fabricated_citations)}):")
            for r in fabricated_citations:
                print(f"    - {r['id']}: cited a chunk_id not present in retrieval")
    print(f"Avg faithfulness score (1-5):    {avg_faithfulness:.2f}  (n={len(judged_rows)} judged)")


if __name__ == "__main__":
    run_eval()
