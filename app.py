"""
Step 7: Minimal Streamlit UI.

Run with: streamlit run app.py

Three tabs:
- "Ask a question": the RAG assistant — question in, answer + sources + a clear
  flag when the fee calculator (not the LLM) produced the number.
- "Bill of costs": a structured form for a full Schedule 6 High Court bill.
  Case-specific facts (letters, folios, hearings, actual disbursements) can't
  come from the corpus, so the user enters them; rates come from the Order via
  fee_calculator, and the supporting Order chunks are shown as sources.
- "Eval results": any eval/results*.csv run, plus the self-hosted fine-tuning
  comparison (base vs SFT vs SFT+DPO) when those runs are present.

LLM backend (via llm_client.py, read from environment / Streamlit secrets):
- default: Groq (GROQ_API_KEY)
- self-hosted vLLM: set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL (e.g. advocate-dpo)
"""

from pathlib import Path

import pandas as pd
import streamlit as st

import fee_calculator as fc
import fee_router
import feedback
from rag_pipeline_v2 import RagPipeline

st.set_page_config(page_title="Advocate Remuneration Assistant", layout="wide")

MAX_SUGGESTIONS_PER_SESSION = 3  # basic spam limit for the public suggestions box


@st.cache_resource
def load_pipeline() -> RagPipeline:
    return RagPipeline()


def render_sources(retrieved: list[dict]) -> None:
    """Shared by every tab so all answers cite their chunks the same way."""
    with st.expander(f"Sources ({len(retrieved)} chunks retrieved)"):
        for chunk in retrieved:
            label = (
                "📘 STATUTE"
                if chunk["doc_type"] == "statute"
                else f"⚖️ {chunk['case_citation']}"
            )
            st.markdown(f"**{label}** — *{chunk['section']}*")
            st.caption(chunk["chunk_id"])
            snippet = chunk["text"][:400]
            st.text(snippet + ("..." if len(chunk["text"]) > 400 else ""))
            st.divider()


def answered_by_llm(query: str, result: dict) -> bool:
    """False when the answer came from deterministic code, not the model."""
    calc = result.get("calculator_result")
    if calc is not None and calc.scenario == "schedule6_full_bill":
        return False  # full bills are returned straight from the calculator
    if calc is None and fee_router.calculation_guard(query) is not None:
        return False  # the calculation guard answered
    return True


def bool_values(series: pd.Series) -> pd.Series:
    """CSV booleans arrive as True/False, 'True'/'False' or blanks; normalise them."""
    return series.dropna().map(lambda v: str(v).strip().lower() == "true")


tab_chat, tab_bill, tab_eval = st.tabs(["Ask a question", "Bill of costs", "Eval results"])


# ---------------------------------------------------------------------------
# Tab 1: Chat
# ---------------------------------------------------------------------------
with tab_chat:
    st.title("Advocate Remuneration Assistant")
    st.caption(
        "Grounded in the Advocates (Remuneration) Order and Kenyan case law on "
        "taxation of costs. Fee calculations are computed exactly, not guessed by the model."
    )

    query = st.text_input(
        "Ask about advocate fees, taxation of costs, or a specific calculation:",
        placeholder="e.g. What is the discharge fee for a charge over a Kshs 2,800,000 loan?",
    )

    if query:
        # Any widget interaction (e.g. submitting the suggestion form below)
        # reruns this whole script. Cache the answer so a rerun for the same
        # question doesn't call the LLM again and burn the free-tier rate limit.
        if st.session_state.get("last_query") != query:
            pipeline = load_pipeline()
            try:
                with st.spinner("Retrieving and generating..."):
                    st.session_state["last_result"] = pipeline.answer(query)
                    st.session_state["last_query"] = query
            except RuntimeError as e:
                err = str(e).lower()
                if "rate" in err or "limit" in err or "429" in err:
                    st.warning(
                        "Rate/token limit on the free LLM tier. "
                        "Please wait about 30 seconds and try again."
                    )
                else:
                    st.error(str(e))
                st.stop()
        result = st.session_state["last_result"]

        if result.get("calculator_result") is not None:
            st.success(
                f"✅ Calculated exactly via fee_router — "
                f"{result['calculator_result'].schedule_citation}"
            )

        st.markdown("### Answer")
        answer = (result.get("answer") or "").strip()
        if answer:
            st.markdown(answer)
        else:
            st.warning(
                "The model returned an empty answer. "
                "Wait a few seconds and try the same question again."
            )

        if answered_by_llm(query, result):
            pipeline = load_pipeline()
            st.caption(f"Written by `{pipeline.generation_model_name}` ({pipeline.backend}).")

        if "not covered in the provided documents" in answer.lower() and feedback.is_configured():
            submitted_for = st.session_state.setdefault("suggested_queries", [])
            if query in submitted_for:
                st.info("Thanks — your suggestion for this question has been recorded.")
            elif len(submitted_for) >= MAX_SUGGESTIONS_PER_SESSION:
                st.info("You've reached the suggestion limit for this session. Thank you for the help!")
            else:
                with st.form("doc_request_form", clear_on_submit=True):
                    st.markdown("#### Help improve the assistant")
                    st.caption(
                        "This question isn't covered yet. If you know a ruling, statute or other "
                        "public document that answers it, suggest it and it may be added. "
                        "Suggestions are posted publicly as GitHub issues, so please don't include "
                        "personal or confidential information."
                    )
                    suggestion = st.text_area(
                        "Which document or ruling should be added?", max_chars=1000,
                        placeholder="e.g. Kenya Law ruling on taxation of costs in probate matters",
                    )
                    link = st.text_input("Link (optional)", max_chars=300,
                                         placeholder="https://new.kenyalaw.org/...")
                    send = st.form_submit_button("Submit suggestion")

                if send:
                    if not suggestion.strip():
                        st.error("Please describe the document before submitting.")
                    else:
                        try:
                            feedback.submit_document_request(query, suggestion.strip(), link.strip())
                            submitted_for.append(query)
                            st.success("Thanks — your suggestion has been recorded.")
                        except RuntimeError as e:
                            st.error(f"Sorry, the suggestion couldn't be saved: {e}")

        render_sources(result["retrieved_chunks"])


# ---------------------------------------------------------------------------
# Tab 2: Bill of costs (structured form)
# ---------------------------------------------------------------------------
with tab_bill:
    st.title("Bill of Costs — High Court (Schedule 6)")
    st.caption(
        "Rates come from the Advocates (Remuneration) Order. Counts and actual "
        "expenses are facts about your matter, so enter them below. Anything left "
        "at 0 appears in the bill as '—' with its rate, so you can see what's missing."
    )

    with st.form("bill_form"):
        st.markdown("#### Matter")
        c1, c2, c3 = st.columns(3)
        subject_value = c1.number_input(
            "Subject matter value (KSh)", min_value=0.0, value=1_000_000.0, step=100_000.0, format="%.0f"
        )
        defended = c2.radio("Defence filed?", ["Defended", "Undefended"]) == "Defended"
        advocate_client = c3.radio(
            "Bill type", ["Party and party", "Advocate-client (+50%)"]
        ).startswith("Advocate")

        st.markdown("#### Drawing and perusals (1 folio = 100 words)")
        c1, c2, c3 = st.columns(3)
        pleading_folios_raw = c1.text_input(
            "Folios per pleading drawn", placeholder="e.g. 3, 6, 2",
            help="One number per pleading (plaint, defence, affidavit, motion...), comma-separated.",
        )
        other_drawing = c2.number_input("Folios of other documents drawn", min_value=0, step=1)
        perusal = c3.number_input("Folios perused", min_value=0, step=1)

        st.markdown("#### Correspondence and attendances")
        c1, c2, c3, c4 = st.columns(4)
        letters_adv = c1.number_input("Letters to opposing advocate", min_value=0, step=1)
        letters_client = c2.number_input("Letters to client", min_value=0, step=1)
        mentions = c3.number_input("Mentions", min_value=0, step=1)
        hearings = c4.number_input("Hearing days", min_value=0, step=1)

        st.markdown("#### Disbursements (actual amounts paid)")
        c1, c2, c3 = st.columns(3)
        service = c1.number_input("Service of documents (KSh)", min_value=0.0, step=500.0, format="%.2f")
        filing = c2.number_input("Court filing fees (KSh)", min_value=0.0, step=500.0, format="%.2f")
        photocopy = c3.number_input("Photocopying/printing (KSh)", min_value=0.0, step=500.0, format="%.2f")
        other_disb_raw = st.text_area(
            "Other disbursements (one per line, as 'label: amount')",
            placeholder="Travel to court: 3500\nCommissioning affidavits: 1000",
        )

        vat_percent = st.number_input("VAT rate (%)", min_value=0.0, max_value=100.0, value=16.0, step=1.0)
        submitted = st.form_submit_button("Calculate bill")

    if submitted:
        errors = []

        pleading_folios = []
        for part in pleading_folios_raw.split(","):
            part = part.strip()
            if not part:
                continue
            if part.isdigit():
                pleading_folios.append(int(part))
            else:
                errors.append(f"'{part}' in folios per pleading isn't a whole number.")

        disbursements = {}
        if service:
            disbursements["Service of documents"] = service
        if filing:
            disbursements["Court filing fees"] = filing
        if photocopy:
            disbursements["Photocopying/printing"] = photocopy
        for line in other_disb_raw.splitlines():
            if not line.strip():
                continue
            label, sep, amount = line.rpartition(":")
            try:
                if not sep or not label.strip():
                    raise ValueError
                disbursements[label.strip()] = float(amount.replace(",", "").strip())
            except ValueError:
                errors.append(f"Couldn't read '{line.strip()}' — use the format 'label: amount'.")

        if subject_value <= 0:
            errors.append("Enter the subject matter value.")

        if errors:
            for e in errors:
                st.error(e)
            st.stop()

        bill = fc.generate_high_court_bill(
            subject_value,
            defended=defended,
            letters_to_advocate=int(letters_adv),
            letters_to_client=int(letters_client),
            mentions=int(mentions),
            hearings=int(hearings),
            disbursements=disbursements or None,
            vat_rate=vat_percent / 100,
            pleading_folios=pleading_folios or None,
            other_drawing_folios=int(other_drawing),
            perusal_folios=int(perusal),
            advocate_client=advocate_client,
        )
        bill_md = fc.format_bill_as_markdown(bill)

        st.success("✅ Calculated exactly via fee_calculator — Advocates Remuneration Order, Schedule 6")
        st.markdown(bill_md)

        rows = [
            {"Item": li.item, "Calculation": li.calculation_note, "Basis": li.basis,
             "Amount (KSh)": li.amount, "Classification": li.classification}
            for li in bill.line_items
        ]
        rows += [
            {"Item": "Professional fees", "Amount (KSh)": bill.professional_fee_subtotal},
            {"Item": "Disbursements", "Amount (KSh)": bill.disbursement_subtotal},
            {"Item": "Subtotal", "Amount (KSh)": bill.subtotal},
            {"Item": "VAT", "Basis": "VAT Act", "Amount (KSh)": bill.vat_amount, "Classification": "Tax"},
            {"Item": "Total bill", "Amount (KSh)": bill.total},
        ]
        d1, d2 = st.columns(2)
        d1.download_button("Download as CSV", pd.DataFrame(rows).to_csv(index=False),
                           file_name="bill_of_costs.csv", mime="text/csv")
        d2.download_button("Download as Markdown", bill_md,
                           file_name="bill_of_costs.md", mime="text/markdown")

        st.caption(
            "Estimate only — the taxing officer has discretion (para 16) and may allow, "
            "increase or reduce items. Confirm the current VAT rate and any Order amendments."
        )

        try:
            render_sources(load_pipeline().retrieve_bill_sources(advocate_client=advocate_client))
        except Exception as e:
            st.info(f"The bill above is complete, but the supporting sources couldn't be loaded: {e}")


# ---------------------------------------------------------------------------
# Tab 3: Eval results
# ---------------------------------------------------------------------------
with tab_eval:
    st.title("Evaluation Results")
    eval_dir = Path("eval")

    # The self-hosted fine-tuning experiment: same 62 questions, same pipeline, only the model differs.
    experiment = {
        "Base Qwen2.5-7B-Instruct": eval_dir / "results_v3_base.csv",
        "SFT (QLoRA)": eval_dir / "results_v3_advocate-ep1.csv",
        "SFT + DPO": eval_dir / "results_v3_advocate-dpo.csv",
    }
    if all(path.exists() for path in experiment.values()):
        st.markdown("#### Fine-tuning experiment (self-hosted, vLLM on one RTX 4090)")
        summary = []
        for name, path in experiment.items():
            d = pd.read_csv(path)
            correct = bool_values(d["answer_correct"]) if "answer_correct" in d else pd.Series(dtype=bool)
            cites = bool_values(d["citation_verifiable"])
            named = bool_values(d["cites_named_case"]) if "cites_named_case" in d else pd.Series(dtype=bool)
            judged = d["faithfulness_score"][d["faithfulness_score"] > 0]
            summary.append({
                "Model": name,
                "Correctness": f"{correct.mean():.1%}" if len(correct) else "—",
                "Fabricated citations": int((~cites).sum()),
                "Cites the case asked about": f"{named.mean():.1%}" if len(named) else "—",
                "Avg faithfulness (1-5)": round(judged.mean(), 2) if len(judged) else None,
            })
        st.dataframe(pd.DataFrame(summary).set_index("Model"), use_container_width=True)
        st.caption(
            "62 questions, 33 of them from rulings the models never saw in training. Fine-tuning removed "
            "fabricated citations; correctness differences of one or two questions are within run-to-run noise. "
            "The live assistant uses Groq unless a self-hosted server is configured."
        )

    runs = sorted(eval_dir.glob("results*.csv"))
    if not runs:
        st.warning("No eval results found yet — run `python eval/eval.py` first, then refresh this page.")
    else:
        names = [p.name for p in runs]
        default = names.index("results_v3_advocate-dpo.csv") if "results_v3_advocate-dpo.csv" in names else 0
        chosen = st.selectbox("Eval run", runs, index=default, format_func=lambda p: p.name)
        df = pd.read_csv(chosen)
        df["category"] = df["id"].str.extract(r"^([a-z]+)_")

        cols = st.columns(5)
        if "answer_correct" in df:
            correct = bool_values(df["answer_correct"])
            if len(correct):
                cols[0].metric("Answer correctness", f"{correct.mean():.0%}")
        cols[1].metric("Retrieval hit rate", f"{bool_values(df['retrieval_hit']).mean():.0%}")

        numeric_df = df[df["expected_type"] == "numeric"]
        if len(numeric_df):
            cols[2].metric("Numeric accuracy", f"{bool_values(numeric_df['numeric_exact_match']).mean():.0%}")

        cites = bool_values(df["citation_verifiable"])
        fabricated = int((~cites).sum())
        if len(cites):
            cols[3].metric(
                "Citation accuracy",
                f"{cites.mean():.0%}",
                delta=f"-{fabricated} fabricated" if fabricated else None,
                delta_color="inverse",
            )

        judged = df[df["faithfulness_score"] > 0]
        if len(judged):
            cols[4].metric("Avg faithfulness", f"{judged['faithfulness_score'].mean():.2f} / 5")

        st.caption(
            "Faithfulness is LLM-judged and should be read as a rough secondary signal — "
            "retrieval hit rate, numeric accuracy, and citation accuracy above are deterministic "
            "and more reliable on their own."
        )

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("#### Faithfulness score distribution")
            st.bar_chart(judged["faithfulness_score"].value_counts().sort_index())
        with chart_col2:
            st.markdown("#### Retrieval hit rate by question category")
            df["retrieval_hit_bool"] = df["retrieval_hit"].map(lambda v: str(v).strip().lower() == "true")
            st.bar_chart(df.groupby("category")["retrieval_hit_bool"].mean())

        if fabricated:
            st.markdown("#### ⚠ Fabricated citations")
            bad_ids = cites[~cites].index
            st.dataframe(df.loc[bad_ids, ["id", "answer"]], use_container_width=True)

        st.markdown("#### Full results")
        st.dataframe(df.drop(columns=["retrieval_hit_bool"]), use_container_width=True)
