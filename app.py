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
- "Eval results": reads eval/results.csv and charts the eval metrics.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

import fee_calculator as fc
from rag_pipeline import RagPipeline

st.set_page_config(page_title="Advocate Remuneration Assistant", layout="wide")


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
        pipeline = load_pipeline()
        try:
            with st.spinner("Retrieving and generating..."):
                result = pipeline.answer(query)
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

    results_path = Path("eval/results.csv")
    if not results_path.exists():
        st.warning("No `eval/results.csv` found yet — run `python eval/eval.py` first, then refresh this page.")
    else:
        df = pd.read_csv(results_path)
        df["category"] = df["id"].str.extract(r"^([a-z]+)_")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Retrieval hit rate", f"{df['retrieval_hit'].mean():.0%}")

        numeric_df = df[df["expected_type"] == "numeric"]
        if len(numeric_df):
            col2.metric("Numeric accuracy", f"{numeric_df['numeric_exact_match'].mean():.0%}")

        fabricated = 0
        cited_df = df[df["citation_verifiable"].notna()]
        if len(cited_df):
            fabricated = (~cited_df["citation_verifiable"].astype(bool)).sum()
            col3.metric(
                "Citation accuracy",
                f"{cited_df['citation_verifiable'].astype(bool).mean():.0%}",
                delta=f"-{fabricated} fabricated" if fabricated else None,
                delta_color="inverse",
            )

        col4.metric("Avg faithfulness", f"{df['faithfulness_score'].mean():.2f} / 5")

        st.caption(
            "Faithfulness is LLM-judged and should be read as a rough secondary signal — "
            "retrieval hit rate, numeric accuracy, and citation accuracy above are deterministic "
            "and more reliable on their own."
        )

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("#### Faithfulness score distribution")
            st.bar_chart(df["faithfulness_score"].value_counts().sort_index())
        with chart_col2:
            st.markdown("#### Retrieval hit rate by question category")
            st.bar_chart(df.groupby("category")["retrieval_hit"].mean())

        if len(cited_df) and fabricated:
            st.markdown("#### ⚠ Fabricated citations")
            st.dataframe(
                cited_df[~cited_df["citation_verifiable"].astype(bool)][["id", "answer"]],
                use_container_width=True,
            )

        st.markdown("#### Full results")
        st.dataframe(df, use_container_width=True)
