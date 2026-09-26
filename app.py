"""
Step 7: Minimal Streamlit UI.

Run with: streamlit run app.py

Two tabs:
- "Ask a question": the RAG assistant itself — question in, answer + sources + a
  clear flag when the fee calculator (not the LLM) produced the number.
- "Eval results": reads eval/results.csv (produced by eval/eval.py) and renders
  the same precision/faithfulness metrics as the CLI summary, as charts.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from rag_pipeline import RagPipeline

st.set_page_config(page_title="Advocate Remuneration Assistant", layout="wide")


@st.cache_resource
def load_pipeline() -> RagPipeline:
    return RagPipeline()


tab_chat, tab_eval = st.tabs(["Ask a question", "Eval results"])


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

        # Calculator badge first (optional — move below answer if you prefer)
        if result.get("calculator_result") is not None:
            st.success(
                f"✅ Calculated exactly via fee_router — "
                f"{result['calculator_result'].schedule_citation}"
            )

        # Answer — always render something visible
        st.markdown("### Answer")
        answer = (result.get("answer") or "").strip()
        if answer:
            st.markdown(answer)   # markdown handles bold/lists better than st.write
        else:
            st.warning(
                "The model returned an empty answer (often a rate limit). "
                "Wait a few seconds and try the same question again."
            )

        retrieved = result["retrieved_chunks"]
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


# ---------------------------------------------------------------------------
# Tab 2: Eval results
# ---------------------------------------------------------------------------
with tab_eval:
    st.title("Evaluation Results")

    results_path = Path("eval/results.csv")
    if not results_path.exists():
        st.warning("No `eval/results.csv` found yet — run `python eval/eval.py` first, then refresh this page.")
    else:
        df = pd.read_csv(results_path)
        df["category"] = df["id"].str.extract(r"^([a-z]+)_")

        # --- summary metrics ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Retrieval hit rate", f"{df['retrieval_hit'].mean():.0%}")

        numeric_df = df[df["expected_type"] == "numeric"]
        if len(numeric_df):
            col2.metric("Numeric accuracy", f"{numeric_df['numeric_exact_match'].mean():.0%}")

        cited_df = df[df["citation_verifiable"].notna()]
        if len(cited_df):
            fabricated = (~cited_df["citation_verifiable"].astype(bool)).sum()
            col3.metric(
                "Citation accuracy",
                f"{cited_df['citation_verifiable'].mean():.0%}",
                delta=f"-{fabricated} fabricated" if fabricated else None,
                delta_color="inverse",
            )

        col4.metric("Avg faithfulness", f"{df['faithfulness_score'].mean():.2f} / 5")

        st.caption(
            "Faithfulness is LLM-judged and should be read as a rough secondary signal — "
            "retrieval hit rate, numeric accuracy, and citation accuracy above are deterministic "
            "and more reliable on their own."
        )

        # --- charts ---
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### Faithfulness score distribution")
            score_counts = df["faithfulness_score"].value_counts().sort_index()
            st.bar_chart(score_counts)

        with chart_col2:
            st.markdown("#### Retrieval hit rate by question category")
            category_hit_rate = df.groupby("category")["retrieval_hit"].mean()
            st.bar_chart(category_hit_rate)

        # --- fabricated citations, called out explicitly ---
        if len(cited_df) and fabricated:
            st.markdown("#### ⚠ Fabricated citations")
            st.dataframe(
                cited_df[~cited_df["citation_verifiable"].astype(bool)][["id", "answer"]],
                use_container_width=True,
            )

        # --- full table ---
        st.markdown("#### Full results")
        st.dataframe(df, use_container_width=True)
