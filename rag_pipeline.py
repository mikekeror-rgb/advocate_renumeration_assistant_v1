"""
Step 4: Retrieval + generation pipeline.

Reads:  ./chroma_db/ (collection built by embed_index.py)
Uses:   Groq hosted API for generation (free tier, fast, Streamlit-friendly)

Requires: `pip install groq chromadb sentence-transformers rank-bm25`
          Set env var: GROQ_API_KEY=gsk_...
"""

import os
import time
import re
from pathlib import Path
import chromadb
from groq import Groq, APIError, RateLimitError, AuthenticationError, APIStatusError
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import fee_router

CHROMA_DIR =  str(Path(__file__).resolve().parent / "chroma_db")
COLLECTION_NAME = "policy_docs"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
GENERATION_MODEL_NAME = "openai/gpt-oss-20b" # fast free-tier model; swap e.g. "llama-3.3-70b-versatile"
TOP_K = 5
# Groq models (Llama 3.1 8B / 3.3 70B) support large context natively (~131k).
# We no longer pass num_ctx; instead we keep CONTEXT_SNIPPET_CHARS so the prompt stays reasonable.
CONTEXT_SNIPPET_CHARS = 1000

SYSTEM_PROMPT = """You are a legal research assistant specializing in advocate remuneration \
under Kenya's Advocates Remuneration Order and related case law. Answer the user's question \
using ONLY the provided context chunks below, which are excerpts from Kenyan court rulings on \
taxation of advocates' bills of costs.

Each chunk is tagged with [chunk_id] and its case citation — cite BOTH the chunk_id and the \
case citation in square brackets at the end of each sentence that relies on them, \
e.g. "...instruction fees are not multiplied per party where only one pleading was filed \
[chunk_id: xyz__s2__c0, Kithi & Co. Advocates v Greenwood Ltd]."

Rules:
- If a VERIFIED CALCULATION block appears in the user's message, it is authoritative, pre-computed \
context — treat it exactly as if it were a figure quoted verbatim from the Order, and build your \
answer around it. NEVER respond "Not covered in the provided documents" when a VERIFIED CALCULATION \
block is present — it exists specifically because the answer required arithmetic the Order's schedule \
supports but never states as a single sentence; the calculation has already been done for you correctly.
- If there is no VERIFIED CALCULATION block and the answer is not contained in the context chunks, \
respond exactly: "Not covered in the provided documents."
- Do not use outside knowledge of the Advocates Remuneration Order or case law, even if you know it.
- Context chunks are labeled by source: some come from THE ADVOCATES REMUNERATION ORDER itself \
(the actual fee schedules and rules) and others from court RULINGS interpreting it. If a specific \
fee amount, percentage, or schedule figure is quoted verbatim in an Order chunk, you may state it \
and cite that chunk. If it only appears discussed or applied within a ruling (not quoted from the \
Order itself), say so and do not treat the ruling's application of it as the authoritative figure.
- Be concise and precise; do not pad the answer with restated context.
"""


class RagPipeline:
    def __init__(
        self,
        chroma_dir: str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        generation_model_name: str = GENERATION_MODEL_NAME,
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name, device="cpu")
        self.generation_model_name = generation_model_name

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
                "and export it (or add it to Streamlit secrets)."
            )
        self.groq_client = Groq(api_key=api_key)

        client = chromadb.PersistentClient(path=chroma_dir)

        existing = [c.name for c in client.list_collections()]
        if collection_name not in existing:
            raise RuntimeError(
                        f"Chroma collection '{collection_name}' not found in '{chroma_dir}'. "
                        f"Available: {existing}. "
                        "Did you push the chroma_db/ folder to GitHub?"
                    )
        self.collection = client.get_collection(collection_name)
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Pull every chunk already in Chroma and build a BM25 keyword index over it."""
        all_data = self.collection.get(include=["documents", "metadatas"])
        self._bm25_ids = all_data["ids"]
        self._bm25_documents = all_data["documents"]
        self._bm25_metadatas = all_data["metadatas"]
        tokenized_corpus = [self._tokenize(doc) for doc in self._bm25_documents]
        self._bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index built over {len(self._bm25_ids)} chunks")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _bm25_search(self, query: str, n_results: int) -> list[dict]:
        scores = self._bm25.get_scores(self._tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
        chunks = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            meta = self._bm25_metadatas[idx]
            chunks.append({
                "chunk_id": self._bm25_ids[idx],
                "text": self._bm25_documents[idx],
                "doc_title": meta["doc_title"],
                "section": meta["section"],
                "source_url": meta["source_url"],
                "doc_type": meta.get("doc_type", "Unknown"),
                "case_number": meta.get("case_number", "Unknown"),
                "court": meta.get("court", "Unknown"),
                "ruling_date": meta.get("ruling_date", "Unknown"),
                "case_citation": meta.get("case_citation", "Unknown"),
                "bm25_score": scores[idx],
            })
        return chunks

    def _query_collection(self, query_embedding, n_results: int, where: dict | None = None) -> list[dict]:
        kwargs = {"query_embeddings": query_embedding.tolist(), "n_results": n_results}
        if where is not None:
            kwargs["where"] = where
        results = self.collection.query(**kwargs)
        if not results["ids"][0]:
            return []
        chunks = []
        for doc, meta, dist, chunk_id in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0], results["ids"][0]
        ):
            chunks.append({
                "chunk_id": chunk_id,
                "text": doc,
                "doc_title": meta["doc_title"],
                "section": meta["section"],
                "source_url": meta["source_url"],
                "doc_type": meta.get("doc_type", "Unknown"),
                "case_number": meta.get("case_number", "Unknown"),
                "court": meta.get("court", "Unknown"),
                "ruling_date": meta.get("ruling_date", "Unknown"),
                "case_citation": meta.get("case_citation", "Unknown"),
                "distance": dist,
            })
        return chunks

    def retrieve(self, query: str, top_k: int = TOP_K, statute_boost: int = 2, bm25_k: int = 3) -> list[dict]:
        query_with_instruction = f"Represent this sentence for searching relevant passages: {query}"
        query_embedding = self.embedding_model.encode([query_with_instruction], normalize_embeddings=True)

        general_chunks = self._query_collection(query_embedding, n_results=top_k)
        statute_chunks = self._query_collection(query_embedding, n_results=statute_boost, where={"doc_type": "statute"})
        bm25_chunks = self._bm25_search(query, n_results=bm25_k)

        rrf_k = 60
        rrf_scores: dict[str, float] = {}
        chunk_by_id: dict[str, dict] = {}
        for ranked_list in (statute_chunks, general_chunks, bm25_chunks):
            for rank, chunk in enumerate(ranked_list):
                cid = chunk["chunk_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
                chunk_by_id.setdefault(cid, chunk)

        merged_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        return [chunk_by_id[cid] for cid in merged_ids[: top_k + statute_boost]]

    def build_context_block(self, retrieved_chunks: list[dict]) -> str:
        parts = []
        for chunk in retrieved_chunks:
            source_label = (
                f"THE ADVOCATES REMUNERATION ORDER ({chunk['case_citation']})"
                if chunk["doc_type"] == "statute"
                else f"{chunk['case_citation']} / {chunk['case_number']} / {chunk['court']} / dated {chunk['ruling_date']}"
            )
            snippet = chunk["text"][:CONTEXT_SNIPPET_CHARS]
            parts.append(
                f"[chunk_id: {chunk['chunk_id']}] (source: {source_label} / section: {chunk['section']})\n{snippet}"
            )
        return "\n\n---\n\n".join(parts)

    def generate(self, query: str, retrieved_chunks: list[dict], calculator_result=None) -> str:
         context_block = self.build_context_block(retrieved_chunks)

         if calculator_result is not None:
                calc_block = (
                    f"VERIFIED CALCULATION (this is your answer's basis — computed exactly from the Order's "
                    f"schedule formula; state this figure precisely, do not recompute, round differently, "
                    f"alter it, or say the question is not covered):\n"
                    f"{calculator_result.explanation}\n"
                    f"[source: {calculator_result.schedule_citation}]"
                )
                user_message = (
                    f"Question: {query}\n\n"
                    f"{calc_block}\n\n"
                    f"Additional citations below are supplementary only — they do not override the "
                    f"VERIFIED CALCULATION above:\n{context_block}\n\n"
                    f"---\n"
                    f"Reminder — Question: {query}\n"
                    f"Reminder — {calc_block}"
                )
         else:
                user_message = f"Context:\n{context_block}\n\nQuestion: {query}"

         last_error = None
         for attempt in range(4):
                try:
                    response = self.groq_client.chat.completions.create(
                        model=self.generation_model_name,  # should be "openai/gpt-oss-20b"
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_message},
                        ],
                        temperature=0.1,
                        max_completion_tokens=4096,  # gpt-oss-20b is a REASONING model — its hidden
                        # chain-of-thought tokens draw from this SAME budget before any visible
                        # answer text is emitted. At max_tokens=1024, harder questions could spend
                        # the entire budget reasoning and return content="" with finish_reason=
                        # "length" — no exception, just silent empty output. This was the actual
                        # root cause of "long/complex questions return nothing" reported in testing.
                        reasoning_effort="low",  # supported only by gpt-oss-20b/120b on Groq;
                        # reduces how many tokens go to invisible reasoning, leaving more of the
                        # budget for the actual visible answer. Raise to "medium"/"high" only if
                        # answer quality suffers — that trades reliability for occasional lost answers again.
                    )
                    choice = response.choices[0]
                    content = choice.message.content or ""
                    if not content.strip() and choice.finish_reason == "length":
                        # Defensive: even with the settings above, don't silently return empty —
                        # this is exactly the failure mode that was invisible before. Surface it
                        # as a clear, catchable error instead of a blank string reaching the UI.
                        raise RuntimeError(
                            "Groq returned empty content because the reasoning-token budget "
                            "was exhausted before any answer text was written (finish_reason="
                            "'length' with empty content). Try again, or reduce retrieved "
                            "context / raise max_completion_tokens further."
                        )
                    return content

                except RateLimitError as e:
                    last_error = e
                    time.sleep(15 * (attempt + 1))  # 15s, 30s, 45s, 60s
                except APIStatusError as e:
                    if getattr(e, "status_code", None) == 429:
                        last_error = e
                        time.sleep(15 * (attempt + 1))
                    else:
                        details = {
                            "type": type(e).__name__,
                            "message": getattr(e, "message", str(e)),
                            "status_code": getattr(e, "status_code", None),
                            "body": getattr(e, "body", None),
                            "model": self.generation_model_name,
                        }
                        raise RuntimeError(f"GROQ FULL ERROR: {details}") from e
                except Exception as e:
                    details = {
                        "type": type(e).__name__,
                        "message": getattr(e, "message", str(e)),
                        "status_code": getattr(e, "status_code", None),
                        "body": getattr(e, "body", None),
                        "model": self.generation_model_name,
                    }
                    raise RuntimeError(f"GROQ FULL ERROR: {details}") from e

         raise RuntimeError(
                f"Groq rate/token limit after retries. Try again in a minute. Last error: {last_error}"
            )
    def retrieve_bill_sources(self, advocate_client: bool = False) -> list[dict]:
        """Retrieve the Order chunks that a Schedule 6 bill's rates come from.
        Uses a fixed query naming the bill's items rather than the user's own
        wording, since a bill request ("my case had 10 letters...") rarely
        shares vocabulary with the Schedule text itself."""
        query = (
            "Schedule 6 costs of proceedings in the High Court party and party costs "
            "instruction fees getting up or preparing for trial drawing folios "
            "perusals correspondence letters attendances service"
        )
        if advocate_client:
            query += " advocate and client costs fees prescribed in A increased by 50%"
        return self.retrieve(query)

    def answer(self, query: str, top_k: int = TOP_K) -> dict:
        route_result = fee_router.route(query)

        # A full itemized bill is already complete and verified — return it
        # directly. Sending a large table through the LLM risks mangled figures
        # and wastes gpt-oss reasoning tokens (the cause of empty answers).
        # Retrieval still runs (it's local and cheap) so the answer shows the
        # Order chunks the bill's rates come from, like every other answer.
        if route_result is not None and route_result.scenario == "schedule6_full_bill":
            return {
                "query": query,
                "answer": route_result.explanation,
                "retrieved_chunks": self.retrieve_bill_sources(
                    advocate_client="Part B" in route_result.explanation
                ),
                "calculator_result": route_result,
            }

        retrieved_chunks = self.retrieve(query, top_k=top_k)

        if route_result is not None:
            answer_text = self.generate(query, retrieved_chunks, calculator_result=route_result)
        else:
            answer_text = self.generate(query, retrieved_chunks)

        return {
            "query": query,
            "answer": answer_text,
            "retrieved_chunks": retrieved_chunks,
            "calculator_result": route_result,
        }


if __name__ == "__main__":
    pipeline = RagPipeline()

    print(f"RAG pipeline ready (generation model: {GENERATION_MODEL_NAME} via Groq). Type 'exit' to quit.\n")
    while True:
        user_query = input("Question: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            break
        if not user_query:
            continue

        result = pipeline.answer(user_query)
        print(f"\nAnswer:\n{result['answer']}\n")
        if result["calculator_result"] is not None:
            print(f"[Calculated exactly via fee_router — {result['calculator_result'].schedule_citation}]")
        print("Sources used:")
        for chunk in result["retrieved_chunks"]:
            print(f"  - {chunk['chunk_id']} ({chunk['case_citation']} / {chunk['section']})")
        print()
