"""
Step 4: Retrieval + generation pipeline.

Reads:  ./chroma_db/ (collection built by embed_index.py)
Uses:   a local Ollama model for generation

Requires: `pip install ollama chromadb sentence-transformers`
          `ollama pull llama3.1:8b`  (or swap MODEL_NAME below)
"""

import re

import chromadb
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import fee_router

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "policy_docs"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
GENERATION_MODEL_NAME = "qwen2.5:7b"   # swap for "llama3.1:8b" or "mistral" if preferred
TOP_K = 10   # raised from 5 — corpus has grown well past the original 2 test rulings,
            # and ruling-specific questions (which don't benefit from the statute
            # boost below) need more room to find the right case among many similar ones
NUM_CTX = 8192   # Ollama defaults to 2048 for most models — far too small once retrieved
                 # context grows with a bigger corpus. Silent truncation at 2048 was the
                 # root cause of the calc_01/06/07 regression: front-loaded Question/
                 # VERIFIED CALCULATION content was getting dropped. Raise further if you
                 # still see the model answering something unrelated to the question —
                 # check `ollama show <model>` for that model's actual max context first.
CONTEXT_SNIPPET_CHARS = 2200  # chunks are built at CHUNK_SIZE_CHARS=1800 in chunk.py, so
                              # this must stay above that or it silently truncates *normal*
                              # chunks mid-sentence — which is exactly what an earlier,
                              # too-aggressive 800-char cap did here (see the statute_01/
                              # statute_02 regression: retrieval was correct, but the
                              # answer-bearing sentence fell past char 800 and got cut).
                              # This is a guard against outliers (e.g. oversized fee-table
                              # chunks), not a routine truncator — 5 chunks at 2200 chars
                              # (~2750 tokens) is still comfortably under NUM_CTX.

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

        client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = client.get_collection(collection_name)
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Pull every chunk already in Chroma and build a BM25 keyword index over it.
        Why: dense embeddings under-serve short, distinctive phrases ("fourteen
        days", a specific case name) once the corpus is large enough that many
        chunks are semantically similar — BM25's exact/near-exact term matching
        catches these where pure semantic distance ranks them too low to surface
        in the top-k. Rebuilt from Chroma directly (not re-reading chunks.jsonl)
        so it's always in sync with whatever's actually indexed."""
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
                continue  # no keyword overlap at all — don't force in irrelevant results
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
        """Low-level Chroma query, returned as a list of chunk dicts."""
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

    def retrieve(self, query: str, top_k: int = TOP_K, statute_boost: int = 3, bm25_k: int = 5) -> list[dict]:
        """Hybrid retrieval: dense semantic search (top_k) + a guaranteed statute-only
        slice (statute_boost) + BM25 keyword search (bm25_k), combined via reciprocal
        rank fusion (RRF). Each method catches what the others miss: semantic search
        finds paraphrased/conceptual matches; the statute boost guarantees the Order's
        own ~150 chunks aren't drowned out by a much larger ruling corpus; BM25 finds
        exact short phrases ("fourteen days", a specific case name) that dense
        embeddings under-rank once many chunks are semantically similar.
        RRF avoids needing to normalize BM25 scores against cosine distances — it
        only uses each method's RANK, which is directly comparable across methods."""
        query_with_instruction = f"Represent this sentence for searching relevant passages: {query}"
        query_embedding = self.embedding_model.encode([query_with_instruction], normalize_embeddings=True)

        general_chunks = self._query_collection(query_embedding, n_results=top_k)
        statute_chunks = self._query_collection(query_embedding, n_results=statute_boost, where={"doc_type": "statute"})
        bm25_chunks = self._bm25_search(query, n_results=bm25_k)

        rrf_k = 60  # standard RRF damping constant
        rrf_scores: dict[str, float] = {}
        chunk_by_id: dict[str, dict] = {}
        for ranked_list in (statute_chunks, general_chunks, bm25_chunks):
            for rank, chunk in enumerate(ranked_list):
                cid = chunk["chunk_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
                chunk_by_id.setdefault(cid, chunk)  # keep first-seen version (has full metadata either way)

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
            # Cap length per chunk — with a larger corpus, TOP_K chunks at full length
            # can still blow the context budget even with NUM_CTX raised.
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
            # Question + calc block appear at BOTH ends of the message: if a context
            # window limit ever truncates one end (front or back, depending on the
            # runtime), the other copy still gets through. This is what actually
            # protects against the calc_01/06/07 regression — reordering alone
            # doesn't, since truncation direction isn't something this code controls.
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

        response = ollama.chat(
            model=self.generation_model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            options={"num_ctx": NUM_CTX},
        )
        return response["message"]["content"]

    def answer(self, query: str, top_k: int = TOP_K) -> dict:
        """Full pipeline: try the fee calculator router first (exact figures for
        computable questions), then retrieve + generate either way — for a routed
        question, retrieval only supplies supporting citations; the LLM is told
        the verified figure and must not alter it."""
        route_result = fee_router.route(query)
        retrieved_chunks = self.retrieve(query, top_k=top_k)

        if route_result is not None:
            answer_text = self.generate(query, retrieved_chunks, calculator_result=route_result)
        else:
            answer_text = self.generate(query, retrieved_chunks)

        return {
            "query": query,
            "answer": answer_text,
            "retrieved_chunks": retrieved_chunks,
            "calculator_result": route_result,  # None if the router didn't classify this query
        }


if __name__ == "__main__":
    pipeline = RagPipeline()

    print(f"RAG pipeline ready (generation model: {GENERATION_MODEL_NAME}). Type 'exit' to quit.\n")
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
