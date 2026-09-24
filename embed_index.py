"""
Step 3: Embed chunks.jsonl and build a persistent Chroma vector index.

Reads:  data/chunks.jsonl   (produced by chunk.py)
Writes: ./chroma_db/        (persistent vector store on disk)
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("./data/chunks.jsonl")
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "policy_docs"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64


def load_chunks(chunks_path: Path) -> list[dict]:
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")
    return chunks


def embed_and_index(chunks: list[dict]) -> chromadb.Collection:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # start clean each run; drop this if you want incremental indexing instead
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [
        {
            "source_url": c["source_url"],
            "doc_title": c["doc_title"],
            "section": c["section"],
            "doc_type": c.get("doc_type", "Unknown"),
            "case_number": c.get("case_number", "Unknown"),
            "court": c.get("court", "Unknown"),
            "ruling_date": c.get("ruling_date", "Unknown"),
            "case_citation": c.get("case_citation", "Unknown"),
        }
        for c in chunks
    ]

    for start in range(0, len(texts), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_texts = texts[start:end]
        # bge models expect a query-side instruction prefix at search time (see query_index below);
        # passage/document side is embedded as-is.
        batch_embeddings = model.encode(
            batch_texts, normalize_embeddings=True, show_progress_bar=False
        )
        collection.add(
            ids=ids[start:end],
            embeddings=batch_embeddings.tolist(),
            documents=batch_texts,
            metadatas=metadatas[start:end],
        )
        print(f"  Indexed {min(end, len(texts))}/{len(texts)} chunks")

    print(f"Built collection '{COLLECTION_NAME}' with {collection.count()} vectors at {CHROMA_DIR}")
    return collection


def query_index(collection: chromadb.Collection, model: SentenceTransformer, query: str, top_k: int = 5):
    """Example retrieval helper — bge-small wants a query instruction prefix for best results."""
    query_with_instruction = f"Represent this sentence for searching relevant passages: {query}"
    query_embedding = model.encode([query_with_instruction], normalize_embeddings=True)
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=top_k)
    return results


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)
    collection = embed_and_index(chunks)

    # quick smoke test
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    test_results = query_index(collection, model, "How are instruction fees calculated under the Advocates Remuneration Order?", top_k=3)
    for i, (doc, meta) in enumerate(zip(test_results["documents"][0], test_results["metadatas"][0])):
        print(f"\n--- Result {i+1} ({meta['case_citation']} / {meta['section']}) ---")
        print(doc[:200], "...")
