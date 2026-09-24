# Run this against your own chroma_db to see what actually got retrieved for the query.
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("policy_docs")

query = "Represent this sentence for searching relevant passages: legal fees for discharge of charge over a loan of 2.8m"
emb = model.encode([query], normalize_embeddings=True)
results = collection.query(query_embeddings=emb.tolist(), n_results=5)

for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    print(f"--- {meta.get('doc_type')} / {meta.get('section')} (distance {dist:.3f}) ---")
    print(doc[:300], "\n")
