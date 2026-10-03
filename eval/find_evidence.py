"""
find_evidence.py: find the chunk(s) containing a phrase.

Use it to pick an 'evidence_hint' for a qa_pairs.jsonl question, and to see whether
a wrong answer was a retrieval problem or a generation problem. It also tells you
whether the phrase sits inside the first CONTEXT_SNIPPET_CHARS of the chunk (what
the generator actually sees) or beyond that cut-off.

Usage (from the project root):
  python eval/find_evidence.py "getting up" --doc Cheruiyot
  python eval/find_evidence.py "preliminary objection" --doc Kariithi
"""

import argparse
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "policy_docs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phrase", help="text to search for (case-insensitive)")
    parser.add_argument("--doc", help="only search chunks whose doc_title contains this, e.g. Cheruiyot")
    parser.add_argument("--chars", type=int, default=1000, help="generator snippet length (CONTEXT_SNIPPET_CHARS)")
    args = parser.parse_args()

    collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"])

    phrase = args.phrase.lower()
    hits = 0
    for chunk_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        if args.doc and args.doc.lower() not in meta.get("doc_title", "").lower():
            continue
        position = text.lower().find(phrase)
        if position == -1:
            continue
        hits += 1
        where = "WITHIN" if position < args.chars else f"BEYOND (at char {position})"
        start = max(0, position - 150)
        snippet = text[start:position + 250].replace("\n", " ")
        print(f"\n[{chunk_id}]  section: {meta.get('section')}")
        print(f"  phrase is {where} the first {args.chars} chars the generator sees")
        print(f"  ...{snippet}...")

    print(f"\n{hits} chunk(s) found.")


if __name__ == "__main__":
    main()
