"""
Run this ONCE (and again any time the knowledge base documents change) to
build the persisted ChromaDB index:

    python scripts/ingest_kb.py

main.py never calls build_index() itself - it only ever reads the existing
persisted collection via src.tools.rag.retrieve(). This script is what
satisfies the "persist the vector store, don't re-embed on every run"
requirement.
"""
import sys
import time

sys.path.insert(0, ".")  # allow running as `python scripts/ingest_kb.py` from repo root

from src.tools.rag import build_index  # noqa: E402

if __name__ == "__main__":
    start = time.time()
    n_chunks = build_index()
    elapsed = time.time() - start
    print(f"Indexed {n_chunks} chunks into ChromaDB at ./chroma_db in {elapsed:.1f}s")
