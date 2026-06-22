"""
RAG retrieval tool (Feature 2).

Handles document loading + chunking + embedding + ChromaDB storage, and
exposes a `retrieve()` function the RAG agent node uses at query time.

IMPORTANT: `build_index()` is meant to be run ONCE (via scripts/ingest_kb.py)
to (re)build the persisted ChromaDB collection on disk. The app itself only
ever calls `retrieve()`, which opens the existing persisted collection - it
never re-embeds the corpus on startup.
"""
import os
from typing import List

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

from src.state import RagChunk

KB_DIR = "data/knowledge_base"
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "university_policies"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# Chosen over the English-centric all-MiniLM-L6-v2 because the knowledge
# base (Feature 2) is in Greek - this model supports 50+ languages,
# including Greek, with much better retrieval quality for non-English text.

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # characters of overlap between consecutive chunks

_embedding_fn = None


def _get_embedding_fn():
    """Lazy-load the embedding function so importing this module doesn't
    itself trigger a model download - only actually calling build_index()
    or retrieve() does."""
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    return _embedding_fn


def _read_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple fixed-size sliding-window chunker over whitespace-normalized
    text. Good enough for a small policy-document corpus; swap for a
    semantic/markdown-aware chunker for larger or more structured corpora."""
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=_get_embedding_fn()
    )


def build_index(kb_dir: str = KB_DIR) -> int:
    """(Re)build the ChromaDB collection from the documents in kb_dir.
    Returns the number of chunks indexed. Run this via scripts/ingest_kb.py,
    not on every app startup."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    # Drop any existing collection so re-running this script is idempotent.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME, embedding_function=_get_embedding_fn()
    )

    ids, documents, metadatas = [], [], []
    for filename in sorted(os.listdir(kb_dir)):
        path = os.path.join(kb_dir, filename)
        if not os.path.isfile(path):
            continue
        text = _read_file(path)
        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            ids.append(f"{filename}::{i}")
            documents.append(chunk)
            metadatas.append({"source": filename, "chunk_index": i})

    if documents:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(documents)


def retrieve(query: str, k: int = 4) -> List[RagChunk]:
    """Query the persisted collection for the top-k most relevant chunks.
    Each result includes its source filename and chunk index so the agent's
    answer is traceable back to the source document."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(query_texts=[query], n_results=min(k, collection.count()))

    chunks: List[RagChunk] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        chunks.append(
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "distance": dist,
            }
        )
    return chunks
