"""
ingest.py — Step 1 of the RAG pipeline: turn raw documents into searchable vectors.

CONCEPT: An LLM can't "read" your files. So we:
  1. Load the raw text out of each document.
  2. Split ("chunk") it into small overlapping pieces — small enough that each
     piece is a focused, self-contained idea the LLM can use as context.
  3. Convert each chunk into a vector embedding — a list of numbers that
     captures the *meaning* of that text (similar meanings -> similar numbers).
  4. Store those vectors in a vector database (ChromaDB here) so we can later
     search by meaning instead of by exact keyword match.

Run this once (or whenever your source docs change):
    python ingest.py
"""

import os
import glob
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb

load_dotenv()

# NOTE: We are NOT using OpenAI embeddings (which cost money per call).
# Instead we let ChromaDB use its own built-in FREE local embedding model
# (a small model called all-MiniLM-L6-v2). It downloads once (~80MB) the
# first time you run this, then runs entirely on your own CPU — no API
# calls, no cost, works offline after the first download.

DOCS_DIR = "sample_docs"
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 800       # characters per chunk — tune this; smaller = more precise, larger = more context
CHUNK_OVERLAP = 150    # overlap so we don't cut a sentence's meaning in half at a chunk boundary


def load_text_from_file(path: str) -> str:
    """Extract raw text from a .pdf or .txt file."""
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks. This is a simple sliding-window
    splitter — good enough to learn on. Production systems often use
    sentence-aware or semantic chunking instead."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def main():
    doc_paths = glob.glob(os.path.join(DOCS_DIR, "*.pdf")) + glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    if not doc_paths:
        print(f"No documents found in {DOCS_DIR}/. Add some .pdf or .txt files and re-run.")
        return

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # No embedding_function passed = ChromaDB uses its free default local model automatically.
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    all_chunks, all_ids, all_metadatas = [], [], []
    for path in doc_paths:
        text = load_text_from_file(path)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{os.path.basename(path)}::chunk-{i}")
            all_metadatas.append({"source": os.path.basename(path), "chunk_index": i})
        print(f"  {path}: {len(chunks)} chunks")

    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metadatas)
    print(f"\nDone. Ingested {len(all_chunks)} chunks from {len(doc_paths)} document(s) into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
