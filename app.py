"""
app.py — Step 2 of the RAG pipeline: answer questions using retrieved context.

CONCEPT (this is "RAG" — Retrieval-Augmented Generation):
  1. Take the user's question and embed it (same embedding model as ingest.py).
  2. Search ChromaDB for the chunks whose vectors are closest in meaning to
     the question's vector — this is "semantic search."
  3. Stuff those chunks into the LLM's prompt as context, then ask the LLM
     to answer USING that context. This grounds the answer in your actual
     documents instead of the model's general training knowledge.

Run the server:
    uvicorn app:app --reload

Then test it:
    curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
         -d '{"question": "What does this document say about X?"}'
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
import google.generativeai as genai

load_dotenv()

CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "knowledge_base"
TOP_K = 8  # how many chunks to retrieve per question — tune based on chunk size/quality

app = FastAPI(title="GenAI RAG Assistant")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-3.6-flash")  # free-tier friendly model

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
# No embedding_function passed = same free default local model used in ingest.py
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(payload: Question):
    # 1. Retrieve — semantic search over the vector store
    results = collection.query(query_texts=[payload.question], n_results=TOP_K)
    retrieved_chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]

    context = "\n\n---\n\n".join(retrieved_chunks)

    # 2. Augment — build a prompt that forces the model to rely on retrieved context
    prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "context provided below. If the answer isn't in the context, say you "
        "don't have enough information — do not make something up.\n\n"
        f"Context:\n{context}\n\nQuestion: {payload.question}"
    )

    # 3. Generate
    response = gemini_model.generate_content(prompt)

    return {
        "answer": response.text,
        "sources": sources,
        "retrieved_chunk_count": len(retrieved_chunks),
    }


@app.get("/health")
def health():
    return {"status": "ok", "collection_count": collection.count()}
