# GenAI RAG Assistant (Phase 1)

A minimal but real Retrieval-Augmented Generation (RAG) system: ask questions
over your own documents and get answers grounded in them, with sources cited.

This is Phase 1 of a bigger build. Once this works, we'll layer on: LlamaIndex,
Docker, GitHub Actions CI/CD, an MCP server, and multi-agent orchestration —
each phase extending this same repo.

## How it works

```
Your documents (PDF/txt)
        |
   ingest.py   -> chunks text, embeds chunks, stores vectors in ChromaDB
        |
   chroma_store/  (a local vector database, persisted to disk)
        |
   app.py (FastAPI) -> /ask endpoint:
        1. Embed the user's question
        2. Semantic search ChromaDB for the most relevant chunks
        3. Stuff those chunks into an LLM prompt as context
        4. LLM generates an answer grounded in that context
```

## Setup (run this yourself, step by step — don't just skim it)

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Add your OpenAI key:**
   ```
   cp .env.example .env
   # edit .env and paste your real key
   ```

3. **Add some documents to ingest:**
   Drop 3-5 PDFs or .txt files into `sample_docs/` — use something you
   actually care about (your old project docs, a textbook chapter, company
   policy docs, anything with real text).

4. **Run ingestion** (turns your docs into searchable vectors):
   ```
   python ingest.py
   ```
   You should see output like `sample_docs/foo.pdf: 12 chunks`.

5. **Start the API:**
   ```
   uvicorn app:app --reload
   ```

6. **Ask it something:**
   ```
   curl -X POST http://localhost:8000/ask \
        -H "Content-Type: application/json" \
        -d '{"question": "What is this document about?"}'
   ```

## Things to actually try breaking (this is how you learn it, not just run it)

- Change `CHUNK_SIZE` in `ingest.py` to 200 vs 2000 — re-ingest and see how
  answer quality changes. This is a real trade-off interviewers ask about.
- Change `TOP_K` in `app.py` from 4 to 1 — watch answers get worse when
  there's not enough context.
- Ask a question with NO answer in your documents — confirm it says
  "I don't have enough information" instead of hallucinating. This is
  what "grounding" means in practice.
- Look at what `results["documents"]` actually contains by printing it —
  understand that the LLM never sees your whole document, only the
  chunks retrieval picked.

## Interview talking points this gives you

- Why chunking size/overlap matters (context vs. precision trade-off)
- What an embedding actually is (a vector representing meaning)
- Why RAG reduces hallucination vs. asking the LLM directly
- The difference between the vector DB (storage + similarity search) and
  the LLM (reasoning + generation) — they're two separate systems working
  together
- Why the system prompt explicitly says "don't make something up" —
  prompt engineering to enforce grounding

## Next phases (same repo, future branches)

- **Phase 2:** Rebuild ingestion/retrieval using LlamaIndex instead of raw ChromaDB calls
- **Phase 3:** Dockerfile + docker-compose (add Postgres for chat history)
- **Phase 4:** GitHub Actions workflow (lint, test, build, push image)
- **Phase 5:** Wrap `/ask` as an MCP server tool
- **Phase 6:** LangGraph/CrewAI agent that calls this as a tool
