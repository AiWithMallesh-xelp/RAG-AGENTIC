# Advanced Agentic RAG

Persistent agentic RAG with hybrid retrieval, parent expansion, and Self-RAG evaluation.

## Setup

```bash
cd agentic_rag
cp .env.example .env
# Set GOOGLE_API_KEY and GEMINI_API_KEY (same key is fine)

pip install -r requirements.txt
```

Requires **Qdrant** (`QDRANT_URL`) and **Ollama** with `gemma4:26b` for reranking.

## Web UI (Flask)

Set `FRONTEND=true` in `.env`, then:

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 — upload PDF/DOCX, chat, view index status, reset.

Default LLM is `gemini-2.5-flash` (set `LLM_MODEL` in `.env`). Ingest runs in the background so the UI stays responsive on large PDFs. Health check: `GET /api/health`.

## CLI

```bash
python main.py ingest --index-dir ./idx --files doc.pdf
python main.py status --index-dir ./idx
python main.py query --index-dir ./idx "Your question"
python main.py interactive --index-dir ./idx
```

## Verification

```bash
python verify.py
```
