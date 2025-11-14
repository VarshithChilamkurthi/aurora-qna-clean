# Aurora Member Q&A

A small RAG-style question-answering API over the provided member messages.

## What it does
- Builds semantic embeddings for member messages (sentence-transformers + FAISS).
- Exposes `GET /ask?q=<question>` that returns a concise answer based on messages.
- Falls back to a local `messages.json` if the public API is inaccessible.
- Includes a simple rule-based answerer so the service returns understandable answers without an OpenAI key.

## Quick start (local)
1. Create and activate venv:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Build the index (uses messages.json fallback if API is unreachable):

bash
Copy code
python -c "from app.embed_index import build_index; build_index(save=True)"
Run:

bash
Copy code
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
Example:

bash
Copy code
curl "http://127.0.0.1:8000/ask?q=When%20is%20Layla%20planning%20her%20trip%20to%20London%3F"
API
GET /ask?q=...
Returns:

json
Copy code
{ "answer": "..." }
Reindexing
To rebuild the vector index (from API or local file):

bash
Copy code
curl -X POST "http://127.0.0.1:8000/reindex"
# or
python -c "from app.embed_index import build_index; build_index(save=True)"
Note about data source
During development the public messages API returned HTTP 403. For reproducibility this repo includes messages.json. The service will attempt to fetch from the API but falls back to messages.json if unavailable.

Design notes (brief)
Retrieval: sentence-transformers (all-MiniLM-L6-v2) + FAISS.

Generation: optional OpenAI. If OPENAI_API_KEY is set the server uses the LLM to synthesize concise answers; otherwise a rule-based fallback returns friendly answers.

Alternatives considered: managed vector DBs (Pinecone/PGVector), open-source LLMs (local) and extractive QA models. See project notes in the repo for details.

Files included
app/ — service code

messages.json — local sample dataset (used as fallback)

Dockerfile, requirements.txt

Example queries
When is Layla planning her trip to London?

How many cars does Vikram Desai have?

What are Amira's favorite restaurants?

