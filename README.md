Aurora Member Q&A

A lightweight RAG-style (Retrieval-Augmented Generation) question-answering API built for the Aurora take-home assignment.
It answers natural-language questions about member messages by performing semantic search over embeddings and returning concise responses.

The system works both offline (using a local messages.json) and online (when the public API is accessible).

🚀 Features

Semantic retrieval using SentenceTransformers (all-MiniLM-L6-v2) + FAISS.

/ask endpoint to answer natural language questions.

Rule-based fallback answers (no OpenAI key required).

Optional LLM-based answers using OpenAI if OPENAI_API_KEY is provided.

/reindex endpoint to rebuild the vector index.

messages.json fallback ensures the project runs even if the external API is down (the provided API returned HTTP 403 during development).

Dockerized for easy deployment (Render / Cloud Run).

📦 Project Structure
app/
  ├─ main.py                  # FastAPI service
  ├─ embed_index.py           # Builds & loads FAISS index
  ├─ model_utils.py           # Prompt construction + OpenAI call
  ├─ config.py                # Environment configs
messages.json                 # Local dataset fallback
requirements.txt
Dockerfile
README.md

🧪 Running Locally
1. Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. Build the index (uses messages.json automatically)
python -c "from app.embed_index import build_index; build_index(save=True)"

3. Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

4. Test example questions
curl "http://127.0.0.1:8000/ask?q=When%20is%20Layla%20planning%20her%20trip%20to%20London%3F"

curl "http://127.0.0.1:8000/ask?q=How%20many%20cars%20does%20Vikram%20Desai%20have%3F"

curl "http://127.0.0.1:8000/ask?q=What%20are%20Amira%27s%20favorite%20restaurants%3F"

🔧 Rebuilding the Index

Either:

curl -X POST "http://127.0.0.1:8000/reindex"


Or directly:

python -c "from app.embed_index import build_index; build_index(save=True)"

🤖 Using OpenAI (optional)

If you want polished, natural answers instead of rule-based responses:

export OPENAI_API_KEY="sk-..."


Restart the server and /ask will use the LLM for final reasoning and answer generation.

📡 Deployment Notes

This project includes a Dockerfile so deployment is simple.

Deploying on Render (recommended)

Go to Render → New → Web Service

Choose your GitHub repo.

Environment: Docker

Add environment variables (optional):

OPENAI_API_KEY=...

MESSAGES_API=https://november7-730026606190.europe-west1.run.app/messages

Deploy and test:

curl "https://<your-render-url>/ask?q=When%20is%20Layla%20planning%20her%20trip%20to%20London%3F"

📁 Data Source Note (Very Important)

The provided public messages API returned HTTP 403 Forbidden throughout development.
To ensure the application remains fully functional, the repository includes a small sample dataset messages.json.

The system will:

Try the API

If API fails → use messages.json (guaranteed working demo)

/reindex will rebuild based on whichever source is available

This ensures reliable, reproducible behavior for evaluators.

🧠 Design Notes (High Level)

Embedding model: all-MiniLM-L6-v2 for speed + accuracy.

Vector store: FAISS (simple, fast, fully local).

Reasoning:

Without OpenAI → rule-based extraction for dates, restaurants, and counts.

With OpenAI → RAG prompt + LLM → natural answers.

Alternatives considered:

Managed vector DB (Pinecone, Weaviate, PGVector)

Local LLM inference (Mistral, Llama-3)

Extractive QA models (e.g., RoBERTa SQuAD)

Full agentic orchestration (out of scope for this take-home)

📝 Example Output
{
  "answer": "June 12 (interpreted as 2025-06-12) — found in member messages."
}

📎 Extras

A small examples.sh script is included to test the three example questions easily.
