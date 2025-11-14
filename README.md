# Aurora Member Q&A

A lightweight RAG-style (Retrieval-Augmented Generation) question-answering API built for the Aurora take-home assignment.  
It answers natural-language questions about member messages by performing semantic search over embeddings and returning concise responses.

The system works both **offline** (using a local `messages.json`) and **online** (when the public API is accessible).

---

## 🚀 Features

- Semantic retrieval using SentenceTransformers (`all-MiniLM-L6-v2`) + FAISS  
- `/ask` endpoint to answer natural language questions  
- Rule-based fallback answers (no OpenAI key required)  
- Optional LLM-based answers using OpenAI if `OPENAI_API_KEY` is set  
- `/reindex` endpoint to rebuild the vector index  
- Local fallback dataset `messages.json` for reliability  
- Dockerized for easy deployment and reproducibility  

---

## 📦 Project Structure
app/
├─ main.py
├─ embed_index.py
├─ model_utils.py
├─ config.py
messages.json
requirements.txt
Dockerfile
README.md


---

## 🧪 Running Locally

### 1. Create & activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### 2. Build the index
