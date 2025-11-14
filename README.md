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
```
app/
├─ main.py
├─ embed_index.py
├─ model_utils.py
├─ config.py
messages.json
requirements.txt
Dockerfile
README.md
```

---

## 🧪 Running Locally

### 1. Create & activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### 2. Build the index

```
python -c "from app.embed_index import build_index; build_index(save=True)"
```

### 3. Start the server
```uvicorn app.main:app --reload --host 0.0.0.0 --port 8000```

### 4. Test example questions
```
curl "http://127.0.0.1:8000/ask?q=When%20is%20Layla%20planning%20her%20trip%20to%20London%3F"

curl "http://127.0.0.1:8000/ask?q=How%20many%20cars%20does%20Vikram%20Desai%20have%3F"

curl 'http://127.0.0.1:8000/ask?q=What%20are%20Amira%27s%20favorite%20restaurants%3F'
```


## 🔧 Rebuilding the Index

Reindex from the API or local file:

``` curl -X POST "http://127.0.0.1:8000/reindex" ```


Or:

``` python -c "from app.embed_index import build_index; build_index(save=True)" ```

## 🤖 Optional: Using OpenAI for Generated Answers

Set your API key:

``` export OPENAI_API_KEY="sk-..." ```

Restart the server and /ask will use the LLM to generate polished answers.
