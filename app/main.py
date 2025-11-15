
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from app.embed_index import load_index, build_index
from app.config import TOP_K, USE_OPENAI
from app.model_utils import build_prompt, call_openai
import re
from collections import Counter

app = FastAPI(title="Aurora Member QnA (light retriever)")

index, docs = load_index()

class AnswerResponse(BaseModel):
    answer: str

def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def semantic_search(query: str, k=TOP_K):
    q_tokens = _tokenize(query)
    q_counts = Counter(q_tokens)
    scores = []

    for i, d in enumerate(docs):
        text = d.get("text", "")
        tokens = _tokenize(text)
        score = sum(q_counts[t] * tokens.count(t) for t in q_counts)

        member = (d.get("member") or "").lower()
        for t in q_tokens:
            if t in member:
                score += 1

        scores.append((score, i))

    scores.sort(key=lambda x: x[0], reverse=True)
    top = [idx for score, idx in scores[:k] if score > 0]
    if not top:
        top = list(range(min(k, len(docs))))

    return [docs[i] for i in top]

def simple_answer(question: str, top_docs):
    q = question.lower()
    all_text = " \n ".join([d.get("text", "") for d in top_docs]).strip()

    if any(w in q for w in ["when", "trip", "travel", "planning"]):
        m = re.search(r"([A-Za-z]+ \d{1,2}(?:,? \d{4})?)", all_text)
        if m:
            return f"{m.group(1)} - found in messages."
        m2 = re.search(r"(\d{4}-\d{2}-\d{2})", all_text)
        if m2:
            return f"{m2.group(1)} - found in messages."
        return "No explicit trip date found."

    if "how many" in q:
        nums = re.findall(r"(\d+)\s+(?:car|cars|vehicles)", all_text, flags=re.IGNORECASE)
        if nums:
            return f"{nums[0]} (inferred)."
        return "No clear car count found."

    if "restaurant" in q:
        m = re.search(r"favorite restaurants?:\s*([A-Za-z0-9 ,'\-&]+)", all_text, flags=re.IGNORECASE)
        if m:
            parts = [p.strip().strip('.') for p in re.split(r",| and | & ", m.group(1)) if p.strip()]
            return "Favorites: " + ", ".join(parts)
        return "No favorite restaurants found."

    return "Top matching messages:\n\n" + all_text

@app.get("/ask", response_model=AnswerResponse)
def ask(q: str):
    try:
        top_docs = semantic_search(q)
        if USE_OPENAI:
            prompt = build_prompt(q, top_docs)
            answer = call_openai(prompt)
        else:
            answer = simple_answer(q, top_docs)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindex")
def reindex():
    global docs
    _, docs = build_index(save=True)
    return {"status": "ok", "total_docs": len(docs)}
