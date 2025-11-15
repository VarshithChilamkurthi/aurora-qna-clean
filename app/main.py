
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from app.embed_index import load_index, build_index
from app.config import TOP_K, USE_OPENAI
from app.model_utils import build_prompt, call_openai
import re
from collections import Counter

app = FastAPI(title="Aurora Member QnA (lightweight retriever)")

# load docs (index unused here)
index, docs = load_index()

class AnswerResponse(BaseModel):
    answer: str

def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def semantic_search(keyword_query: str, k=TOP_K):
    q_tokens = _tokenize(keyword_query)
    q_counts = Counter(q_tokens)
    scores = []
    for i, d in enumerate(docs):
        text = d.get("text", "") or ""
        tokens = _tokenize(text)
        score = sum(q_counts[t] * tokens.count(t) for t in q_counts)
        member = (d.get("member") or d.get("member_name") or "").lower()
        for t in q_tokens:
            if t in member:
                score += 1
        scores.append((score, i))
    scores.sort(key=lambda x: x[0], reverse=True)
    top_idxs = [idx for score, idx in scores[:k] if score > 0]
    if not top_idxs:
        top_idxs = list(range(min(k, len(docs))))
    return [docs[i] for i in top_idxs]

def _top_text(top_docs):
    lines = []
    for d in top_docs:
        member = d.get("member") or d.get("member_name") or "unknown"
        ts = d.get("timestamp","")
        txt = d.get("text","")
        header = f"[{member}] {ts}".strip()
        lines.append(f"{header}\n{txt}")
    return "\n\n".join(lines)

def simple_answer(question: str, top_docs: list) -> str:
    q = question.lower()
    all_text = " \n ".join(d.get("text","") for d in top_docs).strip()

    # Trip / date heuristics
    if any(word in q for word in ["when", "trip", "travel", "planning"]):
        # common month-day (e.g., June 12) or iso date (2025-06-12)
        m = re.search(r'([A-Za-z]+ \d{1,2}(?:,? \d{4})?)', all_text)
        if m:
            return f"{m.group(1)} (interpreted) - found in member messages."
        m2 = re.search(r'(\d{4}-\d{2}-\d{2})', all_text)
        if m2:
            return f"{m2.group(1)} - found in member messages."
        if all_text:
            return "Couldn't find an explicit date. Top matching messages:\n\n" + _top_text(top_docs)
        return "I don't see trip dates in the data."

    # Count heuristics (cars)
    if "how many" in q or "how many cars" in q:
        nums = re.findall(r'(\d+)\s+(?:cars|car|vehicles)', all_text, flags=re.IGNORECASE)
        if nums:
            return f"{nums[0]} (inferred from text)."
        car_keywords = ['car','cars','tesla','range rover','honda','bmw','mercedes']
        car_mentions = sum(all_text.lower().count(kw) for kw in car_keywords)
        if car_mentions:
            return f"Mentions of cars detected (approx {car_mentions}). Top matching messages:\n\n" + _top_text(top_docs)
        return "I don't see any clear car-count info in the data."

    # Restaurants heuristics
    if any(word in q for word in ['favorite restaurant','favorite restaurants','restaurants']):
        # explicit pattern like: "My favorite restaurants: A, B and C"
        m = re.search(r"favorite restaurants?:\s*([A-Za-z0-9 ,'\-&]+)", all_text, flags=re.IGNORECASE)
        if m:
            parts = re.split(r',| and | & ', m.group(1))
            parts = [p.strip().strip('.') for p in parts if p.strip()]
            if parts:
                return "Favorites: " + ", ".join(parts)
        if all_text:
            return "Couldn't find explicit favorites. Top matching messages:\n\n" + _top_text(top_docs)
        return "No favorite restaurants found."

    # Default fallback: return top docs
    if all_text:
        return "Top matching messages:\n\n" + _top_text(top_docs)
    return "I don't see information relevant to your question in the data."

@app.get("/ask", response_model=AnswerResponse)
def ask(q: str = Query(..., description="Natural language question about member data")):
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
