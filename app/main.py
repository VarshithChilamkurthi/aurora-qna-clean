from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from app.embed_index import load_index, build_index
from app.config import TOP_K, USE_OPENAI
from app.model_utils import build_prompt, call_openai
import re
from collections import Counter

app = FastAPI(title="Aurora Member QnA (lightweight retriever)")

# load docs (index is None for the lightweight flow)
index, docs = load_index()

class AnswerResponse(BaseModel):
    answer: str

def _tokenize(text):
    # lowercase words, remove non-alphanum, split
    return re.findall(r"[a-z0-9]+", text.lower())

def semantic_search(keyword_query: str, k=TOP_K):
    """
    Simple lightweight retrieval: score documents by token overlap with query.
    Returns top-k docs (as list of doc dicts).
    """
    q_tokens = _tokenize(keyword_query)
    q_counts = Counter(q_tokens)
    scores = []
    for i, d in enumerate(docs):
        text = d.get("text", "") or ""
        tokens = _tokenize(text)
        if not tokens:
            scores.append((0, i))
            continue
        # score = sum of query token frequencies in doc tokens
        score = sum(q_counts[t] * tokens.count(t) for t in q_counts)
        # small bonus for member name match
        member = (d.get("member") or "").lower()
        for t in q_tokens:
            if t in member:
                score += 1
        scores.append((score, i))
    # pick top k by score
    scores.sort(key=lambda x: x[0], reverse=True)
    top_idxs = [idx for score, idx in scores[:k] if score > 0]
    # if no positive scores, fall back to first k docs
    if not top_idxs:
        top_idxs = list(range(min(k, len(docs))))
    return [docs[i] for i in top_idxs]

# simple fallback logic kept from previous version
def simple_answer(question: str, top_docs: list) -> str:
    q = question.lower()
    all_text = " \n ".join(d.get("text","") for d in top_docs).strip()

    # trip date heuristic
    if any(word in q for word in ["when", "trip", "travel", "planning"]):
        # look for month-day patterns or YYYY-MM-DD
        m = re.search(r'([A-Za-z]+ \d{1,2}(?:,? \d{4})?)', all_text)
        if m:
            return f\"{m.group(1)} (interpreted) — found in member messages.\"
        m2 = re.search(r'(\d{4}-\d{2}-\d{2})', all_text)
        if m2:
            return f\"{m2.group(1)} — found in member messages.\"
        if all_text:
            return "I couldn't find an explicit date, but here are the top matching messages:\\n\\n" + all_text
        return "I don't see trip dates in the data."

    # count heuristics
    if q.strip().startswith("how many") or "how many" in q:
        nums = re.findall(r'(\\d+)\\s+(?:cars|car|vehicles)', all_text, flags=re.IGNORECASE)
        if nums:
            return f\"{nums[0]} (inferred from text).\"
        car_keywords = ['car','cars','tesla','range rover','honda','bmw','mercedes']
        car_mentions = sum(all_text.lower().count(kw) for kw in car_keywords)
        if car_mentions:
            return f\"Mentions of cars found in context (approx {car_mentions} mentions). See messages for details:\\n\\n\" + all_text
        return \"I don't see any clear car-count info in the data.\"

    # restaurants
    if any(word in q for word in ['favorite restaurant','favorite restaurants','restaurants']):
        m = re.search(r'favorite restaurants?:\\s*([A-Za-z0-9 ,\\'\\-&]+)', all_text, flags=re.IGNORECASE)
        if m:
            txt = m.group(1)
            parts = re.split(r',| and | & ', txt)
            parts = [p.strip().strip('.') for p in parts if p.strip()]
            if parts:
                return \"Favorites: \" + \", \".join(parts)
        # fallback: capitalized multiword candidates
        candidates = re.findall(r'([A-Z][a-z]+(?: [A-Z][a-z]+)*)', all_text)
        candidates = [c for c in candidates if len(c) > 3]
        if candidates:
            seen = []
            for c in candidates:
                if c.lower() not in [s.lower() for s in seen]:
                    seen.append(c)
            return \"Restaurants mentioned: \" + \", \".join(seen[:6])
        return \"No favorite restaurants found in the data.\"

    # default
    if all_text:
        return \"Top matching messages:\\n\\n\" + all_text
    return \"I don't see information relevant to your question in the data.\"

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
