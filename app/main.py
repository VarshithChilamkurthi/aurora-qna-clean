from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from app.embed_index import load_index, build_index, embedder
from app.config import TOP_K, USE_OPENAI
from app.model_utils import build_prompt, call_openai
import re
from dateutil import parser as dateparser

app = FastAPI(title="Aurora Member QnA")

index, docs = load_index()

class AnswerResponse(BaseModel):
    answer: str

def semantic_search(question: str, k=TOP_K):
    q_emb = embedder.encode([question], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, k)
    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(docs): continue
        results.append(docs[idx])
    return results

# Simple rule-based fallbacks to produce concise answers without OpenAI
def simple_answer(question: str, top_docs: list) -> str:
    q = question.lower()

    # Helper: join all texts
    all_text = " \n ".join(d.get("text","") for d in top_docs).strip()

    # 1) Trip date question (when is X planning a trip to Y)
    if any(word in q for word in ["when", "trip", "travel", "planning"]):
        # Try to find date-like substrings in the context
        dates = []
        for d in top_docs:
            # search for ISO-like or month name patterns
            txt = d.get("text","")
            # common formats: June 12, 2025 ; Jun 12 ; 2025-06-12
            for m in re.findall(r'([A-Za-z]+ \d{1,2}(?:,? \d{4})?)', txt):
                try:
                    parsed = dateparser.parse(m, fuzzy=True)
                    dates.append((m, parsed.date().isoformat()))
                except Exception:
                    pass
            for m in re.findall(r'(\d{4}-\d{2}-\d{2})', txt):
                try:
                    parsed = dateparser.parse(m)
                    dates.append((m, parsed.date().isoformat()))
                except Exception:
                    pass
        if dates:
            # return the first reasonable match
            m, iso = dates[0]
            return f"{m} (interpreted as {iso}) — found in member messages."
        # fallback: show context
        if all_text:
            return "I couldn't find an explicit date, but here are the top matching messages:\n\n" + all_text
        return "I don't see trip dates in the data."

    # 2) Count questions (how many cars / how many X)
    if q.strip().startswith("how many") or "how many" in q:
        # look for numeric mentions in context, or common car keywords
        # count explicit car mentions (simple heuristic)
        car_keywords = ['car', 'cars', 'tesla', 'range rover', 'honda', 'bmw', 'mercedes']
        car_mentions = 0
        for d in top_docs:
            txt = d.get("text","").lower()
            # detect sentences with 'have' or 'own' + car keyword
            for kw in car_keywords:
                if kw in txt:
                    car_mentions += txt.count(kw)
        # also try to extract explicit phrases "I have X" -> numbers
        nums = re.findall(r'(\d+)\s+(?:cars|car|vehicles)', all_text, flags=re.IGNORECASE)
        if nums:
            return f"{nums[0]} (inferred from text)."
        if car_mentions > 0:
            # heuristically map mentions to a count (may overcount but ok for demo)
            return f"Mentions of cars found in context (approx {car_mentions} mentions). See messages for details:\n\n" + all_text
        return "I don't see any clear car-count info in the data."

    # 3) Favorite restaurants
    if any(word in q for word in ["favorite restaurant", "favorite restaurants", "restaurants"]):
        # Look for list-like patterns or "favorite" mentions
        favorites = []
        # common pattern: "My favorite restaurants: A, B, and C"
        m = re.search(r'favorite restaurants?:\s*([A-Za-z0-9 ,\'\-&]+)', all_text, flags=re.IGNORECASE)
        if m:
            txt = m.group(1)
            # split by comma/and
            parts = re.split(r',| and | & ', txt)
            parts = [p.strip().strip('.') for p in parts if p.strip()]
            if parts:
                return "Favorites: " + ", ".join(parts)
        # fallback: search for proper nouns that look like restaurants (capitalized words)
        candidates = re.findall(r'([A-Z][a-z]+(?: [A-Z][a-z]+)*)', all_text)
        # filter short common words
        candidates = [c for c in candidates if len(c) > 3]
        if candidates:
            # unique
            seen = []
            for c in candidates:
                if c.lower() not in [s.lower() for s in seen]:
                    seen.append(c)
            return "Restaurants mentioned: " + ", ".join(seen[:6])
        return "No favorite restaurants found in the data."

    # Default fallback: show top context
    if all_text:
        return "Top matching messages:\n\n" + all_text
    return "I don't see information relevant to your question in the data."

@app.get("/ask", response_model=AnswerResponse)
def ask(q: str = Query(..., description="Natural language question about member data")):
    try:
        top_docs = semantic_search(q)
        if USE_OPENAI:
            # build prompt and call OpenAI
            prompt = build_prompt(q, top_docs)
            answer = call_openai(prompt)
            return {"answer": answer}
        else:
            # use the simple rule-based fallback to produce a concise answer
            answer = simple_answer(q, top_docs)
            return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindex")
def reindex():
    global index, docs
    index, docs = build_index(save=True)
    return {"status": "ok", "total_docs": len(docs)}
