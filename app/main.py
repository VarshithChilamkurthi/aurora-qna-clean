
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from app.embed_index import load_index, build_index
from app.config import TOP_K, USE_OPENAI
from app.model_utils import build_prompt, call_openai
import re
from collections import Counter
from typing import Optional

app = FastAPI(title="Aurora Member QnA (debuggable)")

index, docs = load_index()

class AnswerResponse(BaseModel):
    answer: str

def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def _member_tokens(member_name):
    if not member_name:
        return []
    return [t for t in re.findall(r"[a-z0-9]+", str(member_name).lower())]

def semantic_search(keyword_query: str, k=TOP_K):
    q = keyword_query or ""
    q_lower = q.lower()
    q_tokens = _tokenize(q)
    q_counts = Counter(q_tokens)

    scores = []
    for i, d in enumerate(docs):
        text = d.get("text", "") or ""
        member = (d.get("member") or d.get("member_name") or "").strip()
        member_lower = member.lower()

        tokens = _tokenize(text)

        score = sum(q_counts[t] * tokens.count(t) for t in q_counts)

        member_toks = _member_tokens(member)
        name_overlap = sum(1 for t in q_tokens if t in member_toks)
        if name_overlap:
            score += 150
        if member and member_lower in q_lower:
            score += 200
        if len(q.strip()) > 3 and q_lower in text.lower():
            score += 120

        ts = d.get("timestamp") or ""
        if isinstance(ts, str) and len(ts) >= 4:
            m = re.search(r'(\d{4})', ts)
            if m:
                try:
                    year = int(m.group(1))
                    score += max(0, year - 2000)
                except:
                    pass

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
    q = (question or "").lower()
    all_text = " \n ".join(d.get("text","") for d in top_docs).strip()

    if any(word in q for word in ["when", "trip", "travel", "planning"]):
        m = re.search(r'([A-Za-z]+ \d{1,2}(?:,? \d{4})?)', all_text)
        if m:
            return f"{m.group(1)} (interpreted) - found in member messages."
        m2 = re.search(r'(\d{4}-\d{2}-\d{2})', all_text)
        if m2:
            return f"{m2.group(1)} - found in member messages."
        if all_text:
            return "Couldn't find an explicit date. Top matching messages:\n\n" + _top_text(top_docs)
        return "I don't see trip dates in the data."

    if "how many" in q or "how many cars" in q:
        nums = re.findall(r'(\d+)\s+(?:cars|car|vehicles)', all_text, flags=re.IGNORECASE)
        if nums:
            return f"{nums[0]} (inferred from text)."
        car_keywords = ['car','cars','tesla','range rover','honda','bmw','mercedes']
        car_mentions = sum(all_text.lower().count(kw) for kw in car_keywords)
        if car_mentions:
            return f"Mentions of cars detected (approx {car_mentions}). Top matching messages:\n\n" + _top_text(top_docs)
        return "I don't see any clear car-count info in the data."

    if any(word in q for word in ['favorite restaurant','favorite restaurants','restaurants']):
        m = re.search(r"favorite restaurants?:\s*([A-Za-z0-9 ,'\-&]+)", all_text, flags=re.IGNORECASE)
        if m:
            parts = re.split(r',| and | & ', m.group(1))
            parts = [p.strip().strip('.') for p in parts if p.strip()]
            if parts:
                return "Favorites: " + ", ".join(parts)
        if all_text:
            return "Couldn't find explicit favorites. Top matching messages:\n\n" + _top_text(top_docs)
        return "No favorite restaurants found."

    if all_text:
        return "Top matching messages:\n\n" + _top_text(top_docs)
    return "I don't see relevant information."

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

# --- Debug endpoint ---
@app.get("/debug_docs")
def debug_docs(n: Optional[int] = 10, include_raw: Optional[bool] = False):
    """
    Return the first n docs from the currently-loaded metadata (for debug).
    include_raw: if true, include the raw field.
    """
    try:
        nd = int(n or 10)
    except:
        nd = 10
    sample = docs[:nd]
    out = []
    for d in sample:
        o = {"member": d.get("member") or d.get("member_name") or None,
             "text": d.get("text"),
             "timestamp": d.get("timestamp")}
        if include_raw:
            o["raw"] = d.get("raw")
        out.append(o)
    return {"total_docs": len(docs), "sample": out}
