
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from app.embed_index import load_index, build_index
from app.config import TOP_K, USE_OPENAI
from app.model_utils import build_prompt, call_openai
import re
from collections import Counter
from typing import List

app = FastAPI(title="Aurora Member QnA (member-lookup fix)")

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
    """
    Lightweight retriever with member-direct-lookup:
    - If query contains a name token that matches a member, return all docs for that member (newest-first).
    - Otherwise use token-overlap + boosts as before.
    """
    q = (keyword_query or "").strip()
    q_lower = q.lower()
    q_tokens = _tokenize(q)
    q_counts = Counter(q_tokens)

    # 1) attempt direct member lookup: if any member name contains a token from query, return those docs
    member_matches = {}
    for i, d in enumerate(docs):
        member = (d.get("member") or d.get("member_name") or "").strip()
        if not member:
            continue
        m_lower = member.lower()
        # if any q token (len>2) is in the member string, count it
        hits = sum(1 for t in q_tokens if len(t) > 2 and t in m_lower)
        if hits:
            if member not in member_matches:
                member_matches[member] = []
            member_matches[member].append((i, d))

    # if we found at least one matching member, prioritize the best-match member (most token hits)
    if member_matches:
        # choose member with most matched docs (or you could weigh hits)
        best_member = max(member_matches.keys(), key=lambda m: len(member_matches[m]))
        # return that member's docs sorted by timestamp descending if possible
        docs_for_member = [d for i,d in sorted(member_matches[best_member], key=lambda x: x[1].get('timestamp',''), reverse=True)]
        # limit to k
        return docs_for_member[:k]

    # 2) fallback scoring (token overlap + boosts)
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

# Keep the improved per-doc extractors from previous version (dates, cars, restaurants)
MONTH_DAY_RE = re.compile(r'\b([A-Z][a-z]+)\s+(\d{1,2})(?:,?\s*(\d{4}))?\b')
ISO_RE = re.compile(r'\b(\d{4}-\d{2}-\d{2})\b')
RANGE_RE = re.compile(r'([A-Z][a-z]+ \d{1,2}(?:,? \d{4})?)\s*(?:to|-|–|—)\s*([A-Z][a-z]+ \d{1,2}(?:,? \d{4})?)', re.IGNORECASE)
CAR_MODEL_RE = re.compile(r'\b(Tesla Model \d+|Tesla|Range Rover|Land Rover|Honda|BMW|Mercedes(?:-Benz)?|Ferrari|Lamborghini|Porsche|Aston Martin)\b', re.IGNORECASE)
RESTAURANT_CAND_RE = re.compile(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})\b')
RESTAURANT_KEYWORDS = set(k.lower() for k in [
    "restaurant","lounge","chez","cafe","bistro","marina","house","grill","inn",
    "bar","trattoria","osteria","kitchen","dining","pub","tavern","club","deli"
])

def _extract_dates_from_docs(top_docs: List[dict]) -> List[str]:
    found = []
    for d in top_docs:
        text = d.get("text","") or ""
        m_range = RANGE_RE.search(text)
        if m_range:
            start, end = m_range.group(1).strip(), m_range.group(2).strip()
            found.append(f"{start} to {end}")
            continue
        m_iso = ISO_RE.search(text)
        if m_iso:
            found.append(m_iso.group(1))
            continue
        m_md = MONTH_DAY_RE.search(text)
        if m_md:
            month, day, year = m_md.group(1), m_md.group(2), m_md.group(3)
            if year:
                found.append(f"{month} {day}, {year}")
            else:
                found.append(f"{month} {day}")
    return found

def _extract_car_models_from_docs(top_docs: List[dict]) -> List[str]:
    models = []
    for d in top_docs:
        text = d.get("text","") or ""
        for m in CAR_MODEL_RE.findall(text):
            nm = m.strip()
            if nm.lower() not in [x.lower() for x in models]:
                models.append(nm)
        m_have = re.search(r'\bI (?:still )?have (.+?)(?:\.|$)', text, flags=re.IGNORECASE)
        if m_have:
            raw = m_have.group(1)
            parts = re.split(r',| and |;| & ', raw)
            for p in parts:
                p = p.strip()
                if len(p) > 2 and re.search(r'[A-Z]', p):
                    if p.lower() not in [x.lower() for x in models]:
                        models.append(p)
    return models

def _extract_favorites_from_docs(top_docs: List[dict]) -> List[str]:
    for d in top_docs:
        text = d.get("text","") or ""
        m = re.search(r'favorite restaurants?:\s*([A-Za-z0-9 ,\'&\-\u2019]+)', text, flags=re.IGNORECASE)
        if m:
            raw = m.group(1)
            parts = re.split(r',| and | & ', raw)
            parts = [p.strip().strip('.') for p in parts if p.strip() and len(p.strip())>2]
            if parts:
                return parts
    candidates = []
    for d in top_docs:
        text = d.get("text","") or ""
        for cand in RESTAURANT_CAND_RE.findall(text):
            cand = cand.strip()
            if len(cand) < 4:
                continue
            low = cand.lower()
            if low in ('there','please','thanks','thank','book','reserve','table','i','my','the'):
                continue
            if ' ' in cand or any(k in low for k in RESTAURANT_KEYWORDS):
                if low not in [c.lower() for c in candidates]:
                    candidates.append(cand)
    return candidates[:6]

def simple_answer(question: str, top_docs: List[dict]) -> str:
    q = (question or "").lower()

    # If a member is mentioned, try direct member documents first (semantic_search already returns them)
    search_docs = top_docs

    # Dates
    if any(word in q for word in ["when", "trip", "travel", "planning"]):
        dates = _extract_dates_from_docs(search_docs)
        if dates:
            for dt in dates:
                if " to " in dt:
                    return f"{dt} — found in member messages."
            return f"{dates[0]} — found in member messages."
        if search_docs:
            return "Couldn't find an explicit date. Top matching messages:\n\n" + _top_text(search_docs)
        return "I don't see trip dates in the data."

    # Cars
    if "how many" in q and "car" in q:
        models = _extract_car_models_from_docs(search_docs)
        if models:
            return f"{len(models)} (models detected: {', '.join(models)})"
        all_text = " ".join(d.get("text","") for d in search_docs)
        nums = re.findall(r'(\d+)\s+(?:cars|car|vehicles)', all_text, flags=re.IGNORECASE)
        if nums:
            return f"{nums[0]} (inferred from text)."
        if search_docs:
            return "No explicit car count found. Top matching messages:\n\n" + _top_text(search_docs)
        return "No car info in data."

    # Restaurants
    if any(word in q for word in ['favorite restaurant','favorite restaurants','restaurants']):
        favs = _extract_favorites_from_docs(search_docs)
        if favs:
            return "Favorites: " + ", ".join(favs)
        if search_docs:
            return "Couldn't find explicit favorites. Top matching messages:\n\n" + _top_text(search_docs)
        return "No favorite restaurants found."

    # Default
    if top_docs:
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
