
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from app.embed_index import load_index, build_index
from app.config import TOP_K, USE_OPENAI
from app.model_utils import build_prompt, call_openai
import re
from collections import Counter
from typing import List

app = FastAPI(title="Aurora Member QnA (extractor polish)")

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

# regexes
MONTH_DAY_RE = re.compile(r'\b([A-Z][a-z]+)\s+(\d{1,2})(?:,?\s*(\d{4}))?\b')
ISO_RE = re.compile(r'\b(\d{4}-\d{2}-\d{2})\b')
RANGE_RE = re.compile(r'([A-Z][a-z]+ \d{1,2}(?:,? \d{4})?)\s*(?:to|-|–|—)\s*([A-Z][a-z]+ \d{1,2}(?:,? \d{4})?)', re.IGNORECASE)
# expanded car models list (case-insensitive)
CAR_MODEL_RE = re.compile(r'\b(Tesla Model \d+|Tesla|Range Rover|Land Rover|Honda|BMW|Mercedes(?:-Benz)?|Ferrari|Lamborghini|Porsche|Aston Martin)\b', re.IGNORECASE)
# generic candidate capitalized name (we will filter later)
RESTAURANT_CAND_RE = re.compile(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})\b')

# helper: filter restaurant candidates more strictly
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
        # model names via regex
        for m in CAR_MODEL_RE.findall(text):
            nm = m.strip()
            if nm.lower() not in [x.lower() for x in models]:
                models.append(nm)
        # also parse "I still have X and Y" or "I have a X, Y and Z"
        m_have = re.search(r'\bI (?:still )?have (.+?)(?:\.|$)', text, flags=re.IGNORECASE)
        if m_have:
            raw = m_have.group(1)
            # split by and / , / ; then take capitalized pieces
            parts = re.split(r',| and |;| & ', raw)
            for p in parts:
                p = p.strip()
                if len(p) > 2:
                    # prefer proper nouns (capitalized)
                    if re.search(r'[A-Z]', p):
                        if p.lower() not in [x.lower() for x in models]:
                            models.append(p)
    return models

def _extract_favorites_from_docs(top_docs: List[dict]) -> List[str]:
    # explicit pattern first
    for d in top_docs:
        text = d.get("text","") or ""
        m = re.search(r'favorite restaurants?:\s*([A-Za-z0-9 ,\'&\-\u2019]+)', text, flags=re.IGNORECASE)
        if m:
            raw = m.group(1)
            parts = re.split(r',| and | & ', raw)
            parts = [p.strip().strip('.') for p in parts if p.strip()]
            if parts:
                # filter obviously wrong small words
                parts = [p for p in parts if len(p) > 2]
                if parts:
                    return parts
    # fallback: candidate extraction but strict filtering
    candidates = []
    for d in top_docs:
        text = d.get("text","") or ""
        for cand in RESTAURANT_CAND_RE.findall(text):
            cand = cand.strip()
            if len(cand) < 4:
                continue
            low = cand.lower()
            # exclude pronouns and common sentence starts
            if low in ('there','please','thanks','thank','book','reserve','table','i','my','the'):
                continue
            # accept if contains space (multi-word name) OR contains a restaurant keyword
            if ' ' in cand or any(k in low for k in RESTAURANT_KEYWORDS):
                if low not in [c.lower() for c in candidates]:
                    candidates.append(cand)
            # also accept "Chez X" (Chez is in keywords)
    # return up to 6 candidates
    return candidates[:6]

def simple_answer(question: str, top_docs: List[dict]) -> str:
    q = (question or "").lower()

    # restrict to docs for the named member if the query contains one
    name_tokens = _tokenize(question)
    member_filtered = []
    for d in top_docs:
        member = (d.get("member") or d.get("member_name") or "").lower()
        if any(t in member for t in name_tokens if len(t) > 2):
            member_filtered.append(d)
    search_docs = member_filtered if member_filtered else top_docs

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
        # numeric fallback
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

    # default
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
