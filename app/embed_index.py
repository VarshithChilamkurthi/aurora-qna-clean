import requests
import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from app.config import MESSAGES_API, FAISS_INDEX_PATH, METADATA_PATH, EMBED_BATCH

MODEL_NAME = "all-MiniLM-L6-v2"

embedder = SentenceTransformer(MODEL_NAME)

def _unwrap_possible_wrapper(data):
    """
    If data is a dict with a single top-level wrapper like {'items':[...]} or {'results':[...]} or {'data':{'items':[...]}}
    try to return the inner list. Otherwise return data unchanged.
    """
    if isinstance(data, dict):
        # common keys that contain lists
        for key in ("items", "results", "data", "messages"):
            if key in data and isinstance(data[key], (list, tuple)):
                return data[key]
        # some APIs return {'data': {'items': [...]}}
        if "data" in data and isinstance(data["data"], dict):
            for key in ("items", "results", "messages"):
                if key in data["data"] and isinstance(data["data"][key], (list, tuple)):
                    return data["data"][key]
    return data

def fetch_messages():
    """
    Prefer local messages.json if present (for reproducibility / offline dev).
    Otherwise try the API and unwrap common wrappers. If API fails, fallback to local file if present.
    """
    local_path = Path("messages.json")
    if local_path.exists():
        print("Using local messages.json")
        with open(local_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = _unwrap_possible_wrapper(data)
        return data

    try:
        resp = requests.get(MESSAGES_API, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            data = _unwrap_possible_wrapper(data)
            return data
        else:
            print(f"Warning: API returned status {resp.status_code}.")
    except Exception as e:
        print(f"Warning: API fetch failed: {e}.")

    # last-ditch fallback
    if local_path.exists():
        print("Falling back to local messages.json")
        with open(local_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = _unwrap_possible_wrapper(data)
        return data

    raise RuntimeError("Could not fetch messages from API and no local messages.json found.")

def build_index(save=True):
    msgs = fetch_messages()
    if not isinstance(msgs, (list, tuple)):
        raise RuntimeError(f"Expected a list of messages but got {type(msgs)}; please check messages.json or API output.")

    docs = []
    for i, m in enumerate(msgs):
        if isinstance(m, dict):
            text = m.get("text") or m.get("message") or json.dumps(m)
            member = m.get("member_name") or m.get("member") or m.get("author") or "unknown"
            ts = m.get("timestamp") or m.get("date") or ""
            raw = m
        else:
            text = str(m)
            member = "unknown"
            ts = ""
            raw = m
        docs.append({"id": str(i), "member": member, "text": text, "timestamp": ts, "raw": raw})

    texts = [d["text"] for d in docs]
    if len(texts) == 0:
        index = faiss.IndexFlatL2(1)
        if save:
            faiss.write_index(index, FAISS_INDEX_PATH)
            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(docs, f, ensure_ascii=False, indent=2)
        return index, docs

    embeddings = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i+EMBED_BATCH]
        emb = embedder.encode(batch, show_progress_bar=True, convert_to_numpy=True)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    if save:
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)

    return index, docs

def load_index():
    if not Path(FAISS_INDEX_PATH).exists() or not Path(METADATA_PATH).exists():
        return build_index(save=True)
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return index, docs
