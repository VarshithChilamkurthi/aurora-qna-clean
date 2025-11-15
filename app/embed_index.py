import requests
import json
from pathlib import Path
from app.config import MESSAGES_API, METADATA_PATH

def _unwrap_possible_wrapper(data):
    if isinstance(data, dict):
        for key in ("items", "results", "data", "messages"):
            if key in data and isinstance(data[key], (list, tuple)):
                return data[key]
        if "data" in data and isinstance(data["data"], dict):
            for key in ("items", "results", "messages"):
                if key in data["data"] and isinstance(data["data"][key], (list, tuple)):
                    return data["data"][key]
    return data

def fetch_messages():
    local_path = Path("messages.json")
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _unwrap_possible_wrapper(data)

    try:
        resp = requests.get(MESSAGES_API, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return _unwrap_possible_wrapper(data)
    except Exception:
        pass

    raise RuntimeError("Could not fetch messages from API and no local messages.json found.")

def build_index(save=True):
    """
    For compatibility keep a build_index function that writes metadata.json.
    This function now simply normalizes the messages into docs and writes metadata.
    """
    msgs = fetch_messages()
    if not isinstance(msgs, (list, tuple)):
        raise RuntimeError("Expected a list of messages.")
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
    if save:
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
    return None, docs

def load_index():
    # Instead of returning a faiss index, we return None and the docs list.
    metadata_path = Path(METADATA_PATH)
    if not metadata_path.exists():
        build_index(save=True)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return None, docs
