
import os
import json
import requests
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

def _possible_message_paths():
    # Candidate locations to look for messages.json
    paths = []
    envp = os.getenv("MESSAGES_PATH")
    if envp:
        paths.append(Path(envp))
    paths.append(Path("messages.json"))
    paths.append(Path("/app/messages.json"))
    here = Path(__file__).resolve()
    repo_root = here.parents[2] if len(here.parents) >= 3 else here.parent
    paths.append(repo_root / "messages.json")
    paths.append(Path.cwd() / "messages.json")
    seen = set()
    for p in paths:
        try:
            pstr = str(p.resolve())
        except Exception:
            pstr = str(p)
        if pstr in seen:
            continue
        seen.add(pstr)
        yield p

def _load_local_messages():
    for p in _possible_message_paths():
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return _unwrap_possible_wrapper(data)
            except Exception:
                continue
    return None

def fetch_messages():
    # 1) Try local files
    local = _load_local_messages()
    if local is not None:
        return local

    # 2) Try remote API (may be blocked - tolerate errors)
    try:
        resp = requests.get(MESSAGES_API, timeout=10)
        if resp.status_code == 200:
            try:
                data = resp.json()
                return _unwrap_possible_wrapper(data)
            except Exception:
                pass
    except Exception:
        pass

    # 3) Nothing found — return empty list (do NOT raise)
    return []

def _extract_text_from_msg(m):
    # m is a dict or other value
    if not isinstance(m, dict):
        return str(m)
    # common text fields
    for k in ("text", "message", "body", "content", "msg", "message_text"):
        v = m.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # fallback: if 'raw' nested object contains message
    raw = m.get("raw") or m.get("data") or m.get("payload")
    if isinstance(raw, dict):
        for k in ("text","message","body","content"):
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # sometimes message is in 'raw' under 'message' string
        v = raw.get("message") or raw.get("msg")
        if isinstance(v, str) and v.strip():
            return v.strip()
    # last resort, stringify the dict
    return json.dumps(m, ensure_ascii=False)

def _extract_member_from_msg(m):
    if not isinstance(m, dict):
        return "unknown"
    for k in ("member_name","member","author","user_name","user","username","userName","client_name"):
        v = m.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # nested raw object
    raw = m.get("raw") or m.get("data") or m.get("payload")
    if isinstance(raw, dict):
        for k in ("member_name","member","author","user_name","user","username","userName"):
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # some APIs use 'user_name' under top keys of raw
        v = raw.get("user_name") or raw.get("user")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "unknown"

def _extract_timestamp_from_msg(m):
    if not isinstance(m, dict):
        return ""
    for k in ("timestamp","time","created_at","date"):
        v = m.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    raw = m.get("raw") or m.get("data") or m.get("payload")
    if isinstance(raw, dict):
        for k in ("timestamp","time","created_at","date"):
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""

def build_index(save=True):
    msgs = fetch_messages()
    if not isinstance(msgs, (list, tuple)):
        msgs = []

    docs = []
    for i, m in enumerate(msgs):
        text = _extract_text_from_msg(m)
        member = _extract_member_from_msg(m)
        ts = _extract_timestamp_from_msg(m)
        raw = m
        docs.append({"id": str(i), "member": member, "text": text, "timestamp": ts, "raw": raw})

    if save:
        try:
            p = Path(METADATA_PATH)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as f:
                json.dump(docs, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return None, docs

def load_index():
    p = Path(METADATA_PATH)
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                docs = json.load(f)
            if isinstance(docs, list):
                return None, docs
        except Exception:
            pass
    return build_index(save=True)
