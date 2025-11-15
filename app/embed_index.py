
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
    # explicit env override
    envp = os.getenv("MESSAGES_PATH")
    if envp:
        paths.append(Path(envp))
    # repo root (where Docker copies source)
    paths.append(Path("messages.json"))
    # common container app path
    paths.append(Path("/app/messages.json"))
    # relative to this file (two levels up typically project root)
    here = Path(__file__).resolve()
    repo_root = here.parents[2] if len(here.parents) >= 3 else here.parent
    paths.append(repo_root / "messages.json")
    # also try current working dir
    paths.append(Path.cwd() / "messages.json")
    # deduplicate and yield existing ones
    seen = set()
    for p in paths:
        if p is None:
            continue
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
                # try next path
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

    # 3) Nothing found — return empty list (do NOT raise) so the app can start.
    return []

def build_index(save=True):
    """
    Normalizes messages into a simple metadata list and writes METADATA_PATH if requested.
    Returns (None, docs) to preserve previous API shape where first value was an index object.
    """
    msgs = fetch_messages()
    if not isinstance(msgs, (list, tuple)):
        msgs = []

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
        try:
            p = Path(METADATA_PATH)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as f:
                json.dump(docs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # If writing metadata fails, silently continue — indexing can be retried via /reindex
            pass

    return None, docs

def load_index():
    """
    Loads metadata.json if present, otherwise calls build_index(save=True) to create it.
    Always returns (None, docs).
    """
    p = Path(METADATA_PATH)
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                docs = json.load(f)
            if isinstance(docs, list):
                return None, docs
        except Exception:
            pass

    # fallback: try to build from fetch_messages (this will not raise)
    return build_index(save=True)
