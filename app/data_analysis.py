import json
from collections import Counter
from datetime import datetime
from embed_index import fetch_messages

def analyze():
    msgs = fetch_messages()
    print(f"Total messages fetched: {len(msgs)}")
    member_counts = Counter()
    missing_text = 0
    timestamps = []
    duplicates = 0
    seen_texts = set()

    for m in msgs:
        text = m.get("text") or m.get("message") or ""
        member = m.get("member_name") or m.get("member") or m.get("author") or "unknown"
        ts = m.get("timestamp") or m.get("date") or None

        member_counts[member] += 1
        if not text:
            missing_text += 1
        if text in seen_texts:
            duplicates += 1
        else:
            seen_texts.add(text)
        if ts:
            try:
                timestamps.append(datetime.fromisoformat(ts))
            except Exception:
                pass

    print("Top members by message count:", member_counts.most_common(10))
    print("Missing text fields:", missing_text)
    print("Duplicate message bodies:", duplicates)
    if timestamps:
        print("Earliest ts:", min(timestamps))
        print("Latest ts:", max(timestamps))

if __name__ == "__main__":
    analyze()
