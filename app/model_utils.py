import os
import openai
from app.config import OPENAI_API_KEY, OPENAI_MODEL, TOP_K

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

def build_prompt(question: str, top_docs: list):
    ctx = "\n\n---\n\n".join([f"Member: {d['member']}\nTimestamp: {d.get('timestamp','')}\nMessage: {d['text']}" for d in top_docs])
    prompt = f"""
You are an assistant that answers questions about member data. Use only the information in the provided context. If the answer is not in the context, say "I don't see that information in the data."

Context:
{ctx}

Question: {question}

Provide a concise answer (one or two sentences). If multiple members are referenced, be explicit about which member. If asked for a count, return just the number and a short note on how you inferred it.
"""
    return prompt.strip()

def call_openai(prompt: str, model: str = OPENAI_MODEL, max_tokens=256):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not configured.")
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp['choices'][0]['message']['content'].strip()
