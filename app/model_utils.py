
import os
from typing import List, Dict

def build_prompt(question: str, top_docs: List[Dict]) -> str:
    context_lines = []
    for d in top_docs:
        member = d.get("member", "unknown")
        text = d.get("text", "")
        context_lines.append(f"- [{member}] {text}")
    context = "\n".join(context_lines)

    prompt = (
        "You are an assistant that answers questions about members using only the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer briefly using only the context above. If the answer is not present, say you don't know."
    )
    return prompt

def call_openai(prompt: str) -> str:
    try:
        import openai
    except:
        raise RuntimeError("OpenAI SDK not installed. Remove USE_OPENAI or install openai package.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing.")

    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        messages=[
            {"role": "system", "content": "You answer only from context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=200,
    )
    return response["choices"][0]["message"]["content"].strip()
