import os
import json
from typing import List, Dict

def build_prompt(question: str, top_docs: List[Dict]) -> str:
    """
    Build a simple RAG-style prompt consisting of the question and top context.
    """
    context = "\n\n".join([f\"- [{d.get('member','unknown')}] {d.get('text','')}\" for d in top_docs])
    prompt = (
        "You are an assistant that answers questions about members using only the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer briefly and only using the context above. If the answer is not present, say you don't know."
    )
    return prompt

def call_openai(prompt: str) -> str:
    """
    Call OpenAI ChatCompletion in a lazy way (import inside function).
    If openai package or API key is missing, raise a clear error.
    """
    try:
        import openai
    except Exception as e:
        raise RuntimeError("OpenAI SDK is not installed in this environment. Install the 'openai' package or run without OpenAI enabled.") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment. Set the key to enable LLM answers.")

    openai.api_key = api_key

    # Use a concise ChatCompletion request
    try:
        response = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions from provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            temperature=0.0,
            n=1,
        )
        # extract text
        content = response["choices"][0]["message"]["content"].strip()
        return content
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}") from e
