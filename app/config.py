import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional (for LLM answers)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change as needed
MESSAGES_API = os.getenv("MESSAGES_API", "https://november7-730026606190.europe-west1.run.app/messages")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.idx")
METADATA_PATH = os.getenv("METADATA_PATH", "metadata.json")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))
TOP_K = int(os.getenv("TOP_K", "5"))
USE_OPENAI = bool(OPENAI_API_KEY)
