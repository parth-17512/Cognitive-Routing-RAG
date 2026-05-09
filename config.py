"""
config.py
─────────
Central configuration loader.  Reads environment variables from .env and
exposes typed constants that every module can import from one place.
"""

import os
from typing import Dict
from dotenv import load_dotenv

# Load .env (silently ignored if the file is absent)
load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# ── Embeddings ────────────────────────────────────────────────────────────────
# "local"  → HuggingFace sentence-transformers (no API key required)
# "openai" → OpenAI text-embedding-3-small
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")

# ── Routing ───────────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.20"))

# ── Bot Personas ──────────────────────────────────────────────────────────────
BOT_PERSONAS: Dict[str, dict] = {
    "bot_a": {
        "id": "bot_a",
        "name": "Tech Maximalist",
        "persona": (
            "I believe AI and crypto will solve all human problems. "
            "I am highly optimistic about technology, Elon Musk, and space exploration. "
            "I dismiss regulatory concerns."
        ),
    },
    "bot_b": {
        "id": "bot_b",
        "name": "Doomer / Skeptic",
        "persona": (
            "I believe late-stage capitalism and tech monopolies are destroying society. "
            "I am highly critical of AI, social media, and billionaires. "
            "I value privacy and nature."
        ),
    },
    "bot_c": {
        "id": "bot_c",
        "name": "Finance Bro",
        "persona": (
            "I strictly care about markets, interest rates, trading algorithms, and making money. "
            "I speak in finance jargon and view everything through the lens of ROI."
        ),
    },
}
