"""
embeddings.py
─────────────
Thin wrapper that returns a LangChain-compatible embedding object based on the
EMBEDDING_PROVIDER environment variable.  Centralises the choice so all three
phases share the same embedding model.

Supported providers
───────────────────
  "local"  → HuggingFaceEmbeddings (sentence-transformers, no API key needed)
  "openai" → OpenAIEmbeddings (text-embedding-3-small, needs OPENAI_API_KEY)
"""

from config import EMBEDDING_PROVIDER, OPENAI_API_KEY


def get_embeddings():
    """
    Factory that returns the configured LangChain embedding object.

    Returns
    -------
    A LangChain BaseEmbeddings-compatible instance.
    """
    if EMBEDDING_PROVIDER == "openai":
        # ── OpenAI embeddings ──────────────────────────────────────────────
        if not OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Either set the key or switch EMBEDDING_PROVIDER=local."
            )
        from langchain_openai import OpenAIEmbeddings

        print("[Embeddings] Using OpenAI text-embedding-3-small")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY,
        )

    else:
        # ── Local HuggingFace embeddings (default) ─────────────────────────
        # 'all-MiniLM-L6-v2' is small (80 MB), fast on CPU, and semantically
        # rich enough to distinguish the three very different bot personas.
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"[Embeddings] Using local HuggingFace model: {model_name}")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # unit-length → cosine = dot-product
        )
