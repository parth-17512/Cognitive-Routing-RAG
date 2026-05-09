"""
llm_provider.py
───────────────
Returns a ready-to-use LangChain ChatModel based on LLM_MODEL and available
API keys.  Priority order:  Groq → OpenAI → Ollama (local).

Every phase imports get_llm() so the LLM choice stays in one place.
"""

from config import GROQ_API_KEY, OPENAI_API_KEY, OLLAMA_BASE_URL, LLM_MODEL


def get_llm(temperature: float = 0.7, json_mode: bool = False):
    """
    Factory that returns a LangChain chat model.

    Parameters
    ----------
    temperature : float
        Sampling temperature (0 = deterministic, 1 = creative).
    json_mode : bool
        When True and the provider supports it, instruct the model to always
        return valid JSON (used in Phase 2 structured output).

    Returns
    -------
    A LangChain BaseChatModel-compatible instance.
    """

    # ── Groq ──────────────────────────────────────────────────────────────────
    if GROQ_API_KEY:
        from langchain_groq import ChatGroq

        kwargs = dict(
            model=LLM_MODEL,
            temperature=temperature,
            groq_api_key=GROQ_API_KEY,
            max_tokens=1024,
        )
        if json_mode:
            # Groq supports the OpenAI-compatible response_format parameter
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

        print(f"[LLM] Using Groq — model: {LLM_MODEL}")
        return ChatGroq(**kwargs)

    # ── OpenAI ────────────────────────────────────────────────────────────────
    if OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI

        model = LLM_MODEL if "gpt" in LLM_MODEL else "gpt-4o-mini"
        kwargs = dict(
            model=model,
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY,
            max_tokens=1024,
        )
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

        print(f"[LLM] Using OpenAI — model: {model}")
        return ChatOpenAI(**kwargs)

    # ── Ollama (fully local, no API key) ──────────────────────────────────────
    try:
        from langchain_community.chat_models import ChatOllama

        model = LLM_MODEL if LLM_MODEL else "llama3"
        print(f"[LLM] Using Ollama (local) — model: {model}")
        return ChatOllama(
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
        )
    except ImportError:
        pass

    raise EnvironmentError(
        "No LLM provider configured.\n"
        "Set GROQ_API_KEY, OPENAI_API_KEY in your .env file,\n"
        "or install Ollama and set OLLAMA_BASE_URL."
    )
