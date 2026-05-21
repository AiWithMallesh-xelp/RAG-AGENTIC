# llm_factory.py
"""Shared Gemini LLM construction for langchain-google-genai."""

from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings


def make_gemini_chat(
    *,
    model: str | None = None,
    temperature: float = 0.0,
    json_mode: bool = False,
    max_output_tokens: int = 4096,
) -> ChatGoogleGenerativeAI:
    """Build ChatGoogleGenerativeAI with optional JSON response mode."""
    kwargs = {
        "model": model or settings.llm_model,
        "google_api_key": settings.google_api_key,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    return ChatGoogleGenerativeAI(**kwargs)
