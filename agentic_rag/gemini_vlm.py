# gemini_vlm.py
"""Gemini VLM helpers using google.genai (replaces deprecated google.generativeai)."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted

from config import settings

logger = logging.getLogger(__name__)

_client: Optional[genai.Client] = None


def get_genai_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def generate_with_retry(
    prompt: str,
    parts: list[Any],
    *,
    model: Optional[str] = None,
    max_retries: Optional[int] = None,
    base_delay: float = 2.0,
) -> str:
    """
    Call Gemini generate_content with exponential backoff on rate limits.
    `parts` may include PIL images or text; prompt is prepended as text.
    """
    client = get_genai_client()
    model = model or settings.llm_model
    max_retries = max_retries or settings.vlm_rate_limit_retries

    contents: list[Any] = [prompt]
    contents.extend(parts)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            return (response.text or "").strip()
        except ResourceExhausted:
            delay = base_delay * (2**attempt)
            logger.warning(
                f"Gemini rate limited (attempt {attempt + 1}/{max_retries}), "
                f"retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Gemini VLM error: {e}")
            return ""
    logger.error(f"Max retries ({max_retries}) exceeded for Gemini VLM call")
    return ""
