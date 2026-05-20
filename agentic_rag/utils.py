# utils.py
"""
Shared utilities used across the Agentic RAG system.
"""

import re
import uuid
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def extract_json(raw: str) -> str:
    """
    Strip markdown code fences and extract the first valid JSON block.
    Uses balanced-brace scanning instead of a greedy regex.
    """
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate and candidate[0] in ("{", "["):
            return candidate

    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = raw.find(start_char)
        if start == -1:
            continue

        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(raw)):
            ch = raw[i]

            if escape_next:
                escape_next = False
                continue

            if ch == "\\":
                escape_next = True
                continue

            if ch == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    candidate = raw[start : i + 1].strip()
                    if candidate:
                        return candidate

    return raw.strip()


def parse_llm_json(raw: str) -> dict:
    """
    Parse JSON from LLM output with fence stripping.
    Returns empty dict on failure (caller should retry or fail gracefully).
    """
    import json

    text = extract_json(raw if isinstance(raw, str) else str(raw))
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


NAMESPACE_DOC = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def make_deterministic_id(*parts: str) -> str:
    """Generate a deterministic UUID5 from the given string parts."""
    seed = ":".join(str(p) for p in parts)
    return str(uuid.uuid5(NAMESPACE_DOC, seed))


def compute_file_hash(file_path: str | Path) -> str:
    """Compute SHA-256 hash of a file for deduplication."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def rate_limited_generate(
    model,
    contents,
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> Optional[str]:
    """Call Gemini generate_content with exponential backoff on rate limits."""
    from google.api_core.exceptions import ResourceExhausted

    for attempt in range(max_retries):
        try:
            response = model.generate_content(contents)
            return response.text
        except ResourceExhausted:
            delay = base_delay * (2 ** attempt)
            logger.warning(
                f"Gemini rate limited (attempt {attempt + 1}/{max_retries}), "
                f"retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return None
    logger.error(f"Max retries ({max_retries}) exceeded for Gemini call")
    return None
