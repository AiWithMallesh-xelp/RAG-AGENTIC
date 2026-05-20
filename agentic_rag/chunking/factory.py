# chunking/factory.py
"""Select chunking strategy from settings."""

from config import settings
from chunking.agentic_chunker import AgenticChunker
from chunking.base_chunker import BaseChunker
from chunking.maxmin_chunker import MaxMinChunker


def create_chunker() -> BaseChunker:
    if settings.use_maxmin_chunking:
        return MaxMinChunker()
    return AgenticChunker()


def chunking_mode_name() -> str:
    return "maxmin" if settings.use_maxmin_chunking else "agentic_llm"
