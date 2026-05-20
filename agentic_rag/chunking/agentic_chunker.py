# chunking/agentic_chunker.py
"""
LLM Agentic Chunking with Dynamic Metadata.

Includes source_id (stable document identity) and file_hash
(content version) in all chunk metadata.
"""

import logging
from difflib import SequenceMatcher

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from config import settings
from llm_factory import make_gemini_chat
from utils import parse_llm_json
from chunking.base_chunker import BaseChunker, ChunkedDocument, ChunkRelation

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    "AgenticChunker",
    "ChunkedDocument",
    "ChunkRelation",
    "CHUNKER_SYSTEM_PROMPT",
]

CHUNKER_SYSTEM_PROMPT = """You are an expert document chunking agent. Your task is to read the provided text and extract exactly ONE logical, self-contained semantic chunk.

Rules:
1. Extract the text VERBATIM. Do not summarize, rephrase, or omit any words.
2. The chunk should represent one complete thought, specification, or topic.
3. After extracting the chunk, identify the EXACT last 5-10 words of your extracted chunk.
4. Extract meaningful metadata as key-value pairs. Do NOT use generic keys like "key1". Use specific keys like "pump_model", "material", "valve_type", "pressure_rating", "section_title", "topic", "entity_type".

You MUST respond ONLY with valid JSON in this exact format:
{
  "chunk_text": "the verbatim text you extracted...",
  "last_few_words": "the exact last 5 to 10 words of the chunk_text",
  "metadata": {
    "specific_key_1": "specific_value_1",
    "specific_key_2": "specific_value_2"
  }
}
"""


class AgenticChunker(BaseChunker):
    """
    Two-level agentic chunker:
    - Parent: RecursiveCharacterTextSplitter (large context)
    - Child: LLM-driven iterative semantic extraction (precise)
    """

    def __init__(self):
        super().__init__()
        self.llm = make_gemini_chat(temperature=0.0, json_mode=True)

    def child_content_type(self) -> str:
        return "semantic_text"

    def _log_chunking_complete(
        self, parents: list[Document], children: list[Document]
    ) -> None:
        logger.info(
            f"Chunking complete: {len(parents)} parents, "
            f"{len(children)} children"
        )

    def _chunk_text_part(
        self,
        text: str,
        source_name: str,
        page_num: int,
        parent_id: str,
        file_hash: str,
        source_id: str,
        parent_idx: int,
        text_part_idx: int,
    ) -> list[tuple[Document, dict]]:
        return self._llm_semantic_chunker(
            text=text,
            source_name=source_name,
            page_num=page_num,
            parent_id=parent_id,
            file_hash=file_hash,
            source_id=source_id,
            parent_idx=parent_idx,
            text_part_idx=text_part_idx,
        )

    @staticmethod
    def _find_end_index(remaining: str, last_few_words: str) -> int:
        search_bound = min(
            len(remaining),
            int(len(remaining) * 0.85) + len(last_few_words),
        )

        matcher = SequenceMatcher(
            None, remaining.lower(), last_few_words.lower(), autojunk=False
        )
        match = matcher.find_longest_match(0, search_bound, 0, len(last_few_words))

        if match.size > len(last_few_words) * 0.6:
            return match.a + match.size
        return -1

    def _llm_semantic_chunker(
        self,
        text: str,
        source_name: str,
        page_num: int,
        parent_id: str,
        file_hash: str = "",
        source_id: str = "",
        parent_idx: int = 0,
        text_part_idx: int = 0,
    ) -> list[tuple[Document, dict]]:
        chunks: list[tuple[Document, dict]] = []
        remaining_text = text.strip()
        chunk_index = 0
        max_iterations = settings.child_max_iterations

        while (
            remaining_text
            and len(remaining_text.strip()) > settings.child_min_text_length
            and chunk_index < max_iterations
        ):
            before_len = len(remaining_text)

            text_for_llm = remaining_text
            if len(text_for_llm) > settings.chunker_max_input_chars:
                text_for_llm = text_for_llm[: settings.chunker_max_input_chars]

            message = HumanMessage(
                content=f"Text to process:\n\n{text_for_llm}"
            )

            data = {}
            for attempt in range(settings.chunker_json_retries + 1):
                try:
                    response = self.llm.invoke(
                        [SystemMessage(content=CHUNKER_SYSTEM_PROMPT), message]
                    )
                    data = parse_llm_json(response.content)
                    if data.get("chunk_text") and data.get("last_few_words"):
                        break
                    logger.warning(
                        f"Chunker empty/invalid JSON (attempt {attempt + 1})"
                    )
                except Exception as e:
                    logger.warning(
                        f"Chunker LLM error (attempt {attempt + 1}): {e}"
                    )
                data = {}

            try:
                chunk_text = (data.get("chunk_text") or "").strip()
                last_few_words = (data.get("last_few_words") or "").strip()
                dynamic_metadata = data.get("metadata") or {}

                if not chunk_text or not last_few_words:
                    logger.warning(
                        "LLM chunker failed after retries; using overflow split."
                    )
                    break

                end_idx = self._find_end_index(remaining_text, last_few_words)

                if end_idx != -1:
                    extracted = remaining_text[:end_idx].strip()
                    remaining_text = remaining_text[end_idx:].strip()
                else:
                    prefix = chunk_text[:30]
                    pos = remaining_text.find(prefix)
                    if pos != -1:
                        end_pos = pos + len(chunk_text)
                        extracted = remaining_text[pos:end_pos].strip()
                        remaining_text = remaining_text[end_pos:].strip()
                    else:
                        logger.warning(
                            "Chunker cannot locate boundary. Breaking."
                        )
                        break

                if len(remaining_text) >= before_len:
                    logger.warning(
                        "Chunker made no progress; stopping to avoid duplication."
                    )
                    break

                meta = {
                    "source_file": source_name,
                    "page": page_num,
                    "chunk_index": chunk_index,
                    "parent_id": parent_id,
                }

                doc = Document(page_content=extracted, metadata=meta)
                chunks.append((doc, dynamic_metadata))
                chunk_index += 1

            except Exception as e:
                logger.error(f"LLM Chunker error: {e}. Breaking.")
                break

        if remaining_text and len(remaining_text.strip()) > 20:
            meta = {
                "source_file": source_name,
                "page": page_num,
                "chunk_index": chunk_index,
                "parent_id": parent_id,
                "content_type": "text_overflow",
            }
            doc = Document(page_content=remaining_text.strip(), metadata=meta)
            chunks.append(
                (doc, {"topic": "overflow", "content_type": "overflow"})
            )

        return chunks
