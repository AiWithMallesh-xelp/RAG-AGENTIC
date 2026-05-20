# chunking/maxmin_chunker.py
"""
Max-Min semantic chunking (Kiss et al. 2025).

Embeds sentences first, then groups boundaries by cosine similarity.
No LLM generation during ingest.
"""

import logging

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

from config import settings
from chunking.base_chunker import BaseChunker
from chunking.embedding_client import embed_texts_batched

logger = logging.getLogger(__name__)

_punkt_ready = False


def _ensure_nltk_punkt() -> None:
    global _punkt_ready
    if _punkt_ready:
        return
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    _punkt_ready = True


def split_sentences(text: str) -> list[str]:
    """Split text into sentences; fall back to line-based split if needed."""
    _ensure_nltk_punkt()
    import nltk

    text = text.strip()
    if not text:
        return []

    sentences = nltk.sent_tokenize(text)
    cleaned: list[str] = []
    for s in sentences:
        s = s.strip()
        if s:
            cleaned.append(s)
    if cleaned:
        return cleaned

    return [line.strip() for line in text.split("\n") if line.strip()]


def maxmin_group_sentences(
    sentences: list[str],
    embeddings: np.ndarray,
    *,
    fixed_threshold: float | None = None,
    c: float | None = None,
    init_constant: float | None = None,
) -> list[str]:
    """
    Group consecutive sentences into chunks using the Max-Min algorithm.

    Reference: https://github.com/hsdslab/MaxMinChunking
    """
    if not sentences:
        return []
    if len(sentences) == 1:
        return [sentences[0]]

    fixed_threshold = (
        fixed_threshold
        if fixed_threshold is not None
        else settings.maxmin_hard_threshold
    )
    c = c if c is not None else settings.maxmin_c
    init_constant = (
        init_constant
        if init_constant is not None
        else settings.maxmin_init_constant
    )

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    paragraphs: list[list[str]] = []
    current_paragraph = [sentences[0]]
    cluster_start, cluster_end = 0, 1
    pairwise_min = -float("inf")

    for i in range(1, len(sentences)):
        cluster_embeddings = embeddings[cluster_start:cluster_end]

        if cluster_end - cluster_start > 1:
            new_sentence_similarities = cosine_similarity(
                embeddings[i].reshape(1, -1), cluster_embeddings
            )[0]
            adjusted_threshold = pairwise_min * c * sigmoid(
                (cluster_end - cluster_start) - 1
            )
            new_sentence_similarity = float(np.max(new_sentence_similarities))
            pairwise_min = min(
                float(np.min(new_sentence_similarities)), pairwise_min
            )
        else:
            adjusted_threshold = 0.0
            sim = float(
                cosine_similarity(
                    embeddings[i].reshape(1, -1), cluster_embeddings
                )[0][0]
            )
            pairwise_min = sim
            new_sentence_similarity = init_constant * sim

        if new_sentence_similarity > max(adjusted_threshold, fixed_threshold):
            current_paragraph.append(sentences[i])
            cluster_end += 1
        else:
            paragraphs.append(current_paragraph)
            current_paragraph = [sentences[i]]
            cluster_start, cluster_end = i, i + 1
            pairwise_min = -float("inf")

    paragraphs.append(current_paragraph)

    return [" ".join(p) for p in paragraphs if p]


def _split_oversized_chunks(chunks: list[str]) -> list[str]:
    """Split chunks that exceed MAXMIN_MAX_CHUNK_CHARS."""
    max_chars = settings.maxmin_max_chunk_chars
    if max_chars <= 0:
        return chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            result.append(chunk)
        else:
            result.extend(splitter.split_text(chunk))
    return result


class MaxMinChunker(BaseChunker):
    """Child chunks via Max-Min semantic clustering on sentence embeddings."""

    def __init__(self):
        super().__init__()
        self._total_sentences_embedded = 0

    def child_content_type(self) -> str:
        return "semantic_text_maxmin"

    def _log_chunking_complete(
        self, parents: list[Document], children: list[Document]
    ) -> None:
        logger.info(
            f"Max-Min chunking complete: {len(parents)} parents, "
            f"{len(children)} children "
            f"({self._total_sentences_embedded} sentences embedded)"
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
        sentences = split_sentences(text)
        if not sentences:
            return []

        vectors = embed_texts_batched(sentences)
        self._total_sentences_embedded += len(sentences)
        embeddings = np.array(vectors, dtype=np.float64)

        grouped = maxmin_group_sentences(sentences, embeddings)
        grouped = _split_oversized_chunks(grouped)

        results: list[tuple[Document, dict]] = []
        for chunk_index, chunk_text in enumerate(grouped):
            if len(chunk_text.strip()) < settings.child_min_text_length:
                continue
            meta = {
                "source_file": source_name,
                "page": page_num,
                "chunk_index": chunk_index,
                "parent_id": parent_id,
                "chunking_method": "maxmin",
            }
            results.append(
                (
                    Document(page_content=chunk_text.strip(), metadata=meta),
                    {},
                )
            )
        return results
