# retrieval/hybrid_retriever.py
"""
Hybrid retrieval: Vector Search + BM25 → Reciprocal Rank Fusion (RRF).
"""

import logging
from typing import Optional

from langchain_core.documents import Document

from config import settings
from storage.vector_store import QdrantVectorStore
from storage.bm25_store import BM25Store

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever that merges vector and BM25 results using RRF."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        bm25_store: BM25Store,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.rrf_k = settings.rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        """Hybrid retrieval: Vector + BM25 → RRF fusion."""
        top_k = top_k or settings.vector_top_k

        vector_results = self.vector_store.search(
            query=query, top_k=top_k, filters=filters
        )
        logger.info(f"Vector search returned {len(vector_results)} results")

        bm25_results = self.bm25_store.search(query=query, top_k=top_k)
        logger.info(f"BM25 search returned {len(bm25_results)} results")

        fused_results = self._reciprocal_rank_fusion(
            vector_results, bm25_results
        )

        logger.info(f"RRF fusion produced {len(fused_results)} results")
        return fused_results

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[tuple[Document, float]],
        bm25_results: list[tuple[Document, float]],
    ) -> list[tuple[Document, float]]:
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, (doc, _) in enumerate(vector_results, start=1):
            doc_id = doc.metadata.get("doc_id", str(hash(doc.page_content)))
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (
                self.rrf_k + rank
            )
            doc_map[doc_id] = doc

        for rank, (doc, _) in enumerate(bm25_results, start=1):
            doc_id = doc.metadata.get("doc_id", str(hash(doc.page_content)))
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (
                self.rrf_k + rank
            )
            doc_map[doc_id] = doc

        sorted_results = sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [(doc_map[doc_id], score) for doc_id, score in sorted_results]
