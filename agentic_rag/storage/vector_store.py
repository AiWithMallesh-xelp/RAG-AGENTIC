# storage/vector_store.py
"""
Qdrant vector store for child chunk embeddings.

- Deterministic uuid5 IDs
- Copy payload dict (no mutation)
- FilterSelector for delete-by-filter
- delete_by_source_id for stable document identity
- Collection name namespaced by index_dir (FIX B)
"""

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import settings
from utils import make_deterministic_id

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Manages child chunk storage and vector search in Qdrant."""

    def __init__(self):
        embed_kwargs = {
            "model": settings.embedding_model,
            "google_api_key": settings.google_api_key,
            "output_dimensionality": settings.embedding_dim,
        }
        self.embeddings = GoogleGenerativeAIEmbeddings(
            **embed_kwargs,
            task_type="retrieval_document",
        )

        self.query_embeddings = GoogleGenerativeAIEmbeddings(
            **embed_kwargs,
            task_type="retrieval_query",
        )

        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

        # FIX B: Namespace collection by index_dir
        self.collection_name = settings.qdrant_collection_name()
        self._ensure_collection()

    def _ensure_collection(self):
        if self.client.collection_exists(self.collection_name):
            info = self.client.get_collection(self.collection_name)
            current_size = info.config.params.vectors.size
            if current_size != settings.embedding_dim:
                logger.warning(
                    f"Qdrant collection dim {current_size} != "
                    f"{settings.embedding_dim}; recreating collection."
                )
                self.client.delete_collection(self.collection_name)

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                f"Created Qdrant collection: {self.collection_name} "
                f"(dim={settings.embedding_dim}, model={settings.embedding_model})"
            )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Embed and store child chunks in Qdrant."""
        if not documents:
            return []

        texts = [doc.page_content for doc in documents]
        vectors = self.embeddings.embed_documents(texts)

        points: list[PointStruct] = []
        ids: list[str] = []
        for doc, vector in zip(documents, vectors):
            point_id = doc.metadata.get("doc_id") or make_deterministic_id(
                doc.page_content
            )
            ids.append(point_id)

            payload = {
                "text": doc.page_content,
                **{
                    k: v
                    for k, v in doc.metadata.items()
                    if isinstance(v, (str, int, float, bool))
                },
            }

            points.append(
                PointStruct(id=point_id, vector=vector, payload=payload)
            )

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
            )

        logger.info(f"Added {len(points)} child chunks to Qdrant")
        return ids

    def search(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        """Vector similarity search."""
        top_k = top_k or settings.vector_top_k
        query_vector = self.query_embeddings.embed_query(query)

        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        documents_with_scores: list[tuple[Document, float]] = []
        for hit in response.points or []:
            payload = dict(hit.payload or {})
            text = payload.pop("text", "")
            doc = Document(page_content=text, metadata=payload)
            documents_with_scores.append((doc, hit.score))

        return documents_with_scores

    def delete_by_source_id(self, source_id: str) -> int:
        """
        Delete all child chunks for a given source_id.

        Primary deletion method for re-ingestion.
        source_id is stable (absolute file path), so old versions
        are correctly removed even when file content changes.
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="source_id",
                                match=MatchValue(value=source_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
            logger.info(
                f"Deleted Qdrant points for source_id={source_id[:50]}..."
            )
            return 1
        except Exception as e:
            logger.warning(f"Failed to delete Qdrant points by source_id: {e}")
            return 0

    def delete_by_file_hash(self, file_hash: str) -> int:
        """Delete by file_hash. Kept for backward compatibility."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="file_hash",
                                match=MatchValue(value=file_hash),
                            )
                        ]
                    )
                ),
                wait=True,
            )
            logger.info(
                f"Deleted Qdrant points for file_hash={file_hash[:12]}..."
            )
            return 1
        except Exception as e:
            logger.warning(f"Failed to delete Qdrant points: {e}")
            return 0

    def count(self) -> int:
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def delete_collection(self):
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
