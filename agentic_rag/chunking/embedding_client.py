# chunking/embedding_client.py
"""Batch sentence embeddings for Max-Min chunking (same model as Qdrant)."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import settings

_embeddings_client: GoogleGenerativeAIEmbeddings | None = None


def get_document_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings_client
    if _embeddings_client is None:
        _embeddings_client = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
            output_dimensionality=settings.embedding_dim,
            task_type="retrieval_document",
        )
    return _embeddings_client


def embed_texts_batched(texts: list[str]) -> list[list[float]]:
    """Embed texts in batches; returns (vectors, num_api_batches)."""
    if not texts:
        return []

    client = get_document_embeddings()
    batch_size = max(1, settings.maxmin_embed_batch_size)
    vectors: list[list[float]] = []
    num_batches = 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(client.embed_documents(batch))
        num_batches += 1
    return vectors
