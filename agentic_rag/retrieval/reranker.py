# retrieval/reranker.py
"""
Re-ranking with Gemma 4 26B via Ollama.

FIX #8: Restore format="json" for reliable JSON output.
Batch reranking for efficiency.
"""

import json
import logging
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from config import settings
from utils import extract_json

logger = logging.getLogger(__name__)

RERANK_SYSTEM_PROMPT = """You are a relevance scoring agent. Score how relevant a document is to a given query.

Scoring Scale:
- 10: Perfectly relevant — directly answers the query
- 7-9: Highly relevant — substantial related information
- 4-6: Moderately relevant — some related information
- 1-3: Not relevant — unrelated to the query

Respond ONLY with valid JSON:
{
  "score": <integer 1-10>,
  "reasoning": "<brief explanation>"
}
"""

BATCH_RERANK_PROMPT = """You are a relevance scoring agent. Score each document's relevance to the query.

Scoring Scale:
- 10: Perfectly relevant
- 7-9: Highly relevant
- 4-6: Moderately relevant
- 1-3: Not relevant

Respond ONLY with a valid JSON object:
{
  "grades": [
    {"doc_index": 0, "score": <1-10>, "reasoning": "<brief>"},
    {"doc_index": 1, "score": <1-10>, "reasoning": "<brief>"}
  ]
}
"""

REVALIDATION_PROMPT = """You are a document re-validation agent. Given a query and retrieved documents, determine:

1. Are there contradictions between the documents?
2. Is the information sufficient to answer the query?
3. Which documents are most trustworthy?

Respond ONLY with valid JSON:
{
  "contradictions_found": true/false,
  "contradiction_details": "<description if any>",
  "sufficient_information": true/false,
  "missing_information": "<what's missing>",
  "trust_ranking": [<list of doc indices from most to least trustworthy>]
}
"""


class GemmaReranker:
    """Re-ranker using Gemma 4 26B via Ollama with batch scoring."""

    def __init__(self):
        # FIX #8: Restore format="json" for reliable structured output
        self.llm = ChatOllama(
            model=settings.reranker_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
            num_ctx=8192,
            format="json",
        )

    def rerank(
        self,
        query: str,
        documents: list[tuple[Document, float]],
        top_k: int = None,
    ) -> list[tuple[Document, float, str]]:
        """Re-rank documents using batch scoring for efficiency."""
        top_k = top_k or settings.rerank_top_k

        if not documents:
            return []

        # Try batch scoring first
        batch_scores = self._batch_score(query, documents)

        if batch_scores is not None:
            scored_docs = []
            for i, (doc, orig_score) in enumerate(documents):
                score, reasoning = batch_scores.get(i, (5, "batch fallback"))
                scored_docs.append((doc, score, reasoning))
        else:
            scored_docs = []
            for doc, orig_score in documents:
                score, reasoning = self._score_document(query, doc.page_content)
                scored_docs.append((doc, score, reasoning))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]

    def _batch_score(
        self, query: str, documents: list[tuple[Document, float]]
    ) -> Optional[dict[int, tuple[int, str]]]:
        """Score all documents in a single LLM call with adaptive truncation."""
        if not documents:
            return None

        docs_text = ""
        available_tokens = 8192 - 500
        chars_per_token = 3
        max_chars_per_doc = max(
            200,
            (available_tokens * chars_per_token) // max(len(documents), 1),
        )

        for i, (doc, _) in enumerate(documents):
            truncated = doc.page_content[:max_chars_per_doc]
            docs_text += f"\n--- Document {i} ---\n{truncated}\n"

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=BATCH_RERANK_PROMPT),
                    HumanMessage(content=f"Query: {query}\n{docs_text}"),
                ]
            )
            raw = extract_json(response.content)
            data = json.loads(raw)

            if isinstance(data, dict) and "grades" not in data:
                # Might be a single-grade response wrapped in dict
                if "doc_index" in data:
                    data = {"grades": [data]}
                else:
                    data = {"grades": []}

            result: dict[int, tuple[int, str]] = {}
            for item in data.get("grades", []):
                idx = int(item.get("doc_index", 0))
                score = max(1, min(10, int(item.get("score", 5))))
                reasoning = item.get("reasoning", "")
                result[idx] = (score, reasoning)

            return result if result else None

        except Exception as e:
            logger.warning(
                f"Batch reranking failed: {e}. Falling back to per-doc."
            )
            return None

    def _score_document(
        self, query: str, document_text: str
    ) -> tuple[int, str]:
        """Score a single document (fallback)."""
        truncated_doc = document_text[:2000]
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=RERANK_SYSTEM_PROMPT),
                    HumanMessage(
                        content=f"Query: {query}\n\nDocument:\n{truncated_doc}"
                    ),
                ]
            )
            data = json.loads(extract_json(response.content))
            score = max(1, min(10, int(data.get("score", 5))))
            reasoning = data.get("reasoning", "No reasoning provided")
            return score, reasoning
        except Exception as e:
            logger.warning(f"Per-doc reranking failed: {e}")
            return 5, f"Scoring failed: {str(e)[:100]}"

    def revalidate(self, query: str, documents: list[Document]) -> dict:
        """Re-validate documents for contradictions and sufficiency."""
        docs_text = ""
        for i, doc in enumerate(documents):
            docs_text += f"\n--- Document {i + 1} ---\n{doc.page_content[:1000]}\n"

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=REVALIDATION_PROMPT),
                    HumanMessage(
                        content=f"Query: {query}\n\nRetrieved Documents:{docs_text}"
                    ),
                ]
            )
            return json.loads(extract_json(response.content))
        except Exception as e:
            logger.warning(f"Re-validation failed: {e}")
            return {
                "contradictions_found": False,
                "sufficient_information": True,
                "error": str(e),
            }
