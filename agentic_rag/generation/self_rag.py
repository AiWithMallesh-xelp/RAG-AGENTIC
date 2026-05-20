# generation/self_rag.py
"""
Self-RAG: Self-Reflective Retrieval-Augmented Generation.

Separate generation modes for no-retrieval path.
Batch relevance grading.
Strict support grading (configurable).
FIX #9: Remove dead code.
"""

import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

from llm_factory import make_gemini_chat
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from config import settings
from utils import parse_llm_json

logger = logging.getLogger(__name__)


class RetrievalDecision(str, Enum):
    RETRIEVE = "retrieve"
    NO_RETRIEVAL = "no_retrieval"


class RelevanceGrade(str, Enum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"


class SupportGrade(str, Enum):
    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NO_SUPPORT = "no_support"


@dataclass
class SelfRAGResult:
    query: str
    answer: str
    retrieval_decision: RetrievalDecision
    relevance_grade: RelevanceGrade
    support_grade: SupportGrade
    usefulness_score: int
    num_retries: int
    documents_used: list[Document] = field(default_factory=list)
    transformation_history: list[str] = field(default_factory=list)
    is_successful: bool = False


# ── Prompts ─────────────────────────────────────────────────────────

RETRIEVAL_DECISION_PROMPT = """You are a retrieval decision agent. Given a query, determine if external knowledge retrieval is needed.

Rules:
- If the query requires specific facts, data, or information → RETRIEVE
- If the query is about general knowledge, greetings, or simple reasoning → NO_RETRIEVAL
- When in doubt, choose RETRIEVE

Respond ONLY with valid JSON:
{
  "decision": "retrieve" or "no_retrieval",
  "reasoning": "<brief explanation>"
}
"""

BATCH_RELEVANCE_PROMPT = """You are a document relevance grading agent. Given a query and multiple documents, grade each document's relevance.

A document is RELEVANT if it contains information that could help answer the query.
A document is IRRELEVANT if it has no connection to the query.

Respond ONLY with a valid JSON object:
{
  "grades": [
    {"doc_index": 0, "grade": "relevant" or "irrelevant", "reasoning": "<brief>"},
    {"doc_index": 1, "grade": "relevant" or "irrelevant", "reasoning": "<brief>"}
  ]
}
"""

SUPPORT_GRADE_PROMPT = """You are a factual support grading agent. Given a generated response and the source documents, assess if the response is supported by the documents.

- FULLY_SUPPORTED: All claims are directly supported by the documents
- PARTIALLY_SUPPORTED: Some claims supported, others extrapolated
- NO_SUPPORT: Significant claims not found in documents (hallucination)

Be strict. Only mark as FULLY_SUPPORTED if you can verify each major claim.

Respond ONLY with valid JSON:
{
  "grade": "fully_supported" or "partially_supported" or "no_support",
  "unsupported_claims": ["<claim 1>", "<claim 2>"],
  "reasoning": "<brief explanation>"
}
"""

USEFULNESS_PROMPT = """You are a response usefulness grading agent. Rate how useful the response is for the query.

Scoring Scale:
- 5: Perfectly useful — directly and completely answers the query
- 4: Mostly useful — answers the main point but could be more complete
- 3: Somewhat useful — partially answers but misses key aspects
- 2: Barely useful — mostly misses the point
- 1: Not useful — does not address the query

Respond ONLY with valid JSON:
{
  "score": <integer 1-5>,
  "reasoning": "<brief explanation>"
}
"""

QUERY_TRANSFORM_PROMPT = """You are a query transformation agent. Given an original query and why previous attempts failed, transform the query.

Strategies:
1. Make more specific
2. Add context
3. Decompose into sub-queries
4. Use synonyms
5. Remove ambiguous terms

Respond ONLY with valid JSON:
{
  "transformed_query": "<improved query>",
  "strategy_used": "<which strategy>",
  "reasoning": "<why this should help>"
}
"""

RAG_GENERATION_PROMPT = """You are a helpful, accurate AI assistant. Answer the user's query based ONLY on the provided context documents.

Rules:
1. Use ONLY information from the provided context
2. If the context doesn't contain enough information, say "I don't have sufficient information to fully answer this question"
3. Cite specific parts of the context when making claims
4. Be precise and avoid speculation
5. Structure your answer clearly

Context Documents:
{context}

User Query: {query}

Provide a comprehensive, accurate answer:
"""

GENERAL_GENERATION_PROMPT = """You are a helpful, accurate AI assistant. Answer the user's query using your general knowledge.

Rules:
1. Be accurate and helpful
2. If you're unsure, say so
3. Structure your answer clearly

User Query: {query}

Answer:
"""


class SelfRAG:
    """Self-Reflective RAG with evaluation and retry loop."""

    def __init__(self):
        self.llm = make_gemini_chat(temperature=0.1, json_mode=True)

    def decide_retrieval(
        self, query: str
    ) -> tuple[RetrievalDecision, str]:
        response = self._invoke_json(
            RETRIEVAL_DECISION_PROMPT, f"Query: {query}"
        )
        decision_str = response.get("decision", "retrieve")
        reasoning = response.get("reasoning", "")
        try:
            decision = RetrievalDecision(decision_str)
        except ValueError:
            decision = RetrievalDecision.RETRIEVE
        return decision, reasoning

    def grade_relevance_batch(
        self, query: str, documents: list[Document]
    ) -> tuple[RelevanceGrade, list[Document], str]:
        """
        Batch relevance grading — single LLM call for all documents.
        FIX #9: Removed unused message variable.
        """
        if not documents:
            return RelevanceGrade.IRRELEVANT, [], "No documents provided"

        docs_text = ""
        for i, doc in enumerate(documents):
            docs_text += f"\n--- Document {i} ---\n{doc.page_content[:1200]}\n"

        # FIX #9: Removed the unused HumanMessage that was created and discarded
        result = self._invoke_json(
            BATCH_RELEVANCE_PROMPT, f"Query: {query}\n{docs_text}"
        )

        if not result or "grades" not in result:
            logger.warning(
                "Batch relevance grading API failure. "
                "Returning all documents as relevant (conservative fallback)."
            )
            return (
                RelevanceGrade.RELEVANT,
                documents,
                "Grading API failed — keeping all docs",
            )

        relevant_docs: list[Document] = []
        failure_count = 0
        all_reasoning: list[str] = []

        for item in result["grades"]:
            idx = int(item.get("doc_index", -1))
            grade_str = item.get("grade", "relevant")
            reasoning = item.get("reasoning", "")

            if idx < 0 or idx >= len(documents):
                failure_count += 1
                continue

            try:
                grade = RelevanceGrade(grade_str)
            except ValueError:
                failure_count += 1
                grade = RelevanceGrade.RELEVANT

            if grade == RelevanceGrade.RELEVANT:
                relevant_docs.append(documents[idx])
            all_reasoning.append(f"Doc {idx}: {grade.value} — {reasoning}")

        if failure_count > 0:
            logger.warning(
                f"Relevance grading had {failure_count} failures "
                f"out of {len(result['grades'])} grades"
            )

        if not relevant_docs:
            return RelevanceGrade.IRRELEVANT, [], "All documents were irrelevant"
        elif len(relevant_docs) < len(documents):
            return (
                RelevanceGrade.RELEVANT,
                relevant_docs,
                f"{len(relevant_docs)}/{len(documents)} docs relevant",
            )
        else:
            return RelevanceGrade.RELEVANT, relevant_docs, "All documents relevant"

    def generate_with_docs(
        self, query: str, documents: list[Document]
    ) -> str:
        """Generate response grounded in documents."""
        context = self._format_context(documents)
        prompt = RAG_GENERATION_PROMPT.format(context=context, query=query)
        response = self.llm.invoke(
            [
                SystemMessage(content="You are a precise, factual assistant."),
                HumanMessage(content=prompt),
            ]
        )
        return response.content

    def generate_general(self, query: str) -> str:
        """Generate response using general knowledge (no documents)."""
        prompt = GENERAL_GENERATION_PROMPT.format(query=query)
        response = self.llm.invoke(
            [
                SystemMessage(content="You are a helpful, accurate assistant."),
                HumanMessage(content=prompt),
            ]
        )
        return response.content

    def grade_support(
        self, query: str, response: str, documents: list[Document]
    ) -> tuple[SupportGrade, list[str], str]:
        context = self._format_context(documents)
        content = (
            f"Query: {query}\n\n"
            f"Generated Response:\n{response}\n\n"
            f"Source Documents:\n{context}"
        )

        result = self._invoke_json(SUPPORT_GRADE_PROMPT, content)
        grade_str = result.get("grade", "partially_supported")
        unsupported = result.get("unsupported_claims", [])
        reasoning = result.get("reasoning", "")

        try:
            grade = SupportGrade(grade_str)
        except ValueError:
            grade = SupportGrade.PARTIALLY_SUPPORTED

        return grade, unsupported, reasoning

    def grade_usefulness(
        self, query: str, response: str
    ) -> tuple[int, str]:
        """Grade response usefulness. Default score on API failure is 1 (fail)."""
        content = f"Query: {query}\n\nResponse:\n{response}"
        result = self._invoke_json(USEFULNESS_PROMPT, content)
        score = int(result.get("score", 1))
        reasoning = result.get("reasoning", "API failure — defaulting to fail")
        return max(1, min(5, score)), reasoning

    def transform_query(
        self, original_query: str, failure_reason: str
    ) -> tuple[str, str]:
        content = (
            f"Original Query: {original_query}\n\n"
            f"Failure Reason: {failure_reason}\n\n"
            f"Transform the query to improve retrieval and generation."
        )
        result = self._invoke_json(QUERY_TRANSFORM_PROMPT, content)
        transformed = result.get("transformed_query", original_query)
        strategy = result.get("strategy_used", "unknown")
        return transformed, strategy

    # ── Helpers ─────────────────────────────────────────────────────

    def _invoke_json(
        self, system_prompt: str, user_content: str
    ) -> dict:
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_content),
                ]
            )
            data = parse_llm_json(response.content)
            return data if data else {}
        except Exception as e:
            logger.warning(f"JSON parsing failed in _invoke_json: {e}")
            return {}

    def _format_context(self, documents: list[Document]) -> str:
        formatted = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", "?")
            formatted.append(
                f"[Document {i}] (Source: {source}, Page: {page})\n"
                f"{doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted)
