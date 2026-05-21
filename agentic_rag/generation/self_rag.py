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
import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

from llm_factory import make_gemini_chat
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from config import settings
from utils import parse_llm_json

logger = logging.getLogger(__name__)

NOT_IN_CONTEXT_MESSAGE = (
    "The provided documents do not contain information to answer this question. "
    "Please ask about content from your uploaded files."
)

GREETING_REPLY = (
    "Hello! I can answer questions based on your uploaded documents only. "
    "What would you like to know?"
)

_SMALL_TALK_RE = re.compile(
    r"^\s*(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening)|"
    r"thanks|thank\s+you|bye|goodbye|ok|okay|yo)[\s!.?]*$",
    re.IGNORECASE,
)


def is_small_talk(query: str) -> bool:
    """True for brief greetings/thanks — the only no-retrieval case in strict mode."""
    q = query.strip()
    if not q or len(q) > 80:
        return False
    return bool(_SMALL_TALK_RE.match(q))


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

RETRIEVAL_DECISION_PROMPT = """You are a retrieval decision agent for a document-only Q&A system.

Rules:
- If the query asks about facts, concepts, people, systems, or anything that could appear in uploaded files → RETRIEVE
- ONLY use no_retrieval for brief greetings or thanks (e.g. "hi", "hello", "thanks")
- Identity questions ("who are you"), general knowledge, and chit-chat → RETRIEVE (documents likely lack an answer; do not use outside knowledge)
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

QUERY_TRANSFORM_PROMPT = """You are a query transformation agent for retrieval search. Given the user's ORIGINAL query and why retrieval failed, produce a search query.

Rules:
1. MUST keep all named entities, model names, benchmarks, and product names from the original query.
2. Do NOT change the topic or invent a different question (e.g. do not turn a paper-specific question into a generic API question).
3. Prefer: add synonyms, expand abbreviations, remove only truly ambiguous filler words.
4. Keep the transformed query concise (one sentence).

Strategies: expand abbreviations, add synonyms, clarify ambiguous terms only.

Respond ONLY with valid JSON:
{
  "transformed_query": "<improved search query>",
  "strategy_used": "<which strategy>",
  "reasoning": "<why this should help>"
}
"""

RAG_GENERATION_PROMPT = """You are a document-only Q&A assistant. You may ONLY use the context below.

Rules:
1. Use ONLY information explicitly stated in the context documents
2. Do NOT use outside knowledge, training data, or assumptions
3. If the context does not contain enough information to answer the query, respond with EXACTLY this sentence and nothing else:
   "The provided documents do not contain information to answer this question. Please ask about content from your uploaded files."
4. When you can answer, cite the document (source/page) for major claims
5. Do not describe yourself as a Google model or general-purpose AI

Context Documents:
{context}

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
        if settings.strict_document_only:
            if is_small_talk(query):
                return (
                    RetrievalDecision.NO_RETRIEVAL,
                    "Greeting or thanks — no document lookup",
                )
            return (
                RetrievalDecision.RETRIEVE,
                "Strict document-only mode — must check indexed files",
            )

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
        if not documents:
            return NOT_IN_CONTEXT_MESSAGE

        context = self._format_context(documents)
        if not context.strip():
            return NOT_IN_CONTEXT_MESSAGE

        prompt = RAG_GENERATION_PROMPT.format(context=context, query=query)
        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You answer only from provided documents. "
                        "Never use general knowledge."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        text = (response.content or "").strip()
        if settings.strict_document_only and self._looks_like_general_knowledge(
            text, query
        ):
            return NOT_IN_CONTEXT_MESSAGE
        return text

    def generate_general(self, query: str) -> str:
        """No-retrieval path: greetings only in strict mode; never general knowledge."""
        if settings.strict_document_only:
            if is_small_talk(query):
                return GREETING_REPLY
            return NOT_IN_CONTEXT_MESSAGE

        return NOT_IN_CONTEXT_MESSAGE

    @staticmethod
    def _looks_like_general_knowledge(answer: str, query: str) -> bool:
        """Detect answers that ignore document-only rules."""
        lower = answer.lower()
        if NOT_IN_CONTEXT_MESSAGE.lower() in lower:
            return False
        identity_markers = (
            "trained by google",
            "large language model",
            "language model developed",
            "i am an ai",
            "i'm an ai",
            "as an ai assistant",
        )
        if any(m in lower for m in identity_markers):
            return True
        q = query.lower().strip()
        if q in ("who are you?", "who are you", "what are you?"):
            return True
        return False

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
