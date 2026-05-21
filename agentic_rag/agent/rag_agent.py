# agent/rag_agent.py
"""
LangGraph-based Agentic RAG agent.

All fixes applied including:
- FIX #6: Revalidation loop guarded by max_retries
- FIX #1 (this round): _check_relevance also guarded by max_retries
- Separate general vs RAG generation paths
- Revalidation routing
- Strict support grading
- Read max_retries from state
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from config import settings
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import GemmaReranker
from retrieval.parent_expander import ParentExpander
from generation.self_rag import (
    SelfRAG,
    RetrievalDecision,
    RelevanceGrade,
    SupportGrade,
    NOT_IN_CONTEXT_MESSAGE,
    GREETING_REPLY,
    is_small_talk,
)
from .state import RAGAgentState

logger = logging.getLogger(__name__)


class AgenticRAGAgent:
    """Full Agentic RAG pipeline with LangGraph."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: GemmaReranker,
        parent_expander: ParentExpander,
        self_rag: SelfRAG,
    ):
        self.retriever = hybrid_retriever
        self.reranker = reranker
        self.expander = parent_expander
        self.self_rag = self_rag
        self.max_retries = settings.max_retries
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(RAGAgentState)

        # Nodes
        workflow.add_node("route_query", self._route_query)
        workflow.add_node("hybrid_retrieve", self._hybrid_retrieve)
        workflow.add_node("rerank", self._rerank)
        workflow.add_node("expand_parents", self._expand_parents)
        workflow.add_node("revalidate", self._revalidate)
        workflow.add_node("grade_relevance", self._grade_relevance)
        workflow.add_node("generate_rag", self._generate_rag)
        workflow.add_node("generate_general", self._generate_general)
        workflow.add_node("evaluate_rag", self._evaluate_rag)
        workflow.add_node("evaluate_general", self._evaluate_general)
        workflow.add_node("transform_query", self._transform_query)

        workflow.set_entry_point("route_query")

        # Edges
        workflow.add_conditional_edges(
            "route_query",
            self._should_retrieve,
            {
                "retrieve": "hybrid_retrieve",
                "no_retrieval": "generate_general",
            },
        )

        workflow.add_edge("hybrid_retrieve", "rerank")
        workflow.add_edge("rerank", "expand_parents")
        workflow.add_edge("expand_parents", "revalidate")

        workflow.add_conditional_edges(
            "revalidate",
            self._check_revalidation,
            {
                "sufficient": "grade_relevance",
                "insufficient": "transform_query",
            },
        )

        workflow.add_conditional_edges(
            "grade_relevance",
            self._check_relevance,
            {
                "relevant": "generate_rag",
                "irrelevant": "transform_query",
            },
        )

        workflow.add_edge("generate_rag", "evaluate_rag")
        workflow.add_edge("generate_general", "evaluate_general")
        workflow.add_edge("evaluate_general", END)

        workflow.add_conditional_edges(
            "evaluate_rag",
            self._check_evaluation,
            {
                "pass": END,
                "fail": "transform_query",
                "max_retries": END,
            },
        )

        workflow.add_edge("transform_query", "hybrid_retrieve")

        return workflow.compile()

    # ── Node Implementations ────────────────────────────────────────

    def _route_query(self, state: RAGAgentState) -> dict:
        query = state["query"]
        decision, reasoning = self.self_rag.decide_retrieval(query)
        logger.info(
            f"Retrieval decision for '{query[:50]}...': "
            f"{decision.value} — {reasoning}"
        )
        return {
            "retrieval_decision": decision,
            "retrieval_reasoning": reasoning,
            "original_query": state.get("original_query", query),
        }

    def _hybrid_retrieve(self, state: RAGAgentState) -> dict:
        results = self.retriever.retrieve(query=state["query"])
        return {"raw_documents": results}

    def _user_query(self, state: RAGAgentState) -> str:
        """User-facing question (stable across retrieval retries)."""
        return state.get("original_query") or state["query"]

    def _search_query(self, state: RAGAgentState) -> str:
        """Query used for vector/BM25/rerank (may be transformed on retry)."""
        return state["query"]

    def _rerank(self, state: RAGAgentState) -> dict:
        raw_docs = state.get("raw_documents", [])
        if not raw_docs:
            return {"reranked_documents": []}
        reranked = self.reranker.rerank(
            query=self._search_query(state), documents=raw_docs
        )
        return {"reranked_documents": reranked}

    def _expand_parents(self, state: RAGAgentState) -> dict:
        reranked = state.get("reranked_documents", [])
        child_results = [(doc, score) for doc, score, _ in reranked]
        expanded = self.expander.expand(child_results)
        return {"expanded_documents": expanded}

    def _revalidate(self, state: RAGAgentState) -> dict:
        query = self._user_query(state)
        documents = state.get("expanded_documents", [])

        if len(documents) < 2:
            return {
                "revalidation_result": {
                    "sufficient_information": True,
                    "contradictions_found": False,
                }
            }

        result = self.reranker.revalidate(query=query, documents=documents)
        logger.info(
            f"Re-validation: sufficient={result.get('sufficient_information')}, "
            f"contradictions={result.get('contradictions_found')}"
        )
        return {"revalidation_result": result}

    def _grade_relevance(self, state: RAGAgentState) -> dict:
        query = self._user_query(state)
        documents = state.get("expanded_documents", [])
        grade, relevant_docs, reasoning = self.self_rag.grade_relevance_batch(
            query, documents
        )
        logger.info(f"Relevance: {grade.value} — {reasoning}")
        return {"relevance_grade": grade, "relevant_documents": relevant_docs}

    def _generate_rag(self, state: RAGAgentState) -> dict:
        query = self._user_query(state)
        documents = (
            state.get("relevant_documents")
            or state.get("expanded_documents", [])
        )
        relevance = state.get("relevance_grade")
        if (
            settings.strict_document_only
            and relevance == RelevanceGrade.IRRELEVANT
            and not state.get("relevant_documents")
        ):
            generation = NOT_IN_CONTEXT_MESSAGE
        else:
            generation = self.self_rag.generate_with_docs(query, documents)
        return {"generation": generation}

    def _generate_general(self, state: RAGAgentState) -> dict:
        query = state["query"]
        generation = self.self_rag.generate_general(query)
        return {"generation": generation}

    def _evaluate_rag(self, state: RAGAgentState) -> dict:
        """Evaluate RAG-generated response: support + usefulness."""
        query = self._user_query(state)
        generation = state.get("generation", "")
        documents = (
            state.get("relevant_documents")
            or state.get("expanded_documents", [])
        )
        num_retries = state.get("num_retries", 0)

        support_grade, unsupported, support_reasoning = self.self_rag.grade_support(
            query, generation, documents
        )
        logger.info(f"Support: {support_grade.value} — {support_reasoning}")

        usefulness_score, usefulness_reasoning = self.self_rag.grade_usefulness(
            query, generation
        )
        logger.info(f"Usefulness: {usefulness_score}/5 — {usefulness_reasoning}")

        if settings.strict_support:
            support_pass = (
                support_grade == SupportGrade.FULLY_SUPPORTED and not unsupported
            )
        else:
            support_pass = support_grade in (
                SupportGrade.FULLY_SUPPORTED,
                SupportGrade.PARTIALLY_SUPPORTED,
            )

        usefulness_pass = usefulness_score >= settings.usefulness_min_score
        is_successful = support_pass and usefulness_pass

        if is_successful:
            final_answer = generation
        elif num_retries >= state.get("max_retries", self.max_retries):
            final_answer = generation  # Best effort
            is_successful = False
        else:
            final_answer = None

        return {
            "support_grade": support_grade,
            "support_unsupported_claims": unsupported,
            "usefulness_score": usefulness_score,
            "usefulness_reasoning": usefulness_reasoning,
            "final_answer": final_answer,
            "is_successful": is_successful,
        }

    def _evaluate_general(self, state: RAGAgentState) -> dict:
        """Evaluate general (no-retrieval) responses. Usefulness only."""
        query = state["query"]
        generation = state.get("generation", "")

        if settings.strict_document_only and generation.strip() in (
            NOT_IN_CONTEXT_MESSAGE,
            GREETING_REPLY,
        ):
            if generation.strip() == GREETING_REPLY:
                usefulness_score, usefulness_reasoning = 5, "Greeting handled"
            else:
                usefulness_score, usefulness_reasoning = (
                    5,
                    "Correctly refused — not in uploaded documents",
                )
            is_successful = True
        else:
            usefulness_score, usefulness_reasoning = self.self_rag.grade_usefulness(
                query, generation
            )
            is_successful = usefulness_score >= settings.usefulness_min_score

        return {
            "support_grade": SupportGrade.FULLY_SUPPORTED,
            "support_unsupported_claims": [],
            "usefulness_score": usefulness_score,
            "usefulness_reasoning": usefulness_reasoning,
            "final_answer": generation,
            "is_successful": is_successful,
        }

    def _transform_query(self, state: RAGAgentState) -> dict:
        original_query = self._user_query(state)
        num_retries = state.get("num_retries", 0)

        support_grade = state.get("support_grade")
        usefulness_score = state.get("usefulness_score", 3)
        relevance_grade = state.get("relevance_grade")
        revalidation = state.get("revalidation_result", {})

        failure_parts: list[str] = []
        if revalidation and not revalidation.get(
            "sufficient_information", True
        ):
            failure_parts.append(
                "Retrieved documents contain insufficient information"
            )
        if revalidation and revalidation.get("contradictions_found"):
            failure_parts.append(
                "Retrieved documents contain contradictions"
            )
        if relevance_grade == RelevanceGrade.IRRELEVANT:
            failure_parts.append(
                "Retrieved documents were irrelevant to the query"
            )
        if support_grade == SupportGrade.NO_SUPPORT:
            failure_parts.append(
                "Generated response was not supported by documents"
            )
        if (
            support_grade == SupportGrade.PARTIALLY_SUPPORTED
            and settings.strict_support
        ):
            failure_parts.append(
                "Generated response was only partially supported"
            )
        if usefulness_score and usefulness_score < settings.usefulness_min_score:
            failure_parts.append(
                f"Response usefulness was low ({usefulness_score}/5)"
            )

        failure_reason = "; ".join(failure_parts) or "Previous attempt failed"

        transformed_query, strategy = self.self_rag.transform_query(
            original_query, failure_reason
        )
        logger.info(
            f"Query transformed (retry {num_retries + 1}): "
            f"'{original_query[:50]}...' → '{transformed_query[:50]}...' "
            f"(strategy: {strategy})"
        )

        return {
            "query": transformed_query,
            "num_retries": num_retries + 1,
            "transformation_history": [f"{strategy}: {transformed_query}"],
        }

    # ── Conditional Edges ───────────────────────────────────────────

    def _should_retrieve(
        self, state: RAGAgentState
    ) -> Literal["retrieve", "no_retrieval"]:
        decision = state.get("retrieval_decision", RetrievalDecision.RETRIEVE)
        return (
            "retrieve"
            if decision == RetrievalDecision.RETRIEVE
            else "no_retrieval"
        )

    def _check_revalidation(
        self, state: RAGAgentState
    ) -> Literal["sufficient", "insufficient"]:
        """Route based on revalidation, guarded against infinite loops."""
        num_retries = state.get("num_retries", 0)
        max_retries = state.get("max_retries", self.max_retries)

        if num_retries >= max_retries:
            logger.warning(
                f"Max retries ({max_retries}) reached in revalidation loop, "
                f"proceeding with current documents."
            )
            return "sufficient"

        result = state.get("revalidation_result", {})
        sufficient = result.get("sufficient_information", True)
        contradictions = result.get("contradictions_found", False)

        # Only re-retrieve on real contradictions; partial/missing info → answer anyway
        if contradictions:
            logger.info(
                f"Revalidation routing: contradictions=True → transform query"
            )
            return "insufficient"

        if not sufficient:
            logger.info(
                "Revalidation: insufficient information but no contradictions "
                "→ proceed to relevance/generation"
            )
        return "sufficient"

    def _check_relevance(
        self, state: RAGAgentState
    ) -> Literal["relevant", "irrelevant"]:
        """
        FIX #1: Mirror the same max_retries guard as _check_revalidation.
        
        Without this guard, the loop:
          grade_relevance: IRRELEVANT
            → transform_query (num_retries increments past max)
            → hybrid_retrieve → rerank → expand
            → revalidate: _check_revalidation sees retries >= max → "sufficient"
            → grade_relevance: IRRELEVANT again
            → transform_query (num_retries keeps climbing)
            → ... ∞
        
        Now both exit points from the retrieve→evaluate cycle check max_retries,
        ensuring the graph always reaches generate or END within bounded steps.
        """
        num_retries = state.get("num_retries", 0)
        max_retries = state.get("max_retries", self.max_retries)

        if num_retries >= max_retries:
            logger.warning(
                f"Max retries ({max_retries}) reached in relevance check — "
                f"proceeding to generation with available documents."
            )
            return "relevant"

        grade = state.get("relevance_grade", RelevanceGrade.RELEVANT)
        return "relevant" if grade == RelevanceGrade.RELEVANT else "irrelevant"

    def _check_evaluation(
        self, state: RAGAgentState
    ) -> Literal["pass", "fail", "max_retries"]:
        if state.get("is_successful"):
            return "pass"
        num_retries = state.get("num_retries", 0)
        max_retries = state.get("max_retries", self.max_retries)
        if num_retries >= max_retries:
            logger.warning(
                f"Max retries ({max_retries}) reached. Returning best effort."
            )
            return "max_retries"
        return "fail"

    # ── Public Interface ────────────────────────────────────────────

    def invoke(self, query: str) -> dict:
        initial_state: RAGAgentState = {
            "query": query,
            "original_query": query,
            "raw_documents": [],
            "reranked_documents": [],
            "expanded_documents": [],
            "relevant_documents": [],
            "revalidation_result": None,
            "generation": None,
            "retrieval_decision": None,
            "retrieval_reasoning": None,
            "relevance_grade": None,
            "support_grade": None,
            "support_unsupported_claims": [],
            "usefulness_score": None,
            "usefulness_reasoning": None,
            "num_retries": 0,
            "max_retries": self.max_retries,
            "transformation_history": [],
            "final_answer": None,
            "is_successful": False,
            "error_message": None,
        }

        result = self.graph.invoke(initial_state)

        return {
            "answer": result.get("final_answer")
            or result.get("generation", ""),
            "is_successful": result.get("is_successful", False),
            "retrieval_decision": result.get("retrieval_decision"),
            "relevance_grade": result.get("relevance_grade"),
            "support_grade": result.get("support_grade"),
            "usefulness_score": result.get("usefulness_score"),
            "documents": result.get("relevant_documents")
            or result.get("expanded_documents", []),
            "num_retries": result.get("num_retries", 0),
            "transformations": result.get("transformation_history", []),
            "revalidation": result.get("revalidation_result"),
        }
