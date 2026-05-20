# agent/state.py

from typing import TypedDict, Optional, Annotated
from operator import add

from langchain_core.documents import Document
from generation.self_rag import RetrievalDecision, RelevanceGrade, SupportGrade


class RAGAgentState(TypedDict):
    query: str
    original_query: str

    retrieval_decision: Optional[RetrievalDecision]
    retrieval_reasoning: Optional[str]

    raw_documents: list[tuple[Document, float]]
    reranked_documents: list[tuple[Document, float, str]]
    expanded_documents: list[Document]
    relevant_documents: list[Document]

    revalidation_result: Optional[dict]

    generation: Optional[str]

    relevance_grade: Optional[RelevanceGrade]
    support_grade: Optional[SupportGrade]
    support_unsupported_claims: list[str]
    usefulness_score: Optional[int]
    usefulness_reasoning: Optional[str]

    num_retries: int
    max_retries: int
    transformation_history: Annotated[list[str], add]

    final_answer: Optional[str]
    is_successful: bool
    error_message: Optional[str]
