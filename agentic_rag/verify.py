#!/usr/bin/env python3
"""Smoke tests for Agentic RAG (no external API calls)."""

import sys
import tempfile
from pathlib import Path


def test_imports():
    import config  # noqa: F401
    import utils  # noqa: F401
    from ingestion.pdf_parser import PDFParser, ParsedPage  # noqa: F401
    from chunking.agentic_chunker import AgenticChunker  # noqa: F401
    from storage.parent_store import ParentDocumentStore  # noqa: F401
    from storage.bm25_store import BM25Store, tokenize  # noqa: F401
    from retrieval.hybrid_retriever import HybridRetriever  # noqa: F401
    from generation.self_rag import SelfRAG  # noqa: F401
    from agent.rag_agent import AgenticRAGAgent  # noqa: F401
    print("OK imports")


def test_qdrant_namespace():
    from config import Settings

    s1 = Settings(**{"index_dir": str(Path("/tmp/agentic_rag_test_alpha").resolve())})
    s2 = Settings(**{"index_dir": str(Path("/tmp/agentic_rag_test_beta").resolve())})
    n1 = s1.qdrant_collection_name()
    n2 = s2.qdrant_collection_name()
    assert n1 != n2, f"Collections should differ: {n1} vs {n2}"
    assert n1.startswith("agentic_rag_child_")
    print(f"OK namespace: {n1} != {n2}")


def test_extract_json():
    from utils import extract_json

    raw = 'Here is output:\n```json\n{"a": 1}\n```\nAnd more text.'
    assert extract_json(raw) == '{"a": 1}'
    raw2 = 'prefix {"nested": {"x": 1}} suffix'
    assert '"nested"' in extract_json(raw2)
    print("OK extract_json")


def test_deterministic_ids():
    from utils import make_deterministic_id

    id1 = make_deterministic_id("/docs/a.pdf", 1, "child", 0, 0, 0)
    id2 = make_deterministic_id("/docs/a.pdf", 1, "child", 0, 1, 0)
    assert id1 != id2
    assert id1 == make_deterministic_id("/docs/a.pdf", 1, "child", 0, 0, 0)
    print("OK deterministic IDs")


def test_parent_store_source_id():
    from config import settings
    from storage.parent_store import ParentDocumentStore
    from langchain_core.documents import Document

    with tempfile.TemporaryDirectory() as tmp:
        settings.index_dir = tmp
        store = ParentDocumentStore()
        sid = str(Path("/tmp/manual.pdf").resolve())
        doc = Document(
            page_content="test parent",
            metadata={
                "doc_id": "p1",
                "source_id": sid,
                "source_file": "manual.pdf",
                "file_hash": "hash_v1",
            },
        )
        store.add_documents([doc])
        assert store.get_file_hash(sid) == "hash_v1"
        assert store.delete_by_source_id(sid) == 1
        assert store.get_file_hash(sid) is None
        store.close()
    print("OK parent_store source_id")


def test_bm25_doc_id_lookup():
    from config import settings
    from storage.bm25_store import BM25Store
    from langchain_core.documents import Document

    with tempfile.TemporaryDirectory() as tmp:
        settings.index_dir = tmp
        store = BM25Store()
        doc = Document(
            page_content="keyword searchable text",
            metadata={
                "doc_id": "c1",
                "source_id": "/tmp/a.pdf",
                "source_file": "a.pdf",
            },
        )
        store.add_documents([doc])
        store.build_index()
        results = store.search("keyword")
        assert len(results) == 1
        assert results[0][0].metadata["doc_id"] == "c1"
        store.close()
    print("OK bm25 doc_id lookup")


def test_maxmin_grouping():
    """Max-Min splits when sentence embedding clusters diverge (no API)."""
    import numpy as np
    from chunking.maxmin_chunker import maxmin_group_sentences

    sentences = [
        "Topic A sentence one.",
        "Topic A sentence two.",
        "Topic A sentence three.",
        "Topic B sentence four.",
        "Topic B sentence five.",
        "Topic B sentence six.",
    ]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.05, 0.95, 0.0],
            [0.1, 0.9, 0.0],
        ],
        dtype=np.float64,
    )
    chunks = maxmin_group_sentences(sentences, embeddings)
    assert len(chunks) >= 2
    joined = " ".join(chunks)
    for s in sentences:
        assert s in joined
    print("OK maxmin_group_sentences")


def test_chunking_factory():
    from config import settings
    from chunking.factory import create_chunker, chunking_mode_name
    from chunking.agentic_chunker import AgenticChunker
    from chunking.maxmin_chunker import MaxMinChunker

    original = settings.use_maxmin_chunking
    try:
        settings.use_maxmin_chunking = False
        assert chunking_mode_name() == "agentic_llm"
        assert isinstance(create_chunker(), AgenticChunker)
        settings.use_maxmin_chunking = True
        assert chunking_mode_name() == "maxmin"
        assert isinstance(create_chunker(), MaxMinChunker)
    finally:
        settings.use_maxmin_chunking = original
    print("OK chunking factory toggle")


def test_chunker_table_columns():
    from chunking.maxmin_chunker import MaxMinChunker

    chunker = MaxMinChunker.__new__(MaxMinChunker)
    table = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
    chunks = chunker._chunk_table(
        table, "", "doc.pdf", 1, "parent1", "hash", "/tmp/doc.pdf", 0, 0
    )
    assert len(chunks) >= 1
    assert "| A |" in chunks[0].page_content
    assert chunks[0].metadata["source_id"] == "/tmp/doc.pdf"
    print("OK chunker table + source_id metadata")


def test_pipeline_delete_by_source_id():
    """Simulate v5 re-ingest: delete by source_id removes old hash version."""
    import tempfile
    from config import settings
    from storage.parent_store import ParentDocumentStore
    from storage.bm25_store import BM25Store
    from langchain_core.documents import Document

    with tempfile.TemporaryDirectory() as tmp:
        settings.index_dir = tmp
        parent = ParentDocumentStore()
        bm25 = BM25Store()
        sid = str(Path("/tmp/manual.pdf").resolve())

        old_child = Document(
            page_content="old content",
            metadata={
                "doc_id": "old_child",
                "source_id": sid,
                "file_hash": "hash_old",
                "source_file": "manual.pdf",
            },
        )
        parent.add_documents([
            Document(
                page_content="old parent",
                metadata={"doc_id": "p1", "source_id": sid, "file_hash": "hash_old"},
            )
        ])
        bm25.add_documents([old_child])

        parent.delete_by_source_id(sid)
        bm25.delete_by_source_id(sid)
        assert parent.count() == 0
        assert bm25.count() == 0
        parent.close()
        bm25.close()
    print("OK source_id deletion lifecycle")


def test_strict_document_only_routing():
    from generation.self_rag import (
        SelfRAG,
        RetrievalDecision,
        NOT_IN_CONTEXT_MESSAGE,
        GREETING_REPLY,
        is_small_talk,
    )

    assert is_small_talk("Hi")
    assert is_small_talk("Hello!")
    assert not is_small_talk("Who are you?")

    rag = SelfRAG.__new__(SelfRAG)
    d, _ = SelfRAG.decide_retrieval(rag, "Who are you?")
    assert d == RetrievalDecision.RETRIEVE
    d2, _ = SelfRAG.decide_retrieval(rag, "Hi")
    assert d2 == RetrievalDecision.NO_RETRIEVAL

    assert SelfRAG.generate_general(rag, "Who are you?") == NOT_IN_CONTEXT_MESSAGE
    assert SelfRAG.generate_general(rag, "Hi") == GREETING_REPLY
    assert SelfRAG.generate_with_docs(rag, "test", []) == NOT_IN_CONTEXT_MESSAGE
    print("OK strict document-only routing")


def test_grade_usefulness_default():
    from generation.self_rag import SelfRAG

    rag = SelfRAG.__new__(SelfRAG)
    # Simulate API failure returning {}
    import generation.self_rag as mod

    original = mod.SelfRAG._invoke_json

    def fail_invoke(self, prompt, content):
        return {}

    mod.SelfRAG._invoke_json = fail_invoke
    try:
        score, _ = SelfRAG.grade_usefulness(rag, "q", "answer")
        assert score == 1, f"Expected fail default 1, got {score}"
    finally:
        mod.SelfRAG._invoke_json = original
    print("OK grade_usefulness fail default")


def main():
    tests = [
        test_imports,
        test_qdrant_namespace,
        test_extract_json,
        test_deterministic_ids,
        test_parent_store_source_id,
        test_bm25_doc_id_lookup,
        test_maxmin_grouping,
        test_chunking_factory,
        test_chunker_table_columns,
        test_pipeline_delete_by_source_id,
        test_strict_document_only_routing,
        test_grade_usefulness_default,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
    if failed:
        sys.exit(1)
    print("\nAll verification checks passed.")


if __name__ == "__main__":
    main()
