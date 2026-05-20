# pipeline.py
"""
Complete Agentic RAG pipeline manager.

- source_id: stable document identity for deletion (survives content changes)
- file_hash: content version for change detection (skip if unchanged)
- Per-file processing with own hash and source_id
- Qdrant collection namespaced by index_dir
- Rewire all dependents after reset()
"""

import logging
from pathlib import Path
from typing import Union

from config import settings
from ingestion.pdf_parser import PDFParser
from ingestion.docx_parser import DocxParser
from chunking.factory import create_chunker, chunking_mode_name
from storage.vector_store import QdrantVectorStore
from storage.parent_store import ParentDocumentStore
from storage.bm25_store import BM25Store
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import GemmaReranker
from retrieval.parent_expander import ParentExpander
from generation.self_rag import SelfRAG
from agent.rag_agent import AgenticRAGAgent

logger = logging.getLogger(__name__)


class AgenticRAGPipeline:
    """Complete Agentic RAG pipeline manager with persistent state."""

    def __init__(self, index_dir: str = None):
        if index_dir:
            settings.index_dir = index_dir
        settings.ensure_index_dir()

        # ── Ingestion ───────────────────────────────────────────────
        self.pdf_parser = PDFParser()
        self.docx_parser = DocxParser()

        # ── Chunking ────────────────────────────────────────────────
        self.chunker = create_chunker()

        # ── Storage (persistent) ────────────────────────────────────
        self.vector_store = QdrantVectorStore()
        self.parent_store = ParentDocumentStore()
        self.bm25_store = BM25Store()

        # ── Retrieval ───────────────────────────────────────────────
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_store=self.bm25_store,
        )
        self.reranker = GemmaReranker()
        self.parent_expander = ParentExpander(parent_store=self.parent_store)

        # ── Generation ──────────────────────────────────────────────
        self.self_rag = SelfRAG()

        # ── Agent ───────────────────────────────────────────────────
        self.agent = AgenticRAGAgent(
            hybrid_retriever=self.hybrid_retriever,
            reranker=self.reranker,
            parent_expander=self.parent_expander,
            self_rag=self.self_rag,
        )

    # ── Ingest & Index ──────────────────────────────────────────────

    def ingest(self, file_paths: list[Union[str, Path]]) -> dict:
        """
        Ingest documents: Parse → Chunk → Store.

        Key behaviors:
        - source_id = absolute file path (stable identity)
        - file_hash = content hash (change detection)
        - Deletes old chunks by source_id before ingesting new version
        - Optionally skips ingestion if file_hash is unchanged
        - Processes each file independently
        - Builds BM25 index once at the end
        """
        total_pages = 0
        total_parents = 0
        total_children = 0
        files_processed = 0
        files_skipped = 0
        errors: list[str] = []
        bm25_needs_build = False

        for file_path in file_paths:
            file_path = Path(file_path)
            ext = file_path.suffix.lower()

            # ── Stable document identity ────────────────────────────
            source_id = str(file_path.resolve())

            try:
                # ── Parse per file ──────────────────────────────────
                if ext == ".pdf":
                    pages, fhash = self.pdf_parser.parse(file_path)
                elif ext == ".docx":
                    pages, fhash = self.docx_parser.parse(file_path)
                elif ext == ".doc":
                    logger.error(
                        f"Legacy .doc format not supported: {file_path.name}. "
                        f"Convert to .docx first."
                    )
                    continue
                else:
                    logger.warning(f"Unsupported file type: {ext}")
                    continue

                # ── Skip if unchanged and fully indexed ─────────────
                existing_hash = self.parent_store.get_file_hash(source_id)
                if existing_hash == fhash and self.bm25_store.has_source(source_id):
                    logger.info(
                        f"Skipping {file_path.name}: content unchanged "
                        f"(hash={fhash[:12]}...)"
                    )
                    files_skipped += 1
                    continue
                if existing_hash == fhash:
                    logger.warning(
                        f"Re-ingesting {file_path.name}: hash unchanged but "
                        f"no indexed child chunks (partial/failed prior run)"
                    )
                    self._delete_existing_chunks(source_id)

                # ── Delete old version by source_id ─────────────────
                # This correctly removes old chunks even when the file
                # content (and thus file_hash) has changed.
                self._delete_existing_chunks(source_id)

                # ── Chunk per file with its own hash and source_id ──
                chunked = self.chunker.chunk_document(pages, fhash, source_id)

                # ── Store ───────────────────────────────────────────
                self.parent_store.add_documents(chunked.parent_chunks)
                self.vector_store.add_documents(chunked.child_chunks)
                self.bm25_store.add_documents(chunked.child_chunks)
                bm25_needs_build = True

                total_pages += len(pages)
                total_parents += len(chunked.parent_chunks)
                total_children += len(chunked.child_chunks)
                files_processed += 1

                logger.info(
                    f"Ingested {file_path.name}: "
                    f"{len(pages)} pages, {len(chunked.parent_chunks)} parents, "
                    f"{len(chunked.child_chunks)} children "
                    f"(source_id={source_id[:50]}...)"
                )

            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")
                errors.append(f"{file_path.name}: {e}")
                self._delete_existing_chunks(source_id)
                continue

        # Build BM25 index once after all files are ingested
        if bm25_needs_build:
            self.bm25_store.build_index()

        stats = {
            "files_processed": files_processed,
            "files_skipped": files_skipped,
            "total_pages": total_pages,
            "parent_chunks": total_parents,
            "child_chunks": total_children,
            "errors": errors,
        }

        logger.info(f"Ingestion complete: {stats}")
        return stats

    def _delete_existing_chunks(self, source_id: str) -> None:
        """
        Remove old chunks for a source_id before re-ingestion.

        Uses source_id (stable file path) instead of file_hash
        so that updated files with different content hashes
        still correctly remove their old versions.
        """
        deleted_parents = self.parent_store.delete_by_source_id(source_id)
        deleted_bm25 = self.bm25_store.delete_by_source_id(source_id)
        deleted_vector = self.vector_store.delete_by_source_id(source_id)
        if deleted_parents > 0 or deleted_bm25 > 0 or deleted_vector > 0:
            logger.info(
                f"Removed old chunks for source_id={source_id[:50]}...: "
                f"{deleted_parents} parents, {deleted_bm25} BM25, "
                f"{deleted_vector} vector"
            )

    # ── Query ───────────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        return self.agent.invoke(question)

    # ── Status ──────────────────────────────────────────────────────

    def status(self) -> dict:
        """Get current index status."""
        return {
            "qdrant_points": self.vector_store.count(),
            "parent_chunks": self.parent_store.count(),
            "bm25_documents": self.bm25_store.count(),
            "collection_name": self.vector_store.collection_name,
            "index_dir": settings.index_dir,
            "chunking_mode": chunking_mode_name(),
        }

    # ── Reset ───────────────────────────────────────────────────────

    def reset(self):
        """Reset all stores AND rewire all dependents."""
        self.vector_store.delete_collection()
        self.vector_store._ensure_collection()

        self.parent_store.close()
        self.bm25_store.close()
        if settings.db_path.exists():
            settings.db_path.unlink()

        self.parent_store = ParentDocumentStore()
        self.bm25_store = BM25Store()
        self.chunker = create_chunker()

        # Rewire ALL dependents
        self.hybrid_retriever = HybridRetriever(
            self.vector_store, self.bm25_store
        )
        self.parent_expander = ParentExpander(self.parent_store)
        self.agent = AgenticRAGAgent(
            hybrid_retriever=self.hybrid_retriever,
            reranker=self.reranker,
            parent_expander=self.parent_expander,
            self_rag=self.self_rag,
        )

        logger.info("Pipeline reset complete — all stores and dependents rewired")

    def close(self):
        """Close all connections."""
        self.parent_store.close()
        self.bm25_store.close()
