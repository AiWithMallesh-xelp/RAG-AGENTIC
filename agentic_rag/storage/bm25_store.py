# storage/bm25_store.py
"""
BM25 keyword index with SQLite-backed persistence.

Now includes source_id for stable document identity.
Deletion by source_id instead of file_hash.
Deferred index build for efficient batch ingestion.
"""

import json
import logging
from typing import Optional
from pathlib import Path

import nltk
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from config import settings

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = word_tokenize(text)
    return [
        t for t in tokens if t.isalnum() and t not in STOP_WORDS and len(t) > 1
    ]


class BM25Store:
    """BM25 keyword index with SQLite-backed persistence and deferred indexing."""

    def __init__(self, db_path: Optional[str] = None):
        import sqlite3

        self.db_path = db_path or str(settings.db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_table()

        self.bm25: Optional[BM25Okapi] = None
        self.doc_ids: list[str] = []
        self.tokenized_corpus: list[list[str]] = []
        self._index_dirty = False

        self._rebuild_index()

    def _ensure_table(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bm25_documents (
                doc_id TEXT PRIMARY KEY,
                source_id TEXT,
                parent_id TEXT,
                source_file TEXT,
                page INTEGER,
                content_type TEXT,
                text TEXT NOT NULL,
                metadata_json TEXT,
                file_hash TEXT
            )
            """
        )
        # Add source_id column if migrating from older schema
        try:
            self.conn.execute(
                "ALTER TABLE bm25_documents ADD COLUMN source_id TEXT"
            )
        except Exception:
            pass

        # Index for fast deletion by source_id
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bm25_source_id "
            "ON bm25_documents(source_id)"
        )
        self.conn.commit()

    def _rebuild_index(self):
        """Rebuild BM25 index from SQLite on startup. ORDER BY rowid for stability."""
        cursor = self.conn.execute(
            "SELECT doc_id, text FROM bm25_documents ORDER BY rowid"
        )
        rows = cursor.fetchall()

        if rows:
            self.doc_ids = []
            self.tokenized_corpus = []
            for doc_id, text in rows:
                self.doc_ids.append(doc_id)
                self.tokenized_corpus.append(tokenize(text))

            self.bm25 = BM25Okapi(self.tokenized_corpus)
            logger.info(
                f"BM25 index rebuilt: {len(self.doc_ids)} documents from SQLite"
            )
        else:
            self.doc_ids = []
            self.tokenized_corpus = []
            self.bm25 = None

        self._index_dirty = False

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to SQLite. Does NOT rebuild index — call build_index() after."""
        for doc in documents:
            doc_id = doc.metadata.get("doc_id", "")
            source_id = doc.metadata.get("source_id", "")
            parent_id = doc.metadata.get("parent_id", "")
            source = doc.metadata.get("source_file", "")
            page = doc.metadata.get("page")
            ctype = doc.metadata.get("content_type", "")
            file_hash = doc.metadata.get("file_hash", "")

            extra_meta = {
                k: v
                for k, v in doc.metadata.items()
                if k
                not in {
                    "doc_id",
                    "source_id",
                    "parent_id",
                    "source_file",
                    "page",
                    "content_type",
                    "file_hash",
                }
                and isinstance(v, (str, int, float, bool))
            }

            self.conn.execute(
                """INSERT OR REPLACE INTO bm25_documents
                   (doc_id, source_id, parent_id, source_file, page,
                    content_type, text, metadata_json, file_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc_id,
                    source_id,
                    parent_id,
                    source,
                    page,
                    ctype,
                    doc.page_content,
                    json.dumps(extra_meta),
                    file_hash,
                ),
            )

        self.conn.commit()
        self._index_dirty = True

    def build_index(self) -> None:
        """Explicitly build/rebuild the BM25 index. Call once after all add_documents()."""
        if self._index_dirty:
            self._rebuild_index()
        else:
            logger.debug("BM25 index is clean, skipping rebuild.")

    def search(
        self, query: str, top_k: int = None
    ) -> list[tuple[Document, float]]:
        """BM25 keyword search. Returns list of (Document, score) tuples."""
        top_k = top_k or settings.bm25_top_k

        if not self.bm25:
            logger.warning(
                "BM25 index is empty — was the store rebuilt after restart? "
                "Hybrid retrieval may degrade to vector-only."
            )
            return []

        if self._index_dirty:
            self.build_index()

        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        scored = list(enumerate(scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        results: list[tuple[Document, float]] = []
        for idx, score in top:
            doc_id = self.doc_ids[idx]
            doc = self._load_document_by_id(doc_id)
            if doc:
                results.append((doc, float(score)))

        return results

    def _load_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Load a full Document from SQLite by its doc_id (primary key)."""
        cursor = self.conn.execute(
            "SELECT doc_id, text, source_id, parent_id, source_file, page, "
            "content_type, metadata_json "
            "FROM bm25_documents WHERE doc_id = ?",
            (doc_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        doc_id, text, source_id, parent_id, source, page, ctype, meta_json = row
        metadata: dict = {"doc_id": doc_id}
        if source_id:
            metadata["source_id"] = source_id
        if parent_id:
            metadata["parent_id"] = parent_id
        if source:
            metadata["source_file"] = source
        if page is not None:
            metadata["page"] = page
        if ctype:
            metadata["content_type"] = ctype
        if meta_json:
            try:
                metadata.update(json.loads(meta_json))
            except json.JSONDecodeError:
                pass

        return Document(page_content=text, metadata=metadata)

    def delete_by_source_id(self, source_id: str) -> int:
        """
        Delete all BM25 docs for a given source_id.

        Primary deletion method for re-ingestion.
        source_id is stable (absolute file path), so old versions
        of a file are correctly removed even when content changes.

        Marks index dirty without rebuilding — build_index() at end
        of ingestion batch handles the rebuild.
        """
        cursor = self.conn.execute(
            "DELETE FROM bm25_documents WHERE source_id = ?", (source_id,)
        )
        self.conn.commit()

        self._index_dirty = True
        # Invalidate in-memory structures
        self.doc_ids = []
        self.tokenized_corpus = []
        self.bm25 = None

        return cursor.rowcount

    def delete_by_file_hash(self, file_hash: str) -> int:
        """Delete by file_hash. Kept for backward compatibility."""
        cursor = self.conn.execute(
            "DELETE FROM bm25_documents WHERE file_hash = ?", (file_hash,)
        )
        self.conn.commit()

        self._index_dirty = True
        self.doc_ids = []
        self.tokenized_corpus = []
        self.bm25 = None

        return cursor.rowcount

    def has_source(self, source_id: str) -> bool:
        """True if at least one child chunk exists for this source_id."""
        cursor = self.conn.execute(
            "SELECT 1 FROM bm25_documents WHERE source_id = ? LIMIT 1",
            (source_id,),
        )
        return cursor.fetchone() is not None

    def count(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM bm25_documents")
        return cursor.fetchone()[0]

    def clear(self) -> None:
        self.conn.execute("DELETE FROM bm25_documents")
        self.conn.commit()
        self.doc_ids = []
        self.tokenized_corpus = []
        self.bm25 = None
        self._index_dirty = False

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
