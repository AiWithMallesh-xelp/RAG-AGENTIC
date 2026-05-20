# storage/parent_store.py
"""
SQLite-backed persistent store for parent chunks.

Now includes source_id for stable document identity.
Deletion by source_id ensures old versions are removed
even when file content (and thus file_hash) changes.
"""

import json
import logging
from typing import Optional
from pathlib import Path

from langchain_core.documents import Document

from config import settings

logger = logging.getLogger(__name__)


class ParentDocumentStore:
    """SQLite-backed store for parent chunks. Persists across restarts."""

    def __init__(self, db_path: Optional[str] = None):
        import sqlite3

        self.db_path = db_path or str(settings.db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_table()
        self._cache: dict[str, Document] = {}
        self._cache_loaded = False

    def _ensure_table(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS parent_chunks (
                doc_id TEXT PRIMARY KEY,
                source_id TEXT,
                source_file TEXT,
                page INTEGER,
                content_type TEXT,
                text TEXT NOT NULL,
                metadata_json TEXT,
                file_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Add source_id column if migrating from older schema
        try:
            self.conn.execute("ALTER TABLE parent_chunks ADD COLUMN source_id TEXT")
        except Exception:
            pass  # Column already exists

        # Index for fast deletion by source_id
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_parent_source_id "
            "ON parent_chunks(source_id)"
        )
        self.conn.commit()

    def _load_cache(self):
        if self._cache_loaded:
            return

        cursor = self.conn.execute(
            "SELECT doc_id, text, source_id, source_file, page, "
            "content_type, metadata_json "
            "FROM parent_chunks"
        )
        for doc_id, text, source_id, source, page, ctype, meta_json in cursor:
            metadata: dict = {}
            if source_id:
                metadata["source_id"] = source_id
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
            metadata["doc_id"] = doc_id
            self._cache[doc_id] = Document(
                page_content=text, metadata=metadata
            )

        self._cache_loaded = True
        logger.info(f"Parent store cache: {len(self._cache)} documents loaded")

    def add_documents(self, documents: list[Document]) -> None:
        """Store parent chunks in SQLite."""
        for doc in documents:
            parent_id = doc.metadata.get("doc_id")
            if not parent_id:
                logger.warning("Parent document missing doc_id, skipping.")
                continue

            source_id = doc.metadata.get("source_id", "")
            source = doc.metadata.get("source_file", "")
            page = doc.metadata.get("page")
            ctype = doc.metadata.get("content_type", "parent")
            file_hash = doc.metadata.get("file_hash", "")

            extra_meta = {
                k: v
                for k, v in doc.metadata.items()
                if k
                not in {
                    "doc_id",
                    "source_id",
                    "source_file",
                    "page",
                    "content_type",
                    "file_hash",
                }
                and isinstance(v, (str, int, float, bool))
            }

            self.conn.execute(
                """INSERT OR REPLACE INTO parent_chunks
                   (doc_id, source_id, source_file, page, content_type,
                    text, metadata_json, file_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    parent_id,
                    source_id,
                    source,
                    page,
                    ctype,
                    doc.page_content,
                    json.dumps(extra_meta),
                    file_hash,
                ),
            )
            self._cache[parent_id] = doc

        self.conn.commit()
        logger.info(f"Stored {len(documents)} parent chunks in SQLite")

    def get_parent(self, parent_id: str) -> Optional[Document]:
        self._load_cache()
        return self._cache.get(parent_id)

    def get_parents(self, parent_ids: list[str]) -> list[Document]:
        self._load_cache()
        return [self._cache[pid] for pid in parent_ids if pid in self._cache]

    def list_all_ids(self) -> list[str]:
        self._load_cache()
        return list(self._cache.keys())

    def delete_by_source_id(self, source_id: str) -> int:
        """
        Delete all parent chunks for a given source_id.

        This is the primary deletion method for re-ingestion.
        source_id is stable (absolute file path), so old versions
        of a file are correctly removed even when content changes.
        """
        cursor = self.conn.execute(
            "DELETE FROM parent_chunks WHERE source_id = ?", (source_id,)
        )
        self.conn.commit()
        # Invalidate cache entries
        self._cache = {
            k: v
            for k, v in self._cache.items()
            if v.metadata.get("source_id") != source_id
        }
        return cursor.rowcount

    def delete_by_file_hash(self, file_hash: str) -> int:
        """Delete by file_hash. Kept for backward compatibility."""
        cursor = self.conn.execute(
            "DELETE FROM parent_chunks WHERE file_hash = ?", (file_hash,)
        )
        self.conn.commit()
        self._cache = {
            k: v
            for k, v in self._cache.items()
            if v.metadata.get("file_hash") != file_hash
        }
        return cursor.rowcount

    def get_file_hash(self, source_id: str) -> Optional[str]:
        """
        Get the file_hash for a given source_id.
        Used to check if re-ingestion is needed (same hash = skip).
        """
        cursor = self.conn.execute(
            "SELECT file_hash FROM parent_chunks "
            "WHERE source_id = ? LIMIT 1",
            (source_id,),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def count(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM parent_chunks")
        return cursor.fetchone()[0]

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
