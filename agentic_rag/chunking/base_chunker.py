# chunking/base_chunker.py
"""Shared parent/table chunking utilities for LLM and Max-Min strategies."""

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from utils import make_deterministic_id
from ingestion.pdf_parser import ParsedPage

logger = logging.getLogger(__name__)


@dataclass
class ChunkRelation:
    parent_id: str
    child_id: str
    parent_text: str
    child_text: str
    child_metadata: dict = field(default_factory=dict)


@dataclass
class ChunkedDocument:
    parent_chunks: list[Document]
    child_chunks: list[Document]
    relations: list[ChunkRelation]


class BaseChunker(ABC):
    """Two-level chunker: parent split + strategy-specific child extraction."""

    def __init__(self):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.parent_chunk_size,
            chunk_overlap=settings.parent_chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        )

    def chunk_document(
        self,
        pages: list[ParsedPage],
        file_hash: str,
        source_id: str,
    ) -> ChunkedDocument:
        all_parent_chunks: list[Document] = []
        all_child_chunks: list[Document] = []
        all_relations: list[ChunkRelation] = []

        for page in pages:
            full_text = self._merge_page_content(page)
            if not full_text.strip():
                continue

            parent_docs = self.parent_splitter.create_documents(
                texts=[full_text],
                metadatas=[
                    {
                        "source_file": page.source_file,
                        "page": page.page_number,
                        "content_type": "parent",
                        "file_hash": file_hash,
                        "source_id": source_id,
                    }
                ],
            )

            for parent_idx, parent_doc in enumerate(parent_docs):
                parent_id = make_deterministic_id(
                    source_id, page.page_number, "parent", parent_idx
                )
                parent_doc.metadata["doc_id"] = parent_id

                table_blocks, text_parts = self._split_tables_and_text(
                    parent_doc.page_content
                )

                for text_part_idx, text_part in enumerate(text_parts):
                    if not text_part.strip() or len(
                        text_part.strip()
                    ) < settings.child_min_text_length:
                        continue

                    child_results = self._chunk_text_part(
                        text=text_part,
                        source_name=page.source_file,
                        page_num=page.page_number,
                        parent_id=parent_id,
                        file_hash=file_hash,
                        source_id=source_id,
                        parent_idx=parent_idx,
                        text_part_idx=text_part_idx,
                    )

                    for child_idx, (child_doc, extra_meta) in enumerate(
                        child_results
                    ):
                        child_id, meta = self._build_child_metadata(
                            child_doc=child_doc,
                            extra_meta=extra_meta,
                            source_id=source_id,
                            page_num=page.page_number,
                            parent_idx=parent_idx,
                            text_part_idx=text_part_idx,
                            child_idx=child_idx,
                            parent_id=parent_id,
                            file_hash=file_hash,
                            source_id_val=source_id,
                            content_type=self.child_content_type(),
                        )
                        child_doc.metadata = meta

                        all_child_chunks.append(child_doc)
                        all_relations.append(
                            ChunkRelation(
                                parent_id=parent_id,
                                child_id=child_id,
                                parent_text=parent_doc.page_content,
                                child_text=child_doc.page_content,
                                child_metadata=child_doc.metadata,
                            )
                        )

                for table_idx, (table_block, table_context) in enumerate(
                    table_blocks
                ):
                    table_chunks = self._chunk_table(
                        table_block,
                        table_context,
                        page.source_file,
                        page.page_number,
                        parent_id,
                        file_hash,
                        source_id,
                        parent_idx,
                        table_idx,
                    )
                    all_child_chunks.extend(table_chunks)

                all_parent_chunks.append(parent_doc)

        self._log_chunking_complete(all_parent_chunks, all_child_chunks)
        return ChunkedDocument(
            parent_chunks=all_parent_chunks,
            child_chunks=all_child_chunks,
            relations=all_relations,
        )

    @abstractmethod
    def _chunk_text_part(
        self,
        text: str,
        source_name: str,
        page_num: int,
        parent_id: str,
        file_hash: str,
        source_id: str,
        parent_idx: int,
        text_part_idx: int,
    ) -> list[tuple[Document, dict]]:
        """Return (Document, extra_metadata) pairs for one prose text block."""

    @abstractmethod
    def child_content_type(self) -> str:
        """content_type value for semantic child chunks."""

    @abstractmethod
    def _log_chunking_complete(
        self, parents: list[Document], children: list[Document]
    ) -> None:
        pass

    @staticmethod
    def _build_child_metadata(
        child_doc: Document,
        extra_meta: dict,
        source_id: str,
        page_num: int,
        parent_idx: int,
        text_part_idx: int,
        child_idx: int,
        parent_id: str,
        file_hash: str,
        source_id_val: str,
        content_type: str,
    ) -> tuple[str, dict]:
        child_id = make_deterministic_id(
            source_id,
            page_num,
            "child",
            parent_idx,
            text_part_idx,
            child_idx,
        )
        meta = dict(child_doc.metadata)
        meta.update(
            {
                "doc_id": child_id,
                "parent_id": parent_id,
                "content_type": content_type,
                "file_hash": file_hash,
                "source_id": source_id_val,
            }
        )
        meta.update(extra_meta)
        return child_id, meta

    @staticmethod
    def _merge_page_content(page: ParsedPage) -> str:
        parts = []
        if page.text_content.strip():
            parts.append(page.text_content)
        if page.vlm_content.strip():
            parts.append(f"\n[VLM-OCR]\n{page.vlm_content}\n[/VLM-OCR]")
        return "\n\n".join(parts)

    @staticmethod
    def _split_tables_and_text(
        text: str,
    ) -> tuple[list[tuple[str, str]], list[str]]:
        table_blocks: list[tuple[str, str]] = []
        text_parts: list[str] = []

        table_pattern = re.compile(
            r"\[TABLE[^\]]*\]\n(.*?)\n\[/TABLE\]", re.DOTALL
        )

        last_end = 0
        for match in table_pattern.finditer(text):
            before = text[last_end : match.start()].strip()
            if before:
                text_parts.append(before)

            context_start = max(0, match.start() - 200)
            table_context = text[context_start : match.start()].strip()
            table_blocks.append((match.group(1), table_context))
            last_end = match.end()

        remaining = text[last_end:].strip()
        if remaining:
            text_parts.append(remaining)

        if not table_blocks and not text_parts:
            text_parts.append(text)

        return table_blocks, text_parts

    def _chunk_table(
        self,
        table_text: str,
        table_context: str,
        source_name: str,
        page_num: int,
        parent_id: str,
        file_hash: str = "",
        source_id: str = "",
        parent_idx: int = 0,
        table_idx: int = 0,
    ) -> list[Document]:
        lines = table_text.strip().split("\n")
        if len(lines) < 2:
            return []

        header_line = lines[0]
        separator_idx = -1
        for i, line in enumerate(lines[1:], start=1):
            if set(line.replace("|", "").strip()) <= {"-", ":", " "}:
                separator_idx = i
                break

        if separator_idx == -1:
            header = header_line
            data_lines = lines[1:]
        else:
            header = header_line
            data_lines = lines[separator_idx + 1 :]

        num_cols = len([c for c in header.strip("|").split("|") if c.strip()])
        separator = "| " + " | ".join(["---"] * num_cols) + " |"

        rows_per_chunk = 3
        chunks: list[Document] = []

        for i in range(0, len(data_lines), rows_per_chunk):
            chunk_rows = data_lines[i : i + rows_per_chunk]
            chunk_text = f"{header}\n{separator}\n" + "\n".join(chunk_rows)

            if table_context:
                chunk_text = f"Context: {table_context}\n\n{chunk_text}"

            child_id = make_deterministic_id(
                source_id, page_num, "table", parent_idx, table_idx, i
            )
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "doc_id": child_id,
                    "parent_id": parent_id,
                    "source_file": source_name,
                    "page": page_num,
                    "content_type": "table_row_chunk",
                    "table_row_start": i,
                    "table_row_end": min(i + rows_per_chunk, len(data_lines)),
                    "file_hash": file_hash,
                    "source_id": source_id,
                },
            )
            chunks.append(doc)

        return chunks
