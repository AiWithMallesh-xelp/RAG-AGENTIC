# ingestion/docx_parser.py
"""python-docx + Gemini VLM for DOCX ingestion."""

import io
import logging
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.table import Table as DocxTable
from PIL import Image

from config import settings
from gemini_vlm import generate_with_retry
from utils import compute_file_hash
from .pdf_parser import ParsedPage, PDFParser

logger = logging.getLogger(__name__)


def _iter_block_items(doc: Document):
    """Iterate over document body elements in document order."""
    parent_elm = doc.element.body
    for child in parent_elm.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield DocxTable(child, doc)


class DocxParser:
    """DOCX parser with in-order body iteration and VLM image description."""

    def parse(self, file_path: str | Path) -> tuple[list[ParsedPage], str]:
        file_path = Path(file_path)

        if file_path.suffix.lower() == ".doc":
            raise ValueError(
                f"Legacy .doc format is not supported: {file_path.name}. "
                f"Convert to .docx using LibreOffice: "
                f"libreoffice --headless --convert-to docx {file_path}"
            )

        file_hash = compute_file_hash(file_path)
        doc = Document(str(file_path))

        pages: list[ParsedPage] = []
        current_text_parts: list[str] = []
        current_char_count = 0
        page_counter = 1
        chars_per_page = settings.docx_chars_per_page

        for block in _iter_block_items(doc):
            if isinstance(block, Paragraph):
                text = block.text.strip()
                if not text:
                    continue

                style_name = block.style.name if block.style else ""
                if "Heading 1" in style_name:
                    current_text_parts.append(f"\n# {text}\n")
                elif "Heading 2" in style_name:
                    current_text_parts.append(f"\n## {text}\n")
                elif "Heading 3" in style_name:
                    current_text_parts.append(f"\n### {text}\n")
                else:
                    current_text_parts.append(text)

                current_char_count += len(text)

                if current_char_count >= chars_per_page:
                    combined = "\n".join(current_text_parts)
                    if combined.strip():
                        pages.append(
                            ParsedPage(
                                page_number=page_counter,
                                text_content=combined,
                                vlm_content="",
                                images=[],
                                source_file=file_path.name,
                            )
                        )
                        page_counter += 1
                    current_text_parts = []
                    current_char_count = 0

            elif isinstance(block, DocxTable):
                table_md = self._table_to_markdown(block)
                current_text_parts.append(f"\n[TABLE]\n{table_md}\n[/TABLE]\n")
                current_char_count += len(table_md)

        if current_text_parts:
            combined = "\n".join(current_text_parts)
            if combined.strip():
                pages.append(
                    ParsedPage(
                        page_number=page_counter,
                        text_content=combined,
                        vlm_content="",
                        images=[],
                        source_file=file_path.name,
                    )
                )

        page_densities = [len(p.text_content.strip()) for p in pages]
        doc_text_native = PDFParser._document_is_text_native(page_densities)
        image_descriptions = ""
        if settings.vlm_enabled and not doc_text_native:
            image_descriptions = self._describe_docx_images(doc)
        elif doc_text_native:
            logger.info(
                f"{file_path.name}: text-native DOCX "
                f"({sum(page_densities)} chars), skipping VLM"
            )

        if image_descriptions and pages:
            pages[-1].vlm_content = (
                pages[-1].vlm_content + "\n\n" + image_descriptions
            ).strip()
        elif image_descriptions and not pages:
            pages.append(
                ParsedPage(
                    page_number=1,
                    text_content="",
                    vlm_content=image_descriptions,
                    images=[],
                    source_file=file_path.name,
                )
            )

        logger.info(f"Parsed DOCX: {file_path.name} -> {len(pages)} page(s)")
        return pages, file_hash

    def _table_to_markdown(self, table: DocxTable) -> str:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            rows.append("| " + " | ".join(cells) + " |")

        if len(rows) > 1:
            num_cols = len(table.rows[0].cells)
            header_sep = "| " + " | ".join(["---"] * num_cols) + " |"
            rows.insert(1, header_sep)

        return "\n".join(rows)

    def _describe_docx_images(self, doc: Document) -> str:
        descriptions = []
        img_idx = 0

        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                try:
                    image_bytes = rel.target_part.blob
                    image = Image.open(io.BytesIO(image_bytes))

                    desc = generate_with_retry(
                        "Describe this image from a document in detail. "
                        "If it contains a chart, diagram, or figure, "
                        "describe all data points, labels, and connections.",
                        [image],
                    )
                    if desc:
                        img_idx += 1
                        descriptions.append(f"[IMAGE {img_idx}]: {desc}")
                except Exception as e:
                    logger.warning(f"Failed to describe DOCX image: {e}")

        return "\n\n".join(descriptions)
