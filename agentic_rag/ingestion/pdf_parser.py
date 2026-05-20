# ingestion/pdf_parser.py
"""
PyMuPDF + Gemini VLM for PDF ingestion (google.genai SDK).
"""

import base64
import io
import logging
from pathlib import Path
from dataclasses import dataclass

import fitz
from PIL import Image

from config import settings
from gemini_vlm import generate_with_retry
from utils import compute_file_hash

logger = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    """Represents a single parsed page with text + VLM enrichment."""
    page_number: int
    text_content: str
    vlm_content: str
    images: list[str]
    source_file: str


VLM_OCR_PROMPT = """You are an expert document OCR and analysis agent.
Analyze this document page image and extract ALL information:

1. **OCR Text**: Extract all visible text verbatim, preserving structure.
2. **Tables**: Describe tables in markdown format with all rows/columns.
3. **Figures/Charts**: Describe any figures, charts, or diagrams in detail.
4. **Layout**: Note headers, footers, sidebars, and callout boxes.

Rules:
- Do NOT summarize. Extract verbatim.
- If text is unclear, make your best attempt and mark with [?].
- Preserve the reading order and document hierarchy.
- For mathematical formulas, use LaTeX notation.
"""

IMAGE_DESC_PROMPT = """Describe this image from a document in detail.
If it contains a chart or graph, describe all data points and labels.
If it's a diagram, describe all components and connections."""


class PDFParser:
    """PDF parser combining PyMuPDF text extraction with Gemini VLM OCR."""

    @staticmethod
    def _document_is_text_native(page_densities: list[int]) -> bool:
        """
        True when the PDF is mostly machine-readable text (PyMuPDF extraction).

        Avoids VLM on short cover pages in otherwise text-native documents.
        """
        if not page_densities:
            return False
        total = sum(page_densities)
        if total < settings.vlm_text_density_threshold:
            return False
        substantive = sum(1 for d in page_densities if d >= 100)
        return substantive / len(page_densities) >= 0.4

    @staticmethod
    def _page_needs_vlm(text_density: int, doc_text_native: bool) -> bool:
        """Whether to call Gemini VLM for this page."""
        if not settings.vlm_enabled or doc_text_native:
            return False
        return text_density < settings.vlm_text_density_threshold

    def parse(self, file_path: str | Path) -> tuple[list[ParsedPage], str]:
        """Parse a PDF file into ParsedPage objects. Returns: (pages, file_hash)"""
        file_path = Path(file_path)
        file_hash = compute_file_hash(file_path)
        pages = []

        doc = fitz.open(str(file_path))
        try:
            page_densities: list[int] = []
            for page_num in range(len(doc)):
                page_densities.append(
                    len(self._extract_page_text(doc[page_num]).strip())
                )

            doc_text_native = self._document_is_text_native(page_densities)
            if doc_text_native:
                logger.info(
                    f"{file_path.name}: text-native PDF "
                    f"({sum(page_densities)} chars extracted), skipping VLM"
                )

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_data = self._parse_page(
                    page,
                    page_num,
                    file_path.name,
                    doc_text_native=doc_text_native,
                )
                pages.append(page_data)
        finally:
            doc.close()

        logger.info(f"Parsed {len(pages)} pages from {file_path.name}")
        return pages, file_hash

    def _extract_page_text(self, page) -> str:
        """PyMuPDF text + table markdown for a single page."""
        text_content = page.get_text("text")
        table_texts = []
        for table in page.find_tables():
            try:
                df = table.to_pandas()
                table_texts.append(f"\n[TABLE]\n{df.to_markdown()}\n[/TABLE]\n")
            except Exception:
                pass
        if table_texts:
            text_content += "\n" + "\n".join(table_texts)
        return text_content

    def _parse_page(
        self,
        page,
        page_num: int,
        source_name: str,
        *,
        doc_text_native: bool = False,
    ) -> ParsedPage:
        """Parse a single PDF page."""
        text_content = self._extract_page_text(page)

        page_images: list[str] = []
        vlm_content = ""
        text_density = len(text_content.strip())
        needs_vlm = self._page_needs_vlm(text_density, doc_text_native)

        if needs_vlm:
            page_images = self._render_page_images(page, page_num)
            if page_images:
                vlm_content = generate_with_retry(
                    VLM_OCR_PROMPT,
                    [self._b64_to_pil(page_images[0])],
                )
            embedded_descriptions = self._describe_embedded_images(page, page_num)
            if embedded_descriptions:
                vlm_content = (vlm_content + "\n\n" + embedded_descriptions).strip()
        elif settings.vlm_enabled and not doc_text_native:
            logger.debug(
                f"Page {page_num + 1}: sparse text ({text_density} chars), "
                f"using VLM"
            )
        elif not settings.vlm_enabled:
            logger.debug(f"Page {page_num + 1}: VLM disabled globally")

        return ParsedPage(
            page_number=page_num + 1,
            text_content=text_content.strip(),
            vlm_content=vlm_content.strip(),
            images=page_images,
            source_file=source_name,
        )

    def _render_page_images(self, page, page_num: int) -> list[str]:
        try:
            mat = fitz.Matrix(
                settings.vlm_image_dpi / 72, settings.vlm_image_dpi / 72
            )
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            return [base64.b64encode(img_bytes).decode("utf-8")]
        except Exception as e:
            logger.warning(f"Failed to render page {page_num}: {e}")
            return []

    @staticmethod
    def _b64_to_pil(image_b64: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(image_b64)))

    def _describe_embedded_images(self, page, page_num: int) -> str:
        descriptions = []
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(
            image_list[: settings.vlm_max_images_per_page]
        ):
            try:
                xref = img_info[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))

                desc = generate_with_retry(IMAGE_DESC_PROMPT, [image])
                if desc:
                    descriptions.append(f"[IMAGE {img_idx + 1}]: {desc}")
            except Exception as e:
                logger.warning(
                    f"Failed to describe image {img_idx} on page {page_num}: {e}"
                )

        return "\n\n".join(descriptions)
