# retrieval/parent_expander.py
"""
Parent Document Expansion.
Tracks max child score per parent instead of dropping duplicates.
"""

import logging
from langchain_core.documents import Document

from storage.parent_store import ParentDocumentStore

logger = logging.getLogger(__name__)


class ParentExpander:
    """Expands child chunk results to parent document context."""

    def __init__(self, parent_store: ParentDocumentStore):
        self.parent_store = parent_store

    def expand(
        self, child_results: list[tuple[Document, float]]
    ) -> list[Document]:
        """
        Given child chunks with scores, return deduplicated parent documents.
        Tracks the max child score per parent.
        """
        # Collect best score per parent
        parent_best: dict[str, tuple[float, Document]] = {}

        for child_doc, score in child_results:
            parent_id = child_doc.metadata.get("parent_id")
            if not parent_id:
                key = f"__child__{child_doc.metadata.get('doc_id', id(child_doc))}"
                if key not in parent_best or score > parent_best[key][0]:
                    parent_best[key] = (score, child_doc)
                continue

            if parent_id not in parent_best or score > parent_best[parent_id][0]:
                parent_best[parent_id] = (score, child_doc)

        # Expand to parent documents
        expanded: list[Document] = []
        for key, (best_score, representative_child) in parent_best.items():
            if key.startswith("__child__"):
                expanded.append(representative_child)
                continue

            parent_doc = self.parent_store.get_parent(key)
            if parent_doc:
                expanded_doc = Document(
                    page_content=parent_doc.page_content,
                    metadata={
                        **parent_doc.metadata,
                        "matched_child_text": representative_child.page_content[
                            :500
                        ],
                        "child_retrieval_score": best_score,
                        "expansion_type": "parent_expansion",
                    },
                )
                expanded.append(expanded_doc)
            else:
                logger.warning(f"Parent {key} not found, using child chunk.")
                expanded.append(representative_child)

        logger.info(
            f"Parent expansion: {len(child_results)} children → "
            f"{len(expanded)} unique parents"
        )
        return expanded
