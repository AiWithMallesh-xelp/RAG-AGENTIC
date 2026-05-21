# config.py
import hashlib
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """Central configuration for the Agentic RAG system."""

    # ── API Keys & Endpoints ────────────────────────────────────────
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(
        default=None, alias="QDRANT_API_KEY"
    )

    # ── Models ──────────────────────────────────────────────────────
    llm_model: str = Field(default="gemini-2.5-flash", alias="LLM_MODEL")
    embedding_model: str = Field(
        default="gemini-embedding-001", alias="EMBEDDING_MODEL"
    )
    embedding_dim: int = Field(default=768, alias="EMBEDDING_DIM")
    # Reranker: gemini (API, default) or ollama (local ChatOllama)
    reranker_backend: str = Field(default="gemini", alias="RERANKER_BACKEND")
    reranker_model: str = Field(
        default="gemini-2.5-flash", alias="RERANKER_MODEL"
    )

    # ── Persistence ─────────────────────────────────────────────────
    index_dir: str = Field(default="./index_data", alias="INDEX_DIR")

    # ── Qdrant ──────────────────────────────────────────────────────
    qdrant_collection_base: str = "agentic_rag_child"

    # ── Chunking ────────────────────────────────────────────────────
    parent_chunk_size: int = 1500
    parent_chunk_overlap: int = 200
    child_max_iterations: int = 25
    child_min_text_length: int = 50
    chunker_json_retries: int = 2
    chunker_max_input_chars: int = 12000

    # ── Max-Min semantic chunking (embedding-based, no LLM) ─────────
    use_maxmin_chunking: bool = Field(default=False, alias="USE_MAXMIN_CHUNKING")
    maxmin_hard_threshold: float = Field(default=0.6, alias="MAXMIN_HARD_THRESHOLD")
    maxmin_c: float = Field(default=0.9, alias="MAXMIN_C")
    maxmin_init_constant: float = Field(default=1.5, alias="MAXMIN_INIT_CONSTANT")
    maxmin_max_chunk_chars: int = Field(default=2000, alias="MAXMIN_MAX_CHUNK_CHARS")
    maxmin_embed_batch_size: int = Field(default=64, alias="MAXMIN_EMBED_BATCH_SIZE")

    # ── Retrieval ───────────────────────────────────────────────────
    vector_top_k: int = 10
    bm25_top_k: int = 10
    rerank_top_k: int = 5
    rrf_k: int = 60

    # ── Self-RAG ────────────────────────────────────────────────────
    max_retries: int = 3
    relevance_threshold: float = 0.6
    support_threshold: float = 0.6
    usefulness_min_score: int = 3
    strict_support: bool = True
    # Answer only from uploaded docs; refuse general knowledge (e.g. "Who are you?")
    strict_document_only: bool = Field(
        default=True, alias="STRICT_DOCUMENT_ONLY"
    )

    # ── VLM OCR ────────────────────────────────────────────────────
    vlm_enabled: bool = Field(default=True, alias="VLM_ENABLED")
    vlm_image_dpi: int = 200
    vlm_max_images_per_page: int = 5
    # Pages with at least this many extracted chars skip all VLM (text-native PDFs).
    vlm_text_density_threshold: int = Field(
        default=500, alias="VLM_TEXT_DENSITY_THRESHOLD"
    )
    vlm_rate_limit_retries: int = 5

    # ── DOCX ────────────────────────────────────────────────────────
    docx_chars_per_page: int = 3000

    # ── Flask UI ────────────────────────────────────────────────────
    frontend: bool = Field(default=False, alias="FRONTEND")
    flask_host: str = Field(default="127.0.0.1", alias="FLASK_HOST")
    flask_port: int = Field(default=5000, alias="FLASK_PORT")

    # ── Derived Paths ──────────────────────────────────────────────

    @property
    def db_path(self) -> Path:
        return Path(self.index_dir) / "agentic_rag.db"

    def ensure_index_dir(self) -> Path:
        """Create index directory if it doesn't exist. Returns the path."""
        p = Path(self.index_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def qdrant_collection_name(self) -> str:
        """
        Derive a Qdrant collection name from the index directory path.
        Different index_dir values produce different collection names.
        """
        resolved = str(Path(self.index_dir).resolve())
        dir_hash = hashlib.md5(resolved.encode()).hexdigest()[:8]
        return f"{self.qdrant_collection_base}_{dir_hash}"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


settings = Settings()
