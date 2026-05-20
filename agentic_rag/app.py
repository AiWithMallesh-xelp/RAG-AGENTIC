# app.py
"""
Flask web UI for Agentic RAG.

Run when FRONTEND=true in .env:
    python app.py
"""

from __future__ import annotations

import logging
import socket
import sys
import threading
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from config import settings
from pipeline import AgenticRAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agentic_rag.log"),
    ],
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(__file__).parent / "uploads"
ALLOWED_EXTENSIONS = {".pdf", ".docx"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB

_pipeline_cache: dict[str, AgenticRAGPipeline] = {}
_cache_lock = threading.Lock()

_ingest_jobs: dict[str, dict] = {}
_ingest_jobs_lock = threading.Lock()


def _index_key(index_dir: Optional[str] = None) -> str:
    return str(Path(index_dir or settings.index_dir).resolve())


def get_pipeline(index_dir: Optional[str] = None) -> AgenticRAGPipeline:
    """Reuse pipeline instance per index directory (expensive to rebuild)."""
    key = _index_key(index_dir)
    with _cache_lock:
        if key not in _pipeline_cache:
            _pipeline_cache[key] = AgenticRAGPipeline(index_dir=index_dir)
        return _pipeline_cache[key]


def invalidate_pipeline(index_dir: Optional[str] = None) -> None:
    key = _index_key(index_dir)
    with _cache_lock:
        pipe = _pipeline_cache.pop(key, None)
    if pipe:
        try:
            pipe.close()
        except Exception:
            pass


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return str(value)


def serialize_query_result(result: dict) -> dict:
    sources = []
    for doc in result.get("documents") or []:
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        sources.append(
            {
                "source_file": meta.get("source_file", "unknown"),
                "page": meta.get("page"),
                "content_type": meta.get("content_type"),
                "snippet": (doc.page_content or "")[:400],
            }
        )

    return {
        "answer": result.get("answer", ""),
        "is_successful": result.get("is_successful", False),
        "retrieval_decision": _json_safe(result.get("retrieval_decision")),
        "relevance_grade": _json_safe(result.get("relevance_grade")),
        "support_grade": _json_safe(result.get("support_grade")),
        "usefulness_score": result.get("usefulness_score"),
        "num_retries": result.get("num_retries", 0),
        "transformations": result.get("transformations") or [],
        "sources": sources,
    }


@app.route("/")
def index():
    if not settings.frontend:
        return (
            jsonify(
                {
                    "error": "Web UI disabled. Set FRONTEND=true in .env and restart."
                }
            ),
            403,
        )
    return render_template(
        "index.html",
        index_dir=settings.index_dir,
        collection_name=settings.qdrant_collection_name(),
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )


@app.route("/api/health")
def api_health():
    """Health check for monitors and frontend polling."""
    return jsonify(
        {
            "ok": True,
            "frontend": settings.frontend,
            "model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "index_dir": settings.index_dir,
        }
    )


@app.route("/api/status")
def api_status():
    index_dir = request.args.get("index_dir") or settings.index_dir
    try:
        pipeline = get_pipeline(index_dir)
        status = pipeline.status()
        return jsonify({"ok": True, "status": status})
    except Exception as e:
        logger.exception("Status failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"ok": False, "error": "Question is required"}), 400

    index_dir = data.get("index_dir") or settings.index_dir
    try:
        pipeline = get_pipeline(index_dir)
        result = pipeline.query(question)
        return jsonify({"ok": True, "result": serialize_query_result(result)})
    except Exception as e:
        logger.exception("Query failed")
        return jsonify({"ok": False, "error": str(e)}), 500


def _run_ingest_job(job_id: str, saved_paths: list[Path], index_dir: str) -> None:
    try:
        pipeline = get_pipeline(index_dir)
        stats = pipeline.ingest(saved_paths)
        status = pipeline.status()
        errors = stats.get("errors") or []
        processed = stats.get("files_processed", 0)

        with _ingest_jobs_lock:
            if processed == 0 and saved_paths and not stats.get("files_skipped"):
                _ingest_jobs[job_id] = {
                    "status": "failed",
                    "error": (
                        "; ".join(errors)
                        if errors
                        else "No files were indexed. Check server logs."
                    ),
                    "stats": stats,
                    "index_status": status,
                }
            else:
                _ingest_jobs[job_id] = {
                    "status": "completed",
                    "stats": stats,
                    "index_status": status,
                }
    except Exception as e:
        logger.exception("Background ingest failed")
        with _ingest_jobs_lock:
            _ingest_jobs[job_id] = {"status": "failed", "error": str(e), "stats": {}}


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    """Start ingest in background; poll GET /api/ingest/<job_id> for completion."""
    index_dir = request.form.get("index_dir") or settings.index_dir
    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"ok": False, "error": "No files uploaded"}), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for f in files:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify(
                {
                    "ok": False,
                    "error": f"Unsupported type: {ext}. Use PDF or DOCX.",
                }
            ),
            400
        safe_name = secure_filename(f.filename)
        dest = UPLOAD_DIR / safe_name
        f.save(dest)
        saved_paths.append(dest)

    if not saved_paths:
        return jsonify({"ok": False, "error": "No valid files saved"}), 400

    job_id = str(uuid.uuid4())
    with _ingest_jobs_lock:
        _ingest_jobs[job_id] = {
            "status": "running",
            "files": [p.name for p in saved_paths],
        }

    thread = threading.Thread(
        target=_run_ingest_job,
        args=(job_id, saved_paths, index_dir),
        daemon=True,
    )
    thread.start()

    return jsonify(
        {
            "ok": True,
            "job_id": job_id,
            "message": "Ingestion started. Large PDFs may take several minutes.",
            "files": [p.name for p in saved_paths],
        }
    ), 202


@app.route("/api/ingest/<job_id>")
def api_ingest_status(job_id: str):
    with _ingest_jobs_lock:
        job = _ingest_jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "Unknown job id"}), 404
    return jsonify({"ok": True, "job": job})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    data = request.get_json(silent=True) or {}
    index_dir = data.get("index_dir") or settings.index_dir
    try:
        invalidate_pipeline(index_dir)
        pipeline = get_pipeline(index_dir)
        pipeline.reset()
        status = pipeline.status()
        return jsonify({"ok": True, "status": status, "message": "Index reset."})
    except Exception as e:
        logger.exception("Reset failed")
        return jsonify({"ok": False, "error": str(e)}), 500


def _find_free_port(host: str, start_port: int, max_tries: int = 10) -> int:
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    return start_port


def run_app():
    if not settings.frontend:
        print(
            "FRONTEND is not enabled. Set FRONTEND=true in .env to run the web UI."
        )
        sys.exit(1)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    port = _find_free_port(settings.flask_host, settings.flask_port)
    if port != settings.flask_port:
        logger.warning(
            f"Port {settings.flask_port} in use; using {port} instead."
        )

    print(f"\n  Agentic RAG Web UI")
    print(f"  http://{settings.flask_host}:{port}")
    print(f"  LLM: {settings.llm_model}")
    print(f"  Embedding: {settings.embedding_model} (dim={settings.embedding_dim})")
    print(f"  Index: {settings.index_dir}")
    print(f"  Collection: {settings.qdrant_collection_name()}\n")

    app.run(
        host=settings.flask_host,
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


if __name__ == "__main__":
    run_app()
