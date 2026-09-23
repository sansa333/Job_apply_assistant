"""Cheap read-only domestic readiness; never load models or mutate catalog state."""
from __future__ import annotations

import json
import sqlite3

from app.chunking import JD_CHUNK_SCHEMA_VERSION
from app.config import settings


def domestic_readiness() -> dict:
    checks = {
        "data_dir": settings.data_dir.is_dir(),
        "outputs_dir": settings.outputs_dir.is_dir(),
        "llm_credentials_configured": bool(settings.zai_api_key or settings.zhipu_api_key
                                           or settings.zhipuai_api_key or settings.openai_api_key),
        "production_auth": settings.environment.lower() != "production" or len((settings.api_token or "").strip()) >= 32,
        "semantic_embeddings": settings.embedding_backend == "huggingface",
    }
    open_jobs = 0
    records = []
    try:
        with sqlite3.connect(settings.job_catalog_path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            records = connection.execute("SELECT job_id, content_hash FROM job_records WHERE is_domestic=1 AND status='open'").fetchall()
            open_jobs = len(records)
        checks["catalog_readable"] = True
    except sqlite3.Error:
        checks["catalog_readable"] = False
    checks["open_domestic_jobs"] = open_jobs > 0
    index_dir = settings.vector_db_dir / settings.job_collection_name
    try:
        manifest = json.loads((index_dir / "collection_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    checks["index_present"] = (index_dir / "chroma.sqlite3").is_file()
    checks["index_compatible"] = all((
        manifest.get("state") == "ready",
        manifest.get("chunk_schema_version") == JD_CHUNK_SCHEMA_VERSION,
        manifest.get("embedding_backend") == settings.embedding_backend,
        manifest.get("embedding_model") == settings.hf_embedding_model,
        manifest.get("max_length") == settings.embedding_max_length,
        manifest.get("normalize_embeddings") == settings.embedding_normalize,
        manifest.get("use_fp16") == settings.embedding_use_fp16,
    ))
    checks["index_nonempty"] = manifest.get("document_count", 0) > 0
    checks["index_covers_open_jobs"] = bool(records) and all(
        manifest.get("indexed_job_hashes", {}).get(job_id) == content_hash for job_id, content_hash in records
    )
    return {"status": "ready" if all(checks.values()) else "not_ready", "checks": checks,
            "open_domestic_jobs": open_jobs,
            "scope": "single-owner domestic deployment; configuration/artifact checks only, not live LLM, freshness or load validation"}
