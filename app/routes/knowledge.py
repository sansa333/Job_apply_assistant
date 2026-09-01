from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.knowledge.catalog import JobCatalog
from app.knowledge.importers import UserUploadAdapter
from app.knowledge.ingestion import JobKnowledgeIngestion
from app.knowledge.ingestion import import_open_source_jobs
from app.knowledge.health import CollectionHealth
from app.knowledge.profiles import CandidateProfileStore
from app.rag import get_embeddings

router = APIRouter(prefix="/api", tags=["knowledge"])


def _ingestion() -> JobKnowledgeIngestion:
    return JobKnowledgeIngestion(
        catalog=JobCatalog(settings.job_catalog_path),
        source_corpus_dir=settings.source_corpus_dir,
        vector_db_dir=settings.vector_db_dir,
        collection_name=settings.job_collection_name,
    )


def _reranker_available_locally(model_name: str) -> bool:
    """Recognize either an explicit local path or a Hugging Face cache entry."""
    if Path(model_name).exists():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(model_name, "config.json"), str)
    except Exception:
        return False


def _record_payload(record: object) -> dict:
    return {
        "job_id": record.job_id,
        "company_name": record.company_name,
        "job_title": record.job_title,
        "location": record.location,
        "language": record.language,
        "source_kind": record.source_kind,
        "source_dataset": record.source_dataset,
        "source_file": record.source_file,
        "source_url": record.source_url,
        "is_user_uploaded": record.is_user_uploaded,
        "created_at": record.created_at,
    }


@router.post("/jobs/upload")
async def upload_job(
    company_name: str = Form(...),
    job_title: str = Form(...),
    jd_text: str = Form(""),
    file: UploadFile | None = File(None),
) -> dict:
    payload = jd_text.strip()
    original_bytes: bytes | None = payload.encode("utf-8") if payload else None
    filename = "pasted_job.md"
    if file is not None:
        file_bytes = await file.read()
        if not payload:
            try:
                payload = file_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise HTTPException(status_code=400, detail="Only UTF-8 JD files are supported here") from exc
        original_bytes = file_bytes
        filename = Path(file.filename or filename).name
    if not payload:
        raise HTTPException(status_code=422, detail="Provide jd_text or a UTF-8 JD file")

    parsed = UserUploadAdapter(
        company_name=company_name,
        job_title=job_title,
        description=payload,
        source_file=filename,
    ).load()
    if not parsed.jobs:
        raise HTTPException(status_code=422, detail="company_name, job_title and JD content are required")
    ingestion = _ingestion()
    try:
        outcome = ingestion.ingest(parsed.jobs[0], original_bytes=original_bytes)
        return {"record": _record_payload(outcome.record), "chunks_added": outcome.chunks_added, "inserted": outcome.inserted}
    finally:
        ingestion.close()


@router.post("/candidates/upload")
async def upload_candidate_profile(
    candidate_id: str = Form(...),
    profile_text: str = Form(""),
    file: UploadFile | None = File(None),
) -> dict:
    text = profile_text.strip()
    filename = "profile.md"
    if file is not None:
        raw = await file.read()
        if not text:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise HTTPException(status_code=400, detail="Only UTF-8 profile files are supported here") from exc
        filename = Path(file.filename or filename).name
    if not text:
        raise HTTPException(status_code=422, detail="Provide profile_text or a UTF-8 profile file")
    profiles = CandidateProfileStore(
        settings.source_corpus_dir,
        settings.vector_db_dir,
        collection_name=settings.candidate_collection_name,
    )
    try:
        chunks = profiles.ingest_text(candidate_id, text, filename)
        return {"candidate_id": candidate_id, "chunks_added": chunks, "collection": settings.candidate_collection_name}
    finally:
        profiles.close()


@router.post("/jobs/import/open-source")
def import_public_jobs(
    csv_path: str | None = None,
    project_markdown_dir: str | None = None,
) -> dict:
    default_csv = settings.source_corpus_dir / "open_source_jobs" / "kyosek_jobs.csv"
    default_project = settings.data_dir / "eval_dataset" / "jds"
    csv = Path(csv_path) if csv_path else default_csv
    project_dir = Path(project_markdown_dir) if project_markdown_dir else default_project
    if not csv.exists() and not project_dir.exists():
        raise HTTPException(status_code=404, detail="No approved local open-source JD corpus was found")
    return import_open_source_jobs(
        csv_path=csv if csv.exists() else None,
        project_markdown_dir=project_dir if project_dir.exists() else None,
        catalog_path=settings.job_catalog_path,
        source_corpus_dir=settings.source_corpus_dir,
        vector_db_dir=settings.vector_db_dir,
    )


@router.get("/jobs/search")
def search_jobs(company_name: str, job_title: str) -> dict:
    records = JobCatalog(settings.job_catalog_path).lookup(company_name, job_title)
    return {"matches": [_record_payload(record) for record in records]}


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    record = JobCatalog(settings.job_catalog_path).get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")
    payload = _record_payload(record)
    payload["description"] = record.description
    return payload


@router.get("/knowledge/health")
def knowledge_health() -> dict:
    embeddings = get_embeddings()
    dimension = int(getattr(embeddings, "dim", 0) or len(embeddings.embed_query("dimension probe")))
    inspector = CollectionHealth(settings.vector_db_dir)
    job = inspector.check(
        settings.job_collection_name,
        embedding_backend=settings.embedding_backend.lower(),
        embedding_dimension=dimension,
    )
    return {
        "collections": [CollectionHealth.payload(job)],
        "reranker": {
            "enabled": settings.enable_reranker,
            "model": settings.reranker_model,
            "available_locally": _reranker_available_locally(settings.reranker_model),
        },
    }
