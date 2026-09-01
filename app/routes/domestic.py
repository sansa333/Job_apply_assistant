from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app.config import settings
from app.domestic.profile import (
    ingest_pdf_profile,
    load_candidate_profile,
    update_candidate_preferences,
)
from app.domestic.service import DomesticJobService, profile_safe_summary, record_payload
from app.knowledge.catalog import JobCatalog
from app.utils.file_io import safe_filename, validate_identifier


router = APIRouter(prefix="/api/domestic", tags=["domestic-job-search"])

ALLOWED_APPLICATION_STAGES = {
    "saved", "ignored", "planned", "applied", "written_test", "interview",
    "offer", "rejected", "withdrawn",
}


class ApplicationStageRequest(BaseModel):
    candidate_id: str = "current_candidate"
    stage: str = Field(..., description="saved/ignored/planned/applied/written_test/interview/offer/rejected/withdrawn")
    notes: str = ""


class CandidatePreferencesRequest(BaseModel):
    graduation_year: int | None = Field(default=None, ge=1900, le=2100)
    target_roles: list[str] = Field(default_factory=list)
    target_cities: list[str] = Field(default_factory=list)


def _split_preferences(value: str) -> list[str]:
    return [item.strip() for item in value.replace("，", ",").split(",") if item.strip()]


@router.post("/profile/pdf")
async def upload_pdf_resume(
    candidate_id: str = "current_candidate",
    graduation_year: int | None = Query(default=None, ge=1900, le=2100),
    target_roles: str = "",
    target_cities: str = "",
    file: UploadFile = File(...),
) -> dict:
    try:
        candidate_id = validate_identifier(candidate_id, field_name="candidate_id")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    filename = safe_filename(file.filename or "resume.pdf")
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=422, detail="Only PDF resumes are supported")
    target_dir = settings.source_corpus_dir / "candidate_profiles" / candidate_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "resume_source.pdf"
    target.write_bytes(await file.read())
    try:
        result = ingest_pdf_profile(
            candidate_id=candidate_id,
            source_pdf=target,
            source_corpus_dir=settings.source_corpus_dir,
            vector_db_dir=settings.vector_db_dir,
            collection_name=settings.candidate_collection_name,
            graduation_year=graduation_year,
            target_roles=_split_preferences(target_roles) if target_roles.strip() else None,
            target_cities=_split_preferences(target_cities) if target_cities.strip() else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "candidate_id": candidate_id,
        "source_filename": filename,
        "managed_pdf": str(result.managed_pdf_path),
        "page_count": result.page_count,
        "text_length": result.text_length,
        "chunks_added": result.chunks_added,
        "profile": profile_safe_summary(result.profile),
        "resume_content_policy": "read_only_no_rewrite",
    }


@router.get("/profile/{candidate_id}")
def get_profile(candidate_id: str) -> dict:
    profile = load_candidate_profile(settings.source_corpus_dir, candidate_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="candidate profile not found")
    return profile_safe_summary(profile) or {}


@router.put("/profile/{candidate_id}/preferences")
def save_candidate_preferences(
    candidate_id: str, request: CandidatePreferencesRequest
) -> dict:
    try:
        profile = update_candidate_preferences(
            settings.source_corpus_dir,
            candidate_id,
            graduation_year=request.graduation_year,
            target_roles=request.target_roles,
            target_cities=request.target_cities,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"candidate_id": candidate_id, "profile": profile_safe_summary(profile)}


@router.get("/sources")
def list_sources() -> dict:
    service = DomesticJobService()
    return {
        "sources": service.catalog.list_sources(),
        "policy": {
            "domestic_only": True,
            "official_public_sources_only": True,
            "restricted_aggregators": ["BOSS直聘", "智联招聘", "猎聘", "前程无忧"],
            "restricted_aggregator_mode": "manual_user_import_only",
        },
    }


@router.get("/stats")
def domestic_stats() -> dict:
    return DomesticJobService().stats()


@router.post("/sources/sync")
def sync_all_sources(build_index: bool = True) -> dict:
    return DomesticJobService().sync_all(build_index=build_index)


@router.post("/sources/refresh")
def refresh_all_sources() -> dict:
    return DomesticJobService().refresh_all()


@router.post("/sources/{source_id}/sync")
def sync_source(source_id: str, build_index: bool = True) -> dict:
    try:
        return DomesticJobService().sync_source(source_id, build_index=build_index)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sources/runs")
def crawl_runs(limit: int = Query(default=50, ge=1, le=200)) -> dict:
    return {"runs": JobCatalog(settings.job_catalog_path).list_crawl_runs(limit)}


@router.post("/index/rebuild")
def rebuild_domestic_index() -> dict:
    return DomesticJobService().rebuild_domestic_index()


@router.get("/jobs/search")
def search_domestic_jobs(
    keyword: str = "",
    company_name: str = "",
    location: str = "",
    recruitment_type: str = "",
    graduation_year: int | None = None,
    status: str = "open",
    candidate_id: str = "current_candidate",
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict:
    return DomesticJobService().search(
        keyword=keyword,
        company_name=company_name,
        location=location,
        recruitment_type=recruitment_type,
        graduation_year=graduation_year,
        status=status,
        candidate_id=candidate_id,
        limit=limit,
        offset=offset,
    )


@router.get("/jobs/{job_id}")
def get_domestic_job(job_id: str) -> dict:
    record = JobCatalog(settings.job_catalog_path).get(job_id)
    if record is None or not record.is_domestic:
        raise HTTPException(status_code=404, detail="domestic job not found")
    return record_payload(record, include_description=True)


@router.post("/jobs/{job_id}/fit")
def analyze_domestic_job(job_id: str, candidate_id: str = "current_candidate") -> dict:
    try:
        return DomesticJobService().analyze_job(job_id=job_id, candidate_id=candidate_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/jobs/{job_id}/application")
def update_application(job_id: str, request: ApplicationStageRequest) -> dict:
    if request.stage not in ALLOWED_APPLICATION_STAGES:
        raise HTTPException(status_code=422, detail="invalid application stage")
    try:
        record = JobCatalog(settings.job_catalog_path).set_application_stage(
            candidate_id=request.candidate_id,
            job_id=job_id,
            stage=request.stage,
            notes=request.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"application": record}


@router.get("/applications/{candidate_id}")
def list_applications(candidate_id: str) -> dict:
    rows = JobCatalog(settings.job_catalog_path).list_applications(candidate_id)
    return {"candidate_id": candidate_id, "applications": rows}
