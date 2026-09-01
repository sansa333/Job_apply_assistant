import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import settings
from app.knowledge.catalog import JobCatalog
from app.knowledge.ingestion import JobKnowledgeIngestion
from app.knowledge.models import NormalizedJob
from app.knowledge.profiles import CandidateProfileStore
from app.schemas import FitRequest
from app.services.application_service import ApplicationService
from app.services.application_service import get_application_service
from app.main import app


class _FakeLlm:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return type("Response", (), {"content": "evidence-based fit report"})()


class JobScopedFitTests(unittest.TestCase):
    def test_fit_api_keeps_evidence_payload(self) -> None:
        class FakeService:
            def analyze_scoped_fit(self, req):
                return {
                    "status": "ok",
                    "job_id": "job_1",
                    "fit_report": "report",
                    "job_evidence": [{"job_id": "job_1", "content": "requirement"}],
                    "candidate_evidence": [{"candidate_id": "current", "content": "experience"}],
                    "historical_notice": "history",
                }

        app.dependency_overrides[get_application_service] = lambda: FakeService()
        try:
            with TestClient(app) as client:
                response = client.post(
                    "/api/fit",
                    json={"candidate_id": "current", "company_name": "Acme", "job_title": "RAG Engineer"},
                )
        finally:
            app.dependency_overrides.clear()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["job_evidence"][0]["job_id"], "job_1")

    def test_missing_job_returns_upload_guidance_without_llm_call(self) -> None:
        service = object.__new__(ApplicationService)
        service.llm = _FakeLlm()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(settings, "job_catalog_path", root / "catalog.sqlite3"),
                patch.object(settings, "source_corpus_dir", root / "corpus"),
                patch.object(settings, "vector_db_dir", root / "vectors"),
                patch.object(settings, "embedding_backend", "hash"),
                patch.object(settings, "enable_reranker", False),
            ):
                result = service.analyze_scoped_fit(
                    FitRequest(candidate_id="current", company_name="Missing", job_title="Engineer", jd_text="")
                )

        self.assertEqual(result["status"], "job_not_found")
        self.assertEqual(service.llm.calls, 0)
        self.assertEqual(result["upload_action"], "/api/jobs/upload")

    def test_exact_match_contains_only_selected_job_and_candidate_evidence(self) -> None:
        service = object.__new__(ApplicationService)
        service.llm = _FakeLlm()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(settings, "job_catalog_path", root / "catalog.sqlite3"),
                patch.object(settings, "source_corpus_dir", root / "corpus"),
                patch.object(settings, "vector_db_dir", root / "vectors"),
                patch.object(settings, "embedding_backend", "hash"),
                patch.object(settings, "enable_reranker", False),
            ):
                catalog = JobCatalog(settings.job_catalog_path)
                jobs = JobKnowledgeIngestion(
                    catalog=catalog,
                    source_corpus_dir=settings.source_corpus_dir,
                    vector_db_dir=settings.vector_db_dir,
                    collection_name=settings.job_collection_name,
                )
                target = jobs.ingest(NormalizedJob("Acme", "RAG Engineer", "Need Python Chroma RAG skills.", None, "open_source", "test", "acme.md", None, "en"))
                jobs.ingest(NormalizedJob("Other", "Engineer", "Need irrelevant Spark skills.", None, "open_source", "test", "other.md", None, "en"))
                jobs.close()
                profiles = CandidateProfileStore(
                    settings.source_corpus_dir,
                    settings.vector_db_dir,
                    collection_name=settings.candidate_collection_name,
                )
                profiles.ingest_text("current", "I built Python Chroma RAG applications.", "current.md")
                profiles.ingest_text("other", "I built Spark pipelines.", "other.md")
                profiles.close()

                result = service.analyze_scoped_fit(
                    FitRequest(candidate_id="current", company_name="Acme", job_title="RAG Engineer", jd_text="")
                )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["job_id"], target.record.job_id)
        self.assertEqual(service.llm.calls, 1)
        self.assertTrue(all(item["job_id"] == target.record.job_id for item in result["job_evidence"]))
        self.assertTrue(all(item["candidate_id"] == "current" for item in result["candidate_evidence"]))

    def test_resume_text_is_usable_without_candidate_id(self) -> None:
        service = object.__new__(ApplicationService)
        service.llm = _FakeLlm()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(settings, "job_catalog_path", root / "catalog.sqlite3"),
                patch.object(settings, "source_corpus_dir", root / "corpus"),
                patch.object(settings, "vector_db_dir", root / "vectors"),
                patch.object(settings, "embedding_backend", "hash"),
                patch.object(settings, "enable_reranker", False),
            ):
                catalog = JobCatalog(settings.job_catalog_path)
                jobs = JobKnowledgeIngestion(
                    catalog=catalog,
                    source_corpus_dir=settings.source_corpus_dir,
                    vector_db_dir=settings.vector_db_dir,
                    collection_name=settings.job_collection_name,
                )
                jobs.ingest(NormalizedJob("Acme", "RAG Engineer", "Need Python RAG skills.", None, "open_source", "test", "acme.md", None, "en"))
                jobs.close()

                result = service.analyze_scoped_fit(
                    FitRequest(
                        company_name="Acme",
                        job_title="RAG Engineer",
                        resume_text="I built a Python RAG system.",
                    )
                )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["evidence_level"], "user_provided")
        self.assertEqual(service.llm.calls, 1)

    def test_missing_candidate_evidence_returns_next_action_without_llm_call(self) -> None:
        service = object.__new__(ApplicationService)
        service.llm = _FakeLlm()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(settings, "job_catalog_path", root / "catalog.sqlite3"),
                patch.object(settings, "source_corpus_dir", root / "corpus"),
                patch.object(settings, "vector_db_dir", root / "vectors"),
                patch.object(settings, "embedding_backend", "hash"),
                patch.object(settings, "enable_reranker", False),
            ):
                catalog = JobCatalog(settings.job_catalog_path)
                jobs = JobKnowledgeIngestion(
                    catalog=catalog,
                    source_corpus_dir=settings.source_corpus_dir,
                    vector_db_dir=settings.vector_db_dir,
                    collection_name=settings.job_collection_name,
                )
                jobs.ingest(NormalizedJob("Acme", "RAG Engineer", "Need Python RAG skills.", None, "open_source", "test", "acme.md", None, "en"))
                jobs.close()

                result = service.analyze_scoped_fit(FitRequest(company_name="Acme", job_title="RAG Engineer"))

        self.assertEqual(result["status"], "needs_candidate_evidence")
        self.assertEqual(result["next_action"], "provide_candidate_id_or_resume_text")
        self.assertEqual(service.llm.calls, 0)


if __name__ == "__main__":
    unittest.main()
