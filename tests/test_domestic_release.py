from __future__ import annotations

import asyncio
import io
import json
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.chunking import JD_CHUNK_SCHEMA_VERSION
from app.config import settings
from app.knowledge.catalog import JobCatalog
from app.knowledge.ingestion import JobKnowledgeIngestion
from app.knowledge.models import NormalizedJob
from app.main import app
from app.readiness import domestic_readiness
from app.security import ApiBoundaryMiddleware, validate_deployment_security
from app.ui_http import backend_request


def test_provider_timeout_returns_actionable_error_without_provider_secrets(monkeypatch):
    import httpx
    from openai import APITimeoutError

    service = Mock()
    service.generate_application_package.side_effect = APITimeoutError(request=httpx.Request("POST", "https://example.com"))
    monkeypatch.setattr("app.routes.domestic.DomesticJobService", lambda: service)
    monkeypatch.setattr(settings, "api_token", None)
    with TestClient(app) as client:
        response = client.post("/api/domestic/jobs/test/package")
    assert response.status_code == 504
    assert response.json()["status"] == "model_timeout"
    assert "example.com" not in response.text


def test_production_auth_fails_closed(monkeypatch):
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "api_token", None)
    with pytest.raises(RuntimeError, match="API_TOKEN"):
        validate_deployment_security()
    monkeypatch.setattr(settings, "api_token", "x" * 40)
    validate_deployment_security()


@pytest.mark.parametrize("headers", [[], [(b"content-length", b"1")]])
def test_actual_stream_limit_cannot_be_bypassed(monkeypatch, headers):
    monkeypatch.setattr(settings, "max_request_bytes", 10)
    downstream = Mock()
    messages = iter([{"type": "http.request", "body": b"123456", "more_body": True},
                     {"type": "http.request", "body": b"789012", "more_body": False}])
    sent = []

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    asyncio.run(ApiBoundaryMiddleware(downstream)({"type": "http", "path": "/health", "headers": headers}, receive, send))
    assert sent[0]["status"] == 413
    downstream.assert_not_called()


def test_ui_does_not_leak_token_to_arbitrary_backend(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://api:8000")
    monkeypatch.setenv("API_TOKEN", "secret")
    request = Mock()
    monkeypatch.setattr("app.ui_http.requests.request", request)
    backend_request("GET", "http://api:8000/api/domestic/stats")
    assert request.call_args.kwargs["headers"]["Authorization"] == "Bearer secret"
    assert request.call_args.kwargs["allow_redirects"] is False
    backend_request("GET", "https://untrusted.example/api/")
    assert "Authorization" not in request.call_args.kwargs["headers"]


def test_readiness_rejects_empty_and_stale_index(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "job_catalog_path", tmp_path / "catalog.sqlite3")
    monkeypatch.setattr(settings, "vector_db_dir", tmp_path / "vectors")
    JobCatalog(settings.job_catalog_path)
    report = domestic_readiness()
    assert report["status"] == "not_ready"
    assert report["checks"]["catalog_readable"]
    assert not report["checks"]["open_domestic_jobs"]
    assert not report["checks"]["index_compatible"]
    directory = settings.vector_db_dir / settings.job_collection_name
    directory.mkdir(parents=True)
    (directory / "collection_manifest.json").write_text(json.dumps({"state": "ready", "chunk_schema_version": JD_CHUNK_SCHEMA_VERSION}), encoding="utf-8")
    assert not domestic_readiness()["checks"]["index_compatible"]


def test_domestic_http_chain_pdf_search_evidence_drafts_history_and_board(tmp_path, monkeypatch):
    for name, path in {"job_catalog_path": tmp_path / "catalog.sqlite3", "source_corpus_dir": tmp_path / "sources",
                       "vector_db_dir": tmp_path / "vectors", "outputs_dir": tmp_path / "outputs"}.items():
        monkeypatch.setattr(settings, name, path)
    monkeypatch.setattr(settings, "api_token", "owner-test-token")
    monkeypatch.setattr(settings, "job_retrieval_strategy", "vector")
    monkeypatch.setattr(settings, "domestic_sync_enabled", False)
    catalog = JobCatalog(settings.job_catalog_path)
    index = JobKnowledgeIngestion(catalog=catalog, source_corpus_dir=settings.source_corpus_dir,
                                  vector_db_dir=settings.vector_db_dir, collection_name=settings.job_collection_name)
    try:
        record = index.ingest(NormalizedJob(company_name="测试公司", job_title="Python AI应用工程师",
            description="岗位职责：\n负责Python应用开发与RAG知识库建设。\n任职要求：\n熟悉Python及FastAPI，具备本科及以上学历。",
            location="上海", source_kind="open_source", source_dataset="official_test", source_file="test.md",
            source_url="https://example.com/careers/test", language="zh", is_domestic=True,
            recruitment_type="campus", graduation_year=2027)).record
    finally:
        index.close()
    model = Mock()
    model.invoke.return_value.content = "根据已有 Python 项目经历准备岗位面试，未提供的信息待确认。"
    monkeypatch.setattr("app.domestic.service.get_llm", lambda **kwargs: model)
    pdf = io.BytesIO()
    document = PdfWriter()
    page = document.add_blank_page(width=600, height=850)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    commands = []
    for i, line in enumerate(["Candidate: Demo. Graduation Year: 2027. Bachelor in Computer Science.",
                              "Project: Python API with FastAPI, RAG knowledge base and Chroma retrieval.",
                              "Implemented document ingestion and evidence search. Target: AI application developer."]):
        commands.append(f"BT /F1 10 Tf 30 {800 - i * 20} Td ({line}) Tj ET")
    stream = DecodedStreamObject()
    stream.set_data("\n".join(commands).encode("ascii"))
    page[NameObject("/Contents")] = stream
    document.write(pdf)
    content = pdf.getvalue()
    with TestClient(app) as client:
        client.headers["Authorization"] = "Bearer owner-test-token"
        response = client.post("/api/domestic/profile/pdf", params={"candidate_id": "demo"}, files={"file": ("resume.pdf", content, "application/pdf")})
        assert response.status_code == 200, response.text
        original = settings.source_corpus_dir / "candidate_profiles/demo/resume_source.pdf"
        assert original.read_bytes() == content
        bad = client.post("/api/domestic/profile/pdf", params={"candidate_id": "demo"}, files={"file": ("bad.pdf", b"%PDF-broken", "application/pdf")})
        assert bad.status_code == 422
        assert original.read_bytes() == content
        search = client.get("/api/domestic/jobs/search", params={"candidate_id": "demo", "keyword": "想找Python相关的开发岗位"})
        assert search.json()["matches"][0]["job_id"] == record.job_id
        base = f"/api/domestic/jobs/{record.job_id}"
        evidence = client.get(base + "/evidence", params={"query": "Python", "k": 5})
        assert evidence.status_code == 200 and evidence.json()["documents"]
        fit = client.post(base + "/fit", params={"candidate_id": "demo"})
        assert fit.status_code == 200 and fit.json()["artifact_id"]
        package = client.post(base + "/package", params={"candidate_id": "demo"})
        assert package.status_code == 200, package.text
        payload = package.json()
        assert payload["status"] == "generated_not_submitted"
        assert payload["source_hashes"]["jd_text_sha256"]
        assert all(payload[key] for key in ("cover_letter", "application_email", "interview_questions"))
        assert model.invoke.call_count == 5  # Standalone fit + package fit + three drafts.
        assert original.read_bytes() == content
        history = client.get(base + "/artifacts", params={"candidate_id": "demo"}).json()["artifacts"]
        assert len(history) == 2
        saved = client.get(base + f"/artifacts/{payload['artifact_id']}", params={"candidate_id": "demo"})
        assert saved.json() == payload
        assert client.get(base + f"/artifacts/{payload['artifact_id']}", params={"candidate_id": "other"}).status_code == 404
        stage = client.put(base + "/application", json={"candidate_id": "demo", "stage": "planned", "notes": "核对官网后手动投递"})
        assert stage.status_code == 200
        board = client.get("/api/domestic/applications/demo").json()["applications"]
        assert board[0]["stage"] == "planned"
