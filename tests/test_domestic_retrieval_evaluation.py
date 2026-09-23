from __future__ import annotations

import json
import re
from dataclasses import replace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.domestic.service import DomesticJobService, search_keyword_terms
from app.embeddings import HashEmbeddings
from app.evaluation.domestic import (
    aggregate, build_dataset, frozen_catalog, paired_comparison, rank_metrics, read_jsonl, verify_dataset,
)
from app.knowledge.catalog import JobCatalog
from app.knowledge.ingestion import JobKnowledgeIngestion
from app.knowledge.models import NormalizedJob
from tools.finalize_domestic_annotations import finalize
from app.routes.domestic import router
from app.chunking import JD_CHUNK_SCHEMA_VERSION, is_heading_only_job_chunk, split_documents_semantic
from app.evaluation.domestic import TOPICS
from app.knowledge.ingestion import StaleJobIndexError
from langchain_core.documents import Document


@pytest.mark.parametrize("topic,text,negative", [
    ("python", "熟悉Python或python等编程工具", "CPython pythonic"),
    ("java", "掌握Java开发", "JavaScript"),
    ("rag", "参与RAG应用", "fragment"),
    ("agent", "负责Agents开发", "agentic"),
    ("sql", "掌握SQL语言", "SQLAlchemy"),
    ("training", "熟悉SFT和RLHF训练", "SFTTrainer"),
])
def test_mixed_chinese_english_label_boundaries(topic, text, negative):
    pattern = next(row[1] for row in TOPICS if row[0] == topic)
    assert re.search(pattern, text, re.I)
    assert not re.search(pattern, negative, re.I)


def test_empty_heading_removed_without_losing_short_facts(tmp_path):
    docs = split_documents_semantic([Document(page_content="岗位职责：\n负责Python开发。\n任职要求：\n岗位要求：\n学历：本科", metadata={"doc_type": "job_description"})])
    assert docs and all(not is_heading_only_job_chunk(doc.page_content) for doc in docs)
    assert "本科" in "\n".join(doc.page_content for doc in docs)
    catalog = JobCatalog(tmp_path / "jobs.sqlite3")
    record = add_job(catalog, description="任职要求：")
    chunker = JobKnowledgeIngestion.__new__(JobKnowledgeIngestion)
    chunker.source_corpus_dir = tmp_path
    assert chunker._chunks_for_record(record) == []


def test_stale_index_cannot_silently_relabel_or_serve(tmp_path):
    catalog = JobCatalog(tmp_path / "jobs.sqlite3")
    job = add_job(catalog)
    args = dict(catalog=catalog, source_corpus_dir=tmp_path / "sources", vector_db_dir=tmp_path / "indexes",
                embeddings=HashEmbeddings(), embedding_backend="hash")
    index = JobKnowledgeIngestion(**args)
    index.rebuild_records([job])
    path = index.persist_dir / "collection_manifest.json"
    original = json.loads(path.read_text(encoding="utf-8"))
    index.close()
    stale = {**original, "chunk_schema_version": "old"}
    path.write_text(json.dumps(stale), encoding="utf-8")
    index = JobKnowledgeIngestion(**args)
    try:
        assert json.loads(path.read_text(encoding="utf-8")) == stale
        with pytest.raises(StaleJobIndexError):
            index.retrieve_for_job(job.job_id, "Python")
        index.rebuild_records([job])
        assert index.retrieve_for_job(job.job_id, "Python")
        assert json.loads(path.read_text(encoding="utf-8"))["chunk_schema_version"] == JD_CHUNK_SCHEMA_VERSION
    finally:
        index.close()


def add_job(catalog, title="Python 工程师", description="岗位职责\n负责 Python 应用开发。\n任职要求\n本科及以上学历，熟悉数据库。", **kwargs):
    return catalog.upsert(NormalizedJob(
        company_name="测试公司", job_title=title, description=description, location="上海",
        source_kind="open_source", source_dataset="official_test", source_file="job.md",
        source_url="https://example.com/job", language="zh", is_domestic=True,
        recruitment_type="campus", graduation_year=2027, **kwargs,
    )).record


def test_metrics_truncation_duplicates_and_unanswerable():
    result = rank_metrics(["x", "a", "a", "y", "z", "c", "b"], {"a", "b"})
    assert result["mrr_at_3"] == .5
    assert result["recall_at_5"] == .5
    assert result["recall_at_10"] == 1
    assert rank_metrics(["x", "y", "z", "a"], {"a"})["mrr_at_3"] == 0
    missing = rank_metrics(["x"], set())
    assert missing["recall_at_5"] is None
    assert aggregate([result, missing])["recall_at_5"] == .5
    assert rank_metrics([], {"a"})["recall_at_5"] == 0


def test_paired_comparison_can_report_regression_and_rejects_unpaired():
    base = [{"query_id": str(i), "group_id": str(i // 2), "mrr_at_3": 1.0} for i in range(8)]
    candidate = [{**row, "mrr_at_3": .5} for row in base]
    delta = paired_comparison(base, candidate, "mrr_at_3", iterations=100)
    assert delta["delta"] == -.5
    assert delta["relative_percent"] == -50
    assert delta["ci95"] == [-.5, -.5]
    assert delta["groups"] == 4
    with pytest.raises(ValueError, match="identical"):
        paired_comparison(base, candidate[:-1], "mrr_at_3")


def test_search_ranks_all_candidates_before_pagination(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "source_corpus_dir", tmp_path / "no_profiles")
    catalog = JobCatalog(tmp_path / "jobs.sqlite3")
    # The best job is older than 205 distractors: SQL pre-pagination loses it.
    best = add_job(catalog, title="Python Agent RAG 大模型应用工程师")
    for index in range(205):
        add_job(catalog, title=f"普通岗位{index}", description="岗位职责：参与文档管理。")
    with catalog._connect() as connection:
        connection.execute("UPDATE job_records SET created_at = '2030-01-01' WHERE job_id != ?", (best.job_id,))
        connection.execute("UPDATE job_records SET created_at = '2000-01-01' WHERE job_id = ?", (best.job_id,))
    service = DomesticJobService(catalog=catalog)
    result = service.search(limit=1)
    assert result["matches"][0]["job_id"] == best.job_id
    assert result["total"] == 206
    page2 = service.search(limit=1, offset=1)
    assert page2["matches"][0]["job_id"] != best.job_id
    assert service.search(location="北京")["count"] == 0
    natural = service.search(keyword="我想找Python或RAG相关的开发岗位", limit=1)
    assert natural["matches"][0]["job_id"] == best.job_id
    assert natural["normalized_keyword"] == "Python RAG"
    assert search_keyword_terms("Python FastAPI") == "Python FastAPI"
    assert search_keyword_terms("不存在的精确职位") == "不存在的精确职位"


def test_frozen_data_uses_production_chunks_and_exhaustive_silver(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    live = JobCatalog(tmp_path / "original.sqlite3")
    record = add_job(live)
    duplicate = replace(record, job_id="duplicate", job_title="另一标题")
    directory = tmp_path / "dataset"
    manifest = build_dataset(directory, [record, duplicate], source)
    assert manifest["jobs"] == 1
    assert len(manifest["duplicates_removed"]) == 1
    assert verify_dataset(directory) == manifest
    queries = read_jsonl(directory / "queries.jsonl")
    assert len({q["split"] for q in queries}) == 1
    labels = read_jsonl(directory / "qrels.jsonl")
    assert all(row["human_grade"] is None for row in labels)
    assert not any(":vision" in row["query_id"] for row in labels)
    evidence = read_jsonl(directory / "evidence.jsonl")
    tasks = read_jsonl(directory / "annotation_tasks.jsonl")
    assert len(tasks) == len(queries) * len(evidence)
    isolated = frozen_catalog(tmp_path / "frozen.sqlite3", [record])
    assert isolated.get(record.job_id).description == record.description
    with pytest.raises(ValueError, match="overwrite"):
        frozen_catalog(tmp_path / "frozen.sqlite3", [record])
    with pytest.raises(ValueError, match="completed"):
        finalize(directory, directory / "annotation_tasks.jsonl", tmp_path / "reviewed", "a", "b")
    for task in tasks:
        task["annotator_1"] = task["silver_grade"]
        task["annotator_2"] = task["silver_grade"]
    review_file = tmp_path / "test_reviews.jsonl"
    review_file.write_text("".join(json.dumps(task) + "\n" for task in tasks), encoding="utf-8")
    agreement = finalize(directory, review_file, tmp_path / "reviewed", "test_a", "test_b")
    assert agreement["agreement"] == 1
    assert verify_dataset(tmp_path / "reviewed")["annotation_status"] == "human_double_reviewed"
    (directory / "qrels.jsonl").write_text(json.dumps(labels[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        verify_dataset(directory)


def test_service_evidence_uses_real_chroma_and_exact_job(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "job_retrieval_strategy", "vector")
    catalog = JobCatalog(tmp_path / "catalog.sqlite3")
    target = add_job(catalog)
    other = add_job(catalog, title="另一个岗位", description="岗位职责：Python Python Python RAG。")
    ingestion = JobKnowledgeIngestion(
        catalog=catalog, source_corpus_dir=tmp_path / "sources", vector_db_dir=tmp_path / "indexes",
        collection_name="test_domestic", embeddings=HashEmbeddings(), embedding_backend="hash",
    )
    try:
        ingestion.rebuild_records([target, other])
        # Test service closure with a factory opening the same isolated index.
        ingestion.close()
        service = DomesticJobService(catalog=catalog, ingestion_factory=lambda: JobKnowledgeIngestion(
            catalog=catalog, source_corpus_dir=tmp_path / "sources", vector_db_dir=tmp_path / "indexes",
            collection_name="test_domestic", embeddings=HashEmbeddings(), embedding_backend="hash",
        ))
        result = service.retrieve_job_evidence(job_id=target.job_id, query="Python", k=5)
        assert result["status"] == "ok"
        assert all(item["job_id"] == target.job_id for item in result["documents"])
        with pytest.raises(ValueError, match="not found"):
            service.retrieve_job_evidence(job_id="missing", query="Python")
    finally:
        ingestion.close()


def test_domestic_fit_passes_retrieved_evidence_to_model(tmp_path, monkeypatch):
    catalog = JobCatalog(tmp_path / "jobs.sqlite3")
    record = add_job(catalog)
    monkeypatch.setattr("app.domestic.service.load_candidate_resume_text", lambda *args: "本科，Python 项目开发经验。")
    monkeypatch.setattr("app.domestic.service.load_candidate_profile", lambda *args: None)
    model = Mock()
    model.invoke.return_value.content = "结合证据准备面试。"
    monkeypatch.setattr("app.domestic.service.get_llm", lambda **kwargs: model)
    service = DomesticJobService(catalog=catalog)
    evidence = {"status": "ok", "documents": [{"chunk_id": "proof:123", "content": "可核验的 Python 岗位要求"}]}
    service.retrieve_job_evidence = Mock(return_value=evidence)
    result = service.analyze_job(job_id=record.job_id)
    assert "[proof:123]" in model.invoke.call_args.args[0]
    assert result["job_evidence"] == evidence
    assert record.description in model.invoke.call_args.args[0]


def test_evidence_http_contract_and_missing_job(monkeypatch):
    service = Mock()
    service.retrieve_job_evidence.return_value = {"status": "ok", "documents": [{"chunk_id": "job:one:0"}]}
    monkeypatch.setattr("app.routes.domestic.DomesticJobService", lambda: service)
    application = FastAPI()
    application.include_router(router)
    client = TestClient(application)
    assert client.get("/api/domestic/jobs/one/evidence", params={"query": "Python", "k": 5}).status_code == 200
    service.retrieve_job_evidence.assert_called_once_with(job_id="one", query="Python", k=5)
    assert client.get("/api/domestic/jobs/one/evidence", params={"query": "Python", "k": 51}).status_code == 422
    assert client.get("/api/domestic/jobs/one/evidence", params={"query": ""}).status_code == 422
    service.retrieve_job_evidence.side_effect = ValueError("domestic job not found")
    assert client.get("/api/domestic/jobs/missing/evidence", params={"query": "Python"}).status_code == 404
