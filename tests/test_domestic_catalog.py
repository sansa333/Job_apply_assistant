from __future__ import annotations

import sqlite3

from app.knowledge.catalog import JobCatalog
from app.knowledge.models import NormalizedJob


def make_job(description: str = "负责 Python RAG Agent 应用开发") -> NormalizedJob:
    return NormalizedJob(
        company_name="测试科技",
        job_title="大模型应用开发工程师",
        description=description,
        location="上海",
        source_kind="open_source",
        source_dataset="official_test",
        source_file="job.json",
        source_url="https://example.com/careers",
        language="zh",
        external_id="external-1",
        apply_url="https://example.com/apply/1",
        source_name="测试科技招聘官网",
        recruitment_type="campus",
        graduation_year=2027,
        job_category="agent_development",
        is_domestic=True,
    )


def test_domestic_upsert_update_snapshot_search_and_application(tmp_path) -> None:
    catalog = JobCatalog(tmp_path / "jobs.sqlite3")
    first = catalog.upsert(make_job())
    assert first.inserted is True
    updated = catalog.upsert(make_job("负责 Python FastAPI RAG Agent 工具调用应用开发"))
    assert updated.inserted is False
    assert updated.updated is True
    assert updated.record.job_id == first.record.job_id
    assert catalog.search(keyword="Python Agent", graduation_year=2027)[0].job_id == first.record.job_id
    application = catalog.set_application_stage(
        candidate_id="current_candidate", job_id=first.record.job_id, stage="saved", notes="优先"
    )
    assert application["stage"] == "saved"
    assert catalog.list_applications("current_candidate")[0]["company_name"] == "测试科技"
    with sqlite3.connect(catalog.path) as connection:
        snapshots = connection.execute(
            "SELECT COUNT(*) FROM job_snapshots WHERE job_id = ?", (first.record.job_id,)
        ).fetchone()[0]
    assert snapshots == 2


def test_job_closes_only_after_three_consecutive_misses(tmp_path) -> None:
    catalog = JobCatalog(tmp_path / "jobs.sqlite3")
    record = catalog.upsert(make_job()).record
    catalog.set_source_missing("official_test", set())
    assert catalog.get(record.job_id).status == "possibly_closed"
    catalog.set_source_missing("official_test", set())
    assert catalog.get(record.job_id).status == "possibly_closed"
    catalog.set_source_missing("official_test", set())
    assert catalog.get(record.job_id).status == "closed"
