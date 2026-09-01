from __future__ import annotations

from unittest.mock import Mock

from app.domestic.service import DomesticJobService


def service_with_results(sync_status: str = "success") -> DomesticJobService:
    service = DomesticJobService.__new__(DomesticJobService)
    failed_row = (
        [{"source_id": "notice_source", "status": "error", "error": "timeout"}]
        if sync_status == "partial"
        else []
    )
    service.sync_all = Mock(
        return_value={
            "status": sync_status,
            "results": [
                {"source_id": "official_source", "status": "success", "fetched": 10},
                *failed_row,
            ],
            "domestic_jobs": 290,
            "open_domestic_jobs": 288,
        }
    )
    service.rebuild_domestic_index = Mock(
        return_value={
            "status": "success",
            "domestic_only": True,
            "jobs_indexed": 288,
            "chunks_indexed": 1200,
            "collection": "job_knowledge_bge_m3",
        }
    )
    return service


def test_refresh_syncs_without_incremental_index_then_rebuilds_once() -> None:
    service = service_with_results()
    result = service.refresh_all()

    service.sync_all.assert_called_once_with(build_index=False)
    service.rebuild_domestic_index.assert_called_once_with()
    assert result["status"] == "success"
    assert result["jobs_indexed"] == 288
    assert result["chunks_indexed"] == 1200
    assert result["failed_sources"] == []


def test_refresh_rebuilds_consistent_index_after_partial_sync() -> None:
    service = service_with_results("partial")
    result = service.refresh_all()

    service.rebuild_domestic_index.assert_called_once_with()
    assert result["status"] == "partial"
    assert result["failed_sources"] == ["notice_source"]
