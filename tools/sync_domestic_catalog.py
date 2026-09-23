"""Refresh the configured live domestic catalog and index during a maintenance window.

Back up existing state first; fetch configured sources rather than importing eval snapshots.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.domestic.service import DomesticJobService
from app.main import app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for backups and the sync report")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
              "catalog": str(settings.job_catalog_path.resolve()),
              "index_directory": str((settings.vector_db_dir / settings.job_collection_name).resolve()),
              "source_policy": "fresh configured source fetches; no evaluation snapshot imports", "sources": []}

    def save():
        (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        if settings.domestic_sync_enabled:
            raise RuntimeError("Disable the scheduler and stop serving requests before maintenance refresh")
        if settings.job_catalog_path.exists():
            with sqlite3.connect(settings.job_catalog_path) as source, sqlite3.connect(root / "catalog_before.sqlite3") as target:
                source.backup(target)
        index_path = (settings.vector_db_dir / settings.job_collection_name).resolve()
        if index_path.is_dir():
            shutil.copytree(index_path, root / "index_before")
        service = DomesticJobService()
        report["before"] = {"total": service.catalog.count(), "domestic": service.catalog.count(domestic_only=True)}
        save()
        for source_id in service.sources:
            print(f"Fetching {source_id}", flush=True)
            try:
                result = service.sync_source(source_id, build_index=False)
            except Exception as exc:
                result = {"source_id": source_id, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:1000]}
            report["sources"].append(result)
            save()
            print(json.dumps(result, ensure_ascii=False), flush=True)
        print("Rebuilding configured production BGE index", flush=True)
        report["index"] = service.rebuild_domestic_index()
        records = service.catalog.domestic_records(status="open")
        report["after"] = {"total": service.catalog.count(), "open_domestic": len(records),
                           "by_company": dict(Counter(r.company_name for r in records)),
                           "by_source": dict(Counter(r.source_dataset for r in records)),
                           "public_notice_bundles": sum("岗位合集" in r.job_title or "notices" in r.source_dataset for r in records)}
        with TestClient(app) as client:
            if settings.api_token:
                client.headers["Authorization"] = f"Bearer {settings.api_token}"
            readiness = client.get("/ready")
            search = client.get("/api/domestic/jobs/search", params={"candidate_id": "catalog_sync_validation", "keyword": "Python", "limit": 5})
            search.raise_for_status()
            body = search.json()
            report["verification"] = {"readiness_http_status": readiness.status_code, "readiness": readiness.json(),
                                      "python_search_total": body["total"], "returned_jobs": len(body["matches"])}
            if body["matches"]:
                job_id = body["matches"][0]["job_id"]
                evidence = client.get(f"/api/domestic/jobs/{job_id}/evidence", params={"query": "Python 技术要求", "k": 5})
                evidence.raise_for_status()
                documents = evidence.json()["documents"]
                report["verification"]["evidence"] = {"job_id": job_id, "count": len(documents),
                    "exact_job_isolation": bool(documents) and all(d["job_id"] == job_id for d in documents)}
            else:
                raise RuntimeError("Fresh domestic catalog has no Python search results")
        passed = readiness.status_code == 200 and report["verification"]["evidence"]["exact_job_isolation"]
        report["status"] = ("success" if all(row["status"] == "success" for row in report["sources"]) else "partial") if passed else "verification_failed"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        print(json.dumps({key: value for key, value in report.items() if key not in {"sources", "source_policy"}}, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
