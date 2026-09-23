"""Opt-in real-model acceptance in a fresh directory using synthetic candidate text.

Does not send email or applications and never reads a personal resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import settings
from app.knowledge.catalog import JobCatalog
from app.knowledge.ingestion import JobKnowledgeIngestion
from app.knowledge.models import NormalizedJob
from app.main import app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True, help="Fresh official records serialized as NormalizedJob objects")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live-llm", action="store_true", help="Send synthetic resume and public JD to the configured LLM")
    args = parser.parse_args()
    if not args.live_llm:
        parser.error("This acceptance requires explicit --live-llm")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "candidate": "synthetic text fixture; no private resume",
              "status": "running", "provider_model": settings.llm_model, "steps": []}
    started = time.perf_counter()
    try:
        with ExitStack() as stack:
            for name, value in {"job_catalog_path": root / "catalog.sqlite3", "source_corpus_dir": root / "sources",
                                "vector_db_dir": root / "indexes", "outputs_dir": root / "artifacts",
                                "domestic_sync_enabled": False, "job_retrieval_strategy": "vector"}.items():
                stack.enter_context(patch.object(settings, name, value))
            jobs = [NormalizedJob(**r) for r in json.loads(args.jobs.read_text(encoding="utf-8"))]
            catalog = JobCatalog(settings.job_catalog_path)
            index = JobKnowledgeIngestion(catalog=catalog, source_corpus_dir=settings.source_corpus_dir,
                                          vector_db_dir=settings.vector_db_dir, collection_name=settings.job_collection_name)
            try:
                records = [index.ingest(job).record for job in jobs]
            finally:
                index.close()
            report["steps"].append({"step": "fresh_official_jobs_indexed", "jobs": len(records)})
            print(f"Indexed {len(records)} fresh official jobs in isolated acceptance storage", flush=True)
            candidate_id = "synthetic_acceptance"
            profile_dir = settings.source_corpus_dir / "candidate_profiles" / candidate_id
            profile_dir.mkdir(parents=True)
            resume = ("这是测试用的虚构候选人资料，不对应真实个人。\n教育背景：计算机科学与技术本科，2027年毕业。\n"
                      "项目经历：使用Python、FastAPI和Chroma实现课程资料检索接口，编写文档导入与检索测试。\n"
                      "没有提供企业实习、论文、竞赛奖项或量化业务收益。求职方向：AI应用开发。")
            (profile_dir / "resume_extracted.txt").write_text(resume, encoding="utf-8")
            (profile_dir / "profile.json").write_text(json.dumps({"candidate_id": candidate_id, "graduation_year": 2027,
                "skills": ["Python", "FastAPI", "Chroma"], "target_roles": ["AI应用开发"], "target_cities": []}), encoding="utf-8")
            checksum = hashlib.sha256((profile_dir / "resume_extracted.txt").read_bytes()).hexdigest()
            report["initial_fixture_sha256"] = checksum
            with TestClient(app) as client:
                if settings.api_token:
                    client.headers["Authorization"] = f"Bearer {settings.api_token}"
                search = client.get("/api/domestic/jobs/search", params={"keyword": "Python", "candidate_id": candidate_id})
                search.raise_for_status()
                matches = search.json()["matches"]
                if not matches:
                    raise RuntimeError("No fresh Python match; acceptance cannot proceed")
                record = catalog.get(matches[0]["job_id"])
                report["job"] = {"job_id": record.job_id, "title": record.job_title, "source_url": record.source_url}
                base = f"/api/domestic/jobs/{record.job_id}"
                evidence = client.get(base + "/evidence", params={"query": "Python 岗位要求", "k": 5})
                evidence.raise_for_status()
                assert evidence.json()["documents"]
                print("Search/evidence passed; requesting real LLM analysis and drafts", flush=True)
                report["steps"].append({"step": "search_and_exact_job_evidence", "evidence_count": len(evidence.json()["documents"])})
                response = client.post(base + "/package", params={"candidate_id": candidate_id})
                response.raise_for_status()
                package = response.json()
                print("Real LLM returned; verifying saved history and unchanged fixture", flush=True)
                assert all(package.get(k) for k in ("cover_letter", "interview_questions", "application_email"))
                report["steps"].append({"step": "real_llm_fit_and_three_drafts", "status": package["status"],
                                        "validation_findings": len(package["validation_findings"])})
                history = client.get(base + f"/artifacts/{package['artifact_id']}", params={"candidate_id": candidate_id})
                assert history.json() == package
                board = client.put(base + "/application", json={"candidate_id": candidate_id, "stage": "planned"})
                board.raise_for_status()
                assert hashlib.sha256((profile_dir / "resume_extracted.txt").read_bytes()).hexdigest() == checksum
                report["steps"].append({"step": "saved_history_board_and_resume_unchanged", "status": "passed"})
                report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error_note"] = "Inspect local logs; provider exceptions are not copied to avoid exposing credentials."
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        (root / "acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
