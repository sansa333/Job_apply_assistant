"""Read-only HTTP checks against the configured production catalog and index."""
import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists")
    headers = {"Authorization": f"Bearer {settings.api_token}"} if settings.api_token else {}
    report = {"scope": "real production retrieval through HTTP, no external LLM or application submission", "search": {}}
    with TestClient(app, headers=headers) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        report["ready"] = response.json()["status"]
        for strategy in ("bm25", "dense", "hybrid"):
            response = client.get("/api/domestic/jobs/search", params={"keyword": "实时嵌入式操作系统",
                "company_name": "麒麟软件", "graduation_year": 2027, "strict_graduation": True,
                "individual_only": True, "strategy": strategy, "limit": 5})
            assert response.status_code == 200, response.text
            rows = response.json()["matches"]
            assert rows and all(r["company_name"] == "麒麟软件" and r["graduation_year"] == 2027 for r in rows)
            report["search"][strategy] = {"count": len(rows), "top_job_id": rows[0]["job_id"], "top_title": rows[0]["job_title"]}
        job_id = rows[0]["job_id"]
        response = client.get(f"/api/domestic/jobs/{job_id}/evidence", params={"query": "岗位要求哪些调试工具？", "k": 3})
        assert response.status_code == 200
        docs = response.json()["documents"]
        assert docs and all(d["job_id"] == job_id for d in docs)
        report["exact_job_evidence"] = True
        response = client.get(f"/api/domestic/jobs/{job_id}/evidence/check", params={"query": "每月工资多少？"})
        assert response.status_code == 200
        report["evidence_check"] = response.json()
        if settings.domestic_evidence_policy_path is None:
            assert report["evidence_check"]["abstained"] and report["evidence_check"]["status"] == "not_calibrated"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
