from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.domestic.service import rank_job
from app.knowledge.catalog import JobCatalog
from app.knowledge.models import JobRecord


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def relevance(record: JobRecord, query: dict) -> int:
    if query.get("company") and query["company"] != record.company_name:
        return 0
    if query.get("category") and query["category"] != record.job_category:
        return 0
    terms = query.get("terms", [])
    haystack = f"{record.job_title}\n{record.description}".lower()
    hits = sum(term.lower() in haystack for term in terms)
    if terms and hits == 0:
        return 0
    return 2 if hits >= 2 or query.get("category") == record.job_category else 1


def dcg(values: list[int]) -> float:
    return sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(values))


def build_dataset(output_dir: Path) -> dict:
    catalog = JobCatalog(settings.job_catalog_path)
    records = catalog.domestic_records(status="open")
    if not records:
        raise RuntimeError("No open domestic jobs found. Sync official sources first.")

    query_specs = [
        {"query_id": "q_ai_application", "query": "AI 全栈 应用 开发 Python", "category": "ai_application", "terms": ["AI", "Python", "应用"]},
        {"query_id": "q_agent", "query": "Agent 智能体 工具调用", "category": "agent_development", "terms": ["Agent", "智能体", "工具调用"]},
        {"query_id": "q_llm_rag", "query": "大模型应用 RAG 知识库", "category": "llm_application", "terms": ["大模型", "RAG", "知识库"]},
        {"query_id": "q_ai_software", "query": "AI DevOps 系统 软件 平台", "category": "ai_software", "terms": ["AI", "系统", "平台"]},
        {"query_id": "q_baidu_ai", "query": "百度 AI 大模型 校招", "company": "百度", "terms": ["AI", "大模型"]},
        {"query_id": "q_dji_ai", "query": "大疆 AI 算法 软件 校招", "company": "大疆创新", "terms": ["AI", "算法", "软件"]},
        {"query_id": "q_kylin_python", "query": "麒麟软件 Python AI 校招", "company": "麒麟软件", "terms": ["Python", "AI"]},
        {"query_id": "q_transwarp_agent", "query": "星环科技 智能体 大模型 后端", "company": "星环科技", "terms": ["智能体", "大模型", "后端"]},
        {"query_id": "q_cecloud_agent", "query": "中国电子云 Agent LLM 后端", "company": "中国电子云", "terms": ["Agent", "LLM", "后端"]},
        {"query_id": "q_qunar_ai_app", "query": "去哪儿 AI 应用 全栈 Java", "company": "去哪儿旅行", "terms": ["AI", "应用", "全栈", "Java"]},
        {"query_id": "q_candidate_stack", "query": "Python FastAPI RAG Agent LangChain", "terms": ["Python", "FastAPI", "RAG", "Agent", "LangChain"]},
    ]
    query_rows: list[dict] = []
    metrics: list[dict] = []
    for spec in query_specs:
        labels = {
            record.job_id: relevance(record, spec)
            for record in records
            if relevance(record, spec) > 0
        }
        query_rows.append({**spec, "relevance": labels})
        candidates = [
            record for record in records
            if not spec.get("company") or record.company_name == spec["company"]
        ]
        ranked_all = sorted(
            candidates,
            key=lambda record: rank_job(record, query=spec["query"]),
            reverse=True,
        )
        ranked = ranked_all[:10]
        actual = [labels.get(record.job_id, 0) for record in ranked]
        ideal = sorted(labels.values(), reverse=True)[:10]
        relevant = set(labels)
        top_10_relevant = len({r.job_id for r in ranked} & relevant)
        top_50_relevant = len({r.job_id for r in ranked_all[:50]} & relevant)
        metrics.append(
            {
                "query_id": spec["query_id"],
                "relevant_count": len(relevant),
                "precision_at_10": round(top_10_relevant / 10, 4),
                "hit_rate_at_10": 1.0 if top_10_relevant else 0.0,
                "recall_at_10": round(top_10_relevant / len(relevant), 4) if relevant else None,
                "recall_at_50": round(top_50_relevant / len(relevant), 4) if relevant else None,
                "ndcg_at_10": round(dcg(actual) / dcg(ideal), 4) if ideal and dcg(ideal) else None,
            }
        )

    job_rows = [
        {
            "job_id": record.job_id,
            "company_name": record.company_name,
            "job_title": record.job_title,
            "description": record.description,
            "location": record.location,
            "job_category": record.job_category,
            "recruitment_type": record.recruitment_type,
            "graduation_year": record.graduation_year,
            "source_dataset": record.source_dataset,
            "source_url": record.source_url,
            "apply_url": record.apply_url,
            "status": record.status,
        }
        for record in records
    ]
    pair_rows = [
        {
            "candidate_id": "current_candidate",
            "job_id": record.job_id,
            "silver_score": rank_job(record),
            "silver_label": "high" if rank_job(record) >= 45 else "medium" if rank_job(record) >= 25 else "low",
            "label_type": "deterministic_silver_not_human_ground_truth",
        }
        for record in records
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "jobs.jsonl", job_rows)
    write_jsonl(output_dir / "queries.jsonl", query_rows)
    write_jsonl(output_dir / "candidate_job_silver.jsonl", pair_rows)
    report = {
        "dataset": "domestic_job_search_v1",
        "job_count": len(job_rows),
        "company_count": len({row["company_name"] for row in job_rows}),
        "query_count": len(query_rows),
        "company_distribution": Counter(row["company_name"] for row in job_rows),
        "category_distribution": Counter(row["job_category"] or "unknown" for row in job_rows),
        "metrics": metrics,
        "aggregate": {
            "mean_precision_at_10": round(
                sum(row["precision_at_10"] for row in metrics) / len(metrics),
                4,
            ),
            "mean_hit_rate_at_10": round(
                sum(row["hit_rate_at_10"] for row in metrics) / len(metrics), 4
            ),
            "mean_recall_at_50": round(
                sum(row["recall_at_50"] for row in metrics if row["recall_at_50"] is not None)
                / max(1, sum(row["recall_at_50"] is not None for row in metrics)),
                4,
            ),
            "mean_ndcg_at_10": round(
                sum(row["ndcg_at_10"] for row in metrics if row["ndcg_at_10"] is not None)
                / max(1, sum(row["ndcg_at_10"] is not None for row in metrics)),
                4,
            ),
        },
        "limitations": [
            "数据由企业官网逐岗位数据与公开校招公告级岗位合集组成，不能代表整个国内校招市场。",
            "公告级记录不是逐岗位 API 数据，具体岗位状态、要求与投递入口必须在企业页面复核。",
            "相关性标签和候选人匹配标签为规则生成的银标，需要后续人工复核。",
            "岗位状态以最近一次成功同步为准。",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=dict), encoding="utf-8"
    )
    (output_dir / "DATASET_CARD.md").write_text(
        "# 国内岗位检索评测集 v1\n\n"
        f"- 岗位数：{len(job_rows)}\n"
        f"- 公司数：{len({row['company_name'] for row in job_rows})}\n"
        f"- 查询数：{len(query_rows)}\n"
        "- 来源：系统登记的国内企业官方招聘页面及高校就业网公开校招公告\n"
        "- 用途：检索回归测试、候选人-岗位银标排序测试\n"
        "- 限制：不是人工金标，不用于宣称招聘市场整体效果；投递前须打开官网复核。\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.data_dir / "eval_dataset" / "domestic_job_search_v1",
    )
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.output_dir), ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
