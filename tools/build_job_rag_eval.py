from __future__ import annotations

import json

from app.config import settings
from app.knowledge.catalog import JobCatalog
from app.knowledge.evaluation import build_job_eval_samples, evaluate_job_retrieval
from app.knowledge.hybrid import JobHybridRetriever
from app.knowledge.ingestion import JobKnowledgeIngestion
from app.multimodal.reranker import CrossEncoderReranker


def _markdown_report(report: dict) -> str:
    retrieval = report["retrieval_config"]
    lines = [
        "# 岗位 RAG 检索评测报告",
        "",
        f"样本数：{report['sample_count']}",
        "",
        "## 检索配置",
        f"- Embedding：{retrieval['embedding_model']}",
        f"- 策略：{retrieval['strategy']}（向量召回 + BM25 + RRF + Cross-Encoder 分数融合）",
        f"- 候选数：{retrieval['candidate_k']}；RRF 常数：{retrieval['rrf_k']}；Cross-Encoder 权重：{retrieval['rerank_weight']}",
        f"- Cross-Encoder：{retrieval['reranker_model']}；实际应用：{retrieval['reranker_applied_samples']}/{report['sample_count']}",
        "",
        "## 总体指标",
    ]
    lines.extend(f"- {name}: {value:.4f}" for name, value in report["metrics"].items())
    diagnostics = report.get("diagnostics", {})
    lines.extend(
        [
            "",
            "## 数据集诊断",
            f"- 唯一问句：{diagnostics.get('unique_query_count', 0)}/{report['sample_count']}",
            f"- 平均相关 chunk 数：{diagnostics.get('mean_relevant_chunks_per_query', 0):.2f}",
            f"- 多相关查询比例：{diagnostics.get('multi_relevant_query_ratio', 0):.4f}",
            f"- 平均候选池大小：{diagnostics.get('mean_candidate_pool_size', 0):.2f}",
            f"- 评测警告：{', '.join(diagnostics.get('warnings', [])) or '无'}",
        ]
    )
    lines.extend(["", "## 问题类型分布", "", "| 问题类型 | 样本数 |", "| --- | ---: |"])
    lines.extend(f"| {question_type} | {count} |" for question_type, count in report["question_type_distribution"].items())
    lines.extend(
        [
            "",
            "## 分类型指标",
            "",
            "| 问题类型 | 样本数 | HitRate@1 | HitRate@3 | Recall@3 | Recall@5 | MRR@3 | MRR@5 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for question_type, metrics in report["metrics_by_question_type"].items():
        lines.append(
            "| {type} | {count} | {h1:.4f} | {h3:.4f} | {r3:.4f} | {r5:.4f} | {mrr3:.4f} | {mrr5:.4f} |".format(
                type=question_type,
                count=metrics["sample_count"],
                h1=metrics.get("hit_rate_at_1", 0.0),
                h3=metrics.get("hit_rate_at_3", 0.0),
                r3=metrics.get("recall_at_3", 0.0),
                r5=metrics.get("recall_at_5", 0.0),
                mrr3=metrics.get("mrr_at_3", 0.0),
                mrr5=metrics.get("mrr_at_5", 0.0),
            )
        )
    bad_cases = [item for item in report["details"] if item["first_relevant_rank"] != 1]
    lines.extend(["", "## Top-1 未命中样本", f"- 数量：{len(bad_cases)}"])
    for item in bad_cases[:10]:
        lines.append(
            f"- {item['query_id']} ({item['question_type']}): rank={item['first_relevant_rank']} | {item['query']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    output_dir = settings.data_dir / "eval_dataset" / "job_rag"
    output_dir.mkdir(parents=True, exist_ok=True)
    ingestion = JobKnowledgeIngestion(
        catalog=JobCatalog(settings.job_catalog_path),
        source_corpus_dir=settings.source_corpus_dir,
        vector_db_dir=settings.vector_db_dir,
        collection_name=settings.job_collection_name,
    )
    try:
        samples = build_job_eval_samples(ingestion.catalog, ingestion, limit=80)
        dataset = output_dir / "real_job_retrieval_eval.jsonl"
        with dataset.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        reranker = CrossEncoderReranker(
            enabled=settings.enable_reranker,
            model_name=settings.reranker_model,
            local_files_only=settings.reranker_local_files_only,
        )
        retriever = JobHybridRetriever(
            ingestion=ingestion,
            reranker=reranker,
            candidate_k=settings.job_retrieval_candidate_k,
            rrf_k=settings.job_retrieval_rrf_k,
            strategy=settings.job_retrieval_strategy,
            rerank_weight=settings.job_reranker_weight,
        )
        report = evaluate_job_retrieval(samples, retriever, ranks=(1, 3, 5))
        reranker_applied = sum(
            bool(item.get("retrieval", {}).get("reranker_applied")) for item in report["details"]
        )
        report["retrieval_config"] = {
            "embedding_model": settings.hf_embedding_model if settings.embedding_backend == "huggingface" else "hash",
            "strategy": settings.job_retrieval_strategy,
            "candidate_k": settings.job_retrieval_candidate_k,
            "rrf_k": settings.job_retrieval_rrf_k,
            "rerank_weight": settings.job_reranker_weight,
            "reranker_model": settings.reranker_model,
            "reranker_applied_samples": reranker_applied,
        }
        report["dataset"] = str(dataset)
        report_path = output_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "report.md").write_text(_markdown_report(report), encoding="utf-8-sig")
        print(json.dumps({"dataset": str(dataset), "report": str(report_path), **report["metrics"]}, ensure_ascii=False, indent=2))
    finally:
        ingestion.close()


if __name__ == "__main__":
    main()
