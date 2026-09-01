from __future__ import annotations

import json

from app.config import settings
from app.knowledge.catalog import JobCatalog
from app.knowledge.evaluation import build_job_eval_samples
from app.knowledge.experiments import (
    EmbeddingExperimentSpec,
    embedding_selection_evidence,
    run_model_experiment,
    select_embedding_winner,
    select_provisional_embedding_winner,
)
from app.knowledge.ingestion import JobKnowledgeIngestion


MODEL_SPECS = (
    EmbeddingExperimentSpec(name="hash_baseline", backend="hash", model_name=None),
    EmbeddingExperimentSpec(name="bge_small_zh", backend="huggingface", model_name="BAAI/bge-small-zh-v1.5"),
    EmbeddingExperimentSpec(
        name="multilingual_e5_small", backend="huggingface", model_name="intfloat/multilingual-e5-small"
    ),
    EmbeddingExperimentSpec(
        name="bge_m3",
        backend="huggingface",
        model_name=str(settings.bge_m3_model_path or "BAAI/bge-m3"),
    ),
)


def _compact_result(result: dict) -> dict:
    return {
        key: result[key]
        for key in (
            "name",
            "backend",
            "model_name",
            "collection",
            "chunk_count",
            "sample_count",
            "model_load_ms",
            "index_build_ms",
            "index_size_bytes",
        )
    } | {
        "strategies": {
            name: {
                key: strategy[key]
                for key in (
                    "metrics",
                    "metrics_by_question_type",
                    "latency_ms",
                    "mean_query_latency_ms",
                    "reranker",
                )
            }
            for name, strategy in result["strategies"].items()
        }
    }


def _markdown_report(report: dict) -> str:
    lines = [
        "# 岗位 RAG 语义模型与检索策略对比",
        "",
        f"自然问句样本数：{report['sample_count']}",
        "",
        "| 模型 | 策略 | MRR@3 | HitRate@1 | Recall@3 | Recall@5 | 平均查询延迟(ms) | Cross-Encoder |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in report["results"]:
        if result.get("status") != "ok":
            lines.append(f"| {result['name']} | - | - | - | - | - | - | 失败：{result['error_type']} |")
            continue
        for strategy_name, strategy in result["strategies"].items():
            metrics = strategy["metrics"]
            reranker = strategy["reranker"]
            rerank_text = f"{reranker['applied_samples']}/{result['sample_count']}"
            lines.append(
                f"| {result['name']} | {strategy_name} | {metrics['mrr_at_3']:.4f} | "
                f"{metrics['hit_rate_at_1']:.4f} | {metrics['recall_at_3']:.4f} | "
                f"{metrics['recall_at_5']:.4f} | {strategy['mean_query_latency_ms']:.2f} | {rerank_text} |"
            )
    lines.extend(
        [
            "",
            "## 模型资源开销",
            "",
            "| 模型 | 加载(ms) | 建索引(ms) | 索引大小(MiB) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for result in report["results"]:
        if result.get("status") == "ok":
            lines.append(
                f"| {result['name']} | {result['model_load_ms']:.1f} | {result['index_build_ms']:.1f} | "
                f"{result['index_size_bytes'] / 1024 / 1024:.2f} |"
            )
    lines.extend(["", "## 选型结果"])
    winner = report.get("selected_model")
    if winner:
        vector_metrics = winner["vector_metrics"]
        final_metrics = winner["hybrid_rerank_metrics"]
        lines.append(
            f"先依据 `vector-only` 选择 `{winner['model_name']}`（{winner['name']}）："
            f"MRR@3={vector_metrics['mrr_at_3']:.4f}、HitRate@1={vector_metrics['hit_rate_at_1']:.4f}、"
            f"Recall@5={vector_metrics['recall_at_5']:.4f}。随后验证 `hybrid_rerank`："
            f"MRR@3={final_metrics['mrr_at_3']:.4f}，Cross-Encoder 实际应用于全部 "
            f"{report['sample_count']} 条样本。"
        )
    else:
        lines.append("没有模型完成全部样本的 Cross-Encoder 重排；生产配置未变更。")
    evidence = report.get("selection_evidence")
    if evidence:
        ci = evidence["paired_bootstrap_95_ci"]
        lines.extend(
            [
                "",
                "## 配对选择证据",
                f"- `{evidence['winner']}` 相对 `{evidence['runner_up']}` 的配对 MRR@3 差值："
                f"{evidence['mrr_at_3_delta']:.4f}，95% Bootstrap CI [{ci[0]:.4f}, {ci[1]:.4f}]。",
                f"- 查询级胜/平/负：{evidence['wins']}/{evidence['ties']}/{evidence['losses']}；"
                f"差异是否明确：{evidence['statistically_clear']}。",
            ]
        )
    diagnostics = report.get("dataset_diagnostics", {})
    lines.extend(
        [
            "",
            "## 口径",
            "- 查询为精确 `job_id` 已解析后的自然中文提问，不包含公司、岗位标题、目标片段关键词或“请重点说明”。",
            "- `vector` 为 Chroma 向量召回；`hybrid` 为向量与 BM25 的 RRF 融合；`hybrid_rerank` 在融合候选上执行 Cross-Encoder。",
            "- Embedding 只按 `vector-only` 的 MRR@3、Recall@5、HitRate@1 和延迟排序；混合检索与重排只用于后验验证。",
            f"- 数据诊断：唯一问句 {diagnostics.get('unique_query_count', 0)}/{report['sample_count']}；"
            f"平均相关 chunk 数 {diagnostics.get('mean_relevant_chunks_per_query', 0):.2f}；"
            f"平均候选池 {diagnostics.get('mean_candidate_pool_size', 0):.2f}。",
            f"- 当前警告：{', '.join(diagnostics.get('warnings', [])) or '无'}。有警告时结果只能作为工程基线，不能作为最终效果声明。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    output_dir = settings.data_dir / "eval_dataset" / "job_rag" / "model_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = JobCatalog(settings.job_catalog_path)
    source = JobKnowledgeIngestion(
        catalog=catalog,
        source_corpus_dir=settings.source_corpus_dir,
        vector_db_dir=settings.vector_db_dir,
        collection_name=settings.job_collection_name,
    )
    try:
        samples = build_job_eval_samples(catalog, source, limit=80)
    finally:
        source.close()

    successful: list[dict] = []
    failures: list[dict] = []
    for spec in MODEL_SPECS:
        try:
            result = run_model_experiment(
                spec,
                samples=samples,
                catalog=catalog,
                source_corpus_dir=settings.source_corpus_dir,
                experiment_dir=output_dir / "indexes",
                reranker_enabled=False,
                reranker_model="BAAI/bge-reranker-v2-m3",
                reranker_local_files_only=True,
                candidate_k=settings.job_retrieval_candidate_k,
                rrf_k=settings.job_retrieval_rrf_k,
                strategies=("vector", "hybrid"),
                embedding_local_files_only=True,
                rerank_weight=settings.job_reranker_weight,
            )
            successful.append(result)
        except Exception as exc:
            failures.append(
                {
                    "status": "failed",
                    "name": spec.name,
                    "backend": spec.backend,
                    "model_name": spec.model_name or "hash",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    provisional = select_provisional_embedding_winner(successful, strategy="vector")
    selection_evidence = embedding_selection_evidence(successful, strategy="vector")
    if provisional is not None:
        rerank_run = run_model_experiment(
            EmbeddingExperimentSpec(
                name=provisional["name"],
                backend=provisional["backend"],
                model_name=None if provisional["backend"] == "hash" else provisional["model_name"],
            ),
            samples=samples,
            catalog=catalog,
            source_corpus_dir=settings.source_corpus_dir,
            experiment_dir=output_dir / "indexes",
            reranker_enabled=True,
            reranker_model="BAAI/bge-reranker-v2-m3",
            reranker_local_files_only=True,
            candidate_k=settings.job_retrieval_candidate_k,
            rrf_k=settings.job_retrieval_rrf_k,
            strategies=("hybrid_rerank",),
            embedding_local_files_only=True,
            rerank_weight=settings.job_reranker_weight,
        )
        provisional["strategies"].update(rerank_run["strategies"])
    winner = select_embedding_winner([provisional] if provisional is not None else [], sample_count=len(samples))
    selection = None
    if winner is not None:
        strategy = winner["strategies"]["hybrid_rerank"]
        selection = {
            "name": winner["name"],
            "backend": winner["backend"],
            "model_name": winner["model_name"],
            "collection": winner["collection"],
            "selection_strategy": "vector",
            "vector_metrics": winner["strategies"]["vector"]["metrics"],
            "hybrid_rerank_metrics": strategy["metrics"],
            "latency_ms": strategy["latency_ms"],
            "reranker": strategy["reranker"],
        }
    report = {
        "sample_count": len(samples),
        "results": [{"status": "ok", **_compact_result(result)} for result in successful] + failures,
        "selected_model": selection,
        "selection_evidence": selection_evidence,
        "dataset_diagnostics": (
            successful[0]["strategies"]["vector"]["report"]["diagnostics"] if successful else {}
        ),
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(_markdown_report(report), encoding="utf-8-sig")
    print(json.dumps({"report": str(output_dir / "report.json"), "selected_model": selection}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
