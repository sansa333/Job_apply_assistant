from __future__ import annotations

import json

from app.config import settings
from app.evaluation.retrieval_strategies import evaluate_bge_m3_strategies
from app.evaluation.retrieval_v2 import load_retrieval_v2, validate_retrieval_v2


def _markdown(report: dict) -> str:
    lines = [
        "# BGE-M3 Retrieval V2 检索策略消融报告",
        "",
        "固定 BGE-M3、数据、切块、查询、qrels 和候选池，仅改变检索策略。",
        "本报告只读取 Development；当前 qrels 为 silver，不表述为人工 gold Test 结果。",
    ]
    for scope_name in ("job_scoped", "hard_pool"):
        scope = report[scope_name]
        lines.extend(
            [
                "",
                f"## {scope_name}",
                "",
                "| 策略 | MRR@3 | HitRate@1 | Recall@3 | Recall@5 | nDCG@5 | First-stage Recall@20 | P95(ms) | Rerank应用 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for result in scope["strategy_results"]:
            metrics = result["metrics"]
            resources = result["resources"]
            lines.append(
                f"| {result['strategy']} | {metrics['mrr_at_3']:.4f} | "
                f"{metrics['hit_rate_at_1']:.4f} | {metrics['recall_at_3']:.4f} | "
                f"{metrics['recall_at_5']:.4f} | {metrics['ndcg_at_5']:.4f} | "
                f"{metrics['first_stage_recall_at_20']:.4f} | "
                f"{resources['p95_query_total_ms']:.2f} | {resources['reranker_applied_count']} |"
            )
        lines.extend(["", "### 相对 Dense 的配对证据", ""])
        for comparison in scope["paired_vs_dense"]:
            ci = comparison["bootstrap_95_ci"]
            lines.append(
                f"- `{comparison['candidate']}`：ΔMRR@3={comparison['delta']:.4f}，"
                f"95% CI [{ci[0]:.4f}, {ci[1]:.4f}]，"
                f"胜/平/负={comparison['wins']}/{comparison['ties']}/{comparison['losses']}，"
                f"P95增量={comparison['p95_latency_delta_ms']:.2f}ms，"
                f"质量门禁={comparison['quality_gate_passed']}。"
            )
    lines.extend(
        [
            "",
            "## 结论与使用边界",
            "",
            "- 当前 silver Development 上以 `dense` 作为 BGE-M3 上线基线；只有候选策略的配对 Bootstrap 95% CI 下界大于 0，才允许替换基线。",
            "- hard-pool 查询不包含 job_id/company/title，但 qrels 将跨岗位证据统一记为 0；语义正确但岗位错误的文本会被判负，因此该范围只作为跨岗位污染压力测试。",
            "- 确定性 query construction 存在 focus 与 query_type 混合的样本；全量重排退化是人工复核优先级信号，不能直接解释为 reranker 模型能力差。",
            "- 在双人标注、仲裁和一致性门禁完成前，不用该报告发布 Test 或人工 gold 结论。",
            "",
            "## 参数与边界",
            "",
            f"- candidate_k={report['parameters']['candidate_k']}，RRF k={report['parameters']['rrf_k']}，"
            f"reranker_weight={report['parameters']['reranker_weight']}。",
            "- BM25 使用项目当前中文字符/英文词元规则；Dense 使用 1024 维归一化 BGE-M3 向量。",
            "- `*_rerank_blend` 复现线上 0.2 权重保守融合；`*_rerank_full` 直接采用 Cross-Encoder 全量排序。",
            "- Reranker 模型加载和首次预热不计入逐查询 P95，在线延迟包含实际重排推理。",
            "- job_scoped 对齐线上精确岗位内排序；hard_pool 使用每条查询固定 50 个跨岗位困难候选。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    dataset_dir = settings.data_dir / "eval_dataset" / "job_retrieval_v2"
    report_dir = dataset_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_retrieval_v2(dataset_dir)
    quality = validate_retrieval_v2(dataset)
    if not quality["valid"]:
        raise ValueError(f"dataset quality gate failed: {quality['errors']}")

    model_name = str(settings.bge_m3_model_path or settings.hf_embedding_model)
    parameters = {
        "candidate_k": 20,
        "rrf_k": settings.job_retrieval_rrf_k,
        "reranker_weight": settings.job_reranker_weight,
    }
    full_scopes = {
        scope: evaluate_bge_m3_strategies(
            dataset,
            model_name=model_name,
            reranker_model=settings.reranker_model,
            split="development",
            evaluation_scope=scope,
            **parameters,
        )
        for scope in ("job_scoped", "hard_pool")
    }
    report = {
        "dataset_name": dataset["manifest"]["name"],
        "dataset_version": dataset["manifest"]["version"],
        "annotation_status": dataset["manifest"]["annotation_status"],
        "evaluation_mode": "development_only",
        "quality_gate": quality,
        "model_name": model_name,
        "parameters": parameters,
        "job_scoped": full_scopes["job_scoped"],
        "hard_pool": full_scopes["hard_pool"],
    }
    json_path = report_dir / "bge_m3_strategy_ablation_report.json"
    markdown_path = report_dir / "bge_m3_strategy_ablation_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "report": str(json_path.resolve()),
                "evaluation_mode": report["evaluation_mode"],
                "model_name": model_name,
                "job_scoped": [
                    {"strategy": item["strategy"], **item["metrics"], **item["resources"]}
                    for item in report["job_scoped"]["strategy_results"]
                ],
                "hard_pool": [
                    {"strategy": item["strategy"], **item["metrics"], **item["resources"]}
                    for item in report["hard_pool"]["strategy_results"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
