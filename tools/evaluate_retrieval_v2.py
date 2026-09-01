from __future__ import annotations

import argparse
import json

from app.config import settings
from app.evaluation.retrieval_v2 import (
    RetrievalV2ModelSpec,
    evaluate_model,
    load_retrieval_v2,
    paired_bootstrap,
    select_on_development,
    validate_retrieval_v2,
)


MODEL_SPECS = (
    RetrievalV2ModelSpec(name="hash_baseline", backend="hash", model_name=None),
    RetrievalV2ModelSpec(
        name="bge_small_zh",
        backend="huggingface",
        model_name="BAAI/bge-small-zh-v1.5",
        query_prefix="为这个句子生成表示以用于检索相关文章：",
    ),
    RetrievalV2ModelSpec(
        name="multilingual_e5_small",
        backend="huggingface",
        model_name="intfloat/multilingual-e5-small",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    RetrievalV2ModelSpec(
        name="bge_m3",
        backend="huggingface",
        model_name=str(settings.bge_m3_model_path or "BAAI/bge-m3"),
    ),
)


def _compact(result: dict) -> dict:
    if result.get("status") != "ok":
        return result
    return {key: value for key, value in result.items() if key != "details"}


def _markdown(report: dict) -> str:
    quality = report["dataset_quality"]
    stats = quality["statistics"]
    lines = [
        "# Job Retrieval V2 Embedding 选型报告",
        "",
        f"数据质量门禁：{'通过' if quality['valid'] else '失败'}",
        f"标注状态：{report['annotation_status']}（不得表述为人工 gold test set）",
        "",
        "## 数据规模",
        "",
        f"- 真实历史岗位：{stats['jobs']}；岗位族：{len(stats['occupation_family_distribution'])}",
        f"- 原子证据单元：{stats['evidence_units']}；自然查询：{stats['queries']}；唯一查询：{stats['unique_queries']}",
        f"- 多相关查询比例：{stats['multi_relevant_query_ratio']:.4f}；平均相关证据数：{stats['mean_relevant_per_query']:.2f}",
        f"- 固定困难候选池：每个查询 {stats['candidate_pool_size']} 个证据单元",
        f"- 岗位级 split：{stats['split_job_distribution']}",
        "",
        "## 主评测：精确岗位内 Development 模型对比",
        "",
        "该任务与线上流程一致：上游已经解析精确 job_id，只在该 JD 的 8–20 个证据单元内排序。",
        "",
        "| 模型 | MRR@3 | HitRate@1 | Recall@3 | Recall@5 | nDCG@5 | P95查询(ms) | 维度 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report["development_results"]:
        if result.get("status") != "ok":
            lines.append(f"| {result['name']} | - | - | - | - | - | - | - | 失败：{result['error_type']} |")
            continue
        metrics = result["metrics"]
        resources = result["resources"]
        lines.append(
            f"| {result['name']} | {metrics['mrr_at_3']:.4f} | {metrics['hit_rate_at_1']:.4f} | "
            f"{metrics['recall_at_3']:.4f} | {metrics['recall_at_5']:.4f} | {metrics['ndcg_at_5']:.4f} | "
            f"{resources['p95_query_total_ms']:.2f} | {resources['dimension']} |"
        )
    lines.extend(["", "## Development 集选型证据"])
    selection = report.get("selection")
    if selection:
        ci = selection["paired_evidence"]["bootstrap_95_ci"] if selection.get("paired_evidence") else None
        lines.append(
            f"按冻结规则选出 `{selection['winner_model']}`。相对 `{selection.get('runner_up_model')}` 的 MRR@3 "
            f"配对差值为 {selection['paired_evidence']['delta']:.4f}，95% CI "
            f"[{ci[0]:.4f}, {ci[1]:.4f}]，替换门禁：{selection['paired_evidence']['replacement_gate_passed']}。"
        )
    else:
        lines.append("没有可用模型，未执行测试集评测。")
    test = report.get("frozen_test_result")
    lines.extend(["", "## 主评测：精确岗位内冻结 Test 结果"])
    if test and test.get("status") == "ok":
        metrics = test["metrics"]
        lines.extend(
            [
                f"只运行 Development 集选出的 `{test['name']}`，未使用 Test 集重新选型。",
                "",
                f"- MRR@3：{metrics['mrr_at_3']:.4f}",
                f"- HitRate@1：{metrics['hit_rate_at_1']:.4f}",
                f"- Recall@3：{metrics['recall_at_3']:.4f}",
                f"- Recall@5：{metrics['recall_at_5']:.4f}",
                f"- nDCG@5：{metrics['ndcg_at_5']:.4f}",
            ]
        )
    else:
        lines.append("未生成冻结测试集结果。")
    stress_development = report.get("stress_development_result")
    stress_test = report.get("stress_test_result")
    lines.extend(["", "## 压力测试：50 个跨岗位困难候选"])
    for label, result in (("Development", stress_development), ("Test", stress_test)):
        if result and result.get("status") == "ok":
            metrics = result["metrics"]
            lines.append(
                f"- {label}：MRR@3={metrics['mrr_at_3']:.4f}，HitRate@1={metrics['hit_rate_at_1']:.4f}，"
                f"Recall@3={metrics['recall_at_3']:.4f}，Recall@5={metrics['recall_at_5']:.4f}，"
                f"nDCG@5={metrics['ndcg_at_5']:.4f}。"
            )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- 当前 qrels 由确定性查询构造产生，状态为 silver，必须完成双人独立标注和第三人仲裁后才可升级为 gold。",
            "- Embedding 仅在 Development 集选择；Test 集只运行一次选中模型。",
            "- 主指标来自与生产一致的 job_scoped 检索；50 候选 hard_pool 只作为压力测试，不用于包装主效果。",
            "- 本报告衡量检索层，不代表匹配评分、Agent 成功率或生成文本质量。",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Retrieval V2 embedding candidates.")
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Run model selection and stress evaluation on Development without reading Test.",
    )
    args = parser.parse_args(argv)
    dataset_dir = settings.data_dir / "eval_dataset" / "job_retrieval_v2"
    report_dir = dataset_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_retrieval_v2(dataset_dir)
    quality = validate_retrieval_v2(dataset)
    if not quality["valid"]:
        raise ValueError(f"dataset quality gate failed: {quality['errors']}")

    development_results: list[dict] = []
    full_results: list[dict] = []
    spec_by_name = {spec.name: spec for spec in MODEL_SPECS}
    for spec in MODEL_SPECS:
        try:
            result = evaluate_model(
                spec,
                dataset,
                split="development",
                evaluation_scope="job_scoped",
                local_files_only=True,
            )
            development_results.append(_compact(result))
            full_results.append(result)
        except Exception as exc:
            failure = {
                "status": "failed",
                "name": spec.name,
                "backend": spec.backend,
                "model_name": spec.model_name or "hash",
                "evaluation_scope": "job_scoped",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            development_results.append(failure)

    winner, runner_up = select_on_development(full_results, evaluation_scope="job_scoped")
    selection = None
    frozen_test = None
    stress_development = None
    stress_test = None
    if winner is not None:
        evidence = paired_bootstrap(winner, runner_up) if runner_up is not None else None
        selection = {
            "policy": "development_mrr_at_3_then_recall_at_5_then_hit_rate_at_1_then_p95_latency",
            "winner_model": winner["name"],
            "runner_up_model": runner_up["name"] if runner_up else None,
            "paired_evidence": evidence,
        }
        stress_development = evaluate_model(
            spec_by_name[winner["name"]],
            dataset,
            split="development",
            evaluation_scope="hard_pool",
            local_files_only=True,
        )
        if not args.development_only:
            frozen_test = evaluate_model(
                spec_by_name[winner["name"]],
                dataset,
                split="test",
                evaluation_scope="job_scoped",
                local_files_only=True,
            )
            stress_test = evaluate_model(
                spec_by_name[winner["name"]],
                dataset,
                split="test",
                evaluation_scope="hard_pool",
                local_files_only=True,
            )

    report = {
        "dataset_name": dataset["manifest"]["name"],
        "dataset_version": dataset["manifest"]["version"],
        "evaluation_mode": "development_only" if args.development_only else "development_and_frozen_test",
        "annotation_status": dataset["manifest"]["annotation_status"],
        "dataset_quality": quality,
        "development_results": development_results,
        "selection": selection,
        "frozen_test_result": _compact(frozen_test) if frozen_test else None,
        "stress_development_result": _compact(stress_development) if stress_development else None,
        "stress_test_result": _compact(stress_test) if stress_test else None,
    }
    report_stem = "embedding_development_report" if args.development_only else "embedding_selection_report"
    json_report_path = report_dir / f"{report_stem}.json"
    markdown_report_path = report_dir / f"{report_stem}.md"
    json_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_report_path.write_text(_markdown(report), encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "report": str(json_report_path),
                "evaluation_mode": report["evaluation_mode"],
                "quality_gate": quality["valid"],
                "selection": selection,
                "frozen_test_metrics": frozen_test["metrics"] if frozen_test else None,
                "stress_test_metrics": stress_test["metrics"] if stress_test else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
