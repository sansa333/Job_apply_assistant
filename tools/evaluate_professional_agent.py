from __future__ import annotations

import json

from app.config import settings
from app.evaluation.professional_eval import evaluate_dataset


def markdown_report(report: dict) -> str:
    extraction = report["extraction"]
    matching = report["matching"]
    trajectory = report["agent_trajectory"]
    validation = report["generation_validation"]
    lines = [
        "# Evidence-Grounded Job Agent v1 评测报告",
        "",
        f"- 数据集：`{report['dataset']['name']}` v{report['dataset']['version']}",
        f"- 标签状态：{report['dataset']['label_status']}",
        f"- 基线：`{report['baseline']['name']}` / `{report['baseline']['scoring_version']}`",
        "",
        "## 结构化 JD 解析",
        "",
        f"- 岗位数：{extraction['job_count']}",
        f"- Skill Micro-F1：{extraction['skill_micro_f1']:.4f}",
        f"- Skill Macro-F1：{extraction['skill_macro_f1']:.4f}",
        f"- 结构化切分相对全文词表 F1 变化：{extraction['ablation']['structured_minus_raw_f1']:+.4f}",
        f"- P95 延迟：{extraction['p95_latency_ms']:.2f} ms",
        "",
        "## 简历–岗位匹配",
        "",
        f"- 标注对数：{matching['pair_count']}",
        f"- Accuracy：{matching['accuracy']:.4f}",
        f"- Macro-F1：{matching['macro_f1']:.4f}",
        f"- Spearman：{matching['spearman_score_vs_label']:.4f}",
        f"- mean nDCG@5：{matching['mean_ndcg_at_5']:.4f}",
        f"- mean MRR（首个 high）：{matching['mean_mrr_first_high']:.4f}",
        f"- P95 延迟：{matching['p95_latency_ms']:.2f} ms",
        "",
        "### 匹配消融",
        "",
        "| 方法 | Macro-F1 | Spearman | mean nDCG@5 |",
        "| --- | ---: | ---: | ---: |",
        "| 技能重合基线 | {f1:.4f} | {rho:.4f} | {ndcg:.4f} |".format(
            f1=matching["ablation"]["skill_overlap_macro_f1"],
            rho=matching["ablation"]["skill_overlap_spearman"],
            ndcg=matching["ablation"]["skill_overlap_mean_ndcg_at_5"],
        ),
        "| Requirement–Evidence 加权 | {f1:.4f} | {rho:.4f} | {ndcg:.4f} |".format(
            f1=matching["ablation"]["evidence_weighted_macro_f1"],
            rho=matching["ablation"]["evidence_weighted_spearman"],
            ndcg=matching["ablation"]["evidence_weighted_mean_ndcg_at_5"],
        ),
        "",
        "### 开发集校准 / 留出测试",
        "",
        f"- Development pairs：{matching['calibrated_holdout']['development_pair_count']}；Test pairs：{matching['calibrated_holdout']['test_pair_count']}",
        "- 技能重合基线阈值：medium={m:.0f}, high={h:.0f}；Test Macro-F1={f1:.4f}，Spearman={rho:.4f}".format(
            m=matching["calibrated_holdout"]["skill_overlap"]["medium_threshold"],
            h=matching["calibrated_holdout"]["skill_overlap"]["high_threshold"],
            f1=matching["calibrated_holdout"]["skill_overlap"]["test_macro_f1"],
            rho=matching["calibrated_holdout"]["skill_overlap"]["test_spearman"],
        ),
        "- Evidence 加权阈值：medium={m:.0f}, high={h:.0f}；Test Macro-F1={f1:.4f}，Spearman={rho:.4f}".format(
            m=matching["calibrated_holdout"]["evidence_weighted"]["medium_threshold"],
            h=matching["calibrated_holdout"]["evidence_weighted"]["high_threshold"],
            f1=matching["calibrated_holdout"]["evidence_weighted"]["test_macro_f1"],
            rho=matching["calibrated_holdout"]["evidence_weighted"]["test_spearman"],
        ),
        "",
        "## Agent 轨迹",
        "",
        f"- 场景数：{trajectory['case_count']}",
        f"- 工具序列准确率：{trajectory['exact_tool_sequence_accuracy']:.4f}",
        f"- 终态准确率：{trajectory['terminal_status_accuracy']:.4f}",
        f"- Next-action 准确率：{trajectory['next_action_accuracy']:.4f}",
        "",
        "## 生成安全校验",
        "",
        f"- 场景数：{validation['case_count']}",
        f"- Case Exact Accuracy：{validation['exact_case_accuracy']:.4f}",
        f"- Finding F1：{validation['finding_f1']:.4f}",
        "",
        "## 匹配 Bad Cases",
        "",
    ]
    bad_cases = [item for item in matching["details"] if item["expected_label"] != item["predicted_label"]]
    if not bad_cases:
        lines.append("- 当前银标集无分档错误；仍需第二标注者与更大 hard-negative 测试集。")
    else:
        lines.extend(
            f"- {item['annotation_id']}: expected={item['expected_label']}, predicted={item['predicted_label']}, score={item['score']:.2f}"
            for item in bad_cases
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "本报告是小规模、可重复的工程回归基线。匹配标签仍为单人首轮银标，不能表述为 HR 专家金标，也不能外推为录用概率。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    dataset_dir = settings.data_dir / "eval_dataset" / "job_agent_v1"
    report = evaluate_dataset(dataset_dir)
    output_dir = dataset_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "baseline_report.md").write_text(markdown_report(report), encoding="utf-8-sig")
    print(json.dumps({
        "report": str(output_dir / "baseline_report.json"),
        "extraction_f1": report["extraction"]["skill_micro_f1"],
        "matching_macro_f1": report["matching"]["macro_f1"],
        "matching_ndcg_at_5": report["matching"]["mean_ndcg_at_5"],
        "trajectory_accuracy": report["agent_trajectory"]["exact_tool_sequence_accuracy"],
        "validation_f1": report["generation_validation"]["finding_f1"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
