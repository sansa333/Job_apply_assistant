"""Summarize a frozen business run and preserve paired failure analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.domestic import paired_comparison, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text(encoding="utf-8"))
    jobs = {r["job_id"]: r for r in read_jsonl(args.dataset / "jobs.jsonl")}
    labels = {}
    for r in read_jsonl(args.dataset / "search_qrels.jsonl"):
        labels.setdefault(r["query_id"], {})[r["job_id"]] = r
    models = {m["name"]: m for m in report["models"]}
    small, m3 = models["small_512"], models["m3_1024"]
    test = lambda rows: [r for r in rows if r["split"] == "test"]
    comparisons = {}
    for label, left, right in [
        ("m3_hybrid_vs_m3_dense", m3["discovery"]["dense"], m3["discovery"]["hybrid"]),
        ("m3_hybrid_vs_bm25", report["baselines"]["discovery"]["bm25"], m3["discovery"]["hybrid"]),
        ("m3_hybrid_vs_small_hybrid", small["discovery"]["hybrid"], m3["discovery"]["hybrid"]),
    ]:
        comparisons[label] = {}
        for metric in ("mrr_at_10", "recall_at_20", "ndcg_at_10"):
            comparison = paired_comparison(test(left["details"]), test(right["details"]), metric)
            comparison["bootstrap_unit"] = "query intent"
            comparisons[label][metric] = comparison
    dense_small = {r["query_id"]: r for r in test(small["discovery"]["dense"]["details"])}
    dense_m3 = {r["query_id"]: r for r in test(m3["discovery"]["dense"]["details"])}
    hybrid = {r["query_id"]: r for r in test(m3["discovery"]["hybrid"]["details"])}
    failures = []
    for qid, a in dense_small.items():
        b = dense_m3[qid]
        if a["ndcg_at_10"] is None:
            continue
        def top(row):
            return [{"id": j, "title": jobs[j]["job_title"], "company": jobs[j]["company_name"],
                "grade": labels[qid][j]["grade"], "facets": labels[qid][j]["facet_matches"]} for j in row["ranking"][:3]]
        failures.append({"query": a["query"], "query_id": qid,
            "delta_ndcg": b["ndcg_at_10"] - a["ndcg_at_10"], "small": top(a), "m3": top(b),
            "hybrid": top(hybrid[qid]), "candidate_pool": a["candidate_pool"], "relevant_jobs": a["relevant_count"]})
    failures.sort(key=lambda r: r["delta_ndcg"])
    evidence_changes = []
    se = {r["query_id"]: r for r in test(small["evidence"]["details"])}
    for row in test(m3["evidence"]["details"]):
        before = se[row["query_id"]]
        if row["mrr_at_3"] != before["mrr_at_3"] or row["complete_support_at_2"] != before["complete_support_at_2"]:
            evidence_changes.append({"query": row["query"], "query_id": row["query_id"],
                "small_mrr": before["mrr_at_3"], "m3_mrr": row["mrr_at_3"],
                "small_complete_at_2": before["complete_support_at_2"], "m3_complete_at_2": row["complete_support_at_2"],
                "support_quotes": row["support_quotes"]})
    audit = {"comparisons": comparisons, "discovery_changes": failures, "evidence_changes": evidence_changes,
        "caution": "Post-run diagnosis, not new validation. Do not tune on these test cases and retain the same test claim."}
    (args.run / "diagnosis.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 国内招聘业务评测原始汇总", "", "标签：原文证据锚点由助手编写；搜索使用需求规则银标。非人工金标、非真实用户日志。", "",
        "## 全库搜索：18 个有完全相关岗位的 test 查询（共 20 题）", "",
        "| 策略 | MRR@10 | Recall@20 | nDCG@10 |", "|---|---:|---:|---:|"]
    rows = [(name, item) for name, item in report["baselines"]["discovery"].items()]
    rows += [(m["name"] + "/" + strategy, item) for m in report["models"] for strategy, item in m["discovery"].items()]
    for name, item in rows:
        g = item["groups"]["test"]
        lines.append(f"| {name} | {g['mrr_at_10']:.4f} | {g['recall_at_20']:.4f} | {g['ndcg_at_10']:.4f} |")
    lines += ["", "## 指定岗位证据：24 个有答案 test 查询（另 16 个缺失事实问题）", "",
        "| 策略 | MRR@3 | Recall@5 | Recall@2 | 完整支持@1 | 完整支持@2 |", "|---|---:|---:|---:|---:|---:|"]
    for name, item in list(report["baselines"]["evidence"].items()) + [(m["name"], m["evidence"]) for m in report["models"]]:
        g = item["groups"]["test"]
        coverage = [f"{g[k]:.4f}" if k in g else "未估计" for k in ("complete_support_at_1", "complete_support_at_2")]
        recall5 = f"{g['recall_at_5']:.4f}" if "recall_at_5" in g else "未检索 Top5"
        lines.append(f"| {name} | {g['mrr_at_3']:.4f} | {recall5} | {g['recall_at_2']:.4f} | {' | '.join(coverage)} |")
    if "evidence_pool_audit" in report:
        lines += ["", "同岗候选池审计（仅有答案问题）：", "",
            "```json", json.dumps(report["evidence_pool_audit"], ensure_ascii=False, indent=2), "```",
            "", "候选池≤5时 Recall@5 易饱和；没有>5样本的子集不提供指标，不能以满分推断模型全面胜出。"]
    lines += ["", "## 全库搜索退步最大的三个查询", ""]
    for r in failures[:3]:
        lines.append(f"- {r['query_id']}：{r['query']}；M3 − small nDCG@10={r['delta_ndcg']:+.4f}。")
    lines += ["", "完整逐题排名、原文和标签见 report.json 与冻结数据；配对区间与错误岗位见 diagnosis.json。", ""]
    (args.run / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"comparisons": comparisons, "evidence_changes": evidence_changes, "worst_queries": failures[:3]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
