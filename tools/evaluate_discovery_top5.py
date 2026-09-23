"""Score frozen full-catalog rankings at five, without rerunning or tuning retrieval."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from app.evaluation.business import graded_metrics, random_expectation, summarize
from app.evaluation.domestic import paired_comparison, read_jsonl, verify_dataset


def score_method(item, queries, labels, eligible, *, random=False):
    rows, seen = [], set()
    for old in item["details"]:
        qid = old["query_id"]
        if qid not in queries or qid in seen:
            raise ValueError("Unknown or duplicate query")
        seen.add(qid)
        ranking = old["ranking"]
        if len(ranking) != len(set(ranking)) or not set(ranking) <= set(eligible[qid]):
            raise ValueError("Duplicate, unknown or hard-filter-ineligible ranked job")
        metrics = (random_expectation(eligible[qid], labels[qid], relevant_grade=2, ks=(3, 5)) if random else
                   graded_metrics(ranking, labels[qid], relevant_grade=2, ks=(3, 5)))
        for key in ("mrr_at_3", "recall_at_5", "hit_at_5", "ndcg_at_5"):
            if key in old:
                a, b = old[key], metrics[key]
                if (a is None) != (b is None) or (a is not None and not math.isclose(a, b, abs_tol=1e-12)):
                    raise ValueError(f"Saved metric disagrees with frozen labels: {qid}/{key}")
        rows.append({"query_id": qid, "query": queries[qid]["query"], "split": queries[qid]["split"],
            "group_id": queries[qid]["group_id"], **metrics, "ranking": ranking,
            "top5_grades": [labels[qid][j] for j in ranking[:5]],
            "returned_at_5": min(5, len(eligible[qid]) if random else len(ranking)),
            "oracle_precision_at_5": min(5, metrics["relevant_count"]) / 5})
    if seen != queries.keys():
        raise ValueError("Incomplete query coverage")
    groups = {}
    for split in ("development", "test"):
        split_rows = [r for r in rows if r["split"] == split]
        answerable = [r for r in split_rows if r["relevant_count"] > 0]
        missing = [r for r in split_rows if r["relevant_count"] == 0]
        groups[split] = {"answerable_only": summarize(answerable),
            "all_query_precision_at_5": sum(r["precision_at_5"] for r in split_rows) / len(split_rows) if split_rows else None,
            "all_query_hit_at_5": sum(r["hit_at_5"] or 0 for r in split_rows) / len(split_rows) if split_rows else None,
            "total_queries": len(split_rows), "no_relevant_queries": len(missing),
            "no_relevant_returned_nonempty": sum(r["returned_at_5"] > 0 for r in missing),
            "oracle_precision_at_5": sum(r["oracle_precision_at_5"] for r in answerable) / len(answerable) if answerable else None,
            "answerable_with_fewer_than_5_relevant": sum(r["relevant_count"] < 5 for r in answerable)}
    return {"groups": groups, "details": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = verify_dataset(args.dataset)
    source_path = args.run / "report.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source["dataset"]["files"] != manifest["files"]:
        raise ValueError("Run/dataset mismatch")
    queries = {r["query_id"]: r for r in read_jsonl(args.dataset / "search_queries.jsonl")}
    labels, eligible = {}, {qid: [] for qid in queries}
    for row in read_jsonl(args.dataset / "search_qrels.jsonl"):
        labels.setdefault(row["query_id"], {})[row["job_id"]] = row["grade"]
        if row["eligible"]:
            eligible[row["query_id"]].append(row["job_id"])
    methods = {name: score_method(item, queries, labels, eligible, random=name == "random")
        for name, item in source["baselines"]["discovery"].items()}
    for model in source["models"]:
        for strategy, item in model["discovery"].items():
            methods[f"{model['name']}/{strategy}"] = score_method(item, queries, labels, eligible)
    comparisons = {}
    for base in ("m3_1024/dense", "bm25", "small_512/hybrid"):
        comparisons[f"m3_1024/hybrid_vs_{base}"] = {}
        for metric in ("precision_at_5", "hit_at_5", "ndcg_at_5"):
            rows = lambda method: [r for r in methods[method]["details"] if r["split"] == "test" and r["relevant_count"] > 0]
            result = paired_comparison(rows(base), rows("m3_1024/hybrid"), metric)
            result["bootstrap_unit"] = "query intent"
            comparisons[f"m3_1024/hybrid_vs_{base}"][metric] = result
    report = {"task": "full-catalog job discovery; post-hoc fixed-ranking supplementary metrics",
        "definitions": {"precision_at_5": "number of grade-2 jobs in top5 / 5; missing slots count as nonrelevant",
            "hit_at_5": "at least one grade-2 job in top5", "ndcg_at_5": "gain 2^grade-1, grades 0/1/2",
            "main_population": "macro mean over 18 answerable test queries; all-20 P/Hit and no-result queries separately reported",
            "no_tuning": "Dataset, labels, models and rankings unchanged. No new independent test or end-to-end claim."},
        "annotation_status": manifest["annotation_status"], "dataset_files": manifest["files"],
        "methods": methods, "comparisons": comparisons,
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (source_path, args.dataset / "manifest.json", Path(__file__), Path("app/evaluation/business.py"), Path("app/evaluation/domestic.py"))}}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 全库岗位搜索 Top5 补充评测", "", "固定 v4.1 语料与既有排名；条件银标回归集，非独立人工金标。", "",
        "主表：18 个有相关岗位测试题，Precision/Hit 以 grade=2 为相关，nDCG 使用 0/1/2 分级。", "",
        "| 方案 | Precision@5 | Hit@5 | nDCG@5 |", "|---|---:|---:|---:|"]
    for name, item in methods.items():
        g = item["groups"]["test"]["answerable_only"]
        lines.append(f"| {name} | {g['precision_at_5']:.4f} | {g['hit_at_5']:.4f} | {g['ndcg_at_5']:.4f} |")
    lines += ["", "Precision 分母固定为5；不足5个结果也不缩小分母。Hit 是至少命中一个，不是五个都相关。", "",
        "另2个无相关岗位测试题的 P/Hit 记0，并另报是否返回无关候选；详见 report.json。",
        "这是已保存实际检索排名的补充计分，没有重新生成排名或以新指标调参。", ""]
    (args.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
