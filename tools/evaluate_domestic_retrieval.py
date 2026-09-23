"""Run real domestic service retrieval against isolated frozen public snapshots.

python -m tools.evaluate_domestic_retrieval --output outputs/domestic_eval_20260918
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
from importlib.metadata import version
import time
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.config import settings
from app.domestic.service import DomesticJobService
from app.embeddings import get_embeddings
from app.evaluation.domestic import (
    aggregate, build_dataset, frozen_catalog, is_notice, paired_comparison,
    rank_metrics, read_jsonl, snapshot_records, verify_dataset,
)
from app.knowledge.ingestion import JobKnowledgeIngestion
from app.knowledge.models import JobRecord


class BorrowedIngestion:
    """A run owns one persistent index; service calls borrow it without reloading."""
    def __init__(self, ingestion):
        self.ingestion = ingestion

    def __getattr__(self, name):
        return getattr(self.ingestion, name)

    def close(self):
        pass


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_search(root: Path, records: list[JobRecord], queries: list[dict]) -> dict:
    details = []
    for stratum in ("individual_job", "notice"):
        subset = [r for r in records if ("notice" if is_notice(r) else "individual_job") == stratum]
        catalog = frozen_catalog(root / f"search_{stratum}.sqlite3", subset)
        service = DomesticJobService(catalog=catalog, sources={})
        for query in [q for q in queries if q["stratum"] == stratum]:
            start = time.perf_counter()
            result = service.search(keyword=query["keyword"], **query["filters"], candidate_id="eval_no_profile", limit=50)
            ranking = [row["job_id"] for row in result["matches"]]
            details.append({**query, **rank_metrics(ranking, set(query["relevant_ids"])),
                            "ranking": ranking, "total_candidates": result["total"],
                            "latency_ms": (time.perf_counter() - start) * 1000})
    return {"model_independent": True, "path": "DomesticJobService.search -> catalog filters -> rank_job -> pagination",
            "details": details, "all": aggregate(details),
            "by_stratum": {s: aggregate([r for r in details if r["stratum"] == s]) for s in ("individual_job", "notice")},
            "by_variant": {s: aggregate([r for r in details if r["variant"] == s]) for s in ("keywords", "natural_language")},
            "unanswerable": {"count": sum(r["relevant_count"] == 0 for r in details),
                             "returned_nonempty": sum(r["relevant_count"] == 0 and bool(r["ranking"]) for r in details)}}


def _groups(details: list[dict]) -> dict:
    groups = {}
    for stratum in ("individual_job", "notice"):
        for split in ("development", "test"):
            rows = [r for r in details if r["stratum"] == stratum and r["split"] == split]
            groups[f"{stratum}/{split}/all"] = aggregate(rows)
            groups[f"{stratum}/{split}/discriminative"] = aggregate([
                r for r in rows if 0 < r["relevant_count"] < r["pool_size"]
            ])
            groups[f"{stratum}/{split}/pool_gt_5"] = aggregate([r for r in rows if r["pool_size"] > 5])
    return groups


def evaluate_embedding(root: Path, dataset: Path, records: list[JobRecord], spec: dict) -> dict:
    verify_dataset(dataset)
    model_root = root / spec["name"]
    model_root.mkdir()
    # The entire index (including dimensions) is isolated under this fresh run.
    if root.resolve() not in model_root.resolve().parents:
        raise ValueError("Index must be within the evaluation run")
    catalog = frozen_catalog(model_root / "catalog.sqlite3", records)
    start = time.perf_counter()
    embeddings = get_embeddings(backend="huggingface", model_name=spec["model"], local_files_only=True,
                                max_length=spec["max_length"], device="cuda", use_fp16=True, batch_size=4)
    load_seconds = time.perf_counter() - start
    ingestion = JobKnowledgeIngestion(
        catalog=catalog, source_corpus_dir=model_root / "sources", vector_db_dir=model_root / "indexes",
        collection_name="domestic_eval", embeddings=embeddings, embedding_backend="huggingface", embedding_model=spec["model"],
    )
    borrowed = BorrowedIngestion(ingestion)
    service = DomesticJobService(catalog=catalog, sources={}, ingestion_factory=lambda borrowed=borrowed: borrowed)
    start = time.perf_counter()
    chunk_count = ingestion.rebuild_records(records)
    index_seconds = time.perf_counter() - start
    queries = read_jsonl(dataset / "queries.jsonl")
    qrels = defaultdict(set)
    for row in read_jsonl(dataset / "qrels.jsonl"):
        qrels[row["query_id"]].add(row["evidence_id"])
    evidence_rows = read_jsonl(dataset / "evidence.jsonl")
    expected = {r["evidence_id"]: r["text"] for r in evidence_rows}
    source_ids = {r["evidence_id"] for r in evidence_rows if r["metadata"].get("evidence_kind") != "derived_keywords"}
    for record in records:
        for doc in ingestion.get_documents_for_job(record.job_id):
            if expected.get(doc.metadata["chunk_id"]) != doc.page_content:
                raise ValueError("Production chunker differs from the frozen dataset")
    if chunk_count != len(expected):
        raise ValueError("Indexed corpus differs from frozen evidence")
    by_job = {r.job_id: r for r in records}
    search_ranks = {}
    for record in records:
        # Joint-stage audit: discover an exact company/title, then retrieve its
        # evidence only if the intended job was exposed in the first ten results.
        result = service.search(keyword=record.job_title, company_name=record.company_name,
                                candidate_id="eval_no_profile", limit=10)
        ranking = [r["job_id"] for r in result["matches"]]
        search_ranks[record.job_id] = ranking.index(record.job_id) + 1 if record.job_id in ranking else None
    details, latencies, source_details = [], [], []
    embeddings.embed_query("检索预热")
    try:
        with patch.object(settings, "job_retrieval_strategy", "vector"), patch.object(settings, "enable_reranker", False):
            for index, query in enumerate(queries):
                relevant = qrels[query["query_id"]]
                if not relevant:
                    details.append({**query, **rank_metrics([], set(), ks=(1, 3, 5)), "ranking": [], "latency_ms": None})
                    continue
                start = time.perf_counter()
                result = service.retrieve_job_evidence(job_id=query["job_id"], query=query["query"], k=5)
                latency = (time.perf_counter() - start) * 1000
                if any(doc["job_id"] != query["job_id"] for doc in result["documents"]):
                    raise ValueError("Cross-job leakage")
                ranking = [doc["chunk_id"] for doc in result["documents"]]
                metrics = rank_metrics(ranking, relevant, ks=(1, 3, 5))
                source_details.append({**query, **rank_metrics(ranking, relevant & source_ids, ks=(1, 3, 5)), "ranking": ranking})
                details.append({**query, **metrics, "ranking": ranking, "relevant_ids": sorted(relevant),
                                "latency_ms": latency, "search_rank": search_ranks[query["job_id"]],
                                "e2e_recall_at_5": metrics["recall_at_5"] if search_ranks[query["job_id"]] else 0.0,
                                "company": by_job[query["job_id"]].company_name})
                latencies.append(latency)
                if index % 300 == 0:
                    print(f"{spec['name']}: {index}/{len(queries)}", flush=True)
        tokenizer = embeddings._model.tokenizer
        token_lengths = [len(tokenizer.encode(text, truncation=False)) for text in expected.values()]
        config_hashes = {}
        for module in embeddings._model:
            config = getattr(getattr(module, "auto_model", None), "config", None)
            if config:
                config_hashes["model_config_sha256"] = hashlib.sha256(config.to_json_string().encode()).hexdigest()
                config_hashes["hub_commit"] = getattr(config, "_commit_hash", None)
        output = {"spec": spec, "model_provenance": config_hashes, "load_seconds": load_seconds, "index_seconds": index_seconds,
                  "chunk_count": chunk_count, "dimension": len(embeddings.embed_query("维度")),
                  "embedding_policy": "CUDA FP16; normalized; no query prefix, matching production adapter; vector-only Chroma; rerank off",
                  "chunks_truncated": sum(n > spec["max_length"] for n in token_lengths),
                  "p50_ms": float(np.percentile(latencies, 50)), "p95_ms": float(np.percentile(latencies, 95)),
                  "groups": _groups(details), "details": details,
                  "source_only_groups": _groups(source_details), "source_only_details": source_details,
                  "unanswerable_queries_excluded": sum(not row["relevant_count"] for row in details)}
        save_json(model_root / "results.json", output)
        return output
    finally:
        ingestion.close()
        del ingestion, borrowed, service, embeddings
        gc.collect()
        import torch
        torch.cuda.empty_cache()


def compare(results: list[dict]) -> dict:
    comparisons = {}
    for candidate in results[1:]:
        comparison = {}
        for split in ("development", "test"):
            for subset in ("all", "discriminative", "pool_gt_5"):
                def selected(result):
                    return [row for row in result["details"] if row["stratum"] == "individual_job" and row["split"] == split
                            and (subset != "discriminative" or 0 < row["relevant_count"] < row["pool_size"])
                            and (subset != "pool_gt_5" or row["pool_size"] > 5)]
                comparison[f"{split}/{subset}"] = {metric: paired_comparison(selected(results[0]), selected(candidate), metric)
                                                   for metric in ("mrr_at_3", "recall_at_5")}
        comparisons[candidate["spec"]["name"]] = comparison
    return comparisons


def markdown(report: dict) -> str:
    manifest = report["dataset"]
    lines = ["# 国内秋招检索链路评测", "", "## 数据与结论边界", "",
             f"冻结快照：{manifest['jobs']} 条记录（{manifest['individual_jobs']} 个具体岗位、{manifest['notices']} 个公告合集）；"
             f"{manifest['evidence']} 个生产切块；{manifest['queries']} 个岗位内问题；{manifest['search_queries']} 个搜索问题。",
             "", f"标注状态：`{manifest['annotation_status']}`。本次初始标签为固定主题规则生成的 silver，未使用模型排名生成标签；人工复核前不能声称 gold。",
             f"{manifest['queries']} 是岗位×问题实例数，问题正文只有 {len({r['query'] for r in report['models'][0]['details']})} 种预设主题模板，并非同等数量的独立用户提问。",
             "数据来自既有国内快照，非本次重新抓取；历史状态不表示现在仍开放。缺失来源时间保留为未知。",
             "同正文重复岗位去重，按正文哈希分组划分 development/test；近似重复和跨公司模板仍可能相关。",
             "本轮修复基于此前错误分析，同源 test 已被观察，属于修正后的回归评测，不是新的盲测；标签仍是 silver。", "",
             "## 岗位搜索（与 embedding 无关）", "",
             "真实服务调用：查询词提取 → SQLite 过滤 → 全量候选规则排序 → 分页。未加载私人简历，使用默认技能偏好。",
             "具体岗位与公告分别建目录评测；混合目录上的精确岗位搜索在下方串联审计中检验。", "",
             "| 范围 | 有答案查询 | MRR@10 | Recall@10 | Recall@50 |", "|---|---:|---:|---:|---:|"]
    for name, row in report["search"]["by_stratum"].items():
        lines.append(f"| {name} | {row['answerable']} | {row['mrr_at_10']:.4f} | {row['recall_at_10']:.4f} | {row['recall_at_50']:.4f} |")
    lines += ["", "自然语言整句与关键词查询另存分组结果；无答案查询单独记录，不纳入 Recall/MRR 平均值。",
              "## BGE 模型对比：具体岗位 Test", "",
              "使用相同岗位、切块、查询、标签与 Chroma 检索路径；每个模型使用独立索引。",
              "512 对比为共同截断上限；M3-1024 对比为当前生产配置收益，包含上下文长度变化。",
              "两模型均不添加查询前缀，以保持现有生产 adapter 行为；这不是各模型独立调优后的最优性能比较。", "",
              "| 配置 | 有答案查询 | MRR@3 | Recall@5 | 有区分度子集 MRR@3 | 有区分度子集 Recall@5 | P95(ms)* |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for result in report["models"]:
        row = result["groups"]["individual_job/test/all"]
        hard = result["groups"]["individual_job/test/discriminative"]
        lines.append(f"| {result['spec']['name']} | {row['answerable']} | {row['mrr_at_3']:.4f} | {row['recall_at_5']:.4f} | {hard['mrr_at_3']:.4f} | {hard['recall_at_5']:.4f} | {result['p95_ms']:.2f} |")
    lines += ["", "* 延迟统计覆盖全部有答案评测问题；复用已加载模型与打开的索引，不包括首次模型加载、HTTP 及 LLM 生成。",
              "有区分度子集指至少一个相关片段、至少一个不相关片段；并不要求候选数超过 5。", "",
              "| 对比 small-512 | 指标 | 绝对差值 | 相对变化 | 配对 JD 分组 Bootstrap 95% CI |",
              "|---|---|---:|---:|---|"]
    for name, comparison in report["comparisons"].items():
        for metric, row in comparison["test/all"].items():
            relative = f"{row['relative_percent']:+.2f}%" if row["relative_percent"] is not None else "N/A"
            lines.append(f"| {name} | {metric} | {row['delta']:+.4f} | {relative} | [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}] |")
    lines += ["", "正差值表示提升，负差值表示下降。置信区间跨 0 时不宣称稳定优于；区间不包含规则标注误差。",
              "## 短候选池与串联链路", "", f"切块数分布：`{report['pool_distribution']}`。",
              "候选池不超过 5 时，Recall@5 容易饱和；完整报告同时保留 pool_gt_5 子集，不能把饱和分数当模型改进。",
              "串联审计使用公司+岗位标题先搜索，目标岗位进入前十后再取证据；未进入时串联 Recall 记 0。",
              "它检验精确目标岗位发现到证据检索的接通情况，不代表开放式个性化推荐或生成质量。", ""]
    for result in report["models"]:
        rows = [r for r in result["details"] if r["stratum"] == "individual_job" and r["split"] == "test" and r["relevant_count"]]
        mean = sum(r["e2e_recall_at_5"] for r in rows) / len(rows)
        found = sum(r["search_rank"] is not None for r in rows) / len(rows)
        lines.append(f"- {result['spec']['name']}：目标岗位进入前十比例（按问题计）{found:.4f}，串联 Recall@5 {mean:.4f}。")
    lines += ["", "## 尚未由本轮证明", "",
              "- 自动标签不是岗位语义相关性的人工判定；完整同岗位片段标注任务已导出，可双人复核与仲裁。",
              "- 无答案问题未实现自动拒答阈值；该能力不能用本轮 MRR/Recall 证明。",
              "- 不评测候选人简历证据检索、招聘匹配准确率、投递成功率或生成内容质量。",
              "- 搜索规则不使用 embedding，因此不能把搜索指标变化归因于更换 BGE。", ""]
    lines += ["## Recall@5 非饱和子集（具体岗位 Test，候选数 > 5）", "",
              "| 配置 | 有答案查询 | MRR@3 | Recall@5 |", "|---|---:|---:|---:|"]
    for result in report["models"]:
        row = result["groups"]["individual_job/test/pool_gt_5"]
        if row["answerable"]:
            lines.append(f"| {result['spec']['name']} | {row['answerable']} | {row['mrr_at_3']:.4f} | {row['recall_at_5']:.4f} |")
    lines += ["", "该子集规模较小，不应替代全部测试集；它用来揭示短候选池对 Recall 的影响。", ""]
    lines += ["## 排除派生关键词正例的敏感性评测", "",
              "保留相同检索排名，仅将派生的‘技术栈关键词’从相关集合中排除；这些词表不能独立回答岗位要求。仍为主题规则代理，不是人工答案判定。", "",
              "| 配置 | 有答案查询 | MRR@3 | Recall@5 |", "|---|---:|---:|---:|"]
    for result in report["models"]:
        row = result["source_only_groups"]["individual_job/test/all"]
        lines.append(f"| {result['spec']['name']} | {row['answerable']} | {row['mrr_at_3']:.4f} | {row['recall_at_5']:.4f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/eval_dataset/domestic_job_search_v1/jobs.jsonl"))
    parser.add_argument("--dataset", type=Path, help="Replay an already frozen dataset, including reviewed qrels, without rebuilding labels")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--small-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--m3-model", default=str(settings.bge_m3_model_path or "BAAI/bge-m3"))
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source_files = tuple(Path(p) for p in (
        "app/evaluation/domestic.py", "tools/evaluate_domestic_retrieval.py", "app/domestic/service.py",
        "app/knowledge/catalog.py", "app/chunking.py", "app/knowledge/ingestion.py", "app/embeddings.py"))
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    dataset = args.dataset.resolve() if args.dataset else root / "dataset"
    manifest = verify_dataset(dataset) if args.dataset else build_dataset(dataset, snapshot_records(args.source), args.source)
    save_json(root / "protocol.json", {
        "dataset_version": manifest["version"],
        "purpose": "Corrected regression evaluation on previously observed public snapshot; not blind generalization",
        "primary": "individual_job/test MRR@3 topic-silver; paired JD-group bootstrap",
        "guardrails": "Report Recall@5, pool>5 and source-only positives; never claim universal improvement",
        "selection": "Quality-first provisional M3 only if development MRR improves and absolute Recall loss <=0.005; test is a regression audit, human gold remains required",
        "latency": "Warm sequential local measurement only; no production concurrency SLA",
    })
    records = [JobRecord(**row) for row in read_jsonl(dataset / "jobs.jsonl")]
    print(f"Frozen: {manifest['jobs']} records, {manifest['evidence']} chunks, {manifest['queries']} queries", flush=True)
    with patch.object(settings, "source_corpus_dir", root / "empty_candidate_profiles"):
        search = evaluate_search(root, records, read_jsonl(dataset / "search_queries.jsonl"))
        save_json(root / "search_results.json", search)
        specs = [dict(name="bge_small_zh_512", model=args.small_model, max_length=512),
                 dict(name="bge_m3_512", model=args.m3_model, max_length=512),
                 dict(name="bge_m3_1024", model=args.m3_model, max_length=1024)]
        results = []
        for spec in specs:
            print(f"Starting {spec['name']}", flush=True)
            results.append(evaluate_embedding(root, dataset, records, spec))
    verify_dataset(dataset)
    pool_sizes = {q["job_id"]: q["pool_size"] for q in read_jsonl(dataset / "queries.jsonl")}
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != source_hashes[str(p)] for p in source_files):
        raise RuntimeError("Evaluation code changed during the run; rerun after edits are complete")
    import torch
    provenance = {"python": platform.python_version(), "packages": {p: version(p) for p in ("torch", "sentence-transformers", "chromadb", "transformers")},
                  "gpu": torch.cuda.get_device_name(0), "source_hashes": source_hashes}
    report = {"dataset": manifest, "search": search, "models": results, "comparisons": compare(results), "provenance": provenance,
              "source_only_comparisons": compare([{**r, "details": r["source_only_details"]} for r in results]),
              "pool_distribution": dict(Counter(pool_sizes.values()))}
    save_json(root / "report.json", report)
    (root / "report.md").write_text(markdown(report), encoding="utf-8")
    print(f"Completed: {root / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
