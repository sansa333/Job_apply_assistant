"""Frozen business-task evaluation. No LLM calls and no production index writes."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np
from langchain_core.documents import Document

from app.config import settings
from app.domestic.service import DomesticJobService
from app.embeddings import get_embeddings
from app.evaluation.business import (calibrate_threshold, calibration_metrics, fits_filters,
    graded_metrics, random_expectation, summarize)
from app.evaluation.domestic import frozen_catalog, paired_comparison, read_jsonl, verify_dataset
from app.knowledge.hybrid import bm25_rank, reciprocal_rank_fusion
from app.knowledge.ingestion import JobKnowledgeIngestion
from app.knowledge.models import JobRecord
from tools.evaluate_domestic_retrieval import BorrowedIngestion


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def groups(rows):
    return {split: summarize([r for r in rows if r["split"] == split]) for split in ("development", "test")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    dataset = args.dataset.resolve()
    manifest = verify_dataset(dataset)
    source_paths = [Path(p) for p in ("app/evaluation/business.py", "app/evaluation/business_cases.py",
        "app/domestic/service.py", "app/domestic/discovery.py", "app/knowledge/ingestion.py", "app/chunking.py",
        "app/knowledge/hybrid.py", "app/domestic/evidence_check.py", "tools/evaluate_domestic_business.py",
        "app/evaluation/domestic.py", "app/embeddings.py", "app/knowledge/catalog.py")]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    save(root / "protocol.json", {"version": "business-v4-same-job-top5", "dataset_files": manifest["files"],
        "primary_discovery": "full specific-job corpus: nDCG@10, Recall@20 (fully satisfying grade 2), MRR@10",
        "primary_evidence": "within-job MRR@3 and Recall@5; actual Top5 retrieval, identical frozen jobs/chunks/questions/qrels",
        "evidence_diagnostics": "Recall@1/2, nDCG@3, complete quoted-support coverage@1/2/3/5, candidate-pool sizes",
        "evidence_retrieval_k": 5,
        "unanswerable": "development-only max false acceptance .10; report test false acceptance/rejection and coverage",
        "model_selection": "do not auto-deploy a winner; report task tradeoffs and label uncertainty",
        "labels": manifest["annotation_status"], "no_blind_claim": "New query suite, but source documents previously observed",
        "baselines": ["analytic random", "source order", "BM25", "existing discovery rules"], "source_hashes": hashes})
    records = [JobRecord(**r) for r in read_jsonl(dataset / "jobs.jsonl")]
    search_queries = read_jsonl(dataset / "search_queries.jsonl")
    evidence_queries = read_jsonl(dataset / "queries.jsonl")
    search_labels, evidence_labels, support = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for row in read_jsonl(dataset / "search_qrels.jsonl"):
        search_labels[row["query_id"]][row["job_id"]] = row["grade"]
    for row in read_jsonl(dataset / "qrels.jsonl"):
        evidence_labels[row["query_id"]][row["evidence_id"]] = row["grade"]
        support[row["query_id"]][row["evidence_id"]] = row["support_indices"]
    by_job, evidence_by_id = defaultdict(list), {}
    for row in read_jsonl(dataset / "evidence.jsonl"):
        doc = Document(page_content=row["text"], metadata=row["metadata"])
        by_job[row["job_id"]].append(doc)
        evidence_by_id[row["evidence_id"]] = doc

    def search_rows(service, strategy):
        rows = []
        for q in search_queries:
            eligible = [r.job_id for r in records if fits_filters(r, q["filters"])]
            started = time.perf_counter()
            if strategy == "random":
                ranking = []
                metrics = random_expectation(eligible, search_labels[q["query_id"]], relevant_grade=2)
            else:
                result = service.search(keyword=q["query"], **q["filters"], candidate_id="eval_no_profile", limit=50,
                    strategy=strategy, individual_only=True, strict_graduation=True)
                ranking = [r["job_id"] for r in result["matches"]]
                metrics = graded_metrics(ranking, search_labels[q["query_id"]], relevant_grade=2)
            rows.append({**q, **metrics, "candidate_pool": len(eligible), "ranking": ranking,
                         "hard_filter_violations_at_20": len(set(ranking[:20]) - set(eligible)),
                         "latency_ms": (time.perf_counter() - started) * 1000})
        return rows

    def evidence_row(q, ranking, metrics=None):
        metrics = metrics or graded_metrics(ranking, evidence_labels[q["query_id"]], ks=(1, 2, 3, 5))
        result = {**q, **metrics, "ranking": ranking}
        for k in (1, 2, 3, 5):
            seen = {i for eid in ranking[:k] for i in support[q["query_id"]].get(eid, [])}
            result[f"complete_support_at_{k}"] = float(seen == set(range(len(q["support_quotes"])))) if q["support_quotes"] else None
        return result

    base_catalog = frozen_catalog(root / "baseline.sqlite3", records)
    baseline_service = DomesticJobService(catalog=base_catalog, sources={})
    baseline_search, baseline_evidence = {}, {}
    with patch.object(settings, "source_corpus_dir", root / "empty_profiles"):
        for method in ("random", "rules", "bm25"):
            rows = search_rows(baseline_service, method)
            baseline_search[method] = {"groups": groups(rows), "details": rows}
        for method in ("random", "source_order", "bm25"):
            rows = []
            for q in evidence_queries:
                documents = by_job[q["job_id"]]
                ids = [d.metadata["chunk_id"] for d in documents]
                if method == "random":
                    row = evidence_row(q, [], random_expectation(ids, evidence_labels[q["query_id"]], ks=(1, 2, 3, 5)))
                    # No false random support-coverage estimate from an empty list.
                    row = {k: v for k, v in row.items() if not k.startswith("complete_support_at_")}
                else:
                    ordered = documents if method == "source_order" else bm25_rank(q["query"], documents, k=len(documents))
                    row = evidence_row(q, [d.metadata["chunk_id"] for d in ordered])
                rows.append(row)
            baseline_evidence[method] = {"groups": groups(rows), "details": rows}
        save(root / "baselines.json", {"discovery": baseline_search, "evidence": baseline_evidence})
        specs = [("small_512", "BAAI/bge-small-zh-v1.5", 512), ("m3_512", str(settings.bge_m3_model_path), 512),
                 ("m3_1024", str(settings.bge_m3_model_path), 1024)]
        results = []
        for name, model_name, max_length in specs:
            print(f"Starting {name}", flush=True)
            model_root = root / name
            model_root.mkdir()
            embeddings = get_embeddings(backend="huggingface", model_name=model_name, max_length=max_length,
                device="cuda", use_fp16=True, batch_size=4, local_files_only=True)
            index = JobKnowledgeIngestion(catalog=base_catalog, source_corpus_dir=model_root / "sources",
                vector_db_dir=model_root / "index", collection_name="business_eval", embeddings=embeddings,
                embedding_backend="huggingface", embedding_model=model_name)
            borrowed = BorrowedIngestion(index)
            service = DomesticJobService(catalog=base_catalog, sources={}, ingestion_factory=lambda borrowed=borrowed: borrowed)
            try:
                count = index.rebuild_records(records)
                if count != len(evidence_by_id):
                    raise ValueError("Corpus mismatch")
                for job_id in by_job:
                    for doc in index.get_documents_for_job(job_id):
                        if doc.page_content != evidence_by_id[doc.metadata["chunk_id"]].page_content:
                            raise ValueError("Production chunks differ from frozen source")
                discovery = {}
                for strategy in ("dense", "hybrid"):
                    rows = search_rows(service, strategy)
                    discovery[strategy] = {"groups": groups(rows), "details": rows}
                evidence_rows, hybrid_rows, scores = [], [], []
                with patch.object(settings, "job_retrieval_strategy", "vector"), patch.object(settings, "enable_reranker", False):
                    for q in evidence_queries:
                        start = time.perf_counter()
                        response = service.retrieve_job_evidence(job_id=q["job_id"], query=q["query"], k=5)
                        ranking = [d["chunk_id"] for d in response["documents"]]
                        assert all(d["job_id"] == q["job_id"] for d in response["documents"])
                        evidence_rows.append({**evidence_row(q, ranking), "latency_ms": (time.perf_counter() - start) * 1000})
                        dense = index.retrieve_for_job(q["job_id"], q["query"], k=max(12, q["pool_size"]))
                        lexical = bm25_rank(q["query"], by_job[q["job_id"]], k=max(12, q["pool_size"]))
                        fused = reciprocal_rank_fusion(dense, lexical)[:5]
                        hybrid_rows.append(evidence_row(q, [d.metadata["chunk_id"] for d in fused]))
                        scored = index.scored_source_evidence(q["job_id"], q["query"], k=3)
                        covered = {i for d, _ in scored for i in support[q["query_id"]].get(d.metadata["chunk_id"], [])}
                        scores.append({"query_id": q["query_id"], "split": q["split"], "group_id": q["group_id"],
                            "answerable": bool(q["support_quotes"]), "score": scored[0][1] if scored else -1.0,
                            "complete_source_support": bool(q["support_quotes"]) and covered == set(range(len(q["support_quotes"]))),
                            "source_ranking": [d.metadata["chunk_id"] for d, _ in scored]})
                threshold = calibrate_threshold([r for r in scores if r["split"] == "development"])
                policy = {"version": "business-v4-development-calibrated", "fit_split": "development", "threshold": threshold,
                    "embedding_identity": {k: v for k, v in index.expected_manifest.items() if k != "collection"},
                    "dataset_manifest_sha256": hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
                    "annotation_status": manifest["annotation_status"], "production_approved": False}
                save(model_root / "evidence_policy.json", policy)
                gate = {split: {"calibrated": calibration_metrics([r for r in scores if r["split"] == split], threshold),
                               "always_return": calibration_metrics([r for r in scores if r["split"] == split], -2),
                               "always_abstain": calibration_metrics([r for r in scores if r["split"] == split], 2)} for split in ("development", "test")}
                token_lengths = [len(embeddings._model.tokenizer.encode(d.page_content, truncation=False)) for d in evidence_by_id.values()]
                result = {"name": name, "model": model_name, "max_length": max_length, "dimension": index.expected_manifest["dimension"],
                    "chunks_truncated": sum(length > max_length for length in token_lengths), "discovery": discovery,
                    "evidence": {"groups": groups(evidence_rows), "details": evidence_rows},
                    "evidence_hybrid": {"groups": groups(hybrid_rows), "details": hybrid_rows},
                    "gate": gate, "gate_threshold": threshold, "gate_scores": scores,
                    "warm_evidence_p95_ms": float(np.percentile([r["latency_ms"] for r in evidence_rows], 95))}
                save(model_root / "results.json", result)
                results.append(result)
                print(f"Completed {name}", flush=True)
            finally:
                index.close()
                del service, borrowed, index, embeddings
                gc.collect()
                import torch
                torch.cuda.empty_cache()
    comparisons = {}
    for result in results[1:]:
        comparisons[result["name"]] = {}
        for task, left, right, metrics in (
            ("evidence", results[0]["evidence"]["details"], result["evidence"]["details"], ("mrr_at_3", "recall_at_5", "recall_at_2", "ndcg_at_3")),
            ("discovery_dense", results[0]["discovery"]["dense"]["details"], result["discovery"]["dense"]["details"], ("mrr_at_10", "recall_at_20", "ndcg_at_10"))):
            comparisons[result["name"]][task] = {m: paired_comparison([r for r in left if r["split"] == "test"],
                [r for r in right if r["split"] == "test"], m) for m in metrics}
            for comparison in comparisons[result["name"]][task].values():
                comparison["bootstrap_unit"] = "query intent" if task == "discovery_dense" else "duplicate-JD group"
    verify_dataset(dataset)
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)] for p in source_paths):
        raise RuntimeError("Code changed during evaluation; preserve partial artifacts and rerun")
    import torch
    pool_audit = {split: {
        "answerable_pool_sizes": dict(Counter(q["pool_size"] for q in evidence_queries if q["split"] == split and q["support_quotes"])),
        "answerable_pool_gt5": sum(q["pool_size"] > 5 for q in evidence_queries if q["split"] == split and q["support_quotes"]),
        "unanswerable": sum(not q["support_quotes"] for q in evidence_queries if q["split"] == split),
    } for split in ("development", "test")}
    report = {"dataset": manifest, "evidence_retrieval_k": 5, "evidence_pool_audit": pool_audit,
        "baselines": {"discovery": baseline_search, "evidence": baseline_evidence},
        "models": results, "comparisons": comparisons, "provenance": {"gpu": torch.cuda.get_device_name(0),
            "packages": {p: version(p) for p in ("torch", "chromadb", "sentence-transformers", "transformers")}, "source_hashes": hashes}}
    save(root / "report.json", report)
    print(f"Report: {root / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
