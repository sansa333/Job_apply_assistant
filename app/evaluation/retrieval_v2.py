from __future__ import annotations

import json
import hashlib
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.embeddings import get_embeddings


@dataclass(frozen=True)
class RetrievalV2ModelSpec:
    name: str
    backend: str
    model_name: str | None
    query_prefix: str = ""
    passage_prefix: str = ""


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_retrieval_v2_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    mismatches = []
    for filename, expected in manifest.get("files", {}).items():
        path = directory / filename
        if not path.exists():
            mismatches.append({"file": filename, "reason": "missing"})
            continue
        digest = hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        if digest != expected.get("sha256"):
            mismatches.append({"file": filename, "reason": "sha256_mismatch"})
        if path.stat().st_size != expected.get("bytes"):
            mismatches.append({"file": filename, "reason": "size_mismatch"})
    return {"valid": not mismatches, "mismatches": mismatches}


def load_retrieval_v2(directory: Path) -> dict[str, list[dict] | dict]:
    return {
        "manifest": json.loads((directory / "manifest.json").read_text(encoding="utf-8")),
        "jobs": _read_jsonl(directory / "job_snapshots.jsonl"),
        "evidence": _read_jsonl(directory / "evidence_units.jsonl"),
        "queries": _read_jsonl(directory / "queries.jsonl"),
        "qrels": _read_jsonl(directory / "qrels.jsonl"),
        "candidate_pools": _read_jsonl(directory / "candidate_pools.jsonl"),
    }


def validate_retrieval_v2(dataset: dict) -> dict:
    jobs = dataset["jobs"]
    evidence = dataset["evidence"]
    queries = dataset["queries"]
    qrels = dataset["qrels"]
    pools = dataset["candidate_pools"]
    errors: list[str] = []

    if len(jobs) != 120:
        errors.append(f"expected_120_jobs_got_{len(jobs)}")
    if len(queries) != 480:
        errors.append(f"expected_480_queries_got_{len(queries)}")
    if len({query["query"] for query in queries}) != len(queries):
        errors.append("queries_not_globally_unique")
    if len({job["occupation_family"] for job in jobs}) < 8:
        errors.append("fewer_than_8_occupation_families")

    evidence_ids = {item["evidence_id"] for item in evidence}
    evidence_counts = Counter(item["job_id"] for item in evidence)
    invalid_unit_jobs = [job_id for job_id, count in evidence_counts.items() if not 8 <= count <= 20]
    if invalid_unit_jobs:
        errors.append(f"evidence_unit_count_out_of_range:{len(invalid_unit_jobs)}")

    queries_by_job: dict[str, list[dict]] = defaultdict(list)
    for query in queries:
        queries_by_job[query["job_id"]].append(query)
    invalid_query_jobs = [
        job_id
        for job_id, items in queries_by_job.items()
        if len(items) != 4 or {item["query_type"] for item in items} != {
            "responsibilities", "technical_skills", "qualifications", "work_context"
        }
    ]
    if invalid_query_jobs:
        errors.append(f"invalid_queries_per_job:{len(invalid_query_jobs)}")

    qrels_by_query: dict[str, list[dict]] = defaultdict(list)
    for qrel in qrels:
        qrels_by_query[qrel["query_id"]].append(qrel)
        if qrel["evidence_id"] not in evidence_ids:
            errors.append(f"unknown_qrel_evidence:{qrel['evidence_id']}")
    multi_ratio = sum(len(qrels_by_query[query["query_id"]]) > 1 for query in queries) / len(queries)
    if multi_ratio < 0.40:
        errors.append(f"multi_relevant_ratio_below_0.40:{multi_ratio:.4f}")

    pools_by_query = {pool["query_id"]: pool for pool in pools}
    candidate_type_counts: Counter = Counter()
    for query in queries:
        pool = pools_by_query.get(query["query_id"], {}).get("candidates", [])
        ids = [item["evidence_id"] for item in pool]
        candidate_type_counts.update(item["candidate_type"] for item in pool)
        if len(ids) != 50 or len(set(ids)) != 50:
            errors.append(f"invalid_candidate_pool:{query['query_id']}")
        relevant = {item["evidence_id"] for item in qrels_by_query[query["query_id"]] if item["relevance_grade"] >= 2}
        if not relevant.issubset(ids):
            errors.append(f"candidate_pool_missing_relevant:{query['query_id']}")

    split_jobs: dict[str, set[str]] = defaultdict(set)
    for job in jobs:
        split_jobs[job["split"]].add(job["job_id"])
    if split_jobs["development"] & split_jobs["test"]:
        errors.append("job_leakage_between_splits")
    test_job_count = len(split_jobs["test"])
    if test_job_count != 36:
        errors.append(f"expected_36_test_jobs_got_{test_job_count}")

    return {
        "valid": not errors,
        "errors": errors,
        "statistics": {
            "jobs": len(jobs),
            "evidence_units": len(evidence),
            "queries": len(queries),
            "unique_queries": len({query["query"] for query in queries}),
            "qrels": len(qrels),
            "multi_relevant_query_ratio": multi_ratio,
            "mean_relevant_per_query": len(qrels) / len(queries),
            "candidate_pool_size": 50,
            "candidate_type_distribution": dict(sorted(candidate_type_counts.items())),
            "occupation_family_distribution": dict(sorted(Counter(job["occupation_family"] for job in jobs).items())),
            "split_job_distribution": {key: len(value) for key, value in sorted(split_jobs.items())},
        },
    }


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def _dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _aggregate(details: list[dict], ranks: tuple[int, ...]) -> dict:
    average = lambda values: sum(values) / len(values) if values else 0.0
    metrics: dict[str, float] = {}
    for rank in ranks:
        metrics[f"hit_rate_at_{rank}"] = average([float(item[f"hit_at_{rank}"]) for item in details])
        metrics[f"recall_at_{rank}"] = average([item[f"recall_at_{rank}"] for item in details])
        metrics[f"mrr_at_{rank}"] = average([item[f"reciprocal_rank_at_{rank}"] for item in details])
        metrics[f"ndcg_at_{rank}"] = average([item[f"ndcg_at_{rank}"] for item in details])
    return metrics


def evaluate_model(
    spec: RetrievalV2ModelSpec,
    dataset: dict,
    *,
    split: str,
    evaluation_scope: str = "hard_pool",
    ranks: tuple[int, ...] = (1, 3, 5, 10),
    local_files_only: bool = True,
) -> dict:
    if evaluation_scope not in {"job_scoped", "hard_pool"}:
        raise ValueError("evaluation_scope must be job_scoped or hard_pool")
    validation = validate_retrieval_v2(dataset)
    if not validation["valid"]:
        raise ValueError(f"invalid retrieval V2 dataset: {validation['errors']}")
    model_started = time.perf_counter()
    embeddings = get_embeddings(backend=spec.backend, model_name=spec.model_name, local_files_only=local_files_only)
    model_load_ms = (time.perf_counter() - model_started) * 1000

    evidence_by_id = {item["evidence_id"]: item for item in dataset["evidence"]}
    queries = [item for item in dataset["queries"] if item["split"] == split]
    stored_pools = {item["query_id"]: item["candidates"] for item in dataset["candidate_pools"]}
    pools = {}
    for query in queries:
        candidates = stored_pools[query["query_id"]]
        if evaluation_scope == "job_scoped":
            candidates = [
                candidate
                for candidate in candidates
                if evidence_by_id[candidate["evidence_id"]]["job_id"] == query["job_id"]
            ]
        pools[query["query_id"]] = candidates
    qrels_by_query: dict[str, dict[str, int]] = defaultdict(dict)
    for item in dataset["qrels"]:
        qrels_by_query[item["query_id"]][item["evidence_id"]] = int(item["relevance_grade"])
    needed_ids = sorted({candidate["evidence_id"] for query in queries for candidate in pools[query["query_id"]]})

    corpus_started = time.perf_counter()
    corpus_matrix = np.asarray(
        embeddings.embed_documents([spec.passage_prefix + evidence_by_id[item]["text"] for item in needed_ids]),
        dtype=np.float32,
    )
    corpus_encode_ms = (time.perf_counter() - corpus_started) * 1000
    corpus_matrix = _normalize(corpus_matrix)
    row_by_id = {evidence_id: index for index, evidence_id in enumerate(needed_ids)}

    query_vectors: list[list[float]] = []
    query_encode_latencies: list[float] = []
    for item in queries:
        query_started = time.perf_counter()
        query_vectors.append(embeddings.embed_query(spec.query_prefix + item["query"]))
        query_encode_latencies.append((time.perf_counter() - query_started) * 1000)
    query_matrix = np.asarray(query_vectors, dtype=np.float32)
    query_encode_ms = sum(query_encode_latencies)
    query_matrix = _normalize(query_matrix)

    ranking_started = time.perf_counter()
    details: list[dict] = []
    ranking_latencies: list[float] = []
    max_k = max(ranks)
    for query_index, query in enumerate(queries):
        item_started = time.perf_counter()
        candidate_ids = [item["evidence_id"] for item in pools[query["query_id"]]]
        candidate_rows = [row_by_id[item] for item in candidate_ids]
        scores = corpus_matrix[candidate_rows] @ query_matrix[query_index]
        ranked = [
            candidate_ids[index]
            for index in sorted(range(len(candidate_ids)), key=lambda index: (-float(scores[index]), candidate_ids[index]))
        ]
        grades = qrels_by_query[query["query_id"]]
        relevant = {evidence_id for evidence_id, grade in grades.items() if grade >= 2}
        first_rank = next((index for index, evidence_id in enumerate(ranked, start=1) if evidence_id in relevant), None)
        detail = {
            "query_id": query["query_id"],
            "job_id": query["job_id"],
            "occupation_family": query["occupation_family"],
            "query_type": query["query_type"],
            "query": query["query"],
            "relevant_count": len(relevant),
            "candidate_count": len(candidate_ids),
            "first_relevant_rank": first_rank,
            "top_evidence_ids": ranked[:max_k],
        }
        ideal_grades = sorted(grades.values(), reverse=True)
        for rank in ranks:
            top = ranked[:rank]
            matched = relevant & set(top)
            detail[f"hit_at_{rank}"] = bool(matched)
            detail[f"recall_at_{rank}"] = len(matched) / len(relevant)
            detail[f"reciprocal_rank_at_{rank}"] = 1.0 / first_rank if first_rank and first_rank <= rank else 0.0
            actual_dcg = _dcg([grades.get(evidence_id, 0) for evidence_id in top])
            ideal_dcg = _dcg(ideal_grades[:rank])
            detail[f"ndcg_at_{rank}"] = actual_dcg / ideal_dcg if ideal_dcg else 0.0
        details.append(detail)
        ranking_latencies.append((time.perf_counter() - item_started) * 1000)
    ranking_ms = (time.perf_counter() - ranking_started) * 1000

    by_family: dict[str, list[dict]] = defaultdict(list)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for detail in details:
        by_family[detail["occupation_family"]].append(detail)
        by_type[detail["query_type"]].append(detail)
    dimension = int(corpus_matrix.shape[1]) if len(corpus_matrix) else 0
    return {
        "status": "ok",
        "name": spec.name,
        "backend": spec.backend,
        "model_name": spec.model_name or "hash",
        "split": split,
        "evaluation_scope": evaluation_scope,
        "sample_count": len(queries),
        "candidate_pool": {
            "mean": sum(len(pools[item["query_id"]]) for item in queries) / len(queries) if queries else 0.0,
            "minimum": min((len(pools[item["query_id"]]) for item in queries), default=0),
            "maximum": max((len(pools[item["query_id"]]) for item in queries), default=0),
        },
        "embedding_protocol": {"query_prefix": spec.query_prefix, "passage_prefix": spec.passage_prefix},
        "metrics": _aggregate(details, ranks),
        "metrics_by_family": {
            key: {"sample_count": len(items), **_aggregate(items, ranks)} for key, items in sorted(by_family.items())
        },
        "metrics_by_query_type": {
            key: {"sample_count": len(items), **_aggregate(items, ranks)} for key, items in sorted(by_type.items())
        },
        "resources": {
            "dimension": dimension,
            "model_load_ms": model_load_ms,
            "corpus_encode_ms": corpus_encode_ms,
            "query_encode_ms": query_encode_ms,
            "ranking_ms": ranking_ms,
            "mean_query_total_ms": (query_encode_ms + ranking_ms) / len(queries) if queries else 0.0,
            "p50_query_total_ms": _percentile(
                [left + right for left, right in zip(query_encode_latencies, ranking_latencies)], 0.50
            ),
            "p95_query_total_ms": _percentile(
                [left + right for left, right in zip(query_encode_latencies, ranking_latencies)], 0.95
            ),
            "embedding_matrix_bytes": int(corpus_matrix.nbytes),
        },
        "bad_cases": sorted(details, key=lambda item: (item["first_relevant_rank"] or 10**9), reverse=True)[:30],
        "details": details,
    }


def select_on_development(
    results: list[dict], *, evaluation_scope: str = "job_scoped"
) -> tuple[dict | None, dict | None]:
    successful = [
        result
        for result in results
        if result.get("status") == "ok"
        and result.get("split") == "development"
        and result.get("evaluation_scope") == evaluation_scope
    ]
    if not successful:
        return None, None
    ranked = sorted(
        successful,
        key=lambda item: (
            -item["metrics"]["mrr_at_3"],
            -item["metrics"]["recall_at_5"],
            -item["metrics"]["hit_rate_at_1"],
            item["resources"]["p95_query_total_ms"],
            item["name"],
        ),
    )
    return ranked[0], ranked[1] if len(ranked) > 1 else None


def paired_bootstrap(winner: dict, runner_up: dict, *, iterations: int = 5000) -> dict:
    runner = {item["query_id"]: item for item in runner_up["details"]}
    deltas = [
        item["reciprocal_rank_at_3"] - runner[item["query_id"]]["reciprocal_rank_at_3"]
        for item in winner["details"]
        if item["query_id"] in runner
    ]
    rng = random.Random(20260827)
    samples = []
    for _ in range(iterations):
        draw = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    lower = samples[int(0.025 * (len(samples) - 1))]
    upper = samples[int(0.975 * (len(samples) - 1))]
    return {
        "winner": winner["name"],
        "runner_up": runner_up["name"],
        "metric": "mrr_at_3",
        "paired_query_count": len(deltas),
        "delta": sum(deltas) / len(deltas),
        "bootstrap_iterations": iterations,
        "bootstrap_95_ci": [lower, upper],
        "wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "replacement_gate_passed": lower > 1e-6,
    }
