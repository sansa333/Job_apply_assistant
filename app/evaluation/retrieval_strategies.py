from __future__ import annotations

import math
import random
import time
from collections import defaultdict

import numpy as np
from langchain_core.documents import Document

from app.embeddings import get_embeddings
from app.evaluation.retrieval_v2 import validate_retrieval_v2
from app.knowledge.hybrid import bm25_rank, reciprocal_rank_fusion
from app.multimodal.reranker import CrossEncoderReranker, RerankResult


STRATEGIES = (
    "bm25",
    "dense",
    "dense_bm25_rrf",
    "dense_rerank_blend",
    "hybrid_rerank_blend",
    "dense_rerank_full",
    "hybrid_rerank_full",
)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))


def _blend_reranker(
    candidates: list[Document], result: RerankResult, *, reranker_weight: float
) -> list[Document]:
    if not result.applied or not result.scores_by_chunk:
        return candidates
    if reranker_weight >= 1.0:
        return result.docs
    scores = result.scores_by_chunk
    values = [scores.get(str(document.metadata["chunk_id"]), 0.0) for document in candidates]
    minimum, maximum = min(values, default=0.0), max(values, default=0.0)
    span = maximum - minimum
    blended: list[tuple[float, int, Document]] = []
    for rank, document in enumerate(candidates, start=1):
        cross_score = scores.get(str(document.metadata["chunk_id"]), minimum)
        normalized_cross = (cross_score - minimum) / span if span else 0.5
        rank_score = 1.0 / rank
        score = (1.0 - reranker_weight) * rank_score + reranker_weight * normalized_cross
        blended.append((score, rank, document))
    return [document for _, _, document in sorted(blended, key=lambda item: (-item[0], item[1]))]


def _query_detail(
    query: dict,
    ranked_ids: list[str],
    grades: dict[str, int],
    *,
    candidate_count: int,
    first_stage_ids: list[str],
    ranks: tuple[int, ...],
) -> dict:
    relevant = {evidence_id for evidence_id, grade in grades.items() if grade >= 2}
    first_rank = next(
        (index for index, evidence_id in enumerate(ranked_ids, start=1) if evidence_id in relevant),
        None,
    )
    detail = {
        "query_id": query["query_id"],
        "job_id": query["job_id"],
        "occupation_family": query["occupation_family"],
        "query_type": query["query_type"],
        "query": query["query"],
        "relevant_count": len(relevant),
        "candidate_count": candidate_count,
        "first_stage_recall_at_20": len(relevant & set(first_stage_ids[:20])) / len(relevant),
        "first_relevant_rank": first_rank,
        "top_evidence_ids": ranked_ids[: max(ranks)],
    }
    ideal_grades = sorted(grades.values(), reverse=True)
    for rank in ranks:
        top = ranked_ids[:rank]
        matched = relevant & set(top)
        detail[f"hit_at_{rank}"] = bool(matched)
        detail[f"recall_at_{rank}"] = len(matched) / len(relevant)
        detail[f"reciprocal_rank_at_{rank}"] = (
            1.0 / first_rank if first_rank and first_rank <= rank else 0.0
        )
        actual_dcg = _dcg([grades.get(evidence_id, 0) for evidence_id in top])
        ideal_dcg = _dcg(ideal_grades[:rank])
        detail[f"ndcg_at_{rank}"] = actual_dcg / ideal_dcg if ideal_dcg else 0.0
    return detail


def _aggregate(details: list[dict], ranks: tuple[int, ...]) -> dict[str, float]:
    average = lambda values: sum(values) / len(values) if values else 0.0
    metrics = {
        "first_stage_recall_at_20": average(
            [item["first_stage_recall_at_20"] for item in details]
        )
    }
    for rank in ranks:
        metrics[f"hit_rate_at_{rank}"] = average(
            [float(item[f"hit_at_{rank}"]) for item in details]
        )
        metrics[f"recall_at_{rank}"] = average(
            [item[f"recall_at_{rank}"] for item in details]
        )
        metrics[f"mrr_at_{rank}"] = average(
            [item[f"reciprocal_rank_at_{rank}"] for item in details]
        )
        metrics[f"ndcg_at_{rank}"] = average(
            [item[f"ndcg_at_{rank}"] for item in details]
        )
    return metrics


def paired_strategy_delta(candidate: dict, baseline: dict, *, iterations: int = 5000) -> dict:
    baseline_by_query = {item["query_id"]: item for item in baseline["details"]}
    deltas = [
        item["reciprocal_rank_at_3"]
        - baseline_by_query[item["query_id"]]["reciprocal_rank_at_3"]
        for item in candidate["details"]
    ]
    rng = random.Random(20260828)
    samples = []
    for _ in range(iterations):
        draw = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    lower = samples[int(0.025 * (len(samples) - 1))]
    upper = samples[int(0.975 * (len(samples) - 1))]
    return {
        "candidate": candidate["strategy"],
        "baseline": baseline["strategy"],
        "metric": "mrr_at_3",
        "paired_query_count": len(deltas),
        "delta": sum(deltas) / len(deltas),
        "bootstrap_iterations": iterations,
        "bootstrap_95_ci": [lower, upper],
        "wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "quality_gate_passed": lower > 1e-6,
        "p95_latency_delta_ms": (
            candidate["resources"]["p95_query_total_ms"]
            - baseline["resources"]["p95_query_total_ms"]
        ),
    }


def evaluate_bge_m3_strategies(
    dataset: dict,
    *,
    model_name: str,
    reranker_model: str,
    split: str = "development",
    evaluation_scope: str = "job_scoped",
    candidate_k: int = 20,
    rrf_k: int = 60,
    reranker_weight: float = 0.2,
    ranks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    if evaluation_scope not in {"job_scoped", "hard_pool"}:
        raise ValueError("evaluation_scope must be job_scoped or hard_pool")
    if not 0.0 <= reranker_weight <= 1.0:
        raise ValueError("reranker_weight must be between 0 and 1")
    validation = validate_retrieval_v2(dataset)
    if not validation["valid"]:
        raise ValueError(f"invalid Retrieval V2 dataset: {validation['errors']}")

    evidence_by_id = {item["evidence_id"]: item for item in dataset["evidence"]}
    query_items = [item for item in dataset["queries"] if item["split"] == split]
    stored_pools = {item["query_id"]: item["candidates"] for item in dataset["candidate_pools"]}
    pools: dict[str, list[str]] = {}
    for query in query_items:
        candidate_ids = [item["evidence_id"] for item in stored_pools[query["query_id"]]]
        if evaluation_scope == "job_scoped":
            candidate_ids = [
                evidence_id
                for evidence_id in candidate_ids
                if evidence_by_id[evidence_id]["job_id"] == query["job_id"]
            ]
        pools[query["query_id"]] = sorted(candidate_ids)

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for item in dataset["qrels"]:
        qrels[item["query_id"]][item["evidence_id"]] = int(item["relevance_grade"])

    needed_ids = sorted({evidence_id for ids in pools.values() for evidence_id in ids})
    documents = {
        evidence_id: Document(
            page_content=evidence_by_id[evidence_id]["text"],
            metadata={"chunk_id": evidence_id, "job_id": evidence_by_id[evidence_id]["job_id"]},
        )
        for evidence_id in needed_ids
    }

    model_started = time.perf_counter()
    embeddings = get_embeddings(
        backend="huggingface",
        model_name=model_name,
        local_files_only=True,
    )
    model_load_ms = (time.perf_counter() - model_started) * 1000
    embeddings.embed_query("BGE-M3 retrieval warmup")

    corpus_started = time.perf_counter()
    corpus_matrix = _normalize(
        np.asarray(
            embeddings.embed_documents([documents[evidence_id].page_content for evidence_id in needed_ids]),
            dtype=np.float32,
        )
    )
    corpus_encode_ms = (time.perf_counter() - corpus_started) * 1000
    row_by_id = {evidence_id: index for index, evidence_id in enumerate(needed_ids)}

    query_vectors = []
    query_encode_latencies = []
    for query in query_items:
        started = time.perf_counter()
        query_vectors.append(embeddings.embed_query(query["query"]))
        query_encode_latencies.append((time.perf_counter() - started) * 1000)
    query_matrix = _normalize(np.asarray(query_vectors, dtype=np.float32))

    rankings: dict[str, dict[str, list[Document]]] = defaultdict(dict)
    latency: dict[str, dict[str, float]] = defaultdict(dict)
    for query_index, query in enumerate(query_items):
        query_id = query["query_id"]
        candidate_ids = pools[query_id]
        candidate_documents = [documents[evidence_id] for evidence_id in candidate_ids]

        dense_started = time.perf_counter()
        scores = corpus_matrix[[row_by_id[evidence_id] for evidence_id in candidate_ids]] @ query_matrix[
            query_index
        ]
        dense_order = [
            candidate_ids[index]
            for index in sorted(
                range(len(candidate_ids)),
                key=lambda index: (-float(scores[index]), candidate_ids[index]),
            )
        ]
        dense_ranking_ms = (time.perf_counter() - dense_started) * 1000
        dense_docs = [documents[evidence_id] for evidence_id in dense_order[:candidate_k]]

        bm25_started = time.perf_counter()
        bm25_docs = bm25_rank(query["query"], candidate_documents, k=min(candidate_k, len(candidate_documents)))
        bm25_ms = (time.perf_counter() - bm25_started) * 1000

        fusion_started = time.perf_counter()
        hybrid_docs = reciprocal_rank_fusion(dense_docs, bm25_docs, rrf_k=rrf_k)[:candidate_k]
        fusion_ms = (time.perf_counter() - fusion_started) * 1000

        rankings[query_id]["bm25"] = bm25_docs
        rankings[query_id]["dense"] = dense_docs
        rankings[query_id]["dense_bm25_rrf"] = hybrid_docs
        encode_ms = query_encode_latencies[query_index]
        latency[query_id]["bm25"] = bm25_ms
        latency[query_id]["dense"] = encode_ms + dense_ranking_ms
        latency[query_id]["dense_bm25_rrf"] = encode_ms + dense_ranking_ms + bm25_ms + fusion_ms

    reranker = CrossEncoderReranker(
        enabled=True,
        model_name=reranker_model,
        local_files_only=True,
    )
    reranker_started = time.perf_counter()
    first_query = query_items[0]
    reranker.rerank(first_query["query"], rankings[first_query["query_id"]]["dense"][:1], top_n=1)
    reranker_load_warmup_ms = (time.perf_counter() - reranker_started) * 1000

    reranker_applied = {
        "dense_rerank_blend": 0,
        "hybrid_rerank_blend": 0,
        "dense_rerank_full": 0,
        "hybrid_rerank_full": 0,
    }
    for query in query_items:
        query_id = query["query_id"]
        for prefix, source_strategy in (
            ("dense", "dense"),
            ("hybrid", "dense_bm25_rrf"),
        ):
            candidates = rankings[query_id][source_strategy]
            started = time.perf_counter()
            rerank_result = reranker.rerank(query["query"], candidates, top_n=len(candidates))
            rerank_ms = (time.perf_counter() - started) * 1000
            blend_strategy = f"{prefix}_rerank_blend"
            full_strategy = f"{prefix}_rerank_full"
            rankings[query_id][blend_strategy] = _blend_reranker(
                candidates,
                rerank_result,
                reranker_weight=reranker_weight,
            )
            rankings[query_id][full_strategy] = (
                rerank_result.docs if rerank_result.applied else candidates
            )
            total_ms = latency[query_id][source_strategy] + rerank_ms
            latency[query_id][blend_strategy] = total_ms
            latency[query_id][full_strategy] = total_ms
            reranker_applied[blend_strategy] += int(rerank_result.applied)
            reranker_applied[full_strategy] += int(rerank_result.applied)

    results = []
    for strategy in STRATEGIES:
        details = []
        latencies = []
        for query in query_items:
            query_id = query["query_id"]
            ranked_ids = [str(document.metadata["chunk_id"]) for document in rankings[query_id][strategy]]
            first_stage_strategy = {
                "dense_rerank_blend": "dense",
                "hybrid_rerank_blend": "dense_bm25_rrf",
                "dense_rerank_full": "dense",
                "hybrid_rerank_full": "dense_bm25_rrf",
            }.get(strategy, strategy)
            first_stage_ids = [
                str(document.metadata["chunk_id"])
                for document in rankings[query_id][first_stage_strategy]
            ]
            details.append(
                _query_detail(
                    query,
                    ranked_ids,
                    qrels[query_id],
                    candidate_count=len(pools[query_id]),
                    first_stage_ids=first_stage_ids,
                    ranks=ranks,
                )
            )
            latencies.append(latency[query_id][strategy])
        result = {
            "status": "ok",
            "model_name": model_name,
            "strategy": strategy,
            "split": split,
            "evaluation_scope": evaluation_scope,
            "sample_count": len(query_items),
            "candidate_k": candidate_k,
            "metrics": _aggregate(details, ranks),
            "resources": {
                "mean_query_total_ms": sum(latencies) / len(latencies),
                "p50_query_total_ms": _percentile(latencies, 0.50),
                "p95_query_total_ms": _percentile(latencies, 0.95),
                "reranker_applied_count": reranker_applied.get(strategy, 0),
            },
            "bad_cases": sorted(
                details,
                key=lambda item: item["first_relevant_rank"] or 10**9,
                reverse=True,
            )[:30],
            "details": details,
        }
        results.append(result)

    dense = next(result for result in results if result["strategy"] == "dense")
    comparisons = [
        paired_strategy_delta(result, dense)
        for result in results
        if result["strategy"] != "dense"
    ]
    return {
        "status": "ok",
        "model_name": model_name,
        "reranker_model": reranker_model,
        "split": split,
        "evaluation_scope": evaluation_scope,
        "sample_count": len(query_items),
        "parameters": {
            "candidate_k": candidate_k,
            "rrf_k": rrf_k,
            "reranker_weight": reranker_weight,
            "ranks": list(ranks),
        },
        "shared_resources": {
            "embedding_model_load_ms": model_load_ms,
            "corpus_encode_ms": corpus_encode_ms,
            "embedding_dimension": int(corpus_matrix.shape[1]),
            "reranker_load_and_warmup_ms": reranker_load_warmup_ms,
        },
        "strategy_results": results,
        "paired_vs_dense": comparisons,
    }
