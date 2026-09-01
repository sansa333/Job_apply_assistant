from __future__ import annotations

import re
import time
import random
from dataclasses import dataclass
from pathlib import Path

from app.embeddings import get_embeddings
from app.knowledge.evaluation import evaluate_job_retrieval
from app.knowledge.hybrid import JobHybridRetriever
from app.knowledge.ingestion import JobKnowledgeIngestion
from app.multimodal.reranker import CrossEncoderReranker


@dataclass(frozen=True)
class EmbeddingExperimentSpec:
    name: str
    backend: str
    model_name: str | None

    @property
    def collection_name(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")
        return f"job_rag_eval_{slug}"


def select_embedding_winner(results: list[dict], *, sample_count: int) -> dict | None:
    """Select only a model whose Cross-Encoder reranked every evaluated sample."""
    eligible: list[dict] = []
    for result in results:
        strategy = result.get("strategies", {}).get("hybrid_rerank")
        if not strategy or strategy.get("reranker", {}).get("applied_samples") != sample_count:
            continue
        metrics = strategy.get("metrics", {})
        if "mrr_at_3" not in metrics or "hit_rate_at_1" not in metrics:
            continue
        eligible.append(result)
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda result: (
            -result["strategies"]["hybrid_rerank"]["metrics"]["mrr_at_3"],
            -result["strategies"]["hybrid_rerank"]["metrics"]["hit_rate_at_1"],
            result["strategies"]["hybrid_rerank"]["latency_ms"],
            result["name"],
        ),
    )[0]


def select_provisional_embedding_winner(results: list[dict], *, strategy: str = "vector") -> dict | None:
    """Choose an embedding on vector-only retrieval before lexical/rerank stages can mask it."""
    eligible = [result for result in results if strategy in result.get("strategies", {})]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda result: (
            -result["strategies"][strategy]["metrics"]["mrr_at_3"],
            -result["strategies"][strategy]["metrics"].get("recall_at_5", 0.0),
            -result["strategies"][strategy]["metrics"]["hit_rate_at_1"],
            result["strategies"][strategy]["latency_ms"],
            result["name"],
        ),
    )[0]


def embedding_selection_evidence(results: list[dict], *, strategy: str = "vector", iterations: int = 2000) -> dict | None:
    """Return a paired query-level comparison for the best two available embedding models."""
    eligible = [result for result in results if strategy in result.get("strategies", {})]
    if len(eligible) < 2:
        return None
    ranked = sorted(
        eligible,
        key=lambda result: (
            -result["strategies"][strategy]["metrics"]["mrr_at_3"],
            -result["strategies"][strategy]["metrics"].get("recall_at_5", 0.0),
            -result["strategies"][strategy]["metrics"]["hit_rate_at_1"],
            result["strategies"][strategy]["latency_ms"],
            result["name"],
        ),
    )
    winner, runner_up = ranked[:2]
    winner_details = winner["strategies"][strategy]["report"]["details"]
    runner_details = runner_up["strategies"][strategy]["report"]["details"]
    runner_by_id = {item["query_id"]: item for item in runner_details}
    pairs: list[tuple[float, float]] = []
    for item in winner_details:
        other = runner_by_id.get(item["query_id"])
        if other is None:
            continue
        winner_rr = 1.0 / item["first_relevant_rank"] if item["first_relevant_rank"] and item["first_relevant_rank"] <= 3 else 0.0
        runner_rr = 1.0 / other["first_relevant_rank"] if other["first_relevant_rank"] and other["first_relevant_rank"] <= 3 else 0.0
        pairs.append((winner_rr, runner_rr))
    if not pairs:
        return None
    deltas = [left - right for left, right in pairs]
    rng = random.Random(20260827)
    bootstrap = []
    for _ in range(max(100, iterations)):
        sampled = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        bootstrap.append(sum(sampled) / len(sampled))
    bootstrap.sort()
    lower = bootstrap[int(0.025 * (len(bootstrap) - 1))]
    upper = bootstrap[int(0.975 * (len(bootstrap) - 1))]
    return {
        "strategy": strategy,
        "primary_metric": "mrr_at_3",
        "winner": winner["name"],
        "runner_up": runner_up["name"],
        "paired_query_count": len(pairs),
        "mrr_at_3_delta": sum(deltas) / len(deltas),
        "paired_bootstrap_95_ci": [lower, upper],
        "wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "statistically_clear": lower > 0 or upper < 0,
    }


def run_model_experiment(
    spec: EmbeddingExperimentSpec,
    *,
    samples: list[dict],
    catalog,
    source_corpus_dir: Path,
    experiment_dir: Path,
    reranker_enabled: bool,
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    reranker_local_files_only: bool = True,
    candidate_k: int = 12,
    rrf_k: int = 60,
    strategies: tuple[str, ...] = ("vector", "hybrid", "hybrid_rerank"),
    embedding_local_files_only: bool = True,
    rerank_weight: float = 1.0,
) -> dict:
    """Build one non-production index and evaluate all retrieval strategies."""
    experiment_dir.mkdir(parents=True, exist_ok=True)
    model_started = time.perf_counter()
    embeddings = get_embeddings(
        backend=spec.backend, model_name=spec.model_name, local_files_only=embedding_local_files_only
    )
    model_load_ms = (time.perf_counter() - model_started) * 1000
    ingestion = JobKnowledgeIngestion(
        catalog=catalog,
        source_corpus_dir=source_corpus_dir,
        vector_db_dir=experiment_dir,
        collection_name=spec.collection_name,
        embeddings=embeddings,
        embedding_backend=spec.backend,
        embedding_model=spec.model_name,
    )
    try:
        build_started = time.perf_counter()
        chunk_count = ingestion.rebuild()
        index_build_ms = (time.perf_counter() - build_started) * 1000
        strategy_names = tuple(strategies)
        strategy_results: dict[str, dict] = {}
        invalid = set(strategy_names) - {"vector", "hybrid", "hybrid_rerank"}
        if invalid:
            raise ValueError(f"unsupported strategies: {sorted(invalid)}")
        for strategy in strategy_names:
            reranker = CrossEncoderReranker(
                enabled=reranker_enabled and strategy == "hybrid_rerank",
                model_name=reranker_model,
                local_files_only=reranker_local_files_only,
            )
            retriever = JobHybridRetriever(
                ingestion=ingestion,
                reranker=reranker,
                candidate_k=candidate_k,
                rrf_k=rrf_k,
                strategy=strategy,
                rerank_weight=rerank_weight,
            )
            started = time.perf_counter()
            report = evaluate_job_retrieval(samples, retriever, ranks=(1, 3, 5))
            latency_ms = (time.perf_counter() - started) * 1000
            details = report["details"]
            retrieval_details = [item.get("retrieval", {}) for item in details]
            strategy_results[strategy] = {
                "metrics": report["metrics"],
                "metrics_by_question_type": report["metrics_by_question_type"],
                "latency_ms": latency_ms,
                "mean_query_latency_ms": latency_ms / len(samples) if samples else 0.0,
                "reranker": {
                    "weight": rerank_weight if strategy == "hybrid_rerank" else None,
                    "applied_samples": sum(bool(item.get("reranker_applied")) for item in retrieval_details),
                    "models": sorted({item.get("reranker_model") for item in retrieval_details if item.get("reranker_model")}),
                    "reasons": sorted({item.get("reranker_reason") for item in retrieval_details if item.get("reranker_reason")}),
                },
                "report": report,
            }
        return {
            "name": spec.name,
            "backend": spec.backend,
            "model_name": spec.model_name or "hash",
            "collection": spec.collection_name,
            "chunk_count": chunk_count,
            "sample_count": len(samples),
            "model_load_ms": model_load_ms,
            "index_build_ms": index_build_ms,
            "index_size_bytes": sum(path.stat().st_size for path in ingestion.persist_dir.rglob("*") if path.is_file()),
            "strategies": strategy_results,
        }
    finally:
        ingestion.close()
