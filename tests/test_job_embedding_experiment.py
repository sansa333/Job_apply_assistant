import tempfile
import unittest
from pathlib import Path

from app.knowledge.catalog import JobCatalog
from app.knowledge.experiments import (
    EmbeddingExperimentSpec,
    run_model_experiment,
    select_embedding_winner,
    select_provisional_embedding_winner,
    embedding_selection_evidence,
)
from app.knowledge.models import NormalizedJob


class JobEmbeddingExperimentTests(unittest.TestCase):
    def test_provisional_selection_uses_vector_metrics_before_hybrid_and_cross_encoder(self) -> None:
        results = [
            {
                "name": "slower",
                "strategies": {"vector": {"metrics": {"mrr_at_3": 0.8, "recall_at_5": 1.0, "hit_rate_at_1": 0.7}, "latency_ms": 30}},
            },
            {
                "name": "winner",
                "strategies": {"vector": {"metrics": {"mrr_at_3": 0.8, "recall_at_5": 1.0, "hit_rate_at_1": 0.7}, "latency_ms": 10}},
            },
        ]

        self.assertEqual(select_provisional_embedding_winner(results)["name"], "winner")

    def test_selection_evidence_uses_paired_query_results(self) -> None:
        def result(name: str, ranks: list[int]) -> dict:
            details = [
                {"query_id": f"q{index}", "first_relevant_rank": rank}
                for index, rank in enumerate(ranks)
            ]
            mrr = sum(1 / rank if rank <= 3 else 0 for rank in ranks) / len(ranks)
            return {
                "name": name,
                "strategies": {
                    "vector": {
                        "metrics": {"mrr_at_3": mrr, "recall_at_5": 1.0, "hit_rate_at_1": 0.0},
                        "latency_ms": 10.0,
                        "report": {"details": details},
                    }
                },
            }

        evidence = embedding_selection_evidence([result("winner", [1, 1, 2]), result("runner", [2, 3, 3])])

        self.assertEqual(evidence["winner"], "winner")
        self.assertGreater(evidence["mrr_at_3_delta"], 0)
        self.assertEqual(evidence["wins"], 3)

    def test_model_selection_requires_applied_reranking_and_uses_mrr_then_hit_rate(self) -> None:
        results = [
            {
                "name": "unavailable-reranker",
                "model_name": "model-a",
                "strategies": {
                    "hybrid_rerank": {
                        "metrics": {"mrr_at_3": 0.99, "hit_rate_at_1": 0.99},
                        "latency_ms": 1.0,
                        "reranker": {"applied_samples": 0},
                    }
                },
            },
            {
                "name": "lower-mrr",
                "model_name": "model-b",
                "strategies": {
                    "hybrid_rerank": {
                        "metrics": {"mrr_at_3": 0.70, "hit_rate_at_1": 0.80},
                        "latency_ms": 10.0,
                        "reranker": {"applied_samples": 5},
                    }
                },
            },
            {
                "name": "winner",
                "model_name": "model-c",
                "strategies": {
                    "hybrid_rerank": {
                        "metrics": {"mrr_at_3": 0.80, "hit_rate_at_1": 0.70},
                        "latency_ms": 20.0,
                        "reranker": {"applied_samples": 5},
                    }
                },
            },
        ]

        winner = select_embedding_winner(results, sample_count=5)

        self.assertEqual(winner["name"], "winner")
        self.assertEqual(winner["model_name"], "model-c")

    def test_hash_experiment_uses_an_isolated_collection_and_reports_all_strategies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = JobCatalog(root / "catalog.sqlite3")
            record = catalog.upsert(
                NormalizedJob(
                    "Acme",
                    "Platform Engineer",
                    "Responsibilities: build retrieval services. Requirements: Python and FastAPI.",
                    None,
                    "open_source",
                    "test",
                    "acme.md",
                    None,
                    "en",
                )
            ).record
            samples = [
                {
                    "query_id": "one",
                    "query": "这个岗位需要哪些技术技能？",
                    "job_id": record.job_id,
                    "expected_chunk_ids": [f"job:{record.job_id}:0"],
                    "expected_keywords": ["Python"],
                    "question_type": "technical_skills",
                    "target_section": "job_overview",
                }
            ]
            result = run_model_experiment(
                EmbeddingExperimentSpec(name="hash", backend="hash", model_name=None),
                samples=samples,
                catalog=catalog,
                source_corpus_dir=root / "corpus",
                experiment_dir=root / "experiments",
                reranker_enabled=False,
                strategies=("vector", "hybrid"),
                embedding_local_files_only=True,
                rerank_weight=0.2,
            )

        self.assertNotEqual(result["collection"], "job_knowledge")
        self.assertEqual(set(result["strategies"]), {"vector", "hybrid"})
        self.assertIn("mrr_at_3", result["strategies"]["hybrid"]["metrics"])
        self.assertIn("recall_at_3", result["strategies"]["hybrid"]["metrics"])
        self.assertGreater(result["index_size_bytes"], 0)
        self.assertTrue(all("latency_ms" in item for item in result["strategies"].values()))


if __name__ == "__main__":
    unittest.main()
