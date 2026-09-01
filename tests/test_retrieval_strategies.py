import unittest

from langchain_core.documents import Document

from app.evaluation.retrieval_strategies import _blend_reranker, paired_strategy_delta
from app.multimodal.reranker import RerankResult


def _doc(chunk_id: str) -> Document:
    return Document(page_content=chunk_id, metadata={"chunk_id": chunk_id})


class RetrievalStrategyEvaluationTests(unittest.TestCase):
    def test_blend_preserves_input_when_reranker_is_not_applied(self) -> None:
        docs = [_doc("one"), _doc("two")]
        result = RerankResult(docs=list(reversed(docs)), applied=False, model=None)

        self.assertEqual(_blend_reranker(docs, result, reranker_weight=0.2), docs)

    def test_full_weight_uses_cross_encoder_order(self) -> None:
        docs = [_doc("one"), _doc("two"), _doc("three")]
        reranked = [docs[2], docs[1], docs[0]]
        result = RerankResult(
            docs=reranked,
            applied=True,
            model="test",
            scores_by_chunk={"one": -2.0, "two": 0.0, "three": 3.0},
        )

        self.assertEqual(_blend_reranker(docs, result, reranker_weight=1.0), reranked)

    def test_paired_delta_uses_query_level_mrr_and_is_reproducible(self) -> None:
        def strategy(name: str, values: list[float], p95: float) -> dict:
            return {
                "strategy": name,
                "details": [
                    {"query_id": f"q{index}", "reciprocal_rank_at_3": value}
                    for index, value in enumerate(values)
                ],
                "resources": {"p95_query_total_ms": p95},
            }

        baseline = strategy("dense", [0.0, 0.5, 0.0, 0.5], 20.0)
        candidate = strategy("candidate", [1.0, 1.0, 0.5, 1.0], 35.0)

        first = paired_strategy_delta(candidate, baseline, iterations=1000)
        second = paired_strategy_delta(candidate, baseline, iterations=1000)

        self.assertEqual(first, second)
        self.assertEqual(first["wins"], 4)
        self.assertAlmostEqual(first["delta"], 0.625)
        self.assertAlmostEqual(first["p95_latency_delta_ms"], 15.0)
        self.assertTrue(first["quality_gate_passed"])


if __name__ == "__main__":
    unittest.main()
