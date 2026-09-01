import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from app.multimodal.reranker import CrossEncoderReranker
from app.routes.knowledge import _reranker_available_locally


class JobIndexHealthTests(unittest.TestCase):
    def test_reranker_cache_probe_accepts_a_huggingface_cached_model(self) -> None:
        with patch("huggingface_hub.try_to_load_from_cache", return_value="C:/cache/config.json"):
            self.assertTrue(_reranker_available_locally("BAAI/bge-reranker-v2-m3"))

    def test_missing_reranker_keeps_vector_order_and_reports_degradation(self) -> None:
        reranker = CrossEncoderReranker(
            enabled=True,
            model_name="definitely-not-a-local-model",
            local_files_only=True,
        )
        docs = [Document(page_content="first vector result"), Document(page_content="second vector result")]

        result = reranker.rerank("中文查询", docs, top_n=2)

        self.assertFalse(result.applied)
        self.assertEqual(result.docs, docs)
        self.assertIsNone(result.model)
        self.assertIn("unavailable", result.reason)


if __name__ == "__main__":
    unittest.main()
