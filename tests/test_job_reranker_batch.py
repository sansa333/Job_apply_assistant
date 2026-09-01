import unittest

from langchain_core.documents import Document

from app.multimodal.reranker import CrossEncoderReranker


def _doc(chunk_id: str) -> Document:
    return Document(page_content=chunk_id, metadata={"chunk_id": chunk_id})


class _BatchModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[list[str]], int]] = []

    def predict(self, pairs: list[list[str]], batch_size: int) -> list[float]:
        self.calls.append((pairs, batch_size))
        return [float(pair[1].split("_")[-1]) for pair in pairs]


class CrossEncoderBatchTests(unittest.TestCase):
    def test_rerank_many_flattens_pairs_once_and_restores_each_query_order(self) -> None:
        reranker = CrossEncoderReranker(enabled=True, model_name="fake")
        model = _BatchModel()
        reranker._model = model
        reranker._load_attempted = True

        results = reranker.rerank_many(
            [
                ("q1", [_doc("item_1"), _doc("item_3")]),
                ("q2", [_doc("item_2"), _doc("item_4")]),
            ],
            top_n=1,
        )

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(model.calls[0][0]), 4)
        self.assertEqual(model.calls[0][1], 8)
        self.assertEqual([[doc.metadata["chunk_id"] for doc in result.docs] for result in results], [["item_3"], ["item_4"]])
        self.assertEqual(results[0].scores_by_chunk, {"item_1": 1.0, "item_3": 3.0})
        self.assertTrue(all(result.applied for result in results))


if __name__ == "__main__":
    unittest.main()
