import unittest

from langchain_core.documents import Document

from app.knowledge.hybrid import JobHybridRetriever
from app.multimodal.reranker import RerankResult


def _doc(chunk_id: str, job_id: str, text: str) -> Document:
    return Document(page_content=text, metadata={"chunk_id": chunk_id, "job_id": job_id})


class _FakeIngestion:
    def __init__(self, vector_docs: list[Document], all_docs: list[Document]):
        self.vector_docs = vector_docs
        self.all_docs = all_docs

    def retrieve_vector_for_job(self, job_id: str, query: str, *, k: int) -> list[Document]:
        return self.vector_docs[:k]

    def get_documents_for_job(self, job_id: str) -> list[Document]:
        return self.all_docs


class _ReverseReranker:
    def __init__(self):
        self.calls: list[tuple[str, list[Document], int]] = []

    def rerank(self, query: str, docs: list[Document], top_n: int) -> RerankResult:
        self.calls.append((query, docs, top_n))
        return RerankResult(docs=list(reversed(docs)), applied=True, model="fake-cross-encoder")


class _BatchReverseReranker(_ReverseReranker):
    def __init__(self):
        super().__init__()
        self.batch_calls: list[list[tuple[str, list[Document]]]] = []

    def rerank_many(self, requests: list[tuple[str, list[Document]]], top_n: int) -> list[RerankResult]:
        self.batch_calls.append(requests)
        return [RerankResult(docs=list(reversed(docs))[:top_n], applied=True, model="fake-cross-encoder") for _, docs in requests]


class _ScoredReverseReranker(_ReverseReranker):
    def rerank(self, query: str, docs: list[Document], top_n: int) -> RerankResult:
        return RerankResult(
            docs=list(reversed(docs)),
            applied=True,
            model="fake-cross-encoder",
            scores_by_chunk={doc.metadata["chunk_id"]: float(index) for index, doc in enumerate(docs)},
        )


class JobHybridRetrieverTests(unittest.TestCase):
    def test_conservative_rerank_fusion_keeps_strong_rrf_first_result_while_applying_cross_encoder(self) -> None:
        general = _doc("general", "job_target", "General role details.")
        skills = _doc("skills", "job_target", "Technical skills include Python and FastAPI.")
        retriever = JobHybridRetriever(
            ingestion=_FakeIngestion([general, skills], [general, skills]),
            reranker=_ScoredReverseReranker(),
            candidate_k=2,
            rrf_k=60,
            rerank_weight=0.2,
        )

        result = retriever.retrieve("job_target", "这个岗位有哪些技能？", k=2)

        self.assertTrue(result.reranker_applied)
        self.assertEqual([doc.metadata["chunk_id"] for doc in result.documents], ["general", "skills"])

    def test_hybrid_batches_cross_encoder_requests_for_multiple_queries(self) -> None:
        general = _doc("general", "job_target", "General role details.")
        skills = _doc("skills", "job_target", "Technical skills include Python and FastAPI.")
        reranker = _BatchReverseReranker()
        retriever = JobHybridRetriever(
            ingestion=_FakeIngestion([general, skills], [general, skills]),
            reranker=reranker,
            candidate_k=2,
            rrf_k=60,
        )

        results = retriever.retrieve_many(
            [("job_target", "这个岗位有哪些技能？"), ("job_target", "这个岗位需要 Python 吗？")], k=1
        )

        self.assertEqual(len(reranker.batch_calls), 1)
        self.assertEqual(len(reranker.batch_calls[0]), 2)
        self.assertTrue(all(result.reranker_applied for result in results))
        self.assertTrue(all(len(result.documents) == 1 for result in results))

    def test_hybrid_fuses_in_job_bm25_candidates_before_cross_encoder_rerank(self) -> None:
        general = _doc("general", "job_target", "This role supports analysis, reporting, and stakeholder communication.")
        skills = _doc("skills", "job_target", "Technical skills include Python, FastAPI, Docker, and SQL.")
        other_job = _doc("other", "job_other", "Python Spark data pipeline experience.")
        reranker = _ReverseReranker()
        retriever = JobHybridRetriever(
            ingestion=_FakeIngestion([general, skills, other_job], [general, skills, other_job]),
            reranker=reranker,
            candidate_k=3,
            rrf_k=60,
        )

        result = retriever.retrieve("job_target", "这个岗位需要哪些 Python 技能？", k=2)

        self.assertEqual(result.strategy, "hybrid_rerank")
        self.assertTrue(result.reranker_applied)
        self.assertEqual([doc.metadata["chunk_id"] for doc in reranker.calls[0][1]], ["general", "skills"])
        self.assertEqual(result.documents[0].metadata["chunk_id"], "skills")
        self.assertTrue(all(doc.metadata["job_id"] == "job_target" for doc in result.documents))


if __name__ == "__main__":
    unittest.main()
