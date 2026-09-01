import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from app.config import settings
from app.knowledge.catalog import JobCatalog
from app.knowledge.hybrid import JobRetrievalResult
from app.knowledge.ingestion import JobKnowledgeIngestion
from app.knowledge.models import NormalizedJob
from app.knowledge.retrieval import JobScopedRetriever


def job(company: str, title: str, description: str) -> NormalizedJob:
    return NormalizedJob(
        company_name=company,
        job_title=title,
        description=description,
        location=None,
        source_kind="open_source",
        source_dataset="unit_test",
        source_file=f"{company}.md",
        source_url=None,
        language="en",
    )


class JobScopedRetrievalTests(unittest.TestCase):
    def test_exact_resolution_uses_hybrid_rerank_metadata(self) -> None:
        class FakeHybridRetriever:
            def retrieve(self, job_id: str, query: str, *, k: int) -> JobRetrievalResult:
                document = Document(page_content="Python requirement", metadata={"job_id": job_id, "chunk_id": "fake"})
                return JobRetrievalResult(
                    documents=[document],
                    strategy="hybrid_rerank",
                    candidate_count=8,
                    reranker_applied=True,
                    reranker_model="fake-cross-encoder",
                    reranker_reason=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = JobCatalog(root / "catalog.sqlite3")
            ingestion = JobKnowledgeIngestion(
                catalog=catalog, source_corpus_dir=root / "corpus", vector_db_dir=root / "vectors"
            )
            try:
                target = ingestion.ingest(job("Acme", "RAG Engineer", "Need Python and Chroma retrieval expertise."))
                retriever = JobScopedRetriever(
                    catalog=catalog, job_ingestion=ingestion, hybrid_retriever=FakeHybridRetriever()
                )
                outcome = retriever.resolve("Acme", "RAG Engineer", "这个岗位有哪些技术要求？", k=3)
            finally:
                ingestion.close()

        self.assertEqual(outcome.status, "ok")
        self.assertEqual(outcome.record.job_id, target.record.job_id)
        self.assertEqual(outcome.retrieval_strategy, "hybrid_rerank")
        self.assertEqual(outcome.candidate_count, 8)
        self.assertTrue(outcome.reranker_applied)

    def test_exact_lookup_returns_not_found_without_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = JobCatalog(root / "catalog.sqlite3")
            ingestion = JobKnowledgeIngestion(
                catalog=catalog, source_corpus_dir=root / "corpus", vector_db_dir=root / "vectors"
            )
            retriever = JobScopedRetriever(catalog=catalog, job_ingestion=ingestion)

            outcome = retriever.resolve("Missing Corp", "RAG Engineer", "Python")
            ingestion.close()

        self.assertEqual(outcome.status, "job_not_found")
        self.assertEqual(outcome.job_documents, [])

    def test_retrieval_never_returns_chunks_from_another_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = JobCatalog(root / "catalog.sqlite3")
            ingestion = JobKnowledgeIngestion(
                catalog=catalog, source_corpus_dir=root / "corpus", vector_db_dir=root / "vectors"
            )
            target = ingestion.ingest(job("Acme", "RAG Engineer", "Need Python and Chroma retrieval expertise."))
            ingestion.ingest(job("Other", "Data Engineer", "Need Python and Spark data pipelines."))
            retriever = JobScopedRetriever(catalog=catalog, job_ingestion=ingestion)

            with patch.object(settings, "enable_reranker", False):
                outcome = retriever.resolve("Acme", "RAG Engineer", "Python", k=5)
            ingestion.close()

        self.assertEqual(outcome.status, "ok")
        self.assertEqual(outcome.record.job_id, target.record.job_id)
        self.assertTrue(outcome.job_documents)
        self.assertTrue(all(doc.metadata["job_id"] == target.record.job_id for doc in outcome.job_documents))


if __name__ == "__main__":
    unittest.main()
