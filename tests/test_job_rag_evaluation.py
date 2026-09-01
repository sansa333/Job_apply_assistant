import tempfile
import unittest
from pathlib import Path

from app.knowledge.catalog import JobCatalog
from app.knowledge.evaluation import build_job_eval_samples, evaluate_job_retrieval
from app.knowledge.ingestion import JobKnowledgeIngestion
from app.knowledge.models import NormalizedJob


class JobRagEvaluationTests(unittest.TestCase):
    def test_generated_samples_cover_multiple_question_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = JobCatalog(root / "catalog.sqlite3")
            ingestion = JobKnowledgeIngestion(
                catalog=catalog, source_corpus_dir=root / "corpus", vector_db_dir=root / "vectors"
            )
            ingestion.ingest(
                NormalizedJob(
                    "Acme",
                    "Platform Engineer",
                    """About the role: Acme builds a platform that helps product teams discover reliable information and make informed decisions. The team partners with engineering, data, security, and customer success groups. This role has broad ownership of a production service, participates in planning meetings, documents operational decisions, and communicates progress to stakeholders. The engineer will work with a collaborative group that values practical delivery, careful technical judgment, and continuous improvement. Responsibilities: build and operate the retrieval platform, mentor engineers, and improve reliability.
Requirements: 5+ years of backend experience and a bachelor's degree in computer science.
Technical skills: Python, FastAPI, SQL, Docker, and Kubernetes.
Location: Shanghai with a hybrid work arrangement.
Benefits: annual bonus, health insurance, and flexible leave.""",
                    "Shanghai",
                    "open_source",
                    "test",
                    "acme.md",
                    None,
                    "en",
                )
            )
            samples = build_job_eval_samples(catalog, ingestion, limit=12)
            report = evaluate_job_retrieval(samples, ingestion, ranks=(1, 3, 5))
            ingestion.close()

        question_types = {sample.get("question_type") for sample in samples}
        self.assertTrue({"responsibilities", "technical_skills", "qualifications"}.issubset(question_types))
        self.assertTrue(all(sample.get("target_section") for sample in samples))
        self.assertTrue(all("请重点说明" not in sample["query"] for sample in samples))
        self.assertTrue(
            all(anchor not in sample["query"] for sample in samples for anchor in sample.get("anchor_terms", []))
        )
        self.assertEqual(sum(report["question_type_distribution"].values()), len(samples))
        self.assertTrue(report["metrics_by_question_type"])
        self.assertTrue(
            all(0.0 <= metrics["mrr_at_5"] <= 1.0 for metrics in report["metrics_by_question_type"].values())
        )

    def test_generated_samples_have_labels_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = JobCatalog(root / "catalog.sqlite3")
            ingestion = JobKnowledgeIngestion(
                catalog=catalog, source_corpus_dir=root / "corpus", vector_db_dir=root / "vectors"
            )
            for company, title, description in [
                ("Acme", "RAG Engineer", "Responsibilities: build Chroma retrieval. Requirements: Python FastAPI."),
                ("Beta", "Data Engineer", "Responsibilities: build Spark pipelines. Requirements: SQL Airflow."),
            ]:
                ingestion.ingest(
                    NormalizedJob(company, title, description, None, "open_source", "test", f"{company}.md", None, "en")
                )
            samples = build_job_eval_samples(catalog, ingestion, limit=4)
            report = evaluate_job_retrieval(samples, ingestion, ranks=(1, 3, 5))
            ingestion.close()

        self.assertGreaterEqual(len(samples), 2)
        self.assertTrue(all(sample["expected_chunk_ids"] for sample in samples))
        self.assertEqual(report["sample_count"], len(samples))
        self.assertIn("hit_rate_at_3", report["metrics"])
        self.assertIn("mrr_at_3", report["metrics"])
        self.assertIn("recall_at_3", report["metrics"])
        self.assertIn("diagnostics", report)
        self.assertGreaterEqual(report["metrics"]["mrr_at_5"], 0.0)
        self.assertLessEqual(report["metrics"]["mrr_at_5"], 1.0)

    def test_recall_counts_all_relevant_chunks_while_hit_rate_needs_only_one(self) -> None:
        from langchain_core.documents import Document

        class FixedRetriever:
            def retrieve_for_job(self, job_id: str, query: str, *, k: int):
                del job_id, query, k
                return [
                    Document(page_content="first", metadata={"chunk_id": "relevant-a"}),
                    Document(page_content="distractor", metadata={"chunk_id": "other"}),
                    Document(page_content="second", metadata={"chunk_id": "relevant-b"}),
                ]

        sample = {
            "query_id": "multi-relevant",
            "query": "候选人需要哪些生产部署与云平台能力？",
            "question_type": "technical_skills",
            "target_section": "requirements",
            "job_id": "job-a",
            "expected_chunk_ids": ["relevant-a", "relevant-b"],
            "expected_keywords": [],
            "candidate_pool_size": 8,
        }
        report = evaluate_job_retrieval([sample], FixedRetriever(), ranks=(1, 3))

        self.assertEqual(report["metrics"]["hit_rate_at_1"], 1.0)
        self.assertEqual(report["metrics"]["recall_at_1"], 0.5)
        self.assertEqual(report["metrics"]["recall_at_3"], 1.0)
        self.assertEqual(report["metrics"]["mrr_at_3"], 1.0)


if __name__ == "__main__":
    unittest.main()
