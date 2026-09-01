import tempfile
from unittest.mock import patch
import unittest
from pathlib import Path

from app.knowledge.catalog import JobCatalog
from app.knowledge.ingestion import JobKnowledgeIngestion
from app.knowledge.models import NormalizedJob


class JobKnowledgeIngestionTests(unittest.TestCase):
    def test_index_directory_removal_retries_a_transient_windows_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = JobCatalog(root / "job_catalog.sqlite3")
            ingestion = JobKnowledgeIngestion(
                catalog=catalog, source_corpus_dir=root / "source_corpus", vector_db_dir=root / "vector_db"
            )
            with patch("app.knowledge.ingestion.shutil.rmtree", side_effect=[PermissionError("locked"), None]) as remove:
                ingestion._remove_index_dir()
            ingestion.close()

        self.assertEqual(remove.call_count, 2)

    def test_rebuild_releases_existing_chroma_handle_before_replacing_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = JobCatalog(root / "job_catalog.sqlite3")
            ingestion = JobKnowledgeIngestion(
                catalog=catalog,
                source_corpus_dir=root / "source_corpus",
                vector_db_dir=root / "vector_db",
            )
            ingestion.ingest(
                NormalizedJob(
                    company_name="Acme",
                    job_title="RAG Engineer",
                    description="Build Python retrieval systems.",
                    location=None,
                    source_kind="open_source",
                    source_dataset="unit_test",
                    source_file="acme.md",
                    source_url=None,
                    language="en",
                )
            )
            rebuilt = ingestion.rebuild()
            ingestion.close()

        self.assertGreater(rebuilt, 0)

    def test_ingest_persists_source_and_adds_job_scoped_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = JobCatalog(root / "job_catalog.sqlite3")
            ingestion = JobKnowledgeIngestion(
                catalog=catalog,
                source_corpus_dir=root / "source_corpus",
                vector_db_dir=root / "vector_db",
            )
            result = ingestion.ingest(
                NormalizedJob(
                    company_name="Acme",
                    job_title="RAG Engineer",
                    description="Responsibilities: build retrieval. Requirements: Python, Chroma, FastAPI.",
                    location=None,
                    source_kind="open_source",
                    source_dataset="unit_test",
                    source_file="acme.md",
                    source_url=None,
                    language="en",
                ),
                original_bytes=b"# original JD",
            )
            documents = ingestion.retrieve_for_job(result.record.job_id, "Python retrieval", k=5)

            stored = root / "source_corpus" / "open_source_jobs" / "acme.md"
            stored_exists = stored.exists()
            ingestion.close()

        self.assertTrue(result.inserted)
        self.assertGreater(result.chunks_added, 0)
        self.assertTrue(stored_exists)
        self.assertTrue(documents)
        self.assertTrue(all(doc.metadata["job_id"] == result.record.job_id for doc in documents))
        self.assertTrue(all(doc.metadata["collection"] == "job_knowledge" for doc in documents))


if __name__ == "__main__":
    unittest.main()
