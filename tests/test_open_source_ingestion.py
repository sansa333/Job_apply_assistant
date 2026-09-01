import tempfile
import unittest
from pathlib import Path

from app.knowledge.ingestion import import_open_source_jobs


class OpenSourceIngestionTests(unittest.TestCase):
    def test_import_reports_inserted_skipped_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "jobs.csv"
            csv_path.write_text(
                "Job_ID,Location,Title,Company,Link,Description\n"
                "1,Shanghai,RAG Engineer,Acme,https://example.test,Build Python retrieval.\n"
                "2,Shanghai,RAG Engineer,Acme,https://example.test,Build Python retrieval.\n"
                "3,Shanghai,,Bad,https://example.test,Missing title.\n",
                encoding="utf-8",
            )
            report = import_open_source_jobs(
                csv_path=csv_path,
                project_markdown_dir=None,
                catalog_path=root / "catalog.sqlite3",
                source_corpus_dir=root / "corpus",
                vector_db_dir=root / "vectors",
            )

        self.assertEqual(report["inserted"], 1)
        self.assertEqual(report["duplicates"], 1)
        self.assertEqual(report["skipped"], 1)
        self.assertGreater(report["chunks_added"], 0)


if __name__ == "__main__":
    unittest.main()
