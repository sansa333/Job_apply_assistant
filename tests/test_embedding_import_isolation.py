import subprocess
import sys
import unittest


class EmbeddingImportIsolationTests(unittest.TestCase):
    def test_job_ingestion_import_does_not_initialize_legacy_rag_stores(self) -> None:
        process = subprocess.run(
            [sys.executable, "-c", "import sys; import app.knowledge.ingestion; print('app.rag' in sys.modules)"],
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertEqual(process.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
