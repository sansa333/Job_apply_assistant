import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


class UserJobUploadTests(unittest.TestCase):
    def test_upload_requires_company_and_job_title(self) -> None:
        with TestClient(app) as client:
            response = client.post("/api/jobs/upload", data={"company_name": "Acme", "jd_text": "Python"})

        self.assertEqual(response.status_code, 422)

    def test_upload_is_immediately_exact_searchable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(settings, "job_catalog_path", root / "job_catalog.sqlite3", create=True),
                patch.object(settings, "source_corpus_dir", root / "source_corpus", create=True),
                patch.object(settings, "vector_db_dir", root / "vector_db"),
            ):
                with TestClient(app) as client:
                    upload = client.post(
                        "/api/jobs/upload",
                        data={
                            "company_name": "Acme Ltd.",
                            "job_title": "RAG Engineer",
                            "jd_text": "Responsibilities: build retrieval. Requirements: Python and Chroma.",
                        },
                    )
                    search = client.get(
                        "/api/jobs/search",
                        params={"company_name": "Acme", "job_title": "RAG Engineer"},
                    )

        self.assertEqual(upload.status_code, 200, upload.text)
        self.assertEqual(search.status_code, 200, search.text)
        self.assertEqual(search.json()["matches"][0]["source_kind"], "user_upload")


if __name__ == "__main__":
    unittest.main()
