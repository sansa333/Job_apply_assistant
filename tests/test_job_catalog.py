import tempfile
import unittest
from pathlib import Path

from app.knowledge.catalog import JobCatalog
from app.knowledge.models import NormalizedJob
from app.knowledge.normalize import normalize_company_name, normalize_job_title


def make_job(*, description: str = "Build RAG systems.", source_kind: str = "open_source") -> NormalizedJob:
    return NormalizedJob(
        company_name="Bird & Bird LLP",
        job_title="AI-Platform Engineer",
        description=description,
        location="London",
        source_kind=source_kind,
        source_dataset="unit_test",
        source_file="job.md",
        source_url=None,
        language="en",
    )


class JobCatalogTests(unittest.TestCase):
    def test_normalization_removes_company_suffix_and_title_punctuation(self) -> None:
        self.assertEqual(normalize_company_name("Bird & Bird LLP"), normalize_company_name("bird and bird"))
        self.assertEqual(normalize_job_title("AI-Platform Engineer"), normalize_job_title("AI Platform Engineer"))

    def test_duplicate_content_is_not_inserted_twice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = JobCatalog(Path(tmp) / "catalog.sqlite3")
            first = catalog.upsert(make_job())
            second = catalog.upsert(make_job())

        self.assertTrue(first.inserted)
        self.assertFalse(second.inserted)
        self.assertEqual(first.record.job_id, second.record.job_id)

    def test_user_upload_is_selected_before_public_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = JobCatalog(Path(tmp) / "catalog.sqlite3")
            public = catalog.upsert(make_job(description="Public requirements.")).record
            uploaded = catalog.upsert(
                make_job(description="Current requirements.", source_kind="user_upload")
            ).record

            matches = catalog.lookup("Bird and Bird", "AI Platform Engineer")

        self.assertEqual(matches[0].job_id, uploaded.job_id)
        self.assertEqual(matches[1].job_id, public.job_id)


if __name__ == "__main__":
    unittest.main()
