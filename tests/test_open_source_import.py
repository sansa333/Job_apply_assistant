import tempfile
import unittest
from pathlib import Path

from app.knowledge.importers import KyosekCsvAdapter, ProjectMarkdownAdapter


class OpenSourceImportTests(unittest.TestCase):
    def test_csv_adapter_skips_incomplete_rows_and_reports_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.csv"
            path.write_text(
                "Job_ID,Location,Title,Company,Link,Description\n"
                "1,Shanghai,RAG Engineer,Acme,https://example.test/1,Build retrieval systems\n"
                "2,Shanghai,,Acme,https://example.test/2,Missing title\n"
                "3,Shanghai,Data Engineer,Acme,https://example.test/3,\n",
                encoding="utf-8",
            )
            result = KyosekCsvAdapter(path).load()

        self.assertEqual(len(result.jobs), 1)
        self.assertEqual(result.skipped, 2)
        self.assertEqual(result.jobs[0].source_kind, "open_source")
        self.assertEqual(result.jobs[0].source_dataset, "kyosek_jobs_csv")

    def test_real_project_markdown_has_open_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "real_en_jd_01.md"
            path.write_text(
                "# Platform Engineer - Acme\n\n"
                "## Metadata\n- source: test/repository\n- language: en\n\n"
                "## Content\nDesign retrieval systems with Python and Chroma.",
                encoding="utf-8",
            )
            result = ProjectMarkdownAdapter(path).load()

        self.assertEqual(result.skipped, 0)
        self.assertEqual(result.jobs[0].company_name, "Acme")
        self.assertEqual(result.jobs[0].job_title, "Platform Engineer")
        self.assertEqual(result.jobs[0].source_kind, "open_source")
        self.assertEqual(result.jobs[0].source_dataset, "project_real_en_jd")

    def test_real_project_markdown_allows_descriptive_filename_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "real_en_jd_01_platform_engineer.md"
            path.write_text(
                "# Platform Engineer - Acme\n\n## Content\nBuild retrieval systems.", encoding="utf-8"
            )
            result = ProjectMarkdownAdapter(path).load()

        self.assertEqual(len(result.jobs), 1)


if __name__ == "__main__":
    unittest.main()
