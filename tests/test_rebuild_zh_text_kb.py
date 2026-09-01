import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import rebuild_zh_text_kb as rebuild


class RebuildZhTextKbTests(unittest.TestCase):
    def test_chinese_policy_allows_technical_english_terms(self) -> None:
        text = "岗位关键词: Python, FastAPI, LangChain, RAG, Chroma, 中文知识库, 向量检索"

        self.assertTrue(rebuild.is_chinese_primary(text, {"language": "zh"}))
        self.assertFalse(
            rebuild.is_chinese_primary(
                "Primary Purpose: Development and maintenance of software applications.",
                {"language": "en"},
            )
        )

    def test_audit_marks_english_and_missing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection = root / "multimodal_knowledge"
            collection.mkdir(parents=True)
            con = sqlite3.connect(collection / "chroma.sqlite3")
            try:
                con.execute("create table collections (dimension integer)")
                con.execute("insert into collections values (384)")
                con.execute(
                    """
                    create table embedding_metadata (
                        id integer,
                        key text,
                        string_value text,
                        int_value integer,
                        float_value real,
                        bool_value integer
                    )
                    """
                )
                rows = [
                    (1, "chroma:document", "Primary Purpose: backend development role.", None, None, None),
                    (1, "filename", "real_en_jd.md", None, None, None),
                    (1, "source", "real_en_jd.md", None, None, None),
                    (1, "language", "en", None, None, None),
                    (2, "chroma:document", "岗位职责: 负责中文 RAG 知识库、向量检索和 FastAPI 接口开发。", None, None, None),
                    (2, "filename", "zh_jd.md", None, None, None),
                    (2, "source", "zh_jd.md", None, None, None),
                    (2, "modality", "text", None, None, None),
                ]
                con.executemany("insert into embedding_metadata values (?, ?, ?, ?, ?, ?)", rows)
                con.commit()
            finally:
                con.close()

            with patch.object(rebuild.settings, "vector_db_dir", root):
                report = rebuild.audit_collection("multimodal_knowledge")

        self.assertEqual(report.dimension, 384)
        self.assertEqual(report.chunk_count, 2)
        self.assertEqual(report.non_chinese_chunks, 1)
        self.assertIn("real_en_jd.md", report.english_sources)
        self.assertEqual(report.missing_metadata_chunks, 2)

    def test_seed_dry_run_does_not_write_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile_docs"
            jd = root / "jd_docs"
            seed_path = profile / "seed.md"
            profile.mkdir()
            jd.mkdir()
            with (
                patch.object(rebuild.settings, "profile_docs_dir", profile),
                patch.object(rebuild.settings, "jd_docs_dir", jd),
                patch.object(rebuild, "SEED_DOCS", {seed_path: "# 中文种子\n\n项目经历:\n测试"}),
            ):
                created = rebuild.ensure_seed_docs(apply=False)

        self.assertEqual(created, [str(seed_path)])
        self.assertFalse(seed_path.exists())


if __name__ == "__main__":
    unittest.main()
