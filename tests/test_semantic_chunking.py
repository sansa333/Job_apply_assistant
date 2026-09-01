import unittest

from langchain_core.documents import Document

from app.chunking import split_documents_semantic


class SemanticChunkingTests(unittest.TestCase):
    def test_resume_keeps_project_experience_items_together(self) -> None:
        doc = Document(
            page_content=(
                "张三\n"
                "技能栈:\nPython、FastAPI、LangChain\n\n"
                "项目经历:\n"
                "- 企业知识库 RAG 项目：背景是内部资料检索；职责是实现文档加载、切块和检索；"
                "技术栈是 FastAPI、Chroma；成果是提升问答可信度。\n"
                "- 招聘匹配助手：根据简历和 JD 生成匹配分析。"
            ),
            metadata={"filename": "resume.md", "collection": "profile"},
        )

        chunks = split_documents_semantic([doc], collection_name="profile")
        project_chunks = [chunk for chunk in chunks if chunk.metadata.get("section") == "project_experience"]
        rag_chunk = next(chunk for chunk in project_chunks if "企业知识库 RAG 项目" in chunk.page_content)

        self.assertEqual(rag_chunk.metadata["doc_type"], "resume")
        self.assertEqual(rag_chunk.metadata["chunk_strategy"], "resume_structural")
        self.assertIn("成果是提升问答可信度", rag_chunk.page_content)
        self.assertNotIn("招聘匹配助手", rag_chunk.page_content)

    def test_jd_responsibilities_and_requirements_are_separate(self) -> None:
        doc = Document(
            page_content=(
                "岗位：大模型应用开发工程师\n\n"
                "岗位职责：\n"
                "1. 负责 RAG 应用开发；\n"
                "2. 负责 Agent 工具链调试。\n\n"
                "任职要求：\n"
                "- 熟悉 Python；\n"
                "- 了解 LangChain。\n\n"
                "加分项：有可演示项目经验优先。"
            ),
            metadata={"filename": "jd.md", "collection": "job_description"},
        )

        chunks = split_documents_semantic([doc], collection_name="job_description")
        responsibilities = "\n".join(
            chunk.page_content for chunk in chunks if chunk.metadata.get("section") == "responsibilities"
        )
        requirements = "\n".join(
            chunk.page_content for chunk in chunks if chunk.metadata.get("section") == "requirements"
        )

        self.assertIn("负责 RAG 应用开发", responsibilities)
        self.assertNotIn("熟悉 Python", responsibilities)
        self.assertIn("熟悉 Python", requirements)
        self.assertNotIn("负责 Agent", requirements)
        self.assertTrue(any(chunk.metadata.get("section") == "bonus" for chunk in chunks))

    def test_jd_does_not_create_isolated_title_or_short_hard_condition_chunks(self) -> None:
        doc = Document(
            page_content=(
                "# 示例岗位 JD\n\n"
                "岗位：大模型应用开发工程师\n\n"
                "岗位职责：\n"
                "1. 负责中文 RAG 应用开发和知识库治理；\n"
                "2. 优化文档分块、向量召回和引用来源展示。\n\n"
                "任职要求：\n"
                "- 熟悉 Python、FastAPI、LangChain 和 Chroma；\n"
                "- 能解释检索质量评估指标和 bad case 复盘方法。\n\n"
                "工作地点：北京或远程。"
            ),
            metadata={"filename": "jd.md", "collection": "job_description"},
        )

        chunks = split_documents_semantic([doc], collection_name="job_description")
        sections = [chunk.metadata.get("section") for chunk in chunks]
        joined = "\n".join(chunk.page_content for chunk in chunks)

        self.assertNotIn("job_title", sections)
        self.assertNotIn("hard_conditions", sections)
        self.assertIn("岗位概览", joined)
        self.assertTrue(all(len(chunk.page_content.strip()) > 30 for chunk in chunks))

    def test_markdown_keeps_heading_path(self) -> None:
        doc = Document(
            page_content=(
                "# 项目说明\n"
                "这是项目首页。\n\n"
                "## RAG 架构\n"
                "入库和问答两阶段。\n\n"
                "### 检索流程\n"
                "先召回候选片段，再进行上下文组装。"
            ),
            metadata={"filename": "guide.md"},
        )

        chunks = split_documents_semantic([doc])
        retrieval_chunk = next(chunk for chunk in chunks if "先召回候选片段" in chunk.page_content)

        self.assertEqual(retrieval_chunk.metadata["doc_type"], "markdown")
        self.assertEqual(retrieval_chunk.metadata["section_path"], "项目说明 / RAG 架构 / 检索流程")
        self.assertEqual(retrieval_chunk.metadata["heading_level"], 3)

    def test_csv_chunks_always_include_header(self) -> None:
        doc = Document(
            page_content=(
                "application_id,company_name,job_title,status\n"
                "1,某科技,大模型应用开发工程师,generated\n"
                "2,某云,后端工程师,submitted\n"
            ),
            metadata={"filename": "applications.csv"},
        )

        chunks = split_documents_semantic([doc])

        self.assertGreaterEqual(len(chunks), 1)
        self.assertTrue(all("字段: application_id, company_name, job_title, status" in c.page_content for c in chunks))
        self.assertTrue(all(c.metadata["doc_type"] == "csv" for c in chunks))
        self.assertTrue(all(c.metadata["chunk_strategy"].startswith("csv_rows") for c in chunks))

    def test_long_image_analysis_splits_by_structured_fields(self) -> None:
        doc = Document(
            page_content=(
                "图片文件: demo.png\n"
                "- 图片主题: 简历截图\n"
                "- 关键信息: 候选人展示了 RAG 和 FastAPI 项目经验。\n"
                "- OCR文本:\n"
                + ("候选人具备 RAG 项目经验，熟悉文档加载、语义切块和向量检索。" * 120)
                + "\n- 检索关键词: RAG, FastAPI, 简历, 项目经验"
            ),
            metadata={"filename": "demo.png", "modality": "image"},
        )

        chunks = split_documents_semantic([doc])

        self.assertTrue(any(chunk.metadata.get("section") == "image_ocr" for chunk in chunks))
        self.assertTrue(any("检索关键词" in chunk.page_content for chunk in chunks))
        self.assertTrue(all(chunk.metadata["doc_type"] == "image_analysis" for chunk in chunks))
        self.assertTrue(all(chunk.metadata.get("source_image") == "demo.png" for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
