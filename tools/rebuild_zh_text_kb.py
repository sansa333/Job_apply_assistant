from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chunking import split_documents_semantic
from app.config import settings
from app.embeddings import get_embeddings


TARGET_COLLECTIONS = ("profile", "job_description", settings.mm_collection_name)
REQUIRED_METADATA = {
    "doc_type",
    "modality",
    "collection",
    "filename",
    "source",
    "section",
    "section_title",
    "section_path",
    "chunk_strategy",
    "parent_doc_id",
    "chunk_index",
}
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv"}
MIN_VALID_CHUNKS = 20
MIN_CHUNK_TOKENS = 80
MAX_CHUNK_TOKENS = 650


SEED_DOCS: dict[Path, str] = {
    settings.profile_docs_dir
    / "zh_resume_rag_engineer.md": """# 中文简历：RAG 应用工程师候选人

基本信息:
候选人定位为大模型应用开发工程师，熟悉 Python、FastAPI、Pydantic、LangChain、Chroma、Redis 和 Docker。过去项目重点围绕企业知识库问答、简历/JD 匹配、Agent 工具调用和检索评估闭环，能够把模型接口、向量检索、提示词约束和工程可观测性组合成可演示的业务系统。

技能栈:
- 后端工程：Python、FastAPI、Pydantic、异步接口、环境变量配置、日志记录和错误处理。
- RAG 能力：文档加载、中文语义切块、Hash/HuggingFace Embedding、Chroma 向量库、TopK 检索、引用来源展示。
- 模型应用：OpenAI-compatible API、GLM/Qwen 调用、Prompt Engineering、结构化输出、Agent 工具选择。
- 质量评估：HitRate@K、MRR@K、关键词召回率、bad case 分析、请求耗时和 token 消耗记录。

项目经历:
- 企业知识库 RAG 问答平台：负责从零搭建文本入库、语义分块、向量召回、答案生成和引用展示链路。项目中将简历、JD、项目说明按章节和业务语义切块，避免标题或孤立关键词单独入库；通过检索日志定位召回偏差，并用 reranker 对候选片段二次排序。面试可重点说明为什么要按项目/职责/要求切块，以及如何降低幻觉。
- 招聘匹配与面试准备助手：负责把候选人简历和岗位 JD 转换为可检索知识库，再根据岗位要求生成匹配分析、面试问题和回答素材。系统会优先召回项目经历、技术栈、岗位职责和硬性条件，生成回答时要求给出证据来源。项目价值是把求职材料准备从人工整理变成可追踪的 RAG 工作流。
- Agent 工具调用演示系统：负责设计工具注册、任务拆解和结果汇总流程。Agent 可以根据用户目标选择读取岗位信息、生成申请材料或查询知识库等工具。为了让面试官能追问实现细节，项目保留了工具入参校验、失败兜底和执行日志，能解释 Agent 与普通 RAG 问答的边界。

自我评价:
候选人适合需要快速落地大模型应用的团队，优势是能把模型调用、后端接口、知识库治理和评估指标串成闭环。对于不确定需求，会先做可观测的最小版本，再用日志和评估集持续优化召回质量。
""",
    settings.profile_docs_dir
    / "zh_resume_project_deep_dive.md": """# 项目深挖素材：中文 RAG 与 Agent 面试回答

项目经历:
- 中文文档语义分块治理：项目背景是旧知识库混入英文样本和大量短 chunk，导致检索时经常命中标题、关键词或无上下文片段。我的方案是先审计 Chroma 中的文档语言、chunk 长度、metadata 完整性和来源路径，再按中文简历/JD 的结构重新生成知识库。实现时保留项目经历按项目切分，JD 按职责、要求和岗位概览切分，避免“岗位名”“硬条件”这类短文本单独入库。结果是检索证据更完整，回答更容易带出项目背景、职责和指标。
- RAG 评估闭环：项目背景是单纯看模型回答很难判断检索质量。我的方案是构造中文查询样本，记录 expected_sources 和 expected_keywords，对比无 RAG、向量召回和 rerank 后结果。实现中记录 HitRate@K、MRR@K、KeywordRecall@K、候选 chunk 数和耗时。面试中可以强调评估不是追求单次答案好看，而是持续定位召回不准、切块过碎和上下文噪声。
- 面试回答生成链路：项目目标是根据用户问题快速生成可信回答。系统先根据问题检索候选人资料和 JD，再把证据片段组织进提示词，要求回答包含结论、依据和建议。对于“RAG 项目难点怎么讲”这类问题，会优先召回分块策略、评估指标和工程治理相关片段；对于“岗位要求哪些技能”，会优先召回 JD 的职责、要求和技术关键词。

STAR 表达素材:
Situation：旧索引包含英文数据和短碎片，中文面试助手检索不稳定。
Task：构建干净的中文文本知识库，并保证每个 chunk 都能作为回答证据。
Action：审计现有 Chroma 数据，剔除英文正文，补充中文种子资料，优化 JD 分块规则，重建 profile、job_description 和文本知识 collection。
Result：知识库 chunk 语言统一，metadata 完整，检索结果能稳定命中简历项目、岗位要求和 RAG 治理素材。
""",
    settings.jd_docs_dir
    / "zh_jd_llm_app_engineer.md": """# 中文JD：大模型应用开发工程师

岗位: 大模型应用开发工程师

公司介绍:
团队负责企业级 AI 助手、知识库问答、招聘匹配和运营自动化产品，希望把大模型能力嵌入真实业务流程，而不是只做离线 Demo。

岗位职责:
1. 负责基于大语言模型的业务应用开发，包括 RAG 问答、Agent 工具调用、结构化生成和评估看板。
2. 设计中文文档入库流程，处理简历、JD、项目材料、Markdown、PDF 和 CSV 等文本资料。
3. 优化检索质量，包括语义切块、向量召回、rerank、查询日志分析和 bad case 复盘。
4. 使用 Python、FastAPI、Pydantic 构建稳定接口，并对接 OpenAI-compatible 模型服务。
5. 与产品和业务团队沟通需求，把候选人资料、岗位要求和面试准备流程沉淀为可追踪的知识库能力。

任职要求:
- 熟悉 Python 后端开发，能独立实现 API、配置管理、异常处理和日志记录。
- 理解 RAG 基本原理，熟悉文档加载、chunking、embedding、向量数据库和上下文组装。
- 熟悉 LangChain、Chroma、FastAPI 或同类技术栈，有可演示项目经验。
- 能解释模型幻觉、检索召回、引用来源、评估指标和工程稳定性之间的关系。
- 具备良好的中文技术表达能力，能把复杂实现讲成面试官容易追问和理解的项目故事。

加分项:
- 有多模态 OCR、简历/JD 匹配、Agent 工具调用或 RAG 评估平台经验。
- 了解 Redis、Docker、异步任务、权限控制和审计日志。
""",
    settings.jd_docs_dir
    / "zh_jd_rag_backend_engineer.md": """# 中文JD：RAG 后端工程师

岗位: RAG 后端工程师

业务背景:
公司正在建设面向内部知识库、招聘辅助和客户支持的智能问答系统，需要工程师负责文本知识库治理、检索链路和模型生成服务。

岗位职责:
1. 建设中文文本知识库，支持简历、岗位 JD、项目复盘和业务说明文档入库。
2. 设计适合 RAG 的分块策略，确保 chunk 既能精准召回，又保留足够上下文。
3. 维护 Chroma collection，处理重复入库、旧数据清理、embedding 维度一致性和索引重建。
4. 开发检索增强问答接口，返回答案、引用来源、候选片段数量和耗时信息。
5. 建立检索评估样本，持续跟踪 HitRate@K、MRR@K、关键词召回率和失败案例。

任职要求:
- 熟悉 Python、FastAPI、Pydantic，能编写清晰可维护的后端代码。
- 熟悉 RAG、embedding、reranker、Prompt Engineering 和向量数据库。
- 能根据中文简历和 JD 的结构设计语义分块规则，避免标题、岗位名、关键词孤立入库。
- 能处理本地开发环境、环境变量、数据目录和一次性维护脚本。
- 有较强的问题定位能力，能通过日志、数据库抽样和测试验证修复效果。

加分项:
- 熟悉 LangChain、Chroma、OpenAI-compatible API、GLM 或 Qwen。
- 有求职助手、知识库问答、Agent 项目或面试辅助产品经验。
""",
}


@dataclass
class AuditReport:
    collection: str
    path: str
    exists: bool
    dimension: int | None
    chunk_count: int
    chinese_primary_chunks: int
    non_chinese_chunks: int
    short_chunks: int
    long_chunks: int
    english_sources: list[str]
    image_chunks: int
    missing_metadata_chunks: int
    sample_issues: list[str]


def estimate_tokens(text: str) -> int:
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    ascii_words = len(re.findall(r"[A-Za-z0-9_./+#-]+", text))
    other_chars = len(re.sub(r"[\u4e00-\u9fffA-Za-z0-9_./+#\-\s]", "", text))
    return max(1, cjk_chars + ascii_words + other_chars // 2)


def chinese_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if "\u4e00" <= c <= "\u9fff") / len(chars)


def is_chinese_primary(text: str, metadata: dict[str, str] | None = None) -> bool:
    metadata = metadata or {}
    if str(metadata.get("language", "")).lower() == "en":
        return False
    if chinese_ratio(text) >= 0.25:
        return True
    technical_cn = ("技术栈", "岗位", "职责", "要求", "项目", "知识库", "向量", "检索", "简历")
    return any(term in text for term in technical_cn)


def load_one_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix in {".docx", ".doc"}:
        return Docx2txtLoader(str(path)).load()
    if suffix in {".txt", ".md", ".csv"}:
        return TextLoader(str(path), encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {path.name}")


def read_chroma_rows(collection_dir: Path) -> tuple[int | None, list[tuple[str, dict[str, str]]]]:
    db_path = collection_dir / "chroma.sqlite3"
    if not db_path.exists():
        return None, []
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        dimension_row = cur.execute("select dimension from collections limit 1").fetchone()
        dimension = dimension_row[0] if dimension_row else None
        rows: dict[int, dict[str, str]] = {}
        for row_id, key, value in cur.execute(
            """
            select id, key, coalesce(string_value, cast(int_value as text), cast(float_value as text), cast(bool_value as text))
            from embedding_metadata
            """
        ):
            rows.setdefault(row_id, {})[key] = value or ""
        return dimension, [(values.get("chroma:document", ""), values) for values in rows.values()]
    finally:
        con.close()


def audit_collection(collection_name: str) -> AuditReport:
    collection_dir = settings.vector_db_dir / collection_name
    dimension, rows = read_chroma_rows(collection_dir)
    english_sources = sorted(
        {
            metadata.get("source") or metadata.get("filename") or "unknown"
            for text, metadata in rows
            if not is_chinese_primary(text, metadata)
        }
    )
    missing = sum(1 for _, metadata in rows if not REQUIRED_METADATA.issubset(metadata))
    sample_issues: list[str] = []
    for text, metadata in rows:
        tokens = estimate_tokens(text)
        if not is_chinese_primary(text, metadata):
            sample_issues.append(f"non_chinese: {metadata.get('filename', 'unknown')} :: {text[:80]}")
        elif tokens < 80:
            sample_issues.append(f"short_chunk: {metadata.get('filename', 'unknown')} :: {text[:80]}")
        elif tokens > 650:
            sample_issues.append(f"long_chunk: {metadata.get('filename', 'unknown')} :: {text[:80]}")
        if len(sample_issues) >= 8:
            break
    return AuditReport(
        collection=collection_name,
        path=str(collection_dir),
        exists=collection_dir.exists(),
        dimension=dimension,
        chunk_count=len(rows),
        chinese_primary_chunks=sum(1 for text, metadata in rows if is_chinese_primary(text, metadata)),
        non_chinese_chunks=sum(1 for text, metadata in rows if not is_chinese_primary(text, metadata)),
        short_chunks=sum(1 for text, _ in rows if estimate_tokens(text) < 80),
        long_chunks=sum(1 for text, _ in rows if estimate_tokens(text) > 650),
        english_sources=english_sources,
        image_chunks=sum(1 for _, metadata in rows if str(metadata.get("modality", "")).lower() == "image"),
        missing_metadata_chunks=missing,
        sample_issues=sample_issues,
    )


def source_files_for(collection_name: str) -> list[Path]:
    if collection_name == "profile":
        folders = [settings.profile_docs_dir]
    elif collection_name == "job_description":
        folders = [settings.jd_docs_dir]
    elif collection_name == settings.mm_collection_name:
        folders = [settings.profile_docs_dir, settings.jd_docs_dir]
    else:
        folders = []
    files: list[Path] = []
    for folder in folders:
        files.extend(
            sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
        )
    return files


def build_chunks(collection_name: str) -> list[Document]:
    docs: list[Document] = []
    for path in source_files_for(collection_name):
        for doc in load_one_file(path):
            metadata_collection = collection_name
            doc.metadata["source"] = str(path)
            doc.metadata["filename"] = path.name
            doc.metadata["collection"] = metadata_collection
            doc.metadata["modality"] = "text"
            docs.append(doc)
    chunks = split_documents_semantic(docs, collection_name=collection_name)
    chinese_chunks = [chunk for chunk in chunks if is_chinese_primary(chunk.page_content, chunk.metadata)]
    return compact_short_chunks(chinese_chunks)


def compact_short_chunks(chunks: list[Document]) -> list[Document]:
    grouped: dict[str, list[Document]] = {}
    for chunk in chunks:
        key = str(
            chunk.metadata.get("parent_doc_id")
            or chunk.metadata.get("source")
            or chunk.metadata.get("filename")
            or "unknown"
        )
        grouped.setdefault(key, []).append(chunk)

    compacted: list[Document] = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda doc: int(doc.metadata.get("chunk_index", 0)))
        current: list[Document] = []
        group_result: list[Document] = []
        for chunk in ordered:
            if not current:
                current = [chunk]
                continue

            current_text = join_chunk_texts(current)
            proposed_text = join_chunk_texts(current + [chunk])
            if (
                estimate_tokens(current_text) < MIN_CHUNK_TOKENS
                or estimate_tokens(chunk.page_content) < MIN_CHUNK_TOKENS
            ) and estimate_tokens(proposed_text) <= MAX_CHUNK_TOKENS:
                current.append(chunk)
                continue

            group_result.append(materialize_merged_chunk(current))
            current = [chunk]

        if current:
            group_result.append(materialize_merged_chunk(current))

        compacted.extend(reindex_group(group_result))
    return compacted


def join_chunk_texts(chunks: list[Document]) -> str:
    return "\n\n".join(chunk.page_content.strip() for chunk in chunks if chunk.page_content.strip())


def materialize_merged_chunk(chunks: list[Document]) -> Document:
    if len(chunks) == 1:
        return chunks[0]

    metadata = dict(chunks[0].metadata)
    sections = [str(chunk.metadata.get("section", "")) for chunk in chunks if chunk.metadata.get("section")]
    paths = [str(chunk.metadata.get("section_path", "")) for chunk in chunks if chunk.metadata.get("section_path")]
    metadata["section"] = "+".join(dict.fromkeys(sections))
    metadata["section_path"] = " + ".join(dict.fromkeys(paths))
    metadata["chunk_strategy"] = f"{metadata.get('chunk_strategy', 'semantic')}_merged_short"
    metadata["merged_chunk_count"] = len(chunks)
    return Document(page_content=join_chunk_texts(chunks), metadata=metadata)


def reindex_group(chunks: list[Document]) -> list[Document]:
    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)
        metadata["chunk_index"] = index
        chunk.metadata = metadata
    return chunks


def count_effective_source_chunks() -> int:
    return len(build_chunks("profile")) + len(build_chunks("job_description"))


def ensure_seed_docs(*, apply: bool) -> list[str]:
    existing_effective_chunks = count_effective_source_chunks()
    if existing_effective_chunks >= MIN_VALID_CHUNKS:
        return []
    created: list[str] = []
    for path, content in SEED_DOCS.items():
        if path.exists():
            continue
        created.append(str(path))
        if apply:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    return created


def rebuild_collection(collection_name: str) -> int:
    report = audit_collection(collection_name)
    if collection_name == settings.mm_collection_name and report.image_chunks:
        raise RuntimeError(
            f"{collection_name} contains {report.image_chunks} image chunks; refusing to rebuild text collection."
        )

    collection_dir = settings.vector_db_dir / collection_name
    if collection_dir.exists():
        shutil.rmtree(collection_dir)
    collection_dir.mkdir(parents=True, exist_ok=True)

    chunks = build_chunks(collection_name)
    if not chunks:
        return 0

    db = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(collection_dir),
    )
    db.add_documents(chunks)
    return len(chunks)


def print_json(data: object) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def run_dry_run() -> None:
    seed_candidates = ensure_seed_docs(apply=False)
    reports = [asdict(audit_collection(name)) for name in TARGET_COLLECTIONS]
    planned_chunks = {name: len(build_chunks(name)) for name in TARGET_COLLECTIONS}
    print_json(
        {
            "mode": "dry-run",
            "vector_db_dir": str(settings.vector_db_dir),
            "embedding_backend": settings.embedding_backend,
            "seed_docs_would_create": seed_candidates,
            "planned_chunks_before_seed": planned_chunks,
            "audit": reports,
        }
    )


def run_apply() -> None:
    created = ensure_seed_docs(apply=True)
    rebuilt = {name: rebuild_collection(name) for name in TARGET_COLLECTIONS}
    reports = [asdict(audit_collection(name)) for name in TARGET_COLLECTIONS]
    print_json(
        {
            "mode": "apply",
            "vector_db_dir": str(settings.vector_db_dir),
            "embedding_backend": settings.embedding_backend,
            "seed_docs_created": created,
            "rebuilt_chunks": rebuilt,
            "audit": reports,
        }
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit and rebuild Chinese text RAG knowledge bases.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Audit current Chroma data without modifying files.")
    mode.add_argument("--apply", action="store_true", help="Create seed docs if needed and rebuild target collections.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.dry_run:
        run_dry_run()
    else:
        run_apply()


if __name__ == "__main__":
    main()
