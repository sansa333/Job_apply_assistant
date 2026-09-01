from __future__ import annotations

import csv
import hashlib
import io
import re
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_MAX_TOKENS = 650
DEFAULT_OVERLAP_TOKENS = 80
JD_MIN_STANDALONE_TOKENS = 80


class ChunkCandidate(TypedDict, total=False):
    text: str
    metadata: dict


RESUME_SECTIONS: dict[str, tuple[str, ...]] = {
    "basic_info": ("基本信息", "个人信息", "联系方式", "个人资料", "候选人资料"),
    "job_intention": ("求职意向", "职业目标", "期望岗位"),
    "education": ("教育经历", "教育背景", "学历背景"),
    "skills": ("技能栈", "专业技能", "技术栈", "技能清单", "技能"),
    "project_experience": ("项目经历", "项目经验", "项目实践", "项目"),
    "internship_experience": ("实习经历", "实习经验"),
    "work_experience": ("工作经历", "工作经验", "职业经历"),
    "awards": ("竞赛", "论文", "证书", "荣誉", "奖项", "资格证书"),
    "self_evaluation": ("自我评价", "个人评价", "个人总结", "自我介绍"),
}

JD_SECTIONS: dict[str, tuple[str, ...]] = {
    "job_title": ("岗位", "岗位名称", "职位", "职位名称", "招聘岗位"),
    "company_background": ("公司介绍", "业务背景", "团队介绍", "部门介绍", "项目背景"),
    "responsibilities": ("职责", "岗位职责", "工作职责", "工作内容", "你将负责"),
    "requirements": ("要求", "任职要求", "岗位要求", "任职资格", "能力要求", "我们希望你"),
    "bonus": ("加分项", "优先条件", "优先考虑", "加分", "bonus"),
    "tech_stack": ("技术栈", "技术要求", "技术关键词", "关键词", "工具链"),
    "hard_conditions": ("地点", "工作地点", "学历", "经验", "薪资", "薪酬", "base", "工作年限"),
}

IMAGE_SECTIONS: dict[str, tuple[str, ...]] = {
    "image_summary": ("图片主题", "主题", "摘要", "图片摘要"),
    "image_key_info": ("关键信息", "关键内容", "主要信息"),
    "image_ocr": ("OCR文本", "OCR 文本", "识别文字", "可见文字", "文字内容"),
    "image_numbers_or_chart": ("图表信息", "数值信息", "图表/数值信息", "图表结论"),
    "image_keywords": ("检索关键词", "关键词"),
    "image_notes": ("解析备注", "备注", "错误信息"),
}

TECH_TERMS = (
    "Python",
    "FastAPI",
    "Pydantic",
    "LangChain",
    "RAG",
    "Agent",
    "Prompt",
    "OpenAI",
    "Qwen",
    "GLM",
    "Chroma",
    "向量数据库",
    "向量检索",
    "知识库",
    "LLM",
    "大模型",
    "Docker",
    "SQL",
    "Redis",
)


def split_documents_semantic(
    docs: list[Document],
    *,
    collection_name: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Document]:
    """Split documents by structure first, then sentence-aware token budget."""

    chunks: list[Document] = []

    for doc in docs:
        doc_type = detect_document_type(doc, collection_name=collection_name)
        try:
            candidates = _split_one_document(doc, doc_type=doc_type)
            if not candidates:
                candidates = _generic_text_candidates(doc, doc_type=doc_type)
            chunks.extend(
                _materialize_candidates(
                    doc,
                    candidates,
                    doc_type=doc_type,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
        except Exception:
            chunks.extend(_fallback_recursive_split(doc, doc_type=doc_type))

    return chunks


def detect_document_type(doc: Document, *, collection_name: str | None = None) -> str:
    metadata = doc.metadata or {}
    explicit = str(metadata.get("doc_type", "")).strip()
    if explicit:
        return explicit

    modality = str(metadata.get("modality", "")).lower()
    if modality == "image":
        return "image_analysis"

    collection = (collection_name or str(metadata.get("collection", ""))).lower()
    if collection == "profile":
        return "resume"
    if collection == "job_description":
        return "job_description"

    source = str(metadata.get("source", "") or metadata.get("filename", ""))
    filename = str(metadata.get("filename", "") or Path(source).name)
    suffix = Path(filename or source).suffix.lower()
    path_hint = source.replace("\\", "/").lower()
    name_hint = filename.lower()
    text = doc.page_content or ""

    if "/profile_docs/" in path_hint or any(hint in name_hint for hint in ("resume", "profile", "cv", "简历")):
        return "resume"
    if "/jd_docs/" in path_hint or any(hint in name_hint for hint in ("jd", "job", "岗位", "职位")):
        return "job_description"
    if suffix == ".csv":
        return "csv"
    if _looks_like_image_analysis(text):
        return "image_analysis"
    if _looks_like_job_description(text):
        return "job_description"
    if _looks_like_resume(text):
        return "resume"
    if suffix in {".md", ".markdown"} or _looks_like_markdown(text):
        return "markdown"
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".doc", ".docx"}:
        return "docx"
    return "generic_text"


def _split_one_document(doc: Document, *, doc_type: str) -> list[ChunkCandidate]:
    if doc_type == "resume":
        return _resume_candidates(doc)
    if doc_type == "job_description":
        return _jd_candidates(doc)
    if doc_type == "markdown":
        return _markdown_candidates(doc)
    if doc_type == "csv":
        return _csv_candidates(doc)
    if doc_type == "image_analysis":
        return _image_analysis_candidates(doc)
    if doc_type in {"pdf", "docx"}:
        return _page_paragraph_candidates(doc, doc_type=doc_type)
    return _generic_text_candidates(doc, doc_type=doc_type)


def _resume_candidates(doc: Document) -> list[ChunkCandidate]:
    sections = _split_by_named_sections(doc.page_content, RESUME_SECTIONS, default_section="basic_info")
    candidates: list[ChunkCandidate] = []

    for section in sections:
        section_key = section["section"]
        text = section["text"].strip()
        if not text:
            continue

        if section_key in {"project_experience", "internship_experience", "work_experience"}:
            items = _split_experience_items(text)
            for item in items:
                title = _derive_item_title(item) or section["title"]
                candidates.append(
                    {
                        "text": item,
                        "metadata": {
                            "section": section_key,
                            "section_title": title,
                            "section_path": _section_path(section["title"], title),
                            "chunk_strategy": "resume_structural",
                        },
                    }
                )
            continue

        candidates.append(
            {
                "text": text,
                "metadata": {
                    "section": section_key,
                    "section_title": section["title"],
                    "section_path": section["title"],
                    "chunk_strategy": "resume_structural",
                },
            }
        )

    return candidates


def _jd_candidates(doc: Document) -> list[ChunkCandidate]:
    sections = _split_by_named_sections(doc.page_content, JD_SECTIONS, default_section="job_description")
    candidates: list[ChunkCandidate] = []
    overview_lines: list[str] = []

    for section in sections:
        section_key = section["section"]
        text = section["text"].strip()
        if not text:
            continue

        if section_key in {"job_description", "job_title", "company_background", "hard_conditions"}:
            overview_lines.append(text)
            continue

        if overview_lines and section_key in {"responsibilities", "requirements"}:
            text = "岗位概览:\n" + "\n".join(overview_lines) + "\n\n" + text
            overview_lines = []

        candidates.append(
            {
                "text": text,
                "metadata": {
                    "section": section_key,
                    "section_title": section["title"],
                    "section_path": section["title"],
                    "chunk_strategy": "jd_structural",
                },
            }
        )

    if overview_lines:
        overview = "\n".join(overview_lines)
        if _estimate_tokens(overview) >= JD_MIN_STANDALONE_TOKENS:
            candidates.insert(
                0,
                {
                    "text": "岗位概览:\n" + overview,
                    "metadata": {
                        "section": "job_overview",
                        "section_title": "岗位概览",
                        "section_path": "岗位概览",
                        "chunk_strategy": "jd_structural",
                    },
                },
            )

    tech_chunk = _jd_tech_keywords_candidate(doc.page_content)
    if tech_chunk:
        candidates.append(tech_chunk)

    hard_conditions = _jd_hard_conditions_candidate(doc.page_content)
    if hard_conditions and _estimate_tokens(hard_conditions["text"]) >= JD_MIN_STANDALONE_TOKENS:
        candidates.append(hard_conditions)

    return candidates


def _markdown_candidates(doc: Document) -> list[ChunkCandidate]:
    lines = doc.page_content.splitlines()
    candidates: list[ChunkCandidate] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_level = 0
    current_title = "正文"

    def flush() -> None:
        nonlocal current_lines, current_level, current_title
        body = "\n".join(current_lines).strip()
        if not body:
            current_lines = []
            return
        path = " / ".join(title for _, title in heading_stack) or current_title
        candidates.append(
            {
                "text": body,
                "metadata": {
                    "section": "markdown_section",
                    "section_title": current_title,
                    "section_path": path,
                    "heading_level": current_level,
                    "chunk_strategy": "markdown_heading",
                },
            }
        )
        current_lines = []

    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            flush()
            current_level = len(match.group(1))
            current_title = match.group(2).strip()
            heading_stack = [(level, title) for level, title in heading_stack if level < current_level]
            heading_stack.append((current_level, current_title))
            current_lines = [line]
            continue
        current_lines.append(line)

    flush()

    if candidates:
        return candidates
    return _generic_text_candidates(doc, doc_type="markdown")


def _csv_candidates(doc: Document) -> list[ChunkCandidate]:
    text = doc.page_content.strip()
    if not text:
        return []

    try:
        sample = text[:2048]
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.excel

    try:
        rows = [row for row in csv.reader(io.StringIO(text), dialect) if any(cell.strip() for cell in row)]
    except csv.Error:
        return _generic_text_candidates(doc, doc_type="csv")

    if not rows:
        return []

    header = [cell.strip() or f"column_{idx + 1}" for idx, cell in enumerate(rows[0])]
    data_rows = rows[1:]
    table_name = str(doc.metadata.get("filename") or Path(str(doc.metadata.get("source", "table.csv"))).name)

    if not data_rows:
        return [
            {
                "text": f"表名: {table_name}\n字段: {', '.join(header)}\n记录: 无数据行",
                "metadata": {
                    "section": "csv_header",
                    "section_title": table_name,
                    "section_path": table_name,
                    "chunk_strategy": "csv_rows",
                    "row_start": 0,
                    "row_end": 0,
                },
            }
        ]

    candidates: list[ChunkCandidate] = []
    group: list[list[str]] = []
    group_start = 1

    def flush(end_row_number: int) -> None:
        nonlocal group, group_start
        if not group:
            return
        candidates.append(
            {
                "text": _format_csv_chunk(table_name, header, group),
                "metadata": {
                    "section": "csv_records",
                    "section_title": table_name,
                    "section_path": table_name,
                    "chunk_strategy": "csv_rows",
                    "row_start": group_start,
                    "row_end": end_row_number,
                },
            }
        )
        group = []
        group_start = end_row_number + 1

    for row_number, row in enumerate(data_rows, start=1):
        group.append(row)
        candidate_text = _format_csv_chunk(table_name, header, group)
        if len(group) >= 20 or _estimate_tokens(candidate_text) >= DEFAULT_MAX_TOKENS:
            flush(row_number)

    flush(len(data_rows))
    return candidates


def _image_analysis_candidates(doc: Document) -> list[ChunkCandidate]:
    text = doc.page_content.strip()
    if not text:
        return []

    filename = str(doc.metadata.get("filename") or Path(str(doc.metadata.get("source", ""))).name)
    if _estimate_tokens(text) <= DEFAULT_MAX_TOKENS:
        return [
            {
                "text": text,
                "metadata": {
                    "section": "image_analysis",
                    "section_title": filename or "image_analysis",
                    "section_path": filename or "image_analysis",
                    "chunk_strategy": "image_structured",
                    "source_image": filename,
                },
            }
        ]

    parsed = _split_labeled_fields(text, IMAGE_SECTIONS, default_section="image_analysis")
    if len(parsed) <= 1:
        return _generic_text_candidates(doc, doc_type="image_analysis", strategy="image_structured")

    candidates: list[ChunkCandidate] = []
    for section in parsed:
        candidates.append(
            {
                "text": section["text"],
                "metadata": {
                    "section": section["section"],
                    "section_title": section["title"],
                    "section_path": _section_path(filename, section["title"]),
                    "chunk_strategy": "image_structured",
                    "source_image": filename,
                },
            }
        )
    return candidates


def _page_paragraph_candidates(doc: Document, *, doc_type: str) -> list[ChunkCandidate]:
    text = doc.page_content.strip()
    if not text:
        return []
    page = doc.metadata.get("page")
    page_label = f"page_{page}" if page is not None else doc_type
    return _paragraph_candidates(
        text,
        metadata={
            "section": page_label,
            "section_title": page_label,
            "section_path": page_label,
            "chunk_strategy": "page_paragraph",
        },
    )


def _generic_text_candidates(
    doc: Document,
    *,
    doc_type: str,
    strategy: str = "generic_paragraph",
) -> list[ChunkCandidate]:
    text = doc.page_content.strip()
    if not text:
        return []
    return _paragraph_candidates(
        text,
        metadata={
            "section": doc_type,
            "section_title": doc_type,
            "section_path": doc_type,
            "chunk_strategy": strategy,
        },
    )


def _paragraph_candidates(text: str, *, metadata: dict) -> list[ChunkCandidate]:
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    candidates: list[ChunkCandidate] = []
    current: list[str] = []
    for paragraph in paragraphs:
        joined = "\n\n".join(current + [paragraph])
        if current and _estimate_tokens(joined) > DEFAULT_MAX_TOKENS:
            candidates.append({"text": "\n\n".join(current), "metadata": dict(metadata)})
            current = [paragraph]
        else:
            current.append(paragraph)

    if current:
        candidates.append({"text": "\n\n".join(current), "metadata": dict(metadata)})
    return candidates


def _materialize_candidates(
    source_doc: Document,
    candidates: list[ChunkCandidate],
    *,
    doc_type: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[Document]:
    parent_doc_id = _parent_doc_id(source_doc)
    base_metadata = dict(source_doc.metadata or {})
    base_metadata["doc_type"] = doc_type
    base_metadata.setdefault("modality", "image" if doc_type == "image_analysis" else "text")

    docs: list[Document] = []
    for candidate in candidates:
        text = candidate["text"].strip()
        if not text:
            continue

        candidate_metadata = dict(candidate.get("metadata", {}))
        parts = _split_long_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        for part_index, part in enumerate(parts):
            metadata = dict(base_metadata)
            metadata.update(candidate_metadata)
            if len(parts) > 1:
                strategy = str(metadata.get("chunk_strategy", "semantic"))
                metadata["chunk_strategy"] = f"{strategy}_token_fallback"
                metadata["chunk_part_index"] = part_index
                metadata["chunk_part_count"] = len(parts)
            metadata["parent_doc_id"] = parent_doc_id
            docs.append(Document(page_content=part, metadata=_clean_metadata(metadata)))

    for index, chunk in enumerate(docs):
        chunk.metadata["chunk_index"] = index
    return docs


def _fallback_recursive_split(doc: Document, *, doc_type: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "！", "？", "；", ";", ". ", ".", "，", ",", " ", ""],
    )
    chunks = splitter.split_documents([doc])
    parent_doc_id = _parent_doc_id(doc)
    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        metadata["doc_type"] = doc_type
        metadata.setdefault("section", doc_type)
        metadata.setdefault("section_title", doc_type)
        metadata.setdefault("section_path", doc_type)
        metadata["chunk_strategy"] = "recursive_fallback"
        metadata["parent_doc_id"] = parent_doc_id
        metadata.setdefault("modality", "image" if doc_type == "image_analysis" else "text")
        metadata["chunk_index"] = index
        chunk.metadata = _clean_metadata(metadata)
    return chunks


def _split_by_named_sections(
    text: str,
    section_map: dict[str, tuple[str, ...]],
    *,
    default_section: str,
) -> list[dict[str, str]]:
    lines = text.splitlines()
    sections: list[dict[str, str]] = []
    current_section = default_section
    current_title = _section_label(default_section, section_map)
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if body:
            sections.append({"section": current_section, "title": current_title, "text": body})
        current_lines = []

    for line in lines:
        heading = _match_section_heading(line, section_map)
        if heading:
            section_key, title, inline_text = heading
            flush()
            current_section = section_key
            current_title = title
            current_lines = [f"{title}: {inline_text}".strip()] if inline_text else [line.strip()]
            continue
        current_lines.append(line)

    flush()
    return sections


def _split_labeled_fields(
    text: str,
    section_map: dict[str, tuple[str, ...]],
    *,
    default_section: str,
) -> list[dict[str, str]]:
    lines = text.splitlines()
    sections: list[dict[str, str]] = []
    current_section = default_section
    current_title = _section_label(default_section, section_map)
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if body:
            sections.append({"section": current_section, "title": current_title, "text": body})
        current_lines = []

    for line in lines:
        heading = _match_section_heading(line, section_map, allow_bullet=True)
        if heading:
            section_key, title, inline_text = heading
            flush()
            current_section = section_key
            current_title = title
            current_lines = [f"{title}: {inline_text}".strip()] if inline_text else [line.strip()]
            continue
        current_lines.append(line)

    flush()
    return sections


def _match_section_heading(
    line: str,
    section_map: dict[str, tuple[str, ...]],
    *,
    allow_bullet: bool = False,
) -> tuple[str, str, str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    if not allow_bullet and re.match(r"^(?:[-*+]|\d+[.)、])\s+\S+", stripped):
        return None
    if allow_bullet:
        stripped = re.sub(r"^[-*+]\s*", "", stripped)

    markdown_match = re.match(r"^#{1,6}\s+(.+?)\s*$", stripped)
    if markdown_match:
        stripped = markdown_match.group(1).strip()

    colon_match = re.match(r"^(.{1,30}?)[：:]\s*(.*)$", stripped)
    candidate_title = colon_match.group(1).strip() if colon_match else stripped
    inline_text = colon_match.group(2).strip() if colon_match else ""
    normalized = _normalize_label(candidate_title)

    for section_key, labels in section_map.items():
        for label in labels:
            norm_label = _normalize_label(label)
            if normalized == norm_label:
                return section_key, label, inline_text

    for section_key, labels in section_map.items():
        for label in labels:
            norm_label = _normalize_label(label)
            if normalized.endswith(norm_label) and len(normalized) <= len(norm_label) + 4:
                return section_key, label, inline_text

    if colon_match:
        return None

    return None


def _split_experience_items(text: str) -> list[str]:
    lines = text.splitlines()
    if len(lines) <= 2:
        return [text.strip()]

    heading = lines[0].strip() if _is_section_title_line(lines[0]) else ""
    items: list[list[str]] = []
    current: list[str] = []

    for line in lines[1 if heading else 0 :]:
        if re.match(r"^\s*(?:[-*+]|\d+[.)、])\s+\S+", line) and current:
            items.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        items.append(current)

    if len(items) <= 1:
        return [text.strip()]

    chunks = []
    for item in items:
        item_text = "\n".join(item).strip()
        chunks.append(f"{heading}\n{item_text}".strip() if heading else item_text)
    return chunks


def _derive_item_title(text: str) -> str:
    for line in text.splitlines():
        stripped = re.sub(r"^\s*(?:[-*+]|\d+[.)、])\s*", "", line.strip())
        if not stripped or _is_section_title_line(stripped):
            continue
        title = re.split(r"[：:，,。;；|-]", stripped, maxsplit=1)[0].strip()
        if title:
            return title[:40]
    return ""


def _jd_tech_keywords_candidate(text: str) -> ChunkCandidate | None:
    found = []
    lowered = text.lower()
    for term in TECH_TERMS:
        if term.lower() in lowered and term not in found:
            found.append(term)
    if not found:
        return None
    return {
        "text": "技术栈关键词:\n" + "\n".join(f"- {term}" for term in found),
        "metadata": {
            "section": "tech_stack",
            "section_title": "技术栈关键词",
            "section_path": "技术栈关键词",
            "chunk_strategy": "jd_structural_enhancement",
        },
    }


def _jd_hard_conditions_candidate(text: str) -> ChunkCandidate | None:
    pattern = re.compile(r"(地点|工作地点|学历|本科|硕士|博士|经验|年经验|薪资|薪酬|base|全职|实习)")
    lines = [line.strip() for line in text.splitlines() if pattern.search(line)]
    if not lines:
        return None
    unique_lines = list(dict.fromkeys(lines))
    return {
        "text": "硬条件:\n" + "\n".join(f"- {line}" for line in unique_lines[:12]),
        "metadata": {
            "section": "hard_conditions",
            "section_title": "硬条件",
            "section_path": "硬条件",
            "chunk_strategy": "jd_structural_enhancement",
        },
    }


def _format_csv_chunk(table_name: str, header: list[str], rows: list[list[str]]) -> str:
    lines = [f"表名: {table_name}", f"字段: {', '.join(header)}", "记录:"]
    for row in rows:
        padded = list(row) + [""] * max(0, len(header) - len(row))
        pairs = [f"{header[idx]}: {padded[idx].strip()}" for idx in range(len(header))]
        lines.append("- " + "; ".join(pairs))
    return "\n".join(lines)


def _split_long_text(text: str, *, max_tokens: int, overlap_tokens: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if _estimate_tokens(text) <= max_tokens:
        return [text]

    segments = _atomic_segments(text, max_tokens=max_tokens)
    chunks: list[str] = []
    current: list[str] = []

    for segment in segments:
        proposed = "\n".join(current + [segment]).strip()
        if current and _estimate_tokens(proposed) > max_tokens:
            chunks.append("\n".join(current).strip())
            current = _tail_overlap(current, overlap_tokens)
        current.append(segment)

    if current:
        chunks.append("\n".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def _atomic_segments(text: str, *, max_tokens: int) -> list[str]:
    paragraphs = _split_paragraphs(text)
    segments: list[str] = []
    for paragraph in paragraphs:
        if _estimate_tokens(paragraph) <= max_tokens:
            segments.append(paragraph)
            continue
        sentences = _split_sentences(paragraph)
        for sentence in sentences:
            if _estimate_tokens(sentence) <= max_tokens:
                segments.append(sentence)
            else:
                segments.extend(_hard_split(sentence, max_chars=max(200, max_tokens)))
    return segments


def _tail_overlap(segments: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []
    selected: list[str] = []
    total = 0
    for segment in reversed(segments):
        segment_tokens = _estimate_tokens(segment)
        if total + segment_tokens > overlap_tokens:
            break
        selected.append(segment)
        total += segment_tokens
    return list(reversed(selected))


def _split_paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    return [part.strip() for part in normalized.splitlines() if part.strip()]


def _split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def _hard_split(text: str, *, max_chars: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end].strip())
        start = end
    return [chunk for chunk in chunks if chunk]


def _estimate_tokens(text: str) -> int:
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    ascii_words = len(re.findall(r"[A-Za-z0-9_./+#-]+", text))
    other_chars = len(re.sub(r"[\u4e00-\u9fffA-Za-z0-9_./+#\-\s]", "", text))
    return max(1, cjk_chars + ascii_words + other_chars // 2)


def _parent_doc_id(doc: Document) -> str:
    metadata = doc.metadata or {}
    basis = "|".join(
        [
            str(metadata.get("source", "")),
            str(metadata.get("filename", "")),
            str(metadata.get("page", "")),
            hashlib.md5((doc.page_content or "").encode("utf-8")).hexdigest(),
        ]
    )
    return hashlib.md5(basis.encode("utf-8")).hexdigest()


def _clean_metadata(metadata: dict) -> dict:
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def _section_label(section_key: str, section_map: dict[str, tuple[str, ...]]) -> str:
    labels = section_map.get(section_key)
    if labels:
        return labels[0]
    return section_key


def _section_path(*parts: str) -> str:
    return " / ".join(part for part in parts if part)


def _normalize_label(label: str) -> str:
    return re.sub(r"[\s#*：:()（）\[\]【】_-]+", "", label).lower()


def _is_section_title_line(line: str) -> bool:
    stripped = line.strip()
    return bool(re.match(r"^#{1,6}\s+", stripped) or re.match(r"^.{1,30}[：:]$", stripped))


def _looks_like_markdown(text: str) -> bool:
    return bool(re.search(r"(?m)^#{1,6}\s+\S+", text))


def _looks_like_resume(text: str) -> bool:
    hints = ("项目经历", "教育经历", "技能栈", "工作经历", "实习经历", "求职意向", "候选人")
    return sum(1 for hint in hints if hint in text) >= 2


def _looks_like_job_description(text: str) -> bool:
    hints = ("岗位职责", "任职要求", "职责", "要求", "职位", "岗位", "薪资", "工作地点")
    return ("职责" in text and "要求" in text) or sum(1 for hint in hints if hint in text) >= 3


def _looks_like_image_analysis(text: str) -> bool:
    hints = ("图片文件", "图片主题", "OCR文本", "OCR 文本", "检索关键词", "关键信息")
    return sum(1 for hint in hints if hint in text) >= 2
