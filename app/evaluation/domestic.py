"""Frozen domestic retrieval fixtures and metrics, independent of rank_job labels.

Labels are deliberately silver. Never promote them to human gold automatically.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from app.knowledge.catalog import JobCatalog
from app.knowledge.ingestion import JobKnowledgeIngestion
from app.knowledge.models import JobRecord
from app.knowledge.normalize import normalize_company_name, normalize_job_title
from app.chunking import JD_CHUNK_SCHEMA_VERSION


def ascii_term(term: str) -> str:
    """English identifiers adjacent to Han characters still form a full term."""
    return rf"(?<![A-Za-z0-9_])(?:{term})(?![A-Za-z0-9_])"


# Written before running either embedding model. Matching is exhaustive across
# the candidate corpus, not a top-N pseudo-label from the evaluated retriever.
TOPICS = (
    ("python", ascii_term("python"), "Python", "这个岗位对 Python 开发有哪些要求？"),
    ("rag", ascii_term("RAG") + r"|检索增强|知识库|向量检索", "RAG 知识库", "需要负责哪些检索增强和知识库相关工作？"),
    ("agent", ascii_term("agents?") + r"|智能体|工具调用", "Agent 智能体", "这个岗位涉及智能体开发的哪些能力和任务？"),
    ("llm", r"大模型|大语言模型|" + ascii_term("LLM"), "大模型 LLM", "岗位对大语言模型应用或研发有什么要求？"),
    ("java", ascii_term("java"), "Java", "哪些职责或任职条件涉及 Java 开发？"),
    ("cpp", r"C\+\+", "C++", "岗位对 C++ 编程能力有哪些要求？"),
    ("sql", ascii_term("SQL") + r"|数据库|MySQL", "SQL 数据库", "工作中需要哪些数据库相关技能？"),
    ("training", r"微调|模型训练|强化学习|" + ascii_term("SFT|RLHF"), "微调 模型训练", "是否涉及模型训练或微调，具体需要做什么？"),
    ("vision", r"计算机视觉|图像|视觉|目标检测", "图像 视觉", "有哪些图像处理或视觉方向的工作？"),
    ("nlp", ascii_term("NLP") + r"|自然语言|文本理解", "NLP 自然语言", "对自然语言处理有哪些技术或经验要求？"),
    ("distributed", r"分布式|高并发|高可用|性能优化", "分布式 高并发", "岗位涉及哪些系统性能与可靠性方面的工程任务？"),
    ("data", r"数据分析|数据处理|数据清洗|数据集|数据挖掘", "数据分析 数据处理", "需要承担哪些数据处理、分析或数据集建设工作？"),
    ("education", r"本科|硕士|博士|学历|学位", "硕士 本科", "这个岗位对学历和教育背景有哪些要求？"),
    ("bonus", r"优先|加分|竞赛|论文", "论文 竞赛", "哪些经历、成果或技能属于优先考虑条件？"),
    ("responsibilities", r"负责|参与|承担|工作内容", "开发 研发", "入职后主要负责哪些工作和交付？"),
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    # JSONL is separated by LF, not Unicode paragraph separators inside JD strings.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def is_notice(record: JobRecord) -> bool:
    return record.source_kind == "public_notice" or "岗位合集" in record.job_title or "notices" in record.source_dataset


def snapshot_records(path: Path) -> list[JobRecord]:
    """Read the existing public snapshot without touching the user's live DB."""
    records = []
    for row in read_jsonl(path):
        defaults = dict(
            company_key=normalize_company_name(row["company_name"]),
            job_title_key=normalize_job_title(row["job_title"]),
            source_kind="open_source",
            source_file=f"{row['job_id']}.json", language="zh",
            content_hash=digest(row["description"]), is_user_uploaded=False,
            created_at="1970-01-01T00:00:00+00:00", is_domestic=True,
        )
        defaults.update({key: value for key, value in row.items() if key in JobRecord.__dataclass_fields__})
        records.append(JobRecord(**defaults))
    return records


def frozen_catalog(path: Path, records: list[JobRecord]) -> JobCatalog:
    if path.exists():
        raise ValueError("Refusing to overwrite a catalog; use a new evaluation run directory")
    catalog = JobCatalog(path)
    with catalog._connect() as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(job_records)")}
        for record in records:
            values = {key: value for key, value in asdict(record).items() if key in columns}
            names = list(values)
            connection.execute(
                f"INSERT INTO job_records ({','.join(names)}) VALUES ({','.join('?' for _ in names)})",
                [values[key] for key in names],
            )
    return catalog


def rank_metrics(ranking: list[str], relevant: set[str], ks=(1, 3, 5, 10, 50)) -> dict:
    """Macro metrics; unanswerable queries are excluded, never given recall=0."""
    ranking = list(dict.fromkeys(ranking))
    first = next((rank for rank, item in enumerate(ranking, 1) if item in relevant), None)
    result = {"relevant_count": len(relevant), "first_relevant_rank": first}
    for k in ks:
        result[f"mrr_at_{k}"] = (1 / first if first is not None and first <= k else 0.0) if relevant else None
        result[f"recall_at_{k}"] = len(set(ranking[:k]) & relevant) / len(relevant) if relevant else None
        result[f"hit_at_{k}"] = float(bool(set(ranking[:k]) & relevant)) if relevant else None
    return result


def aggregate(rows: list[dict]) -> dict:
    keys = sorted({key for row in rows for key in row if key.startswith(("mrr_at_", "recall_at_", "hit_at_"))})
    return {"queries": len(rows), "answerable": sum(row["relevant_count"] > 0 for row in rows), **{
        key: sum(values) / len(values) if (values := [row[key] for row in rows if row.get(key) is not None]) else None
        for key in keys
    }}


def paired_comparison(baseline: list[dict], candidate: list[dict], metric: str, iterations=3000) -> dict:
    base = {row["query_id"]: row for row in baseline}
    other = {row["query_id"]: row for row in candidate}
    if base.keys() != other.keys():
        raise ValueError("Paired comparison requires identical query IDs")
    clusters = defaultdict(list)
    for qid, left in base.items():
        right = other[qid]
        if left.get(metric) is not None and right.get(metric) is not None:
            clusters[left.get("group_id", qid)].append((left[metric], right[metric]))
    pairs = [pair for items in clusters.values() for pair in items]
    if not pairs:
        return {"baseline": None, "candidate": None, "delta": None, "relative_percent": None, "ci95": None}
    before = sum(pair[0] for pair in pairs) / len(pairs)
    after = sum(pair[1] for pair in pairs) / len(pairs)
    rng = random.Random(20260918)
    groups = list(clusters.values())
    draws = []
    for _ in range(iterations):
        sample = [pair for _ in groups for pair in rng.choice(groups)]
        draws.append(sum(right - left for left, right in sample) / len(sample))
    draws.sort()
    return {"baseline": before, "candidate": after, "delta": after - before,
            "relative_percent": 100 * (after - before) / before if before else None,
            "ci95": [draws[int(.025 * (iterations - 1))], draws[int(.975 * (iterations - 1))]],
            "queries": len(pairs), "groups": len(groups), "bootstrap_unit": "duplicate-JD group"}


def _split(group: str) -> str:
    return "test" if int(group[:8], 16) % 10 < 3 else "development"


def build_dataset(directory: Path, records: list[JobRecord], source: Path) -> dict:
    if (directory / "manifest.json").exists():
        raise ValueError("Dataset is frozen; choose a fresh directory")
    directory.mkdir(parents=True, exist_ok=True)
    unique, duplicates = {}, []
    for record in sorted(records, key=lambda item: item.job_id):
        key = digest(re.sub(r"\s+", "", record.description).lower())
        if key in unique:
            duplicates.append({"excluded": record.job_id, "kept": unique[key].job_id})
        else:
            unique[key] = record
    records = list(unique.values())
    # Reuse precisely the production chunker, including its real short-JD pools.
    chunker = JobKnowledgeIngestion.__new__(JobKnowledgeIngestion)
    chunker.source_corpus_dir = directory / "public_sources"
    evidence, queries, qrels, search = [], [], [], []
    for record in records:
        group = digest(re.sub(r"\s+", "", record.description).lower())
        stratum = "notice" if is_notice(record) else "individual_job"
        chunks = chunker._chunks_for_record(record)
        for chunk in chunks:
            evidence.append({"evidence_id": chunk.metadata["chunk_id"], "job_id": record.job_id,
                             "text": chunk.page_content, "metadata": chunk.metadata})
        for topic, pattern, _, question in TOPICS:
            relevant = [chunk for chunk in chunks if re.search(pattern, chunk.page_content, re.I)]
            # Preserve absent-answer tasks for coverage reporting; no fake positives.
            qid = f"e:{record.job_id}:{topic}"
            queries.append({"query_id": qid, "job_id": record.job_id, "query": question,
                            "topic": topic, "stratum": stratum, "group_id": group, "split": _split(group),
                            "pool_size": len(chunks), "annotation_status": "silver_rule_unreviewed"})
            for chunk in relevant:
                qrels.append({"query_id": qid, "evidence_id": chunk.metadata["chunk_id"],
                              "grade": 1, "label_source": f"topic_regex:{topic}", "human_grade": None})
    # Fixed intent queries, independent of the selected embedding and its results.
    for stratum in ("individual_job", "notice"):
        subset = [r for r in records if ("notice" if is_notice(r) else "individual_job") == stratum]
        for topic, pattern, keyword, _ in TOPICS[:12]:
            variants = [({}, keyword), ({}, f"想找{keyword.replace(' ', '或')}相关的开发岗位")]
            variants.extend(({"company_name": company}, keyword) for company in sorted({r.company_name for r in subset}))
            variants.extend(({"location": city, "graduation_year": 2027}, keyword) for city in ("北京", "上海", "深圳"))
            for index, (filters, text) in enumerate(variants):
                relevant = [r.job_id for r in subset if re.search(pattern, r.job_title + "\n" + r.description, re.I)
                            and (not filters.get("company_name") or filters["company_name"] in r.company_name)
                            and (not filters.get("location") or filters["location"] in (r.location or ""))
                            and (not filters.get("graduation_year") or r.graduation_year == filters["graduation_year"])]
                search.append({"query_id": f"s:{stratum}:{topic}:{index}", "keyword": text, "filters": filters,
                               "relevant_ids": relevant, "stratum": stratum, "variant": "natural_language" if index == 1 else "keywords",
                               "annotation_status": "silver_rule_unreviewed"})
    write_jsonl(directory / "jobs.jsonl", [asdict(r) for r in records])
    write_jsonl(directory / "evidence.jsonl", evidence)
    write_jsonl(directory / "queries.jsonl", queries)
    write_jsonl(directory / "qrels.jsonl", qrels)
    write_jsonl(directory / "search_queries.jsonl", search)
    # All same-job query/chunk judgments, including negatives, are available for
    # independent human annotation, not only the current positive pseudo-labels.
    by_job = defaultdict(list)
    labels = {(r["query_id"], r["evidence_id"]) for r in qrels}
    for row in evidence:
        by_job[row["job_id"]].append(row)
    write_jsonl(directory / "annotation_tasks.jsonl", [
        {"query_id": q["query_id"], "query": q["query"], "evidence_id": e["evidence_id"], "text": e["text"],
         "silver_grade": int((q["query_id"], e["evidence_id"]) in labels),
         "annotator_1": None, "annotator_2": None, "adjudicated_grade": None}
        for q in queries for e in by_job[q["job_id"]]
    ])
    manifest = {
        "version": "domestic_retrieval_v3_corrected_silver", "created_at": datetime.now(timezone.utc).isoformat(),
        "chunk_schema_version": JD_CHUNK_SCHEMA_VERSION,
        "label_schema_version": "topic-v3-ascii-identifier-boundaries",
        "source_path": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_note": "Existing public domestic snapshot; no fresh crawl; missing timestamps/source kinds reconstructed and not verified.",
        "jobs": len(records), "individual_jobs": sum(not is_notice(r) for r in records),
        "notices": sum(is_notice(r) for r in records), "evidence": len(evidence), "queries": len(queries),
        "search_queries": len(search), "duplicates_removed": duplicates,
        "annotation_status": "silver_rule_unreviewed", "label_definition": "All chunks matching a fixed topic regex; broad relevance proxy, NOT verified answers.",
        "split_policy": "normalized exact-JD-content hash grouped 70/30; reused source and observed test: regression evaluation, NOT a fresh blind holdout",
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob("*.jsonl")},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def verify_dataset(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, checksum in manifest["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Frozen dataset changed: {name}")
    return manifest
