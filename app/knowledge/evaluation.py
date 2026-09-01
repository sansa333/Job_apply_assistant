from __future__ import annotations

import re
from collections import Counter, defaultdict

from app.knowledge.catalog import JobCatalog
from app.knowledge.ingestion import JobKnowledgeIngestion


_STOPWORDS = {
    "about", "with", "that", "this", "from", "will", "have", "your", "the", "and", "for", "are", "job",
    "overview", "responsibilities", "requirements", "technical", "skills", "location", "benefits", "岗位", "职位",
    "要求", "负责", "岗位概览", "技术栈", "关键词", "硬条件", "福利", "career", "without", "company",
    "description", "senior", "data", "role", "position", "candidate", "candidates",
}
_QUESTION_TYPE_ORDER = (
    "overview",
    "responsibilities",
    "technical_skills",
    "qualifications",
    "experience_education",
    "location_work_mode",
    "benefits",
)
_QUESTION_TEMPLATES = {
    "overview": "这个岗位主要是做什么的？",
    "responsibilities": "这个岗位需要承担哪些工作职责？",
    "technical_skills": "这个岗位需要哪些技术技能或工具能力？",
    "qualifications": "这个岗位有哪些任职资格或必备条件？",
    "experience_education": "申请这个岗位需要怎样的工作经验或学历背景？",
    "location_work_mode": "这个岗位的工作地点和远程或混合办公安排是什么？",
    "benefits": "这个岗位提供哪些薪酬、福利或假期待遇？",
}
_INTENT_PATTERNS = {
    "responsibilities": r"\b(responsibilit(?:y|ies)|duties|you will|you'll|responsible for|role includes)\b|职责|负责",
    "technical_skills": r"\b(python|java|sql|docker|kubernetes|fastapi|aws|azure|gcp|spark|airflow|react|javascript|typescript|machine learning|llm|rag|api)\b|技术栈|编程|开发",
    "qualifications": r"\b(requirements?|qualifications?|must have|essential|eligible|candidate should)\b|任职要求|资格|必备",
    "experience_education": r"\b(\d+\+?\s*years?|years? of experience|bachelor'?s|master'?s|degree|phd)\b|学历|经验|年",
    "location_work_mode": r"\b(location|remote|hybrid|on[- ]site|office|based in|london|shanghai|beijing)\b|地点|远程|混合办公",
    "benefits": r"\b(benefits?|salary|compensation|bonus|insurance|pension|holiday|leave|equity|rewards?)\b|福利|薪酬|奖金|假期",
}


def _keywords(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#./-]{2,}|[\u4e00-\u9fff]{2,}", text)
    unique: list[str] = []
    for token in tokens:
        lower = token.lower()
        if lower not in _STOPWORDS and lower not in {item.lower() for item in unique}:
            unique.append(token)
        if len(unique) == 4:
            break
    return unique or ["job description"]


def _chunks_for_job(ingestion: JobKnowledgeIngestion, job_id: str) -> list[dict]:
    result = ingestion.db.get(where={"job_id": job_id}, include=["documents", "metadatas"])
    chunks = [
        {"content": content, "metadata": metadata or {}}
        for content, metadata in zip(result.get("documents", []), result.get("metadatas", []))
    ]
    return sorted(chunks, key=lambda item: int(item["metadata"].get("chunk_index", 0)))


def _question_types_for_chunk(content: str, section: str) -> list[str]:
    text = content.lower()
    found: set[str] = set()
    if section in {"job_overview", "job_description"}:
        found.add("overview")
    if section == "tech_stack":
        found.add("technical_skills")
    if section == "hard_conditions":
        found.add("qualifications")
    if section == "bonus":
        found.add("benefits")
    for question_type, pattern in _INTENT_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.add(question_type)
    return [question_type for question_type in _QUESTION_TYPE_ORDER if question_type in found] or ["overview"]


def _sample_for_chunk(record, chunk: dict, question_type: str, query_id: int) -> dict:
    terms = _keywords(chunk["content"])
    return {
        "query_id": f"job_eval_{query_id:03d}",
        "query": _QUESTION_TEMPLATES[question_type],
        "question_type": question_type,
        "target_section": chunk["metadata"].get("section", "unknown"),
        "anchor_terms": terms,
        "company_name": record.company_name,
        "job_title": record.job_title,
        "job_id": record.job_id,
        "expected_chunk_ids": [chunk["metadata"].get("chunk_id")],
        "expected_sources": [record.source_file],
        "expected_keywords": terms,
        "candidate_pool_size": None,
        "source_kind": record.source_kind,
        "source_dataset": record.source_dataset,
    }


def build_job_eval_samples(catalog: JobCatalog, ingestion: JobKnowledgeIngestion, *, limit: int = 80) -> list[dict]:
    """Build balanced, evidence-backed, job-scoped retrieval queries from indexed real jobs."""
    candidates: dict[str, list[tuple[object, dict]]] = defaultdict(list)
    for record in catalog.all_records():
        if record.source_kind not in {"open_source", "user_upload"}:
            continue
        for chunk in _chunks_for_job(ingestion, record.job_id):
            for question_type in _question_types_for_chunk(chunk["content"], chunk["metadata"].get("section", "")):
                candidates[question_type].append((record, chunk))

    for question_type in candidates:
        candidates[question_type].sort(
            key=lambda item: (item[0].source_dataset, item[0].job_id, item[1]["metadata"].get("chunk_index", 0))
        )

    selected: list[tuple[str, object, dict]] = []
    positions = Counter()
    while len(selected) < limit:
        added = False
        for question_type in _QUESTION_TYPE_ORDER:
            position = positions[question_type]
            if position >= len(candidates[question_type]) or len(selected) >= limit:
                continue
            record, chunk = candidates[question_type][position]
            selected.append((question_type, record, chunk))
            positions[question_type] += 1
            added = True
        if not added:
            break

    samples = [
        _sample_for_chunk(record, chunk, question_type, query_id=index)
        for index, (question_type, record, chunk) in enumerate(selected, start=1)
    ]
    pool_sizes = {
        sample["job_id"]: len(_chunks_for_job(ingestion, sample["job_id"]))
        for sample in samples
    }
    for sample in samples:
        sample["candidate_pool_size"] = pool_sizes[sample["job_id"]]
    return samples


def _aggregate_metrics(details: list[dict], ranks: tuple[int, ...]) -> dict:
    average = lambda values: sum(values) / len(values) if values else 0.0
    metrics = {
        f"hit_rate_at_{rank}": average([float(item[f"hit_at_{rank}"]) for item in details])
        for rank in ranks
    }
    for rank in ranks:
        metrics[f"mrr_at_{rank}"] = average(
            [
                1.0 / item["first_relevant_rank"]
                if item["first_relevant_rank"] is not None and item["first_relevant_rank"] <= rank
                else 0.0
                for item in details
            ]
        )
        metrics[f"recall_at_{rank}"] = average([item[f"recall_at_{rank}"] for item in details])
    max_k = max(ranks)
    metrics[f"keyword_recall_at_{max_k}"] = average([item["keyword_recall"] for item in details])
    return metrics


def _diagnostics(samples: list[dict], ranks: tuple[int, ...]) -> dict:
    average = lambda values: sum(values) / len(values) if values else 0.0
    max_k = max(ranks)
    pool_sizes = [int(sample["candidate_pool_size"]) for sample in samples if sample.get("candidate_pool_size")]
    relevant_counts = [len(set(sample.get("expected_chunk_ids", []))) for sample in samples]
    unique_queries = len({sample.get("query", "").strip().lower() for sample in samples})
    warnings: list[str] = []
    if samples and unique_queries / len(samples) < 0.5:
        warnings.append("low_query_diversity")
    if relevant_counts and max(relevant_counts) <= 1:
        warnings.append("single_relevant_label_recall_equals_hit_rate")
    if pool_sizes and average([size <= max_k for size in pool_sizes]) >= 0.5:
        warnings.append("candidate_pool_at_or_below_max_k_metric_saturation")
    return {
        "unique_query_count": unique_queries,
        "unique_query_ratio": unique_queries / len(samples) if samples else 0.0,
        "mean_relevant_chunks_per_query": average(relevant_counts),
        "multi_relevant_query_ratio": average([count > 1 for count in relevant_counts]),
        "mean_candidate_pool_size": average(pool_sizes),
        f"candidate_pool_le_{max_k}_ratio": average([size <= max_k for size in pool_sizes]),
        "warnings": warnings,
    }


def evaluate_job_retrieval(samples: list[dict], ingestion: JobKnowledgeIngestion, *, ranks: tuple[int, ...] = (1, 3, 5)) -> dict:
    ranks = tuple(sorted({max(1, rank) for rank in ranks}))
    details: list[dict] = []
    max_k = max(ranks)
    batched_retrievals = None
    if hasattr(ingestion, "retrieve_many"):
        batched_retrievals = ingestion.retrieve_many(
            [(sample["job_id"], sample["query"]) for sample in samples], k=max_k
        )
    for index, sample in enumerate(samples):
        retrieval = None
        if batched_retrievals is not None:
            retrieval = batched_retrievals[index]
            docs = retrieval.documents
        elif hasattr(ingestion, "retrieve"):
            retrieval = ingestion.retrieve(sample["job_id"], sample["query"], k=max_k)
            docs = retrieval.documents
        else:
            docs = ingestion.retrieve_for_job(sample["job_id"], sample["query"], k=max_k)
        expected = set(sample["expected_chunk_ids"])
        if not expected:
            raise ValueError(f"sample {sample.get('query_id', index)} has no relevant chunk labels")
        ranked_ids = [document.metadata.get("chunk_id") for document in docs]
        first_rank = next((index for index, chunk_id in enumerate(ranked_ids, start=1) if chunk_id in expected), None)
        context = "\n".join(document.page_content.lower() for document in docs)
        terms = [term.lower() for term in sample["expected_keywords"]]
        detail = {
            "query_id": sample["query_id"],
            "query": sample["query"],
            "question_type": sample.get("question_type", "unclassified"),
            "target_section": sample.get("target_section", "unknown"),
            "job_id": sample["job_id"],
            "expected_chunk_ids": sample["expected_chunk_ids"],
            "retrieved_chunk_ids": ranked_ids,
            "relevant_chunk_count": len(expected),
            "candidate_pool_size": sample.get("candidate_pool_size"),
            "first_relevant_rank": first_rank,
            "reciprocal_rank": 1.0 / first_rank if first_rank and first_rank <= max_k else 0.0,
            "keyword_recall": sum(term in context for term in terms) / len(terms) if terms else 0.0,
        }
        if retrieval is not None:
            detail["retrieval"] = {
                "strategy": retrieval.strategy,
                "candidate_count": retrieval.candidate_count,
                "reranker_applied": retrieval.reranker_applied,
                "reranker_model": retrieval.reranker_model,
                "reranker_reason": retrieval.reranker_reason,
            }
        for rank in ranks:
            retrieved_at_k = set(ranked_ids[:rank])
            matched = expected & retrieved_at_k
            detail[f"hit_at_{rank}"] = bool(matched)
            detail[f"relevant_retrieved_at_{rank}"] = len(matched)
            detail[f"recall_at_{rank}"] = len(matched) / len(expected)
        details.append(detail)

    by_question_type: dict[str, list[dict]] = defaultdict(list)
    for detail in details:
        by_question_type[detail["question_type"]].append(detail)
    metrics_by_question_type = {
        question_type: {"sample_count": len(items), **_aggregate_metrics(items, ranks)}
        for question_type, items in sorted(by_question_type.items())
    }
    return {
        "sample_count": len(samples),
        "question_type_distribution": dict(sorted(Counter(sample.get("question_type", "unclassified") for sample in samples).items())),
        "metrics": _aggregate_metrics(details, ranks),
        "metrics_by_question_type": metrics_by_question_type,
        "diagnostics": _diagnostics(samples, ranks),
        "metric_definitions": {
            "hit_rate_at_k": "mean(1 if at least one relevant chunk occurs in top-k else 0)",
            "recall_at_k": "mean(number of unique relevant chunks in top-k / number of labelled relevant chunks)",
            "mrr_at_k": "mean(reciprocal rank of the first relevant chunk if its rank is at most k, else 0)",
            "keyword_recall_at_k": "diagnostic lexical coverage; not used for embedding model selection",
        },
        "details": details,
    }
