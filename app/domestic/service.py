from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import asdict

from app.config import settings
from app.domestic.profile import load_candidate_profile, load_candidate_resume_text
from app.domestic.sources import DomesticJobSource, default_domestic_sources
from app.knowledge.catalog import JobCatalog
from app.knowledge.ingestion import JobKnowledgeIngestion
from app.knowledge.models import JobRecord
from app.llm import get_llm, message_to_text
from app.prompts import FIT_ANALYSIS_PROMPT
from app.services.structured_analysis import (
    align_evidence,
    parse_candidate_profile,
    parse_job_description,
    render_evidence_report,
    score_evidence,
    validate_grounded_text,
)


TARGET_TITLE_TERMS = (
    "AI应用", "人工智能应用", "Agent", "智能体", "大模型应用", "LLM应用",
    "RAG", "AI软件", "AI平台", "AI后端", "AIGC应用", "智能问答",
)
TARGET_SKILL_TERMS = (
    "Python", "FastAPI", "LangChain", "LangGraph", "RAG", "Agent", "智能体",
    "BGE", "Embedding", "向量数据库", "Chroma", "PyTorch", "Transformer",
    "Tool Calling", "工具调用", "Prompt", "Docker", "评测", "召回", "重排",
)
EXCLUDED_TITLE_TERMS = (
    "销售", "运营", "产品经理", "设计师", "法务", "财务", "行政", "人力资源",
)


def record_payload(
    record: JobRecord, *, include_description: bool = False, query: str = ""
) -> dict:
    data = asdict(record)
    if not include_description:
        data.pop("description", None)
        data.pop("raw_payload_json", None)
    data["relevance_score"] = rank_job(record, query=query)
    return data


def rank_job(
    record: JobRecord, profile: dict | None = None, *, query: str = ""
) -> float:
    title = record.job_title.lower()
    description = record.description.lower()
    if any(term.lower() in title for term in EXCLUDED_TITLE_TERMS):
        return 0.0
    score = 0.0
    for term in TARGET_TITLE_TERMS:
        if term.lower() in title:
            score += 12.0
        elif term.lower() in description:
            score += 3.0
    profile_skills = (profile or {}).get("skills") or TARGET_SKILL_TERMS
    for term in profile_skills:
        normalized = str(term).lower()
        if normalized in title:
            score += 3.0
        if normalized in description:
            score += 1.5
    if record.recruitment_type in {"campus", "internship"}:
        score += 8.0
    profile_year = (profile or {}).get("graduation_year")
    if record.graduation_year is None:
        score += 2.0
    elif profile_year is not None and record.graduation_year == profile_year:
        score += 5.0
    for role in (profile or {}).get("target_roles") or []:
        normalized = str(role).lower().strip()
        if normalized and normalized in title:
            score += 8.0
        elif normalized and normalized in description:
            score += 2.0
    for city in (profile or {}).get("target_cities") or []:
        normalized = str(city).lower().strip()
        if normalized and normalized in (record.location or "").lower():
            score += 4.0
    if record.job_category in {
        "agent_development", "llm_application", "ai_application", "ai_software"
    }:
        score += 10.0
    if record.degree_requirement in {None, "本科", "硕士"}:
        score += 3.0
    for term in query.split():
        normalized = term.lower().strip()
        if not normalized:
            continue
        if normalized in title:
            score += 15.0
        elif normalized in description:
            score += 5.0
    query_lower = query.lower()
    category_query_terms = {
        "agent_development": ("agent", "智能体", "codeagent", "multi-agent"),
        "llm_application": ("rag", "知识库", "大模型应用", "llm应用", "智能问答"),
        "ai_application": ("ai应用", "应用开发", "全栈", "aigc应用"),
        "ai_software": ("ai软件", "ai系统", "ai devops", "ai平台", "模型服务"),
    }
    for category, terms in category_query_terms.items():
        if record.job_category == category and any(term in query_lower for term in terms):
            score += 25.0
    return round(min(100.0, score), 2)


class DomesticJobService:
    def __init__(
        self,
        *,
        catalog: JobCatalog | None = None,
        sources: dict[str, DomesticJobSource] | None = None,
        ingestion_factory=None,
    ):
        self.catalog = catalog or JobCatalog(settings.job_catalog_path)
        self.sources = sources or default_domestic_sources()
        self.ingestion_factory = ingestion_factory or self._default_ingestion
        self.register_sources()

    def _default_ingestion(self) -> JobKnowledgeIngestion:
        return JobKnowledgeIngestion(
            catalog=self.catalog,
            source_corpus_dir=settings.source_corpus_dir,
            vector_db_dir=settings.vector_db_dir,
            collection_name=settings.job_collection_name,
        )

    def register_sources(self) -> None:
        for source in self.sources.values():
            metadata = source.metadata
            self.catalog.upsert_source(
                source_id=metadata.source_id,
                name=metadata.name,
                source_type=metadata.source_type,
                base_url=metadata.base_url,
                terms_url=metadata.terms_url,
                robots_url=metadata.robots_url,
                schedule_minutes=metadata.schedule_minutes,
                config={
                    "domestic_only": True,
                    "official_source": metadata.source_type.startswith("official_"),
                    "supplemental_public_notice": metadata.source_type == "public_notice",
                },
            )

    def sync_source(self, source_id: str, *, build_index: bool = True) -> dict:
        source = self.sources.get(source_id)
        if source is None:
            raise ValueError(f"unknown domestic source: {source_id}")
        run_id = f"crawl_{uuid.uuid4().hex}"
        self.catalog.start_crawl_run(run_id=run_id, source_id=source_id)
        fetched = inserted = updated = chunks_added = 0
        ingestion = None
        try:
            jobs = source.fetch()
            fetched = len(jobs)
            if fetched == 0:
                raise RuntimeError(
                    "source returned zero jobs; refusing to change existing posting statuses"
                )
            observed = {job.external_id for job in jobs if job.external_id}
            if build_index:
                ingestion = self.ingestion_factory()
            for job in jobs:
                if ingestion is not None:
                    result = ingestion.ingest(
                        job, original_bytes=(job.raw_payload_json or job.description).encode("utf-8")
                    )
                    inserted += int(result.inserted)
                    updated += int(result.updated)
                    chunks_added += result.chunks_added
                else:
                    result = self.catalog.upsert(job)
                    inserted += int(result.inserted)
                    updated += int(result.updated)
            possibly_closed = self.catalog.set_source_missing(source_id, observed)
            self.catalog.finish_crawl_run(
                run_id=run_id,
                status="success",
                fetched_count=fetched,
                inserted_count=inserted,
                updated_count=updated,
            )
            return {
                "run_id": run_id,
                "source_id": source_id,
                "status": "success",
                "fetched": fetched,
                "inserted": inserted,
                "updated": updated,
                "chunks_added": chunks_added,
                "possibly_closed_or_closed": possibly_closed,
                "build_index": build_index,
            }
        except Exception as exc:
            self.catalog.finish_crawl_run(
                run_id=run_id,
                status="error",
                fetched_count=fetched,
                inserted_count=inserted,
                updated_count=updated,
                error_message=f"{type(exc).__name__}: {exc}"[:1000],
            )
            raise
        finally:
            if ingestion is not None:
                ingestion.close()

    def sync_all(self, *, build_index: bool = True) -> dict:
        results: list[dict] = []
        for source_id in self.sources:
            try:
                results.append(self.sync_source(source_id, build_index=build_index))
            except Exception as exc:
                results.append(
                    {
                        "source_id": source_id,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return {
            "status": "success" if all(row["status"] == "success" for row in results) else "partial",
            "results": results,
            "domestic_jobs": self.catalog.count(domestic_only=True),
            "open_domestic_jobs": self.catalog.count(domestic_only=True, status="open"),
        }

    def refresh_all(self) -> dict:
        """Synchronize every source, then rebuild one consistent domestic index."""
        sync_result = self.sync_all(build_index=False)
        index_result = self.rebuild_domestic_index()
        failed_sources = [
            row["source_id"]
            for row in sync_result["results"]
            if row.get("status") != "success"
        ]
        return {
            "status": "success" if not failed_sources else "partial",
            "sync": sync_result,
            "index": index_result,
            "open_domestic_jobs": sync_result["open_domestic_jobs"],
            "jobs_indexed": index_result["jobs_indexed"],
            "chunks_indexed": index_result["chunks_indexed"],
            "failed_sources": failed_sources,
        }

    def stats(self) -> dict:
        records = self.catalog.domestic_records(status="open")
        return {
            "open_jobs": len(records),
            "company_count": len({record.company_name for record in records}),
            "company_distribution": dict(
                sorted(Counter(record.company_name for record in records).items())
            ),
            "category_distribution": dict(
                sorted(Counter(record.job_category or "unknown" for record in records).items())
            ),
            "recruitment_type_distribution": dict(
                sorted(Counter(record.recruitment_type or "unknown" for record in records).items())
            ),
            "domestic_only": True,
        }

    def rebuild_domestic_index(self) -> dict:
        """Build the production job index from domestic records only."""
        records = self.catalog.domestic_records(status="open")
        ingestion = self.ingestion_factory()
        try:
            chunks = ingestion.rebuild_records(records)
        finally:
            ingestion.close()
        return {
            "status": "success",
            "domestic_only": True,
            "jobs_indexed": len(records),
            "chunks_indexed": chunks,
            "collection": settings.job_collection_name,
        }

    def search(
        self,
        *,
        keyword: str = "",
        company_name: str = "",
        location: str = "",
        recruitment_type: str = "",
        graduation_year: int | None = None,
        status: str = "open",
        candidate_id: str = "current_candidate",
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        records = self.catalog.search(
            keyword=keyword,
            company_name=company_name,
            location=location,
            recruitment_type=recruitment_type,
            graduation_year=graduation_year,
            status=status,
            domestic_only=True,
            limit=limit,
            offset=offset,
        )
        profile = load_candidate_profile(settings.source_corpus_dir, candidate_id)
        values = [record_payload(record, query=keyword) for record in records]
        for payload, record in zip(values, records):
            payload["relevance_score"] = rank_job(record, profile, query=keyword)
        values.sort(
            key=lambda row: (
                float(row["relevance_score"]),
                str(row.get("posted_at") or row.get("created_at") or ""),
            ),
            reverse=True,
        )
        return {
            "matches": values,
            "count": len(values),
            "candidate_id": candidate_id,
            "profile_available": profile is not None,
            "filters": {
                "keyword": keyword,
                "company_name": company_name,
                "location": location,
                "recruitment_type": recruitment_type,
                "graduation_year": graduation_year,
                "status": status,
                "domestic_only": True,
            },
        }

    def analyze_job(self, *, job_id: str, candidate_id: str = "current_candidate") -> dict:
        record = self.catalog.get(job_id)
        if record is None or not record.is_domestic:
            raise ValueError("domestic job not found")
        resume_text = load_candidate_resume_text(settings.source_corpus_dir, candidate_id)
        if not resume_text.strip():
            raise ValueError("candidate PDF profile not found")
        parsed_job = parse_job_description(
            company_name=record.company_name,
            job_title=record.job_title,
            description=record.description,
            location=record.location,
            language=record.language,
            source_url=record.source_url,
        )
        candidate_source = {
            "content": resume_text,
            "filename": "resume_extracted.txt",
            "section": "pdf_resume_original_text",
            "chunk_id": "resume:full_text",
        }
        parsed_candidate = parse_candidate_profile(
            candidate_id=candidate_id,
            sources=[candidate_source],
            source_kind="verified_profile",
        )
        matrix = align_evidence(parsed_job, parsed_candidate)
        score = score_evidence(parsed_job, matrix)
        requirements_available = bool(parsed_job.requirements)
        if requirements_available:
            evidence_report = render_evidence_report(job=parsed_job, matrix=matrix, score=score)
        else:
            evidence_report = (
                "## 结构化匹配评分\n\n"
                "企业官网当前页面只提供岗位方向摘要，未提供可逐项核验的任职要求，"
                "因此本次不输出 0–100 匹配分，避免把信息不足误判为不匹配。"
            )
        prompt = FIT_ANALYSIS_PROMPT.format(
            company_name=record.company_name,
            job_title=record.job_title,
            jd_text=record.description,
            resume_text=resume_text,
            profile_context="PDF 简历原文已直接提供；不得改写或补造事实。",
            jd_context="企业招聘官网当前岗位正文已直接提供。",
        )
        result = get_llm(temperature=0.2).invoke(prompt)
        narrative = message_to_text(result.content)
        findings = validate_grounded_text(narrative, [resume_text, record.description])
        return {
            "status": "ok",
            "stage": "analyzed",
            "job_id": record.job_id,
            "company_name": record.company_name,
            "job_title": record.job_title,
            "match_score": score.overall_score if requirements_available else None,
            "match_score_status": (
                "evidence_weighted" if requirements_available else "insufficient_job_requirements"
            ),
            "search_relevance_score": rank_job(
                record, load_candidate_profile(settings.source_corpus_dir, candidate_id)
            ),
            "fit_report": f"{evidence_report}\n\n## 模型辅助的定性建议\n\n{narrative}",
            "evidence_level": "verified_profile",
            "resume_content_policy": "read_only_no_rewrite",
            "parsed_job": parsed_job.model_dump(mode="json"),
            "parsed_candidate": parsed_candidate.model_dump(mode="json"),
            "evidence_matrix": [item.model_dump(mode="json") for item in matrix],
            "score_breakdown": score.model_dump(mode="json"),
            "validation_findings": [item.model_dump(mode="json") for item in findings],
            "source_url": record.source_url,
            "apply_url": record.apply_url,
            "notice": "分析基于只读 PDF 原文和企业官网岗位正文；投递前请打开官网复核。",
        }


def profile_safe_summary(profile: dict | None) -> dict | None:
    if profile is None:
        return None
    allowed = {
        "candidate_id", "highest_degree", "schools", "majors", "graduation_year",
        "graduation_year_evidence", "target_country", "target_cities", "target_roles",
        "skills", "evidence_terms", "default_filters", "excluded_role_signals",
        "resume_content_policy", "source_sha256",
    }
    return {key: value for key, value in profile.items() if key in allowed}
