from __future__ import annotations

import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.agent.skill_runtime import JOB_APPLICATION_SKILL, SkillSession
from app.config import settings
from app.knowledge.catalog import JobCatalog
from app.knowledge.ingestion import JobKnowledgeIngestion
from app.knowledge.profiles import CandidateProfileStore
from app.knowledge.retrieval import JobScopedRetriever
from app.schemas import FitRequest, OneClickApplyRequest
from app.services.application_service import ApplicationService


class ProfileRetrieveInput(BaseModel):
    candidate_id: str | None = Field(default=None, description="候选人标识；未提供时返回待补充状态")
    query: str = Field(..., description="检索问题或关键词")
    k: int = Field(default=5, description="返回片段数量")


class JobRetrieveInput(BaseModel):
    company_name: str = Field(..., description="目标公司")
    job_title: str = Field(..., description="目标岗位")
    query: str = Field(..., description="检索问题或关键词")
    k: int = Field(default=5, description="返回片段数量")


class FitToolInput(BaseModel):
    candidate_id: str | None = None
    company_name: str
    job_title: str
    jd_text: str = ""
    resume_text: str = ""


class OneClickInput(FitToolInput):
    candidate_name: str | None = None
    candidate_email: str | None = None
    candidate_phone: str | None = None


def build_job_tools(
    service: ApplicationService,
    session: SkillSession | None = None,
) -> list[StructuredTool]:
    """Build domain tools whose availability is enforced by active Skill state."""

    def to_json(payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False, default=str)

    def skill_guard() -> str | None:
        if session is not None and not session.is_active(JOB_APPLICATION_SKILL):
            return to_json(
                {
                    "status": "skill_not_active",
                    "stage": "skill_selection",
                    "required_skill": JOB_APPLICATION_SKILL,
                    "next_action": "activate_skill",
                }
            )
        return None

    def retrieve_profile(candidate_id: str | None, query: str, k: int = 5) -> str:
        """从候选人简历、项目、实习知识库检索可验证证据。"""
        if blocked := skill_guard():
            return blocked
        if not candidate_id:
            return to_json(
                {
                    "status": "needs_candidate_evidence",
                    "stage": "evidence_collected",
                    "evidence_level": "missing",
                    "documents": [],
                    "missing_fields": ["candidate_id"],
                    "next_action": "provide_candidate_id_or_resume_text",
                }
            )

        profiles = CandidateProfileStore(
            settings.source_corpus_dir,
            settings.vector_db_dir,
            collection_name=settings.candidate_collection_name,
        )
        try:
            docs = profiles.retrieve(candidate_id, query, k=k)
            return to_json(
                {
                    "status": "ok" if docs else "no_profile_evidence",
                    "stage": "evidence_collected",
                    "evidence_level": "verified_profile" if docs else "missing",
                    "candidate_id": candidate_id,
                    "documents": [
                        {
                            "filename": doc.metadata.get("filename", "unknown"),
                            "section": doc.metadata.get("section", "unknown"),
                            "content": doc.page_content,
                        }
                        for doc in docs
                    ],
                    "missing_fields": [] if docs else ["candidate_id_or_resume_text"],
                    "next_action": None if docs else "provide_resume_text",
                }
            )
        finally:
            profiles.close()

    def retrieve_job(company_name: str, job_title: str, query: str, k: int = 5) -> str:
        """仅在精确命中的公司和岗位内检索 JD。"""
        if blocked := skill_guard():
            return blocked
        jobs = JobKnowledgeIngestion(
            catalog=JobCatalog(settings.job_catalog_path),
            source_corpus_dir=settings.source_corpus_dir,
            vector_db_dir=settings.vector_db_dir,
            collection_name=settings.job_collection_name,
        )
        try:
            result = JobScopedRetriever(catalog=jobs.catalog, job_ingestion=jobs).resolve(
                company_name, job_title, query, k=k
            )
            if result.status != "ok" or result.record is None:
                return to_json(
                    {
                        "status": "job_not_found",
                        "stage": "job_resolved",
                        "job_documents": [],
                        "missing_fields": ["jd_text"],
                        "next_action": "upload_target_jd",
                    }
                )
            return to_json(
                {
                    "status": "ok",
                    "stage": "job_resolved",
                    "job_id": result.record.job_id,
                    "job_documents": [
                        {
                            "filename": doc.metadata.get("filename", "unknown"),
                            "content": doc.page_content,
                        }
                        for doc in result.job_documents
                    ],
                    "missing_fields": [],
                    "next_action": None,
                }
            )
        finally:
            jobs.close()

    def analyze_job_fit(
        company_name: str,
        job_title: str,
        candidate_id: str | None = None,
        jd_text: str = "",
        resume_text: str = "",
    ) -> str:
        """分析候选人与精确岗位的匹配度，并返回机器可读状态。"""
        if blocked := skill_guard():
            return blocked
        return to_json(
            service.analyze_scoped_fit(
                FitRequest(
                    candidate_id=candidate_id,
                    company_name=company_name,
                    job_title=job_title,
                    jd_text=jd_text,
                    resume_text=resume_text,
                )
            )
        )

    def generate_application_package(
        company_name: str,
        job_title: str,
        jd_text: str,
        candidate_id: str | None = None,
        resume_text: str = "",
        candidate_name: str | None = None,
        candidate_email: str | None = None,
        candidate_phone: str | None = None,
    ) -> str:
        """生成本地申请包和邮件草稿；不会执行外部投递。"""
        if blocked := skill_guard():
            return blocked
        result = service.one_click_apply(
            OneClickApplyRequest(
                candidate_id=candidate_id,
                company_name=company_name,
                job_title=job_title,
                jd_text=jd_text,
                resume_text=resume_text,
                candidate_name=candidate_name,
                candidate_email=candidate_email,
                candidate_phone=candidate_phone,
            )
        )
        return to_json(result)

    return [
        StructuredTool.from_function(
            func=retrieve_profile,
            name="retrieve_profile",
            description="检索候选人简历、项目、实习资料；需要已激活求职工作流 Skill。",
            args_schema=ProfileRetrieveInput,
        ),
        StructuredTool.from_function(
            func=retrieve_job,
            name="retrieve_job",
            description="精确检索公司和岗位对应的 JD；需要已激活求职工作流 Skill。",
            args_schema=JobRetrieveInput,
        ),
        StructuredTool.from_function(
            func=analyze_job_fit,
            name="analyze_job_fit",
            description="分析岗位匹配度并返回 status、stage、evidence_level 等状态。",
            args_schema=FitToolInput,
        ),
        StructuredTool.from_function(
            func=generate_application_package,
            name="generate_application_package",
            description="一键生成本地申请包和草稿，返回生成状态与待确认字段。",
            args_schema=OneClickInput,
        ),
    ]
