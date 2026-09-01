from __future__ import annotations

import csv
from time import perf_counter

from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.agent.workflow import WorkflowTrace
from app.domain.job_application import AgentStage
from app.knowledge.catalog import JobCatalog
from app.knowledge.importers import UserUploadAdapter
from app.knowledge.ingestion import JobKnowledgeIngestion
from app.knowledge.profiles import CandidateProfileStore
from app.knowledge.retrieval import JobScopedRetriever
from app.llm import extract_token_usage, get_llm, message_to_text
from app.prompts import COVER_LETTER_PROMPT, EMAIL_PROMPT, FIT_ANALYSIS_PROMPT, INTERVIEW_PROMPT
from app.schemas import ContactStatus, EvidenceLevel, FitRequest, OneClickApplyRequest
from app.utils.file_io import now_id, safe_filename, write_json, write_text
from app.utils.request_log import elapsed_ms, log_request_event, new_request_id, now_ms
from app.services.structured_analysis import (
    align_evidence,
    parse_candidate_profile,
    parse_job_description,
    render_evidence_report,
    score_evidence,
    validate_grounded_text,
)


class ApplicationService:
    """Core service for fit analysis and one-click application package generation."""

    def __init__(self):
        self.llm = get_llm(temperature=0.2)

    def _run_prompt(self, template: str, variables: dict) -> str:
        text, _ = self._run_prompt_with_usage(template, variables)
        return text

    def _run_prompt_with_usage(self, template: str, variables: dict) -> tuple[str, dict | None]:
        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(**variables)
        result = self.llm.invoke(messages)
        return message_to_text(getattr(result, "content", result)), extract_token_usage(result)

    def build_context(self, req: FitRequest) -> dict:
        return {
            "company_name": req.company_name,
            "job_title": req.job_title,
            "jd_text": req.jd_text,
            "resume_text": req.resume_text or "用户未直接输入简历文本，请优先依据候选人RAG检索上下文。",
            "profile_context": "请使用 candidate_profile 的受控检索路径。",
            "jd_context": "请使用 job_knowledge 的精确岗位检索路径。",
        }

    @staticmethod
    def _evidence(documents: list, identifier: str) -> list[dict]:
        return [
            {
                identifier: document.metadata.get(identifier, ""),
                "filename": document.metadata.get("filename", "unknown"),
                "section": document.metadata.get("section", "unknown"),
                "content": document.page_content,
            }
            for document in documents
        ]

    @staticmethod
    def _context_from_docs(documents: list, label: str) -> str:
        if not documents:
            return "未检索到相关资料。"
        return "\n\n".join(
            f"[{label}{index} | {doc.metadata.get('filename', 'unknown')}]\n{doc.page_content}"
            for index, doc in enumerate(documents, start=1)
        )

    @staticmethod
    def _evidence_level(req: FitRequest, candidate_documents: list) -> EvidenceLevel:
        if candidate_documents:
            return EvidenceLevel.VERIFIED_PROFILE
        if req.resume_text.strip():
            return EvidenceLevel.USER_PROVIDED
        return EvidenceLevel.MISSING

    @staticmethod
    def _contact_metadata(req: OneClickApplyRequest) -> tuple[ContactStatus, list[str]]:
        fields = {
            "candidate_name": req.candidate_name,
            "candidate_email": req.candidate_email,
            "candidate_phone": req.candidate_phone,
        }
        missing = [name for name, value in fields.items() if not value]
        if not missing:
            return ContactStatus.CONFIRMED, []
        if len(missing) == len(fields):
            return ContactStatus.PENDING_CONFIRMATION, missing
        return ContactStatus.PARTIAL, missing

    @staticmethod
    def _contact_value(value: object | None) -> str:
        return str(value) if value else "待确认"

    @staticmethod
    def _generic_candidate_guidance(req: FitRequest, job_description: str) -> str:
        return (
            "## 候选人证据待补充\n"
            "当前没有候选人资料库证据或当轮简历文本，因此未生成个人匹配分或个人经历断言。\n\n"
            "## 岗位准备建议\n"
            f"请围绕该岗位 JD 补充可验证的项目、职责与成果：{job_description[:600]}\n\n"
            "## 下一步\n"
            "粘贴简历/项目经历，或提供 candidate_id 后重新生成证据驱动的匹配分析。"
        )

    def analyze_scoped_fit(self, req: FitRequest) -> dict:
        """Perform exact catalogue resolution before any model generation."""
        trace = WorkflowTrace()
        trace.record(
            AgentStage.REQUEST_ACCEPTED,
            status="ok",
            detail="已接收岗位匹配请求，并建立请求级工作流状态。",
        )
        catalog = JobCatalog(settings.job_catalog_path)
        jobs = JobKnowledgeIngestion(
            catalog=catalog,
            source_corpus_dir=settings.source_corpus_dir,
            vector_db_dir=settings.vector_db_dir,
            collection_name=settings.job_collection_name,
        )
        profiles = CandidateProfileStore(
            settings.source_corpus_dir,
            settings.vector_db_dir,
            collection_name=settings.candidate_collection_name,
        )
        try:
            query = req.question or f"{req.company_name} {req.job_title}"
            resolution_started = perf_counter()
            resolution = JobScopedRetriever(catalog=catalog, job_ingestion=jobs).resolve(
                req.company_name, req.job_title, query, k=6
            )
            if resolution.status == "job_not_found" and req.jd_text.strip():
                parsed = UserUploadAdapter(
                    company_name=req.company_name,
                    job_title=req.job_title,
                    description=req.jd_text,
                    source_file="inline_job.md",
                ).load()
                if parsed.jobs:
                    jobs.ingest(parsed.jobs[0], original_bytes=req.jd_text.encode("utf-8"))
                    resolution = JobScopedRetriever(catalog=catalog, job_ingestion=jobs).resolve(
                        req.company_name, req.job_title, query, k=6
                    )
            if resolution.status == "job_not_found" or resolution.record is None:
                trace.record(
                    AgentStage.BLOCKED,
                    status="job_not_found",
                    detail="未精确找到公司与岗位组合，工作流未使用相似岗位替代。",
                    tool_name="retrieve_job",
                    started_at=resolution_started,
                )
                return {
                    "status": "job_not_found",
                    "stage": "job_resolved",
                    "company_name": req.company_name,
                    "job_title": req.job_title,
                    "message": "知识库中没有该公司与岗位的岗位描述。请上传或粘贴对应 JD 后再进行匹配分析。",
                    "upload_action": "/api/jobs/upload",
                    "job_evidence": [],
                    "candidate_evidence": [],
                    "evidence_level": EvidenceLevel.USER_PROVIDED if req.resume_text.strip() else EvidenceLevel.MISSING,
                    "missing_fields": ["jd_text"],
                    "next_action": "upload_target_jd",
                    "workflow_trace": trace.as_dict(),
                }

            trace.record(
                AgentStage.JOB_RESOLVED,
                status="ok",
                detail=f"已精确解析岗位 {resolution.record.company_name} / {resolution.record.job_title}。",
                tool_name="retrieve_job",
                started_at=resolution_started,
            )

            parse_started = perf_counter()
            parsed_job = parse_job_description(
                company_name=resolution.record.company_name,
                job_title=resolution.record.job_title,
                description=resolution.record.description,
                location=resolution.record.location,
                language=resolution.record.language,
                source_url=resolution.record.source_url,
            )
            trace.record(
                AgentStage.REQUIREMENTS_PARSED,
                status="ok" if parsed_job.requirements else "no_requirements",
                detail=f"结构化解析得到 {len(parsed_job.requirements)} 条岗位要求。",
                tool_name="parse_job_requirements",
                started_at=parse_started,
            )

            evidence_started = perf_counter()
            candidate_documents = profiles.retrieve(req.candidate_id, query, k=6) if req.candidate_id else []
            evidence_level = self._evidence_level(req, candidate_documents)
            job_evidence = self._evidence(resolution.job_documents, "job_id")
            candidate_evidence = self._evidence(candidate_documents, "candidate_id")
            candidate_sources = [
                {
                    **item,
                    "chunk_id": document.metadata.get("chunk_id", ""),
                }
                for item, document in zip(candidate_evidence, candidate_documents)
            ]
            if req.resume_text.strip():
                candidate_sources.append(
                    {
                        "content": req.resume_text,
                        "filename": "request_resume_text.md",
                        "section": "user_provided_resume",
                        "chunk_id": "request:resume_text",
                    }
                )
            trace.record(
                AgentStage.EVIDENCE_COLLECTED,
                status="ok" if candidate_sources else "missing",
                detail=f"收集到 {len(job_evidence)} 条岗位证据和 {len(candidate_sources)} 条候选人证据来源。",
                tool_name="retrieve_profile" if req.candidate_id else None,
                started_at=evidence_started,
            )
            retrieval = {
                "strategy": resolution.retrieval_strategy,
                "candidate_count": resolution.candidate_count,
                "reranker_applied": resolution.reranker_applied,
                "reranker_model": resolution.reranker_model,
                "reranker_reason": resolution.reranker_reason,
            }

            if evidence_level == EvidenceLevel.MISSING:
                return {
                    "status": "needs_candidate_evidence",
                    "stage": "evidence_collected",
                    "job_id": resolution.record.job_id,
                    "fit_report": self._generic_candidate_guidance(req, resolution.record.description),
                    "job_evidence": job_evidence,
                    "candidate_evidence": [],
                    "retrieval": retrieval,
                    "source_kind": resolution.record.source_kind,
                    "source_dataset": resolution.record.source_dataset,
                    "historical_notice": "基于知识库中的公开历史岗位描述或用户上传 JD；岗位当前开放状态请以官方渠道为准。",
                    "evidence_level": evidence_level,
                    "missing_fields": ["candidate_id", "resume_text"],
                    "next_action": "provide_candidate_id_or_resume_text",
                    "parsed_job": parsed_job.model_dump(mode="json"),
                    "workflow_trace": trace.as_dict(),
                }

            alignment_started = perf_counter()
            parsed_candidate = parse_candidate_profile(
                candidate_id=req.candidate_id,
                sources=candidate_sources,
                source_kind=evidence_level.value,
            )
            evidence_matrix = align_evidence(parsed_job, parsed_candidate)
            trace.record(
                AgentStage.EVIDENCE_ALIGNED,
                status="ok",
                detail=f"完成 {len(evidence_matrix)} 条岗位要求与候选人证据的逐项对齐。",
                tool_name="align_candidate_evidence",
                started_at=alignment_started,
            )
            scoring_started = perf_counter()
            score_breakdown = score_evidence(parsed_job, evidence_matrix)
            trace.record(
                AgentStage.SCORED,
                status="ok",
                detail=f"证据加权匹配分为 {score_breakdown.overall_score:.2f}。",
                tool_name="score_job_fit",
                started_at=scoring_started,
            )

            context = {
                "company_name": resolution.record.company_name,
                "job_title": resolution.record.job_title,
                "jd_text": resolution.record.description,
                "resume_text": req.resume_text or "用户未直接输入简历文本，请优先依据候选人 RAG 检索上下文。",
                "profile_context": self._context_from_docs(candidate_documents, "候选人资料"),
                "jd_context": self._context_from_docs(resolution.job_documents, "岗位资料"),
            }
            narrative = self._run_prompt(FIT_ANALYSIS_PROMPT, context)
            allowed_evidence = [item.get("content", "") for item in job_evidence + candidate_sources]
            validation_findings = validate_grounded_text(narrative, allowed_evidence)
            evidence_report = render_evidence_report(
                job=parsed_job,
                matrix=evidence_matrix,
                score=score_breakdown,
            )
            report = f"{evidence_report}\n\n## 模型辅助的定性建议\n\n{narrative}"
            trace.record(
                AgentStage.OUTPUT_VALIDATED,
                status="warning" if validation_findings else "ok",
                detail=(
                    f"发现 {len(validation_findings)} 条需要人工复核的生成断言。"
                    if validation_findings
                    else "生成建议通过量化断言与外部投递声明检查。"
                ),
                tool_name="validate_grounded_output",
            )
            return {
                "status": "ok",
                "stage": "analyzed",
                "job_id": resolution.record.job_id,
                "fit_report": report,
                "job_evidence": job_evidence,
                "candidate_evidence": candidate_evidence,
                "retrieval": retrieval,
                "source_kind": resolution.record.source_kind,
                "source_dataset": resolution.record.source_dataset,
                "historical_notice": "基于知识库中的公开历史岗位描述或用户上传 JD 进行分析；岗位当前开放状态请以官方渠道为准。",
                "evidence_level": evidence_level,
                "missing_fields": [],
                "next_action": "review_validation_findings" if validation_findings else None,
                "match_score": score_breakdown.overall_score,
                "parsed_job": parsed_job.model_dump(mode="json"),
                "parsed_candidate": parsed_candidate.model_dump(mode="json"),
                "evidence_matrix": [item.model_dump(mode="json") for item in evidence_matrix],
                "score_breakdown": score_breakdown.model_dump(mode="json"),
                "validation_findings": [item.model_dump(mode="json") for item in validation_findings],
                "workflow_trace": trace.as_dict(),
            }
        finally:
            jobs.close()
            profiles.close()

    def analyze_fit(self, req: FitRequest) -> str:
        result = self.analyze_scoped_fit(req)
        return result.get("fit_report") or result["message"]

    def generate_cover_letter(self, req: FitRequest) -> str:
        scoped = self.analyze_scoped_fit(req)
        if scoped["status"] not in {"ok", "needs_candidate_evidence"}:
            return scoped["message"]
        return self._run_prompt(COVER_LETTER_PROMPT, self._context_from_scoped_result(req, scoped))

    def generate_interview_questions(self, req: FitRequest) -> str:
        scoped = self.analyze_scoped_fit(req)
        if scoped["status"] not in {"ok", "needs_candidate_evidence"}:
            return scoped["message"]
        return self._run_prompt(INTERVIEW_PROMPT, self._context_from_scoped_result(req, scoped))

    @staticmethod
    def _context_from_scoped_result(req: FitRequest, result: dict) -> dict:
        job_evidence = result.get("job_evidence", [])
        candidate_evidence = result.get("candidate_evidence", [])
        job_text = "\n\n".join(item.get("content", "") for item in job_evidence)
        profile_text = "\n\n".join(item.get("content", "") for item in candidate_evidence)
        return {
            "company_name": req.company_name,
            "job_title": req.job_title,
            "jd_text": job_text,
            "resume_text": req.resume_text or "候选人经历待补充；不得虚构个人事实。",
            "profile_context": profile_text or "未检索到候选人资料；仅可给通用或待确认表述。",
            "jd_context": job_text or "未检索到岗位资料。",
        }

    def generate_email(self, req: OneClickApplyRequest, fit_report: str, cover_letter: str) -> str:
        variables = {
            "candidate_name": self._contact_value(req.candidate_name),
            "candidate_email": self._contact_value(req.candidate_email),
            "candidate_phone": self._contact_value(req.candidate_phone),
            "company_name": req.company_name,
            "job_title": req.job_title,
            "fit_report": fit_report,
            "cover_letter": cover_letter,
        }
        return self._run_prompt(EMAIL_PROMPT, variables)

    def one_click_apply(self, req: OneClickApplyRequest) -> dict:
        request_id = new_request_id()
        start = now_ms()
        token_usage_total: dict[str, int] = {}
        application_id = now_id()
        company = safe_filename(req.company_name)
        job = safe_filename(req.job_title)
        output_dir = settings.outputs_dir / f"{application_id}_{company}_{job}"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            scoped = self.analyze_scoped_fit(req)
            if scoped["status"] not in {"ok", "needs_candidate_evidence"}:
                raise ValueError(scoped["message"])
            context = self._context_from_scoped_result(req, scoped)
            fit_report = scoped["fit_report"]
            cover_letter, usage = self._run_prompt_with_usage(COVER_LETTER_PROMPT, context)
            self._merge_token_usage(token_usage_total, usage)
            interview_questions, usage = self._run_prompt_with_usage(INTERVIEW_PROMPT, context)
            self._merge_token_usage(token_usage_total, usage)
            application_email, usage = self._run_prompt_with_usage(
                EMAIL_PROMPT,
                {
                    "candidate_name": self._contact_value(req.candidate_name),
                    "candidate_email": self._contact_value(req.candidate_email),
                    "candidate_phone": self._contact_value(req.candidate_phone),
                    "company_name": req.company_name,
                    "job_title": req.job_title,
                    "fit_report": fit_report,
                    "cover_letter": cover_letter,
                },
            )
            self._merge_token_usage(token_usage_total, usage)

            trace = WorkflowTrace.model_validate(scoped.get("workflow_trace", {}))
            trace.record(
                AgentStage.MATERIALS_GENERATED,
                status="ok",
                detail="已生成匹配报告、求职信、面试准备材料和邮件草稿。",
                tool_name="generate_materials",
            )
            allowed_evidence = [
                item.get("content", "")
                for item in scoped.get("job_evidence", []) + scoped.get("candidate_evidence", [])
            ]
            if req.resume_text.strip():
                allowed_evidence.append(req.resume_text)
            validation_findings = [
                finding
                for artifact in (cover_letter, interview_questions, application_email)
                for finding in validate_grounded_text(artifact, allowed_evidence)
            ]
            trace.record(
                AgentStage.OUTPUT_VALIDATED,
                status="warning" if validation_findings else "ok",
                detail=(
                    f"申请包发现 {len(validation_findings)} 条需要人工复核的断言。"
                    if validation_findings
                    else "申请包通过量化断言与虚假投递声明检查。"
                ),
                tool_name="validate_grounded_output",
            )

            write_text(output_dir / "fit_report.md", fit_report)
            write_text(output_dir / "cover_letter.md", cover_letter)
            write_text(output_dir / "interview_questions.md", interview_questions)
            write_text(output_dir / "application_email.md", application_email)
            write_json(output_dir / "evidence_matrix.json", scoped.get("evidence_matrix", []))
            write_json(output_dir / "score_breakdown.json", scoped.get("score_breakdown", {}))
            write_json(
                output_dir / "validation_report.json",
                [item.model_dump(mode="json") for item in validation_findings],
            )

            contact_status, contact_missing_fields = self._contact_metadata(req)
            trace.record(
                AgentStage.AWAITING_HUMAN_CONFIRMATION,
                status="generated_not_submitted",
                detail="材料仅保存为本地草稿，等待用户核验事实、联系方式和发送渠道。",
                tool_name="persist_application_draft",
            )
            write_json(output_dir / "workflow_trace.json", trace.as_dict())
            summary = {
                "application_id": application_id,
                "company_name": req.company_name,
                "job_title": req.job_title,
                "candidate_name": self._contact_value(req.candidate_name),
                "candidate_email": self._contact_value(req.candidate_email),
                "candidate_phone": self._contact_value(req.candidate_phone),
                "output_dir": str(output_dir),
                "status": "generated_not_submitted",
                "stage": "generated",
                "evidence_level": scoped["evidence_level"],
                "contact_status": contact_status,
                "missing_fields": contact_missing_fields,
                "next_action": "confirm_contact_details_before_sending" if contact_missing_fields else None,
                "note": "系统已生成投递材料和邮件草稿，真实提交前需要人工确认。",
                "match_score": scoped.get("match_score"),
                "validation_findings": [item.model_dump(mode="json") for item in validation_findings],
                "workflow_trace": trace.as_dict(),
            }

            write_json(output_dir / "submission.json", summary)
            self._append_application_record(summary)

            output_paths = [
                str(output_dir / "fit_report.md"),
                str(output_dir / "cover_letter.md"),
                str(output_dir / "interview_questions.md"),
                str(output_dir / "application_email.md"),
                str(output_dir / "submission.json"),
            ]
            log_request_event(
                route="/api/one-click-apply",
                request_id=request_id,
                elapsed_ms_value=elapsed_ms(start),
                token_usage=token_usage_total or None,
                output_paths=output_paths,
                status="success",
            )

            return {
                **summary,
                "fit_report": fit_report,
                "cover_letter": cover_letter,
                "interview_questions": interview_questions,
                "application_email": application_email,
                "match_score": scoped.get("match_score"),
                "score_breakdown": scoped.get("score_breakdown", {}),
                "evidence_matrix": scoped.get("evidence_matrix", []),
                "validation_findings": [item.model_dump(mode="json") for item in validation_findings],
                "workflow_trace": trace.as_dict(),
            }
        except Exception as exc:
            log_request_event(
                route="/api/one-click-apply",
                request_id=request_id,
                elapsed_ms_value=elapsed_ms(start),
                token_usage=token_usage_total or None,
                output_paths=[str(output_dir)],
                status="error",
                error_type=type(exc).__name__,
            )
            raise

    def _append_application_record(self, row: dict) -> None:
        path = settings.data_dir / "applications.csv"
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "application_id",
            "company_name",
            "job_title",
            "candidate_name",
            "candidate_email",
            "candidate_phone",
            "output_dir",
            "status",
            "stage",
            "evidence_level",
            "contact_status",
            "missing_fields",
            "next_action",
            "note",
        ]

        exists = path.exists()
        with path.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    @staticmethod
    def _merge_token_usage(total: dict[str, int], usage: dict | None) -> None:
        if not usage:
            return
        for key, value in usage.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                total[key] = total.get(key, 0) + value
            elif isinstance(value, float):
                total[key] = total.get(key, 0) + int(value)


def get_application_service() -> ApplicationService:
    return ApplicationService()
