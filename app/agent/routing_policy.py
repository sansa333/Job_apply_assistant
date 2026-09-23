"""Execution admission and tool scope, independent of LLM prose."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool

from app.agent.intent_router import INTENT_CONFIDENCE_THRESHOLD, IntentRecognition, rule_candidates
from app.agent.workflow import WorkflowPolicyInput, plan_workflow
from app.schemas import AgentRequest, ApplicationIntent

INTENT_LABELS = {
    ApplicationIntent.FIT_ANALYSIS: "分析岗位匹配度",
    ApplicationIntent.APPLICATION_PACKAGE: "生成完整申请材料",
    ApplicationIntent.COVER_LETTER: "撰写求职信",
    ApplicationIntent.APPLICATION_EMAIL: "撰写投递邮件",
    ApplicationIntent.INTERVIEW_PREP: "准备面试",
    ApplicationIntent.GENERAL_ADVICE: "获取一般求职建议",
}


def admission_response(req: AgentRequest, recognition: IntentRecognition) -> dict | None:
    candidates = rule_candidates(req.goal)
    ambiguous = (len(candidates) > 1 and recognition.source not in {"declared", "confirmed", "rule"})
    if req.intent is None or recognition.confidence < INTENT_CONFIDENCE_THRESHOLD or ambiguous:
        choices = candidates if len(candidates) > 1 else list(INTENT_LABELS)
        question = "你希望完成哪项任务？" + "；".join(
            f"{i}. {INTENT_LABELS[intent]}" for i, intent in enumerate(choices, 1)
        ) + "。请回复序号或任务名称。"
        return {
            "status": "needs_clarification", "stage": "awaiting_intent_confirmation",
            "next_action": "confirm_intent", "result": question,
            "clarification_question": question, "intent_options": [c.value for c in choices],
            "missing_fields": ["intent"],
        }

    # This is only admission. Real catalogue/ACL/evidence and output validation still
    # execute in ApplicationService; do not assert their success from classifier fields.
    plan = plan_workflow(WorkflowPolicyInput(
        intent=req.intent, has_company=bool(req.company_name.strip()),
        has_job_title=bool(req.job_title.strip()), has_candidate_id=bool(req.candidate_id),
        has_resume_text=bool(req.resume_text.strip()),
    ))
    if plan.terminal_status == "needs_job_identity":
        missing = [name for name in ("company_name", "job_title") if not getattr(req, name).strip()]
        question = "请补充" + "和".join({"company_name": "目标公司", "job_title": "目标岗位"}[f] for f in missing) + "。"
    elif plan.terminal_status == "needs_candidate_evidence":
        missing = ["candidate_id_or_resume_text"]
        question = "请提供候选人标识或简历正文，以便基于真实经历继续。"
    else:
        return None
    return {"status": plan.terminal_status, "stage": "awaiting_input", "result": question,
            "clarification_question": question, "missing_fields": missing,
            "intent_options": [], "next_action": plan.next_action}


def allowed_tool_names(intent: ApplicationIntent | None) -> set[str]:
    if intent is None:
        return set()
    names = {"activate_skill", "read_skill_reference", "retrieve_profile", "retrieve_job"}
    if intent != ApplicationIntent.GENERAL_ADVICE:
        names.add("analyze_job_fit")
    if intent == ApplicationIntent.APPLICATION_PACKAGE:
        names.add("generate_application_package")
    return names


def scope_tools(tools: list[StructuredTool], req: AgentRequest) -> list[StructuredTool]:
    """Guard even retained tool references; callers also filter the advertised list."""
    allowed = allowed_tool_names(req.intent)
    wrapped = []
    for tool in tools:
        def make_call(original: StructuredTool):
            def call(**kwargs: Any) -> Any:
                reason = None
                if original.name not in allowed:
                    reason = "tool_not_allowed_for_intent"
                for field in ("candidate_id", "company_name", "job_title"):
                    if field in kwargs and kwargs[field] and kwargs[field] != getattr(req, field):
                        reason = "tool_scope_mismatch"
                if reason:
                    return json.dumps({"status": reason, "stage": "blocked",
                                       "next_action": "confirm_intent_or_scope"}, ensure_ascii=False)
                # Request evidence/contacts must not be invented by the Agent.
                for field in ("candidate_id", "company_name", "job_title", "jd_text", "resume_text",
                              "candidate_name", "candidate_email", "candidate_phone"):
                    if field in original.args_schema.model_fields:
                        kwargs[field] = getattr(req, field)
                return original.invoke(kwargs)
            return call
        wrapped.append(StructuredTool.from_function(
            func=make_call(tool), name=tool.name, description=tool.description,
            args_schema=tool.args_schema, return_direct=tool.return_direct,
        ))
    return wrapped
