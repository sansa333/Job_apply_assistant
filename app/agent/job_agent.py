from __future__ import annotations

import json
from typing import Any

from app.agent.context_manager import ContextManager, RollingSummaryManager, estimate_tokens
from app.agent.conversation_store import ConversationStore
from app.agent.intent_router import (
    INTENT_CONFIDENCE_THRESHOLD, IntentRecognition, recognize_intent, has_new_job_reference, rule_candidates,
)
from app.agent.routing_policy import admission_response, allowed_tool_names, scope_tools
from app.agent.skill_runtime import (
    SkillSession,
    auto_activate_for_request,
    build_skill_runtime_tools,
    create_skill_session,
)
from app.agent.tools import build_job_tools
from app.agent.tool_results import ToolResultManager, wrap_tools_for_context
from app.config import settings
from app.llm import get_llm, message_to_text
from app.schemas import AgentRequest, ApplicationIntent
from app.services.application_service import ApplicationService


SYSTEM_PROMPT = """
你是求职投递 Agent，目标是帮助候选人完成岗位分析、材料生成和面试准备。

你必须按以下运行规则行动：
1. 先查看“可用按需 Skill”；涉及专门工作流时，先调用 activate_skill，再读取需要的 reference。
2. 只在已激活的 Skill 所允许的范围内调用领域工具。工具返回的 status、stage、next_action 等字段是流程事实来源，不能由自然语言猜测或覆盖。
3. candidate_id 和联系方式是可选输入。缺失时不得编造、不得填默认值；根据工具状态继续生成通用材料或带“待确认”的草稿。
4. 对特定岗位必须精确解析 JD；job_not_found 时引导用户提供 JD，不能用相似岗位替代。
5. 不得声称已向招聘网站、邮箱或 ATS 完成真实投递；申请包仅代表本地草稿已生成。
6. 输出中保留不确定性和下一步，不要把工具错误改写为已完成。
7. 工具返回的 tool_result_ref 指向服务端完整结果；当前上下文只包含预算内摘要，不得假装摘要中省略的细节已经核验。
"""

_DETERMINISTIC_INTENTS = {
    ApplicationIntent.FIT_ANALYSIS,
    ApplicationIntent.APPLICATION_PACKAGE,
}


def _system_prompt(session: SkillSession) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"## 可用 Skill 目录\n{session.catalog_context()}\n\n"
        f"## 当前已激活 Skill\n{session.active_context()}"
    )


def _decode_tool_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return {"status": "tool_output_unstructured", "result": str(value)}
    return decoded if isinstance(decoded, dict) else {"status": "tool_output_unstructured", "result": str(value)}


def _agent_response(payload: dict[str, Any], session: SkillSession) -> dict[str, Any]:
    """Map a real tool return value to the API's stable response contract."""
    result = (
        payload.get("fit_report")
        or payload.get("application_email")
        or payload.get("message")
        or payload.get("result")
        or payload.get("summary")
        or "请求已处理；请根据 status、stage 和 next_action 继续。"
    )
    artifacts = {
        key: str(value)
        for key, value in {
            "application_id": payload.get("application_id"),
            "output_dir": payload.get("output_dir"),
            "tool_result_ref": payload.get("tool_result_ref"),
        }.items()
        if value
    }
    return {
        "job_id": payload.get("job_id"),
        "result": str(result),
        "status": payload.get("status", "ok"),
        "stage": payload.get("stage", "completed"),
        "active_skills": session.active_names,
        "evidence_level": payload.get("evidence_level", "missing"),
        "contact_status": payload.get("contact_status", "pending_confirmation"),
        "missing_fields": payload.get("missing_fields", []),
        "next_action": payload.get("next_action"),
        "artifacts": artifacts,
        "match_score": payload.get("match_score"),
        "score_breakdown": payload.get("score_breakdown", {}),
        "evidence_matrix": payload.get("evidence_matrix", []),
        "validation_findings": payload.get("validation_findings", []),
        "workflow_trace": payload.get("workflow_trace", {}),
    }


def _apply_recognized_intent(req: AgentRequest, recognition: IntentRecognition) -> AgentRequest:
    """Promote only high-confidence classification and entity extraction into routing state."""
    if recognition.confidence < INTENT_CONFIDENCE_THRESHOLD or recognition.intent is None:
        return req
    return req.model_copy(
        update={
            "intent": recognition.intent,
            "company_name": req.company_name or recognition.company_name,
            "job_title": req.job_title or recognition.job_title,
        }
    )


def _with_intent_metadata(response: dict[str, Any], recognition: IntentRecognition) -> dict[str, Any]:
    response["recognized_intent"] = recognition.intent.value if recognition.intent else None
    response["intent_confidence"] = recognition.confidence
    response["intent_source"] = recognition.source
    response["intent_missing_fields"] = recognition.missing_fields
    response["intent_summary"] = recognition.summary
    response["extracted_entities"] = {
        key: value
        for key, value in {
            "company_name": recognition.company_name,
            "job_title": recognition.job_title,
        }.items()
        if value
    }
    return response


def _latest_structured_tool_result(messages: list[Any]) -> dict[str, Any] | None:
    """Prefer a returned tool state over an Agent's prose summary when present."""
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if isinstance(content, list):
            content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        payload = _decode_tool_payload(content)
        if "status" in payload and ("stage" in payload or "next_action" in payload):
            return payload
    return None


def _run_declared_workflow(
    req: AgentRequest,
    tools: list[Any],
    session: SkillSession,
    result_manager: ToolResultManager | None = None,
) -> dict[str, Any] | None:
    """Use real structured tools for UI-declared actions, avoiding an LLM routing hop."""
    if req.intent not in _DETERMINISTIC_INTENTS:
        return None

    by_name = {tool.name: tool for tool in tools}
    if req.intent == ApplicationIntent.FIT_ANALYSIS:
        tool_name = "analyze_job_fit"
        output = by_name[tool_name].invoke(
            {
                "candidate_id": req.candidate_id,
                "company_name": req.company_name,
                "job_title": req.job_title,
                "jd_text": req.jd_text,
                "resume_text": req.resume_text,
            }
        )
    else:
        tool_name = "generate_application_package"
        output = by_name[tool_name].invoke(
            {
                "candidate_id": req.candidate_id,
                "company_name": req.company_name,
                "job_title": req.job_title,
                "jd_text": req.jd_text,
                "resume_text": req.resume_text,
                "candidate_name": req.candidate_name,
                "candidate_email": str(req.candidate_email) if req.candidate_email else None,
                "candidate_phone": req.candidate_phone,
            }
        )
    payload = _decode_tool_payload(output)
    if result_manager is not None:
        _, _, _, reference_id = result_manager.archive(tool_name, output)
        if reference_id:
            payload["tool_result_ref"] = reference_id
    return _agent_response(payload, session)


def run_job_agent(
    req: AgentRequest,
    *,
    conversation_store: ConversationStore | None = None,
) -> dict[str, Any]:
    """Run a request-scoped, Skill-aware Agent and return structured workflow state."""
    store = conversation_store
    conversation: dict[str, Any] | None = None
    recent_turns: list[dict[str, Any]] = []
    rolling_summary: dict[str, Any] = {}
    original_req = req
    pending_state: dict[str, Any] = {}
    context_manager = ContextManager()
    summary_manager = RollingSummaryManager()
    if req.conversation_id:
        store = store or ConversationStore(settings.agent_conversation_db_path)
        req, conversation, recent_turns = store.prepare_request(
            req,
            recent_turns=settings.agent_recent_turns,
        )
        pending_state = store.get_routing_state(req.conversation_id, req.candidate_id)
        rolling_summary, _ = summary_manager.maybe_roll(
            store,
            req.conversation_id,
            req.candidate_id or "",
        )
        store.append_turn(
            req.conversation_id,
            req.candidate_id or "",
            role="user",
            content=req.goal,
            max_chars=settings.agent_max_turn_chars,
        )

    intent_turns, intent_summary = context_manager.intent_inputs(
        recent_turns,
        rolling_summary,
    )
    intent_context_tokens = estimate_tokens(json.dumps(intent_summary, ensure_ascii=False)) + sum(
        estimate_tokens(str(turn.get("content", ""))) + 4 for turn in intent_turns
    )
    # Never feed an inherited target as an explicit current-turn constraint when
    # the user announces a switch. Bound conversations require a new conversation.
    # Filling an empty identity field continues the pending task.
    target_change = has_new_job_reference(req.goal) or any(
        getattr(original_req, field) and conversation and conversation.get(field) and
        getattr(original_req, field) != conversation[field]
        for field in ("company_name", "job_title")
    )
    if target_change:
        req = req.model_copy(update={"company_name": original_req.company_name,
                                     "job_title": original_req.job_title})
    cancelled = req.goal.strip().rstrip("。！!") in {"取消", "取消任务", "先别生成了", "先不做了", "停止"}
    # A field-only reply resumes the confirmed pending intent. Arbitrary prose
    # still goes through classification; ordinal answers are handled by the router.
    if (pending_state.get("intent") and original_req.intent is None
            and not rule_candidates(req.goal) and not target_change
            and any(getattr(original_req, field) for field in
                    ("company_name", "job_title", "candidate_id", "resume_text", "jd_text"))
            and req.goal.strip() in {"", "继续", "已补充", "补充资料", "好了"}):
        req = req.model_copy(update={"intent": ApplicationIntent(pending_state["intent"])})
    recognition = IntentRecognition(source="cancelled") if cancelled else recognize_intent(
        req,
        recent_turns=intent_turns,
        last_intent=conversation.get("last_intent") if conversation else None,
        rolling_summary=intent_summary,
        pending_state=pending_state,
    )
    conflicting_fields = [
        field for field in ("company_name", "job_title")
        if recognition.source == "llm" and recognition.confidence >= INTENT_CONFIDENCE_THRESHOLD
        and getattr(original_req, field) and getattr(recognition, field)
        and getattr(original_req, field) != getattr(recognition, field)
    ]
    # Current-text entities override inherited values, but explicit request fields
    # win. The classifier validates its extractions against the current user text.
    if recognition.confidence >= INTENT_CONFIDENCE_THRESHOLD and recognition.source == "llm":
        req = req.model_copy(update={field: getattr(original_req, field) or getattr(recognition, field)
                                    for field in ("company_name", "job_title")})
    req = _apply_recognized_intent(req, recognition)
    field_sources = {
        field: ("request" if getattr(original_req, field) else
                "current_text" if recognition.source == "llm" and getattr(req, field) and
                getattr(req, field).casefold() in req.goal.casefold() else "conversation")
        for field in ("company_name", "job_title") if getattr(req, field)
    }
    context_usage: dict[str, Any] = {
        "budget_status": "not_applicable",
        "model_invoked": recognition.source == "llm" or recognition.model_attempts > 0,
        "intent_model_invoked": recognition.source == "llm" or recognition.model_attempts > 0,
        "agent_model_invoked": False,
        "intent_context_estimated_tokens": intent_context_tokens,
        "intent_context_budget_tokens": settings.agent_intent_context_tokens,
    }

    def finish(response: dict[str, Any]) -> dict[str, Any]:
        completed = _with_intent_metadata(response, recognition)
        completed["conversation_id"] = req.conversation_id
        completed["job_id"] = completed.get("job_id") or req.job_id
        completed["context_usage"] = dict(context_usage)
        completed["conversation_summary"] = dict(rolling_summary)
        completed["field_sources"] = field_sources
        completed["routing_diagnostics"] = {
            "model_attempts": recognition.model_attempts, "error_kind": recognition.error_kind,
            "latency_ms": round(recognition.latency_ms, 2),
        }
        if store is not None and req.conversation_id and req.candidate_id:
            waiting = completed.get("status") in {"needs_clarification", "needs_job_identity", "needs_candidate_evidence"}
            accepted = req.intent is not None and completed.get("status") not in {"needs_clarification", "needs_new_conversation", "cancelled"}
            intent_value = req.intent.value if accepted else None
            if completed.get("status") != "needs_new_conversation":
                store.update_state(req, completed, intent=intent_value)
            store.save_routing_state(req.conversation_id, req.candidate_id, {
                "intent": intent_value, "intent_options": completed.get("intent_options", []),
                "missing_fields": completed.get("missing_fields", []),
            } if waiting else {})
            store.append_turn(
                req.conversation_id,
                req.candidate_id,
                role="assistant",
                content=str(completed.get("result", "")),
                intent=intent_value,
                max_chars=settings.agent_max_turn_chars,
            )
            updated_summary, summary_updated = summary_manager.maybe_roll(
                store,
                req.conversation_id,
                req.candidate_id,
            )
            completed["conversation_summary"] = updated_summary
            completed["context_usage"]["rolling_summary_updated"] = summary_updated
        return completed

    if cancelled:
        return finish({"result": "已取消当前待处理任务。", "status": "cancelled", "stage": "cancelled",
                       "next_action": None})
    if conflicting_fields:
        question = "本轮文字中的目标与公司/岗位输入框不一致，请把输入框改为你确认的目标后继续。"
        return finish({"result": question, "clarification_question": question,
                       "status": "needs_job_identity", "stage": "awaiting_input",
                       "missing_fields": conflicting_fields, "next_action": "confirm_job_identity"})
    if conversation and conversation.get("job_id") and (target_change or any(
        getattr(req, field) and getattr(req, field) != conversation[field]
        for field in ("company_name", "job_title")
    )):
        return finish({"result": "本会话已绑定原岗位。请为新岗位建立会话，避免混用旧岗位证据。",
                       "status": "needs_new_conversation", "stage": "blocked",
                       "next_action": "create_conversation_for_new_job"})
    if response := admission_response(req, recognition):
        return finish(response)

    session = create_skill_session()
    auto_activate_for_request(session, req)
    service = ApplicationService()
    raw_tools = scope_tools([*build_skill_runtime_tools(session), *build_job_tools(service, session)], req)
    tool_result_manager = ToolResultManager(
        store=store,
        conversation_id=req.conversation_id,
        candidate_id=req.candidate_id,
    )

    deterministic = _run_declared_workflow(
        req,
        raw_tools,
        session,
        result_manager=tool_result_manager,
    )
    if deterministic is not None:
        return finish(deterministic)

    tools = wrap_tools_for_context(
        [tool for tool in raw_tools if tool.name in allowed_tool_names(req.intent)],
        tool_result_manager,
    )

    agent_context = context_manager.build(
        system_instructions=_system_prompt(session),
        request=req,
        recent_turns=recent_turns,
        rolling_summary=rolling_summary,
        conversation=conversation,
    )
    context_usage = {
        **agent_context.usage,
        "model_invoked": True,
        "intent_model_invoked": recognition.source == "llm" or recognition.model_attempts > 0,
        "agent_model_invoked": True,
        "intent_context_estimated_tokens": intent_context_tokens,
        "intent_context_budget_tokens": settings.agent_intent_context_tokens,
    }
    model = get_llm(temperature=0.1)
    prompt = agent_context.system_prompt

    agent_invoked = False
    try:
        from langchain.agents import create_agent

        agent = create_agent(model=model, tools=tools, system_prompt=prompt)
        agent_invoked = True
        # Bound the v1 graph so repeated tool calls cannot run indefinitely.
        result = agent.invoke({"messages": agent_context.messages}, config={"recursion_limit": 32})
        messages = result.get("messages", [])
        content = message_to_text(messages[-1].content) if messages else str(result)
        if payload := _latest_structured_tool_result(messages):
            response = _agent_response(payload, session)
            response["result"] = str(content)
            return finish(response)
        return finish({
            "result": str(content),
            "status": "completed",
            "stage": "responded",
            "active_skills": session.active_names,
            "evidence_level": "missing",
            "contact_status": "pending_confirmation",
            "missing_fields": [],
            "next_action": None,
            "artifacts": {},
            "match_score": None,
            "score_breakdown": {},
            "evidence_matrix": [],
            "validation_findings": [],
            "workflow_trace": {},
        })
    except Exception as agent_exc:
        if agent_invoked:
            # A tool may already have run. Never replay the whole request using
            # another agent after a runtime failure.
            return finish({
                "result": "Agent 执行中断，部分工具可能已执行。请先核对已有结果，再决定是否重试。",
                "status": "agent_execution_error", "stage": "blocked",
                "next_action": "review_existing_results_before_retry",
                "artifacts": {"error_kind": type(agent_exc).__name__},
            })
        return finish({
            "result": "Agent 工具运行初始化失败，未执行未受约束的求职操作。",
            "status": "agent_runtime_error",
            "stage": "agent_initialization",
            "active_skills": session.active_names,
            "evidence_level": "missing",
            "contact_status": "pending_confirmation",
            "missing_fields": [],
            "next_action": "retry_or_check_agent_runtime",
            "artifacts": {"error_kind": type(agent_exc).__name__},
            "match_score": None,
            "score_breakdown": {},
            "evidence_matrix": [],
            "validation_findings": [],
            "workflow_trace": {},
        })
