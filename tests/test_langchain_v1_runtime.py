"""Exercise the installed v1 agent runtime without a network model."""
import json
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from app.agent.job_agent import run_job_agent
from app.llm import get_llm
from app.schemas import AgentRequest


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_real_agent_executes_scoped_tool_and_preserves_structured_result():
    calls = []

    def retrieve_profile(candidate_id: str) -> str:
        """Read the candidate profile."""
        calls.append(candidate_id)
        return json.dumps({"status": "ok", "stage": "retrieved", "evidence_level": "verified"})

    model = ToolCallingModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "retrieve_profile", "args": {"candidate_id": "alice"}, "id": "call-1",
        }]),
        AIMessage(content=[{"type": "text", "text": "已读取候选人资料。"}]),
    ])
    with (patch("app.agent.job_agent.get_llm", return_value=model),
          patch("app.agent.job_agent.ApplicationService"),
          patch("app.agent.job_agent.build_job_tools", return_value=[
              StructuredTool.from_function(retrieve_profile),
          ])):
        result = run_job_agent(AgentRequest(
            goal="提供职业建议", intent="general_advice", candidate_id="alice",
        ))

    assert calls == ["alice"]
    assert result["status"] == "ok"
    assert result["stage"] == "retrieved"
    assert result["evidence_level"] == "verified"
    assert result["result"] == "已读取候选人资料。"


def test_real_agent_rejects_cross_candidate_tool_call():
    effect = Mock()

    def retrieve_profile(candidate_id: str) -> str:
        """Read the candidate profile."""
        effect(candidate_id)
        return "{}"

    model = ToolCallingModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "retrieve_profile", "args": {"candidate_id": "bob"}, "id": "call-2",
        }]),
        AIMessage(content="无法读取其他候选人的资料。"),
    ])
    with (patch("app.agent.job_agent.get_llm", return_value=model),
          patch("app.agent.job_agent.ApplicationService"),
          patch("app.agent.job_agent.build_job_tools", return_value=[
              StructuredTool.from_function(retrieve_profile),
          ])):
        result = run_job_agent(AgentRequest(
            goal="提供职业建议", intent="general_advice", candidate_id="alice",
        ))
    effect.assert_not_called()
    assert result["status"] == "tool_scope_mismatch"


def test_agent_initialization_failure_is_reported_without_retry():
    with (patch("app.agent.job_agent.get_llm", return_value=object()),
          patch("app.agent.job_agent.ApplicationService"),
          patch("langchain.agents.create_agent", side_effect=ValueError("invalid model")) as create):
        result = run_job_agent(AgentRequest(goal="职业建议", intent="general_advice"))
    create.assert_called_once()
    assert result["status"] == "agent_runtime_error"
    assert result["artifacts"] == {"error_kind": "ValueError"}


def test_repeated_tool_calls_stop_at_graph_limit():
    calls = []

    def retrieve_profile(candidate_id: str) -> str:
        """Read the candidate profile."""
        calls.append(candidate_id)
        return "{}"

    model = ToolCallingModel(responses=[AIMessage(content="", tool_calls=[{
        "name": "retrieve_profile", "args": {"candidate_id": "alice"}, "id": f"repeat-{i}",
    }]) for i in range(40)])
    with (patch("app.agent.job_agent.get_llm", return_value=model),
          patch("app.agent.job_agent.ApplicationService"),
          patch("app.agent.job_agent.build_job_tools", return_value=[
              StructuredTool.from_function(retrieve_profile),
          ])):
        result = run_job_agent(AgentRequest(
            goal="职业建议", intent="general_advice", candidate_id="alice",
        ))
    assert 0 < len(calls) <= 32
    assert result["status"] == "agent_execution_error"
    assert result["artifacts"]["error_kind"] == "GraphRecursionError"
    assert result["next_action"] == "review_existing_results_before_retry"


def test_glm_chat_completions_tool_round_trip_without_network():
    model = get_llm()
    assert model.use_responses_api is False
    response = {
        "id": "offline", "model": "glm-test", "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "完成"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }
    with patch.object(model.client.with_raw_response, "create") as create:
        create.return_value.parse.return_value = response
        result = model.invoke([
            AIMessage(content="", tool_calls=[{"name": "lookup", "args": {}, "id": "c1"}]),
            ToolMessage(content="找到资料", tool_call_id="c1"),
        ])
    assert result.content == "完成"
    assert result.usage_metadata["total_tokens"] == 12
    assert create.call_args.kwargs["messages"][-1]["tool_call_id"] == "c1"
