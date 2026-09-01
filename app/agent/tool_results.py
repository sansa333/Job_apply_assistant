from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool

from app.agent.context_manager import estimate_tokens, truncate_to_tokens
from app.agent.conversation_store import ConversationStore
from app.config import settings


_STATE_KEYS = (
    "status",
    "stage",
    "next_action",
    "job_id",
    "application_id",
    "evidence_level",
    "contact_status",
    "missing_fields",
    "match_score",
)
_CONTENT_KEYS = (
    "message",
    "result",
    "fit_report",
    "application_email",
    "cover_letter",
    "interview_questions",
    "content",
    "instructions",
)
_DOCUMENT_KEYS = ("documents", "job_documents")


class ToolResultManager:
    """Archive bounded raw tool results and return a smaller model-facing view."""

    def __init__(
        self,
        *,
        store: ConversationStore | None,
        conversation_id: str | None,
        candidate_id: str | None,
    ) -> None:
        self.store = store
        self.conversation_id = conversation_id
        self.candidate_id = candidate_id

    @staticmethod
    def _decode(output: Any) -> tuple[str, dict[str, Any] | None]:
        if isinstance(output, dict):
            return json.dumps(output, ensure_ascii=False, default=str), output
        raw = str(output)
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return raw, None
        return raw, payload if isinstance(payload, dict) else None

    @staticmethod
    def _summary_text(payload: dict[str, Any] | None, raw: str, *, budget: int) -> str:
        fragments: list[str] = []
        if payload:
            for key in _CONTENT_KEYS:
                value = payload.get(key)
                if value:
                    fragments.append(f"{key}: {value}")
            for key in _DOCUMENT_KEYS:
                documents = payload.get(key)
                if not isinstance(documents, list):
                    continue
                for index, document in enumerate(documents[:3], start=1):
                    if isinstance(document, dict):
                        content = document.get("content") or document.get("page_content") or ""
                        source = document.get("filename") or document.get("source") or "unknown"
                        fragments.append(f"{key}[{index}] {source}: {content}")
        summary_source = "\n".join(fragments) or raw
        return truncate_to_tokens(summary_source, max(1, budget))[0]

    def handle(self, tool_name: str, output: Any) -> str:
        prompt_budget = max(64, settings.agent_tool_result_prompt_tokens)
        raw, payload, summary, reference_id = self.archive(
            tool_name,
            output,
            summary_budget=min(300, prompt_budget // 3),
        )

        if payload is not None and estimate_tokens(raw) <= prompt_budget:
            compact = dict(payload)
            compact["tool_result_ref"] = reference_id
            compact["tool_result_truncated"] = False
            return json.dumps(compact, ensure_ascii=False, default=str)

        compact: dict[str, Any] = {
            "tool_name": tool_name,
            "tool_result_ref": reference_id,
            "tool_result_truncated": True,
        }
        if payload:
            compact.update({key: payload[key] for key in _STATE_KEYS if key in payload})
        compact["summary"] = summary
        encoded = json.dumps(compact, ensure_ascii=False, default=str)
        if estimate_tokens(encoded) > prompt_budget:
            compact["summary"] = truncate_to_tokens(summary, max(16, prompt_budget // 2))[0]
        return json.dumps(compact, ensure_ascii=False, default=str)

    def archive(
        self,
        tool_name: str,
        output: Any,
        *,
        summary_budget: int = 300,
    ) -> tuple[str, dict[str, Any] | None, str, str | None]:
        raw, payload = self._decode(output)
        summary = self._summary_text(payload, raw, budget=summary_budget)
        reference_id: str | None = None
        if self.store and self.conversation_id and self.candidate_id:
            saved = self.store.save_tool_result(
                self.conversation_id,
                self.candidate_id,
                tool_name=tool_name,
                content=raw,
                summary=summary,
                status=str(payload.get("status")) if payload and payload.get("status") else None,
                stage=str(payload.get("stage")) if payload and payload.get("stage") else None,
                max_chars=settings.agent_tool_result_max_chars,
            )
            reference_id = saved["tool_result_id"]
        return raw, payload, summary, reference_id


def wrap_tools_for_context(
    tools: list[StructuredTool],
    manager: ToolResultManager,
) -> list[StructuredTool]:
    wrapped_tools: list[StructuredTool] = []
    for original in tools:
        def make_call(tool: StructuredTool):
            def call(**kwargs: Any) -> str:
                return manager.handle(tool.name, tool.invoke(kwargs))

            return call

        wrapped_tools.append(
            StructuredTool.from_function(
                func=make_call(original),
                name=original.name,
                description=original.description,
                args_schema=original.args_schema,
                return_direct=original.return_direct,
            )
        )
    return wrapped_tools
