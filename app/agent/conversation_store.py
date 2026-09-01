from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from app.schemas import AgentRequest, ConversationCreateRequest


class ConversationError(ValueError):
    """Base error for an invalid or inaccessible conversation."""


class ConversationNotFoundError(ConversationError):
    pass


class ConversationScopeError(ConversationError):
    pass


class ConversationStore:
    """SQLite-backed Agent working memory with candidate and job isolation."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_conversations (
                  conversation_id TEXT PRIMARY KEY,
                  candidate_id TEXT NOT NULL,
                  conversation_type TEXT NOT NULL,
                  job_id TEXT,
                  company_name TEXT NOT NULL DEFAULT '',
                  job_title TEXT NOT NULL DEFAULT '',
                  last_intent TEXT,
                  current_stage TEXT NOT NULL DEFAULT 'created',
                  status TEXT NOT NULL DEFAULT 'active',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_agent_conversations_candidate
                ON agent_conversations(candidate_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS agent_conversation_turns (
                  message_id TEXT PRIMARY KEY,
                  conversation_id TEXT NOT NULL,
                  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                  content TEXT NOT NULL,
                  intent TEXT,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(conversation_id) REFERENCES agent_conversations(conversation_id)
                    ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_agent_turns_conversation
                ON agent_conversation_turns(conversation_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS agent_conversation_summaries (
                  conversation_id TEXT PRIMARY KEY,
                  summary_json TEXT NOT NULL,
                  summarized_until_message_id TEXT,
                  summarized_message_count INTEGER NOT NULL DEFAULT 0,
                  version INTEGER NOT NULL DEFAULT 1,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(conversation_id) REFERENCES agent_conversations(conversation_id)
                    ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS agent_tool_results (
                  tool_result_id TEXT PRIMARY KEY,
                  conversation_id TEXT NOT NULL,
                  tool_name TEXT NOT NULL,
                  status TEXT,
                  stage TEXT,
                  summary TEXT NOT NULL DEFAULT '',
                  content TEXT NOT NULL,
                  content_chars INTEGER NOT NULL,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(conversation_id) REFERENCES agent_conversations(conversation_id)
                    ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_agent_tool_results_conversation
                ON agent_tool_results(conversation_id, created_at DESC);
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        return dict(row) if row is not None else None

    def create(self, request: ConversationCreateRequest) -> dict:
        now = self._now()
        conversation_id = f"conv_{uuid4().hex}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_conversations (
                  conversation_id, candidate_id, conversation_type, job_id,
                  company_name, job_title, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    request.candidate_id.strip(),
                    request.conversation_type.value,
                    request.job_id,
                    request.company_name.strip(),
                    request.job_title.strip(),
                    now,
                    now,
                ),
            )
        return self.get(conversation_id, request.candidate_id)

    def get(self, conversation_id: str, candidate_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            raise ConversationNotFoundError("Conversation not found")
        if row["candidate_id"] != candidate_id.strip():
            raise ConversationScopeError("Conversation does not belong to this candidate")
        return dict(row)

    def list(self, candidate_id: str, *, limit: int = 50) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM agent_conversations
                WHERE candidate_id = ? AND status != 'deleted'
                ORDER BY updated_at DESC LIMIT ?
                """,
                (candidate_id.strip(), max(1, min(limit, 100))),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, conversation_id: str, candidate_id: str) -> None:
        self.get(conversation_id, candidate_id)
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM agent_conversations WHERE conversation_id = ? AND candidate_id = ?",
                (conversation_id, candidate_id.strip()),
            )

    def recent_turns(self, conversation_id: str, candidate_id: str, *, limit: int) -> list[dict]:
        self.get(conversation_id, candidate_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT message_id, role, content, intent, created_at
                FROM agent_conversation_turns
                WHERE conversation_id = ?
                ORDER BY created_at DESC, rowid DESC LIMIT ?
                """,
                (conversation_id, max(1, limit)),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def summary_candidates(
        self,
        conversation_id: str,
        candidate_id: str,
        *,
        keep_recent: int,
        trigger_messages: int,
    ) -> list[dict]:
        """Return only unsummarized turns older than the protected recent window."""
        self.get(conversation_id, candidate_id)
        with self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM agent_conversation_turns WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()[0]
            if total <= trigger_messages or total <= keep_recent:
                return []
            cutoff = connection.execute(
                """
                SELECT rowid FROM agent_conversation_turns
                WHERE conversation_id = ?
                ORDER BY rowid DESC LIMIT 1 OFFSET ?
                """,
                (conversation_id, max(0, keep_recent - 1)),
            ).fetchone()
            summary_row = connection.execute(
                """
                SELECT summarized_until_message_id
                FROM agent_conversation_summaries WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            summarized_rowid = 0
            if summary_row and summary_row["summarized_until_message_id"]:
                row = connection.execute(
                    "SELECT rowid FROM agent_conversation_turns WHERE message_id = ?",
                    (summary_row["summarized_until_message_id"],),
                ).fetchone()
                summarized_rowid = row["rowid"] if row else 0
            rows = connection.execute(
                """
                SELECT message_id, role, content, intent, created_at
                FROM agent_conversation_turns
                WHERE conversation_id = ? AND rowid > ? AND rowid < ?
                ORDER BY rowid ASC
                """,
                (conversation_id, summarized_rowid, cutoff["rowid"]),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_summary(self, conversation_id: str, candidate_id: str) -> dict:
        self.get(conversation_id, candidate_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_conversation_summaries WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return {}
        try:
            summary = json.loads(row["summary_json"])
        except (TypeError, json.JSONDecodeError):
            summary = {}
        if not isinstance(summary, dict):
            summary = {}
        summary["version"] = row["version"]
        summary["summarized_message_count"] = row["summarized_message_count"]
        summary["summarized_until_message_id"] = row["summarized_until_message_id"]
        summary["updated_at"] = row["updated_at"]
        return summary

    def save_summary(
        self,
        conversation_id: str,
        candidate_id: str,
        *,
        summary: dict,
        summarized_until_message_id: str,
        added_message_count: int,
    ) -> dict:
        self.get(conversation_id, candidate_id)
        now = self._now()
        current = self.get_summary(conversation_id, candidate_id)
        version = int(current.get("version", 0)) + 1
        total_count = int(current.get("summarized_message_count", 0)) + added_message_count
        payload = {
            key: value
            for key, value in summary.items()
            if key not in {"version", "summarized_message_count", "summarized_until_message_id", "updated_at"}
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_conversation_summaries (
                  conversation_id, summary_json, summarized_until_message_id,
                  summarized_message_count, version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                  summary_json = excluded.summary_json,
                  summarized_until_message_id = excluded.summarized_until_message_id,
                  summarized_message_count = excluded.summarized_message_count,
                  version = excluded.version,
                  updated_at = excluded.updated_at
                """,
                (
                    conversation_id,
                    json.dumps(payload, ensure_ascii=False),
                    summarized_until_message_id,
                    total_count,
                    version,
                    now,
                    now,
                ),
            )
        return self.get_summary(conversation_id, candidate_id)

    def save_tool_result(
        self,
        conversation_id: str,
        candidate_id: str,
        *,
        tool_name: str,
        content: str,
        summary: str = "",
        status: str | None = None,
        stage: str | None = None,
        max_chars: int = 100000,
    ) -> dict:
        self.get(conversation_id, candidate_id)
        tool_result_id = f"toolres_{uuid4().hex}"
        stored_content = content[: max(1, max_chars)]
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_tool_results (
                  tool_result_id, conversation_id, tool_name, status, stage,
                  summary, content, content_chars, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tool_result_id,
                    conversation_id,
                    tool_name,
                    status,
                    stage,
                    summary[:1000],
                    stored_content,
                    len(content),
                    now,
                ),
            )
        return self.get_tool_result(tool_result_id, candidate_id)

    def list_tool_results(
        self,
        conversation_id: str,
        candidate_id: str,
        *,
        limit: int = 50,
    ) -> list[dict]:
        self.get(conversation_id, candidate_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT tool_result_id, conversation_id, tool_name, status, stage,
                       summary, content_chars, created_at
                FROM agent_tool_results WHERE conversation_id = ?
                ORDER BY rowid DESC LIMIT ?
                """,
                (conversation_id, max(1, min(limit, 100))),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_tool_result(self, tool_result_id: str, candidate_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT result.* FROM agent_tool_results AS result
                JOIN agent_conversations AS conversation
                  ON conversation.conversation_id = result.conversation_id
                WHERE result.tool_result_id = ? AND conversation.candidate_id = ?
                """,
                (tool_result_id, candidate_id.strip()),
            ).fetchone()
        if row is None:
            raise ConversationNotFoundError("Tool result not found")
        return dict(row)

    def delete_tool_result(self, tool_result_id: str, candidate_id: str) -> None:
        self.get_tool_result(tool_result_id, candidate_id)
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM agent_tool_results
                WHERE tool_result_id = ? AND conversation_id IN (
                  SELECT conversation_id FROM agent_conversations WHERE candidate_id = ?
                )
                """,
                (tool_result_id, candidate_id.strip()),
            )

    def append_turn(
        self,
        conversation_id: str,
        candidate_id: str,
        *,
        role: str,
        content: str,
        intent: str | None = None,
        max_chars: int = 4000,
    ) -> dict:
        self.get(conversation_id, candidate_id)
        now = self._now()
        message_id = f"msg_{uuid4().hex}"
        normalized = content.strip()[:max_chars]
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_conversation_turns (
                  message_id, conversation_id, role, content, intent, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (message_id, conversation_id, role, normalized, intent, now),
            )
            connection.execute(
                "UPDATE agent_conversations SET updated_at = ? WHERE conversation_id = ?",
                (now, conversation_id),
            )
        return {
            "message_id": message_id,
            "role": role,
            "content": normalized,
            "intent": intent,
            "created_at": now,
        }

    def prepare_request(self, request: AgentRequest, *, recent_turns: int) -> tuple[AgentRequest, dict, list[dict]]:
        if not request.conversation_id:
            raise ConversationError("conversation_id is required")
        if not request.candidate_id:
            raise ConversationScopeError("candidate_id is required for conversation requests")
        conversation = self.get(request.conversation_id, request.candidate_id)
        if conversation["job_id"] and request.job_id and conversation["job_id"] != request.job_id:
            raise ConversationScopeError("Conversation is already bound to another job")
        effective = request.model_copy(
            update={
                "candidate_id": request.candidate_id,
                "job_id": request.job_id or conversation["job_id"],
                "company_name": request.company_name or conversation["company_name"],
                "job_title": request.job_title or conversation["job_title"],
            }
        )
        turns = self.recent_turns(
            request.conversation_id,
            request.candidate_id,
            limit=recent_turns,
        )
        return effective, conversation, turns

    def update_state(
        self,
        request: AgentRequest,
        response: dict,
        *,
        intent: str | None,
    ) -> None:
        if not request.conversation_id or not request.candidate_id:
            return
        conversation = self.get(request.conversation_id, request.candidate_id)
        resolved_job_id = response.get("job_id") or request.job_id or conversation["job_id"]
        if conversation["job_id"] and resolved_job_id and conversation["job_id"] != resolved_job_id:
            raise ConversationScopeError("Conversation cannot be rebound to another job")
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE agent_conversations SET
                  job_id = ?, company_name = ?, job_title = ?, last_intent = ?,
                  current_stage = ?, updated_at = ?
                WHERE conversation_id = ? AND candidate_id = ?
                """,
                (
                    resolved_job_id,
                    request.company_name or conversation["company_name"],
                    request.job_title or conversation["job_title"],
                    intent or conversation["last_intent"],
                    response.get("stage") or conversation["current_stage"],
                    now,
                    request.conversation_id,
                    request.candidate_id,
                ),
            )

    def detail(self, conversation_id: str, candidate_id: str, *, turn_limit: int = 50) -> dict:
        conversation = self.get(conversation_id, candidate_id)
        conversation["turns"] = self.recent_turns(
            conversation_id,
            candidate_id,
            limit=turn_limit,
        )
        conversation["rolling_summary"] = self.get_summary(conversation_id, candidate_id)
        conversation["tool_results"] = self.list_tool_results(
            conversation_id,
            candidate_id,
            limit=50,
        )
        return conversation
