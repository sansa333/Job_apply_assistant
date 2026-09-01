import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.agent.conversation_store import ConversationScopeError, ConversationStore
from app.agent.job_agent import run_job_agent
from app.config import settings
from app.main import app
from app.schemas import AgentRequest, ConversationCreateRequest


class _FitService:
    def analyze_scoped_fit(self, req):
        return {
            "job_id": "job_acme_rag",
            "fit_report": "匹配分析完成",
            "status": "ok",
            "stage": "analyzed",
            "evidence_level": "user_provided",
        }


class AgentConversationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "conversations.sqlite3"
        self.store = ConversationStore(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create(self, candidate_id: str = "candidate-a") -> dict:
        return self.store.create(
            ConversationCreateRequest(
                candidate_id=candidate_id,
                company_name="Acme",
                job_title="RAG Engineer",
            )
        )

    def test_recent_turn_window_is_bounded_and_ordered(self) -> None:
        conversation = self._create()
        for index in range(8):
            self.store.append_turn(
                conversation["conversation_id"],
                "candidate-a",
                role="user" if index % 2 == 0 else "assistant",
                content=f"turn-{index}",
            )

        turns = self.store.recent_turns(
            conversation["conversation_id"], "candidate-a", limit=4
        )

        self.assertEqual([turn["content"] for turn in turns], ["turn-4", "turn-5", "turn-6", "turn-7"])

    def test_candidate_and_job_scope_cannot_be_rebound(self) -> None:
        conversation = self._create()
        with self.assertRaises(ConversationScopeError):
            self.store.get(conversation["conversation_id"], "candidate-b")

        self.store.update_state(
            AgentRequest(
                goal="分析",
                conversation_id=conversation["conversation_id"],
                candidate_id="candidate-a",
                job_id="job-1",
            ),
            {"stage": "analyzed", "job_id": "job-1"},
            intent="fit_analysis",
        )
        with self.assertRaises(ConversationScopeError):
            self.store.prepare_request(
                AgentRequest(
                    goal="继续",
                    conversation_id=conversation["conversation_id"],
                    candidate_id="candidate-a",
                    job_id="job-2",
                ),
                recent_turns=6,
            )

    def test_agent_persists_compact_turns_and_task_state(self) -> None:
        conversation = self._create()
        request = AgentRequest(
            goal="请分析岗位匹配度",
            conversation_id=conversation["conversation_id"],
            candidate_id="candidate-a",
            jd_text="sensitive full jd",
            resume_text="sensitive full resume",
        )
        with (
            patch("app.agent.job_agent.ApplicationService", return_value=_FitService()),
            patch.object(settings, "agent_summary_trigger_messages", 1),
            patch.object(settings, "agent_summary_keep_recent_messages", 1),
        ):
            result = run_job_agent(request, conversation_store=self.store)

        detail = self.store.detail(conversation["conversation_id"], "candidate-a")
        self.assertEqual(result["conversation_id"], conversation["conversation_id"])
        self.assertEqual(detail["job_id"], "job_acme_rag")
        self.assertEqual(detail["last_intent"], "fit_analysis")
        self.assertEqual(detail["current_stage"], "analyzed")
        self.assertEqual(len(detail["turns"]), 2)
        self.assertEqual(result["context_usage"]["budget_status"], "not_applicable")
        self.assertFalse(result["context_usage"]["model_invoked"])
        self.assertTrue(result["context_usage"]["rolling_summary_updated"])
        self.assertEqual(result["conversation_summary"]["summarized_message_count"], 1)
        self.assertEqual(detail["rolling_summary"]["version"], 1)
        self.assertTrue(result["artifacts"]["tool_result_ref"].startswith("toolres_"))
        self.assertEqual(len(detail["tool_results"]), 1)
        self.assertEqual(detail["tool_results"][0]["tool_name"], "analyze_job_fit")
        persisted_text = " ".join(turn["content"] for turn in detail["turns"])
        self.assertNotIn("sensitive full jd", persisted_text)
        self.assertNotIn("sensitive full resume", persisted_text)

    def test_api_hides_and_deletes_other_candidates_conversations(self) -> None:
        with patch.object(settings, "agent_conversation_db_path", self.db_path):
            with TestClient(app) as client:
                created = client.post(
                    "/api/conversations",
                    json={"candidate_id": "candidate-a"},
                )
                conversation_id = created.json()["conversation_id"]
                hidden = client.get(
                    f"/api/conversations/{conversation_id}",
                    params={"candidate_id": "candidate-b"},
                )
                deleted = client.delete(
                    f"/api/conversations/{conversation_id}",
                    params={"candidate_id": "candidate-a"},
                )
                missing = client.get(
                    f"/api/conversations/{conversation_id}",
                    params={"candidate_id": "candidate-a"},
                )

        self.assertEqual(created.status_code, 201)
        self.assertEqual(hidden.status_code, 404)
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
