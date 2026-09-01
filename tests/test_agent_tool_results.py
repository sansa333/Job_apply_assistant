import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from app.agent.conversation_store import ConversationNotFoundError, ConversationStore
from app.agent.tool_results import ToolResultManager, wrap_tools_for_context
from app.config import settings
from app.schemas import ConversationCreateRequest


class _ToolInput(BaseModel):
    query: str


class AgentToolResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = ConversationStore(Path(self.temp_dir.name) / "tools.sqlite3")
        self.conversation = self.store.create(
            ConversationCreateRequest(candidate_id="candidate-a")
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_large_tool_result_is_archived_and_model_receives_bounded_reference(self) -> None:
        raw_payload = {
            "status": "ok",
            "stage": "evidence_collected",
            "documents": [
                {"filename": "resume.md", "content": "完整证据内容" * 1000}
            ],
        }

        def retrieve(query: str) -> str:
            return json.dumps({**raw_payload, "query": query}, ensure_ascii=False)

        tool = StructuredTool.from_function(
            func=retrieve,
            name="retrieve_test",
            description="test",
            args_schema=_ToolInput,
        )
        manager = ToolResultManager(
            store=self.store,
            conversation_id=self.conversation["conversation_id"],
            candidate_id="candidate-a",
        )
        with patch.object(settings, "agent_tool_result_prompt_tokens", 120):
            output = wrap_tools_for_context([tool], manager)[0].invoke({"query": "RAG"})

        compact = json.loads(output)
        self.assertTrue(compact["tool_result_truncated"])
        self.assertEqual(compact["status"], "ok")
        self.assertTrue(compact["tool_result_ref"].startswith("toolres_"))
        self.assertNotIn("完整证据内容" * 50, output)

        stored = self.store.get_tool_result(compact["tool_result_ref"], "candidate-a")
        self.assertIn("完整证据内容" * 50, stored["content"])
        self.assertEqual(stored["tool_name"], "retrieve_test")
        listed = self.store.list_tool_results(
            self.conversation["conversation_id"], "candidate-a"
        )
        self.assertNotIn("content", listed[0])

        self.store.delete_tool_result(compact["tool_result_ref"], "candidate-a")
        with self.assertRaises(ConversationNotFoundError):
            self.store.get_tool_result(compact["tool_result_ref"], "candidate-a")

    def test_stateless_tool_result_is_still_bounded_without_false_reference(self) -> None:
        tool = StructuredTool.from_function(
            func=lambda query: "unstructured " + query * 1000,
            name="stateless_test",
            description="test",
            args_schema=_ToolInput,
        )
        with patch.object(settings, "agent_tool_result_prompt_tokens", 80):
            output = wrap_tools_for_context(
                [tool],
                ToolResultManager(store=None, conversation_id=None, candidate_id=None),
            )[0].invoke({"query": "long"})

        compact = json.loads(output)
        self.assertTrue(compact["tool_result_truncated"])
        self.assertIsNone(compact["tool_result_ref"])


if __name__ == "__main__":
    unittest.main()
