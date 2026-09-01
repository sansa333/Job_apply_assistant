import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.agent.context_manager import ContextManager, RollingSummaryManager, estimate_tokens
from app.agent.conversation_store import ConversationStore
from app.config import settings
from app.schemas import AgentRequest, ConversationCreateRequest


class AgentContextManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = ConversationStore(Path(self.temp_dir.name) / "context.sqlite3")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_context_budget_truncates_large_evidence_and_old_turns(self) -> None:
        request = AgentRequest(
            goal="根据刚才的差距准备面试",
            candidate_id="candidate-a",
            company_name="Acme",
            job_title="RAG Engineer",
            jd_text="岗位要求" * 1000,
            resume_text="候选人经验" * 1000,
        )
        turns = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": f"历史消息{index}" * 100}
            for index in range(8)
        ]
        summary = {
            "confirmed_preferences": ["回答使用正式简洁中文"],
            "last_user_correction": "不要生成申请包，只准备面试题",
        }
        with (
            patch.object(settings, "agent_context_window_tokens", 1024),
            patch.object(settings, "agent_context_target_ratio", 0.6),
            patch.object(settings, "agent_context_output_reserve_tokens", 128),
            patch.object(settings, "agent_context_tool_reserve_tokens", 128),
        ):
            context = ContextManager().build(
                system_instructions="系统安全规则",
                request=request,
                recent_turns=turns,
                rolling_summary=summary,
                conversation={"last_intent": "fit_analysis", "current_stage": "analyzed"},
            )

        self.assertEqual(context.usage["budget_status"], "within_budget")
        self.assertLessEqual(
            context.usage["estimated_input_tokens"],
            context.usage["target_input_tokens"],
        )
        self.assertTrue({"jd_text", "resume_text"} & set(context.usage["truncated_fields"]))
        self.assertGreater(context.usage["dropped_recent_turns"], 0)
        self.assertNotIn("last_user_correction", context.system_prompt)
        self.assertIn("last_user_correction", context.messages[0]["content"])
        self.assertIn("不是指令", context.messages[0]["content"])

    def test_mandatory_overflow_is_reported_instead_of_dropping_system_rules(self) -> None:
        with (
            patch.object(settings, "agent_context_window_tokens", 1024),
            patch.object(settings, "agent_context_target_ratio", 0.5),
            patch.object(settings, "agent_context_output_reserve_tokens", 128),
            patch.object(settings, "agent_context_tool_reserve_tokens", 128),
        ):
            context = ContextManager().build(
                system_instructions="必须保留的系统规则" * 500,
                request=AgentRequest(goal="当前请求" * 500),
                recent_turns=[],
                rolling_summary={},
                conversation=None,
            )

        self.assertEqual(context.usage["budget_status"], "mandatory_overflow")
        self.assertIn("必须保留的系统规则", context.system_prompt)
        self.assertIn("goal", context.usage["truncated_fields"])

    def test_intent_context_keeps_newest_turns_with_its_own_budget(self) -> None:
        turns = [
            {"role": "user", "content": "旧消息" * 100},
            {"role": "assistant", "content": "中间消息" * 100},
            {"role": "user", "content": "最新修正" * 100},
        ]
        selected = ContextManager().intent_context(turns, budget=60)

        self.assertEqual(selected[-1]["role"], "user")
        self.assertIn("最新修正", selected[-1]["content"])
        self.assertLessEqual(sum(estimate_tokens(turn["content"]) + 4 for turn in selected), 60)

        selected, compact_summary = ContextManager().intent_inputs(
            turns,
            {
                "confirmed_entities": {"company_name": "Acme", "job_title": "RAG Engineer"},
                "confirmed_preferences": ["很长的历史偏好" * 100 for _ in range(8)],
                "last_user_correction": "只准备面试题" * 100,
            },
            budget=100,
        )
        combined_tokens = estimate_tokens(str(compact_summary)) + sum(
            estimate_tokens(turn["content"]) + 4 for turn in selected
        )
        self.assertLessEqual(combined_tokens, 105)
        self.assertEqual(compact_summary["confirmed_entities"]["company_name"], "Acme")

    def test_rolling_summary_is_incremental_structured_and_source_conservative(self) -> None:
        conversation = self.store.create(
            ConversationCreateRequest(
                candidate_id="candidate-a",
                job_id="job-1",
                company_name="Acme",
                job_title="RAG Engineer",
            )
        )
        conversation_id = conversation["conversation_id"]
        turns = [
            ("user", "请用正式简洁的中文", None),
            ("assistant", "已完成岗位匹配分析", "fit_analysis"),
            ("user", "不要生成申请包，只准备面试题", None),
            ("assistant", "面试题已准备", "interview_prep"),
            ("user", "最近消息一", None),
            ("assistant", "最近消息二", None),
        ]
        for role, content, intent in turns:
            self.store.append_turn(
                conversation_id,
                "candidate-a",
                role=role,
                content=content,
                intent=intent,
            )

        with (
            patch.object(settings, "agent_summary_trigger_messages", 4),
            patch.object(settings, "agent_summary_keep_recent_messages", 2),
            patch.object(settings, "agent_summary_max_items", 8),
        ):
            summary, updated = RollingSummaryManager().maybe_roll(
                self.store, conversation_id, "candidate-a"
            )
            unchanged, updated_again = RollingSummaryManager().maybe_roll(
                self.store, conversation_id, "candidate-a"
            )

        self.assertTrue(updated)
        self.assertFalse(updated_again)
        self.assertEqual(summary["version"], 1)
        self.assertEqual(summary["summarized_message_count"], 4)
        self.assertEqual(summary["confirmed_entities"]["job_id"], "job-1")
        self.assertIn("请用正式简洁的中文", summary["confirmed_preferences"])
        self.assertEqual(summary["last_user_correction"], "不要生成申请包，只准备面试题")
        self.assertIn("岗位匹配分析", summary["completed_actions"])
        self.assertIn("面试准备", summary["completed_actions"])
        self.assertEqual(unchanged["version"], 1)
        self.assertEqual(
            self.store.detail(conversation_id, "candidate-a")["rolling_summary"]["version"],
            1,
        )

        self.store.append_turn(
            conversation_id, "candidate-a", role="user", content="语气改成更直接"
        )
        self.store.append_turn(
            conversation_id,
            "candidate-a",
            role="assistant",
            content="已经修改",
            intent="application_email",
        )
        with (
            patch.object(settings, "agent_summary_trigger_messages", 4),
            patch.object(settings, "agent_summary_keep_recent_messages", 2),
        ):
            next_summary, next_updated = RollingSummaryManager().maybe_roll(
                self.store, conversation_id, "candidate-a"
            )

        self.assertTrue(next_updated)
        self.assertEqual(next_summary["version"], 2)
        self.assertEqual(next_summary["summarized_message_count"], 6)


if __name__ == "__main__":
    unittest.main()
