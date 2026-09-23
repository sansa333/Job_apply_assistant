import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from app.agent.conversation_store import ConversationStore
from app.agent.intent_router import IntentRecognition, recognize_intent
from app.agent.job_agent import run_job_agent
from app.agent.routing_policy import scope_tools
from app.main import app
from app.schemas import AgentRequest, ApplicationIntent, ConversationCreateRequest


class RoutingAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ConversationStore(Path(self.tmp.name) / "conversation.db")
        self.conv = self.store.create(ConversationCreateRequest(
            candidate_id="alice", company_name="Acme", job_title="Engineer"))

    def tearDown(self):
        self.tmp.cleanup()

    def request(self, goal, **kwargs):
        return AgentRequest(goal=goal, conversation_id=self.conv["conversation_id"],
                            candidate_id="alice", **kwargs)

    def test_uncertain_route_never_constructs_service_or_agent(self):
        for recognition in (IntentRecognition(), IntentRecognition(
                intent=ApplicationIntent.APPLICATION_PACKAGE, confidence=0.4)):
            with (patch("app.agent.job_agent.recognize_intent", return_value=recognition),
                  patch("app.agent.job_agent.ApplicationService") as service,
                  patch("app.agent.job_agent.get_llm") as model):
                result = run_job_agent(self.request("处理一下"), conversation_store=self.store)
            self.assertEqual(result["status"], "needs_clarification")
            service.assert_not_called()
            model.assert_not_called()
        self.assertIsNone(self.store.get(self.conv["conversation_id"], "alice")["last_intent"])

    def test_ordinal_reply_resumes_after_store_reopen_and_does_not_generate_package(self):
        with patch("app.agent.job_agent.recognize_intent", return_value=IntentRecognition()):
            first = run_job_agent(self.request("处理一下"), conversation_store=self.store)
        self.assertEqual(first["intent_options"][0], "fit_analysis")
        service = Mock()
        service.analyze_scoped_fit.return_value = {"status": "ok", "stage": "analyzed", "fit_report": "完成"}
        reopened = ConversationStore(self.store.path)
        with (patch("app.agent.job_agent.ApplicationService", return_value=service),
              patch("app.agent.intent_router.get_llm") as classifier):
            second = run_job_agent(self.request("第一个"), conversation_store=reopened)
        self.assertEqual(second["status"], "ok")
        self.assertEqual(second["intent_source"], "confirmed")
        service.one_click_apply.assert_not_called()
        classifier.assert_not_called()
        self.assertEqual(reopened.get_routing_state(self.conv["conversation_id"], "alice"), {})

    def test_confirmed_task_missing_fields_blocks_before_service(self):
        with patch("app.agent.job_agent.ApplicationService") as service:
            missing_job = run_job_agent(AgentRequest(goal="匹配度", intent="fit_analysis"))
            missing_resume = run_job_agent(AgentRequest(goal="匹配度", intent="fit_analysis",
                                                       company_name="Acme", job_title="Engineer"))
        service.assert_not_called()
        self.assertEqual(missing_job["status"], "needs_job_identity")
        self.assertEqual(missing_resume["status"], "needs_candidate_evidence")

    def test_missing_field_reply_resumes_confirmed_task(self):
        conv = self.store.create(ConversationCreateRequest(candidate_id="bob"))
        first = AgentRequest(goal="分析匹配度", intent="fit_analysis", candidate_id="bob",
                             conversation_id=conv["conversation_id"])
        self.assertEqual(run_job_agent(first, conversation_store=self.store)["status"], "needs_job_identity")
        service = Mock()
        service.analyze_scoped_fit.return_value = {"status": "ok", "stage": "analyzed"}
        with patch("app.agent.job_agent.ApplicationService", return_value=service):
            result = run_job_agent(AgentRequest(goal="已补充", candidate_id="bob",
                conversation_id=conv["conversation_id"], company_name="Acme", job_title="Engineer"),
                conversation_store=self.store)
        self.assertEqual(result["recognized_intent"], "fit_analysis")
        service.analyze_scoped_fit.assert_called_once()

    def test_cancel_clears_pending_and_previous_intent(self):
        self.store.update_state(self.request("分析"), {}, intent="fit_analysis")
        self.store.save_routing_state(self.conv["conversation_id"], "alice", {"intent_options": ["application_package"]})
        with patch("app.agent.job_agent.ApplicationService") as service:
            result = run_job_agent(self.request("先别生成了"), conversation_store=self.store)
        self.assertEqual(result["status"], "cancelled")
        service.assert_not_called()
        self.assertEqual(self.store.get_routing_state(self.conv["conversation_id"], "alice"), {})
        self.assertIsNone(self.store.get(self.conv["conversation_id"], "alice")["last_intent"])

    def test_low_confidence_never_overwrites_confirmed_last_intent(self):
        self.store.update_state(self.request("分析"), {}, intent="fit_analysis")
        with patch("app.agent.job_agent.recognize_intent", return_value=IntentRecognition(
                intent=ApplicationIntent.APPLICATION_PACKAGE, confidence=0.4)):
            run_job_agent(self.request("处理"), conversation_store=self.store)
        self.assertEqual(self.store.get(self.conv["conversation_id"], "alice")["last_intent"], "fit_analysis")

    def test_new_target_in_text_overrides_inherited_unbound_target(self):
        model = Mock()
        model.invoke.return_value = SimpleNamespace(content=json.dumps({
            "intent": "fit_analysis", "confidence": 0.95, "company_name": "腾讯", "job_title": "后端工程师"}))
        service = Mock()
        service.analyze_scoped_fit.return_value = {"status": "ok", "stage": "analyzed"}
        with (patch("app.agent.intent_router.get_llm", return_value=model),
              patch("app.agent.job_agent.ApplicationService", return_value=service)):
            result = run_job_agent(self.request("分析腾讯后端工程师匹配度"), conversation_store=self.store)
        received = service.analyze_scoped_fit.call_args.args[0]
        self.assertEqual(received.company_name, "腾讯")
        self.assertEqual(received.job_title, "后端工程师")
        self.assertEqual(result["field_sources"]["company_name"], "current_text")

    def test_bound_job_switch_never_runs_or_rebinds(self):
        self.store.update_state(self.request("分析", job_id="job-acme"), {}, intent="fit_analysis")
        with (patch("app.agent.intent_router.get_llm", side_effect=RuntimeError("offline")),
              patch("app.agent.job_agent.ApplicationService") as service):
            result = run_job_agent(self.request("换一家公司的岗位"), conversation_store=self.store)
        self.assertEqual(result["status"], "needs_new_conversation")
        service.assert_not_called()
        self.assertEqual(self.store.get(self.conv["conversation_id"], "alice")["company_name"], "Acme")

    def test_current_text_and_explicit_form_conflict_requires_confirmation(self):
        recognition = IntentRecognition(intent=ApplicationIntent.FIT_ANALYSIS, confidence=0.95,
                                        source="llm", company_name="腾讯", job_title="Engineer")
        with (patch("app.agent.job_agent.recognize_intent", return_value=recognition),
              patch("app.agent.job_agent.ApplicationService") as service):
            result = run_job_agent(self.request("分析腾讯Engineer", company_name="Acme", job_title="Engineer"),
                                   conversation_store=self.store)
        self.assertEqual(result["next_action"], "confirm_job_identity")
        service.assert_not_called()

    def test_tool_scope_blocks_package_and_candidate_drift_before_side_effect(self):
        class Args(BaseModel):
            candidate_id: str
        effect = Mock(return_value="executed")
        tools = [StructuredTool.from_function(func=effect, name=name, description="test", args_schema=Args)
                 for name in ("generate_application_package", "retrieve_profile")]
        guarded = scope_tools(tools, AgentRequest(goal="分析", intent="fit_analysis", candidate_id="alice"))
        self.assertEqual(json.loads(guarded[0].invoke({"candidate_id": "alice"}))["status"], "tool_not_allowed_for_intent")
        self.assertEqual(json.loads(guarded[1].invoke({"candidate_id": "bob"}))["status"], "tool_scope_mismatch")
        effect.assert_not_called()

    def test_api_preserves_clarification_contract(self):
        with patch("app.agent.job_agent.recognize_intent", return_value=IntentRecognition()):
            with TestClient(app) as client:
                result = client.post("/api/agent", json={"goal": "处理一下"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "needs_clarification")
        self.assertTrue(result.json()["clarification_question"])
        self.assertTrue(result.json()["intent_options"])

    def test_ambiguous_alternatives_do_not_execute_even_with_high_model_confidence(self):
        with (patch("app.agent.job_agent.recognize_intent", return_value=IntentRecognition(
                intent=ApplicationIntent.APPLICATION_PACKAGE, confidence=0.99, source="llm")),
              patch("app.agent.job_agent.ApplicationService") as service):
            result = run_job_agent(self.request("生成申请包还是分析匹配度"), conversation_store=self.store)
        self.assertEqual(result["status"], "needs_clarification")
        service.assert_not_called()

    def test_agent_runtime_failure_is_not_replayed_through_legacy_agent(self):
        agent = Mock()
        agent.invoke.side_effect = RuntimeError("tool already ran before failure")
        with (patch("app.agent.job_agent.get_llm", return_value=object()),
              patch("app.agent.job_agent.ApplicationService"),
              patch("langchain.agents.create_agent", return_value=agent),
              patch("langchain.agents.initialize_agent", create=True) as legacy):
            result = run_job_agent(AgentRequest(goal="职业规划", intent="general_advice"))
        self.assertEqual(result["status"], "agent_execution_error")
        self.assertEqual(result["next_action"], "review_existing_results_before_retry")
        legacy.assert_not_called()

    def test_clarification_isolation_and_cascade_delete(self):
        from app.agent.conversation_store import ConversationScopeError, ConversationNotFoundError
        self.store.save_routing_state(self.conv["conversation_id"], "alice", {"intent": "fit_analysis"})
        with self.assertRaises(ConversationScopeError):
            self.store.get_routing_state(self.conv["conversation_id"], "bob")
        self.store.delete(self.conv["conversation_id"], "alice")
        with self.assertRaises(ConversationNotFoundError):
            self.store.get_routing_state(self.conv["conversation_id"], "alice")
        with self.store._connect() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM agent_routing_state").fetchone()[0], 0)


class ClassifierFailureTests(unittest.TestCase):
    def test_classifier_disables_sdk_retries_without_mutating_shared_model(self):
        from langchain_openai import ChatOpenAI
        from app.config import settings
        model = ChatOpenAI(api_key="offline-test", model="test", timeout=90, max_retries=1)
        observed = {}

        def invoke(instance, messages):
            observed.update(timeout=instance.request_timeout, retries=instance.max_retries,
                            max_tokens=instance.max_tokens)
            return SimpleNamespace(content='{"intent":"general_advice","confidence":0.9}')

        with (patch("app.agent.intent_router.get_llm", return_value=model),
              patch.object(ChatOpenAI, "invoke", invoke)):
            recognize_intent(AgentRequest(goal="如何找第一份工作"))
        self.assertEqual(observed["timeout"], settings.agent_intent_timeout_seconds)
        self.assertEqual(observed["retries"], 0)
        self.assertEqual(observed["max_tokens"], 400)
        self.assertEqual(model.max_retries, 1)
        self.assertEqual(model.request_timeout, 90)

    def test_transient_failure_retries_only_once(self):
        model = Mock()
        model.invoke.side_effect = TimeoutError()
        with (patch("app.agent.intent_router.get_llm", return_value=model),
              patch("app.agent.intent_router.time.sleep")):
            result = recognize_intent(AgentRequest(goal="分析匹配度"))
        self.assertEqual(model.invoke.call_count, 2)
        self.assertEqual(result.source, "rule_fallback")
        self.assertEqual(result.error_kind, "transient_error")

    def test_format_repair_shares_two_call_budget(self):
        model = Mock()
        model.invoke.side_effect = [SimpleNamespace(content="not json"), SimpleNamespace(content='{"intent":"general_advice","confidence":0.9}')]
        with patch("app.agent.intent_router.get_llm", return_value=model):
            result = recognize_intent(AgentRequest(goal="如何找第一份工作"))
        self.assertEqual(result.source, "llm")
        self.assertEqual(result.model_attempts, 2)

    def test_provider_error_not_retried(self):
        model = Mock()
        model.invoke.side_effect = RuntimeError("configuration error")
        with patch("app.agent.intent_router.get_llm", return_value=model):
            result = recognize_intent(AgentRequest(goal="处理一下"))
        self.assertEqual(model.invoke.call_count, 1)
        self.assertEqual(result.error_kind, "provider_error")

    def test_hallucinated_company_is_not_accepted(self):
        model = Mock()
        model.invoke.return_value = SimpleNamespace(content='{"intent":"fit_analysis","confidence":0.99,"company_name":"捏造公司"}')
        with patch("app.agent.intent_router.get_llm", return_value=model):
            result = recognize_intent(AgentRequest(goal="分析一下"))
        self.assertIsNone(result.intent)
        self.assertEqual(result.error_kind, "ungrounded_identity")

    def test_single_again_character_does_not_inherit_old_intent(self):
        with patch("app.agent.intent_router.get_llm", side_effect=RuntimeError("offline")):
            result = recognize_intent(AgentRequest(goal="再看看另一家公司的岗位"), last_intent="application_package")
        self.assertNotEqual(result.source, "context_rule")

    def test_failed_entity_extraction_cannot_silently_use_old_target(self):
        with patch("app.agent.intent_router.get_llm", side_effect=RuntimeError("offline")):
            result = recognize_intent(AgentRequest(goal="分析腾讯后端工程师匹配度", company_name="Acme",
                                                   job_title="Engineer"))
        self.assertIsNone(result.intent)


if __name__ == "__main__":
    unittest.main()
