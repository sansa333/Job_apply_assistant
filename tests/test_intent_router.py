import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.agent.intent_router import recognize_intent
from app.schemas import AgentRequest, ApplicationIntent


class _JsonIntentModel:
    def invoke(self, messages):
        return SimpleNamespace(
            content=(
                '{"intent":"fit_analysis","confidence":0.96,'
                '"company_name":"腾讯","job_title":"大模型应用开发工程师",'
                '"missing_fields":[],"summary":"用户要求岗位匹配分析"}'
            )
        )


class IntentRouterTests(unittest.TestCase):
    def test_declared_intent_does_not_call_classifier_model(self) -> None:
        request = AgentRequest(goal="生成申请包", intent=ApplicationIntent.APPLICATION_PACKAGE)
        with patch(
            "app.agent.intent_router.get_llm",
            side_effect=AssertionError("declared intent must skip classification"),
        ):
            result = recognize_intent(request)

        self.assertEqual(result.intent, ApplicationIntent.APPLICATION_PACKAGE)
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(result.source, "declared")

    def test_natural_language_is_classified_and_entities_are_extracted(self) -> None:
        request = AgentRequest(goal="帮我分析腾讯大模型应用开发工程师岗位是否适合我")
        with patch("app.agent.intent_router.get_llm", return_value=_JsonIntentModel()):
            result = recognize_intent(request)

        self.assertEqual(result.intent, ApplicationIntent.FIT_ANALYSIS)
        self.assertEqual(result.source, "llm")
        self.assertGreater(result.confidence, 0.9)
        self.assertEqual(result.company_name, "腾讯")
        self.assertEqual(result.job_title, "大模型应用开发工程师")

    def test_high_precision_rule_runs_before_classifier(self) -> None:
        request = AgentRequest(
            goal="请分析岗位匹配度",
            company_name="Acme",
            job_title="RAG Engineer",
        )
        with patch("app.agent.intent_router.get_llm", side_effect=RuntimeError("no model")):
            result = recognize_intent(request)

        self.assertEqual(result.intent, ApplicationIntent.FIT_ANALYSIS)
        self.assertEqual(result.source, "rule")
        self.assertEqual(result.missing_fields, [])

    def test_rule_fallback_is_available_when_entities_are_missing_and_model_fails(self) -> None:
        request = AgentRequest(goal="请分析岗位匹配度")
        with patch("app.agent.intent_router.get_llm", side_effect=RuntimeError("no model")):
            result = recognize_intent(request)

        self.assertEqual(result.intent, ApplicationIntent.FIT_ANALYSIS)
        self.assertEqual(result.source, "rule_fallback")
        self.assertEqual(result.missing_fields, ["company_name", "job_title"])

    def test_negated_package_and_explicit_interview_request_are_not_misrouted(self) -> None:
        request = AgentRequest(
            goal="不要生成申请包，只准备面试题",
            company_name="Acme",
            job_title="RAG Engineer",
        )
        with patch(
            "app.agent.intent_router.get_llm",
            side_effect=AssertionError("unambiguous correction should use a rule"),
        ):
            result = recognize_intent(request)

        self.assertEqual(result.intent, ApplicationIntent.INTERVIEW_PREP)
        self.assertEqual(result.source, "rule")

    def test_continuation_uses_last_conversation_intent(self) -> None:
        request = AgentRequest(
            goal="语气再正式一点",
            company_name="Acme",
            job_title="RAG Engineer",
        )
        result = recognize_intent(request, last_intent="application_email")

        self.assertEqual(result.intent, ApplicationIntent.APPLICATION_EMAIL)
        self.assertEqual(result.source, "context_rule")

    def test_unknown_goal_remains_unresolved_when_model_fails(self) -> None:
        request = AgentRequest(goal="帮我处理一下")
        with patch("app.agent.intent_router.get_llm", side_effect=RuntimeError("no model")):
            result = recognize_intent(request)

        self.assertIsNone(result.intent)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.source, "unresolved")


if __name__ == "__main__":
    unittest.main()
