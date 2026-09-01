import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.agent.intent_router import IntentRecognition
from app.agent.job_agent import _apply_recognized_intent, run_job_agent
from app.agent.skill_runtime import JOB_APPLICATION_SKILL, SkillRegistry, SkillSession
from app.main import app
from app.schemas import AgentRequest, ApplicationIntent


class _PackageService:
    def one_click_apply(self, req):
        return {
            "application_id": "application-1",
            "output_dir": "outputs/application-1",
            "fit_report": "基于用户提供简历的匹配摘要",
            "cover_letter": "求职信草稿",
            "interview_questions": "面试题草稿",
            "application_email": "邮件草稿",
            "status": "generated_not_submitted",
            "stage": "generated",
            "evidence_level": "user_provided",
            "contact_status": "pending_confirmation",
            "missing_fields": ["candidate_email", "candidate_phone"],
            "next_action": "confirm_contact_details_before_sending",
        }


class _FitService:
    def analyze_scoped_fit(self, req):
        return {
            "fit_report": f"已分析 {req.company_name} / {req.job_title}",
            "status": "ok",
            "stage": "analyzed",
            "evidence_level": "user_provided",
        }


class _CapturingAgent:
    payload = None

    def invoke(self, payload):
        type(self).payload = payload
        return {"messages": [SimpleNamespace(content="已给出求职建议")]}


class AgentSkillWorkflowTests(unittest.TestCase):
    def test_agent_endpoint_exposes_structured_workflow_state(self) -> None:
        with patch(
            "app.main.run_job_agent",
            return_value={
                "result": "申请包草稿已生成",
                "status": "generated_not_submitted",
                "stage": "generated",
                "active_skills": [JOB_APPLICATION_SKILL],
                "evidence_level": "missing",
                "contact_status": "pending_confirmation",
                "missing_fields": ["candidate_email"],
                "next_action": "confirm_contact_details_before_sending",
                "artifacts": {"output_dir": "outputs/application-1"},
            },
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/api/agent",
                    json={"goal": "生成申请包", "intent": "application_package"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "generated_not_submitted")
        self.assertEqual(response.json()["next_action"], "confirm_contact_details_before_sending")

    def test_application_package_intent_uses_activated_skill_and_structured_tool_result(self) -> None:
        registry = SkillRegistry(__import__("app.config", fromlist=["settings"]).settings.skills_dir)
        session = SkillSession(registry)
        request = AgentRequest(
            goal="生成申请包",
            intent=ApplicationIntent.APPLICATION_PACKAGE,
            company_name="Acme",
            job_title="RAG Engineer",
            jd_text="Need RAG experience",
            resume_text="Built a RAG application",
        )
        with (
            patch("app.agent.job_agent.create_skill_session", return_value=session),
            patch("app.agent.job_agent.ApplicationService", return_value=_PackageService()),
            patch("app.agent.job_agent.get_llm", side_effect=AssertionError("deterministic workflow should not call an LLM")),
        ):
            result = run_job_agent(request)

        self.assertEqual(result["status"], "generated_not_submitted")
        self.assertEqual(result["stage"], "generated")
        self.assertEqual(result["active_skills"], [JOB_APPLICATION_SKILL])
        self.assertEqual(result["contact_status"], "pending_confirmation")
        self.assertEqual(result["artifacts"]["application_id"], "application-1")
        self.assertEqual(result["recognized_intent"], "application_package")
        self.assertEqual(result["intent_source"], "declared")

    def test_natural_language_high_confidence_intent_uses_deterministic_tool(self) -> None:
        registry = SkillRegistry(__import__("app.config", fromlist=["settings"]).settings.skills_dir)
        session = SkillSession(registry)
        request = AgentRequest(goal="帮我分析腾讯的大模型应用开发工程师岗位是否适合我", resume_text="RAG 项目经验")
        recognition = IntentRecognition(
            intent=ApplicationIntent.FIT_ANALYSIS,
            confidence=0.96,
            source="llm",
            company_name="腾讯",
            job_title="大模型应用开发工程师",
            summary="用户要求岗位匹配分析。",
        )
        with (
            patch("app.agent.job_agent.recognize_intent", return_value=recognition),
            patch("app.agent.job_agent.create_skill_session", return_value=session),
            patch("app.agent.job_agent.ApplicationService", return_value=_FitService()),
            patch("app.agent.job_agent.get_llm", side_effect=AssertionError("high-confidence route should be deterministic")),
        ):
            result = run_job_agent(request)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["recognized_intent"], "fit_analysis")
        self.assertEqual(result["intent_source"], "llm")
        self.assertEqual(result["extracted_entities"]["company_name"], "腾讯")
        self.assertIn("腾讯 / 大模型应用开发工程师", result["result"])

    def test_low_confidence_intent_does_not_override_agent_request(self) -> None:
        request = AgentRequest(goal="帮我处理一下")
        recognition = IntentRecognition(
            intent=ApplicationIntent.APPLICATION_PACKAGE,
            confidence=0.45,
            source="llm",
            company_name="模型猜测的公司",
            job_title="模型猜测的岗位",
        )

        routed = _apply_recognized_intent(request, recognition)

        self.assertIsNone(routed.intent)
        self.assertEqual(routed.company_name, "")
        self.assertEqual(routed.job_title, "")

    def test_dynamic_agent_receives_context_manager_budgeted_messages(self) -> None:
        registry = SkillRegistry(__import__("app.config", fromlist=["settings"]).settings.skills_dir)
        session = SkillSession(registry)
        request = AgentRequest(
            goal="给我一些求职建议",
            intent=ApplicationIntent.GENERAL_ADVICE,
            jd_text="超长岗位内容" * 1000,
            resume_text="超长简历内容" * 1000,
        )
        settings = __import__("app.config", fromlist=["settings"]).settings
        with (
            patch("app.agent.job_agent.create_skill_session", return_value=session),
            patch("app.agent.job_agent.get_llm", return_value=object()),
            patch("langchain.agents.create_agent", return_value=_CapturingAgent(), create=True),
            patch.object(settings, "agent_context_window_tokens", 4096),
            patch.object(settings, "agent_context_target_ratio", 0.6),
            patch.object(settings, "agent_context_output_reserve_tokens", 128),
            patch.object(settings, "agent_context_tool_reserve_tokens", 128),
        ):
            result = run_job_agent(request)

        self.assertTrue(result["context_usage"]["model_invoked"])
        self.assertEqual(result["context_usage"]["budget_status"], "within_budget")
        self.assertLessEqual(
            result["context_usage"]["estimated_input_tokens"],
            result["context_usage"]["target_input_tokens"],
        )
        self.assertTrue({"jd_text", "resume_text"} & set(result["context_usage"]["truncated_fields"]))
        self.assertIsNotNone(_CapturingAgent.payload)
        self.assertEqual(_CapturingAgent.payload["messages"][-1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
