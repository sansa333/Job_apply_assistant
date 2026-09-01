import unittest

from app.agent.workflow import WorkflowPolicyInput, plan_workflow
from app.domain.job_application import AgentStage


class AgentWorkflowPolicyTests(unittest.TestCase):
    def test_application_package_requires_human_confirmation(self) -> None:
        plan = plan_workflow(
            WorkflowPolicyInput(
                intent="application_package",
                has_candidate_id=True,
                has_all_contacts=False,
            )
        )

        self.assertEqual(plan.terminal_status, "generated_not_submitted")
        self.assertEqual(plan.terminal_stage, AgentStage.AWAITING_HUMAN_CONFIRMATION)
        self.assertEqual(plan.next_action, "confirm_contact_details_before_sending")
        self.assertIn("validate_grounded_output", plan.expected_tools)

    def test_missing_public_job_snapshot_stops_before_candidate_analysis(self) -> None:
        plan = plan_workflow(
            WorkflowPolicyInput(
                intent="fit_analysis",
                has_jd=False,
                job_exists=False,
                has_candidate_id=True,
            )
        )

        self.assertEqual(plan.expected_tools, ["activate_skill", "retrieve_job"])
        self.assertEqual(plan.terminal_status, "job_not_found")


if __name__ == "__main__":
    unittest.main()
