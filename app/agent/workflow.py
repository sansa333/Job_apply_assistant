from __future__ import annotations

from time import perf_counter

from pydantic import BaseModel, Field

from app.domain.job_application import AgentStage, WorkflowEvent
from app.schemas import ApplicationIntent


class WorkflowTrace(BaseModel):
    workflow_version: str = "evidence_grounded_agent_v1"
    events: list[WorkflowEvent] = Field(default_factory=list)

    def record(
        self,
        stage: AgentStage,
        *,
        status: str,
        detail: str,
        tool_name: str | None = None,
        started_at: float | None = None,
    ) -> None:
        duration_ms = None if started_at is None else round((perf_counter() - started_at) * 1000, 3)
        self.events.append(
            WorkflowEvent(
                stage=stage,
                status=status,
                detail=detail,
                tool_name=tool_name,
                duration_ms=duration_ms,
            )
        )

    @property
    def stages(self) -> list[str]:
        return [event.stage.value for event in self.events]

    @property
    def tools(self) -> list[str]:
        return [event.tool_name for event in self.events if event.tool_name]

    def as_dict(self) -> dict:
        return self.model_dump(mode="json") | {"stages": self.stages, "tools": self.tools}


class WorkflowPolicyInput(BaseModel):
    intent: ApplicationIntent
    has_company: bool = True
    has_job_title: bool = True
    has_jd: bool = True
    job_exists: bool = True
    has_candidate_id: bool = False
    has_resume_text: bool = False
    has_all_contacts: bool = False
    generation_valid: bool = True


class WorkflowPolicyPlan(BaseModel):
    expected_tools: list[str]
    terminal_status: str
    terminal_stage: AgentStage
    next_action: str | None = None


def plan_workflow(value: WorkflowPolicyInput) -> WorkflowPolicyPlan:
    """Deterministic safety policy used by runtime routing and trajectory evals."""
    tools: list[str] = ["activate_skill"]
    if value.intent == ApplicationIntent.GENERAL_ADVICE:
        return WorkflowPolicyPlan(
            expected_tools=tools,
            terminal_status="completed",
            terminal_stage=AgentStage.COMPLETED,
        )
    if not value.has_company or not value.has_job_title:
        return WorkflowPolicyPlan(
            expected_tools=tools,
            terminal_status="needs_job_identity",
            terminal_stage=AgentStage.BLOCKED,
            next_action="provide_company_and_job_title",
        )
    tools.append("retrieve_job")
    if not value.has_jd and not value.job_exists:
        return WorkflowPolicyPlan(
            expected_tools=tools,
            terminal_status="job_not_found",
            terminal_stage=AgentStage.BLOCKED,
            next_action="upload_target_jd",
        )
    if value.has_candidate_id:
        tools.append("retrieve_profile")
    if not value.has_candidate_id and not value.has_resume_text:
        return WorkflowPolicyPlan(
            expected_tools=tools,
            terminal_status="needs_candidate_evidence",
            terminal_stage=AgentStage.BLOCKED,
            next_action="provide_candidate_id_or_resume_text",
        )
    tools.extend(["parse_job_requirements", "align_candidate_evidence", "score_job_fit"])
    if value.intent == ApplicationIntent.FIT_ANALYSIS:
        return WorkflowPolicyPlan(
            expected_tools=tools,
            terminal_status="ok",
            terminal_stage=AgentStage.OUTPUT_VALIDATED,
        )
    if value.intent in {
        ApplicationIntent.COVER_LETTER,
        ApplicationIntent.APPLICATION_EMAIL,
        ApplicationIntent.INTERVIEW_PREP,
        ApplicationIntent.APPLICATION_PACKAGE,
    }:
        tools.extend(["generate_materials", "validate_grounded_output"])
        if not value.generation_valid:
            return WorkflowPolicyPlan(
                expected_tools=tools,
                terminal_status="validation_failed",
                terminal_stage=AgentStage.BLOCKED,
                next_action="review_unsupported_claims",
            )
        if value.intent == ApplicationIntent.APPLICATION_PACKAGE and not value.has_all_contacts:
            return WorkflowPolicyPlan(
                expected_tools=tools,
                terminal_status="generated_not_submitted",
                terminal_stage=AgentStage.AWAITING_HUMAN_CONFIRMATION,
                next_action="confirm_contact_details_before_sending",
            )
        return WorkflowPolicyPlan(
            expected_tools=tools,
            terminal_status="generated_not_submitted",
            terminal_stage=AgentStage.AWAITING_HUMAN_CONFIRMATION,
            next_action="review_and_confirm_before_sending",
        )
    return WorkflowPolicyPlan(
        expected_tools=tools,
        terminal_status="completed",
        terminal_stage=AgentStage.COMPLETED,
    )
