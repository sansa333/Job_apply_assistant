"""Typed domain objects for the evidence-grounded job application workflow."""

from app.domain.job_application import (
    AgentStage,
    EvidenceMatch,
    EvidenceSupport,
    JobRequirement,
    ParsedCandidateProfile,
    ParsedJobDescription,
    RequirementCategory,
    ScoreBreakdown,
    ValidationFinding,
    WorkflowEvent,
)

__all__ = [
    "AgentStage",
    "EvidenceMatch",
    "EvidenceSupport",
    "JobRequirement",
    "ParsedCandidateProfile",
    "ParsedJobDescription",
    "RequirementCategory",
    "ScoreBreakdown",
    "ValidationFinding",
    "WorkflowEvent",
]
