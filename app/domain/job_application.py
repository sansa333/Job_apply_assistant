from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AgentStage(str, Enum):
    REQUEST_ACCEPTED = "request_accepted"
    JOB_RESOLVED = "job_resolved"
    EVIDENCE_COLLECTED = "evidence_collected"
    REQUIREMENTS_PARSED = "requirements_parsed"
    EVIDENCE_ALIGNED = "evidence_aligned"
    SCORED = "scored"
    MATERIALS_GENERATED = "materials_generated"
    OUTPUT_VALIDATED = "output_validated"
    AWAITING_HUMAN_CONFIRMATION = "awaiting_human_confirmation"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class RequirementCategory(str, Enum):
    TECHNICAL_SKILL = "technical_skill"
    RESPONSIBILITY = "responsibility"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    DOMAIN = "domain"
    LANGUAGE = "language"
    LOCATION_WORK_MODE = "location_work_mode"
    SOFT_SKILL = "soft_skill"
    OTHER = "other"


class EvidenceSupport(str, Enum):
    DIRECT = "direct"
    PARTIAL = "partial"
    MISSING = "missing"


class JobRequirement(BaseModel):
    requirement_id: str
    category: RequirementCategory
    text: str
    normalized_terms: list[str] = Field(default_factory=list)
    must_have: bool = False
    preferred: bool = False
    weight: float = 1.0
    source_section: str = "unknown"


class ParsedJobDescription(BaseModel):
    company_name: str
    job_title: str
    location: str | None = None
    language: str = "unknown"
    requirements: list[JobRequirement] = Field(default_factory=list)
    source_url: str | None = None
    content_hash: str | None = None


class CandidateFact(BaseModel):
    fact_id: str
    text: str
    normalized_terms: list[str] = Field(default_factory=list)
    category: RequirementCategory = RequirementCategory.OTHER
    source_id: str
    source_name: str
    section: str = "unknown"


class ParsedCandidateProfile(BaseModel):
    candidate_id: str | None = None
    facts: list[CandidateFact] = Field(default_factory=list)
    source_kind: str = "user_provided"


class EvidenceMatch(BaseModel):
    requirement_id: str
    requirement_text: str
    category: RequirementCategory
    must_have: bool
    support: EvidenceSupport
    confidence: float = Field(ge=0.0, le=1.0)
    matched_terms: list[str] = Field(default_factory=list)
    evidence_fact_ids: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(default_factory=list)
    source_names: list[str] = Field(default_factory=list)
    explanation: str


class CategoryScore(BaseModel):
    category: RequirementCategory
    earned_weight: float
    total_weight: float
    score: float = Field(ge=0.0, le=100.0)


class ScoreBreakdown(BaseModel):
    overall_score: float = Field(ge=0.0, le=100.0)
    coverage_score: float = Field(ge=0.0, le=100.0)
    must_have_coverage: float = Field(ge=0.0, le=100.0)
    category_scores: list[CategoryScore] = Field(default_factory=list)
    direct_matches: int = 0
    partial_matches: int = 0
    missing_requirements: int = 0
    missing_must_haves: list[str] = Field(default_factory=list)
    scoring_version: str = "evidence_weighted_v1"
    calibration_status: str = "uncalibrated_baseline"


class ValidationFinding(BaseModel):
    code: str
    severity: str
    message: str
    claim: str | None = None


class WorkflowEvent(BaseModel):
    stage: AgentStage
    status: str
    detail: str
    tool_name: str | None = None
    duration_ms: float | None = None
