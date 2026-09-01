from enum import Enum

from pydantic import BaseModel, EmailStr, Field


class ApplicationIntent(str, Enum):
    GENERAL_ADVICE = "general_advice"
    FIT_ANALYSIS = "fit_analysis"
    COVER_LETTER = "cover_letter"
    APPLICATION_EMAIL = "application_email"
    INTERVIEW_PREP = "interview_prep"
    APPLICATION_PACKAGE = "application_package"


class ConversationType(str, Enum):
    CAREER_GENERAL = "career_general"
    JOB_APPLICATION = "job_application"
    KNOWLEDGE_CHAT = "knowledge_chat"


class EvidenceLevel(str, Enum):
    VERIFIED_PROFILE = "verified_profile"
    USER_PROVIDED = "user_provided"
    MISSING = "missing"


class ContactStatus(str, Enum):
    CONFIRMED = "confirmed"
    PARTIAL = "partial"
    PENDING_CONFIRMATION = "pending_confirmation"


class IngestResponse(BaseModel):
    saved_files: list[str]
    chunks_added: int


class FitRequest(BaseModel):
    candidate_id: str | None = Field(default=None, examples=["current_candidate"])
    company_name: str = Field(..., examples=["某某科技有限公司"])
    job_title: str = Field(..., examples=["大模型应用开发工程师"])
    question: str = Field(default="请分析我的匹配度、主要证据与能力缺口。")
    jd_text: str = Field(default="", examples=["负责RAG、Agent、LangChain应用开发..."])
    resume_text: str = Field(default="", examples=["熟悉Python/FastAPI/LangChain..."])


class FitResponse(BaseModel):
    status: str = "ok"
    stage: str = "analyzed"
    fit_report: str = ""
    job_id: str | None = None
    message: str | None = None
    upload_action: str | None = None
    job_evidence: list[dict] = Field(default_factory=list)
    candidate_evidence: list[dict] = Field(default_factory=list)
    source_kind: str | None = None
    source_dataset: str | None = None
    historical_notice: str | None = None
    evidence_level: EvidenceLevel = EvidenceLevel.MISSING
    missing_fields: list[str] = Field(default_factory=list)
    next_action: str | None = None
    match_score: float | None = None
    parsed_job: dict = Field(default_factory=dict)
    parsed_candidate: dict = Field(default_factory=dict)
    evidence_matrix: list[dict] = Field(default_factory=list)
    score_breakdown: dict = Field(default_factory=dict)
    validation_findings: list[dict] = Field(default_factory=list)
    workflow_trace: dict = Field(default_factory=dict)


class OneClickApplyRequest(FitRequest):
    candidate_name: str | None = None
    candidate_email: EmailStr | None = None
    candidate_phone: str | None = None


class OneClickApplyResponse(BaseModel):
    application_id: str
    output_dir: str
    fit_report: str
    cover_letter: str
    interview_questions: str
    application_email: str
    status: str = "generated_not_submitted"
    stage: str = "generated"
    evidence_level: EvidenceLevel = EvidenceLevel.MISSING
    contact_status: ContactStatus = ContactStatus.PENDING_CONFIRMATION
    missing_fields: list[str] = Field(default_factory=list)
    next_action: str | None = None
    match_score: float | None = None
    score_breakdown: dict = Field(default_factory=dict)
    evidence_matrix: list[dict] = Field(default_factory=list)
    validation_findings: list[dict] = Field(default_factory=list)
    workflow_trace: dict = Field(default_factory=dict)


class AgentRequest(BaseModel):
    goal: str = Field(..., examples=["请分析岗位并生成申请包"])
    conversation_id: str | None = None
    intent: ApplicationIntent | None = None
    candidate_id: str | None = None
    job_id: str | None = None
    candidate_name: str | None = None
    candidate_email: EmailStr | None = None
    candidate_phone: str | None = None
    company_name: str = ""
    job_title: str = ""
    jd_text: str = ""
    resume_text: str = ""


class AgentResponse(BaseModel):
    conversation_id: str | None = None
    job_id: str | None = None
    result: str = ""
    status: str = "ok"
    stage: str = "completed"
    active_skills: list[str] = Field(default_factory=list)
    evidence_level: EvidenceLevel = EvidenceLevel.MISSING
    contact_status: ContactStatus = ContactStatus.PENDING_CONFIRMATION
    missing_fields: list[str] = Field(default_factory=list)
    next_action: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    match_score: float | None = None
    score_breakdown: dict = Field(default_factory=dict)
    evidence_matrix: list[dict] = Field(default_factory=list)
    validation_findings: list[dict] = Field(default_factory=list)
    workflow_trace: dict = Field(default_factory=dict)
    recognized_intent: ApplicationIntent | None = None
    intent_confidence: float = 0.0
    intent_source: str = "unresolved"
    intent_missing_fields: list[str] = Field(default_factory=list)
    intent_summary: str = ""
    extracted_entities: dict[str, str] = Field(default_factory=dict)
    context_usage: dict = Field(default_factory=dict)
    conversation_summary: dict = Field(default_factory=dict)


class ConversationCreateRequest(BaseModel):
    candidate_id: str = Field(..., min_length=1)
    conversation_type: ConversationType = ConversationType.JOB_APPLICATION
    job_id: str | None = None
    company_name: str = ""
    job_title: str = ""


class ConversationTurn(BaseModel):
    message_id: str
    role: str
    content: str
    intent: ApplicationIntent | None = None
    created_at: str


class ConversationSummary(BaseModel):
    conversation_id: str
    candidate_id: str
    conversation_type: ConversationType
    job_id: str | None = None
    company_name: str = ""
    job_title: str = ""
    last_intent: ApplicationIntent | None = None
    current_stage: str = "created"
    status: str = "active"
    created_at: str
    updated_at: str


class ConversationDetail(ConversationSummary):
    turns: list[ConversationTurn] = Field(default_factory=list)
    rolling_summary: dict = Field(default_factory=dict)
    tool_results: list[dict] = Field(default_factory=list)


class ToolResultDetail(BaseModel):
    tool_result_id: str
    conversation_id: str
    tool_name: str
    status: str | None = None
    stage: str | None = None
    summary: str = ""
    content: str
    content_chars: int
    created_at: str
