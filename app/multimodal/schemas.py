from pydantic import BaseModel, Field


class IngestResult(BaseModel):
    saved_files: list[str]
    chunks_added: int
    modality: str
    collection_name: str | None = None
    pipeline: str | None = None
    document_count: int | None = None


class ChatTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$", description="user or assistant")
    content: str


class Citation(BaseModel):
    filename: str
    modality: str
    source: str


class MultimodalChatResponse(BaseModel):
    conversation_id: str | None = None
    answer: str
    citations: list[Citation]
    retrieved_chunks: int
    candidate_chunks: int
    reranker_applied: bool
    reranker_model: str | None = None
    reranker_reason: str | None = None
    context_usage: dict = Field(default_factory=dict)
    conversation_summary: dict = Field(default_factory=dict)


class EvalSample(BaseModel):
    query_id: str | None = None
    sample_id: str | None = None
    scenario: str | None = None
    query: str = Field(..., description="Evaluation query")
    expected_sources: list[str] = Field(
        default_factory=list,
        description="Expected source filenames for hit-rate/MRR evaluation",
    )
    expected_keywords: list[str] = Field(
        default_factory=list,
        description="Expected keywords that should appear in retrieved context",
    )
    expected_answer: str | None = None


class EvalRequest(BaseModel):
    samples: list[EvalSample]
    retrieve_k: int = 6
    candidate_k: int | None = None
    rerank_top_n: int | None = None
    include_answer_check: bool = False
    dataset_name: str | None = None


class EvalMetrics(BaseModel):
    sample_count: int
    baseline_hit_rate: float
    rerank_hit_rate: float
    baseline_mrr: float
    rerank_mrr: float
    baseline_keyword_recall: float
    rerank_keyword_recall: float
    citation_hit_rate: float | None = None


class EvalExperimentMetrics(BaseModel):
    name: str
    label: str
    hit_rate: float
    mrr: float
    keyword_recall: float


class EvalSampleExperimentResult(BaseModel):
    hit: bool
    mrr: float
    keyword_recall: float
    retrieved_sources: list[str] = Field(default_factory=list)


class EvalSampleResult(BaseModel):
    query: str
    query_id: str | None = None
    sample_id: str | None = None
    scenario: str | None = None
    expected_answer: str | None = None
    experiments: dict[str, EvalSampleExperimentResult] = Field(default_factory=dict)
    baseline_hit: bool
    rerank_hit: bool
    baseline_mrr: float
    rerank_mrr: float
    baseline_keyword_recall: float
    rerank_keyword_recall: float
    citation_hit: bool | None = None


class EvalResponse(BaseModel):
    config: dict[str, int | bool | str | None]
    metrics: EvalMetrics
    experiments: list[EvalExperimentMetrics] = Field(default_factory=list)
    samples: list[EvalSampleResult]


class EvalDatasetIngestRequest(BaseModel):
    dataset_name: str = Field(default="retrieval", description="retrieval, multimodal, all, zh_retrieval, zh_multimodal, or zh_all")
    sample_limit: int | None = Field(default=None, ge=1)
    include_images: bool = True


class EvalDatasetSamplesResponse(BaseModel):
    dataset_name: str
    samples: list[EvalSample]
