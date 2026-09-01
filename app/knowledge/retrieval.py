from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from app.knowledge.catalog import JobCatalog
from app.knowledge.hybrid import JobHybridRetriever
from app.knowledge.ingestion import JobKnowledgeIngestion
from app.knowledge.models import JobRecord
from app.multimodal.reranker import CrossEncoderReranker
from app.config import settings


@dataclass(frozen=True)
class JobResolution:
    status: str
    record: JobRecord | None
    job_documents: list[Document]
    retrieval_strategy: str | None = None
    candidate_count: int = 0
    reranker_applied: bool = False
    reranker_model: str | None = None
    reranker_reason: str | None = None


class JobScopedRetriever:
    """Resolve an exact catalog record before semantic retrieval."""

    def __init__(self, *, catalog: JobCatalog, job_ingestion: JobKnowledgeIngestion, hybrid_retriever=None):
        self.catalog = catalog
        self.job_ingestion = job_ingestion
        self.hybrid_retriever = hybrid_retriever or JobHybridRetriever(
            ingestion=job_ingestion,
            reranker=CrossEncoderReranker(
                enabled=settings.enable_reranker,
                model_name=settings.reranker_model,
                local_files_only=settings.reranker_local_files_only,
            ),
            candidate_k=settings.job_retrieval_candidate_k,
            rrf_k=settings.job_retrieval_rrf_k,
            strategy=settings.job_retrieval_strategy,
            rerank_weight=settings.job_reranker_weight,
        )

    def resolve(self, company_name: str, job_title: str, query: str, *, k: int = 5) -> JobResolution:
        matches = self.catalog.lookup(company_name, job_title)
        if not matches:
            return JobResolution(status="job_not_found", record=None, job_documents=[])
        record = matches[0]
        retrieval = self.hybrid_retriever.retrieve(record.job_id, query, k=k)
        return JobResolution(
            status="ok",
            record=record,
            job_documents=retrieval.documents,
            retrieval_strategy=retrieval.strategy,
            candidate_count=retrieval.candidate_count,
            reranker_applied=retrieval.reranker_applied,
            reranker_model=retrieval.reranker_model,
            reranker_reason=retrieval.reranker_reason,
        )
