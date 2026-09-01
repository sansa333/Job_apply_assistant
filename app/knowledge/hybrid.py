from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from langchain_core.documents import Document

from app.multimodal.reranker import CrossEncoderReranker, RerankResult


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#./-]*|[\u4e00-\u9fff]")


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _chunk_id(document: Document) -> str:
    return str(document.metadata.get("chunk_id", document.page_content))


def reciprocal_rank_fusion(*rankings: list[Document], rrf_k: int = 60) -> list[Document]:
    """Fuse rankings deterministically while retaining the first ranking's tie order."""
    scores: dict[str, float] = defaultdict(float)
    documents: dict[str, Document] = {}
    first_seen: dict[str, int] = {}
    position = 0
    for ranking in rankings:
        for rank, document in enumerate(ranking, start=1):
            identifier = _chunk_id(document)
            scores[identifier] += 1.0 / (rrf_k + rank)
            documents.setdefault(identifier, document)
            first_seen.setdefault(identifier, position)
            position += 1
    return [
        documents[identifier]
        for identifier in sorted(scores, key=lambda identifier: (-scores[identifier], first_seen[identifier]))
    ]


def bm25_rank(query: str, documents: list[Document], *, k: int) -> list[Document]:
    """A compact BM25 ranker suitable for the small chunk set of one exact job."""
    if not documents:
        return []
    query_terms = _tokens(query)
    if not query_terms:
        return documents[:k]
    tokenized = [_tokens(document.page_content) for document in documents]
    lengths = [len(tokens) for tokens in tokenized]
    average_length = sum(lengths) / len(lengths) or 1.0
    document_frequency = Counter(term for terms in tokenized for term in set(terms))
    query_counts = Counter(query_terms)
    scores: list[float] = []
    k1, b = 1.5, 0.75
    for terms, length in zip(tokenized, lengths):
        term_counts = Counter(terms)
        score = 0.0
        for term, query_count in query_counts.items():
            frequency = term_counts.get(term, 0)
            if not frequency:
                continue
            idf = math.log(1.0 + (len(documents) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            denominator = frequency + k1 * (1.0 - b + b * length / average_length)
            score += query_count * idf * frequency * (k1 + 1.0) / denominator
        scores.append(score)
    ranked = sorted(enumerate(documents), key=lambda item: (-scores[item[0]], item[0]))
    return [document for _, document in ranked[:k]]


@dataclass(frozen=True)
class JobRetrievalResult:
    documents: list[Document]
    strategy: str
    candidate_count: int
    reranker_applied: bool
    reranker_model: str | None
    reranker_reason: str | None


class JobHybridRetriever:
    """Exact-job candidate generation, RRF fusion, and optional Cross-Encoder reranking."""

    def __init__(
        self,
        *,
        ingestion,
        reranker: CrossEncoderReranker,
        candidate_k: int = 12,
        rrf_k: int = 60,
        strategy: str = "hybrid_rerank",
        rerank_weight: float = 1.0,
    ):
        if strategy not in {"vector", "hybrid", "hybrid_rerank"}:
            raise ValueError("strategy must be vector, hybrid, or hybrid_rerank")
        if not 0.0 <= rerank_weight <= 1.0:
            raise ValueError("rerank_weight must be between 0 and 1")
        self.ingestion = ingestion
        self.reranker = reranker
        self.candidate_k = max(1, candidate_k)
        self.rrf_k = max(1, rrf_k)
        self.strategy = strategy
        self.rerank_weight = rerank_weight

    def retrieve(self, job_id: str, query: str, *, k: int = 5) -> JobRetrievalResult:
        candidates = self._candidate_documents(job_id, query)
        if self.strategy != "hybrid_rerank":
            return self._without_rerank(candidates, k=k)
        rerank_result: RerankResult = self.reranker.rerank(query, candidates, top_n=k)
        return self._with_rerank(candidates, rerank_result, k=k)

    def retrieve_many(self, requests: list[tuple[str, str]], *, k: int = 5) -> list[JobRetrievalResult]:
        """Batch the final rerank step while preserving each query's exact-job candidates."""
        candidates = [self._candidate_documents(job_id, query) for job_id, query in requests]
        if self.strategy != "hybrid_rerank":
            return [self._without_rerank(items, k=k) for items in candidates]
        reranked = self.reranker.rerank_many(
            [(query, items) for (_, query), items in zip(requests, candidates)], top_n=k
        )
        return [self._with_rerank(items, result, k=k) for items, result in zip(candidates, reranked)]

    def _candidate_documents(self, job_id: str, query: str) -> list[Document]:
        candidates = [
            document
            for document in self.ingestion.retrieve_vector_for_job(job_id, query, k=self.candidate_k)
            if document.metadata.get("job_id") == job_id
        ]
        if self.strategy != "vector":
            lexical_documents = [
                document for document in self.ingestion.get_documents_for_job(job_id) if document.metadata.get("job_id") == job_id
            ]
            lexical = bm25_rank(query, lexical_documents, k=self.candidate_k)
            candidates = reciprocal_rank_fusion(candidates, lexical, rrf_k=self.rrf_k)[: self.candidate_k]
        return candidates

    def _without_rerank(self, candidates: list[Document], *, k: int) -> JobRetrievalResult:
        return JobRetrievalResult(
            documents=candidates[:k],
            strategy=self.strategy,
            candidate_count=len(candidates),
            reranker_applied=False,
            reranker_model=None,
            reranker_reason="not_requested",
        )

    def _with_rerank(self, candidates: list[Document], rerank_result: RerankResult, *, k: int) -> JobRetrievalResult:
        documents = self._blend_rerank_scores(candidates, rerank_result)[:k] if rerank_result.applied else candidates[:k]
        return JobRetrievalResult(
            documents=documents,
            strategy=self.strategy,
            candidate_count=len(candidates),
            reranker_applied=rerank_result.applied,
            reranker_model=rerank_result.model,
            reranker_reason=rerank_result.reason,
        )

    def _blend_rerank_scores(self, candidates: list[Document], rerank_result: RerankResult) -> list[Document]:
        if self.rerank_weight >= 1.0 or not rerank_result.scores_by_chunk:
            return rerank_result.docs
        scores = rerank_result.scores_by_chunk
        values = [scores.get(_chunk_id(document), 0.0) for document in candidates]
        minimum, maximum = min(values, default=0.0), max(values, default=0.0)
        span = maximum - minimum
        blended: list[tuple[float, int, Document]] = []
        for rank, document in enumerate(candidates, start=1):
            cross_score = scores.get(_chunk_id(document), minimum)
            normalized_cross = (cross_score - minimum) / span if span else 0.5
            rrf_rank_score = 1.0 / rank
            score = (1.0 - self.rerank_weight) * rrf_rank_score + self.rerank_weight * normalized_cross
            blended.append((score, rank, document))
        return [document for _, _, document in sorted(blended, key=lambda item: (-item[0], item[1]))]

    def retrieve_for_job(self, job_id: str, query: str, *, k: int = 5) -> list[Document]:
        return self.retrieve(job_id, query, k=k).documents
