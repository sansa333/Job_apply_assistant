from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    docs: list[Document]
    applied: bool
    model: str | None
    reason: str | None = None
    scores_by_chunk: dict[str, float] | None = None


class CrossEncoderReranker:
    """Optional cross-encoder reranker with graceful fallback."""

    def __init__(self, enabled: bool, model_name: str, local_files_only: bool = True):
        self.enabled = enabled
        self.model_name = model_name
        self.local_files_only = local_files_only
        self._model = None
        self._load_error: str | None = None
        self._load_attempted = False

    @property
    def available(self) -> bool:
        return self.enabled and self._model is not None

    def _ensure_model(self) -> None:
        if not self.enabled or self._model is not None or self._load_attempted:
            return

        self._load_attempted = True
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, local_files_only=self.local_files_only)
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Reranker cross-encoder unavailable: %s", self._load_error)

    def rerank(self, query: str, docs: list[Document], top_n: int) -> RerankResult:
        top_n = max(1, top_n)

        if not docs:
            return RerankResult(docs=[], applied=False, model=None, reason="no_documents")

        if not self.enabled:
            return RerankResult(docs=docs[:top_n], applied=False, model=None, reason="disabled")

        self._ensure_model()

        if not self.available:
            return RerankResult(
                docs=docs[:top_n],
                applied=False,
                model=None,
                reason=f"unavailable: {self._load_error or 'model_not_loaded'}",
            )

        pairs = [[query, doc.page_content] for doc in docs]

        try:
            scores = self._model.predict(pairs)
            ranked = sorted(zip(scores, docs), key=lambda x: float(x[0]), reverse=True)
            sorted_docs = [doc for _, doc in ranked]
            return RerankResult(
                docs=sorted_docs[:top_n],
                applied=True,
                model=self.model_name,
                reason=None,
                scores_by_chunk={str(doc.metadata.get("chunk_id", index)): float(score) for index, (score, doc) in enumerate(ranked)},
            )
        except Exception as exc:
            logger.warning("Reranker inference failed, fallback to vector order: %s", exc)
            return RerankResult(
                docs=docs[:top_n],
                applied=False,
                model=None,
                reason=f"inference_failed: {type(exc).__name__}",
            )

    def rerank_many(self, requests: list[tuple[str, list[Document]]], top_n: int) -> list[RerankResult]:
        """Batch Cross-Encoder pairs across independent queries to reduce CPU inference overhead."""
        top_n = max(1, top_n)
        if not requests:
            return []
        if not self.enabled:
            return [RerankResult(docs=docs[:top_n], applied=False, model=None, reason="disabled") for _, docs in requests]

        self._ensure_model()
        if not self.available:
            return [
                RerankResult(
                    docs=docs[:top_n],
                    applied=False,
                    model=None,
                    reason=f"unavailable: {self._load_error or 'model_not_loaded'}",
                )
                for _, docs in requests
            ]

        pairs: list[list[str]] = []
        offsets: list[tuple[int, int]] = []
        for query, docs in requests:
            start = len(pairs)
            pairs.extend([query, document.page_content] for document in docs)
            offsets.append((start, len(pairs)))
        if not pairs:
            return [RerankResult(docs=[], applied=False, model=None, reason="no_documents") for _, _ in requests]

        try:
            scores = self._model.predict(pairs, batch_size=8)
            results: list[RerankResult] = []
            for (_, docs), (start, end) in zip(requests, offsets):
                if not docs:
                    results.append(RerankResult(docs=[], applied=False, model=None, reason="no_documents"))
                    continue
                ranked = sorted(zip(scores[start:end], docs), key=lambda item: float(item[0]), reverse=True)
                results.append(
                    RerankResult(
                        docs=[document for _, document in ranked[:top_n]],
                        applied=True,
                        model=self.model_name,
                        reason=None,
                        scores_by_chunk={
                            str(document.metadata.get("chunk_id", index)): float(score)
                            for index, (score, document) in enumerate(ranked)
                        },
                    )
                )
            return results
        except Exception as exc:
            logger.warning("Batched reranker inference failed, fallback to candidate order: %s", exc)
            return [
                RerankResult(
                    docs=docs[:top_n],
                    applied=False,
                    model=None,
                    reason=f"inference_failed: {type(exc).__name__}",
                )
                for _, docs in requests
            ]

    @staticmethod
    def _lexical_rerank(query: str, docs: list[Document]) -> list[Document]:
        q_tokens = {tok for tok in query.lower().split() if tok.strip()}
        if not q_tokens:
            return docs

        def score(doc: Document) -> int:
            text = doc.page_content.lower()
            return sum(1 for tok in q_tokens if tok in text)

        return sorted(docs, key=score, reverse=True)
