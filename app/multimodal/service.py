from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import UploadFile
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from app.agent.context_manager import ContextManager, RollingSummaryManager
from app.agent.conversation_store import ConversationScopeError, ConversationStore
from app.config import settings
from app.llm import extract_token_usage, get_llm, get_vision_llm, message_to_text
from app.multimodal.prompts import IMAGE_ANALYSIS_PROMPT, QA_SYSTEM_PROMPT, QA_USER_PROMPT
from app.multimodal.reranker import CrossEncoderReranker, RerankResult
from app.multimodal.schemas import (
    ChatTurn,
    Citation,
    EvalDatasetIngestRequest,
    EvalExperimentMetrics,
    EvalMetrics,
    EvalRequest,
    EvalResponse,
    EvalSample,
    EvalSampleExperimentResult,
    EvalSampleResult,
    MultimodalChatResponse,
)
from app.embeddings import get_embeddings
from app.rag import load_one_file, split_documents
from app.utils.file_io import safe_filename
from app.utils.request_log import elapsed_ms, log_request_event, new_request_id, now_ms

TEXT_SUFFIXES = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
EVAL_DATASET_NAMES = {"retrieval", "multimodal", "all", "zh_retrieval", "zh_multimodal", "zh_all"}


class MultimodalAssistantService:
    """Knowledge assistant with text RAG and optional transient image context."""

    def __init__(self):
        self.chat_llm = get_llm(temperature=0.2)
        self.vision_llm = get_vision_llm(temperature=0.1)
        self.reranker = CrossEncoderReranker(
            enabled=settings.enable_reranker,
            model_name=settings.reranker_model,
            local_files_only=settings.reranker_local_files_only,
        )
        self.db = Chroma(
            collection_name=settings.mm_collection_name,
            embedding_function=get_embeddings(),
            persist_directory=str(settings.vector_db_dir / settings.mm_collection_name),
        )
        self.eval_db = Chroma(
            collection_name=settings.eval_collection_name,
            embedding_function=get_embeddings(),
            persist_directory=str(settings.vector_db_dir / settings.eval_collection_name),
        )

    async def ingest_text_files(self, files: list[UploadFile]) -> tuple[list[str], int]:
        request_id = new_request_id()
        start = now_ms()
        try:
            paths = await self._save_uploads(files, settings.mm_text_docs_dir, TEXT_SUFFIXES)

            docs: list[Document] = []
            for path in paths:
                loaded = load_one_file(path)
                for doc in loaded:
                    doc.metadata["source"] = str(path)
                    doc.metadata["filename"] = path.name
                    doc.metadata["modality"] = "text"
                    doc.metadata["collection"] = settings.mm_collection_name
                docs.extend(loaded)

            if not docs:
                log_request_event(
                    route="/api/mm/ingest/text",
                    request_id=request_id,
                    document_count=0,
                    chunk_count=0,
                    collection_name=settings.mm_collection_name,
                    elapsed_ms_value=elapsed_ms(start),
                )
                return [p.name for p in paths], 0

            chunks = split_documents(docs, collection_name=settings.mm_collection_name)
            self.db.add_documents(chunks)
            log_request_event(
                route="/api/mm/ingest/text",
                request_id=request_id,
                document_count=len(docs),
                chunk_count=len(chunks),
                collection_name=settings.mm_collection_name,
                elapsed_ms_value=elapsed_ms(start),
                extra={"saved_files": [p.name for p in paths]},
            )
            return [p.name for p in paths], len(chunks)
        except Exception as exc:
            log_request_event(
                route="/api/mm/ingest/text",
                request_id=request_id,
                collection_name=settings.mm_collection_name,
                elapsed_ms_value=elapsed_ms(start),
                status="error",
                error_type=type(exc).__name__,
            )
            raise

    async def chat(
        self,
        question: str,
        top_k: int = 6,
        history: list[ChatTurn] | None = None,
        image_file: UploadFile | None = None,
        conversation_id: str | None = None,
        candidate_id: str | None = None,
    ) -> MultimodalChatResponse:
        request_id = new_request_id()
        start = now_ms()
        top_k = max(1, min(top_k, 12))
        token_usage_total: dict[str, int] = {}
        conversation_store: ConversationStore | None = None
        rolling_summary: dict = {}
        recent_turns = [turn.model_dump() for turn in (history or [])]

        if bool(conversation_id) != bool(candidate_id):
            raise ConversationScopeError(
                "conversation_id and candidate_id must be provided together"
            )
        if conversation_id and candidate_id:
            conversation_store = ConversationStore(settings.agent_conversation_db_path)
            conversation = conversation_store.get(conversation_id, candidate_id)
            if conversation["conversation_type"] != "knowledge_chat":
                raise ConversationScopeError("Conversation is not a knowledge chat")
            rolling_summary, _ = RollingSummaryManager().maybe_roll(
                conversation_store,
                conversation_id,
                candidate_id,
            )
            recent_turns = conversation_store.recent_turns(
                conversation_id,
                candidate_id,
                limit=settings.agent_recent_turns,
            )
            conversation_store.append_turn(
                conversation_id,
                candidate_id,
                role="user",
                content=question,
                max_chars=settings.agent_max_turn_chars,
            )

        try:
            image_context = "未提供临时图片。"
            if image_file is not None:
                saved = await self._save_uploads([image_file], settings.mm_image_docs_dir, IMAGE_SUFFIXES)
                if saved:
                    image_context, image_tokens = self._describe_image_with_usage(saved[0])
                    self._merge_token_usage(token_usage_total, image_tokens)

            docs, candidate_docs, rerank_result = self._retrieve_docs(question, final_k=top_k)
            qa_context = ContextManager().build_qa_context(
                system_instructions=QA_SYSTEM_PROMPT,
                question=question,
                recent_turns=recent_turns,
                rolling_summary=rolling_summary,
                rag_context=self._format_rag_context(docs),
                image_context=image_context,
            )

            generation_error: Exception | None = None
            try:
                answer, answer_tokens = self._generate_answer_with_usage(
                    question=qa_context.question,
                    docs=docs,
                    history_text=qa_context.history_text,
                    image_context=qa_context.image_context,
                    rag_context=qa_context.rag_context,
                )
                self._merge_token_usage(token_usage_total, answer_tokens)
            except Exception as exc:
                generation_error = exc
                answer = self._fallback_answer(
                    question=question,
                    docs=docs,
                    image_context=image_context,
                    error=exc,
                )

            log_request_event(
                route="/api/mm/chat",
                request_id=request_id,
                document_count=len(self._extract_citations(candidate_docs)),
                chunk_count=len(docs),
                top_k=top_k,
                candidate_count=len(candidate_docs),
                rerank_enabled=self.reranker.enabled,
                rerank_applied=rerank_result.applied,
                collection_name=settings.mm_collection_name,
                elapsed_ms_value=elapsed_ms(start),
                token_usage=token_usage_total or None,
                status="degraded" if generation_error else "success",
                error_type=type(generation_error).__name__ if generation_error else None,
                extra=(
                    {"llm_error": self._safe_error_summary(generation_error)}
                    if generation_error
                    else None
                ),
            )

            if conversation_store and conversation_id and candidate_id:
                conversation_store.append_turn(
                    conversation_id,
                    candidate_id,
                    role="assistant",
                    content=answer,
                    max_chars=settings.agent_max_turn_chars,
                )
                rolling_summary, summary_updated = RollingSummaryManager().maybe_roll(
                    conversation_store,
                    conversation_id,
                    candidate_id,
                )
                qa_context.usage["rolling_summary_updated"] = summary_updated

            citations = self._extract_citations(docs)
            return MultimodalChatResponse(
                conversation_id=conversation_id,
                answer=answer,
                citations=citations,
                retrieved_chunks=len(docs),
                candidate_chunks=len(candidate_docs),
                reranker_applied=rerank_result.applied,
                reranker_model=rerank_result.model,
                reranker_reason=rerank_result.reason,
                context_usage=qa_context.usage,
                conversation_summary=rolling_summary,
            )
        except Exception as exc:
            log_request_event(
                route="/api/mm/chat",
                request_id=request_id,
                top_k=top_k,
                rerank_enabled=self.reranker.enabled,
                collection_name=settings.mm_collection_name,
                elapsed_ms_value=elapsed_ms(start),
                token_usage=token_usage_total or None,
                status="error",
                error_type=type(exc).__name__,
            )
            raise

    def evaluate(self, request: EvalRequest) -> EvalResponse:
        request_id = new_request_id()
        start = now_ms()
        retrieve_k = max(1, min(request.retrieve_k, 12))
        candidate_k = request.candidate_k or max(retrieve_k, settings.reranker_candidate_k)
        candidate_k = max(retrieve_k, min(candidate_k, 30))
        rerank_top_n = request.rerank_top_n or max(retrieve_k, settings.reranker_top_n)
        rerank_top_n = max(retrieve_k, min(rerank_top_n, candidate_k))

        try:
            sample_results: list[EvalSampleResult] = []
            max_candidate_count = 0
            rerank_applied = False

            for sample in request.samples:
                candidate_docs = self.eval_db.similarity_search(sample.query, k=candidate_k)
                max_candidate_count = max(max_candidate_count, len(candidate_docs))
                baseline_docs = candidate_docs[:retrieve_k]
                rerank_result = self.reranker.rerank(sample.query, candidate_docs, top_n=rerank_top_n)
                rerank_applied = rerank_applied or rerank_result.applied
                reranked_docs = rerank_result.docs[:retrieve_k] if rerank_result.applied else baseline_docs

                no_rag_result = self._sample_experiment_result([], sample)
                baseline_result = self._sample_experiment_result(baseline_docs, sample)
                rerank_result_metrics = self._sample_experiment_result(reranked_docs, sample)

                citation_hit: bool | None = None
                if request.include_answer_check and sample.expected_sources:
                    answer = self._generate_answer(sample.query, reranked_docs)
                    citation_hit = self._citation_hit(answer, sample.expected_sources)

                sample_results.append(
                    EvalSampleResult(
                        query=sample.query,
                        query_id=sample.query_id,
                        sample_id=sample.sample_id,
                        scenario=sample.scenario,
                        expected_answer=sample.expected_answer,
                        experiments={
                            "no_rag": no_rag_result,
                            "vector": baseline_result,
                            "vector_rerank": rerank_result_metrics,
                        },
                        baseline_hit=baseline_result.hit,
                        rerank_hit=rerank_result_metrics.hit,
                        baseline_mrr=baseline_result.mrr,
                        rerank_mrr=rerank_result_metrics.mrr,
                        baseline_keyword_recall=baseline_result.keyword_recall,
                        rerank_keyword_recall=rerank_result_metrics.keyword_recall,
                        citation_hit=citation_hit,
                    )
                )

            metrics = self._aggregate_metrics(sample_results)
            experiments = self._aggregate_experiment_metrics(sample_results)
            config = {
                "retrieve_k": retrieve_k,
                "candidate_k": candidate_k,
                "rerank_top_n": rerank_top_n,
                "k_label": f"@{retrieve_k}",
                "dataset_name": request.dataset_name,
                "collection_name": settings.eval_collection_name,
                "reranker_enabled": settings.enable_reranker,
                "reranker_loaded": self.reranker.available,
                "reranker_model": settings.reranker_model,
                "reranker_local_files_only": settings.reranker_local_files_only,
                "include_answer_check": request.include_answer_check,
            }

            log_request_event(
                route="/api/mm/evaluate",
                request_id=request_id,
                document_count=len(request.samples),
                chunk_count=sum(
                    len(result.experiments["vector"].retrieved_sources)
                    for result in sample_results
                    if "vector" in result.experiments
                ),
                top_k=retrieve_k,
                candidate_count=max_candidate_count,
                rerank_enabled=self.reranker.enabled,
                rerank_applied=rerank_applied,
                collection_name=settings.eval_collection_name,
                elapsed_ms_value=elapsed_ms(start),
                extra={"dataset_name": request.dataset_name},
            )

            return EvalResponse(config=config, metrics=metrics, experiments=experiments, samples=sample_results)
        except Exception as exc:
            log_request_event(
                route="/api/mm/evaluate",
                request_id=request_id,
                document_count=len(request.samples),
                top_k=retrieve_k,
                rerank_enabled=self.reranker.enabled,
                collection_name=settings.eval_collection_name,
                elapsed_ms_value=elapsed_ms(start),
                status="error",
                error_type=type(exc).__name__,
            )
            raise

    def load_eval_dataset_samples(self, dataset_name: str, sample_limit: int | None = None) -> list[EvalSample]:
        dataset_name = dataset_name.lower().strip()
        self._validate_eval_dataset_name(dataset_name)
        samples: list[EvalSample] = []

        retrieval_path = self._eval_retrieval_path(dataset_name)
        if retrieval_path:
            path = retrieval_path
            for record in self._read_jsonl(path):
                samples.append(
                    EvalSample(
                        query_id=record.get("query_id"),
                        scenario=record.get("scenario"),
                        query=record["query"],
                        expected_sources=record.get("expected_sources", []),
                        expected_keywords=record.get("expected_keywords", []),
                    )
                )
                if sample_limit and len(samples) >= sample_limit:
                    return samples

        multimodal_path = self._eval_multimodal_path(dataset_name)
        if multimodal_path:
            path = multimodal_path
            for record in self._read_jsonl(path):
                choices = record.get("choices") or {}
                choice_text = " ".join(f"{key}. {value}" for key, value in choices.items())
                query = f"{record['question']}\n{choice_text}".strip()
                samples.append(
                    EvalSample(
                        sample_id=record.get("sample_id"),
                        scenario=record.get("scenario"),
                        query=query,
                        expected_sources=record.get("expected_sources", []),
                        expected_keywords=record.get("expected_keywords", []),
                        expected_answer=record.get("answer"),
                    )
                )
                if sample_limit and len(samples) >= sample_limit:
                    return samples

        return samples

    def ingest_eval_dataset(self, request: EvalDatasetIngestRequest) -> tuple[list[str], int, int]:
        request_id = new_request_id()
        start = now_ms()
        try:
            docs, saved_files, token_usage = self._load_eval_dataset_documents(request)
            if not docs:
                log_request_event(
                    route="/api/mm/ingest/eval-dataset",
                    request_id=request_id,
                    document_count=0,
                    chunk_count=0,
                    collection_name=settings.eval_collection_name,
                    elapsed_ms_value=elapsed_ms(start),
                    token_usage=token_usage or None,
                    extra={"dataset_name": request.dataset_name},
                )
                return saved_files, 0, 0

            for document in docs:
                document.metadata["collection"] = settings.eval_collection_name
                document.metadata["scope"] = "eval_demo"
            chunks = split_documents(docs, collection_name=settings.eval_collection_name)
            added = self._add_eval_documents_once(chunks, id_prefix=f"eval:{request.dataset_name}")
            log_request_event(
                route="/api/mm/ingest/eval-dataset",
                request_id=request_id,
                document_count=len(docs),
                chunk_count=added,
                collection_name=settings.eval_collection_name,
                elapsed_ms_value=elapsed_ms(start),
                token_usage=token_usage or None,
                extra={
                    "dataset_name": request.dataset_name,
                    "sample_limit": request.sample_limit,
                    "include_images": request.include_images,
                    "skipped_duplicate_chunks": len(chunks) - added,
                },
            )
            return saved_files, added, len(docs)
        except Exception as exc:
            log_request_event(
                route="/api/mm/ingest/eval-dataset",
                request_id=request_id,
                collection_name=settings.eval_collection_name,
                elapsed_ms_value=elapsed_ms(start),
                status="error",
                error_type=type(exc).__name__,
                extra={"dataset_name": request.dataset_name},
            )
            raise

    def _retrieve_docs(self, query: str, final_k: int) -> tuple[list[Document], list[Document], RerankResult]:
        candidate_k = max(final_k, settings.reranker_candidate_k)
        candidate_k = max(1, min(candidate_k, 30))
        candidate_docs = self.db.similarity_search(
            query,
            k=candidate_k,
            filter={"modality": "text"},
        )

        if not candidate_docs:
            return [], [], RerankResult(docs=[], applied=False, model=None)

        rerank_top_n = max(final_k, settings.reranker_top_n)
        rerank_top_n = min(rerank_top_n, len(candidate_docs))

        rerank_result = self.reranker.rerank(query, candidate_docs, top_n=rerank_top_n)
        if rerank_result.applied:
            final_docs = rerank_result.docs[:final_k]
        else:
            final_docs = candidate_docs[:final_k]

        return final_docs, candidate_docs, rerank_result

    def _add_eval_documents_once(self, docs: list[Document], id_prefix: str) -> int:
        return self._add_documents_once_to(self.eval_db, docs, id_prefix)

    def _load_eval_dataset_documents(
        self,
        request: EvalDatasetIngestRequest,
    ) -> tuple[list[Document], list[str], dict[str, int]]:
        dataset_name = request.dataset_name.lower().strip()
        self._validate_eval_dataset_name(dataset_name)

        docs: list[Document] = []
        saved_files: list[str] = []
        token_usage_total: dict[str, int] = {}
        base = settings.data_dir / "eval_dataset"

        retrieval_path = self._eval_retrieval_path(dataset_name)
        if retrieval_path:
            sample_count = 0
            sources: set[str] = set()
            retrieval_dataset_label = "zh_retrieval" if dataset_name.startswith("zh_") else "retrieval"
            for record in self._read_jsonl(retrieval_path):
                sources.update(record.get("expected_sources", []))
                sample_count += 1
                if request.sample_limit and sample_count >= request.sample_limit:
                    break

            for source in sorted(sources):
                path = self._find_eval_source_file(source)
                if not path:
                    continue
                loaded = load_one_file(path)
                for doc in loaded:
                    doc.metadata["source"] = str(path)
                    doc.metadata["filename"] = path.name
                    doc.metadata["modality"] = "text"
                    doc.metadata["collection"] = settings.mm_collection_name
                    doc.metadata["dataset_name"] = retrieval_dataset_label
                docs.extend(loaded)
                saved_files.append(path.name)

        multimodal_path = self._eval_multimodal_path(dataset_name)
        if multimodal_path:
            sample_count = 0
            multimodal_dataset_label = "zh_multimodal" if dataset_name.startswith("zh_") else "multimodal"
            for record in self._read_jsonl(multimodal_path):
                text_file = base / record["text_file"]
                if text_file.exists():
                    loaded = load_one_file(text_file)
                    for doc in loaded:
                        doc.metadata["source"] = str(text_file)
                        doc.metadata["filename"] = text_file.name
                        doc.metadata["modality"] = "text"
                        doc.metadata["collection"] = settings.mm_collection_name
                        doc.metadata["dataset_name"] = multimodal_dataset_label
                        doc.metadata["sample_id"] = record.get("sample_id")
                    docs.extend(loaded)
                    saved_files.append(text_file.name)

                image_file = base / record["image_file"]
                if request.include_images and image_file.exists():
                    parsed, token_usage = self._describe_image_with_usage(image_file)
                    self._merge_token_usage(token_usage_total, token_usage)
                    docs.append(
                        Document(
                            page_content=parsed,
                            metadata={
                                "source": str(image_file),
                                "filename": image_file.name,
                                "modality": "image",
                                "collection": settings.mm_collection_name,
                                "dataset_name": multimodal_dataset_label,
                                "sample_id": record.get("sample_id"),
                                "pipeline": "VLM OCR/semantic extraction -> Chroma unified retrieval",
                            },
                        )
                    )
                    saved_files.append(image_file.name)

                sample_count += 1
                if request.sample_limit and sample_count >= request.sample_limit:
                    break

        return docs, sorted(set(saved_files)), token_usage_total

    def _find_eval_source_file(self, filename: str) -> Path | None:
        base = settings.data_dir / "eval_dataset"
        for folder in [base / "resumes", base / "jds"]:
            path = folder / filename
            if path.exists():
                return path
        return None

    @staticmethod
    def _validate_eval_dataset_name(dataset_name: str) -> None:
        if dataset_name not in EVAL_DATASET_NAMES:
            allowed = ", ".join(sorted(EVAL_DATASET_NAMES))
            raise ValueError(f"dataset_name must be one of: {allowed}")

    @staticmethod
    def _eval_retrieval_path(dataset_name: str) -> Path | None:
        base = settings.data_dir / "eval_dataset" / "rag_queries"
        if dataset_name in {"retrieval", "all"}:
            return base / "retrieval_eval.jsonl"
        if dataset_name in {"zh_retrieval", "zh_all"}:
            return base / "zh_retrieval_eval.jsonl"
        return None

    @staticmethod
    def _eval_multimodal_path(dataset_name: str) -> Path | None:
        base = settings.data_dir / "eval_dataset"
        if dataset_name in {"multimodal", "all"}:
            return base / "multimodal" / "mrag_eval.jsonl"
        if dataset_name in {"zh_multimodal", "zh_all"}:
            return base / "multimodal_zh" / "zh_mrag_eval.jsonl"
        return None

    def _add_documents_once(self, docs: list[Document], id_prefix: str) -> int:
        return self._add_documents_once_to(self.db, docs, id_prefix)

    def _add_documents_once_to(self, db: Chroma, docs: list[Document], id_prefix: str) -> int:
        if not docs:
            return 0

        ids = [self._stable_doc_id(doc, idx, id_prefix) for idx, doc in enumerate(docs)]
        existing: set[str] = set()
        try:
            found = db.get(ids=ids)
            existing = set(found.get("ids", []))
        except Exception:
            existing = set()

        filtered_docs: list[Document] = []
        filtered_ids: list[str] = []
        for doc_id, doc in zip(ids, docs):
            if doc_id in existing:
                continue
            filtered_ids.append(doc_id)
            filtered_docs.append(doc)

        if filtered_docs:
            db.add_documents(filtered_docs, ids=filtered_ids)
        return len(filtered_docs)

    @staticmethod
    def _stable_doc_id(doc: Document, idx: int, id_prefix: str) -> str:
        source = str(doc.metadata.get("source", ""))
        filename = str(doc.metadata.get("filename", ""))
        content = doc.page_content[:500]
        digest = hashlib.md5(f"{id_prefix}|{idx}|{source}|{filename}|{content}".encode("utf-8")).hexdigest()
        return f"{id_prefix}:{digest}"

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _generate_answer(self, question: str, docs: list[Document]) -> str:
        answer, _ = self._generate_answer_with_usage(
            question=question,
            docs=docs,
            history_text="无",
            image_context="无",
        )
        return answer

    def _generate_answer_with_usage(
        self,
        *,
        question: str,
        docs: list[Document],
        history_text: str,
        image_context: str,
        rag_context: str | None = None,
    ) -> tuple[str, dict | None]:
        rag_context = rag_context or self._format_rag_context(docs)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QA_SYSTEM_PROMPT),
                ("human", QA_USER_PROMPT),
            ]
        )
        messages = prompt.format_messages(
            question=question,
            history=history_text,
            image_context=image_context,
            rag_context=rag_context,
        )
        result = self.chat_llm.invoke(messages)
        content = message_to_text(getattr(result, "content", result))
        return content, extract_token_usage(result)

    @classmethod
    def _fallback_answer(
        cls,
        *,
        question: str,
        docs: list[Document],
        image_context: str,
        error: Exception,
    ) -> str:
        reason = cls._safe_error_summary(error)
        lines = [
            "模型调用失败，已切换为本地检索摘要。",
            "",
            f"失败原因：{reason}",
            "",
            f"问题：{question}",
        ]

        if image_context and image_context not in {"无", "未提供临时图片。"}:
            lines.extend(["", "## 临时图片解析", cls._compact_text(image_context, limit=700)])

        if not docs:
            lines.extend(
                [
                    "",
                    "## 检索结果",
                    "未检索到可用知识片段。请稍后重试模型调用，或先在“知识入库”中导入文本/图片资料。",
                ]
            )
            return "\n".join(lines)

        lines.extend(["", "## 检索到的主要内容"])
        for idx, doc in enumerate(docs[:5], start=1):
            filename = doc.metadata.get("filename", "unknown")
            modality = doc.metadata.get("modality", "unknown")
            snippet = cls._compact_text(doc.page_content, limit=450)
            lines.append(f"{idx}. 来源 {filename}（{modality}）：{snippet}")

        lines.extend(
            [
                "",
                "## 建议",
                "1. 根据以上来源先核对关键事实，再重新发送问题以获取模型生成版回答。",
                "2. 如果多次出现限流，请稍后重试，或更换可用的模型/API Key。",
                "3. 需要正式输出时，请优先以 citations 中列出的原始文件为准。",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _compact_text(text: str, limit: int) -> str:
        compacted = " ".join(str(text).split())
        if len(compacted) <= limit:
            return compacted
        return f"{compacted[:limit].rstrip()}..."

    @classmethod
    def _safe_error_summary(cls, error: Exception) -> str:
        message = str(error).splitlines()[0] if str(error) else type(error).__name__
        return cls._compact_text(f"{type(error).__name__}: {message}", limit=300)

    async def _save_uploads(
        self,
        files: list[UploadFile],
        target_dir: Path,
        allowed_suffixes: set[str],
    ) -> list[Path]:
        target_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []

        for upload in files:
            filename = safe_filename(upload.filename or f"upload_{uuid.uuid4().hex}")
            suffix = Path(filename).suffix.lower()
            if suffix not in allowed_suffixes:
                raise ValueError(f"Unsupported file type: {filename}")

            if not suffix:
                raise ValueError(f"File has no extension: {filename}")

            unique_name = f"{Path(filename).stem}_{uuid.uuid4().hex[:8]}{suffix}"
            save_path = target_dir / unique_name
            content = await upload.read()
            save_path.write_bytes(content)
            saved_paths.append(save_path)

        return saved_paths

    def _describe_image(self, image_path: Path) -> str:
        text, _ = self._describe_image_with_usage(image_path)
        return text

    def _describe_image_with_usage(self, image_path: Path) -> tuple[str, dict | None]:
        try:
            image_data_uri = self._image_to_data_uri(image_path)
            message = HumanMessage(
                content=[
                    {"type": "text", "text": IMAGE_ANALYSIS_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ]
            )
            result = self.vision_llm.invoke([message])
            content = message_to_text(getattr(result, "content", result))
            return f"图片文件: {image_path.name}\n{content}", extract_token_usage(result)
        except Exception as exc:
            return (
                f"图片文件: {image_path.name}\n"
                f"图片解析失败，保留文件元信息用于检索。\n"
                f"错误信息: {type(exc).__name__}: {exc}"
            ), None

    @staticmethod
    def _image_to_data_uri(image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        mime_type = mime_type or "image/png"
        b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    @staticmethod
    def _message_to_text(content: object) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
            return "\n".join(parts).strip() or str(content)

        return str(content)

    @staticmethod
    def _format_history(history: list[ChatTurn]) -> str:
        if not history:
            return "无"

        lines: list[str] = []
        for turn in history:
            role = "用户" if turn.role.lower() == "user" else "助手"
            lines.append(f"{role}: {turn.content}")
        return "\n".join(lines)

    @staticmethod
    def _format_rag_context(docs: list[Document]) -> str:
        if not docs:
            return "未检索到可用知识，请谨慎作答。"

        lines: list[str] = []
        for idx, doc in enumerate(docs, start=1):
            filename = doc.metadata.get("filename", "unknown")
            modality = doc.metadata.get("modality", "unknown")
            lines.append(f"[Chunk {idx} | {modality} | {filename}]\n{doc.page_content}")
        return "\n\n".join(lines)

    @staticmethod
    def _extract_citations(docs: list[Document]) -> list[Citation]:
        seen: set[tuple[str, str, str]] = set()
        citations: list[Citation] = []

        for doc in docs:
            filename = str(doc.metadata.get("filename", "unknown"))
            modality = str(doc.metadata.get("modality", "unknown"))
            source = str(doc.metadata.get("source", "unknown"))
            key = (filename, modality, source)
            if key in seen:
                continue
            seen.add(key)
            citations.append(Citation(filename=filename, modality=modality, source=source))

        return citations

    def _sample_experiment_result(self, docs: list[Document], sample: EvalSample) -> EvalSampleExperimentResult:
        return EvalSampleExperimentResult(
            hit=self._has_source_hit(docs, sample.expected_sources),
            mrr=self._mrr(docs, sample.expected_sources),
            keyword_recall=self._keyword_recall(docs, sample.expected_keywords),
            retrieved_sources=self._retrieved_sources(docs),
        )

    @staticmethod
    def _retrieved_sources(docs: list[Document]) -> list[str]:
        sources: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            filename = str(doc.metadata.get("filename", "")).strip()
            if filename and filename not in seen:
                sources.append(filename)
                seen.add(filename)
        return sources

    @staticmethod
    def _merge_token_usage(total: dict[str, int], usage: dict | None) -> None:
        if not usage:
            return
        for key, value in usage.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                total[key] = total.get(key, 0) + value
            elif isinstance(value, float):
                total[key] = total.get(key, 0) + int(value)

    @staticmethod
    def _normalize_sources(sources: list[str]) -> set[str]:
        return {s.strip().lower() for s in sources if s.strip()}

    def _has_source_hit(self, docs: list[Document], expected_sources: list[str]) -> bool:
        expected = self._normalize_sources(expected_sources)
        if not expected:
            return False

        retrieved = {
            str(doc.metadata.get("filename", "")).strip().lower() for doc in docs if doc.metadata.get("filename")
        }
        return bool(retrieved & expected)

    def _mrr(self, docs: list[Document], expected_sources: list[str]) -> float:
        expected = self._normalize_sources(expected_sources)
        if not expected:
            return 0.0

        for idx, doc in enumerate(docs, start=1):
            filename = str(doc.metadata.get("filename", "")).strip().lower()
            if filename in expected:
                return 1.0 / float(idx)
        return 0.0

    @staticmethod
    def _keyword_recall(docs: list[Document], expected_keywords: list[str]) -> float:
        keywords = [kw.strip().lower() for kw in expected_keywords if kw.strip()]
        if not keywords:
            return 0.0

        context = "\n".join(doc.page_content for doc in docs).lower()
        hit_count = sum(1 for kw in keywords if kw in context)
        return hit_count / float(len(keywords))

    @staticmethod
    def _citation_hit(answer: str, expected_sources: list[str]) -> bool:
        for src in expected_sources:
            if f"[{src}]" in answer:
                return True
        return False

    @staticmethod
    def _aggregate_metrics(results: list[EvalSampleResult]) -> EvalMetrics:
        n = len(results)
        if n == 0:
            return EvalMetrics(
                sample_count=0,
                baseline_hit_rate=0.0,
                rerank_hit_rate=0.0,
                baseline_mrr=0.0,
                rerank_mrr=0.0,
                baseline_keyword_recall=0.0,
                rerank_keyword_recall=0.0,
                citation_hit_rate=None,
            )

        def avg(values: list[float]) -> float:
            return sum(values) / float(len(values))

        citation_values = [float(bool(r.citation_hit)) for r in results if r.citation_hit is not None]

        return EvalMetrics(
            sample_count=n,
            baseline_hit_rate=avg([float(r.baseline_hit) for r in results]),
            rerank_hit_rate=avg([float(r.rerank_hit) for r in results]),
            baseline_mrr=avg([r.baseline_mrr for r in results]),
            rerank_mrr=avg([r.rerank_mrr for r in results]),
            baseline_keyword_recall=avg([r.baseline_keyword_recall for r in results]),
            rerank_keyword_recall=avg([r.rerank_keyword_recall for r in results]),
            citation_hit_rate=avg(citation_values) if citation_values else None,
        )

    @staticmethod
    def _aggregate_experiment_metrics(results: list[EvalSampleResult]) -> list[EvalExperimentMetrics]:
        labels = {
            "no_rag": "No RAG",
            "vector": "Vector Retrieval",
            "vector_rerank": "Vector + Rerank",
        }
        if not results:
            return [
                EvalExperimentMetrics(name=name, label=label, hit_rate=0.0, mrr=0.0, keyword_recall=0.0)
                for name, label in labels.items()
            ]

        metrics: list[EvalExperimentMetrics] = []
        for name, label in labels.items():
            values = [result.experiments.get(name) for result in results]
            present = [value for value in values if value is not None]
            if not present:
                metrics.append(EvalExperimentMetrics(name=name, label=label, hit_rate=0.0, mrr=0.0, keyword_recall=0.0))
                continue
            count = float(len(present))
            metrics.append(
                EvalExperimentMetrics(
                    name=name,
                    label=label,
                    hit_rate=sum(float(value.hit) for value in present) / count,
                    mrr=sum(value.mrr for value in present) / count,
                    keyword_recall=sum(value.keyword_recall for value in present) / count,
                )
            )
        return metrics


@lru_cache(maxsize=1)
def get_multimodal_service() -> MultimodalAssistantService:
    return MultimodalAssistantService()
