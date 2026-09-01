from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.agent.conversation_store import ConversationNotFoundError, ConversationScopeError
from app.config import settings
from app.multimodal.schemas import (
    ChatTurn,
    EvalDatasetIngestRequest,
    EvalDatasetSamplesResponse,
    EvalRequest,
    EvalResponse,
    IngestResult,
    MultimodalChatResponse,
)
from app.multimodal.service import get_multimodal_service

router = APIRouter(prefix="/api/mm", tags=["multimodal-assistant"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "module": "multimodal-assistant"}


@router.post("/ingest/text", response_model=IngestResult)
async def ingest_text(files: list[UploadFile] = File(...)) -> IngestResult:
    service = get_multimodal_service()
    try:
        saved, chunks = await service.ingest_text_files(files)
        return IngestResult(
            saved_files=saved,
            chunks_added=chunks,
            modality="text",
            collection_name=settings.mm_collection_name,
            pipeline="Text load -> semantic chunking -> Chroma unified retrieval",
            document_count=len(saved),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/chat", response_model=MultimodalChatResponse)
async def chat(
    question: str = Form(...),
    top_k: int = Form(6),
    history_json: str = Form("[]"),
    conversation_id: str | None = Form(None),
    candidate_id: str | None = Form(None),
    image: UploadFile | None = File(None),
) -> MultimodalChatResponse:
    service = get_multimodal_service()

    try:
        history_data = json.loads(history_json)
        turns = [ChatTurn.model_validate(item) for item in history_data]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid history_json: {exc}") from exc

    try:
        return await service.chat(
            question=question,
            top_k=top_k,
            history=turns,
            image_file=image,
            conversation_id=conversation_id,
            candidate_id=candidate_id,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConversationScopeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/evaluate", response_model=EvalResponse)
def evaluate(req: EvalRequest) -> EvalResponse:
    service = get_multimodal_service()
    try:
        return service.evaluate(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/eval-dataset/samples", response_model=EvalDatasetSamplesResponse)
def eval_dataset_samples(dataset_name: str = "retrieval", sample_limit: int = 10) -> EvalDatasetSamplesResponse:
    service = get_multimodal_service()
    try:
        samples = service.load_eval_dataset_samples(dataset_name=dataset_name, sample_limit=sample_limit)
        return EvalDatasetSamplesResponse(dataset_name=dataset_name, samples=samples)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ingest/eval-dataset", response_model=IngestResult)
def ingest_eval_dataset(req: EvalDatasetIngestRequest) -> IngestResult:
    service = get_multimodal_service()
    try:
        saved, chunks, documents = service.ingest_eval_dataset(req)
        return IngestResult(
            saved_files=saved,
            chunks_added=chunks,
            modality=req.dataset_name,
            collection_name=settings.eval_collection_name,
            pipeline="Eval text + VLM image extraction -> semantic chunking -> Chroma unified retrieval",
            document_count=documents,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
