from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.agent.conversation_store import (
    ConversationNotFoundError,
    ConversationScopeError,
    ConversationStore,
)
from app.agent.job_agent import run_job_agent
from app.config import ensure_dirs, settings
from app.domestic.scheduler import start_domestic_scheduler, stop_domestic_scheduler
from app.multimodal.routes import router as multimodal_router
from app.routes.domestic import router as domestic_router
from app.routes.knowledge import router as knowledge_router
from app.schemas import (
    AgentRequest,
    AgentResponse,
    ConversationCreateRequest,
    ConversationDetail,
    ConversationSummary,
    FitRequest,
    FitResponse,
    IngestResponse,
    OneClickApplyRequest,
    OneClickApplyResponse,
    ToolResultDetail,
)
from app.security import ApiBoundaryMiddleware
from app.services.application_service import ApplicationService, get_application_service
from app.services.document_service import ingest_jd_files, ingest_profile_files

ensure_dirs()


@asynccontextmanager
async def lifespan(_: FastAPI):
    start_domestic_scheduler()
    try:
        yield
    finally:
        stop_domestic_scheduler()

app = FastAPI(
    title="AI Job Apply + Multimodal Assistant",
    description="LLM + LangChain + RAG + Agent + Multimodal",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(ApiBoundaryMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (static_dir / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": settings.llm_model,
        "vision_model": settings.vision_model,
        "reranker_enabled": settings.enable_reranker,
        "reranker_model": settings.reranker_model,
        "reranker_local_files_only": settings.reranker_local_files_only,
        "base_url": settings.llm_base_url,
        "embedding_backend": settings.embedding_backend,
        "embedding_model": settings.hf_embedding_model,
        "embedding_device": settings.embedding_device,
        "embedding_use_fp16": settings.embedding_use_fp16,
        "embedding_max_length": settings.embedding_max_length,
        "embedding_batch_size": settings.embedding_batch_size,
        "embedding_local_files_only": settings.embedding_local_files_only,
        "job_retrieval_strategy": settings.job_retrieval_strategy,
        "job_retrieval_candidate_k": settings.job_retrieval_candidate_k,
        "job_retrieval_rrf_k": settings.job_retrieval_rrf_k,
        "agent_context_window_tokens": settings.agent_context_window_tokens,
        "agent_context_target_ratio": settings.agent_context_target_ratio,
        "agent_summary_trigger_messages": settings.agent_summary_trigger_messages,
        "agent_summary_keep_recent_messages": settings.agent_summary_keep_recent_messages,
    }


@app.get("/ready")
def ready() -> JSONResponse:
    checks = {
        "data_dir": settings.data_dir.exists(),
        "outputs_dir": settings.outputs_dir.exists(),
        "job_catalog": settings.job_catalog_path.exists(),
        "llm_credentials": bool(
            settings.zai_api_key
            or settings.zhipu_api_key
            or settings.zhipuai_api_key
            or settings.openai_api_key
        ),
    }
    ready_status = all(checks.values())
    return JSONResponse(
        status_code=200 if ready_status else 503,
        content={"status": "ready" if ready_status else "not_ready", "checks": checks},
    )


@app.post("/api/ingest/profile", response_model=IngestResponse)
async def ingest_profile(files: list[UploadFile] = File(...)) -> IngestResponse:
    saved, chunks = await ingest_profile_files(files)
    return IngestResponse(saved_files=saved, chunks_added=chunks)


@app.post("/api/ingest/job", response_model=IngestResponse)
async def ingest_job(files: list[UploadFile] = File(...)) -> IngestResponse:
    saved, chunks = await ingest_jd_files(files)
    return IngestResponse(saved_files=saved, chunks_added=chunks)


@app.post("/api/analyze", response_model=FitResponse)
def analyze(
    req: FitRequest,
    service: ApplicationService = Depends(get_application_service),
) -> FitResponse:
    result = service.analyze_scoped_fit(req)
    return FitResponse(**result)


@app.post("/api/fit", response_model=FitResponse)
def fit(
    req: FitRequest,
    service: ApplicationService = Depends(get_application_service),
) -> FitResponse:
    return FitResponse(**service.analyze_scoped_fit(req))


@app.post("/api/one-click-apply", response_model=OneClickApplyResponse)
def one_click_apply(
    req: OneClickApplyRequest,
    service: ApplicationService = Depends(get_application_service),
) -> OneClickApplyResponse:
    result = service.one_click_apply(req)
    return OneClickApplyResponse(**result)


@app.post("/api/agent", response_model=AgentResponse)
def agent(req: AgentRequest) -> AgentResponse:
    try:
        return AgentResponse(**run_job_agent(req))
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConversationScopeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _conversation_store() -> ConversationStore:
    return ConversationStore(settings.agent_conversation_db_path)


@app.post("/api/conversations", response_model=ConversationSummary, status_code=201)
def create_conversation(req: ConversationCreateRequest) -> ConversationSummary:
    return ConversationSummary(**_conversation_store().create(req))


@app.get("/api/conversations", response_model=list[ConversationSummary])
def list_conversations(
    candidate_id: str = Query(..., min_length=1),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[ConversationSummary]:
    return [
        ConversationSummary(**item)
        for item in _conversation_store().list(candidate_id, limit=limit)
    ]


@app.get("/api/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: str,
    candidate_id: str = Query(..., min_length=1),
) -> ConversationDetail:
    try:
        return ConversationDetail(
            **_conversation_store().detail(conversation_id, candidate_id)
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConversationScopeError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    candidate_id: str = Query(..., min_length=1),
) -> dict[str, str]:
    try:
        _conversation_store().delete(conversation_id, candidate_id)
    except (ConversationNotFoundError, ConversationScopeError) as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    return {"status": "deleted", "conversation_id": conversation_id}


@app.get("/api/conversations/{conversation_id}/tool-results")
def list_conversation_tool_results(
    conversation_id: str,
    candidate_id: str = Query(..., min_length=1),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[dict]:
    try:
        return _conversation_store().list_tool_results(
            conversation_id,
            candidate_id,
            limit=limit,
        )
    except (ConversationNotFoundError, ConversationScopeError) as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc


@app.get(
    "/api/conversations/{conversation_id}/tool-results/{tool_result_id}",
    response_model=ToolResultDetail,
)
def get_conversation_tool_result(
    conversation_id: str,
    tool_result_id: str,
    candidate_id: str = Query(..., min_length=1),
) -> ToolResultDetail:
    store = _conversation_store()
    try:
        store.get(conversation_id, candidate_id)
        result = store.get_tool_result(tool_result_id, candidate_id)
        if result["conversation_id"] != conversation_id:
            raise ConversationNotFoundError("Tool result not found")
        return ToolResultDetail(**result)
    except (ConversationNotFoundError, ConversationScopeError) as exc:
        raise HTTPException(status_code=404, detail="Tool result not found") from exc


@app.delete("/api/conversations/{conversation_id}/tool-results/{tool_result_id}")
def delete_conversation_tool_result(
    conversation_id: str,
    tool_result_id: str,
    candidate_id: str = Query(..., min_length=1),
) -> dict[str, str]:
    store = _conversation_store()
    try:
        store.get(conversation_id, candidate_id)
        result = store.get_tool_result(tool_result_id, candidate_id)
        if result["conversation_id"] != conversation_id:
            raise ConversationNotFoundError("Tool result not found")
        store.delete_tool_result(tool_result_id, candidate_id)
    except (ConversationNotFoundError, ConversationScopeError) as exc:
        raise HTTPException(status_code=404, detail="Tool result not found") from exc
    return {"status": "deleted", "tool_result_id": tool_result_id}


app.include_router(multimodal_router)
app.include_router(knowledge_router)
app.include_router(domestic_router)
