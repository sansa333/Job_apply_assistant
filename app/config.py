from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from .env first."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = "127.0.0.1"
    app_port: int = 8000
    environment: str = "development"
    api_token: str | None = None
    cors_origins: str = "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:8501,http://localhost:8501"
    max_request_bytes: int = 10 * 1024 * 1024

    # Zhipu AI (GLM) OpenAI-compatible settings
    zai_api_key: str | None = None
    zhipu_api_key: str | None = None  # optional alias
    zhipuai_api_key: str | None = None  # optional alias
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    zhipu_model: str = "glm-4.5-flash"
    zhipu_vision_model: str = "glm-4.5v"
    zhipu_disable_thinking: bool = True
    llm_timeout_seconds: int = 90
    llm_max_retries: int = 1
    llm_max_output_tokens: int = 1200

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_vision_model: str = "gpt-4o-mini"

    # Production job retrieval should use a prepared multilingual embedding model.
    # Set EMBEDDING_BACKEND=hash only for offline/demo fallback.
    embedding_backend: str = "huggingface"
    hf_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    bge_m3_model_path: Path | None = None
    embedding_device: str = "auto"
    embedding_use_fp16: bool = False
    embedding_max_length: int = 1024
    embedding_batch_size: int = 4
    embedding_normalize: bool = True
    embedding_local_files_only: bool = False

    data_dir: Path = BASE_DIR / "data"
    source_corpus_dir: Path = BASE_DIR / "data" / "source_corpus"
    job_catalog_path: Path = BASE_DIR / "data" / "job_catalog.sqlite3"
    profile_docs_dir: Path = BASE_DIR / "data" / "profile_docs"
    jd_docs_dir: Path = BASE_DIR / "data" / "jd_docs"
    mm_text_docs_dir: Path = BASE_DIR / "data" / "mm_docs" / "text"
    mm_image_docs_dir: Path = BASE_DIR / "data" / "mm_docs" / "images"
    vector_db_dir: Path = BASE_DIR / "data" / "vector_db"
    outputs_dir: Path = BASE_DIR / "outputs"
    # Trusted project Skill root. Skills are discovered for each Agent request;
    # this directory is the production policy source and is never shadowed by
    # default.
    skills_dir: Path = BASE_DIR / "skills"
    # Optional workspace overrides are disabled by default. Enabling them also
    # requires an explicit allowlist of Skill names, so production deployments
    # keep the trusted project root as their source of authority.
    agent_workspace_skills_dir: Path | None = None
    agent_allow_workspace_skill_overrides: bool = False
    agent_approved_workspace_skills: str = ""
    agent_disabled_skills: str = ""
    # Comma-separated names of Skills that should be active for every Agent
    # request (for example a future memory-policy Skill). Domain Skills remain
    # on-demand unless explicitly configured here.
    agent_always_on_skills: str = ""
    # Agent conversations are opt-in: requests without conversation_id remain
    # stateless. The store keeps only user/assistant text and compact task state;
    # raw JD and resume fields are not copied into conversation turns.
    agent_conversation_db_path: Path = BASE_DIR / "data" / "agent_conversations.sqlite3"
    agent_recent_turns: int = 6
    agent_max_turn_chars: int = 4000
    # Context assembly starts trimming before the provider's hard limit. The
    # target ratio leaves room for retries and provider-specific message framing.
    agent_context_window_tokens: int = 16000
    agent_context_target_ratio: float = 0.65
    agent_context_output_reserve_tokens: int = 1200
    agent_context_tool_reserve_tokens: int = 2000
    agent_intent_context_tokens: int = 1500
    agent_summary_trigger_messages: int = 10
    agent_summary_keep_recent_messages: int = 6
    agent_summary_max_items: int = 8
    agent_tool_result_prompt_tokens: int = 1200
    agent_tool_result_max_chars: int = 100000
    request_log_path: Path = BASE_DIR / "data" / "request_logs" / "rag_requests.jsonl"
    output_text_encoding: str = "utf-8-sig"

    mm_collection_name: str = "multimodal_knowledge"
    candidate_collection_name: str = "candidate_profile"
    job_collection_name: str = "job_knowledge"
    eval_collection_name: str = "eval_demo"
    enable_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_candidate_k: int = 12
    reranker_top_n: int = 6
    reranker_local_files_only: bool = True

    job_retrieval_strategy: str = "hybrid_rerank"
    job_retrieval_candidate_k: int = 12
    job_retrieval_rrf_k: int = 60
    job_reranker_weight: float = 0.2

    # Domestic campus job discovery. Disabled by default for public deployments;
    # it can be enabled after the operator reviews the configured official sources.
    domestic_sync_enabled: bool = False
    domestic_sync_interval_minutes: int = 360
    domestic_sync_build_index: bool = True

    @property
    def llm_api_key(self) -> str:
        key = (
            self.zai_api_key
            or self.zhipu_api_key
            or self.zhipuai_api_key
            or self.openai_api_key
        )
        if not key:
            raise RuntimeError(
                "No LLM API key found. Set ZAI_API_KEY (or ZHIPU_API_KEY) in .env."
            )
        return key

    @property
    def llm_base_url(self) -> str:
        if self.zai_api_key or self.zhipu_api_key or self.zhipuai_api_key:
            return self.zhipu_base_url
        if self.openai_api_key:
            return self.openai_base_url
        return self.zhipu_base_url

    @property
    def llm_model(self) -> str:
        if self.zai_api_key or self.zhipu_api_key or self.zhipuai_api_key:
            return self.zhipu_model
        if self.openai_api_key:
            return self.openai_model
        return self.zhipu_model

    @property
    def vision_model(self) -> str:
        if self.zai_api_key or self.zhipu_api_key or self.zhipuai_api_key:
            return self.zhipu_vision_model
        if self.openai_api_key:
            return self.openai_vision_model
        return self.zhipu_vision_model

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()


def ensure_dirs() -> None:
    for path in [
        settings.data_dir,
        settings.source_corpus_dir,
        settings.profile_docs_dir,
        settings.jd_docs_dir,
        settings.mm_text_docs_dir,
        settings.mm_image_docs_dir,
        settings.vector_db_dir,
        settings.outputs_dir,
        settings.agent_conversation_db_path.parent,
        settings.request_log_path.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)
