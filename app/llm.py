from langchain_openai import ChatOpenAI

from app.config import settings


def _build_chat_model(model_name: str, temperature: float) -> ChatOpenAI:
    extra_body = None
    if settings.zhipu_disable_thinking and "bigmodel.cn" in settings.llm_base_url:
        extra_body = {"thinking": {"type": "disabled"}}
    return ChatOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=model_name,
        temperature=temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        max_tokens=settings.llm_max_output_tokens,
        extra_body=extra_body,
    )


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Primary text model (Zhipu GLM/OpenAI-compatible endpoint)."""
    return _build_chat_model(settings.llm_model, temperature)


def get_vision_llm(temperature: float = 0.1) -> ChatOpenAI:
    """Vision-capable model for image understanding."""
    return _build_chat_model(settings.vision_model, temperature)


def message_to_text(content: object) -> str:
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


def extract_token_usage(result: object) -> dict | None:
    usage = getattr(result, "usage_metadata", None)
    if usage:
        return dict(usage)

    metadata = getattr(result, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage") or metadata.get("usage")
    return dict(token_usage) if token_usage else None
