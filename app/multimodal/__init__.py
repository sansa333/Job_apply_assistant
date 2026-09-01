__all__ = ["MultimodalAssistantService", "get_multimodal_service"]


def __getattr__(name: str):
    if name in __all__:
        from app.multimodal.service import MultimodalAssistantService, get_multimodal_service

        values = {
            "MultimodalAssistantService": MultimodalAssistantService,
            "get_multimodal_service": get_multimodal_service,
        }
        return values[name]
    raise AttributeError(name)
