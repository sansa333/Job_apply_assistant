from app.main import app
from app.multimodal.reranker import RerankResult
from app.multimodal.service import MultimodalAssistantService


def test_production_image_ingestion_is_not_exposed() -> None:
    paths = {route.path for route in app.routes}

    assert "/api/mm/ingest/image" not in paths
    assert not hasattr(MultimodalAssistantService, "ingest_image_files")


def test_production_retrieval_excludes_legacy_image_documents() -> None:
    class EmptyTextOnlyStore:
        def __init__(self) -> None:
            self.filter: dict[str, str] | None = None

        def similarity_search(self, _query: str, *, k: int, filter: dict[str, str]) -> list:
            assert k > 0
            self.filter = filter
            return []

    service = object.__new__(MultimodalAssistantService)
    service.db = EmptyTextOnlyStore()
    service.reranker = object()

    docs, candidates, rerank = service._retrieve_docs("测试问题", final_k=3)

    assert docs == []
    assert candidates == []
    assert isinstance(rerank, RerankResult)
    assert service.db.filter == {"modality": "text"}
