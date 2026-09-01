import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from app.config import settings
from app.multimodal.schemas import EvalDatasetIngestRequest
from app.multimodal.service import MultimodalAssistantService


class _FakeDb:
    def __init__(self) -> None:
        self.added: list[Document] = []

    def get(self, ids):
        return {"ids": []}

    def add_documents(self, docs, ids=None):
        self.added.extend(docs)


class EvalIsolationTests(unittest.TestCase):
    def test_eval_dataset_writes_only_to_eval_demo_collection(self) -> None:
        service = object.__new__(MultimodalAssistantService)
        production_db = _FakeDb()
        eval_db = _FakeDb()
        service.db = production_db
        service.eval_db = eval_db
        service._load_eval_dataset_documents = lambda request: (
            [Document(page_content="synthetic evaluation data", metadata={"filename": "synth.md"})],
            ["synth.md"],
            {},
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(settings, "request_log_path", Path(tmp) / "log.jsonl"):
                service.ingest_eval_dataset(EvalDatasetIngestRequest(dataset_name="retrieval", include_images=False))

        self.assertEqual(production_db.added, [])
        self.assertEqual(len(eval_db.added), 1)
        self.assertEqual(eval_db.added[0].metadata["collection"], settings.eval_collection_name)


if __name__ == "__main__":
    unittest.main()
