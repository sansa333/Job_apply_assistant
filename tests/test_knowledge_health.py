import json
import tempfile
import unittest
from pathlib import Path

from app.knowledge.health import CollectionHealth


class KnowledgeHealthTests(unittest.TestCase):
    def test_manifest_dimension_mismatch_is_unhealthy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "job_knowledge" / "collection_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps({"collection": "job_knowledge", "embedding_backend": "hash", "dimension": 384}),
                encoding="utf-8",
            )
            result = CollectionHealth(root).check("job_knowledge", embedding_backend="hash", embedding_dimension=1024)

        self.assertFalse(result.healthy)
        self.assertIn("dimension_mismatch", result.issues)


if __name__ == "__main__":
    unittest.main()
