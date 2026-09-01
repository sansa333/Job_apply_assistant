import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.knowledge.profiles import CandidateProfileStore
from app.main import app


class CandidateProfileTests(unittest.TestCase):
    def test_profile_upload_api_requires_candidate_id(self) -> None:
        with TestClient(app) as client:
            response = client.post("/api/candidates/upload", data={"profile_text": "Python RAG"})

        self.assertEqual(response.status_code, 422)

    def test_profile_is_retrieved_only_for_the_requested_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles = CandidateProfileStore(root / "source_corpus", root / "vectors")
            profiles.ingest_text("candidate-a", "Alice has Python RAG and Chroma project experience.", "alice.md")
            profiles.ingest_text("candidate-b", "Bob has Spark data engineering experience.", "bob.md")
            docs = profiles.retrieve("candidate-a", "Python RAG", k=5)
            profiles.close()

        self.assertTrue(docs)
        self.assertTrue(all(doc.metadata["candidate_id"] == "candidate-a" for doc in docs))
        self.assertTrue(all(doc.metadata["scope"] == "profile" for doc in docs))

    def test_reupload_replaces_previous_candidate_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles = CandidateProfileStore(root / "source_corpus", root / "vectors")
            profiles.ingest_text("candidate-a", "Legacy Java profile evidence.", "old.md")
            profiles.ingest_text("candidate-a", "Current Python RAG profile evidence.", "current.md")
            result = profiles.db.get(where={"candidate_id": "candidate-a"}, include=["documents"])
            profiles.close()

        documents = result.get("documents", [])
        self.assertTrue(documents)
        self.assertTrue(all("Current Python RAG" in document for document in documents))
        self.assertTrue(all("Legacy Java" not in document for document in documents))


if __name__ == "__main__":
    unittest.main()
