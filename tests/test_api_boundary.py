import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


class ApiBoundaryTests(unittest.TestCase):
    def test_api_token_protects_api_routes_but_not_health(self) -> None:
        with patch.object(settings, "api_token", "deployment-secret"):
            with TestClient(app) as client:
                unauthorized = client.post(
                    "/api/agent",
                    json={"goal": "Prepare for the Arm Full Stack Data Scientist role"},
                )
                health = client.get("/health")

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(health.status_code, 200)
        self.assertIn(
            health.json()["job_retrieval_strategy"],
            {"vector", "hybrid", "hybrid_rerank"},
        )

    def test_request_size_guard_rejects_declared_oversized_body(self) -> None:
        with patch.object(settings, "max_request_bytes", 10):
            with TestClient(app) as client:
                response = client.post(
                    "/api/agent",
                    content=b"01234567890",
                    headers={"content-type": "application/json"},
                )

        self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main()
