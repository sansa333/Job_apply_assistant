import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from app.multimodal.reranker import RerankResult
from app.multimodal.schemas import ChatTurn, EvalSampleExperimentResult, EvalSampleResult
from app.multimodal.service import MultimodalAssistantService
from app.agent.conversation_store import ConversationStore
from app.schemas import ConversationCreateRequest
from app.utils.file_io import write_json, write_text
from app.utils.request_log import _redact_secrets, log_request_event


class EvalObservabilityTests(unittest.TestCase):
    def test_aggregate_experiment_metrics_handles_three_variants(self) -> None:
        results = [
            EvalSampleResult(
                query="q1",
                experiments={
                    "no_rag": EvalSampleExperimentResult(hit=False, mrr=0.0, keyword_recall=0.0),
                    "vector": EvalSampleExperimentResult(hit=True, mrr=0.5, keyword_recall=0.25),
                    "vector_rerank": EvalSampleExperimentResult(hit=True, mrr=1.0, keyword_recall=0.75),
                },
                baseline_hit=True,
                rerank_hit=True,
                baseline_mrr=0.5,
                rerank_mrr=1.0,
                baseline_keyword_recall=0.25,
                rerank_keyword_recall=0.75,
            ),
            EvalSampleResult(
                query="q2",
                experiments={
                    "no_rag": EvalSampleExperimentResult(hit=False, mrr=0.0, keyword_recall=0.0),
                    "vector": EvalSampleExperimentResult(hit=False, mrr=0.0, keyword_recall=0.5),
                    "vector_rerank": EvalSampleExperimentResult(hit=True, mrr=0.5, keyword_recall=1.0),
                },
                baseline_hit=False,
                rerank_hit=True,
                baseline_mrr=0.0,
                rerank_mrr=0.5,
                baseline_keyword_recall=0.5,
                rerank_keyword_recall=1.0,
            ),
        ]

        metrics = {item.name: item for item in MultimodalAssistantService._aggregate_experiment_metrics(results)}

        self.assertEqual(metrics["no_rag"].hit_rate, 0.0)
        self.assertEqual(metrics["vector"].hit_rate, 0.5)
        self.assertEqual(metrics["vector_rerank"].hit_rate, 1.0)
        self.assertEqual(metrics["vector_rerank"].keyword_recall, 0.875)

    def test_write_text_uses_bom_and_json_stays_standard_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text_path = Path(tmp) / "report.md"
            json_path = Path(tmp) / "submission.json"

            write_text(text_path, "中文报告")
            write_json(json_path, {"message": "中文"})

            self.assertTrue(text_path.read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertFalse(json_path.read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["message"], "中文")

    def test_log_redacts_configured_secret(self) -> None:
        from app.config import settings

        secret = settings.zai_api_key or settings.zhipu_api_key or settings.zhipuai_api_key or settings.openai_api_key
        event = _redact_secrets({"token": f"prefix-{secret}-suffix" if secret else "prefix"})

        if secret:
            self.assertEqual(event["token"], "prefix-<redacted>-suffix")
        else:
            self.assertEqual(event["token"], "prefix")

    def test_log_request_event_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from app.config import settings

            original = settings.request_log_path
            settings.request_log_path = Path(tmp) / "rag_requests.jsonl"
            try:
                log_request_event(
                    route="/api/mm/evaluate",
                    request_id="test",
                    top_k=6,
                    candidate_count=12,
                    rerank_enabled=True,
                    elapsed_ms_value=10,
                )
                lines = settings.request_log_path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(lines), 1)
                payload = json.loads(lines[0])
                self.assertEqual(payload["route"], "/api/mm/evaluate")
                self.assertEqual(payload["top_k"], 6)
            finally:
                settings.request_log_path = original

    def test_chat_returns_fallback_answer_when_llm_is_rate_limited(self) -> None:
        from app.config import settings

        with tempfile.TemporaryDirectory() as tmp:
            original = settings.request_log_path
            settings.request_log_path = Path(tmp) / "rag_requests.jsonl"
            try:
                service = object.__new__(MultimodalAssistantService)
                service.reranker = type("FakeReranker", (), {"enabled": True})()
                doc = Document(
                    page_content="RAG 系统使用 LangChain 和 Chroma 做统一检索，并要求答案给出来源。",
                    metadata={
                        "filename": "guide.md",
                        "modality": "text",
                        "source": "data/guide.md",
                    },
                )
                service._retrieve_docs = lambda question, final_k: (
                    [doc],
                    [doc],
                    RerankResult(docs=[doc], applied=False, model=None),
                )

                def raise_rate_limit(**kwargs):
                    raise RuntimeError("rate limit exceeded")

                service._generate_answer_with_usage = raise_rate_limit

                result = asyncio.run(
                    service.chat(
                        question="请总结知识库核心内容",
                        top_k=5,
                        history=[],
                    )
                )

                self.assertIn("模型调用失败", result.answer)
                self.assertIn("RAG 系统使用 LangChain 和 Chroma", result.answer)
                self.assertEqual(result.retrieved_chunks, 1)
                self.assertEqual(result.citations[0].filename, "guide.md")
            finally:
                settings.request_log_path = original

    def test_chat_uses_server_conversation_instead_of_client_history(self) -> None:
        from app.config import settings

        with tempfile.TemporaryDirectory() as tmp:
            original_log = settings.request_log_path
            original_db = settings.agent_conversation_db_path
            settings.request_log_path = Path(tmp) / "rag_requests.jsonl"
            settings.agent_conversation_db_path = Path(tmp) / "conversations.sqlite3"
            try:
                store = ConversationStore(settings.agent_conversation_db_path)
                conversation = store.create(
                    ConversationCreateRequest(
                        candidate_id="candidate-a",
                        conversation_type="knowledge_chat",
                    )
                )
                service = object.__new__(MultimodalAssistantService)
                service.reranker = type("FakeReranker", (), {"enabled": False})()
                doc = Document(
                    page_content="服务端知识证据" * 100,
                    metadata={"filename": "guide.md", "modality": "text", "source": "guide.md"},
                )
                service._retrieve_docs = lambda question, final_k: (
                    [doc],
                    [doc],
                    RerankResult(docs=[doc], applied=False, model=None),
                )
                captured = {}

                def generate(**kwargs):
                    captured.update(kwargs)
                    return "服务端会话回答", {"input_tokens": 10, "output_tokens": 5}

                service._generate_answer_with_usage = generate
                with (
                    patch.object(settings, "agent_summary_trigger_messages", 1),
                    patch.object(settings, "agent_summary_keep_recent_messages", 1),
                ):
                    result = asyncio.run(
                        service.chat(
                            question="当前问题",
                            history=[ChatTurn(role="user", content="CLIENT HISTORY MUST BE IGNORED")],
                            conversation_id=conversation["conversation_id"],
                            candidate_id="candidate-a",
                        )
                    )

                self.assertEqual(result.conversation_id, conversation["conversation_id"])
                self.assertNotIn("CLIENT HISTORY MUST BE IGNORED", captured["history_text"])
                self.assertEqual(result.context_usage["budget_status"], "within_budget")
                self.assertTrue(result.context_usage["rolling_summary_updated"])
                detail = store.detail(conversation["conversation_id"], "candidate-a")
                self.assertEqual(len(detail["turns"]), 2)
                self.assertEqual(detail["turns"][0]["content"], "当前问题")
                self.assertEqual(result.conversation_summary["summarized_message_count"], 1)
            finally:
                settings.request_log_path = original_log
                settings.agent_conversation_db_path = original_db


if __name__ == "__main__":
    unittest.main()
