"""Deterministic, zero-download defaults for the test suite."""

from __future__ import annotations

import os


# These variables are set before test modules import the global application
# settings. Individual tests can still override them with monkeypatch.
os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["ENABLE_RERANKER"] = "false"
os.environ["EMBEDDING_LOCAL_FILES_ONLY"] = "true"

# Do not let a developer's shell or .env credentials turn offline tests into
# provider calls. Tests that exercise client construction supply their own key.
for name in ("ZAI_API_KEY", "ZHIPU_API_KEY", "ZHIPUAI_API_KEY", "OPENAI_API_KEY"):
    os.environ[name] = ""
