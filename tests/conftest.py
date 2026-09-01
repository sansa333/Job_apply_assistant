"""Deterministic, zero-download defaults for the test suite."""

from __future__ import annotations

import os


# These variables are set before test modules import the global application
# settings. Individual tests can still override them with monkeypatch.
os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["ENABLE_RERANKER"] = "false"
os.environ["EMBEDDING_LOCAL_FILES_ONLY"] = "true"
