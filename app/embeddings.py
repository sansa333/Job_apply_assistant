from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any

import numpy as np
from langchain_core.embeddings import Embeddings

from app.config import settings


class HashEmbeddings(Embeddings):
    """Zero-download fallback embedding for offline demos."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        normalized = text.lower().replace("\n", " ")
        chars = [char for char in normalized if not char.isspace()]
        tokens = chars + ["".join(chars[index : index + 2]) for index in range(max(0, len(chars) - 1))]
        tokens.extend(normalized.split())
        if not tokens:
            return vec.tolist()
        for token in tokens:
            digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[digest % self.dim] += 1.0 if (digest >> 8) % 2 == 0 else -1.0
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class SentenceTransformerEmbeddings(Embeddings):
    """Reproducible SentenceTransformer adapter with explicit runtime settings."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        use_fp16: bool,
        max_length: int,
        batch_size: int,
        normalize_embeddings: bool,
        local_files_only: bool,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        resolved_device = device.lower()
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "EMBEDDING_DEVICE=cuda was requested, but the installed PyTorch build "
                "cannot access CUDA. Install the CUDA PyTorch wheel from requirements-gpu.txt."
            )
        if use_fp16 and resolved_device != "cuda":
            raise RuntimeError("EMBEDDING_USE_FP16=true requires EMBEDDING_DEVICE=cuda.")
        if max_length <= 0 or batch_size <= 0:
            raise ValueError("Embedding max_length and batch_size must be positive integers.")

        transformer_kwargs: dict[str, Any] = {}
        if use_fp16:
            transformer_kwargs["torch_dtype"] = torch.float16

        self.model_name = model_name
        self.device = resolved_device
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model = SentenceTransformer(
            model_name,
            device=resolved_device,
            local_files_only=local_files_only,
            model_kwargs=transformer_kwargs,
        )
        self._model.max_seq_length = max_length

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32, copy=False).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]


@lru_cache(maxsize=1)
def _configured_sentence_transformer(
    model_name: str,
    device: str,
    use_fp16: bool,
    max_length: int,
    batch_size: int,
    normalize_embeddings: bool,
    local_files_only: bool,
) -> SentenceTransformerEmbeddings:
    """Load the configured production model once instead of duplicating GPU weights."""
    return SentenceTransformerEmbeddings(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16,
        max_length=max_length,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        local_files_only=local_files_only,
    )


def get_embeddings(
    *,
    backend: str | None = None,
    model_name: str | None = None,
    local_files_only: bool | None = None,
    device: str | None = None,
    use_fp16: bool | None = None,
    max_length: int | None = None,
    batch_size: int | None = None,
) -> Embeddings:
    selected_backend = (backend or settings.embedding_backend).lower()
    if selected_backend == "huggingface":
        resolved = (
            model_name or settings.hf_embedding_model,
            device or settings.embedding_device,
            settings.embedding_use_fp16 if use_fp16 is None else use_fp16,
            max_length or settings.embedding_max_length,
            batch_size or settings.embedding_batch_size,
            settings.embedding_normalize,
            settings.embedding_local_files_only
            if local_files_only is None
            else local_files_only,
        )
        configured_request = all(
            value is None
            for value in (backend, model_name, local_files_only, device, use_fp16, max_length, batch_size)
        )
        if configured_request:
            return _configured_sentence_transformer(*resolved)
        return SentenceTransformerEmbeddings(
            model_name=resolved[0],
            device=resolved[1],
            use_fp16=resolved[2],
            max_length=resolved[3],
            batch_size=resolved[4],
            normalize_embeddings=resolved[5],
            local_files_only=resolved[6],
        )
    return HashEmbeddings()
