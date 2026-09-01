from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

from app.chunking import split_documents_semantic
from app.config import settings
from app.embeddings import get_embeddings


def load_one_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()

    if suffix in {".docx", ".doc"}:
        return Docx2txtLoader(str(path)).load()

    if suffix in {".txt", ".md", ".csv"}:
        return TextLoader(str(path), encoding="utf-8").load()

    raise ValueError(f"Unsupported file type: {path.name}")


def split_documents(docs: list[Document], collection_name: str | None = None) -> list[Document]:
    return split_documents_semantic(docs, collection_name=collection_name)


class RAGStore:
    """Thin wrapper around Chroma vector store."""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.persist_dir = settings.vector_db_dir / collection_name
        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),
            persist_directory=str(self.persist_dir),
        )

    def ingest_files(self, paths: Iterable[Path]) -> int:
        docs: list[Document] = []

        for path in paths:
            loaded = load_one_file(path)
            for doc in loaded:
                doc.metadata["source"] = str(path)
                doc.metadata["filename"] = path.name
                doc.metadata["collection"] = self.collection_name
            docs.extend(loaded)

        if not docs:
            return 0

        chunks = split_documents(docs, collection_name=self.collection_name)
        self.db.add_documents(chunks)
        return len(chunks)

    def ingest_folder(self, folder: Path) -> int:
        supported = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv"}
        paths = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in supported]
        if not paths:
            return 0
        return self.ingest_files(paths)

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        return self.db.similarity_search(query, k=k)

    def as_context(self, query: str, k: int = 5) -> str:
        docs = self.retrieve(query, k=k)
        if not docs:
            return "未检索到相关资料。"

        blocks: list[str] = []
        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("filename") or doc.metadata.get("source", "unknown")
            blocks.append(f"[资料{idx} | {source}]\n{doc.page_content}")
        return "\n\n".join(blocks)

    def close(self) -> None:
        close = getattr(getattr(self.db, "_client", None), "close", None)
        if callable(close):
            close()
