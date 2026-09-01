from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.chunking import split_documents_semantic
from app.embeddings import get_embeddings
from app.knowledge.chroma_lifecycle import close_chroma
from app.utils.file_io import validate_identifier


class CandidateProfileStore:
    collection_name = "candidate_profile"

    def __init__(
        self,
        source_corpus_dir: Path,
        vector_db_dir: Path,
        collection_name: str | None = None,
    ):
        self.source_corpus_dir = source_corpus_dir
        self.collection_name = collection_name or type(self).collection_name
        self.persist_dir = vector_db_dir / self.collection_name
        self.db = Chroma(
            collection_name=self.collection_name,
            embedding_function=get_embeddings(),
            persist_directory=str(self.persist_dir),
        )

    def ingest_text(self, candidate_id: str, text: str, filename: str = "profile.md") -> int:
        candidate_id = validate_identifier(candidate_id, field_name="candidate_id")
        if not text.strip():
            raise ValueError("profile content is required")
        target = self.source_corpus_dir / "candidate_profiles" / candidate_id / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        document = Document(
            page_content=text,
            metadata={
                "source": str(target),
                "filename": filename,
                "collection": self.collection_name,
                "candidate_id": candidate_id,
                "scope": "profile",
                "source_kind": "user_upload",
                "doc_type": "resume",
            },
        )
        chunks = split_documents_semantic([document], collection_name="profile") or [document]
        ids: list[str] = []
        for index, chunk in enumerate(chunks):
            chunk.metadata.update(document.metadata)
            chunk.metadata["chunk_id"] = f"profile:{candidate_id}:{index}"
            chunk.metadata["chunk_index"] = index
            ids.append(f"profile:{candidate_id}:{index}")
        self.db.delete(where={"candidate_id": candidate_id})
        self.db.add_documents(chunks, ids=ids)
        return len(chunks)

    def retrieve(self, candidate_id: str, query: str, *, k: int = 5) -> list[Document]:
        candidate_id = validate_identifier(candidate_id, field_name="candidate_id")
        return self.db.similarity_search(query, k=max(1, k), filter={"candidate_id": candidate_id})

    def close(self) -> None:
        db = getattr(self, "db", None)
        close_chroma(db)
        self.db = None
