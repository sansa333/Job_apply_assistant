from __future__ import annotations

import json
import gc
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.chunking import JD_CHUNK_SCHEMA_VERSION, is_heading_only_job_chunk, split_documents_semantic
from app.config import settings
from app.knowledge.catalog import JobCatalog
from app.knowledge.chroma_lifecycle import close_chroma
from app.knowledge.importers import KyosekCsvAdapter, ProjectMarkdownAdapter
from app.knowledge.models import JobRecord, NormalizedJob
from app.embeddings import get_embeddings


@dataclass(frozen=True)
class JobIngestionResult:
    record: JobRecord
    inserted: bool
    updated: bool
    chunks_added: int


class StaleJobIndexError(RuntimeError):
    """An explicit offline rebuild is required before serving this index."""


class JobKnowledgeIngestion:
    collection_name = "job_knowledge"

    def __init__(
        self,
        *,
        catalog: JobCatalog,
        source_corpus_dir: Path,
        vector_db_dir: Path,
        collection_name: str | None = None,
        embeddings: Embeddings | None = None,
        embedding_backend: str | None = None,
        embedding_model: str | None = None,
    ):
        self.catalog = catalog
        self.source_corpus_dir = source_corpus_dir
        self.collection_name = collection_name or type(self).collection_name
        self.embeddings = embeddings or get_embeddings(backend=embedding_backend, model_name=embedding_model)
        self.embedding_backend = (embedding_backend or settings.embedding_backend).lower()
        self.embedding_model = embedding_model or (
            settings.hf_embedding_model if self.embedding_backend == "huggingface" else "hash"
        )
        self.vector_db_dir = vector_db_dir.resolve()
        self.persist_dir = self.vector_db_dir / self.collection_name
        self.db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )
        self.expected_manifest = self._manifest_identity()
        self.indexed_job_hashes = {}
        manifest_path = self.persist_dir / "collection_manifest.json"
        if manifest_path.exists():
            try:
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                previous = {}
            self.indexed_job_hashes = previous.get("indexed_job_hashes", {})
            self.index_needs_rebuild = any(previous.get(key) != value for key, value in self.expected_manifest.items())
            self.index_needs_rebuild |= previous.get("state") != "ready"
        elif self.db._collection.count():
            self.index_needs_rebuild = True
        else:
            self.index_needs_rebuild = False
            self._write_manifest()

    def _manifest_identity(self) -> dict:
        dimension = int(getattr(self.embeddings, "dim", 0) or len(self.embeddings.embed_query("dimension probe")))
        return {
            "collection": self.collection_name,
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
            "dimension": dimension,
            "chunk_schema_version": JD_CHUNK_SCHEMA_VERSION,
            "max_length": getattr(self.embeddings, "max_length", None),
            "normalize_embeddings": getattr(self.embeddings, "normalize_embeddings", None),
            "use_fp16": getattr(self.embeddings, "use_fp16", None),
        }

    def _assert_current_index(self) -> None:
        if self.index_needs_rebuild:
            raise StaleJobIndexError("岗位索引版本或模型配置不一致；请停服后运行 python -m tools.rebuild_job_knowledge 重建索引。")

    def _write_manifest(self, *, state: str = "ready") -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        (self.persist_dir / "collection_manifest.json").write_text(
            json.dumps(
                {**self.expected_manifest, "state": state, "document_count": self.db._collection.count(),
                 "indexed_job_hashes": self.indexed_job_hashes},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def ingest(self, job: NormalizedJob, *, original_bytes: bytes | None = None) -> JobIngestionResult:
        self._assert_current_index()
        try:
            return self._ingest(job, original_bytes=original_bytes)
        except Exception:
            self.index_needs_rebuild = True
            self._write_manifest(state="error")
            raise

    def _ingest(self, job: NormalizedJob, *, original_bytes: bytes | None = None) -> JobIngestionResult:
        catalog_result = self.catalog.upsert(job)
        record = catalog_result.record
        if not catalog_result.inserted and not catalog_result.updated:
            return JobIngestionResult(record=record, inserted=False, updated=False, chunks_added=0)

        self._persist_original(record, original_bytes or job.description.encode("utf-8"))
        chunks = self._chunks_for_record(record)
        ids = [f"job:{record.job_id}:{index}" for index in range(len(chunks))]
        if catalog_result.updated:
            self.db.delete(where={"job_id": record.job_id})
        if chunks:
            self.db.add_documents(chunks, ids=ids)
            self.indexed_job_hashes[record.job_id] = record.content_hash
        self._write_manifest()
        return JobIngestionResult(
            record=record,
            inserted=catalog_result.inserted,
            updated=catalog_result.updated,
            chunks_added=len(chunks),
        )

    def _persist_original(self, record: JobRecord, content: bytes) -> Path:
        if record.is_user_uploaded:
            path = self.source_corpus_dir / "user_jobs" / record.job_id / record.source_file
        else:
            path = self.source_corpus_dir / "open_source_jobs" / record.source_file
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or record.source_kind != "user_upload":
            path.write_bytes(content)
        return path

    def _chunks_for_record(self, record: JobRecord) -> list[Document]:
        if record.is_user_uploaded:
            source_path = self.source_corpus_dir / "user_jobs" / record.job_id / record.source_file
        else:
            source_path = self.source_corpus_dir / "open_source_jobs" / record.source_file
        source = str(source_path)
        document = Document(
            page_content=record.description,
            metadata={
                "source": source,
                "filename": record.source_file,
                "collection": self.collection_name,
                "doc_type": "job_description",
                "job_id": record.job_id,
                "company_name": record.company_name,
                "company_key": record.company_key,
                "job_title": record.job_title,
                "job_title_key": record.job_title_key,
                "source_kind": record.source_kind,
                "source_dataset": record.source_dataset,
                "language": record.language,
                "scope": "job",
            },
        )
        chunks = split_documents_semantic([document], collection_name="job_description")
        if not chunks and not is_heading_only_job_chunk(document.page_content):
            chunks = [document]
        for index, chunk in enumerate(chunks):
            chunk.metadata.update(document.metadata)
            chunk.metadata["chunk_id"] = f"job:{record.job_id}:{index}"
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_schema_version"] = JD_CHUNK_SCHEMA_VERSION
            chunk.metadata["evidence_kind"] = (
                "derived_keywords" if chunk.metadata.get("section_title") == "技术栈关键词"
                else "source_text"
            )
        return chunks

    def retrieve_vector_for_job(self, job_id: str, query: str, *, k: int = 5) -> list[Document]:
        self._assert_current_index()
        return self.db.similarity_search(query, k=max(1, k), filter={"job_id": job_id})

    def retrieve_for_job(self, job_id: str, query: str, *, k: int = 5) -> list[Document]:
        """Compatibility alias for vector-only retrieval callers."""
        return self.retrieve_vector_for_job(job_id, query, k=k)

    def retrieve_across_jobs(self, query: str, job_ids: list[str], *, k: int = 100) -> list[Document]:
        """Global evidence candidate recall, constrained only by explicit job filters."""
        self._assert_current_index()
        if not job_ids:
            return []
        return self.db.similarity_search(query, k=max(1, k), filter={"job_id": {"$in": job_ids}})

    def scored_source_evidence(self, job_id: str, query: str, *, k: int = 3) -> list[tuple[Document, float]]:
        """Cosine from normalized dense vectors; derived keyword lists are not answer evidence."""
        self._assert_current_index()
        if not getattr(self.embeddings, "normalize_embeddings", False):
            raise ValueError("Evidence sufficiency requires normalized semantic embeddings")
        pairs = self.db.similarity_search_with_score(query, k=max(1, k), filter={"$and": [
            {"job_id": job_id}, {"evidence_kind": "source_text"},
        ]})
        # This collection uses Chroma's default squared L2 distance.
        return [(doc, max(-1.0, min(1.0, 1 - float(distance) / 2))) for doc, distance in pairs]

    def get_documents_for_job(self, job_id: str) -> list[Document]:
        """Return every indexed chunk for a job, ordered by stable chunk index."""
        self._assert_current_index()
        result = self.db.get(where={"job_id": job_id}, include=["documents", "metadatas"])
        documents = [
            Document(page_content=content, metadata=metadata or {})
            for content, metadata in zip(result.get("documents", []), result.get("metadatas", []))
        ]
        return sorted(documents, key=lambda document: int(document.metadata.get("chunk_index", 0)))

    def close(self) -> None:
        """Release local Chroma file handles (important for Windows rebuilds/tests)."""
        db = getattr(self, "db", None)
        close_chroma(db)
        self.db = None
        del db
        gc.collect()

    def _remove_index_dir(self) -> None:
        self.close()
        if not self.persist_dir.exists():
            return
        if self.persist_dir.resolve().parent != self.vector_db_dir or self.persist_dir.resolve() == self.vector_db_dir:
            raise ValueError("Refusing to remove an index outside its vector DB directory")
        for attempt in range(10):
            try:
                shutil.rmtree(self.persist_dir)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.2)

    def rebuild_records(self, records: list[JobRecord]) -> int:
        """Replace this collection with exactly the supplied catalog records."""
        self._remove_index_dir()
        self.db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )
        self.index_needs_rebuild = True
        self.indexed_job_hashes = {}
        self._write_manifest(state="building")
        chunks: list[Document] = []
        ids: list[str] = []
        for record in records:
            record_chunks = self._chunks_for_record(record)
            chunks.extend(record_chunks)
            if record_chunks:
                self.indexed_job_hashes[record.job_id] = record.content_hash
            ids.extend(f"job:{record.job_id}:{index}" for index in range(len(record_chunks)))
        if chunks:
            self.db.add_documents(chunks, ids=ids)
        self._write_manifest()
        self.index_needs_rebuild = False
        return len(chunks)

    def rebuild(self) -> int:
        return self.rebuild_records(self.catalog.all_records())


def import_open_source_jobs(
    *,
    csv_path: Path | None,
    project_markdown_dir: Path | None,
    catalog_path: Path,
    source_corpus_dir: Path,
    vector_db_dir: Path,
) -> dict[str, int]:
    """Import only approved public CSV and `real_en_jd_*` Markdown sources."""
    ingestion = JobKnowledgeIngestion(
        catalog=JobCatalog(catalog_path),
        source_corpus_dir=source_corpus_dir,
        vector_db_dir=vector_db_dir,
        collection_name=settings.job_collection_name,
    )
    inserted = duplicates = skipped = chunks_added = 0
    try:
        if csv_path is not None and csv_path.exists():
            parsed = KyosekCsvAdapter(csv_path).load()
            skipped += parsed.skipped
            original = csv_path.read_bytes()
            for job in parsed.jobs:
                outcome = ingestion.ingest(job, original_bytes=original)
                if outcome.inserted:
                    inserted += 1
                    chunks_added += outcome.chunks_added
                else:
                    duplicates += 1
        if project_markdown_dir is not None and project_markdown_dir.exists():
            for path in sorted(project_markdown_dir.glob("real_en_jd_*.md")):
                parsed = ProjectMarkdownAdapter(path).load()
                skipped += parsed.skipped
                original = path.read_bytes()
                for job in parsed.jobs:
                    outcome = ingestion.ingest(job, original_bytes=original)
                    if outcome.inserted:
                        inserted += 1
                        chunks_added += outcome.chunks_added
                    else:
                        duplicates += 1
        return {"inserted": inserted, "duplicates": duplicates, "skipped": skipped, "chunks_added": chunks_added}
    finally:
        ingestion.close()
