from pathlib import Path
from fastapi import UploadFile

from app.config import settings
from app.rag import RAGStore
from app.utils.file_io import safe_filename


async def save_uploads(files: list[UploadFile], target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for f in files:
        name = safe_filename(f.filename or "upload.txt")
        path = target_dir / name
        content = await f.read()
        path.write_bytes(content)
        saved_paths.append(path)

    return saved_paths


async def ingest_profile_files(files: list[UploadFile]) -> tuple[list[str], int]:
    paths = await save_uploads(files, settings.profile_docs_dir)
    store = RAGStore("profile")
    try:
        chunks = store.ingest_files(paths)
    finally:
        store.close()
    return [p.name for p in paths], chunks


async def ingest_jd_files(files: list[UploadFile]) -> tuple[list[str], int]:
    paths = await save_uploads(files, settings.jd_docs_dir)
    store = RAGStore("job_description")
    try:
        chunks = store.ingest_files(paths)
    finally:
        store.close()
    return [p.name for p in paths], chunks


def rebuild_profile_index() -> int:
    store = RAGStore("profile")
    try:
        return store.ingest_folder(settings.profile_docs_dir)
    finally:
        store.close()


def rebuild_jd_index() -> int:
    store = RAGStore("job_description")
    try:
        return store.ingest_folder(settings.jd_docs_dir)
    finally:
        store.close()
