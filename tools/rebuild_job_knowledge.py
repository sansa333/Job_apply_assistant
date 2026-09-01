from __future__ import annotations

import json

from app.config import settings
from app.knowledge.catalog import JobCatalog
from app.knowledge.ingestion import JobKnowledgeIngestion


def main() -> None:
    ingestion = JobKnowledgeIngestion(
        catalog=JobCatalog(settings.job_catalog_path),
        source_corpus_dir=settings.source_corpus_dir,
        vector_db_dir=settings.vector_db_dir,
        collection_name=settings.job_collection_name,
    )
    try:
        print(json.dumps({"chunks_rebuilt": ingestion.rebuild()}, ensure_ascii=False, indent=2))
    finally:
        ingestion.close()


if __name__ == "__main__":
    main()
