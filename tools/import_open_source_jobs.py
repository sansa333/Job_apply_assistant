from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import settings
from app.knowledge.ingestion import import_open_source_jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Import approved public JD sources into job_knowledge.")
    parser.add_argument("--csv", type=Path, default=settings.source_corpus_dir / "open_source_jobs" / "kyosek_jobs.csv")
    parser.add_argument("--project-jds", type=Path, default=settings.data_dir / "eval_dataset" / "jds")
    args = parser.parse_args()
    report = import_open_source_jobs(
        csv_path=args.csv if args.csv.exists() else None,
        project_markdown_dir=args.project_jds if args.project_jds.exists() else None,
        catalog_path=settings.job_catalog_path,
        source_corpus_dir=settings.source_corpus_dir,
        vector_db_dir=settings.vector_db_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
