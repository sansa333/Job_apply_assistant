"""Freeze business-task cases and local official-source jobs before retrieval runs."""
import argparse
import json
from pathlib import Path

from app.config import settings
from app.evaluation.business import build_business_dataset
from app.knowledge.catalog import JobCatalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=settings.job_catalog_path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_business_dataset(args.output, JobCatalog(args.catalog).domestic_records(status="open"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
