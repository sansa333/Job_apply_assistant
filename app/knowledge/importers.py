from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from app.knowledge.models import NormalizedJob
from app.knowledge.normalize import detect_language


@dataclass(frozen=True)
class ImportParseResult:
    jobs: list[NormalizedJob]
    skipped: int = 0


class KyosekCsvAdapter:
    """Read the supported public CSV format without accepting incomplete rows."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> ImportParseResult:
        jobs: list[NormalizedJob] = []
        skipped = 0
        with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                company = (row.get("Company") or "").strip()
                title = (row.get("Title") or "").strip()
                description = (row.get("Description") or "").strip()
                if not company or not title or not description:
                    skipped += 1
                    continue
                jobs.append(
                    NormalizedJob(
                        company_name=company,
                        job_title=title,
                        description=description,
                        location=(row.get("Location") or "").strip() or None,
                        source_kind="open_source",
                        source_dataset="kyosek_jobs_csv",
                        source_file=self.path.name,
                        source_url=(row.get("Link") or "").strip() or None,
                        language=detect_language(description),
                    )
                )
        return ImportParseResult(jobs=jobs, skipped=skipped)


class ProjectMarkdownAdapter:
    """Parse checked-in real public evaluation JD Markdown, never synthetic files."""

    def __init__(self, path: Path):
        self.path = path

    @staticmethod
    def _split_title(title: str) -> tuple[str, str]:
        for separator in (" - ", " — ", " | "):
            if separator in title:
                left, right = (part.strip() for part in title.split(separator, 1))
                if left and right:
                    return left, right
        return title.strip(), "公开职位来源"

    def load(self) -> ImportParseResult:
        if not re.fullmatch(r"real_en_jd_\d+(?:_[\w-]+)*\.md", self.path.name):
            return ImportParseResult(jobs=[], skipped=1)
        text = self.path.read_text(encoding="utf-8")
        title_match = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
        content_match = re.search(r"^##\s+Content\s*\n(.+)$", text, re.MULTILINE | re.DOTALL)
        if not title_match or not content_match or not content_match.group(1).strip():
            return ImportParseResult(jobs=[], skipped=1)
        title, company = self._split_title(title_match.group(1))
        description = content_match.group(1).strip()
        language_match = re.search(r"^-\s+language:\s*(zh|en)\s*$", text, re.MULTILINE | re.I)
        return ImportParseResult(
            jobs=[
                NormalizedJob(
                    company_name=company,
                    job_title=title,
                    description=description,
                    location=None,
                    source_kind="open_source",
                    source_dataset="project_real_en_jd",
                    source_file=self.path.name,
                    source_url=None,
                    language=(language_match.group(1).lower() if language_match else detect_language(description)),
                )
            ]
        )


class UserUploadAdapter:
    def __init__(self, *, company_name: str, job_title: str, description: str, source_file: str):
        self.company_name = company_name
        self.job_title = job_title
        self.description = description
        self.source_file = source_file

    def load(self) -> ImportParseResult:
        if not self.company_name.strip() or not self.job_title.strip() or not self.description.strip():
            return ImportParseResult(jobs=[], skipped=1)
        return ImportParseResult(
            jobs=[
                NormalizedJob(
                    company_name=self.company_name.strip(),
                    job_title=self.job_title.strip(),
                    description=self.description.strip(),
                    location=None,
                    source_kind="user_upload",
                    source_dataset="manual_upload",
                    source_file=self.source_file,
                    source_url=None,
                    language=detect_language(self.description),
                )
            ]
        )
