from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SourceKind = Literal[
    "open_source",
    "user_upload",
    "official_career",
    "official_api",
    "public_notice",
]


@dataclass(frozen=True)
class NormalizedJob:
    company_name: str
    job_title: str
    description: str
    location: str | None
    source_kind: SourceKind
    source_dataset: str
    source_file: str
    source_url: str | None
    language: Literal["zh", "en"]
    external_id: str | None = None
    apply_url: str | None = None
    source_name: str | None = None
    recruitment_type: str | None = None
    graduation_year: int | None = None
    degree_requirement: str | None = None
    major_requirement: str | None = None
    job_category: str | None = None
    employment_type: str | None = None
    posted_at: str | None = None
    deadline_at: str | None = None
    first_seen_at: str | None = None
    last_seen_at: str | None = None
    status: str = "open"
    is_domestic: bool = False
    raw_payload_json: str | None = None


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    company_name: str
    company_key: str
    job_title: str
    job_title_key: str
    description: str
    location: str | None
    language: str
    source_kind: str
    source_dataset: str
    source_file: str
    source_url: str | None
    content_hash: str
    is_user_uploaded: bool
    created_at: str
    external_id: str | None = None
    apply_url: str | None = None
    source_name: str | None = None
    recruitment_type: str | None = None
    graduation_year: int | None = None
    degree_requirement: str | None = None
    major_requirement: str | None = None
    job_category: str | None = None
    employment_type: str | None = None
    posted_at: str | None = None
    deadline_at: str | None = None
    first_seen_at: str | None = None
    last_seen_at: str | None = None
    status: str = "open"
    is_domestic: bool = False
    miss_count: int = 0
    raw_payload_json: str | None = None


@dataclass(frozen=True)
class CatalogUpsertResult:
    record: JobRecord
    inserted: bool
    updated: bool = False
