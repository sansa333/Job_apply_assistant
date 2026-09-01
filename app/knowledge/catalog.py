from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.knowledge.models import CatalogUpsertResult, JobRecord, NormalizedJob
from app.knowledge.normalize import normalize_company_name, normalize_job_title


class JobCatalog:
    """SQLite source of truth for exact company and job-title lookup."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS job_records (
                  job_id TEXT PRIMARY KEY,
                  company_name TEXT NOT NULL,
                  company_key TEXT NOT NULL,
                  job_title TEXT NOT NULL,
                  job_title_key TEXT NOT NULL,
                  description TEXT NOT NULL,
                  location TEXT,
                  language TEXT NOT NULL,
                  source_kind TEXT NOT NULL CHECK (source_kind IN ('open_source', 'user_upload')),
                  source_dataset TEXT NOT NULL,
                  source_file TEXT NOT NULL,
                  source_url TEXT,
                  content_hash TEXT NOT NULL,
                  is_user_uploaded INTEGER NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL,
                  UNIQUE(company_key, job_title_key, content_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_job_lookup
                ON job_records(company_key, job_title_key, is_user_uploaded, created_at);

                CREATE TABLE IF NOT EXISTS job_snapshots (
                  snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job_id TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  description TEXT NOT NULL,
                  raw_payload_json TEXT,
                  captured_at TEXT NOT NULL,
                  UNIQUE(job_id, content_hash),
                  FOREIGN KEY(job_id) REFERENCES job_records(job_id)
                );

                CREATE TABLE IF NOT EXISTS job_sources (
                  source_id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  source_type TEXT NOT NULL,
                  base_url TEXT NOT NULL,
                  terms_url TEXT,
                  robots_url TEXT,
                  enabled INTEGER NOT NULL DEFAULT 1,
                  schedule_minutes INTEGER NOT NULL DEFAULT 360,
                  config_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS crawl_runs (
                  run_id TEXT PRIMARY KEY,
                  source_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  started_at TEXT NOT NULL,
                  finished_at TEXT,
                  fetched_count INTEGER NOT NULL DEFAULT 0,
                  inserted_count INTEGER NOT NULL DEFAULT 0,
                  updated_count INTEGER NOT NULL DEFAULT 0,
                  error_message TEXT,
                  FOREIGN KEY(source_id) REFERENCES job_sources(source_id)
                );

                CREATE TABLE IF NOT EXISTS application_records (
                  candidate_id TEXT NOT NULL,
                  job_id TEXT NOT NULL,
                  stage TEXT NOT NULL,
                  notes TEXT NOT NULL DEFAULT '',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY(candidate_id, job_id),
                  FOREIGN KEY(job_id) REFERENCES job_records(job_id)
                );
                """
            )
            self._ensure_domestic_columns(connection)
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_job_source_external
                ON job_records(source_dataset, external_id);
                CREATE INDEX IF NOT EXISTS idx_job_domestic_search
                ON job_records(is_domestic, status, recruitment_type, graduation_year, posted_at);
                """
            )

    @staticmethod
    def _ensure_domestic_columns(connection: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(job_records)").fetchall()
        }
        columns = {
            "external_id": "TEXT",
            "apply_url": "TEXT",
            "source_name": "TEXT",
            "recruitment_type": "TEXT",
            "graduation_year": "INTEGER",
            "degree_requirement": "TEXT",
            "major_requirement": "TEXT",
            "job_category": "TEXT",
            "employment_type": "TEXT",
            "posted_at": "TEXT",
            "deadline_at": "TEXT",
            "first_seen_at": "TEXT",
            "last_seen_at": "TEXT",
            "status": "TEXT NOT NULL DEFAULT 'open'",
            "is_domestic": "INTEGER NOT NULL DEFAULT 0",
            "miss_count": "INTEGER NOT NULL DEFAULT 0",
            "raw_payload_json": "TEXT",
        }
        for name, declaration in columns.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE job_records ADD COLUMN {name} {declaration}")

    @staticmethod
    def _content_hash(company_key: str, title_key: str, description: str) -> str:
        normalized = "\n".join([company_key, title_key, " ".join(description.split())])
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _to_record(row: sqlite3.Row) -> JobRecord:
        boolean_fields = {"is_user_uploaded", "is_domestic"}
        return JobRecord(
            **{
                key: (bool(row[key]) if key in boolean_fields else row[key])
                for key in row.keys()
            }
        )

    def upsert(self, job: NormalizedJob) -> CatalogUpsertResult:
        company_key = normalize_company_name(job.company_name)
        title_key = normalize_job_title(job.job_title)
        if not company_key or not title_key or not job.description.strip():
            raise ValueError("company_name, job_title and description must be non-empty")

        content_hash = self._content_hash(company_key, title_key, job.description)
        job_id = f"job_{content_hash[:24]}"
        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        is_user_uploaded = int(job.source_kind == "user_upload")
        is_domestic = int(job.is_domestic or job.language == "zh")
        first_seen_at = job.first_seen_at or created_at
        last_seen_at = job.last_seen_at or created_at
        with self._connect() as connection:
            if job.external_id:
                row = connection.execute(
                    """
                    SELECT * FROM job_records
                    WHERE source_dataset = ? AND external_id = ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (job.source_dataset, job.external_id),
                ).fetchone()
                if row:
                    changed = row["content_hash"] != content_hash
                    connection.execute(
                        """
                        UPDATE job_records SET
                          company_name = ?, company_key = ?, job_title = ?, job_title_key = ?,
                          description = ?, location = ?, language = ?, source_kind = ?, source_file = ?,
                          source_url = ?, content_hash = ?, apply_url = ?, source_name = ?,
                          recruitment_type = ?, graduation_year = ?, degree_requirement = ?,
                          major_requirement = ?, job_category = ?, employment_type = ?, posted_at = ?,
                          deadline_at = ?, last_seen_at = ?, status = ?, is_domestic = ?, miss_count = 0,
                          raw_payload_json = ?
                        WHERE job_id = ?
                        """,
                        (
                            job.company_name.strip(), company_key, job.job_title.strip(), title_key,
                            job.description.strip(), job.location, job.language, job.source_kind,
                            job.source_file, job.source_url, content_hash, job.apply_url,
                            job.source_name, job.recruitment_type, job.graduation_year,
                            job.degree_requirement, job.major_requirement, job.job_category,
                            job.employment_type, job.posted_at, job.deadline_at, last_seen_at,
                            job.status, is_domestic, job.raw_payload_json, row["job_id"],
                        ),
                    )
                    if changed:
                        self._save_snapshot(
                            connection,
                            row["job_id"],
                            content_hash,
                            job.description,
                            job.raw_payload_json,
                            created_at,
                        )
                    updated_row = connection.execute(
                        "SELECT * FROM job_records WHERE job_id = ?", (row["job_id"],)
                    ).fetchone()
                    return CatalogUpsertResult(
                        record=self._to_record(updated_row), inserted=False, updated=changed
                    )
            row = connection.execute(
                "SELECT * FROM job_records WHERE company_key = ? AND job_title_key = ? AND content_hash = ?",
                (company_key, title_key, content_hash),
            ).fetchone()
            if row:
                return CatalogUpsertResult(record=self._to_record(row), inserted=False)
            connection.execute(
                """
                INSERT INTO job_records (
                    job_id, company_name, company_key, job_title, job_title_key, description, location,
                    language, source_kind, source_dataset, source_file, source_url, content_hash,
                    is_user_uploaded, created_at, external_id, apply_url, source_name,
                    recruitment_type, graduation_year, degree_requirement, major_requirement,
                    job_category, employment_type, posted_at, deadline_at, first_seen_at, last_seen_at,
                    status, is_domestic, miss_count, raw_payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id, job.company_name.strip(), company_key, job.job_title.strip(), title_key,
                    job.description.strip(), job.location, job.language, job.source_kind, job.source_dataset,
                    job.source_file, job.source_url, content_hash, is_user_uploaded, created_at,
                    job.external_id, job.apply_url, job.source_name, job.recruitment_type,
                    job.graduation_year, job.degree_requirement, job.major_requirement,
                    job.job_category, job.employment_type, job.posted_at, job.deadline_at,
                    first_seen_at, last_seen_at, job.status, is_domestic, 0, job.raw_payload_json,
                ),
            )
            row = connection.execute("SELECT * FROM job_records WHERE job_id = ?", (job_id,)).fetchone()
            self._save_snapshot(
                connection, job_id, content_hash, job.description, job.raw_payload_json, created_at
            )
        return CatalogUpsertResult(record=self._to_record(row), inserted=True, updated=False)

    @staticmethod
    def _save_snapshot(
        connection: sqlite3.Connection,
        job_id: str,
        content_hash: str,
        description: str,
        raw_payload_json: str | None,
        captured_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT OR IGNORE INTO job_snapshots (
              job_id, content_hash, description, raw_payload_json, captured_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (job_id, content_hash, description.strip(), raw_payload_json, captured_at),
        )

    def lookup(self, company_name: str, job_title: str) -> list[JobRecord]:
        company_key = normalize_company_name(company_name)
        title_key = normalize_job_title(job_title)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM job_records
                WHERE company_key = ? AND job_title_key = ?
                ORDER BY is_user_uploaded DESC, created_at DESC, job_id ASC
                """,
                (company_key, title_key),
            ).fetchall()
        return [self._to_record(row) for row in rows]

    def get(self, job_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM job_records WHERE job_id = ?", (job_id,)).fetchone()
        return self._to_record(row) if row else None

    def all_records(self) -> list[JobRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM job_records ORDER BY created_at, job_id").fetchall()
        return [self._to_record(row) for row in rows]

    def domestic_records(self, *, status: str = "") -> list[JobRecord]:
        clauses = ["is_domestic = 1"]
        params: list[object] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM job_records WHERE {' AND '.join(clauses)} "
                "ORDER BY COALESCE(posted_at, created_at) DESC, job_id",
                params,
            ).fetchall()
        return [self._to_record(row) for row in rows]

    def search(
        self,
        *,
        keyword: str = "",
        company_name: str = "",
        location: str = "",
        recruitment_type: str = "",
        graduation_year: int | None = None,
        status: str = "open",
        domestic_only: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> list[JobRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if domestic_only:
            clauses.append("is_domestic = 1")
        if status:
            clauses.append("status = ?")
            params.append(status)
        if keyword.strip():
            terms = [term for term in keyword.split() if term]
            if not terms:
                terms = [keyword.strip()]
            term_clauses: list[str] = []
            for term in terms[:8]:
                value = f"%{term}%"
                term_clauses.append("(job_title LIKE ? OR description LIKE ? OR job_category LIKE ?)")
                params.extend([value, value, value])
            clauses.append(f"({' OR '.join(term_clauses)})")
        if company_name.strip():
            clauses.append("company_name LIKE ?")
            params.append(f"%{company_name.strip()}%")
        if location.strip():
            clauses.append("location LIKE ?")
            params.append(f"%{location.strip()}%")
        if recruitment_type.strip():
            clauses.append("recruitment_type = ?")
            params.append(recruitment_type.strip())
        if graduation_year is not None:
            clauses.append("(graduation_year IS NULL OR graduation_year = ?)")
            params.append(graduation_year)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([max(1, min(limit, 200)), max(0, offset)])
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM job_records
                {where}
                ORDER BY COALESCE(posted_at, created_at) DESC, company_name, job_title
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
        return [self._to_record(row) for row in rows]

    def count(self, *, domestic_only: bool = False, status: str = "") -> int:
        clauses: list[str] = []
        params: list[object] = []
        if domestic_only:
            clauses.append("is_domestic = 1")
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) AS count FROM job_records {where}", params
            ).fetchone()
        return int(row["count"])

    def upsert_source(
        self,
        *,
        source_id: str,
        name: str,
        source_type: str,
        base_url: str,
        terms_url: str | None = None,
        robots_url: str | None = None,
        enabled: bool = True,
        schedule_minutes: int = 360,
        config: dict | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO job_sources (
                  source_id, name, source_type, base_url, terms_url, robots_url, enabled,
                  schedule_minutes, config_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                  name = excluded.name, source_type = excluded.source_type,
                  base_url = excluded.base_url, terms_url = excluded.terms_url,
                  robots_url = excluded.robots_url, enabled = excluded.enabled,
                  schedule_minutes = excluded.schedule_minutes,
                  config_json = excluded.config_json, updated_at = excluded.updated_at
                """,
                (
                    source_id, name, source_type, base_url, terms_url, robots_url,
                    int(enabled), schedule_minutes, json.dumps(config or {}, ensure_ascii=False),
                    now, now,
                ),
            )

    def list_sources(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM job_sources ORDER BY name, source_id"
            ).fetchall()
        return [
            {
                **dict(row),
                "enabled": bool(row["enabled"]),
                "config": json.loads(row["config_json"] or "{}"),
            }
            for row in rows
        ]

    def start_crawl_run(self, *, run_id: str, source_id: str) -> None:
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO crawl_runs (run_id, source_id, status, started_at)
                VALUES (?, ?, 'running', ?)
                """,
                (run_id, source_id, now),
            )

    def finish_crawl_run(
        self,
        *,
        run_id: str,
        status: str,
        fetched_count: int = 0,
        inserted_count: int = 0,
        updated_count: int = 0,
        error_message: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE crawl_runs SET
                  status = ?, finished_at = ?, fetched_count = ?, inserted_count = ?,
                  updated_count = ?, error_message = ?
                WHERE run_id = ?
                """,
                (
                    status, now, fetched_count, inserted_count, updated_count,
                    error_message, run_id,
                ),
            )

    def list_crawl_runs(self, limit: int = 50) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT r.*, s.name AS source_name
                FROM crawl_runs r
                LEFT JOIN job_sources s ON s.source_id = r.source_id
                ORDER BY r.started_at DESC LIMIT ?
                """,
                (max(1, min(limit, 200)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_source_missing(self, source_dataset: str, observed_external_ids: set[str]) -> int:
        """Increment miss counts and close a posting only after three consecutive misses."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT job_id, external_id, miss_count FROM job_records
                WHERE source_dataset = ? AND external_id IS NOT NULL AND status != 'closed'
                """,
                (source_dataset,),
            ).fetchall()
            changed = 0
            for row in rows:
                if row["external_id"] in observed_external_ids:
                    continue
                miss_count = int(row["miss_count"] or 0) + 1
                status = "closed" if miss_count >= 3 else "possibly_closed"
                connection.execute(
                    "UPDATE job_records SET miss_count = ?, status = ? WHERE job_id = ?",
                    (miss_count, status, row["job_id"]),
                )
                changed += 1
        return changed

    def set_application_stage(
        self,
        *,
        candidate_id: str,
        job_id: str,
        stage: str,
        notes: str = "",
    ) -> dict:
        if self.get(job_id) is None:
            raise ValueError("job not found")
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO application_records (
                  candidate_id, job_id, stage, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id, job_id) DO UPDATE SET
                  stage = excluded.stage, notes = excluded.notes, updated_at = excluded.updated_at
                """,
                (candidate_id, job_id, stage, notes, now, now),
            )
            row = connection.execute(
                "SELECT * FROM application_records WHERE candidate_id = ? AND job_id = ?",
                (candidate_id, job_id),
            ).fetchone()
        return dict(row)

    def list_applications(self, candidate_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT a.*, j.company_name, j.job_title, j.location, j.apply_url, j.source_url
                FROM application_records a
                JOIN job_records j ON j.job_id = a.job_id
                WHERE a.candidate_id = ?
                ORDER BY a.updated_at DESC
                """,
                (candidate_id,),
            ).fetchall()
        return [dict(row) for row in rows]
