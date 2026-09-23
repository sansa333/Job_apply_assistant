"""Immutable local analysis/package history for a single-owner deployment."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from app.config import settings
from app.utils.file_io import validate_identifier


def artifact_root(candidate_id: str, job_id: str):
    return settings.outputs_dir / "domestic" / validate_identifier(candidate_id, field_name="candidate_id") / validate_identifier(job_id, field_name="job_id")


def save_artifact(candidate_id: str, job_id: str, payload: dict) -> dict:
    artifact_id = uuid.uuid4().hex
    directory = artifact_root(candidate_id, job_id)
    directory.mkdir(parents=True, exist_ok=True)
    result = {**payload, "artifact_id": artifact_id, "candidate_id": candidate_id, "job_id": job_id,
              "created_at": datetime.now(timezone.utc).isoformat()}
    temporary = directory / f"{artifact_id}.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(directory / f"{artifact_id}.json")
    return result


def list_artifacts(candidate_id: str, job_id: str) -> list[dict]:
    rows = []
    for path in artifact_root(candidate_id, job_id).glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append({key: value.get(key) for key in ("artifact_id", "candidate_id", "job_id", "created_at", "kind", "status")})
    return sorted(rows, key=lambda row: row["created_at"], reverse=True)


def read_artifact(candidate_id: str, job_id: str, artifact_id: str) -> dict:
    artifact_id = validate_identifier(artifact_id, field_name="artifact_id")
    path = artifact_root(candidate_id, job_id) / f"{artifact_id}.json"
    if not path.is_file():
        raise ValueError("artifact not found")
    return json.loads(path.read_text(encoding="utf-8"))
