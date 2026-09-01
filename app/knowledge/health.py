from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HealthResult:
    collection: str
    healthy: bool
    issues: list[str]
    manifest: dict | None


class CollectionHealth:
    def __init__(self, vector_db_dir: Path):
        self.vector_db_dir = vector_db_dir

    def check(self, collection: str, *, embedding_backend: str, embedding_dimension: int) -> HealthResult:
        path = self.vector_db_dir / collection / "collection_manifest.json"
        if not path.exists():
            return HealthResult(collection=collection, healthy=False, issues=["manifest_missing"], manifest=None)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        issues: list[str] = []
        if manifest.get("collection") != collection:
            issues.append("collection_mismatch")
        if manifest.get("embedding_backend") != embedding_backend:
            issues.append("embedding_backend_mismatch")
        if manifest.get("dimension") != embedding_dimension:
            issues.append("dimension_mismatch")
        return HealthResult(collection=collection, healthy=not issues, issues=issues, manifest=manifest)

    @staticmethod
    def payload(result: HealthResult) -> dict:
        return asdict(result)
