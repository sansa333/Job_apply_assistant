import json
from datetime import datetime
from pathlib import Path
from typing import Any


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_filename(name: str) -> str:
    keep = []
    for ch in name.strip():
        if ch.isalnum() or ch in "-_.":
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    result = "".join(keep).strip("._")
    return result or "file"


def validate_identifier(value: str, *, field_name: str = "identifier") -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    if len(normalized) > 80:
        raise ValueError(f"{field_name} must be at most 80 characters")
    if normalized in {".", ".."} or any(
        not (character.isalnum() or character in "-_.") for character in normalized
    ):
        raise ValueError(
            f"{field_name} may contain only letters, numbers, Chinese characters, '-', '_' and '.'"
        )
    return normalized


def write_text(path: Path, content: str, encoding: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if encoding is None:
        from app.config import settings

        encoding = settings.output_text_encoding
    path.write_text(content, encoding=encoding)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
