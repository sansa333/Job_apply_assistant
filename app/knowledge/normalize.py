from __future__ import annotations

import re


_COMPANY_SUFFIXES = re.compile(r"\b(?:incorporated|inc|limited|ltd|llp|llc|corp(?:oration)?|co(?:mpany)?)\b", re.I)


def _compact(value: str) -> str:
    value = value.strip().lower().replace("&", " and ")
    value = _COMPANY_SUFFIXES.sub(" ", value)
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value)


def normalize_company_name(value: str) -> str:
    return _compact(value)


def normalize_job_title(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value.strip().lower())


def detect_language(text: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"
