from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from app.knowledge.profiles import CandidateProfileStore
from app.utils.file_io import validate_identifier


KNOWN_SKILLS = [
    "Python", "PyTorch", "Transformer", "FastAPI", "LangChain", "LangGraph",
    "RAG", "Agent", "BGE-M3", "Chroma", "FTS5", "RRF", "Embedding",
    "Docker", "Git", "Prompt", "LiteLLM", "Qwen", "Kalman", "MLP",
]

DEGREE_TERMS = ("博士", "硕士", "本科", "专科", "大专", "PhD", "Master", "Bachelor")
SCHOOL_PATTERN = re.compile(
    r"(?<![\u4e00-\u9fffA-Za-z])"
    r"([\u4e00-\u9fff·]{2,30}(?:大学|学院|学校|研究院)"
    r"|[A-Za-z][A-Za-z&（）()' -]{1,50}?(?:University|College|Institute))"
    r"(?=$|[\s|｜,，;；])",
    flags=re.IGNORECASE,
)
DATE_RANGE_PATTERN = re.compile(
    r"(?P<start>(?:19|20)\d{2})(?:[./年-]\d{1,2}(?:月)?)?\s*"
    r"(?:-|—|–|~|～|至)\s*"
    r"(?P<end>(?:19|20)\d{2}|至今|现在|Present)"
    r"(?:[./年-]\d{1,2}(?:月)?)?",
    flags=re.IGNORECASE,
)


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", str(value)).strip(" \t,，;；|｜")
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _extract_labeled_values(resume_text: str, labels: tuple[str, ...]) -> list[str]:
    label_pattern = "|".join(re.escape(label) for label in labels)
    values: list[str] = []
    for line in resume_text.splitlines():
        match = re.search(
            rf"(?:{label_pattern})(?:\s*[:：]\s*|\s+)([^\n]{{1,100}})",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        values.extend(re.split(r"[,，、;/；|｜]", match.group(1)))
    return _unique(values)


def extract_education(resume_text: str) -> tuple[list[str], list[str], str | None, int | None]:
    """Extract education facts without relying on a known person or institution list."""
    schools = _unique(match.group(1) for match in SCHOOL_PATTERN.finditer(resume_text))
    degrees = [term for term in DEGREE_TERMS if re.search(re.escape(term), resume_text, re.I)]
    degree_rank = {
        "专科": 1, "大专": 1, "本科": 2, "Bachelor": 2,
        "硕士": 3, "Master": 3, "博士": 4, "PhD": 4,
    }
    highest_degree = max(degrees, key=lambda item: degree_rank[item], default=None)

    majors = _extract_labeled_values(resume_text, ("专业", "主修", "Major"))
    degree_pattern = "|".join(re.escape(term) for term in DEGREE_TERMS)
    for line in resume_text.splitlines():
        if not SCHOOL_PATTERN.search(line):
            continue
        match = re.search(
            rf"(?:大学|学院|学校|研究院|University|College|Institute)\s+"
            rf"([^|｜,，;；]{{2,40}}?)\s+(?:{degree_pattern})(?:\s|$)",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            majors.append(match.group(1))
    majors = _unique(majors)

    explicit_years = [
        int(value)
        for value in re.findall(
            r"(?:毕业(?:时间|年份)?|Graduation(?:\s+Year)?)\s*[:：]?\s*((?:19|20)\d{2})",
            resume_text,
            flags=re.IGNORECASE,
        )
    ]
    education_end_years: list[int] = []
    for line in resume_text.splitlines():
        if not (SCHOOL_PATTERN.search(line) or any(term.lower() in line.lower() for term in DEGREE_TERMS)):
            continue
        for match in DATE_RANGE_PATTERN.finditer(line):
            end = match.group("end")
            if end.isdigit():
                education_end_years.append(int(end))
    graduation_year = max(explicit_years or education_end_years, default=None)
    return schools, majors, highest_degree, graduation_year


def _preference_values(explicit: Iterable[str] | None, inferred: list[str]) -> list[str]:
    return _unique(explicit) if explicit is not None else inferred


@dataclass(frozen=True)
class PdfProfileResult:
    candidate_id: str
    managed_pdf_path: Path
    extracted_text_path: Path
    profile_json_path: Path
    page_count: int
    text_length: int
    chunks_added: int
    profile: dict


def extract_pdf_text(path: Path) -> tuple[str, int]:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append((page.extract_text() or "").strip())
    text = "\n\n".join(value for value in pages if value).strip()
    if len(text) < 100:
        raise ValueError(
            "PDF contains too little extractable text. A scanned resume requires OCR/vision processing."
        )
    return text, len(reader.pages)


def derive_candidate_profile(
    candidate_id: str,
    resume_text: str,
    source_pdf: Path,
    *,
    graduation_year: int | None = None,
    target_roles: Iterable[str] | None = None,
    target_cities: Iterable[str] | None = None,
) -> dict:
    candidate_id = validate_identifier(candidate_id, field_name="candidate_id")
    compact = " ".join(resume_text.split())
    name_match = re.search(r"姓\s*名\s*([\u4e00-\u9fff]{2,4})", compact)
    school_values, majors, degree, inferred_graduation_year = extract_education(resume_text)
    inferred_roles = _extract_labeled_values(resume_text, ("求职意向", "目标岗位", "期望职位"))
    inferred_cities = _extract_labeled_values(resume_text, ("期望城市", "意向城市", "工作地点"))
    effective_graduation_year = graduation_year or inferred_graduation_year
    effective_roles = _preference_values(target_roles, inferred_roles)
    effective_cities = _preference_values(target_cities, inferred_cities)
    skills = [skill for skill in KNOWN_SKILLS if skill.lower() in resume_text.lower()]
    evidence_terms = _unique(
        match.group(0)
        for match in re.finditer(
            r"[^。；;\n]{0,40}(?:\d+(?:\.\d+)?%|MRR@?\d*|Recall@?\d*)[^。；;\n]{0,40}",
            resume_text,
            flags=re.IGNORECASE,
        )
    )
    return {
        "candidate_id": candidate_id,
        "name": name_match.group(1) if name_match else None,
        "source_pdf": str(source_pdf),
        "source_sha256": hashlib.sha256(source_pdf.read_bytes()).hexdigest(),
        "resume_content_policy": "read_only_no_rewrite",
        "highest_degree": degree,
        "schools": school_values,
        "majors": majors,
        "graduation_year": effective_graduation_year,
        "graduation_year_evidence": (
            "user_preference" if graduation_year is not None
            else ("resume_education_date" if inferred_graduation_year is not None else None)
        ),
        "target_country": "中国",
        "target_cities": effective_cities,
        "target_roles": effective_roles,
        "skills": skills,
        "evidence_terms": evidence_terms,
        "default_filters": {
            "domestic_only": True,
            "recruitment_types": ["campus", "internship"],
            "graduation_year": effective_graduation_year,
            "salary_filter": None,
            "city_filter": effective_cities,
            "auto_submit": False,
        },
        "excluded_role_signals": [
            "纯预训练", "纯CUDA算子", "纯推荐算法", "仅基础模型训练", "仅芯片编译优化"
        ],
    }


def ingest_pdf_profile(
    *,
    candidate_id: str,
    source_pdf: Path,
    source_corpus_dir: Path,
    vector_db_dir: Path,
    collection_name: str,
    graduation_year: int | None = None,
    target_roles: Iterable[str] | None = None,
    target_cities: Iterable[str] | None = None,
) -> PdfProfileResult:
    candidate_id = validate_identifier(candidate_id, field_name="candidate_id")
    if source_pdf.suffix.lower() != ".pdf":
        raise ValueError("Only PDF resumes are accepted by this endpoint")
    resume_text, page_count = extract_pdf_text(source_pdf)
    target_dir = source_corpus_dir / "candidate_profiles" / candidate_id
    target_dir.mkdir(parents=True, exist_ok=True)
    managed_pdf = target_dir / "resume_source.pdf"
    if source_pdf.resolve() != managed_pdf.resolve():
        shutil.copy2(source_pdf, managed_pdf)
    extracted_path = target_dir / "resume_extracted.txt"
    extracted_path.write_text(resume_text, encoding="utf-8")
    profile = derive_candidate_profile(
        candidate_id,
        resume_text,
        managed_pdf,
        graduation_year=graduation_year,
        target_roles=target_roles,
        target_cities=target_cities,
    )
    profile_path = target_dir / "profile.json"
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    store = CandidateProfileStore(
        source_corpus_dir,
        vector_db_dir,
        collection_name=collection_name,
    )
    try:
        chunks = store.ingest_text(candidate_id, resume_text, "resume_extracted.txt")
    finally:
        store.close()
    return PdfProfileResult(
        candidate_id=candidate_id,
        managed_pdf_path=managed_pdf,
        extracted_text_path=extracted_path,
        profile_json_path=profile_path,
        page_count=page_count,
        text_length=len(resume_text),
        chunks_added=chunks,
        profile=profile,
    )


def load_candidate_profile(source_corpus_dir: Path, candidate_id: str) -> dict | None:
    candidate_id = validate_identifier(candidate_id, field_name="candidate_id")
    path = source_corpus_dir / "candidate_profiles" / candidate_id / "profile.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_candidate_resume_text(source_corpus_dir: Path, candidate_id: str) -> str:
    candidate_id = validate_identifier(candidate_id, field_name="candidate_id")
    path = source_corpus_dir / "candidate_profiles" / candidate_id / "resume_extracted.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def update_candidate_preferences(
    source_corpus_dir: Path,
    candidate_id: str,
    *,
    graduation_year: int | None,
    target_roles: Iterable[str],
    target_cities: Iterable[str],
) -> dict:
    candidate_id = validate_identifier(candidate_id, field_name="candidate_id")
    profile = load_candidate_profile(source_corpus_dir, candidate_id)
    if profile is None:
        raise ValueError("candidate profile not found; import a PDF resume first")
    roles = _unique(target_roles)
    cities = _unique(target_cities)
    profile["graduation_year"] = graduation_year
    profile["graduation_year_evidence"] = "user_preference" if graduation_year else None
    profile["target_roles"] = roles
    profile["target_cities"] = cities
    filters = profile.setdefault("default_filters", {})
    filters["graduation_year"] = graduation_year
    filters["city_filter"] = cities
    path = source_corpus_dir / "candidate_profiles" / candidate_id / "profile.json"
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile
