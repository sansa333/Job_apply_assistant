from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Iterable

from app.domain.job_application import (
    CandidateFact,
    CategoryScore,
    EvidenceMatch,
    EvidenceSupport,
    JobRequirement,
    ParsedCandidateProfile,
    ParsedJobDescription,
    RequirementCategory,
    ScoreBreakdown,
    ValidationFinding,
)


# This lexicon is deliberately versioned and deterministic. It is a transparent
# baseline, not a claim that every occupational skill can be represented here.
SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "python": ("python",),
    "java": ("java",),
    "javascript": ("javascript", "js"),
    "typescript": ("typescript", "ts"),
    "c++": ("c++", "cpp"),
    "c#": ("c#", "csharp"),
    "sql": ("sql",),
    "fastapi": ("fastapi",),
    "flask": ("flask",),
    "django": ("django",),
    "react": ("react", "react.js", "reactjs"),
    "angular": ("angular",),
    "vue": ("vue", "vue.js", "vuejs"),
    "spring boot": ("spring boot", "springboot"),
    "langchain": ("langchain",),
    "langgraph": ("langgraph",),
    "rag": ("rag", "retrieval augmented generation", "检索增强生成"),
    "llm": ("llm", "large language model", "大语言模型", "大模型"),
    "machine learning": ("machine learning", "机器学习", "ml"),
    "deep learning": ("deep learning", "深度学习"),
    "nlp": ("nlp", "natural language processing", "自然语言处理"),
    "computer vision": ("computer vision", "计算机视觉"),
    "pytorch": ("pytorch",),
    "tensorflow": ("tensorflow",),
    "scikit-learn": ("scikit-learn", "sklearn"),
    "pandas": ("pandas",),
    "numpy": ("numpy",),
    "spark": ("spark", "pyspark"),
    "airflow": ("airflow",),
    "kafka": ("kafka",),
    "redis": ("redis",),
    "mysql": ("mysql",),
    "postgresql": ("postgresql", "postgres"),
    "mongodb": ("mongodb", "mongo"),
    "elasticsearch": ("elasticsearch", "elastic search"),
    "chroma": ("chroma", "chromadb"),
    "faiss": ("faiss",),
    "pgvector": ("pgvector",),
    "docker": ("docker", "containerization", "容器化"),
    "kubernetes": ("kubernetes", "k8s"),
    "helm": ("helm",),
    "linux": ("linux",),
    "git": ("git",),
    "ci/cd": ("ci/cd", "cicd", "continuous integration", "持续集成"),
    "aws": ("aws", "amazon web services"),
    "azure": ("azure",),
    "gcp": ("gcp", "google cloud"),
    "mlflow": ("mlflow",),
    "tableau": ("tableau",),
    "power bi": ("power bi", "powerbi"),
    "a/b testing": ("a/b testing", "a/b test", "ab test", "ab testing", "a/b 测试"),
    "statistics": ("statistics", "statistical", "统计学", "统计分析"),
    "data visualization": ("data visualization", "visualisation", "数据可视化"),
    "credit risk": ("credit risk", "credit analysis", "credit underwriting", "counterparty risk", "settlement risk", "信用风险"),
    "risk management": ("risk management", "风险管理"),
    "broadcast engineering": ("broadcast engineering", "broadcast chain", "广播工程"),
    "networking": ("networking", "network technologies", "network faults", "网络"),
    "virtualization": ("virtualization", "virtualisation", "虚拟化"),
    "cloud services": ("cloud services", "cloud infrastructure", "云服务", "云基础设施"),
    "signal processing": ("signal processing", "信号处理"),
    "medical devices": ("medical device", "medical devices", "biosensor", "biosensors", "医疗器械"),
    "murine models": ("murine model", "murine models", "mouse model", "mouse models", "小鼠模型"),
    "scientific writing": ("manuscript writing", "scientific writing", "论文写作"),
    "transcriptomics": ("transcriptomics", "转录组学"),
    "proteomics": ("proteomics", "蛋白质组学"),
    "microscopy": ("microscopy", "显微镜"),
}

_SECTION_HEADERS = {
    "responsibilities": RequirementCategory.RESPONSIBILITY,
    "responsibility": RequirementCategory.RESPONSIBILITY,
    "accountabilities": RequirementCategory.RESPONSIBILITY,
    "scope": RequirementCategory.RESPONSIBILITY,
    "main duties": RequirementCategory.RESPONSIBILITY,
    "岗位职责": RequirementCategory.RESPONSIBILITY,
    "工作职责": RequirementCategory.RESPONSIBILITY,
    "requirements": RequirementCategory.OTHER,
    "required skills and experience": RequirementCategory.OTHER,
    "qualifications": RequirementCategory.OTHER,
    "what you'll bring": RequirementCategory.OTHER,
    "任职要求": RequirementCategory.OTHER,
    "岗位要求": RequirementCategory.OTHER,
    "nice to have": RequirementCategory.OTHER,
    "preferred qualifications": RequirementCategory.OTHER,
    "加分项": RequirementCategory.OTHER,
}

_IGNORE_SECTION_HEADERS = (
    "about us",
    "about the company",
    "company description",
    "in return",
    "benefits",
    "what we offer",
    "additional information",
    "equal opportunities",
    "equal opportunity",
    "inclusion & diversity",
    "inclusion and diversity",
    "accommodations",
    "how we get things done",
    "the legal bits",
    "key notes for applicants",
)

_MUST_PATTERNS = re.compile(
    r"\b(required|must|essential|need to|strong proficiency|proven experience|you will need)\b|"
    r"必须|必备|要求|熟练|精通|需要具备",
    re.I,
)
_PREFERRED_PATTERNS = re.compile(
    r"\b(preferred|desirable|nice to have|ideally|a plus|familiarity)\b|优先|加分|最好|熟悉",
    re.I,
)
_EXPERIENCE_PATTERNS = re.compile(
    r"\b(?:at least\s+)?\d+(?:\s*[-–]\s*\d+)?\+?\s*years?\b|\bproven experience\b|"
    r"\btrack record\b|\d+\s*年|工作经验|项目经验",
    re.I,
)
_EDUCATION_PATTERNS = re.compile(
    r"\b(ph\.?d|doctorate|master'?s?|bachelor'?s?|degree|computer science|data science)\b|"
    r"博士|硕士|本科|学历|学位",
    re.I,
)
_LANGUAGE_PATTERNS = re.compile(r"\b(english|mandarin|chinese|german|french|spanish)\b|英语|中文|德语|法语", re.I)
_LOCATION_PATTERNS = re.compile(r"\b(remote|hybrid|on[- ]site|office|relocat|visa|sponsorship)\b|远程|混合办公|工作地点|签证", re.I)
_SOFT_PATTERNS = re.compile(r"\b(communication|collaboration|stakeholder|leadership|organis|self-motivat|team)\w*\b|沟通|协作|领导力|自驱|团队", re.I)

_TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9+#./-]{1,}|[\u4e00-\u9fff]{2,}", re.I)
_STOPWORDS = {
    "and", "the", "for", "with", "that", "this", "from", "your", "you", "will", "have", "has",
    "are", "our", "into", "using", "skills", "skill", "experience", "role", "work", "team", "strong",
    "ability", "knowledge", "including", "related", "负责", "要求", "岗位", "工作", "能力", "经验", "相关",
    "以及", "进行", "具备", "熟悉", "能够", "使用", "参与",
}


def _contains_alias(text: str, alias: str) -> bool:
    if re.fullmatch(r"[a-z0-9+#./ -]+", alias, flags=re.I):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text, flags=re.I))
    return alias.lower() in text.lower()


def extract_skills(text: str) -> list[str]:
    return sorted(
        canonical
        for canonical, aliases in SKILL_ALIASES.items()
        if any(_contains_alias(text, alias) for alias in aliases)
    )


def content_tokens(text: str) -> set[str]:
    skills = set(extract_skills(text))
    tokens = {token.lower() for token in _TOKEN_PATTERN.findall(text) if token.lower() not in _STOPWORDS}
    return skills | tokens


def _clean_line(line: str) -> str:
    line = re.sub(r"^[\s\-•*\d.)]+", "", line).strip()
    return re.sub(r"\s+", " ", line)


def _line_category(text: str, section_category: RequirementCategory) -> RequirementCategory:
    if extract_skills(text):
        return RequirementCategory.TECHNICAL_SKILL
    if _EXPERIENCE_PATTERNS.search(text):
        return RequirementCategory.EXPERIENCE
    if _EDUCATION_PATTERNS.search(text):
        return RequirementCategory.EDUCATION
    if _LANGUAGE_PATTERNS.search(text):
        return RequirementCategory.LANGUAGE
    if _LOCATION_PATTERNS.search(text):
        return RequirementCategory.LOCATION_WORK_MODE
    if _SOFT_PATTERNS.search(text):
        return RequirementCategory.SOFT_SKILL
    if section_category == RequirementCategory.RESPONSIBILITY:
        return RequirementCategory.RESPONSIBILITY
    return RequirementCategory.OTHER


def _requirement_weight(category: RequirementCategory, must_have: bool, preferred: bool) -> float:
    base = {
        RequirementCategory.TECHNICAL_SKILL: 1.4,
        RequirementCategory.EXPERIENCE: 1.3,
        RequirementCategory.EDUCATION: 1.1,
        RequirementCategory.RESPONSIBILITY: 1.0,
        RequirementCategory.DOMAIN: 1.1,
        RequirementCategory.LANGUAGE: 0.8,
        RequirementCategory.LOCATION_WORK_MODE: 0.9,
        RequirementCategory.SOFT_SKILL: 0.7,
        RequirementCategory.OTHER: 0.6,
    }[category]
    if must_have:
        base *= 1.5
    if preferred:
        base *= 0.65
    return round(base, 3)


def parse_job_description(
    *,
    company_name: str,
    job_title: str,
    description: str,
    location: str | None = None,
    language: str = "unknown",
    source_url: str | None = None,
) -> ParsedJobDescription:
    section_name = "job_description"
    section_category = RequirementCategory.OTHER
    in_preferred_section = False
    candidates: list[tuple[str, str, RequirementCategory, bool]] = []

    for raw_line in description.replace("\r", "").split("\n"):
        cleaned = _clean_line(raw_line)
        if not cleaned:
            continue
        lower = cleaned.lower().rstrip(":：")
        ignored_header = next((name for name in _IGNORE_SECTION_HEADERS if lower == name or lower.startswith(name + " ")), None)
        if ignored_header and len(cleaned) < 120:
            section_name = "ignored_boilerplate"
            section_category = RequirementCategory.OTHER
            in_preferred_section = False
            continue
        matched_header = next((name for name in _SECTION_HEADERS if lower == name or lower.startswith(name + " ")), None)
        if matched_header and len(cleaned) < 100:
            section_name = matched_header
            section_category = _SECTION_HEADERS[matched_header]
            in_preferred_section = "nice" in matched_header or "preferred" in matched_header or "加分" in matched_header
            continue
        if section_name == "ignored_boilerplate":
            continue
        if len(cleaned) < 18 and cleaned.endswith((":", "：")):
            section_name = lower
            continue
        if len(cleaned) < 24 and not any(char in cleaned for char in ".。;；") and not extract_skills(cleaned):
            continue
        category = _line_category(cleaned, section_category)
        requirement_signal = (
            section_name != "job_description"
            or category != RequirementCategory.OTHER
            or _MUST_PATTERNS.search(cleaned)
            or _PREFERRED_PATTERNS.search(cleaned)
        )
        if requirement_signal:
            candidates.append((cleaned, section_name, category, in_preferred_section))

    # Deduplicate boilerplate and near-identical repeated responsibility/requirement lines.
    seen: set[str] = set()
    requirements: list[JobRequirement] = []
    for text, section, category, preferred_section in candidates:
        key = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        preferred = bool(preferred_section or _PREFERRED_PATTERNS.search(text))
        must_have = bool(_MUST_PATTERNS.search(text)) and not preferred
        terms = extract_skills(text)
        requirements.append(
            JobRequirement(
                requirement_id=f"req_{len(requirements) + 1:03d}",
                category=category,
                text=text,
                normalized_terms=terms,
                must_have=must_have,
                preferred=preferred,
                weight=_requirement_weight(category, must_have, preferred),
                source_section=section,
            )
        )

    return ParsedJobDescription(
        company_name=company_name,
        job_title=job_title,
        location=location,
        language=language,
        requirements=requirements,
        source_url=source_url,
        content_hash=hashlib.sha256(description.encode("utf-8")).hexdigest(),
    )


def parse_candidate_profile(
    *,
    candidate_id: str | None,
    sources: Iterable[dict],
    source_kind: str,
) -> ParsedCandidateProfile:
    facts: list[CandidateFact] = []
    seen: set[str] = set()
    for source_index, source in enumerate(sources, start=1):
        content = str(source.get("content", ""))
        source_name = str(source.get("filename") or source.get("source_name") or f"source_{source_index}")
        source_id = str(source.get("chunk_id") or source.get("source_id") or f"candidate_source_{source_index}")
        section = str(source.get("section", "unknown"))
        for raw_line in content.replace("\r", "").split("\n"):
            text = _clean_line(raw_line)
            if len(text) < 3:
                continue
            key = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text.lower())
            if not key or key in seen:
                continue
            seen.add(key)
            category = _line_category(text, RequirementCategory.OTHER)
            facts.append(
                CandidateFact(
                    fact_id=f"fact_{len(facts) + 1:03d}",
                    text=text,
                    normalized_terms=extract_skills(text),
                    category=category,
                    source_id=source_id,
                    source_name=source_name,
                    section=section,
                )
            )
    return ParsedCandidateProfile(candidate_id=candidate_id, facts=facts, source_kind=source_kind)


def _fact_match_score(requirement: JobRequirement, fact: CandidateFact) -> tuple[float, list[str]]:
    required_terms = set(requirement.normalized_terms)
    fact_terms = set(fact.normalized_terms)
    matched_terms = sorted(required_terms & fact_terms)
    if required_terms:
        coverage = len(matched_terms) / len(required_terms)
        if coverage:
            return min(1.0, 0.75 + 0.25 * coverage), matched_terms

    requirement_tokens = content_tokens(requirement.text)
    fact_tokens = content_tokens(fact.text)
    if not requirement_tokens or not fact_tokens:
        return 0.0, []
    intersection = requirement_tokens & fact_tokens
    containment = len(intersection) / max(1, min(len(requirement_tokens), len(fact_tokens)))
    jaccard = len(intersection) / len(requirement_tokens | fact_tokens)
    score = 0.65 * containment + 0.35 * jaccard
    return min(1.0, score), sorted(intersection)[:8]


def align_evidence(job: ParsedJobDescription, candidate: ParsedCandidateProfile) -> list[EvidenceMatch]:
    matrix: list[EvidenceMatch] = []
    for requirement in job.requirements:
        ranked: list[tuple[float, CandidateFact, list[str]]] = []
        for fact in candidate.facts:
            score, terms = _fact_match_score(requirement, fact)
            if score > 0:
                ranked.append((score, fact, terms))
        ranked.sort(key=lambda item: (-item[0], item[1].fact_id))
        best = ranked[:2]
        best_score = best[0][0] if best else 0.0
        if best_score >= 0.72:
            support = EvidenceSupport.DIRECT
        elif best_score >= 0.28:
            support = EvidenceSupport.PARTIAL
        else:
            support = EvidenceSupport.MISSING
        selected = [item for item in best if item[0] >= 0.28]
        matrix.append(
            EvidenceMatch(
                requirement_id=requirement.requirement_id,
                requirement_text=requirement.text,
                category=requirement.category,
                must_have=requirement.must_have,
                support=support,
                confidence=round(best_score, 4),
                matched_terms=sorted({term for _, _, terms in selected for term in terms}),
                evidence_fact_ids=[fact.fact_id for _, fact, _ in selected],
                evidence_quotes=[fact.text[:320] for _, fact, _ in selected],
                source_names=sorted({fact.source_name for _, fact, _ in selected}),
                explanation={
                    EvidenceSupport.DIRECT: "候选人资料中存在直接技能或职责证据。",
                    EvidenceSupport.PARTIAL: "候选人资料存在相关证据，但未完整覆盖要求。",
                    EvidenceSupport.MISSING: "未在候选人资料中找到足以支持该要求的证据。",
                }[support],
            )
        )
    return matrix


def score_evidence(job: ParsedJobDescription, matrix: list[EvidenceMatch]) -> ScoreBreakdown:
    by_id = {requirement.requirement_id: requirement for requirement in job.requirements}
    support_value = {
        EvidenceSupport.DIRECT: 1.0,
        EvidenceSupport.PARTIAL: 0.5,
        EvidenceSupport.MISSING: 0.0,
    }
    earned_by_category: dict[RequirementCategory, float] = defaultdict(float)
    total_by_category: dict[RequirementCategory, float] = defaultdict(float)
    total_weight = 0.0
    earned_weight = 0.0
    must_total = 0
    must_earned = 0.0
    missing_must_haves: list[str] = []
    direct = partial = missing = 0

    for item in matrix:
        requirement = by_id[item.requirement_id]
        value = support_value[item.support]
        total_weight += requirement.weight
        earned_weight += requirement.weight * value
        total_by_category[requirement.category] += requirement.weight
        earned_by_category[requirement.category] += requirement.weight * value
        if requirement.must_have:
            must_total += 1
            must_earned += value
            if item.support == EvidenceSupport.MISSING:
                missing_must_haves.append(requirement.text)
        if item.support == EvidenceSupport.DIRECT:
            direct += 1
        elif item.support == EvidenceSupport.PARTIAL:
            partial += 1
        else:
            missing += 1

    coverage = 100.0 * earned_weight / total_weight if total_weight else 0.0
    must_coverage = 100.0 * must_earned / must_total if must_total else coverage
    # Missing mandatory requirements are reported explicitly and conservatively
    # reduce the headline score; the raw coverage remains available for audit.
    hard_gate_factor = 1.0 if not missing_must_haves else max(0.55, 1.0 - 0.08 * len(missing_must_haves))
    overall = min(100.0, coverage * hard_gate_factor)
    category_scores = [
        CategoryScore(
            category=category,
            earned_weight=round(earned_by_category[category], 4),
            total_weight=round(total, 4),
            score=round(100.0 * earned_by_category[category] / total, 2) if total else 0.0,
        )
        for category, total in sorted(total_by_category.items(), key=lambda item: item[0].value)
    ]
    return ScoreBreakdown(
        overall_score=round(overall, 2),
        coverage_score=round(coverage, 2),
        must_have_coverage=round(must_coverage, 2),
        category_scores=category_scores,
        direct_matches=direct,
        partial_matches=partial,
        missing_requirements=missing,
        missing_must_haves=missing_must_haves,
    )


def render_evidence_report(
    *,
    job: ParsedJobDescription,
    matrix: list[EvidenceMatch],
    score: ScoreBreakdown,
) -> str:
    lines = [
        "## 可解释匹配结果",
        "",
        f"- 证据加权匹配分：**{score.overall_score:.2f}/100**",
        f"- 原始要求覆盖率：{score.coverage_score:.2f}%",
        f"- 必备条件覆盖率：{score.must_have_coverage:.2f}%",
        f"- 评分版本：`{score.scoring_version}`（{score.calibration_status}）",
        "",
        "> 该分数是透明规则基线，不代表招聘方决定；未经过目标岗位域人工标注校准前，不应解释为录用概率。",
        "",
        "## Requirement–Evidence Matrix",
        "",
        "| 要求 | 类型 | 必备 | 支持程度 | 证据 | 来源 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in matrix:
        quote = "；".join(item.evidence_quotes) or "未找到证据"
        source = "、".join(item.source_names) or "无"
        lines.append(
            f"| {item.requirement_text.replace('|', '/')} | {item.category.value} | "
            f"{'是' if item.must_have else '否'} | {item.support.value} | "
            f"{quote.replace('|', '/')[:240]} | {source.replace('|', '/')} |"
        )
    lines.extend(["", "## 需要补充或核验的必备条件", ""])
    if score.missing_must_haves:
        lines.extend(f"- {item}" for item in score.missing_must_haves)
    else:
        lines.append("- 当前解析出的必备条件均找到了直接或部分证据；仍需人工核对证据真实性和完整性。")
    return "\n".join(lines)


def validate_grounded_text(text: str, allowed_evidence: Iterable[str]) -> list[ValidationFinding]:
    evidence = "\n".join(allowed_evidence).lower()
    findings: list[ValidationFinding] = []
    for match in re.finditer(
        r"(?<![\w.])\d+(?:\.\d+)?\s*(?:%|倍|万|亿|ms|秒|年|years?|months?)",
        text,
        flags=re.I,
    ):
        claim = match.group(0)
        if claim.lower() not in evidence:
            findings.append(
                ValidationFinding(
                    code="unsupported_quantified_claim",
                    severity="error",
                    message="生成内容包含候选人证据中未出现的量化断言。",
                    claim=claim,
                )
            )
    fabricated_markers = ("已完成投递", "已提交申请", "已发送给招聘方", "application submitted")
    for marker in fabricated_markers:
        if marker.lower() in text.lower():
            findings.append(
                ValidationFinding(
                    code="false_submission_claim",
                    severity="error",
                    message="系统只能生成本地草稿，不能声称已完成外部投递。",
                    claim=marker,
                )
            )
    return findings
