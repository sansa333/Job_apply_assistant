from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from app.config import settings
from app.knowledge.catalog import JobCatalog


VERSION = "2.0.0-silver"
TARGET_JOBS = 120
QUERIES_PER_JOB = 4
CANDIDATE_POOL_SIZE = 50
QUERY_TYPES = ("responsibilities", "technical_skills", "qualifications", "work_context")
PUBLISHED_JSONL_FILES = (
    "candidate_pools.jsonl",
    "evidence_units.jsonl",
    "job_snapshots.jsonl",
    "qrels.jsonl",
    "queries.jsonl",
)

FAMILY_RULES = {
    "life_sciences_research": (
        "bioinformatic", "genomic", "clinical", "pharma", "medical device", "senior scientist",
        "postdoctoral", "research fellow", "life science", "statistician intern",
    ),
    "finance_risk_quant": (
        "quantitative", "quant researcher", "risk", "credit", "finance", "accounting", "audit",
        "capital market", "portfolio", "stock controller", "asset management", "investment",
    ),
    "data_engineering": (
        "data engineer", "bi engineer", "data manager", "data sourcing", "data platform",
        "data & reporting", "data reporting",
    ),
    "ml_ai_research": (
        "machine learning", "artificial intelligence", " ai ", "nlp", "deep learning",
        "applied scientist", "research engineer", "ml engineer", "simulation research",
    ),
    "data_science_analytics": (
        "data scientist", "data analyst", "analytics", "data science", "research officer",
        "research assistant",
    ),
    "software_security_engineering": (
        "software engineer", "software developer", "developer", "full-stack", "full stack",
        "python developer", "platform engineer", "penetration tester", "technical support engineer",
    ),
    "operations_infrastructure": (
        "operator", "controller", "broadcast", "maintenance", "electrician", "ocean import",
        "control room", "cio", "material controller", "linux platform", "real estate",
    ),
    "business_policy_management": (
        "manager", "policy", "consultant", "campaign", "alliances", "data protection",
        "associate", "business development", "student development", "head of",
    ),
}

INTENT_TERMS = {
    "responsibilities": (
        "responsibil", "accountabil", "you will", "role", "deliver", "develop", "build", "manage",
        "lead", "collaborat", "support", "design", "implement", "maintain", "create", "work with",
    ),
    "technical_skills": (
        "python", "sql", "java", "javascript", "react", "machine learning", "deep learning", "nlp",
        "cloud", "aws", "azure", "gcp", "docker", "kubernetes", "spark", "airflow", "linux",
        "tableau", "power bi", "pytorch", "tensorflow", "statistics", "algorithm", "api", "database",
        "technology", "digital", "architecture", "enterprise service", "vendor", "analyse", "analysis",
        "evidence", "cost-benefit", "modelling", "modeling",
    ),
    "qualifications": (
        "experience", "qualification", "degree", "bachelor", "master", "phd", "education", "years",
        "essential", "required", "proven", "knowledge", "ability", "background", "certification",
    ),
    "work_context": (
        "location", "remote", "hybrid", "office", "benefit", "salary", "pension", "holiday", "leave",
        "culture", "diversity", "inclusion", "travel", "working hours", "flexible", "team", "company",
    ),
}

CONCEPT_LABELS = {
    "machine learning": "机器学习",
    "deep learning": "深度学习",
    "natural language processing": "自然语言处理",
    "large language models": "大语言模型",
    "data pipeline": "数据管道",
    "data product": "数据产品",
    "predictive model": "预测模型",
    "production system": "生产系统",
    "stakeholder": "利益相关方协作",
    "credit risk": "信用风险",
    "market risk": "市场风险",
    "quantitative research": "量化研究",
    "clinical data": "临床数据",
    "genomic": "基因组数据",
    "bioinformatic": "生物信息学",
    "cloud": "云平台",
    "software development": "软件开发",
    "technical architecture": "技术架构",
    "architecture": "企业技术架构",
    "technology strategy": "技术战略",
    "enterprise service": "企业级服务",
    "solution design": "解决方案设计",
    "critical system": "关键业务系统",
    "data visual": "数据可视化",
    "dashboard": "分析仪表板",
    "broadcast chain": "广播链路",
    "troubleshoot": "故障排查",
    "cyber security": "网络安全",
    "financial report": "财务报告",
    "product analytics": "产品分析",
    "business requirement": "业务需求",
    "cross-functional": "跨职能协作",
    "publication": "研究发表",
    "experimental design": "实验设计",
    "risk management": "风险管理",
    "credit rating": "信用评级",
    "structured finance": "结构化金融",
    "securitization": "证券化分析",
    "clo": "CLO产品",
    "cost-benefit": "成本收益分析",
    "policy analysis": "政策分析",
    "data governance": "数据治理",
    "data quality": "数据质量",
    "data management": "数据管理",
    "project management": "项目管理",
    "people management": "团队管理",
    "product development": "产品开发",
    "capital market": "资本市场",
    "regulatory": "监管政策",
    "customer service": "客户服务",
    "benefit": "员工福利",
    "salary": "薪酬安排",
    "remote": "远程办公",
    "hybrid": "混合办公",
    "diversity": "多元与包容",
    "Python": "Python",
    "SQL": "SQL",
    "AWS": "AWS",
    "Azure": "Azure",
    "GCP": "GCP",
    "Docker": "Docker",
    "Kubernetes": "Kubernetes",
    "React": "React",
    "Linux": "Linux",
    "Tableau": "Tableau",
}
EXACT_CONCEPTS = {"clo", "python", "sql", "aws", "azure", "gcp", "docker", "kubernetes", "react", "linux", "tableau"}

STOPWORDS = {
    "about", "after", "also", "among", "and", "are", "been", "being", "business", "candidate",
    "company", "could", "from", "have", "into", "more", "must", "other", "role", "should", "team",
    "that", "their", "these", "they", "this", "through", "using", "what", "when", "where", "which",
    "will", "with", "work", "working", "would", "years", "your", "you", "job", "position", "skills",
    "opportunity", "time", "life", "option", "global", "people", "including", "across", "within", "make",
    "provide", "help", "looking", "offer", "part", "well", "need", "want", "great", "service",
}

QUERY_TEMPLATES = {
    "responsibilities": (
        "围绕{focus}开展工作时，入职后需要承担哪些核心交付和协作职责？",
        "如果实际负责{focus}相关任务，这个岗位日常要推进哪些工作？",
        "在{focus}这条工作线上，岗位对成果交付和跨团队配合有什么具体要求？",
        "从执行角度看，处理{focus}时需要负责哪些环节和结果？",
        "该岗位会怎样参与{focus}，个人需要直接承担哪些责任？",
    ),
    "technical_skills": (
        "为了完成{focus}相关工作，需要掌握哪些技术、工具和分析方法？",
        "在处理{focus}时，招聘方看重哪些工程或技术能力？",
        "候选人要胜任{focus}，技术栈和方法论方面需要具备什么？",
        "实现{focus}涉及哪些编程语言、平台或专业工具？",
        "针对{focus}场景，岗位要求哪些可以落地使用的技术能力？",
    ),
    "qualifications": (
        "针对{focus}这部分工作，经验年限、学历或专业背景有哪些门槛？",
        "申请人要承担{focus}，过往经历和教育背景需要达到什么程度？",
        "招聘方如何要求候选人在{focus}方面的经验、知识和资质？",
        "胜任{focus}需要哪些可证明的从业经历或学术训练？",
        "关于{focus}，岗位列出了哪些必备资格和背景条件？",
    ),
    "work_context": (
        "除核心职责外，与{focus}相关的团队环境、办公安排或员工支持是怎样的？",
        "这个岗位开展{focus}时，地点、协作方式和工作环境有哪些说明？",
        "招聘信息对{focus}所在的组织环境、灵活办公或福利支持如何描述？",
        "如果加入后参与{focus}，还需要了解哪些工作方式、地点或团队条件？",
        "围绕{focus}的实际工作环境，岗位还提供了哪些安排或支持？",
    ),
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _family(record) -> str:
    text = f" {record.job_title} {record.description[:1500]} ".lower()
    scores = {
        family: sum(3 if term in record.job_title.lower() else 1 for term in terms if term in text)
        for family, terms in FAMILY_RULES.items()
    }
    return max(scores, key=lambda family: (scores[family], family))


def _sentences(text: str) -> list[str]:
    normalized = re.sub(r"\r\n?", "\n", text)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n+", normalized):
        for line in paragraph.splitlines():
            line = re.sub(r"^[\s•*\-–—]+", "", line).strip()
            if not line:
                continue
            split = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line)
            pieces.extend(item.strip() for item in split if len(item.strip()) >= 18)
    return pieces


def _atomic_units(text: str) -> list[str]:
    sentences = _sentences(text)
    units: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        words = sentence.split()
        if current and current_words + len(words) > 105:
            units.append(" ".join(current))
            current, current_words = [], 0
        current.append(sentence)
        current_words += len(words)
        if current_words >= 55:
            units.append(" ".join(current))
            current, current_words = [], 0
    if current:
        if units and current_words < 28:
            units[-1] = f"{units[-1]} {' '.join(current)}"
        else:
            units.append(" ".join(current))

    while len(units) < 8:
        index = max(range(len(units)), key=lambda idx: len(units[idx].split()))
        words = units[index].split()
        if len(words) < 18:
            break
        middle = len(words) // 2
        units[index : index + 1] = [" ".join(words[:middle]), " ".join(words[middle:])]
    if len(units) < 8:
        words = " ".join(units).split()
        chunk_size = max(1, math.ceil(len(words) / 8))
        units = [" ".join(words[index : index + chunk_size]) for index in range(0, len(words), chunk_size)]
        while len(units) < 8 and any(len(unit.split()) >= 2 for unit in units):
            index = max(range(len(units)), key=lambda idx: len(units[idx].split()))
            item_words = units[index].split()
            middle = max(1, len(item_words) // 2)
            units[index : index + 1] = [" ".join(item_words[:middle]), " ".join(item_words[middle:])]
    while len(units) > 20:
        index = min(range(len(units) - 1), key=lambda idx: len(units[idx].split()) + len(units[idx + 1].split()))
        units[index : index + 2] = [f"{units[index]} {units[index + 1]}"]
    return [unit.strip() for unit in units if unit.strip()]


def _intent_scores(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {
        intent: sum(bool(re.search(rf"\b{re.escape(term)}\w*\b", lowered)) for term in terms)
        for intent, terms in INTENT_TERMS.items()
    }


def _primary_intent(text: str) -> str:
    scores = _intent_scores(text)
    return max(scores, key=lambda intent: (scores[intent], -QUERY_TYPES.index(intent)))


def _focus(texts: list[str]) -> str:
    combined = " ".join(texts)
    lowered = combined.lower()
    found = []
    for concept, label in CONCEPT_LABELS.items():
        suffix = "" if concept.lower() in EXACT_CONCEPTS else r"\w*"
        if re.search(rf"\b{re.escape(concept.lower())}{suffix}\b", lowered):
            found.append(label)
    if found:
        return "、".join(found[:2])
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z-]{3,}", combined)
        if token.lower() not in STOPWORDS
    ]
    phrase_counts = Counter(
        f"{left} {right}" for left, right in zip(tokens, tokens[1:]) if left != right
    )
    selected = [phrase for phrase, _ in phrase_counts.most_common(2)]
    return "、".join(selected) if selected else "岗位核心业务"


def _select_relevant(units: list[dict], query_type: str, *, multi: bool) -> list[dict]:
    ranked = sorted(
        units,
        key=lambda unit: (
            -unit["intent_scores"][query_type],
            unit["primary_intent"] != query_type,
            unit["unit_index"],
        ),
    )
    count = 2 if multi and len(ranked) >= 2 else 1
    return ranked[:count]


def _query_text(job: dict, query_type: str, relevant: list[dict], used: set[str]) -> str:
    focus = _focus([item["text"] for item in relevant])
    offset = int(_sha256(f"{job['job_id']}:{query_type}")[:8], 16) % len(QUERY_TEMPLATES[query_type])
    query = QUERY_TEMPLATES[query_type][offset].format(focus=focus)
    base = query.rstrip("？")
    if query in used:
        query = base + f"，在“{job['job_title']}”的招聘语境下如何理解？"
    if query in used:
        query = base + f"，请依据{job['company_name']}发布的该职位原文说明？"
    if query in used:
        raise ValueError(f"could not construct a unique query for {job['job_id']} {query_type}")
    used.add(query)
    return query


def _lexical_terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z-]{3,}", text)
        if token.lower() not in STOPWORDS
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


def _stratified_split(jobs: list[dict]) -> dict[str, str]:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for job in jobs:
        by_family[job["occupation_family"]].append(job)
    target_test = round(len(jobs) * 0.30)
    quotas = {family: max(1, math.floor(len(items) * 0.30)) for family, items in by_family.items()}
    remaining = target_test - sum(quotas.values())
    remainders = sorted(
        by_family,
        key=lambda family: (-(len(by_family[family]) * 0.30 - math.floor(len(by_family[family]) * 0.30)), family),
    )
    for family in remainders[:remaining]:
        quotas[family] += 1
    split: dict[str, str] = {}
    for family, family_jobs in by_family.items():
        ordered = sorted(family_jobs, key=lambda job: _sha256(job["job_id"]))
        test_count = quotas[family]
        for index, job in enumerate(ordered):
            split[job["job_id"]] = "test" if index < test_count else "development"
    return split


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _manifest(directory: Path, counts: dict, family_counts: dict, split_counts: dict) -> dict:
    files = {}
    for filename in PUBLISHED_JSONL_FILES:
        path = directory / filename
        files[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path.read_text(encoding="utf-8"))}
    return {
        "name": "job_retrieval_v2",
        "version": VERSION,
        "annotation_status": "silver_expert_review_required",
        "selection_policy": "development_only_then_single_selected_model_on_frozen_test",
        "counts": counts,
        "occupation_family_distribution": family_counts,
        "split_distribution": split_counts,
        "local_artifacts": {
            "annotation_tasks.jsonl": {
                "rows": counts["annotation_tasks"],
                "versioned": False,
                "regenerate_with": "python -m tools.build_retrieval_v2_dataset",
            }
        },
        "files": files,
    }


def build_dataset(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [record for record in JobCatalog(settings.job_catalog_path).all_records() if record.source_kind == "open_source"]
    records = sorted(records, key=lambda record: (-len(record.description), record.job_id))[:TARGET_JOBS]
    if len(records) != TARGET_JOBS:
        raise ValueError(f"expected {TARGET_JOBS} public jobs, found {len(records)}")

    jobs = [
        {
            "job_id": record.job_id,
            "company_name": record.company_name,
            "job_title": record.job_title,
            "description": record.description,
            "location": record.location,
            "language": record.language,
            "source_dataset": record.source_dataset,
            "source_file": record.source_file,
            "source_url": record.source_url,
            "content_hash": record.content_hash,
            "occupation_family": _family(record),
            "historical_snapshot": True,
        }
        for record in records
    ]
    splits = _stratified_split(jobs)
    for job in jobs:
        job["split"] = splits[job["job_id"]]

    evidence: list[dict] = []
    evidence_by_job: dict[str, list[dict]] = {}
    for job in jobs:
        units = []
        for index, text in enumerate(_atomic_units(job["description"])):
            scores = _intent_scores(text)
            unit = {
                "evidence_id": f"ev_{job['job_id'][4:]}_{index:02d}",
                "job_id": job["job_id"],
                "occupation_family": job["occupation_family"],
                "split": job["split"],
                "unit_index": index,
                "text": text,
                "primary_intent": _primary_intent(text),
                "intent_scores": scores,
                "content_sha256": _sha256(text),
            }
            units.append(unit)
            evidence.append(unit)
        if not 8 <= len(units) <= 20:
            raise ValueError(f"{job['job_id']} produced {len(units)} evidence units")
        evidence_by_job[job["job_id"]] = units

    queries: list[dict] = []
    qrels: list[dict] = []
    used_queries: set[str] = set()
    for job in jobs:
        units = evidence_by_job[job["job_id"]]
        for query_index, query_type in enumerate(QUERY_TYPES):
            multi = query_index < 2
            relevant = _select_relevant(units, query_type, multi=multi)
            query_id = f"q_{job['job_id'][4:]}_{query_index + 1}"
            query = _query_text(job, query_type, relevant, used_queries)
            queries.append(
                {
                    "query_id": query_id,
                    "job_id": job["job_id"],
                    "occupation_family": job["occupation_family"],
                    "split": job["split"],
                    "query_type": query_type,
                    "query": query,
                    "focus": _focus([item["text"] for item in relevant]),
                    "annotation_status": "silver_expert_review_required",
                }
            )
            for rank, unit in enumerate(relevant):
                qrels.append(
                    {
                        "query_id": query_id,
                        "evidence_id": unit["evidence_id"],
                        "relevance_grade": 3 if rank == 0 else 2,
                        "label_source": "deterministic_query_construction",
                        "annotator_1": None,
                        "annotator_2": None,
                        "adjudicated_grade": None,
                        "status": "expert_review_required",
                    }
                )

    evidence_by_id = {item["evidence_id"]: item for item in evidence}
    lexical_by_id = {item["evidence_id"]: _lexical_terms(item["text"]) for item in evidence}
    qrels_by_query: dict[str, list[dict]] = defaultdict(list)
    for qrel in qrels:
        qrels_by_query[qrel["query_id"]].append(qrel)
    candidate_pools: list[dict] = []
    annotation_tasks: list[dict] = []
    for query in queries:
        query_terms = _lexical_terms(query["query"])
        relevant_ids = {item["evidence_id"] for item in qrels_by_query[query["query_id"]]}
        same_job = evidence_by_job[query["job_id"]]
        candidates: list[dict] = [
            {"evidence_id": item["evidence_id"], "candidate_type": "relevant" if item["evidence_id"] in relevant_ids else "same_job_hard_negative"}
            for item in same_job
        ]
        selected_ids = {item["evidence_id"] for item in candidates}
        cross_job = [
            item for item in evidence
            if item["job_id"] != query["job_id"] and item["occupation_family"] == query["occupation_family"]
        ]
        cross_job.sort(
            key=lambda item: (
                -_jaccard(query_terms, lexical_by_id[item["evidence_id"]]),
                item["primary_intent"] != query["query_type"],
                item["evidence_id"],
            )
        )
        for item in cross_job:
            if len(candidates) >= CANDIDATE_POOL_SIZE:
                break
            if item["evidence_id"] not in selected_ids:
                candidates.append({"evidence_id": item["evidence_id"], "candidate_type": "same_family_hard_negative"})
                selected_ids.add(item["evidence_id"])
        if len(candidates) < CANDIDATE_POOL_SIZE:
            fallback = sorted(
                (item for item in evidence if item["evidence_id"] not in selected_ids),
                key=lambda item: (-_jaccard(query_terms, lexical_by_id[item["evidence_id"]]), item["evidence_id"]),
            )
            for item in fallback[: CANDIDATE_POOL_SIZE - len(candidates)]:
                candidates.append({"evidence_id": item["evidence_id"], "candidate_type": "cross_family_hard_negative"})
        candidates = candidates[:CANDIDATE_POOL_SIZE]
        candidate_pools.append({"query_id": query["query_id"], "candidates": candidates})
        annotation_tasks.append(
            {
                "query_id": query["query_id"],
                "query": query["query"],
                "job_id": query["job_id"],
                "split": query["split"],
                "candidate_passages": [
                    {
                        "evidence_id": item["evidence_id"],
                        "text": evidence_by_id[item["evidence_id"]]["text"],
                        "silver_grade": next(
                            (
                                qrel["relevance_grade"]
                                for qrel in qrels_by_query[query["query_id"]]
                                if qrel["evidence_id"] == item["evidence_id"]
                            ),
                            0,
                        ),
                        "annotator_1_grade": None,
                        "annotator_2_grade": None,
                        "adjudicated_grade": None,
                    }
                    for item in candidates
                ],
            }
        )

    _write_jsonl(output_dir / "job_snapshots.jsonl", jobs)
    _write_jsonl(output_dir / "evidence_units.jsonl", evidence)
    _write_jsonl(output_dir / "queries.jsonl", queries)
    _write_jsonl(output_dir / "qrels.jsonl", qrels)
    _write_jsonl(output_dir / "candidate_pools.jsonl", candidate_pools)
    _write_jsonl(output_dir / "annotation_tasks.jsonl", annotation_tasks)

    family_counts = dict(sorted(Counter(job["occupation_family"] for job in jobs).items()))
    split_counts = dict(sorted(Counter(job["split"] for job in jobs).items()))
    counts = {
        "jobs": len(jobs),
        "evidence_units": len(evidence),
        "queries": len(queries),
        "qrels": len(qrels),
        "multi_relevant_queries": sum(len(qrels_by_query[item["query_id"]]) > 1 for item in queries),
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "annotation_tasks": len(annotation_tasks),
    }
    manifest = _manifest(output_dir, counts, family_counts, split_counts)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    output_dir = settings.data_dir / "eval_dataset" / "job_retrieval_v2"
    manifest = build_dataset(output_dir)
    print(json.dumps({"output_dir": str(output_dir), **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
