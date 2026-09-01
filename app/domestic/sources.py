from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from app.knowledge.models import NormalizedJob


USER_AGENT = "AIJobApplyAssistant/1.0 (+personal campus job research; contact local user)"
DEFAULT_TIMEOUT_SECONDS = 30


def compact_text(value: str) -> str:
    lines = [" ".join(line.split()) for line in value.replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def classify_job(title: str, description: str) -> str:
    value = f"{title}\n{description}".lower()
    if any(term in value for term in ("agent", "智能体", "tool calling", "工具调用", "多agent")):
        return "agent_development"
    if any(term in value for term in ("大模型应用", "llm应用", "rag", "知识库", "智能问答")):
        return "llm_application"
    if any(term in value for term in (
        "ai应用", "aigc应用", "ai原生", "人工智能应用", "ai全栈", "智能应用开发",
        "空间智能应用", "应用算法",
    )):
        return "ai_application"
    if any(term in value for term in (
        "ai软件", "ai平台", "算法平台", "ai后端", "模型服务", "ai devops", "ai系统",
    )):
        return "ai_software"
    if any(term in value for term in ("预训练", "强化学习", "cuda", "算子", "推理部署", "基础模型")):
        return "foundation_or_infra"
    return "other"


def infer_degree_requirement(text: str) -> str | None:
    if "博士" in text:
        return "博士"
    if "硕士" in text or "研究生" in text:
        return "硕士"
    if "本科" in text:
        return "本科"
    if "大专" in text:
        return "大专"
    return None


def infer_major_requirement(text: str) -> str | None:
    match = re.search(
        r"((?:计算机|人工智能|软件工程|电子信息|自动化|数学|通信工程)[^，。；\n]{0,50}(?:相关专业|专业))",
        text,
    )
    return compact_text(match.group(1)) if match else None


def stable_external_id(source_id: str, title: str, url: str) -> str:
    value = f"{source_id}\n{title}\n{url}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:24]


def epoch_millis_to_iso(value: object) -> str | None:
    try:
        timestamp = int(value) / 1000
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class SourceMetadata:
    source_id: str
    name: str
    source_type: str
    base_url: str
    terms_url: str | None = None
    robots_url: str | None = None
    schedule_minutes: int = 360


class DomesticSourceError(RuntimeError):
    pass


class DomesticJobSource(ABC):
    metadata: SourceMetadata

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"})

    def _robots_allows(self, url: str) -> bool:
        robots_url = self.metadata.robots_url
        if not robots_url:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        try:
            response = self.session.get(robots_url, timeout=10)
            if response.status_code == 404:
                return True
            content_type = response.headers.get("content-type", "").lower()
            if response.status_code != 200 or "html" in content_type:
                return True
            parser = RobotFileParser()
            parser.set_url(robots_url)
            parser.parse(response.text.splitlines())
            return parser.can_fetch(USER_AGENT, url)
        except requests.RequestException:
            return True

    def _get_text(self, url: str) -> str:
        if not self._robots_allows(url):
            raise DomesticSourceError(f"robots.txt disallows automated access: {url}")
        response = self.session.get(url, timeout=DEFAULT_TIMEOUT_SECONDS)
        response.raise_for_status()
        if not response.encoding or response.encoding.lower() == "iso-8859-1":
            response.encoding = response.apparent_encoding or "utf-8"
        return response.text

    def _post_json(
        self, url: str, payload: dict, *, headers: dict[str, str] | None = None
    ) -> dict:
        if not self._robots_allows(url):
            raise DomesticSourceError(f"robots.txt disallows automated access: {url}")
        response = self.session.post(
            url,
            json=payload,
            headers=headers,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise DomesticSourceError(f"unexpected JSON payload from {url}")
        return data

    @abstractmethod
    def fetch(self) -> list[NormalizedJob]:
        raise NotImplementedError


class BaiduCampusSource(DomesticJobSource):
    metadata = SourceMetadata(
        source_id="baidu_campus_2027_aidu",
        name="百度 2027 AIDU 校园招聘",
        source_type="official_career",
        base_url="https://talent.baidu.com/jobs/list?projectType=3&recruitType=GRADUATE",
        terms_url="https://talent.baidu.com/",
    )

    def fetch(self) -> list[NormalizedJob]:
        html = self._get_text(self.metadata.base_url)
        marker = "window.__INITIAL_DATA__ ="
        if marker not in html:
            raise DomesticSourceError("Baidu initial job data was not found")
        raw_payload = html[html.index(marker) + len(marker):].lstrip()
        # The page occasionally serializes an absent JavaScript value as
        # `undefined`, which is valid JavaScript but invalid JSON.
        raw_payload = re.sub(
            r"(?<=:)\s*undefined(?=\s*[,}])",
            "null",
            raw_payload,
        )
        payload, _ = json.JSONDecoder().raw_decode(raw_payload)
        rows = payload.get("listData", {}).get("listDetailData", [])
        jobs: list[NormalizedJob] = []
        for row in rows:
            title = compact_text(str(row.get("name") or ""))
            responsibilities = compact_text(str(row.get("workContent") or ""))
            requirements = compact_text(str(row.get("serviceCondition") or ""))
            description = compact_text(
                f"岗位职责：\n{responsibilities}\n任职要求：\n{requirements}"
            )
            if not title or not description:
                continue
            external_id = str(row.get("jobId") or row.get("postId") or "")
            jobs.append(
                NormalizedJob(
                    company_name="百度",
                    job_title=title,
                    description=description,
                    location=compact_text(str(row.get("workPlace") or "")) or None,
                    source_kind="open_source",
                    source_dataset=self.metadata.source_id,
                    source_file=f"baidu_{external_id or stable_external_id(self.metadata.source_id, title, self.metadata.base_url)}.json",
                    source_url=self.metadata.base_url,
                    language="zh",
                    external_id=external_id or stable_external_id(self.metadata.source_id, title, self.metadata.base_url),
                    apply_url=self.metadata.base_url,
                    source_name=self.metadata.name,
                    recruitment_type="campus",
                    graduation_year=2027,
                    degree_requirement=infer_degree_requirement(requirements),
                    major_requirement=infer_major_requirement(requirements),
                    job_category=classify_job(title, description),
                    employment_type="full_time",
                    posted_at=str(row.get("updateDate") or row.get("publishDate") or "") or None,
                    status="open",
                    is_domestic=True,
                    raw_payload_json=json.dumps(row, ensure_ascii=False, sort_keys=True),
                )
            )
        return jobs


class DjiCampusHotJobsSource(DomesticJobSource):
    metadata = SourceMetadata(
        source_id="dji_campus_2027_hot_jobs",
        name="大疆 2027 拓疆者校园招聘",
        source_type="official_career",
        base_url="https://careers.dji.com/zh-CN/campus/hot-jobs?source=campus_hotjobs",
        terms_url="https://careers.dji.com/zh-CN/",
    )

    def fetch(self) -> list[NormalizedJob]:
        html = self._get_text(self.metadata.base_url)
        soup = BeautifulSoup(html, "html.parser")
        jobs: list[NormalizedJob] = []
        for card in soup.select('div[class*="pc_card"]'):
            heading = card.find("h4")
            intro = card.find("p")
            link = card.find("a", href=True)
            if heading is None or intro is None:
                continue
            title = compact_text(heading.get_text(" ", strip=True))
            description = compact_text(intro.get_text(" ", strip=True))
            locations = [
                compact_text(tag.get_text(" ", strip=True))
                for tag in card.select('span[class*="pc_tag"]')
                if compact_text(tag.get_text(" ", strip=True))
            ]
            apply_url = urljoin(
                self.metadata.base_url,
                str(link.get("href")) if link else self.metadata.base_url,
            )
            external_id = stable_external_id(self.metadata.source_id, title, apply_url)
            jobs.append(
                NormalizedJob(
                    company_name="大疆创新",
                    job_title=title,
                    description=description,
                    location="、".join(locations) or None,
                    source_kind="open_source",
                    source_dataset=self.metadata.source_id,
                    source_file=f"dji_{external_id}.html",
                    source_url=self.metadata.base_url,
                    language="zh",
                    external_id=external_id,
                    apply_url=apply_url,
                    source_name=self.metadata.name,
                    recruitment_type="campus",
                    graduation_year=2027,
                    degree_requirement=infer_degree_requirement(description),
                    major_requirement=infer_major_requirement(description),
                    job_category=classify_job(title, description),
                    employment_type="full_time",
                    status="open",
                    is_domestic=True,
                    raw_payload_json=json.dumps(
                        {"title": title, "locations": locations, "description": description},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )
            )
        return jobs


class KylinCampusSource(DomesticJobSource):
    metadata = SourceMetadata(
        source_id="kylin_campus_2027",
        name="麒麟软件 2027 校园招聘",
        source_type="official_career",
        base_url="https://www.kylinos.cn/about/job/campusRecruitment/index.html",
        terms_url="https://www.kylinos.cn/about/job/campusRecruitment/index.html",
    )

    def fetch(self) -> list[NormalizedJob]:
        html = self._get_text(self.metadata.base_url)
        soup = BeautifulSoup(html, "html.parser")
        jobs: list[NormalizedJob] = []
        for item in soup.select("li.content-item"):
            heading = item.find("h4")
            section = item.find("section", class_="item-section")
            if heading is None or section is None:
                continue
            title = compact_text(heading.get_text(" ", strip=True))
            full_text = compact_text(section.get_text("\n", strip=True))
            location_match = re.search(r"工作地点[：:]\s*([^\n]+)", full_text)
            date_match = re.search(r"发布时间[：:]\s*(\d{4}-\d{2}-\d{2})", full_text)
            link = section.find("a", href=True)
            apply_url = urljoin(
                self.metadata.base_url,
                str(link.get("href")) if link else self.metadata.base_url,
            )
            query = parse_qs(urlparse(apply_url).query)
            external_id = (query.get("jobAdId") or [""])[0]
            if not external_id:
                external_id = stable_external_id(self.metadata.source_id, title, apply_url)
            jobs.append(
                NormalizedJob(
                    company_name="麒麟软件",
                    job_title=title,
                    description=full_text,
                    location=compact_text(location_match.group(1)) if location_match else None,
                    source_kind="open_source",
                    source_dataset=self.metadata.source_id,
                    source_file=f"kylin_{external_id}.html",
                    source_url=self.metadata.base_url,
                    language="zh",
                    external_id=external_id,
                    apply_url=apply_url,
                    source_name=self.metadata.name,
                    recruitment_type="campus",
                    graduation_year=2027,
                    degree_requirement=infer_degree_requirement(full_text),
                    major_requirement=infer_major_requirement(full_text),
                    job_category=classify_job(title, full_text),
                    employment_type="full_time",
                    posted_at=date_match.group(1) if date_match else None,
                    status="open",
                    is_domestic=True,
                    raw_payload_json=json.dumps(
                        {"title": title, "text": full_text, "apply_url": apply_url},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )
            )
        return jobs


class OppoCampusSource(DomesticJobSource):
    metadata = SourceMetadata(
        source_id="oppo_campus_2027_technical",
        name="OPPO 2027 校园招聘技术岗位",
        source_type="official_api",
        base_url="https://careers.oppo.com/university/oppo/campus/",
        terms_url="https://careers.oppo.com/",
    )
    api_url = "https://careers.oppo.com/openapi/position/pageNew"
    target_position_types = {"AI/算法类", "软件类", "测试类", "工程技术类"}
    target_terms = (
        "ai", "人工智能", "大模型", "智能体", "agent", "llm", "rag", "aigc",
        "全栈", "后端", "服务端", "it开发", "应用开发", "软件开发",
    )

    def fetch(self) -> list[NormalizedJob]:
        jobs: list[NormalizedJob] = []
        page = 1
        total = 1
        while (page - 1) * 100 < total and page <= 10:
            payload = {
                "pageNum": page,
                "pageSize": 100,
                "positionName": "",
                "projectList": [],
                "positionTypeList": [],
                "workCityCodeList": [],
                "shareId": "",
            }
            result = self._post_json(
                self.api_url, payload, headers={"Tenant-Id": "1000"}
            )
            if result.get("code") != 0:
                raise DomesticSourceError(f"OPPO API error: {result.get('msg')}")
            data = result.get("data") or {}
            total = int(data.get("total") or 0)
            rows = data.get("records") or []
            for row in rows:
                project_name = compact_text(str(row.get("projectName") or ""))
                title = compact_text(str(row.get("positionName") or ""))
                position_type = compact_text(str(row.get("positionTypeName") or ""))
                if "2027" not in project_name or "博士" in f"{project_name}{title}":
                    continue
                duty = compact_text(str(row.get("positionDesc") or ""))
                requirement = compact_text(str(row.get("positionRequire") or ""))
                ai_capability = compact_text(str(row.get("aiCapabilityLevelDesc") or ""))
                searchable = f"{title}\n{duty}\n{requirement}".lower()
                if position_type not in self.target_position_types:
                    continue
                if not any(term in searchable for term in self.target_terms):
                    continue
                description = compact_text(
                    f"岗位职责：\n{duty}\n任职要求：\n{requirement}"
                    + (f"\nAI能力要求：\n{ai_capability}" if ai_capability else "")
                )
                external_id = str(row.get("idProjPosition") or row.get("idRecruitPosition") or "")
                if not external_id or not description:
                    continue
                recruit_type = str(row.get("recruitmentType") or "").lower()
                recruitment_type = "internship" if "intern" in recruit_type else "campus"
                apply_url = (
                    "https://careers.oppo.com/university/oppo/campus/post/"
                    f"{external_id}?recruitType={row.get('recruitmentType') or 'Graduate'}"
                )
                jobs.append(
                    NormalizedJob(
                        company_name="OPPO",
                        job_title=title,
                        description=description,
                        location=compact_text(str(row.get("workCityName") or "")) or None,
                        source_kind="open_source",
                        source_dataset=self.metadata.source_id,
                        source_file=f"oppo_{external_id}.json",
                        source_url=self.metadata.base_url,
                        language="zh",
                        external_id=external_id,
                        apply_url=apply_url,
                        source_name=self.metadata.name,
                        recruitment_type=recruitment_type,
                        graduation_year=2027,
                        degree_requirement=infer_degree_requirement(requirement),
                        major_requirement=infer_major_requirement(requirement),
                        job_category=classify_job(title, description),
                        employment_type=("internship" if recruitment_type == "internship" else "full_time"),
                        posted_at=str(row.get("releaseTime") or "") or None,
                        status="open",
                        is_domestic=True,
                        raw_payload_json=json.dumps(row, ensure_ascii=False, sort_keys=True),
                    )
                )
            if not rows:
                break
            page += 1
        return jobs


class MeituanCampusAiSource(DomesticJobSource):
    metadata = SourceMetadata(
        source_id="meituan_campus_ai_technical",
        name="美团校园招聘 AI/大模型技术岗位",
        source_type="official_api",
        base_url="https://careers.meituan.com/position",
        terms_url="https://careers.meituan.com/",
    )
    api_url = "https://careers.meituan.com/api/official/job/getJobList"
    keywords = ("AI Agent", "大模型", "RAG", "智能体", "AIGC", "AI应用", "LLM")

    def fetch(self) -> list[NormalizedJob]:
        rows_by_id: dict[str, dict] = {}
        for keyword in self.keywords:
            payload = {
                "page": {"pageNo": 1, "pageSize": 50},
                "jobShareType": "1",
                "keywords": keyword,
                "cityList": [],
                "department": [],
                "jfJgList": [],
                "jobType": [
                    {"code": "1", "subCode": []},
                    {"code": "2", "subCode": []},
                ],
                "typeCode": [],
                "specialCode": [],
            }
            result = self._post_json(self.api_url, payload)
            if result.get("status") not in {0, 1, "0", "1", None}:
                raise DomesticSourceError(f"Meituan API error: {result.get('message')}")
            for row in (result.get("data") or {}).get("list") or []:
                external_id = str(row.get("jobUnionId") or "")
                if external_id and row.get("jobFamily") == "技术类":
                    rows_by_id[external_id] = row

        jobs: list[NormalizedJob] = []
        for external_id, row in rows_by_id.items():
            if str(row.get("jobStatus") or "000") != "000":
                continue
            title = compact_text(str(row.get("name") or ""))
            duty = compact_text(str(row.get("jobDuty") or ""))
            requirement = compact_text(str(row.get("jobRequirement") or ""))
            if not title or not (duty or requirement):
                continue
            description = compact_text(f"岗位职责：\n{duty}\n任职要求：\n{requirement}")
            cities = [
                compact_text(str(item.get("name") or ""))
                for item in row.get("cityList") or []
                if compact_text(str(item.get("name") or ""))
            ]
            recruitment_type = "internship" if str(row.get("jobType")) == "2" else "campus"
            apply_url = (
                "https://careers.meituan.com/position/detail?jobUnionId="
                f"{external_id}&highlightType=campus"
            )
            jobs.append(
                NormalizedJob(
                    company_name="美团",
                    job_title=title,
                    description=description,
                    location="、".join(cities) or None,
                    source_kind="open_source",
                    source_dataset=self.metadata.source_id,
                    source_file=f"meituan_{external_id}.json",
                    source_url=self.metadata.base_url,
                    language="zh",
                    external_id=external_id,
                    apply_url=apply_url,
                    source_name=self.metadata.name,
                    recruitment_type=recruitment_type,
                    graduation_year=2027,
                    degree_requirement=infer_degree_requirement(requirement),
                    major_requirement=infer_major_requirement(requirement),
                    job_category=classify_job(title, description),
                    employment_type=("internship" if recruitment_type == "internship" else "full_time"),
                    posted_at=epoch_millis_to_iso(row.get("refreshTime") or row.get("firstPostTime")),
                    status="open",
                    is_domestic=True,
                    raw_payload_json=json.dumps(row, ensure_ascii=False, sort_keys=True),
                )
            )
        return jobs


class HonorCampusHighlightsSource(DomesticJobSource):
    metadata = SourceMetadata(
        source_id="honor_campus_2027_highlights",
        name="荣耀 2027 校园招聘重点职位",
        source_type="official_career",
        base_url="https://www.honor.com/cn/career/",
        terms_url="https://www.honor.com/cn/career/",
    )

    def fetch(self) -> list[NormalizedJob]:
        html = self._get_text(self.metadata.base_url)
        soup = BeautifulSoup(html, "html.parser")
        rows: dict[str, tuple[str, str]] = {}
        for link in soup.find_all("a", href=True):
            apply_url = str(link.get("href") or "")
            if "postType=campus" not in apply_url or "postId=" not in apply_url:
                continue
            title = compact_text(link.get_text(" ", strip=True))
            post_id = (parse_qs(urlparse(apply_url).query).get("postId") or [""])[0]
            if title and post_id:
                rows[post_id] = (title, apply_url)
        jobs: list[NormalizedJob] = []
        for post_id, (title, apply_url) in rows.items():
            description = (
                f"荣耀招聘官网将“{title}”列为当前重点校招职位。"
                "官网公开首页未提供完整任职要求，投递和匹配分析前请打开详情页复核。"
            )
            jobs.append(
                NormalizedJob(
                    company_name="荣耀",
                    job_title=title,
                    description=description,
                    location=None,
                    source_kind="open_source",
                    source_dataset=self.metadata.source_id,
                    source_file=f"honor_{post_id}.html",
                    source_url=self.metadata.base_url,
                    language="zh",
                    external_id=post_id,
                    apply_url=apply_url,
                    source_name=self.metadata.name,
                    recruitment_type="campus",
                    graduation_year=2027,
                    degree_requirement=None,
                    major_requirement=None,
                    job_category=classify_job(title, description),
                    employment_type="full_time",
                    status="open",
                    is_domestic=True,
                    raw_payload_json=json.dumps(
                        {"title": title, "apply_url": apply_url, "detail_level": "title_only"},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )
            )
        return jobs


@dataclass(frozen=True)
class CuratedCampusNotice:
    slug: str
    company_name: str
    title: str
    source_url: str
    focus: str
    publisher: str
    company_token: str
    location: str | None = None
    apply_url: str | None = None
    posted_at: str | None = None


CURATED_2027_TECH_NOTICES: tuple[CuratedCampusNotice, ...] = (
    CuratedCampusNotice(
        "transsion", "传音控股", "2027 校招 AI/软件技术岗位合集（公开公告）",
        "https://job.hust.edu.cn/zpinfo1/2407476.htm",
        "AI 技术、算法、软件开发、测试和数据方向", "华中科技大学就业信息网", "传音",
        apply_url="https://transsion.zhiye.com/campus", posted_at="2026-08-25",
    ),
    CuratedCampusNotice(
        "uisee", "驭势科技", "2027 校招物理 AI/软件岗位合集（公开公告）",
        "https://job.hust.edu.cn/zpinfo1/2407233.htm",
        "物理 AI、自动驾驶算法和软件研发方向", "华中科技大学就业信息网", "驭势",
        posted_at="2026-08-11",
    ),
    CuratedCampusNotice(
        "hellogroup", "挚文集团", "2027 校招 Agent/软件岗位合集（公开公告）",
        "https://job.hust.edu.cn/zpinfo1/2407229.htm",
        "Agent 开发、AI 算法、服务端、Java、Web、全栈和后端开发", "华中科技大学就业信息网", "挚文",
        posted_at="2026-08-10",
    ),
    CuratedCampusNotice(
        "duxiaoman", "度小满", "2027 校招 AI 应用岗位合集（公开公告）",
        "https://career.nankai.edu.cn/correcruit/content/id/116782.html",
        "AI 全栈、AI 测试、AI 应用开发和 AI 基础设施方向", "南开大学就业信息网", "度小满",
        location="北京、上海", apply_url="https://campus.duxiaoman.com", posted_at="2026-08-18",
    ),
    CuratedCampusNotice(
        "stics", "中景芯创", "2027 校招 AI Agent/应用岗位合集（公开公告）",
        "https://career.nankai.edu.cn/correcruit/content/id/116908.html",
        "AI Agent、AI 应用、Java、测试开发、自动化算法和 BI", "南开大学就业信息网", "中景芯创",
        location="北京、上海", apply_url="https://stics.zhiye.com/campus", posted_at="2026-08-24",
    ),
    CuratedCampusNotice(
        "sugon", "中科曙光", "2027 校招大模型/智能体岗位合集（公开公告）",
        "https://career.nankai.edu.cn/correcruit/content/id/116837.html",
        "材料大模型、科学智能体、AI 计算框架、AI 系统、AI Infra 和软件研发", "南开大学就业信息网", "中科曙光",
        location="北京、天津、南京、西安等", apply_url="https://go.sugon.com/campus", posted_at="2026-08-20",
    ),
    CuratedCampusNotice(
        "pudutech", "普渡科技", "2027 校招 AI/软件技术岗位合集（公开公告）",
        "https://job.hust.edu.cn/zpinfo1/2406953.htm",
        "深度学习算法、机器人软件及相关技术研发方向", "华中科技大学就业信息网", "普渡",
        apply_url="https://pudutech.zhiye.com/campus", posted_at="2026-06-26",
    ),
    CuratedCampusNotice(
        "transwarp", "星环科技", "2027 校招智能体/大模型岗位合集（公开公告）",
        "https://career.nankai.edu.cn/correcruit/content/id/116990.html",
        "智能体开发、大模型算法、AI 测试、AI 部署、后端和分布式系统", "南开大学就业信息网", "星环科技",
        location="上海、北京、南京、广州、成都",
        apply_url="https://app.mokahr.com/campus_apply/transwarp", posted_at="2026-08-25",
    ),
    CuratedCampusNotice(
        "goertek", "歌尔集团", "2027 校招 AI/软件研发岗位合集（公开公告）",
        "https://myjob.dlmu.edu.cn/campus/view/id/868506",
        "软件研发、算法研发、机器视觉和应用设计开发", "大连海事大学就业信息网", "歌尔",
        location="潍坊、青岛、上海、北京、深圳等",
    ),
    CuratedCampusNotice(
        "sigmastar", "星宸科技", "2027 校招 AI/软件岗位合集（公开公告）",
        "https://career.nankai.edu.cn/correcruit/content/id/116535.html",
        "AI 算法、AI-ISP、Linux 驱动、嵌入式软件和编译工具链", "南开大学就业信息网", "星宸",
        location="上海、厦门、深圳、成都、杭州",
        apply_url="https://sigmastar.zhiye.com/campus/jobs", posted_at="2026-07-31",
    ),
    CuratedCampusNotice(
        "ronds", "容知日新", "2027 校招软件技术岗位合集（公开公告）",
        "https://myjob.dlmu.edu.cn/campus/view/id/868533",
        "智能策略软件、嵌入式软件、软件测试和工业 AI", "大连海事大学就业信息网", "容知日新",
        apply_url="https://job.ronds.com/campus-recruitment/anhuirohgzhirixin/",
    ),
    CuratedCampusNotice(
        "xiaomi", "小米集团", "2027 校招算法/软件研发岗位合集（公开公告）",
        "https://job.hust.edu.cn/zpinfo1/2407338.htm",
        "算法、软件研发及相关技术职类", "华中科技大学就业信息网", "小米",
        location="北京、上海、武汉、南京、西安、深圳", apply_url="https://hr.xiaomi.com/campus",
        posted_at="2026-08-18",
    ),
    CuratedCampusNotice(
        "cecloud", "中国电子云", "2027 校招 Agent/大模型岗位合集（公开公告）",
        "https://job.hust.edu.cn/zpinfo1/2407567.htm",
        "LLM、多模态、训推引擎、AI Coding、Agent、后端和大模型数据工程", "华中科技大学就业信息网", "中国电子云",
        location="北京、武汉、南京、杭州、成都", apply_url="https://www.cecloud.com/",
        posted_at="2026-08-27",
    ),
    CuratedCampusNotice(
        "dewu", "得物", "2027 校招算法/软件开发岗位合集（公开公告）",
        "https://job.hust.edu.cn/zpinfo1/2407279.htm",
        "算法策略、搜推广、Java、Golang、测试开发和客户端开发", "华中科技大学就业信息网", "得物",
        location="上海、杭州、北京、长沙", posted_at="2026-08-13",
    ),
    CuratedCampusNotice(
        "reolink", "睿联技术", "2027 校招智能算法/软件岗位合集（公开公告）",
        "https://job.hust.edu.cn/zpinfo1/2407507.htm",
        "智能算法、音视频处理、网络通信和软件研发", "华中科技大学就业信息网", "睿联技术",
        posted_at="2026-08-26",
    ),
    CuratedCampusNotice(
        "qiyunfang", "启云方", "2027 校招 AI Agent/软件岗位合集（公开公告）",
        "https://myjob.dlmu.edu.cn/campus/view/id/868418",
        "AI 软件、AI 模型、AI Agent、RAG、Tool Calling 和软件开发", "大连海事大学就业信息网", "启云方",
        location="武汉、深圳",
    ),
    CuratedCampusNotice(
        "spirit_ai", "千寻智能", "2027 校招 Agent/AI 平台岗位合集（公开公告）",
        "https://job.hust.edu.cn/zpinfo1/2407316.htm",
        "Agent 算法、具身智能 AI 平台、AI Infra、模型评测和机器人软件", "华中科技大学就业信息网", "千寻智能",
        location="北京、杭州、深圳", apply_url="https://nwd4iy9rd2s.jobs.feishu.cn/campusofSpiritAI",
        posted_at="2026-08-17",
    ),
    CuratedCampusNotice(
        "hygon", "海光信息", "2027 校招大模型/软件岗位合集（公开公告）",
        "https://job.hust.edu.cn/zpinfo1/2407309.htm",
        "CPU/GPU 软件开发、大模型与算法软件优化", "华中科技大学就业信息网", "海光",
        location="北京、上海、成都、苏州、天津、深圳等", posted_at="2026-08-17",
    ),
    CuratedCampusNotice(
        "dongfeng_rd", "东风研发总院", "2027 校招智能体/大模型岗位合集（公开公告）",
        "https://career.nankai.edu.cn/correcruit/content/id/116880.html",
        "大语言模型、多模态、VLA、世界模型、智能体开发与应用", "南开大学就业信息网", "东风汽车集团",
        posted_at="2026-08-22",
    ),
    CuratedCampusNotice(
        "zoomlion", "中联重科", "2027 校招机器人/AI 软件岗位合集（公开公告）",
        "https://career.nankai.edu.cn/correcruit/content/id/116863.html",
        "具身智能、AI 算法、感知导航、嵌入式与系统软件", "南开大学就业信息网", "中联重科",
        location="长沙", posted_at="2026-08-21",
    ),
    CuratedCampusNotice(
        "bytedance_seed", "字节跳动", "2027 Seed 大模型校招岗位合集（企业官网）",
        "https://seed.bytedance.com/zh/seedearlycareer",
        "基础模型、大模型算法、研究和工程方向", "字节跳动 Seed 官网", "2027",
        apply_url="https://seed.bytedance.com/zh/seedearlycareer",
    ),
    CuratedCampusNotice(
        "pdd", "拼多多", "2027 校招算法/研发岗位合集（公开公告）",
        "https://jy.bsu.edu.cn/front/zwxx.jspa?xqzwId=2078018788040658946&zpxxId=2078020991216275457",
        "算法、AI Agent、服务端研发和客户端研发方向", "北京体育大学就业信息网", "拼多多",
        apply_url="https://careers.pddglobalhr.com/campus/grad",
    ),
    CuratedCampusNotice(
        "xiaopeng", "小鹏汽车", "2027 校招 AI/软件技术岗位合集（公开公告）",
        "https://career.nankai.edu.cn/correcruit/content/id/116251.html",
        "AI、自动驾驶、软件研发和智能汽车技术方向", "南开大学就业信息网", "小鹏",
        posted_at="2026-07-06",
    ),
    CuratedCampusNotice(
        "huanuo", "华诺星空", "2027 校招 AI/后端岗位合集（公开公告）",
        "https://myjob.dlmu.edu.cn/campus/view/id/868428",
        "AI 算法、Java 后端、RAG 和 Agent 应用方向", "大连海事大学就业信息网", "华诺星空",
    ),
    CuratedCampusNotice(
        "y-t", "扬腾创新", "2027 校招 AI 应用岗位合集（公开公告）",
        "https://job.hust.edu.cn/zpinfo1/2407468.htm",
        "AI 应用、软件工程及数智化技术方向", "华中科技大学就业信息网", "扬腾创新",
        posted_at="2026-08-25",
    ),
    CuratedCampusNotice(
        "qunar", "去哪儿旅行", "2027 校招 AI 应用/全栈岗位合集（公开公告）",
        "https://myjob.dlmu.edu.cn/campus/view/id/868380",
        "AI 全栈、AI 应用开发（Java/测开/客户端）和 AI 算法", "大连海事大学就业信息网", "去哪儿",
        location="北京、上海", apply_url="https://campus.qunar.com",
    ),
    CuratedCampusNotice(
        "catl", "宁德时代", "2027 校招计算机/AI 岗位合集（公开公告）",
        "https://job.hust.edu.cn/zpinfo1/2407371.htm",
        "计算机、AI、数据科学、软件工程和 AI for Science", "华中科技大学就业信息网", "宁德时代",
        posted_at="2026-08-19",
    ),
    CuratedCampusNotice(
        "huaqin", "华勤技术", "2027 校招 AI/软件技术岗位合集（公开公告）",
        "https://job.hust.edu.cn/zpinfo1/2407478.htm",
        "人工智能、云计算、AIoT、软件和智能产品技术方向", "华中科技大学就业信息网", "华勤",
        apply_url="https://jobs.huaqin.com", posted_at="2026-08-25",
    ),
)


class CuratedCampusNoticeSource(DomesticJobSource):
    """Current public 2027 notices used when no stable per-job API is available."""

    metadata = SourceMetadata(
        source_id="curated_public_2027_ai_software_notices_v1",
        name="2027 届 AI/软件校招公开公告",
        source_type="public_notice",
        base_url="https://career.nankai.edu.cn/",
        terms_url="https://career.nankai.edu.cn/",
        schedule_minutes=720,
    )

    def __init__(
        self,
        session: requests.Session | None = None,
        notices: tuple[CuratedCampusNotice, ...] = CURATED_2027_TECH_NOTICES,
    ):
        super().__init__(session=session)
        self.notices = notices

    @staticmethod
    def _extract_visible_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.select("script, style, noscript, svg"):
            tag.decompose()
        return compact_text(soup.get_text("\n", strip=True))[:24000]

    def fetch(self) -> list[NormalizedJob]:
        jobs: list[NormalizedJob] = []
        failures: list[str] = []
        for notice in self.notices:
            try:
                text = self._extract_visible_text(self._get_text(notice.source_url))
            except Exception as exc:
                failures.append(f"{notice.company_name}: {type(exc).__name__}: {exc}")
                continue
            if "2027" not in text or notice.company_token not in text:
                failures.append(f"{notice.company_name}: page validation failed")
                continue
            description = compact_text(
                "来源说明：该记录是企业招聘官网或高校就业网公开的 2027 届校招公告级岗位合集，"
                "不是招聘平台内部 API 的逐岗位数据；岗位状态和要求以投递页为准。\n"
                f"重点技术方向：{notice.focus}\n公开公告正文：\n{text}"
            )
            jobs.append(
                NormalizedJob(
                    company_name=notice.company_name,
                    job_title=notice.title,
                    description=description,
                    location=notice.location,
                    source_kind="open_source",
                    source_dataset=self.metadata.source_id,
                    source_file=f"notice_{notice.slug}.html",
                    source_url=notice.source_url,
                    language="zh",
                    external_id=notice.slug,
                    apply_url=notice.apply_url or notice.source_url,
                    source_name=f"{self.metadata.name} · {notice.publisher}",
                    recruitment_type="campus",
                    graduation_year=2027,
                    degree_requirement=infer_degree_requirement(text),
                    major_requirement=infer_major_requirement(text),
                    job_category=classify_job(notice.title, description),
                    employment_type="full_time",
                    posted_at=notice.posted_at,
                    status="open",
                    is_domestic=True,
                    raw_payload_json=json.dumps(
                        {
                            "company_name": notice.company_name,
                            "focus": notice.focus,
                            "publisher": notice.publisher,
                            "source_url": notice.source_url,
                            "apply_url": notice.apply_url or notice.source_url,
                            "detail_level": "public_notice_bundle",
                            "visible_text_sha256": hashlib.sha256(
                                text.encode("utf-8")
                            ).hexdigest(),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )
            )
        if failures:
            raise DomesticSourceError(
                "curated notice sync was incomplete; existing records were preserved: "
                + "; ".join(failures)
            )
        if len(jobs) != len(self.notices):
            raise DomesticSourceError("curated notice count did not match configuration")
        return jobs


def default_domestic_sources() -> dict[str, DomesticJobSource]:
    values: list[DomesticJobSource] = [
        BaiduCampusSource(),
        DjiCampusHotJobsSource(),
        KylinCampusSource(),
        OppoCampusSource(),
        MeituanCampusAiSource(),
        HonorCampusHighlightsSource(),
        CuratedCampusNoticeSource(),
    ]
    return {source.metadata.source_id: source for source in values}
