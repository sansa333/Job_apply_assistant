from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "data" / "eval_dataset"
RESUME_DIR = DATASET / "resumes"
JD_DIR = DATASET / "jds"
PAIR_DIR = DATASET / "pairs"
RAG_DIR = DATASET / "rag_queries"
MM_DIR = DATASET / "multimodal_zh"
MM_IMAGE_DIR = MM_DIR / "images"
MM_TEXT_DIR = MM_DIR / "text"


RESUMES = [
    {
        "id": "001",
        "name": "张伟",
        "target": "大模型应用开发工程师",
        "years": "3年",
        "skills": ["Python", "FastAPI", "LangChain", "RAG", "Chroma", "Redis", "Docker"],
        "projects": [
            "企业知识库 RAG：完成文档加载、语义切块、向量召回、Rerank 与答案引用，HitRate@5 从 0.62 提升到 0.81。",
            "多模态招聘助手：使用 GLM-V 对简历截图和岗位海报做 OCR/语义抽取，写入 Chroma 实现图文统一检索。",
        ],
        "strength": "RAG 工程化、FastAPI 服务封装、检索评估闭环",
    },
    {
        "id": "002",
        "name": "李晨",
        "target": "后端开发工程师",
        "years": "4年",
        "skills": ["Java", "Spring Boot", "MySQL", "Redis", "Kafka", "Docker", "Kubernetes"],
        "projects": [
            "订单中台：负责高并发接口、缓存预热和消息队列削峰，核心接口 P95 降到 120ms。",
            "权限系统：基于 RBAC 设计多租户权限模型，支持审计日志和灰度发布。",
        ],
        "strength": "后端架构、数据库优化、微服务稳定性",
    },
    {
        "id": "003",
        "name": "王雨",
        "target": "数据分析师",
        "years": "2年",
        "skills": ["SQL", "Python", "Pandas", "Tableau", "A/B Test", "指标体系"],
        "projects": [
            "增长分析看板：搭建用户留存、转化漏斗和渠道 ROI 指标体系，支持周报自动化。",
            "A/B 实验平台：设计实验分层、显著性检验和异常样本过滤规则。",
        ],
        "strength": "业务指标拆解、SQL 分析、可视化报表",
    },
    {
        "id": "004",
        "name": "赵敏",
        "target": "前端工程师",
        "years": "3年",
        "skills": ["TypeScript", "React", "Next.js", "Vite", "Tailwind CSS", "ECharts"],
        "projects": [
            "SaaS 控制台：实现复杂表格、权限路由、实时筛选和可视化仪表盘。",
            "低代码表单：封装 Schema 渲染器和组件物料库，提升运营配置效率。",
        ],
        "strength": "前端工程化、组件库、数据可视化",
    },
    {
        "id": "005",
        "name": "陈航",
        "target": "机器学习工程师",
        "years": "4年",
        "skills": ["PyTorch", "TensorFlow", "特征工程", "推荐系统", "MLflow", "模型监控"],
        "projects": [
            "推荐召回模型：基于双塔模型和负采样优化召回率，线上 CTR 提升 6.3%。",
            "模型监控平台：跟踪特征漂移、AUC、延迟和回滚状态。",
        ],
        "strength": "模型训练、推荐算法、实验追踪",
    },
    {
        "id": "006",
        "name": "刘洋",
        "target": "DevOps 工程师",
        "years": "5年",
        "skills": ["Linux", "Docker", "Kubernetes", "Helm", "GitLab CI", "Prometheus", "Grafana"],
        "projects": [
            "K8s 发布平台：实现蓝绿发布、回滚、镜像扫描和资源配额管理。",
            "监控告警体系：建设服务可用性、错误率、延迟和容量看板。",
        ],
        "strength": "CI/CD、容器化、可观测性",
    },
    {
        "id": "007",
        "name": "周宁",
        "target": "产品经理",
        "years": "4年",
        "skills": ["用户访谈", "需求分析", "PRD", "原型设计", "数据分析", "增长策略"],
        "projects": [
            "招聘 SaaS 产品：设计岗位发布、简历筛选、候选人跟进和面试评价流程。",
            "AI 简历助手：定义匹配评分、推荐理由和反馈闭环。",
        ],
        "strength": "招聘业务流程、AI 产品设计、需求优先级",
    },
    {
        "id": "008",
        "name": "孙悦",
        "target": "测试开发工程师",
        "years": "3年",
        "skills": ["Python", "Pytest", "接口自动化", "Playwright", "性能测试", "CI"],
        "projects": [
            "自动化测试平台：支持接口用例、UI 用例、定时任务和报告归档。",
            "压测治理：定位接口慢查询和线程池瓶颈，推动 P95 延迟下降 30%。",
        ],
        "strength": "自动化测试、质量平台、性能诊断",
    },
]


JDS = [
    {
        "id": "001",
        "company": "星河智能科技",
        "title": "大模型应用开发工程师",
        "keywords": ["Python", "FastAPI", "LangChain", "RAG", "Chroma", "Rerank", "多模态"],
        "responsibilities": [
            "负责企业知识库问答、简历/JD 匹配和多模态检索应用开发。",
            "建设 RAG 评估闭环，跟踪 HitRate@K、MRR@K、关键词召回率和 token 消耗。",
            "将图片 OCR/语义抽取结果与文本统一写入 Chroma collection。",
        ],
        "requirements": ["熟悉 Python/FastAPI", "有 RAG、向量数据库、Rerank 或 LangChain 项目经验", "能做日志埋点和效果评估"],
    },
    {
        "id": "002",
        "company": "云杉云计算",
        "title": "后端开发工程师",
        "keywords": ["Java", "Spring Boot", "MySQL", "Redis", "Kafka", "Kubernetes"],
        "responsibilities": ["负责交易与订单系统后端服务", "优化数据库查询、缓存和消息队列", "参与微服务稳定性治理"],
        "requirements": ["熟悉 Java/Spring Boot", "掌握 MySQL 与 Redis", "了解容器化和 CI/CD"],
    },
    {
        "id": "003",
        "company": "数启增长实验室",
        "title": "数据分析师",
        "keywords": ["SQL", "Python", "Pandas", "Tableau", "A/B Test", "指标体系"],
        "responsibilities": ["搭建业务指标体系和增长看板", "负责用户行为分析和 A/B 实验评估", "输出可落地的数据洞察"],
        "requirements": ["熟练 SQL 和 Python", "理解统计检验和实验设计", "能与业务团队沟通指标口径"],
    },
    {
        "id": "004",
        "company": "北辰协同办公",
        "title": "前端工程师",
        "keywords": ["TypeScript", "React", "Next.js", "Vite", "Tailwind CSS", "ECharts"],
        "responsibilities": ["开发 SaaS 控制台和数据可视化页面", "沉淀组件库和工程化规范", "优化前端性能和交互体验"],
        "requirements": ["熟悉 React/TypeScript", "有复杂表格或图表经验", "关注可维护性和用户体验"],
    },
    {
        "id": "005",
        "company": "青枫推荐系统",
        "title": "机器学习工程师",
        "keywords": ["PyTorch", "推荐系统", "特征工程", "MLflow", "模型监控"],
        "responsibilities": ["训练和优化推荐召回/排序模型", "建设实验追踪和模型监控", "与后端协作完成在线推理部署"],
        "requirements": ["熟悉 PyTorch", "理解推荐算法和特征工程", "有模型上线和监控经验"],
    },
    {
        "id": "006",
        "company": "启航基础设施",
        "title": "DevOps 工程师",
        "keywords": ["Linux", "Docker", "Kubernetes", "Helm", "GitLab CI", "Prometheus", "Grafana"],
        "responsibilities": ["维护 Kubernetes 集群和发布平台", "建设 CI/CD、监控告警和容量治理", "推动服务稳定性改进"],
        "requirements": ["熟悉 Linux 和容器化", "掌握 Kubernetes/Helm", "能建设 Prometheus/Grafana 监控"],
    },
]


RETRIEVAL_QUERIES = [
    ("001", "001", "high"),
    ("001", "002", "low"),
    ("002", "002", "high"),
    ("002", "006", "medium"),
    ("003", "003", "high"),
    ("003", "005", "low"),
    ("004", "004", "high"),
    ("004", "003", "low"),
    ("005", "005", "high"),
    ("005", "001", "medium"),
    ("006", "006", "high"),
    ("006", "002", "medium"),
    ("007", "001", "medium"),
    ("007", "003", "low"),
    ("008", "001", "medium"),
    ("008", "006", "medium"),
    ("001", "004", "low"),
    ("002", "001", "low"),
    ("005", "006", "low"),
    ("008", "002", "medium"),
    ("004", "001", "medium"),
    ("003", "001", "low"),
    ("006", "001", "low"),
    ("007", "004", "low"),
]


MM_SAMPLES = [
    {
        "id": "0001",
        "title": "候选人简历截图",
        "image_title": "张伟 - 大模型应用开发工程师简历摘要",
        "lines": ["核心技能: Python / FastAPI / LangChain / RAG / Chroma", "项目: 多模态招聘助手, Rerank 评估闭环", "成果: HitRate@5 0.62 -> 0.81"],
        "question": "这张简历截图中的候选人最匹配哪个岗位？",
        "answer": "大模型应用开发工程师",
        "keywords": ["张伟", "大模型应用开发工程师", "RAG", "FastAPI", "Chroma"],
    },
    {
        "id": "0002",
        "title": "岗位JD海报",
        "image_title": "星河智能科技招聘: 大模型应用开发工程师",
        "lines": ["要求: Python, FastAPI, LangChain, RAG, Rerank", "职责: 建设 RAG 评估闭环与图文统一检索", "指标: HitRate@K / MRR@K / KeywordRecall@K"],
        "question": "这张岗位海报要求候选人熟悉哪些 RAG 相关技术？",
        "answer": "Python、FastAPI、LangChain、RAG、Rerank、Chroma、评估指标",
        "keywords": ["星河智能科技", "RAG", "Rerank", "HitRate@K", "MRR@K"],
    },
    {
        "id": "0003",
        "title": "候选人岗位匹配矩阵",
        "image_title": "候选人-岗位匹配矩阵",
        "lines": ["张伟 x 大模型应用开发: 92分", "李晨 x 后端开发: 88分", "王雨 x 数据分析: 90分", "赵敏 x 前端工程师: 91分"],
        "question": "匹配矩阵中张伟对应的大模型应用开发岗位分数是多少？",
        "answer": "92分",
        "keywords": ["匹配矩阵", "张伟", "大模型应用开发", "92分"],
    },
    {
        "id": "0004",
        "title": "RAG项目架构图",
        "image_title": "求职助手 RAG 架构",
        "lines": ["文本简历/JD -> 语义切块 -> Chroma", "图片简历/JD -> GLM-V OCR/语义抽取 -> Chroma", "检索 -> Rerank -> Prompt -> GLM 生成"],
        "question": "架构图中图片资料如何进入统一检索链路？",
        "answer": "图片先经过 GLM-V OCR/语义抽取，再写入 Chroma，与文本统一检索。",
        "keywords": ["GLM-V", "OCR", "语义抽取", "Chroma", "统一检索"],
    },
    {
        "id": "0005",
        "title": "面试反馈卡",
        "image_title": "面试反馈: 大模型应用开发工程师",
        "lines": ["候选人: 张伟", "优势: RAG评估闭环、FastAPI、日志埋点", "风险: Reranker 本地模型缓存需补齐", "建议: 进入二面"],
        "question": "面试反馈卡中候选人的主要优势是什么？",
        "answer": "RAG评估闭环、FastAPI、日志埋点",
        "keywords": ["面试反馈", "RAG评估闭环", "FastAPI", "日志埋点"],
    },
    {
        "id": "0006",
        "title": "技能雷达摘要",
        "image_title": "技能雷达: 大模型应用开发",
        "lines": ["RAG: 90", "FastAPI: 85", "Chroma: 82", "Prompt工程: 80", "多模态: 78"],
        "question": "技能雷达摘要中 RAG 分数是多少？",
        "answer": "90",
        "keywords": ["技能雷达", "RAG", "90", "FastAPI", "多模态"],
    },
    {
        "id": "0007",
        "title": "JD关键词看板",
        "image_title": "JD关键词看板",
        "lines": ["岗位: 后端开发工程师", "高频词: Java / Spring Boot / MySQL / Redis / Kafka", "加分项: Kubernetes, 可观测性"],
        "question": "JD关键词看板中后端岗位的高频词有哪些？",
        "answer": "Java、Spring Boot、MySQL、Redis、Kafka",
        "keywords": ["后端开发工程师", "Java", "Spring Boot", "MySQL", "Redis", "Kafka"],
    },
    {
        "id": "0008",
        "title": "投递材料清单",
        "image_title": "投递材料生成结果",
        "lines": ["输出文件: fit_report.md", "输出文件: cover_letter.md", "输出文件: interview_questions.md", "编码: utf-8-sig", "状态: generated_not_submitted"],
        "question": "投递材料清单中生成了哪些文件？",
        "answer": "fit_report.md、cover_letter.md、interview_questions.md",
        "keywords": ["投递材料", "fit_report.md", "cover_letter.md", "interview_questions.md", "utf-8-sig"],
    },
]


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def find_font(bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/mnt/c/Windows/Fonts/msyhbd.ttc" if bold else "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/simhei.ttf",
        "/mnt/c/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), 28 if bold else 24)
    return ImageFont.load_default()


FONT = find_font(False)
FONT_BOLD = find_font(True)
FONT_SMALL = ImageFont.truetype(FONT.path, 20) if hasattr(FONT, "path") else FONT


def wrap_zh(text: str, width: int = 28) -> list[str]:
    if len(text) <= width:
        return [text]
    return textwrap.wrap(text, width=width, break_long_words=True, replace_whitespace=False)


def draw_dataset_image(sample: dict) -> None:
    image = Image.new("RGB", (1200, 760), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((36, 36, 1164, 724), radius=24, fill="#ffffff", outline="#cbd5e1", width=2)
    draw.rectangle((36, 36, 1164, 130), fill="#0f766e")
    draw.text((72, 66), sample["image_title"], font=FONT_BOLD, fill="#ffffff")

    y = 168
    colors = ["#ecfeff", "#fef9c3", "#eef2ff", "#f0fdf4", "#fff7ed"]
    for idx, line in enumerate(sample["lines"]):
        fill = colors[idx % len(colors)]
        draw.rounded_rectangle((72, y, 1128, y + 78), radius=12, fill=fill, outline="#cbd5e1")
        for offset, wrapped in enumerate(wrap_zh(line, width=42)):
            draw.text((100, y + 22 + offset * 24), wrapped, font=FONT, fill="#111827")
        y += 100

    draw.line((72, 620, 1128, 620), fill="#94a3b8", width=2)
    draw.text((72, 650), "场景标签: 中文求职 / 简历JD匹配 / 多模态RAG评估", font=FONT_SMALL, fill="#334155")
    image_path = MM_IMAGE_DIR / f"zh_mrag_{sample['id']}_query.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)


def render_resume(item: dict) -> str:
    projects = "\n".join(f"- {project}" for project in item["projects"])
    return f"""
# 中文简历：{item["name"]}

## 基本信息
- 姓名：{item["name"]}
- 目标岗位：{item["target"]}
- 工作年限：{item["years"]}

## 技能栈
{", ".join(item["skills"])}

## 项目经历
{projects}

## 优势总结
{item["strength"]}
"""


def render_jd(item: dict) -> str:
    responsibilities = "\n".join(f"- {line}" for line in item["responsibilities"])
    requirements = "\n".join(f"- {line}" for line in item["requirements"])
    return f"""
# 中文JD：{item["company"]} - {item["title"]}

## 岗位关键词
{", ".join(item["keywords"])}

## 岗位职责
{responsibilities}

## 任职要求
{requirements}
"""


def main() -> None:
    for folder in [RESUME_DIR, JD_DIR, PAIR_DIR, RAG_DIR, MM_IMAGE_DIR, MM_TEXT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    resume_by_id = {item["id"]: item for item in RESUMES}
    jd_by_id = {item["id"]: item for item in JDS}

    for resume in RESUMES:
        write_text(RESUME_DIR / f"zh_resume_{resume['id']}.md", render_resume(resume))

    for jd in JDS:
        write_text(JD_DIR / f"zh_jd_{jd['id']}.md", render_jd(jd))

    pair_rows = []
    query_rows = []
    for idx, (resume_id, jd_id, label) in enumerate(RETRIEVAL_QUERIES, start=1):
        resume = resume_by_id[resume_id]
        jd = jd_by_id[jd_id]
        expected_keywords = sorted(set(resume["skills"]) & set(jd["keywords"]))
        expected_keywords = [resume["name"], jd["company"], jd["title"], *expected_keywords]
        if len(expected_keywords) < 5:
            expected_keywords.extend(jd["keywords"][: 5 - len(expected_keywords)])

        pair_rows.append(
            {
                "pair_id": f"zh_pair_{idx:04d}",
                "source": "synthetic_zh_job_apply_local",
                "resume_id": f"zh_resume_{resume_id}",
                "resume_file": f"zh_resume_{resume_id}.md",
                "jd_id": f"zh_jd_{jd_id}",
                "jd_file": f"zh_jd_{jd_id}.md",
                "relevance_label": label,
                "expected_keywords": expected_keywords,
            }
        )
        query_rows.append(
            {
                "query_id": f"zh_rq_{idx:04d}",
                "source_pair_id": f"zh_pair_{idx:04d}",
                "scenario": "resume_job_matching_zh",
                "query": f"候选人{resume['name']}是否适合{jd['company']}的{jd['title']}岗位？请结合中文简历和JD给出证据。",
                "expected_sources": [f"zh_resume_{resume_id}.md", f"zh_jd_{jd_id}.md"],
                "expected_keywords": expected_keywords,
                "relevance_label": label,
            }
        )

    write_jsonl(PAIR_DIR / "zh_resume_jd_pairs.jsonl", pair_rows)
    write_jsonl(RAG_DIR / "zh_retrieval_eval.jsonl", query_rows)

    mm_rows = []
    for sample in MM_SAMPLES:
        draw_dataset_image(sample)
        text_path = MM_TEXT_DIR / f"zh_mrag_{sample['id']}_qa.md"
        write_text(
            text_path,
            f"""
# {sample["title"]}

## 图片主题
{sample["image_title"]}

## OCR文本
{"; ".join(sample["lines"])}

## 问题
{sample["question"]}

## 标准答案
{sample["answer"]}

## 检索关键词
{", ".join(sample["keywords"])}
""",
        )
        mm_rows.append(
            {
                "sample_id": f"zh_mrag_{sample['id']}",
                "source": "synthetic_zh_job_apply_local",
                "scenario": "job_apply_multimodal_zh",
                "aspect": "resume_jd_ocr_semantic_extraction",
                "image_type": "JobApplicationDocument",
                "image_file": f"multimodal_zh/images/zh_mrag_{sample['id']}_query.png",
                "text_file": f"multimodal_zh/text/zh_mrag_{sample['id']}_qa.md",
                "question": sample["question"],
                "choices": {},
                "answer_choice": None,
                "answer": sample["answer"],
                "expected_sources": [f"zh_mrag_{sample['id']}_query.png", f"zh_mrag_{sample['id']}_qa.md"],
                "expected_keywords": sample["keywords"],
            }
        )

    write_jsonl(MM_DIR / "zh_mrag_eval.jsonl", mm_rows)
    print(f"generated {len(RESUMES)} resumes, {len(JDS)} jds, {len(query_rows)} retrieval queries, {len(mm_rows)} multimodal samples")


if __name__ == "__main__":
    main()
