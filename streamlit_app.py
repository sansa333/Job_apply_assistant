from __future__ import annotations

import json
import os
import re
from datetime import date
from typing import Any
from urllib.parse import quote

import requests
import streamlit as st

st.set_page_config(page_title="AI Job Apply Assistant", page_icon="🧠", layout="wide")

DEFAULT_BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


def post_files(url: str, files: list[Any]) -> dict:
    payload = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files]
    response = requests.post(url, files=payload, timeout=120)
    if not response.ok:
        detail = None
        try:
            body = response.json()
            detail = body.get("detail") if isinstance(body, dict) else None
        except Exception:
            detail = response.text
        raise RuntimeError(f"HTTP {response.status_code}: {detail or response.reason}")
    return response.json()


def render_json(data: Any) -> None:
    st.code(json.dumps(data, ensure_ascii=False, indent=2), language="json")


def request_json(method: str, url: str, **kwargs: Any) -> dict:
    response = requests.request(method, url, timeout=kwargs.pop("timeout", 120), **kwargs)
    if not response.ok:
        detail = None
        try:
            body = response.json()
            detail = body.get("detail") if isinstance(body, dict) else None
        except Exception:
            detail = response.text
        raise RuntimeError(f"HTTP {response.status_code}: {detail or response.reason}")
    return response.json()


def preference_list(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,，、;；\n]", value) if item.strip()]


st.title("国内 AI 求职助手")
st.caption("国内官网在招岗位 + PDF 简历只读分析 + 匹配与投递看板")

with st.sidebar:
    st.header("连接配置")
    backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND)
    health_check = st.button("健康检查")

    if health_check:
        try:
            res = requests.get(f"{backend_url}/health", timeout=30)
            res.raise_for_status()
            st.success("连接成功")
            render_json(res.json())
        except Exception as exc:
            st.error(f"连接失败: {exc}")

    st.divider()
    st.header("当前候选人与求职偏好")
    active_candidate_id = st.text_input(
        "candidate_id",
        value="current_candidate",
        key="active_candidate_id",
        help="所有简历、匹配分析和投递记录都使用此标识；可使用中英文、数字、点、横线和下划线。",
    ).strip()
    if not active_candidate_id:
        st.warning("candidate_id 不能为空，当前暂用 current_candidate。")
        active_candidate_id = "current_candidate"
    target_roles_text = st.text_area(
        "目标岗位（逗号或换行分隔）",
        value="AI应用开发工程师，Agent开发工程师，大模型应用开发工程师，AI软件开发工程师",
        key="target_roles_text",
        height=100,
    )
    target_cities_text = st.text_input(
        "目标城市（逗号分隔，可留空）", key="target_cities_text"
    )
    graduation_year_value = st.number_input(
        "毕业年份（0 表示不限）",
        min_value=0,
        max_value=2100,
        value=date.today().year + 1,
        step=1,
        key="graduation_year_value",
    )
    target_roles = preference_list(target_roles_text)
    target_cities = preference_list(target_cities_text)
    graduation_year = int(graduation_year_value) or None
    if st.button("保存求职偏好", use_container_width=True):
        try:
            encoded_candidate_id = quote(active_candidate_id, safe="")
            data = request_json(
                "PUT",
                f"{backend_url}/api/domestic/profile/{encoded_candidate_id}/preferences",
                json={
                    "graduation_year": graduation_year,
                    "target_roles": target_roles,
                    "target_cities": target_cities,
                },
            )
            st.success("求职偏好已保存到当前候选人画像")
            render_json(data)
        except Exception as exc:
            st.info(f"请先导入当前候选人的 PDF 简历，再保存偏好：{exc}")

previous_candidate_id = st.session_state.get("_previous_candidate_id")
if previous_candidate_id != active_candidate_id:
    for state_key in (
        "domestic_matches",
        "domestic_job_detail",
        "domestic_fit",
        "agent_result",
        "agent_conversation_id",
        "agent_conversation_selector",
        "mm_conversation_id",
        "chat_history",
    ):
        st.session_state.pop(state_key, None)
    st.session_state._previous_candidate_id = active_candidate_id

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

domestic_tab, applications_tab, job_tab, fit_tab, agent_tab, ingest_tab, chat_tab, eval_tab = st.tabs(
    [
        "国内岗位发现",
        "投递看板",
        "岗位库管理",
        "匹配工作台",
        "自然语言 Agent",
        "多模态知识入库",
        "多模态对话",
        "隔离评测报告",
    ]
)

with domestic_tab:
    st.subheader("国内官网岗位发现")
    st.caption("只自动采集允许公开访问的企业招聘官网；不会绕过登录、验证码或调用招聘平台内部接口。")
    try:
        domestic_stats = request_json("GET", f"{backend_url}/api/domestic/stats", timeout=30)
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("国内在招岗位", domestic_stats.get("open_jobs", 0))
        metric_b.metric("覆盖公司", domestic_stats.get("company_count", 0))
        metric_c.metric(
            "校招 / 实习",
            f"{domestic_stats.get('recruitment_type_distribution', {}).get('campus', 0)} / "
            f"{domestic_stats.get('recruitment_type_distribution', {}).get('internship', 0)}",
        )
        with st.expander("公司岗位分布", expanded=False):
            st.bar_chart(domestic_stats.get("company_distribution", {}))
    except Exception:
        st.info("后端连接后将显示最新岗位统计。")
    control_a, control_b = st.columns(2)
    if control_a.button("一键更新岗位", type="primary", use_container_width=True):
        try:
            data = request_json(
                "POST", f"{backend_url}/api/domestic/sources/refresh", timeout=1800,
            )
            message = (
                f"更新完成：在招 {data.get('open_domestic_jobs', 0)} 个，"
                f"已索引 {data.get('jobs_indexed', 0)} 个岗位 / "
                f"{data.get('chunks_indexed', 0)} 个分块"
            )
            if data.get("status") == "success":
                st.success(message)
            else:
                failed = "、".join(data.get("failed_sources", [])) or "未知来源"
                st.warning(f"{message}；部分来源失败：{failed}")
            render_json(data)
        except Exception as exc:
            st.error(f"岗位更新失败: {exc}")
    if control_b.button("查看数据源", use_container_width=True):
        try:
            render_json(request_json("GET", f"{backend_url}/api/domestic/sources"))
        except Exception as exc:
            st.error(f"读取失败: {exc}")

    with st.expander("只读导入 PDF 简历", expanded=False):
        st.warning("系统只复制并解析 PDF 原文，不会改写或覆盖简历内容。")
        st.caption(f"当前 candidate_id：{active_candidate_id}")
        resume_pdf = st.file_uploader("PDF 简历", type=["pdf"], key="domestic_resume_pdf")
        if st.button("导入 PDF", use_container_width=True):
            if resume_pdf is None:
                st.warning("请先选择 PDF")
            else:
                try:
                    files = {"file": (resume_pdf.name, resume_pdf.getvalue(), "application/pdf")}
                    data = request_json(
                        "POST", f"{backend_url}/api/domestic/profile/pdf",
                        params={
                            "candidate_id": active_candidate_id,
                            "graduation_year": graduation_year,
                            "target_roles": ",".join(target_roles),
                            "target_cities": ",".join(target_cities),
                        },
                        files=files,
                        timeout=600,
                    )
                    st.success("简历已按只读策略导入")
                    render_json(data)
                except Exception as exc:
                    st.error(f"PDF 导入失败: {exc}")

    st.markdown("#### 搜索与筛选")
    filter_a, filter_b, filter_c, filter_d = st.columns(4)
    domestic_keyword = filter_a.text_input(
        "岗位关键词",
        value=" ".join(target_roles),
        key="domestic_keyword",
        help="多个关键词用空格分隔，匹配任意一个。",
    )
    domestic_company = filter_b.text_input("公司", key="domestic_company")
    domestic_location = filter_c.text_input(
        "城市", value=target_cities[0] if target_cities else "", key="domestic_location"
    )
    domestic_type = filter_d.selectbox(
        "招聘类型", ["", "campus", "internship", "social"], key="domestic_type"
    )
    if st.button("搜索岗位", type="primary", use_container_width=True):
        try:
            data = request_json(
                "GET", f"{backend_url}/api/domestic/jobs/search",
                params={
                    "keyword": domestic_keyword,
                    "company_name": domestic_company,
                    "location": domestic_location,
                    "recruitment_type": domestic_type,
                    "graduation_year": graduation_year,
                    "candidate_id": active_candidate_id,
                    "limit": 100,
                },
            )
            st.session_state.domestic_matches = data.get("matches", [])
        except Exception as exc:
            st.error(f"搜索失败: {exc}")

    matches = st.session_state.get("domestic_matches", [])
    if matches:
        st.dataframe(
            [
                {
                    "匹配分": row.get("relevance_score"),
                    "公司": row.get("company_name"),
                    "岗位": row.get("job_title"),
                    "地点": row.get("location"),
                    "类型": row.get("recruitment_type"),
                    "类别": row.get("job_category"),
                    "投递链接": row.get("apply_url") or row.get("source_url"),
                }
                for row in matches
            ],
            use_container_width=True,
            hide_index=True,
            column_config={"投递链接": st.column_config.LinkColumn("投递链接")},
        )
        labels = [f"{row['company_name']}｜{row['job_title']}｜{row.get('location') or '地点未标注'}" for row in matches]
        selected_label = st.selectbox("选择岗位", labels)
        selected = matches[labels.index(selected_label)]
        action_a, action_b = st.columns(2)
        if action_a.button("查看完整 JD", use_container_width=True):
            try:
                st.session_state.domestic_job_detail = request_json(
                    "GET", f"{backend_url}/api/domestic/jobs/{selected['job_id']}"
                )
            except Exception as exc:
                st.error(f"读取 JD 失败: {exc}")
        if action_b.button("结合简历分析", type="primary", use_container_width=True):
            try:
                result = request_json(
                    "POST", f"{backend_url}/api/domestic/jobs/{selected['job_id']}/fit",
                    params={"candidate_id": active_candidate_id}, timeout=300,
                )
                st.session_state.domestic_fit = result
            except Exception as exc:
                st.error(f"分析失败: {exc}")
        stage_a, stage_b = st.columns([1, 2])
        selected_stage = stage_a.selectbox(
            "投递阶段",
            ["saved", "planned", "applied", "written_test", "interview", "offer", "rejected", "withdrawn", "ignored"],
            key=f"stage_{selected['job_id']}",
        )
        stage_notes = stage_b.text_input("备注", key=f"notes_{selected['job_id']}")
        if st.button("保存到投递看板", use_container_width=True):
            try:
                request_json(
                    "PUT", f"{backend_url}/api/domestic/jobs/{selected['job_id']}/application",
                    json={
                        "candidate_id": active_candidate_id,
                        "stage": selected_stage,
                        "notes": stage_notes,
                    },
                )
                st.success("投递状态已保存")
            except Exception as exc:
                st.error(f"保存失败: {exc}")
        detail = st.session_state.get("domestic_job_detail")
        if detail and detail.get("job_id") == selected.get("job_id"):
            st.markdown("#### 岗位详情")
            st.write(detail.get("description", ""))
        fit_result = st.session_state.get("domestic_fit")
        if fit_result:
            st.markdown("#### 简历匹配分析")
            st.write(fit_result.get("fit_report") or fit_result.get("result") or fit_result)
    else:
        st.info("点击“搜索岗位”查看国内在招岗位。")

with applications_tab:
    st.subheader("投递状态看板")
    try:
        applications = request_json(
            "GET", f"{backend_url}/api/domestic/applications/{quote(active_candidate_id, safe='')}"
        ).get("applications", [])
        if applications:
            st.dataframe(applications, use_container_width=True, hide_index=True)
        else:
            st.info("暂无保存或投递记录。可通过 API 或岗位详情将岗位加入看板。")
    except Exception as exc:
        st.error(f"看板加载失败: {exc}")

with job_tab:
    st.subheader("岗位库管理")
    st.caption("仅导入真实公开历史 JD 或用户上传 JD；历史公开岗位不代表当前仍在招聘。")
    import_col, health_col = st.columns(2)
    if import_col.button("导入本地开源 JD 语料", use_container_width=True):
        try:
            data = request_json("POST", f"{backend_url}/api/jobs/import/open-source", timeout=600)
            st.success("开源语料导入完成")
            render_json(data)
        except Exception as exc:
            st.error(f"导入失败: {exc}")
    if health_col.button("检查索引健康", use_container_width=True):
        try:
            render_json(request_json("GET", f"{backend_url}/api/knowledge/health"))
        except Exception as exc:
            st.error(f"健康检查失败: {exc}")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("#### 上传目标 JD")
        with st.form("job_upload_form"):
            upload_company = st.text_input("公司名称")
            upload_title = st.text_input("岗位名称")
            upload_text = st.text_area("JD 正文（可粘贴）", height=180)
            upload_file = st.file_uploader("或上传 UTF-8 Markdown/TXT", type=["md", "txt"], key="target_jd")
            submit_job = st.form_submit_button("保存并建立岗位索引", use_container_width=True)
        if submit_job:
            try:
                form_data = {"company_name": upload_company, "job_title": upload_title, "jd_text": upload_text}
                files = None
                if upload_file is not None:
                    files = {"file": (upload_file.name, upload_file.getvalue(), upload_file.type or "text/plain")}
                response = requests.post(f"{backend_url}/api/jobs/upload", data=form_data, files=files, timeout=180)
                response.raise_for_status()
                st.success("目标 JD 已入库")
                render_json(response.json())
            except Exception as exc:
                st.error(f"JD 上传失败: {exc}")
    with right:
        st.markdown("#### 上传候选人资料")
        with st.form("profile_upload_form"):
            st.caption(f"当前 candidate_id：{active_candidate_id}")
            profile_text = st.text_area("简历/项目正文（可粘贴）", height=180)
            profile_file = st.file_uploader("或上传 UTF-8 Markdown/TXT", type=["md", "txt"], key="profile_file")
            submit_profile = st.form_submit_button("保存候选人资料", use_container_width=True)
        if submit_profile:
            try:
                form_data = {"candidate_id": active_candidate_id, "profile_text": profile_text}
                files = None
                if profile_file is not None:
                    files = {"file": (profile_file.name, profile_file.getvalue(), profile_file.type or "text/plain")}
                response = requests.post(f"{backend_url}/api/candidates/upload", data=form_data, files=files, timeout=180)
                response.raise_for_status()
                st.success("候选人资料已入库")
                render_json(response.json())
            except Exception as exc:
                st.error(f"候选人资料上传失败: {exc}")

with fit_tab:
    st.subheader("精确岗位匹配工作台")
    st.caption("必须先精确命中“公司 + 岗位”。未命中时不会用相似职位代替，而是引导上传对应 JD。")
    with st.form("fit_form"):
        st.caption(f"当前 candidate_id：{active_candidate_id}")
        company_name = st.text_input("目标公司")
        job_title = st.text_input("目标岗位")
        fit_question = st.text_area("补充问题", value="请分析我的匹配度、主要证据与能力缺口。", height=100)
        run_fit = st.form_submit_button("开始精确匹配", type="primary", use_container_width=True)
    if run_fit:
        try:
            result = request_json(
                "POST",
                f"{backend_url}/api/fit",
                json={"candidate_id": active_candidate_id, "company_name": company_name, "job_title": job_title, "question": fit_question},
                timeout=180,
            )
            if result.get("status") == "job_not_found":
                st.warning(result.get("message"))
                st.info("请在“岗位库管理”中上传该公司的对应 JD 后重试。")
            else:
                st.success(f"流程状态：{result.get('status')} / {result.get('stage')}")
                if result.get("next_action"):
                    st.info(f"下一步：{result.get('next_action')}")
                if result.get("missing_fields"):
                    st.warning(f"待补充字段：{', '.join(result['missing_fields'])}")
                st.markdown("#### 匹配分析")
                st.write(result.get("fit_report", ""))
                evidence_a, evidence_b = st.columns(2)
                evidence_a.markdown("#### 岗位要求证据")
                evidence_a.json(result.get("job_evidence", []))
                evidence_b.markdown("#### 候选人经历证据")
                evidence_b.json(result.get("candidate_evidence", []))
                st.caption(result.get("historical_notice", ""))
        except Exception as exc:
            st.error(f"匹配失败: {exc}")


with agent_tab:
    st.subheader("自然语言 Agent")
    st.caption("先识别意图和公司/岗位实体；高置信度固定任务走确定性工具，其余请求由 Agent 动态选择工具。")
    if st.button("新建 Agent 会话", help="清除当前会话绑定；下次执行时创建新会话。"):
        st.session_state.pop("agent_conversation_id", None)
        st.session_state.pop("agent_result", None)
        st.session_state.agent_conversation_selector = ""
    try:
        conversations = request_json(
            "GET",
            f"{backend_url}/api/conversations",
            params={"candidate_id": active_candidate_id, "limit": 20},
            timeout=3,
        )
    except Exception:
        conversations = []
    if conversations:
        conversation_ids = [item["conversation_id"] for item in conversations]
        current_id = st.session_state.get("agent_conversation_id")
        choices = [""] + conversation_ids
        selector = st.session_state.get("agent_conversation_selector")
        if selector not in choices:
            st.session_state.agent_conversation_selector = current_id if current_id in choices else ""

        def select_agent_conversation() -> None:
            selected = st.session_state.agent_conversation_selector
            if selected:
                st.session_state.agent_conversation_id = selected
            else:
                st.session_state.pop("agent_conversation_id", None)
                st.session_state.pop("agent_result", None)

        st.selectbox(
            "打开历史会话",
            choices,
            key="agent_conversation_selector",
            on_change=select_agent_conversation,
            format_func=lambda value: next(
                (
                    f"{item.get('company_name') or '通用'} / {item.get('job_title') or '求职会话'} · {value[-8:]}"
                    for item in conversations
                    if item["conversation_id"] == value
                ),
                "新会话（尚未保存）" if not value else value,
            ),
        )
    if conversation_id := st.session_state.get("agent_conversation_id"):
        st.caption(f"当前会话：{conversation_id}")
    with st.form("natural_language_agent_form"):
        agent_goal = st.text_area(
            "用自然语言描述你的目标",
            value="请分析我与目标公司大模型应用开发工程师岗位的匹配度，并指出主要能力缺口。",
            height=120,
        )
        st.caption(f"当前 candidate_id：{active_candidate_id}")
        agent_company, agent_job = st.columns(2)
        agent_company_name = agent_company.text_input(
            "公司名称（可选，未填写时尝试从自然语言提取）",
            key="agent_company_name",
        )
        agent_job_title = agent_job.text_input(
            "岗位名称（可选，未填写时尝试从自然语言提取）",
            key="agent_job_title",
        )
        with st.expander("补充材料与联系方式（可选）"):
            agent_jd_text = st.text_area("岗位 JD", key="agent_jd_text", height=140)
            agent_resume_text = st.text_area("候选人补充资料", key="agent_resume_text", height=140)
            contact_a, contact_b, contact_c = st.columns(3)
            agent_candidate_name = contact_a.text_input("姓名", key="agent_candidate_name")
            agent_candidate_email = contact_b.text_input("邮箱", key="agent_candidate_email")
            agent_candidate_phone = contact_c.text_input("电话", key="agent_candidate_phone")
        run_natural_language_agent = st.form_submit_button(
            "识别意图并执行",
            type="primary",
            use_container_width=True,
        )

    if run_natural_language_agent:
        if not agent_goal.strip():
            st.warning("请输入自然语言目标。")
        else:
            try:
                conversation_id = st.session_state.get("agent_conversation_id")
                if not conversation_id:
                    conversation = request_json(
                        "POST",
                        f"{backend_url}/api/conversations",
                        json={
                            "candidate_id": active_candidate_id,
                            "conversation_type": "job_application",
                            "company_name": agent_company_name.strip(),
                            "job_title": agent_job_title.strip(),
                        },
                    )
                    conversation_id = conversation["conversation_id"]
                    st.session_state.agent_conversation_id = conversation_id
                result = request_json(
                    "POST",
                    f"{backend_url}/api/agent",
                    json={
                        "goal": agent_goal.strip(),
                        "conversation_id": conversation_id,
                        "candidate_id": active_candidate_id,
                        "candidate_name": agent_candidate_name.strip() or None,
                        "candidate_email": agent_candidate_email.strip() or None,
                        "candidate_phone": agent_candidate_phone.strip() or None,
                        "company_name": agent_company_name.strip(),
                        "job_title": agent_job_title.strip(),
                        "jd_text": agent_jd_text.strip(),
                        "resume_text": agent_resume_text.strip(),
                    },
                    timeout=300,
                )
                st.session_state.agent_result = result
            except Exception as exc:
                st.error(f"Agent 执行失败: {exc}")

    if result := st.session_state.get("agent_result"):
        intent_a, intent_b, intent_c = st.columns(3)
        intent_a.metric("识别意图", result.get("recognized_intent") or "未确定")
        intent_b.metric("置信度", f"{float(result.get('intent_confidence') or 0):.2f}")
        intent_c.metric("识别来源", result.get("intent_source") or "unresolved")
        st.write(f"流程状态：{result.get('status')} / {result.get('stage')}")
        if result.get("conversation_id"):
            st.caption(f"会话 ID：{result['conversation_id']}")
        if usage := result.get("context_usage"):
            if usage.get("model_invoked"):
                st.caption(
                    "上下文预算："
                    f"约 {usage.get('estimated_input_tokens', 0)} / "
                    f"{usage.get('target_input_tokens', 0)} tokens；"
                    f"保留最近 {usage.get('included_recent_turns', 0)} 条消息"
                )
            if usage.get("budget_status") == "mandatory_overflow":
                st.warning("系统指令和当前请求已超过目标上下文预算，请缩短当前输入或提高预算配置。")
            if usage.get("truncated_fields"):
                st.info(f"为满足上下文预算已裁剪：{', '.join(usage['truncated_fields'])}")
        if result.get("conversation_summary"):
            with st.expander("查看滚动会话摘要"):
                render_json(result["conversation_summary"])
        if result.get("extracted_entities"):
            st.info(f"提取实体：{result['extracted_entities']}")
        if result.get("intent_missing_fields"):
            st.warning(f"意图路由待补充：{', '.join(result['intent_missing_fields'])}")
        if result.get("missing_fields"):
            st.warning(f"待补充字段：{', '.join(result['missing_fields'])}")
        if result.get("next_action"):
            st.info(f"下一步：{result['next_action']}")
        st.markdown("#### Agent 结果")
        st.write(result.get("result", ""))
        if result.get("artifacts"):
            st.markdown("#### 本地产物")
            st.json(result["artifacts"])
        with st.expander("查看完整结构化响应"):
            render_json(result)


with ingest_tab:
    st.subheader("文本知识入库")
    st.caption("文本会被语义切块后写入 multimodal_knowledge collection，供 RAG 检索使用。")
    text_files = st.file_uploader(
        "上传文本文档",
        type=["pdf", "docx", "doc", "txt", "md", "csv"],
        accept_multiple_files=True,
        key="text_uploader",
    )
    if st.button("执行文本入库", use_container_width=True):
        if not text_files:
            st.warning("请先选择文本文件")
        else:
            try:
                data = post_files(f"{backend_url}/api/mm/ingest/text", text_files)
                st.success("文本入库完成")
                render_json(data)
            except Exception as exc:
                st.error(f"文本入库失败: {exc}")

with chat_tab:
    st.subheader("RAG 对话")
    st.caption("检索阶段从文本知识库召回候选片段，再由 Reranker 做二阶段排序。")
    col_a, col_b = st.columns([3, 1])
    with col_a:
        question = st.text_area("问题", value="请总结知识库核心内容，并给出三条落地建议。", height=120)
    with col_b:
        top_k = st.slider("TopK", min_value=1, max_value=12, value=6)

    temp_image = st.file_uploader(
        "临时图片（仅本轮使用，可选）",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files=False,
        key="temp_chat_image",
    )

    if st.button("发送提问", type="primary", use_container_width=True):
        try:
            mm_conversation_id = st.session_state.get("mm_conversation_id")
            if not mm_conversation_id:
                conversation = request_json(
                    "POST",
                    f"{backend_url}/api/conversations",
                    json={
                        "candidate_id": active_candidate_id,
                        "conversation_type": "knowledge_chat",
                    },
                )
                mm_conversation_id = conversation["conversation_id"]
                st.session_state.mm_conversation_id = mm_conversation_id
            form_data = {
                "question": question,
                "top_k": str(top_k),
                "history_json": "[]",
                "conversation_id": mm_conversation_id,
                "candidate_id": active_candidate_id,
            }
            files = None
            if temp_image is not None:
                files = {
                    "image": (
                        temp_image.name,
                        temp_image.getvalue(),
                        temp_image.type or "application/octet-stream",
                    )
                }

            res = requests.post(
                f"{backend_url}/api/mm/chat",
                data=form_data,
                files=files,
                timeout=180,
            )
            res.raise_for_status()
            data = res.json()

            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": data.get("answer", "")})

            st.success("回答生成完成")
            st.caption(f"服务端会话：{data.get('conversation_id') or mm_conversation_id}")
            st.markdown("### 回答")
            st.write(data.get("answer", ""))

            st.markdown("### 召回信息")
            st.write(
                {
                    "retrieved_chunks": data.get("retrieved_chunks"),
                    "candidate_chunks": data.get("candidate_chunks"),
                    "reranker_applied": data.get("reranker_applied"),
                    "reranker_model": data.get("reranker_model"),
                    "reranker_reason": data.get("reranker_reason"),
                }
            )

            st.markdown("### 引用来源")
            render_json(data.get("citations", []))
            if data.get("context_usage"):
                with st.expander("上下文预算与滚动摘要"):
                    render_json(
                        {
                            "context_usage": data["context_usage"],
                            "conversation_summary": data.get("conversation_summary", {}),
                        }
                    )
        except Exception as exc:
            st.error(f"对话失败: {exc}")

    if st.button("清空对话历史", use_container_width=True):
        mm_conversation_id = st.session_state.pop("mm_conversation_id", None)
        if mm_conversation_id:
            try:
                request_json(
                    "DELETE",
                    f"{backend_url}/api/conversations/{mm_conversation_id}",
                    params={"candidate_id": active_candidate_id},
                )
            except Exception as exc:
                st.warning(f"服务端会话删除失败：{exc}")
        st.session_state.chat_history = []
        st.info("已清空")

with eval_tab:
    st.subheader("隔离评测报告")
    st.caption("评测语料仅写入 eval_demo collection，绝不会进入岗位匹配或候选人资料检索。")

    dataset_col, limit_col, image_col = st.columns([2, 1, 1])
    dataset_name = dataset_col.selectbox(
        "评估数据集",
        ["zh_retrieval", "zh_multimodal", "zh_all", "retrieval", "multimodal", "all"],
        index=0,
    )
    sample_limit = limit_col.number_input("样本数", min_value=1, max_value=80, value=10)
    include_images = image_col.checkbox("导入图片解析", value=True)

    ingest_col, load_col = st.columns(2)
    if ingest_col.button("导入评估语料到 Chroma", use_container_width=True):
        try:
            data = request_json(
                "POST",
                f"{backend_url}/api/mm/ingest/eval-dataset",
                json={
                    "dataset_name": dataset_name,
                    "sample_limit": int(sample_limit),
                    "include_images": include_images,
                },
                timeout=600,
            )
            st.success("评估语料导入完成")
            render_json(data)
        except Exception as exc:
            st.error(f"导入失败: {exc}")

    if load_col.button("加载评估样本", use_container_width=True):
        try:
            data = request_json(
                "GET",
                f"{backend_url}/api/mm/eval-dataset/samples",
                params={"dataset_name": dataset_name, "sample_limit": int(sample_limit)},
                timeout=120,
            )
            st.session_state.eval_samples = data.get("samples", [])
            st.success(f"已加载 {len(st.session_state.eval_samples)} 条样本")
        except Exception as exc:
            st.error(f"加载失败: {exc}")

    if "eval_samples" not in st.session_state:
        st.session_state.eval_samples = [
            {
                "query": "请总结系统中关于RAG的设计",
                "expected_sources": ["example_jd.md"],
                "expected_keywords": ["RAG", "LangChain", "向量"],
            }
        ]

    eval_json_text = st.text_area(
        "评估样本JSON",
        value=json.dumps(st.session_state.eval_samples, ensure_ascii=False, indent=2),
        height=240,
    )

    col1, col2, col3 = st.columns(3)
    retrieve_k = col1.number_input("TopK / retrieve_k", min_value=1, max_value=12, value=6)
    candidate_k = col2.number_input("候选集 candidate_k", min_value=1, max_value=30, value=12)
    rerank_top_n = col3.number_input("Rerank top_n", min_value=1, max_value=30, value=6)
    include_answer_check = st.checkbox("执行答案引用检查（更慢）", value=False)

    if st.button("运行评估报告", type="primary", use_container_width=True):
        try:
            samples = json.loads(eval_json_text)
            req = {
                "samples": samples,
                "retrieve_k": int(retrieve_k),
                "candidate_k": int(candidate_k),
                "rerank_top_n": int(rerank_top_n),
                "include_answer_check": include_answer_check,
                "dataset_name": dataset_name,
            }
            data = request_json("POST", f"{backend_url}/api/mm/evaluate", json=req, timeout=300)

            st.success("评估完成")
            experiments = data.get("experiments", [])
            rows = [
                {
                    "experiment": item.get("label"),
                    f"HitRate@{retrieve_k}": round(float(item.get("hit_rate", 0)), 4),
                    f"MRR@{retrieve_k}": round(float(item.get("mrr", 0)), 4),
                    f"KeywordRecall@{retrieve_k}": round(float(item.get("keyword_recall", 0)), 4),
                }
                for item in experiments
            ]
            st.markdown("### 三组对比实验")
            st.dataframe(rows, use_container_width=True)
            st.bar_chart(rows, x="experiment", y=[f"HitRate@{retrieve_k}", f"MRR@{retrieve_k}", f"KeywordRecall@{retrieve_k}"])

            metrics = data.get("metrics", {})
            hit_delta = metrics.get("rerank_hit_rate", 0) - metrics.get("baseline_hit_rate", 0)
            mrr_delta = metrics.get("rerank_mrr", 0) - metrics.get("baseline_mrr", 0)
            kw_delta = metrics.get("rerank_keyword_recall", 0) - metrics.get("baseline_keyword_recall", 0)
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Baseline HitRate@{retrieve_k}", f"{metrics.get('baseline_hit_rate', 0):.3f}", f"{hit_delta:+.3f}")
            m2.metric(f"Rerank MRR@{retrieve_k}", f"{metrics.get('rerank_mrr', 0):.3f}", f"{mrr_delta:+.3f}")
            m3.metric(f"Rerank KeywordRecall@{retrieve_k}", f"{metrics.get('rerank_keyword_recall', 0):.3f}", f"{kw_delta:+.3f}")

            sample_rows = []
            bad_cases = []
            for item in data.get("samples", []):
                experiments_map = item.get("experiments", {})
                vector = experiments_map.get("vector", {})
                rerank = experiments_map.get("vector_rerank", {})
                row = {
                    "id": item.get("query_id") or item.get("sample_id") or "",
                    "scenario": item.get("scenario") or "",
                    "query": item.get("query"),
                    "vector_hit": vector.get("hit"),
                    "rerank_hit": rerank.get("hit"),
                    "vector_mrr": vector.get("mrr"),
                    "rerank_mrr": rerank.get("mrr"),
                    "vector_sources": ", ".join(vector.get("retrieved_sources", [])),
                    "rerank_sources": ", ".join(rerank.get("retrieved_sources", [])),
                }
                sample_rows.append(row)
                if not rerank.get("hit"):
                    bad_cases.append(row)

            st.markdown("### 逐样本明细")
            st.dataframe(sample_rows, use_container_width=True)
            st.markdown("### Bad Case")
            st.dataframe(bad_cases, use_container_width=True)

            st.markdown("### 评估配置")
            render_json(data.get("config", {}))
            st.markdown("### 全量结果")
            render_json(data)
        except Exception as exc:
            st.error(f"评估失败: {exc}")
