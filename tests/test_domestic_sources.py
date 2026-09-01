from __future__ import annotations

import json

from app.domestic.sources import (
    BaiduCampusSource,
    CuratedCampusNotice,
    CuratedCampusNoticeSource,
    DjiCampusHotJobsSource,
    HonorCampusHighlightsSource,
    KylinCampusSource,
    MeituanCampusAiSource,
    OppoCampusSource,
)


def test_baidu_parser_normalizes_a_campus_job(monkeypatch) -> None:
    payload = {
        "listData": {
            "listDetailData": [
                {
                    "jobId": "baidu-1",
                    "name": "大模型应用开发工程师",
                    "workContent": "负责 RAG 和 Agent 应用开发",
                    "serviceCondition": "计算机相关专业，硕士及以上，熟悉 Python",
                    "workPlace": "北京",
                    "updateDate": "2026-08-20",
                }
            ]
        }
    }
    source = BaiduCampusSource()
    monkeypatch.setattr(
        source,
        "_get_text",
        lambda _: f"<script>window.__INITIAL_DATA__ = {json.dumps(payload, ensure_ascii=False)};</script>",
    )
    jobs = source.fetch()
    assert len(jobs) == 1
    assert jobs[0].external_id == "baidu-1"
    assert jobs[0].job_category == "agent_development"
    assert jobs[0].graduation_year == 2027
    assert jobs[0].is_domestic is True


def test_baidu_parser_accepts_javascript_undefined_metadata(monkeypatch) -> None:
    payload = {
        "listData": {
            "listDetailData": [
                {
                    "jobId": "baidu-undefined",
                    "name": "AI 应用开发工程师",
                    "workContent": "负责大模型应用开发",
                    "serviceCondition": "本科及以上，熟悉 Python",
                    "workPlace": "北京",
                }
            ]
        }
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    source = BaiduCampusSource()
    monkeypatch.setattr(
        source,
        "_get_text",
        lambda _: (
            "<script>window.__INITIAL_DATA__ = "
            f'{{"projectType":undefined,"listData":{serialized[12:-1]}}};</script>'
        ),
    )
    jobs = source.fetch()
    assert [job.external_id for job in jobs] == ["baidu-undefined"]


def test_dji_parser_resolves_relative_apply_url(monkeypatch) -> None:
    html = """
    <div class="pc_card_test"><h4>AI 全栈开发工程师</h4>
      <p>使用 Python 开发 AI 应用平台</p>
      <span class="pc_tag_test">深圳</span><a href="/zh-CN/jobs/123">详情</a>
    </div>
    """
    source = DjiCampusHotJobsSource()
    monkeypatch.setattr(source, "_get_text", lambda _: html)
    jobs = source.fetch()
    assert len(jobs) == 1
    assert jobs[0].apply_url == "https://careers.dji.com/zh-CN/jobs/123"
    assert jobs[0].location == "深圳"


def test_kylin_parser_extracts_location_date_and_id(monkeypatch) -> None:
    html = """
    <li class="content-item"><h4>智能体平台开发工程师</h4>
      <section class="item-section">工作地点：上海\n发布时间：2026-08-01\n负责 Agent 工具调用
        <a href="/about/job/detail.html?jobAdId=88">投递</a>
      </section>
    </li>
    """
    source = KylinCampusSource()
    monkeypatch.setattr(source, "_get_text", lambda _: html)
    jobs = source.fetch()
    assert len(jobs) == 1
    assert jobs[0].external_id == "88"
    assert jobs[0].posted_at == "2026-08-01"
    assert jobs[0].apply_url.startswith("https://www.kylinos.cn/")


def test_oppo_public_api_filters_and_normalizes_target_jobs(monkeypatch) -> None:
    source = OppoCampusSource()
    monkeypatch.setattr(
        source,
        "_post_json",
        lambda *_args, **_kwargs: {
            "code": 0,
            "data": {
                "total": 1,
                "records": [
                    {
                        "idProjPosition": 1821,
                        "projectName": "2027届应届生校园招聘",
                        "recruitmentType": "Graduate",
                        "positionTypeName": "软件类",
                        "positionName": "AI工程师（AI Agent方向）",
                        "positionDesc": "负责 Agent 产品研发落地",
                        "positionRequire": "本科及以上，熟悉 Python 和 Linux",
                        "workCityName": "深圳市,成都市",
                        "releaseTime": "2026-07-15",
                    }
                ],
            },
        },
    )
    jobs = source.fetch()
    assert len(jobs) == 1
    assert jobs[0].company_name == "OPPO"
    assert jobs[0].external_id == "1821"
    assert jobs[0].job_category == "agent_development"


def test_meituan_official_api_deduplicates_keyword_results(monkeypatch) -> None:
    source = MeituanCampusAiSource()
    row = {
        "jobUnionId": "4697317646",
        "name": "AI Agent开发工程师",
        "jobType": "1",
        "jobStatus": "000",
        "jobFamily": "技术类",
        "jobDuty": "负责 Agentic 工作流研发",
        "jobRequirement": "本科及以上，熟悉 Python、RAG 和 LangChain",
        "cityList": [{"name": "北京市"}, {"name": "上海市"}],
        "refreshTime": 1788055369000,
    }
    monkeypatch.setattr(
        source,
        "_post_json",
        lambda *_args, **_kwargs: {"status": 1, "data": {"list": [row]}},
    )
    jobs = source.fetch()
    assert len(jobs) == 1
    assert jobs[0].company_name == "美团"
    assert jobs[0].location == "北京市、上海市"
    assert jobs[0].recruitment_type == "campus"


def test_honor_highlight_parser_keeps_unique_campus_links(monkeypatch) -> None:
    html = """
    <a href="https://career.honor.com/pb/posDetail.html?postId=abc&postType=campus">大模型算法工程师</a>
    <a href="https://career.honor.com/pb/posDetail.html?postId=abc&postType=campus">大模型算法工程师</a>
    """
    source = HonorCampusHighlightsSource()
    monkeypatch.setattr(source, "_get_text", lambda _: html)
    jobs = source.fetch()
    assert len(jobs) == 1
    assert jobs[0].company_name == "荣耀"
    assert jobs[0].graduation_year == 2027


def test_curated_notice_parser_marks_bundle_and_validates_2027(monkeypatch) -> None:
    notice = CuratedCampusNotice(
        slug="example",
        company_name="示例科技",
        title="2027 校招 Agent 岗位合集（公开公告）",
        source_url="https://career.example.test/2027",
        focus="Agent、RAG 和软件开发",
        publisher="示例大学就业网",
        company_token="示例科技",
        location="北京",
        posted_at="2026-08-30",
    )
    source = CuratedCampusNoticeSource(notices=(notice,))
    monkeypatch.setattr(
        source,
        "_get_text",
        lambda _: "<html><body>示例科技 2027 届校园招聘，招聘 Agent 开发工程师，要求本科。</body></html>",
    )
    jobs = source.fetch()
    assert len(jobs) == 1
    assert jobs[0].company_name == "示例科技"
    assert jobs[0].job_category == "agent_development"
    assert jobs[0].source_name.endswith("示例大学就业网")
    assert '"detail_level": "public_notice_bundle"' in jobs[0].raw_payload_json
