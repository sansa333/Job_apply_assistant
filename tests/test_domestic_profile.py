from __future__ import annotations

from app.domestic.profile import derive_candidate_profile, update_candidate_preferences


def test_derived_profile_keeps_resume_read_only_policy(tmp_path) -> None:
    source = tmp_path / "resume.pdf"
    original = b"immutable-pdf-bytes"
    source.write_bytes(original)
    text = (
        "2024.09-2027.06 浙江大学 软件工程 硕士\n"
        "求职意向：后端开发工程师、Agent开发工程师\n"
        "期望城市：杭州，北京\n"
        "Python FastAPI LangChain RAG Agent 项目经历"
    )
    profile = derive_candidate_profile("candidate", text, source)
    assert source.read_bytes() == original
    assert profile["resume_content_policy"] == "read_only_no_rewrite"
    assert profile["target_country"] == "中国"
    assert profile["graduation_year"] == 2027
    assert profile["schools"] == ["浙江大学"]
    assert profile["majors"] == ["软件工程"]
    assert profile["target_roles"] == ["后端开发工程师", "Agent开发工程师"]
    assert profile["target_cities"] == ["杭州", "北京"]
    assert {"Python", "FastAPI", "LangChain", "RAG", "Agent"}.issubset(profile["skills"])


def test_profile_preferences_override_inference_without_person_specific_defaults(tmp_path) -> None:
    source = tmp_path / "resume.pdf"
    source.write_bytes(b"immutable-pdf-bytes")
    text = "2019.09-2023.06 北京理工大学 计算机科学与技术 本科"
    profile = derive_candidate_profile(
        "candidate",
        text,
        source,
        graduation_year=2026,
        target_roles=["数据开发工程师"],
        target_cities=["深圳"],
    )
    assert profile["schools"] == ["北京理工大学"]
    assert profile["majors"] == ["计算机科学与技术"]
    assert profile["graduation_year"] == 2026
    assert profile["graduation_year_evidence"] == "user_preference"
    assert profile["target_roles"] == ["数据开发工程师"]
    assert profile["target_cities"] == ["深圳"]


def test_update_candidate_preferences_persists_current_user_choices(tmp_path) -> None:
    profile_dir = tmp_path / "candidate_profiles" / "candidate-b"
    profile_dir.mkdir(parents=True)
    (profile_dir / "profile.json").write_text(
        '{"candidate_id":"candidate-b","default_filters":{}}', encoding="utf-8"
    )
    result = update_candidate_preferences(
        tmp_path,
        "candidate-b",
        graduation_year=2028,
        target_roles=["算法工程师", "算法工程师"],
        target_cities=["成都", "上海"],
    )
    assert result["graduation_year"] == 2028
    assert result["target_roles"] == ["算法工程师"]
    assert result["target_cities"] == ["成都", "上海"]
    assert result["default_filters"]["city_filter"] == ["成都", "上海"]
