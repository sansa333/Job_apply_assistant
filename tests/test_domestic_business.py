from __future__ import annotations

import itertools
import json
from dataclasses import replace
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from app.config import settings
from app.domestic.evidence_check import check_evidence
from app.domestic.service import DomesticJobService
from app.embeddings import HashEmbeddings
from app.evaluation.business import (
    build_business_dataset, calibrate_threshold, calibration_metrics,
    graded_metrics, random_expectation, summarize,
)
from app.evaluation.domestic import read_jsonl, verify_dataset
from app.knowledge.catalog import JobCatalog
from app.knowledge.ingestion import JobKnowledgeIngestion
from app.knowledge.models import NormalizedJob
from tools.evaluate_business_reviews import review_metric_ks, validate_reviews


def add_job(catalog, title="Linux设备驱动", description="岗位职责：\n负责Linux设备驱动适配。\n任职要求：\n掌握C++编程，Rust经验优先。", year=2027):
    return catalog.upsert(NormalizedJob(company_name="甲公司", job_title=title, description=description,
        location="北京", source_kind="open_source", source_dataset="official_test", source_file="job.md",
        source_url="https://example.com/job", language="zh", is_domestic=True,
        recruitment_type="campus", graduation_year=year)).record


def test_random_baseline_matches_exhaustive_permutations():
    grades = {"a": 2, "b": 1, "c": 0, "d": 2}
    empirical = summarize([graded_metrics(p, grades, relevant_grade=2) for p in itertools.permutations(grades)])
    analytic = random_expectation(list(grades), grades, relevant_grade=2)
    for key in analytic:
        if key.startswith(("mrr_", "recall_", "ndcg_", "precision_", "hit_")):
            assert analytic[key] == pytest.approx(empirical[key])


def test_top5_precision_uses_fixed_slots_and_hit_requires_full_relevance():
    grades = {"full": 2, "partial": 1, "irrelevant": 0}
    m = graded_metrics(["partial", "full", "full"], grades, relevant_grade=2, ks=(5,))
    assert m["precision_at_5"] == .2  # one full hit, not 1/2 or 2/5
    assert m["hit_at_5"] == 1
    assert 0 < m["ndcg_at_5"] < 1
    partial = graded_metrics(["partial"], grades, relevant_grade=2, ks=(5,))
    assert partial["precision_at_5"] == 0 and partial["hit_at_5"] == 0
    assert partial["ndcg_at_5"] > 0
    assert graded_metrics([], {"none": 0}, ks=(5,))["precision_at_5"] == 0


def test_saved_top5_rescore_retains_missing_queries_and_validates_labels():
    from tools.evaluate_discovery_top5 import score_method
    queries = {q: {"query": q, "split": "test", "group_id": q} for q in ("a", "b")}
    labels = {"a": {"job": 2}, "b": {"job": 0}}
    rows = {"details": [{"query_id": q, "ranking": ["job"]} for q in queries]}
    result = score_method(rows, queries, labels, {q: ["job"] for q in queries})["groups"]["test"]
    assert result["answerable_only"]["precision_at_5"] == .2
    assert result["all_query_precision_at_5"] == .1
    assert result["all_query_hit_at_5"] == .5
    assert result["no_relevant_returned_nonempty"] == 1
    rows["details"][0]["ndcg_at_5"] = .1
    with pytest.raises(ValueError, match="disagrees"):
        score_method(rows, queries, labels, {q: ["job"] for q in queries})


def test_partial_support_is_not_fully_satisfying_job_recall():
    result = graded_metrics(["partial", "missing", "full", "full"], {"partial": 1, "full": 2}, relevant_grade=2)
    assert result["mrr_at_3"] == pytest.approx(1 / 3)
    assert result["recall_at_2"] == 0
    assert result["recall_at_3"] == 1
    assert 0 < result["ndcg_at_2"] < 1
    assert graded_metrics(["missing"], {"missing": 0})["mrr_at_3"] is None
    assert random_expectation([], {})["recall_at_2"] is None


def test_same_job_top5_recall_and_top3_mrr_have_separate_cutoffs():
    grades = {"answer_a": 2, "answer_b": 1, "unrelated": 0}
    result = graded_metrics(["n1", "n2", "n3", "answer_a", "answer_b"], grades, ks=(3, 5))
    assert result["mrr_at_3"] == 0
    assert result["recall_at_3"] == 0
    assert result["recall_at_5"] == 1
    assert graded_metrics(["answer_a", "n1", "n2", "n3", "n4"], grades)["recall_at_5"] == .5
    assert 5 not in review_metric_ks({}, "evidence")
    assert 5 in review_metric_ks({"evidence_retrieval_k": 5}, "evidence")


def test_jsonl_preserves_unicode_paragraphs_in_official_jd(tmp_path):
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps({"text": "原文\u2028第二行\u2029第三行"}, ensure_ascii=False) + "\n", encoding="utf-8")
    assert read_jsonl(path) == [{"text": "原文\u2028第二行\u2029第三行"}]


def test_calibration_is_development_only_and_reports_abstention_cost():
    rows = [{"split": "development", "answerable": a, "score": s}
            for a, s in [(True, .8), (True, .4), (False, .6), (False, .2)]]
    threshold = calibrate_threshold(rows, max_false_accept=0)
    m = calibration_metrics(rows, threshold)
    assert m["false_accept_rate"] == 0
    assert m["false_reject_rate"] == .5
    assert m["coverage"] == .25
    assert calibration_metrics(rows, -2)["false_accept_rate"] == 1
    assert calibration_metrics(rows, 2)["accepted_unsupported_rate"] is None
    with pytest.raises(ValueError, match="development"):
        calibrate_threshold([{**rows[0], "split": "test"}, rows[2]])
    with pytest.raises(ValueError, match="both"):
        calibrate_threshold([rows[0]])


def test_dataset_requires_source_support_and_separates_absence(tmp_path, monkeypatch):
    catalog = JobCatalog(tmp_path / "jobs.sqlite3")
    job = add_job(catalog)
    monkeypatch.setattr("app.evaluation.business.EVIDENCE_CASES", [(job.job_id, "test",
        [("学历条件？", ["掌握C++编程", "Rust经验优先"], "multi_support")], ["薪资多少？"])])
    monkeypatch.setattr("app.evaluation.business.DISCOVERY_CASES", [("t1", "test", "驱动岗位", ["驱动"], {})])
    directory = tmp_path / "frozen"
    manifest = build_business_dataset(directory, [job, replace(job, job_id="z_duplicate")])
    assert manifest["jobs"] == 1
    assert verify_dataset(directory) == manifest
    labels = read_jsonl(directory / "qrels.jsonl")
    assert any(r["grade"] == 2 for r in labels if r["query_id"].endswith(":0"))
    assert not any(r["grade"] for r in labels if r["query_id"].endswith(":1"))
    assert all(r["reviewer_a_grade"] is None for r in read_jsonl(directory / "human_review_evidence.jsonl"))
    monkeypatch.setattr("app.evaluation.business.EVIDENCE_CASES", [(job.job_id, "test",
        [("工资？", ["月薪三万元"], "numeric")], [])])
    with pytest.raises(ValueError, match="Quote not present"):
        build_business_dataset(tmp_path / "invalid", [job])


def test_full_catalog_search_keeps_hard_filters_and_unknown_year_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "source_corpus_dir", tmp_path / "profiles")
    catalog = JobCatalog(tmp_path / "jobs.sqlite3")
    target = add_job(catalog)
    unknown = add_job(catalog, title="Linux未知年份", year=None)
    add_job(catalog, title="Qt客户端", description="岗位职责：负责Qt界面开发。", year=2026)
    factory = Mock(side_effect=AssertionError("BM25 should not load a dense model"))
    service = DomesticJobService(catalog=catalog, sources={}, ingestion_factory=factory)
    result = service.search(keyword="我希望参与Linux设备驱动的移植工作", strategy="bm25",
        graduation_year=2027, strict_graduation=True, limit=1)
    assert result["matches"][0]["job_id"] == target.job_id
    assert result["eligible_candidates"] == 1
    loose = service.search(keyword="Linux", strategy="bm25", graduation_year=2027)
    assert {r["job_id"] for r in loose["matches"]} == {target.job_id, unknown.job_id}
    assert service.search(keyword="Linux", strategy="bm25", location="上海")["count"] == 0


def test_global_vector_recall_cannot_escape_allowed_jobs(tmp_path):
    catalog = JobCatalog(tmp_path / "jobs.sqlite3")
    allowed = add_job(catalog)
    other = add_job(catalog, title="Qt客户端", description="负责Qt界面开发。")
    index = JobKnowledgeIngestion(catalog=catalog, source_corpus_dir=tmp_path / "sources",
        vector_db_dir=tmp_path / "index", embeddings=HashEmbeddings(), embedding_backend="hash")
    try:
        index.rebuild_records([allowed, other])
        docs = index.retrieve_across_jobs("Qt界面开发", [allowed.job_id], k=20)
        assert docs and {d.metadata["job_id"] for d in docs} == {allowed.job_id}
        assert index.retrieve_across_jobs("Qt", []) == []
        with pytest.raises(ValueError, match="normalized"):
            index.scored_source_evidence(allowed.job_id, "Linux")
    finally:
        index.close()


@pytest.mark.parametrize("content,status", [
    ("invalid", "invalid_calibration"),
    ('{"threshold": NaN}', "invalid_calibration"),
    ('{"threshold": 0.5}', "calibration_not_approved"),
    ('{"threshold": 0.5, "production_approved": true, "fit_split": "test"}', "calibration_mismatch"),
])
def test_unusable_gate_never_returns_evidence(tmp_path, content, status):
    path = tmp_path / "policy.json"
    path.write_text(content, encoding="utf-8")
    index = Mock(expected_manifest={"model": "model", "collection": "test"})
    assert check_evidence(index, "job", "query", path)["status"] == status
    index.scored_source_evidence.assert_not_called()


def test_valid_gate_retains_provenance_and_handles_empty_candidates(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"threshold": .7, "embedding_identity": {"model": "model"},
        "fit_split": "development", "production_approved": True, "version": "v1"}), encoding="utf-8")
    index = Mock(expected_manifest={"model": "model", "collection": "test"})
    index.scored_source_evidence.return_value = [(Document(page_content="原文", metadata={"chunk_id": "c1", "job_id": "j"}), .8)]
    result = check_evidence(index, "j", "问题", path)
    assert not result["abstained"] and result["documents"][0]["chunk_id"] == "c1"
    index.scored_source_evidence.return_value = []
    assert check_evidence(index, "j", "问题", path)["abstained"]
    assert check_evidence(None, "j", "问题", None)["status"] == "not_calibrated"


def test_human_review_import_rejects_missing_and_unadjudicated_grades():
    expected = [{"query_id": "q", "job_id": "j", "eligible": True}]
    with pytest.raises(ValueError, match="Incomplete"):
        validate_reviews(expected, [], "job_id", "alice", "bob")
    row = {"query_id": "q", "job_id": "j", "reviewer_a_grade": 2, "reviewer_b_grade": 0}
    with pytest.raises(ValueError, match="adjudicated"):
        validate_reviews(expected, [row], "job_id", "alice", "bob")
    row.update(adjudicated_grade=1, adjudicator="charlie")
    grades, agreement = validate_reviews(expected, [row], "job_id", "alice", "bob")
    assert grades == {"q": {"j": 1}} and agreement["agreement"] == 0
    with pytest.raises(ValueError, match="ineligible"):
        validate_reviews([{**expected[0], "eligible": False}], [row], "job_id", "alice", "bob")
