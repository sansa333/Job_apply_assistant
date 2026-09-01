import json
from collections import Counter, defaultdict
from pathlib import Path

from app.evaluation.retrieval_v2 import (
    load_retrieval_v2,
    paired_bootstrap,
    select_on_development,
    validate_retrieval_v2,
    verify_retrieval_v2_manifest,
)


DATASET_DIR = Path("data/eval_dataset/job_retrieval_v2")


def test_retrieval_v2_dataset_passes_full_quality_gate() -> None:
    dataset = load_retrieval_v2(DATASET_DIR)
    quality = validate_retrieval_v2(dataset)

    assert quality["valid"], quality["errors"]
    assert quality["statistics"]["jobs"] == 120
    assert quality["statistics"]["queries"] == 480
    assert quality["statistics"]["unique_queries"] == 480
    assert quality["statistics"]["multi_relevant_query_ratio"] >= 0.40
    assert quality["statistics"]["split_job_distribution"] == {"development": 84, "test": 36}


def test_retrieval_v2_manifest_freezes_every_published_jsonl_file() -> None:
    verification = verify_retrieval_v2_manifest(DATASET_DIR)

    assert verification["valid"], verification["mismatches"]
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert "annotation_tasks.jsonl" not in manifest["files"]


def test_queries_and_candidate_pools_are_concrete_and_complete() -> None:
    dataset = load_retrieval_v2(DATASET_DIR)
    queries = dataset["queries"]
    qrels_by_query: dict[str, list[dict]] = defaultdict(list)
    for qrel in dataset["qrels"]:
        qrels_by_query[qrel["query_id"]].append(qrel)
    pools = {item["query_id"]: item["candidates"] for item in dataset["candidate_pools"]}

    assert all("某某" not in item["query"] for item in queries)
    assert all(item["annotation_status"] == "silver_expert_review_required" for item in queries)
    assert all(len({candidate["evidence_id"] for candidate in pools[item["query_id"]]}) == 50 for item in queries)
    assert all(
        {qrel["evidence_id"] for qrel in qrels_by_query[item["query_id"]]}
        <= {candidate["evidence_id"] for candidate in pools[item["query_id"]]}
        for item in queries
    )
    assert Counter(item["query_type"] for item in queries) == {
        "responsibilities": 120,
        "technical_skills": 120,
        "qualifications": 120,
        "work_context": 120,
    }


def test_annotation_tasks_are_documented_as_a_local_regenerable_artifact() -> None:
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text(encoding="utf-8"))
    policy = manifest["local_artifacts"]["annotation_tasks.jsonl"]

    assert policy == {
        "rows": 480,
        "versioned": False,
        "regenerate_with": "python -m tools.build_retrieval_v2_dataset",
    }


def test_model_selection_uses_development_and_bootstrap_is_paired() -> None:
    def result(name: str, reciprocal_ranks: list[float]) -> dict:
        return {
            "status": "ok",
            "name": name,
            "split": "development",
            "evaluation_scope": "job_scoped",
            "metrics": {
                "mrr_at_3": sum(reciprocal_ranks) / len(reciprocal_ranks),
                "recall_at_5": 1.0,
                "hit_rate_at_1": 0.5,
            },
            "resources": {"mean_query_total_ms": 10.0, "p95_query_total_ms": 12.0},
            "details": [
                {"query_id": f"q{index}", "reciprocal_rank_at_3": value}
                for index, value in enumerate(reciprocal_ranks)
            ],
        }

    winner, runner = select_on_development([result("winner", [1.0, 1.0, 0.5]), result("runner", [0.5, 0.5, 0.0])])
    evidence = paired_bootstrap(winner, runner, iterations=200)

    assert winner["name"] == "winner"
    assert evidence["paired_query_count"] == 3
    assert evidence["delta"] == 0.5
    assert evidence["wins"] == 3
