from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter

from app.agent.workflow import WorkflowPolicyInput, plan_workflow
from app.services.structured_analysis import (
    align_evidence,
    extract_skills,
    parse_candidate_profile,
    parse_job_description,
    score_evidence,
    validate_grounded_text,
)


LABEL_VALUE = {"low": 0, "medium": 1, "high": 2}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def precision_recall_f1(expected: set[str], predicted: set[str]) -> tuple[float, float, float]:
    true_positive = len(expected & predicted)
    precision = true_positive / len(predicted) if predicted else float(not expected)
    recall = true_positive / len(expected) if expected else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _rank(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end
    return ranks


def spearman(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) < 2 or len(values_a) != len(values_b):
        return 0.0
    ranks_a, ranks_b = _rank(values_a), _rank(values_b)
    avg_a, avg_b = mean(ranks_a), mean(ranks_b)
    numerator = sum((a - avg_a) * (b - avg_b) for a, b in zip(ranks_a, ranks_b))
    denominator = math.sqrt(
        sum((a - avg_a) ** 2 for a in ranks_a) * sum((b - avg_b) ** 2 for b in ranks_b)
    )
    return numerator / denominator if denominator else 0.0


def macro_f1(expected: list[str], predicted: list[str]) -> float:
    scores = []
    for label in sorted(set(LABEL_VALUE)):
        tp = sum(e == label and p == label for e, p in zip(expected, predicted))
        fp = sum(e != label and p == label for e, p in zip(expected, predicted))
        fn = sum(e == label and p != label for e, p in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return mean(scores)


def label_from_score(score: float, *, medium_threshold: float = 20, high_threshold: float = 55) -> str:
    if score >= high_threshold:
        return "high"
    if score >= medium_threshold:
        return "medium"
    return "low"


def calibrate_thresholds(scores: list[float], expected: list[str]) -> tuple[float, float, float]:
    """Select thresholds on development data only; return medium, high, Macro-F1."""
    best = (0.0, 20.0, 55.0)
    for medium_threshold in range(0, 61):
        for high_threshold in range(medium_threshold + 1, 101):
            predicted = [
                label_from_score(score, medium_threshold=medium_threshold, high_threshold=high_threshold)
                for score in scores
            ]
            value = macro_f1(expected, predicted)
            candidate = (value, float(medium_threshold), float(high_threshold))
            if candidate > best:
                best = candidate
    return best[1], best[2], best[0]


def ndcg(labels: list[int], scores: list[float], k: int = 5) -> float:
    def dcg(items: list[int]) -> float:
        return sum((2**relevance - 1) / math.log2(index + 2) for index, relevance in enumerate(items[:k]))

    ranked = [label for _, label in sorted(zip(scores, labels), key=lambda item: -item[0])]
    ideal = sorted(labels, reverse=True)
    ideal_score = dcg(ideal)
    return dcg(ranked) / ideal_score if ideal_score else 0.0


def first_high_mrr(labels: list[int], scores: list[float]) -> float:
    ranked = [label for _, label in sorted(zip(scores, labels), key=lambda item: -item[0])]
    rank = next((index for index, label in enumerate(ranked, start=1) if label == 2), None)
    return 1.0 / rank if rank else 0.0


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def evaluate_dataset(dataset_dir: Path) -> dict:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    jobs = read_jsonl(dataset_dir / "job_snapshots.jsonl")
    candidates = read_jsonl(dataset_dir / "candidate_profiles.jsonl")
    annotations = read_jsonl(dataset_dir / "match_annotations.jsonl")
    trajectories = read_jsonl(dataset_dir / "agent_trajectories.jsonl")
    generation_cases = read_jsonl(dataset_dir / "generation_validation.jsonl")

    parsed_jobs = {}
    extraction_details = []
    extraction_latencies = []
    aggregate_expected: set[str] = set()
    aggregate_predicted: set[str] = set()
    aggregate_raw_predicted: set[str] = set()
    for row in jobs:
        started = perf_counter()
        parsed = parse_job_description(
            company_name=row["company_name"],
            job_title=row["job_title"],
            description=row["description"],
            location=row.get("location"),
            language=row.get("language", "unknown"),
            source_url=row.get("source_url"),
        )
        extraction_latencies.append((perf_counter() - started) * 1000)
        parsed_jobs[row["job_id"]] = parsed
        expected = set(row["annotation"].get("expected_skills", []))
        predicted = {term for requirement in parsed.requirements for term in requirement.normalized_terms}
        aggregate_expected |= {f"{row['job_id']}:{term}" for term in expected}
        aggregate_predicted |= {f"{row['job_id']}:{term}" for term in predicted}
        aggregate_raw_predicted |= {f"{row['job_id']}:{term}" for term in extract_skills(row["description"])}
        precision, recall, f1 = precision_recall_f1(expected, predicted)
        extraction_details.append(
            {
                "job_id": row["job_id"],
                "company_name": row["company_name"],
                "job_title": row["job_title"],
                "expected_skills": sorted(expected),
                "predicted_skills": sorted(predicted),
                "skill_precision": precision,
                "skill_recall": recall,
                "skill_f1": f1,
                "requirement_count": len(parsed.requirements),
            }
        )
    micro_precision, micro_recall, micro_f1 = precision_recall_f1(aggregate_expected, aggregate_predicted)
    raw_precision, raw_recall, raw_f1 = precision_recall_f1(aggregate_expected, aggregate_raw_predicted)

    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    parsed_candidates = {
        candidate_id: parse_candidate_profile(
            candidate_id=candidate_id,
            sources=[{"content": row["resume_text"], "filename": f"{candidate_id}.md", "section": "complete_resume"}],
            source_kind="synthetic_eval",
        )
        for candidate_id, row in candidate_by_id.items()
    }

    match_details = []
    match_latencies = []
    expected_labels, predicted_labels, baseline_labels, label_values, scores, baseline_scores = [], [], [], [], [], []
    by_candidate: dict[str, list[dict]] = defaultdict(list)
    for annotation in annotations:
        started = perf_counter()
        job = parsed_jobs[annotation["job_id"]]
        candidate = parsed_candidates[annotation["candidate_id"]]
        matrix = align_evidence(job, candidate)
        breakdown = score_evidence(job, matrix)
        match_latencies.append((perf_counter() - started) * 1000)
        predicted_label = label_from_score(breakdown.overall_score)
        job_skills = {term for requirement in job.requirements for term in requirement.normalized_terms}
        candidate_skills = {term for fact in candidate.facts for term in fact.normalized_terms}
        baseline_score = 100.0 * len(job_skills & candidate_skills) / len(job_skills) if job_skills else 0.0
        baseline_label = label_from_score(baseline_score)
        detail = {
            "annotation_id": annotation["annotation_id"],
            "candidate_id": annotation["candidate_id"],
            "job_id": annotation["job_id"],
            "expected_label": annotation["relevance_label"],
            "predicted_label": predicted_label,
            "score": breakdown.overall_score,
            "skill_overlap_baseline_score": round(baseline_score, 2),
            "must_have_coverage": breakdown.must_have_coverage,
            "missing_must_haves": breakdown.missing_must_haves,
        }
        match_details.append(detail)
        by_candidate[annotation["candidate_id"]].append(detail)
        expected_labels.append(annotation["relevance_label"])
        predicted_labels.append(predicted_label)
        baseline_labels.append(baseline_label)
        label_values.append(LABEL_VALUE[annotation["relevance_label"]])
        scores.append(breakdown.overall_score)
        baseline_scores.append(baseline_score)

    ranking_metrics = []
    baseline_ranking_metrics = []
    for candidate_id, details in sorted(by_candidate.items()):
        labels = [LABEL_VALUE[item["expected_label"]] for item in details]
        candidate_scores = [item["score"] for item in details]
        candidate_baseline_scores = [item["skill_overlap_baseline_score"] for item in details]
        ranking_metrics.append(
            {
                "candidate_id": candidate_id,
                "ndcg_at_5": ndcg(labels, candidate_scores, k=5),
                "mrr_first_high": first_high_mrr(labels, candidate_scores),
            }
        )
        baseline_ranking_metrics.append(
            {
                "candidate_id": candidate_id,
                "ndcg_at_5": ndcg(labels, candidate_baseline_scores, k=5),
                "mrr_first_high": first_high_mrr(labels, candidate_baseline_scores),
            }
        )

    splits = {candidate_id: row["split"] for candidate_id, row in candidate_by_id.items()}
    development_indices = [
        index for index, item in enumerate(match_details) if splits[item["candidate_id"]] == "development"
    ]
    test_indices = [index for index, item in enumerate(match_details) if splits[item["candidate_id"]] == "test"]
    dev_expected = [expected_labels[index] for index in development_indices]
    evidence_medium, evidence_high, evidence_dev_f1 = calibrate_thresholds(
        [scores[index] for index in development_indices], dev_expected
    )
    baseline_medium, baseline_high, baseline_dev_f1 = calibrate_thresholds(
        [baseline_scores[index] for index in development_indices], dev_expected
    )
    test_expected = [expected_labels[index] for index in test_indices]
    evidence_test_predicted = [
        label_from_score(scores[index], medium_threshold=evidence_medium, high_threshold=evidence_high)
        for index in test_indices
    ]
    baseline_test_predicted = [
        label_from_score(baseline_scores[index], medium_threshold=baseline_medium, high_threshold=baseline_high)
        for index in test_indices
    ]

    trajectory_details = []
    for row in trajectories:
        plan = plan_workflow(WorkflowPolicyInput.model_validate(row["policy_input"]))
        trajectory_details.append(
            {
                "case_id": row["case_id"],
                "tool_sequence_match": plan.expected_tools == row["expected_tools"],
                "terminal_status_match": plan.terminal_status == row["terminal_status"],
                "terminal_stage_match": plan.terminal_stage.value == row["terminal_stage"],
                "next_action_match": plan.next_action == row.get("next_action"),
                "actual_tools": plan.expected_tools,
            }
        )

    generation_details = []
    expected_codes_flat: list[str] = []
    actual_codes_flat: list[str] = []
    for row in generation_cases:
        actual_codes = sorted({finding.code for finding in validate_grounded_text(row["text"], row["evidence"])})
        expected_codes = sorted(set(row["expected_codes"]))
        generation_details.append(
            {"case_id": row["case_id"], "expected_codes": expected_codes, "actual_codes": actual_codes, "exact_match": expected_codes == actual_codes}
        )
        for code in sorted(set(expected_codes) | set(actual_codes)):
            expected_codes_flat.append(code if code in expected_codes else f"not:{code}")
            actual_codes_flat.append(code if code in actual_codes else f"not:{code}")

    true_positive = sum(
        bool(set(item["expected_codes"]) & set(item["actual_codes"])) for item in generation_details
    )
    predicted_positive = sum(bool(item["actual_codes"]) for item in generation_details)
    expected_positive = sum(bool(item["expected_codes"]) for item in generation_details)
    generation_precision = true_positive / predicted_positive if predicted_positive else 0.0
    generation_recall = true_positive / expected_positive if expected_positive else 0.0
    generation_f1 = (
        2 * generation_precision * generation_recall / (generation_precision + generation_recall)
        if generation_precision + generation_recall
        else 0.0
    )

    return {
        "dataset": {
            "name": manifest["dataset_name"],
            "version": manifest["version"],
            "file_sha256": manifest["file_sha256"],
            "label_status": manifest["label_status"],
        },
        "baseline": {
            "name": "deterministic_skill_lexicon_plus_evidence_weighting",
            "scoring_version": "evidence_weighted_v1",
            "thresholds": {"high": ">=55", "medium": ">=20", "low": "<20"},
        },
        "extraction": {
            "job_count": len(jobs),
            "skill_micro_precision": micro_precision,
            "skill_micro_recall": micro_recall,
            "skill_micro_f1": micro_f1,
            "skill_macro_f1": mean(item["skill_f1"] for item in extraction_details),
            "ablation": {
                "raw_full_text_skill_f1": raw_f1,
                "raw_full_text_skill_precision": raw_precision,
                "raw_full_text_skill_recall": raw_recall,
                "structured_section_skill_f1": micro_f1,
                "structured_minus_raw_f1": micro_f1 - raw_f1,
            },
            "p50_latency_ms": percentile(extraction_latencies, 0.50),
            "p95_latency_ms": percentile(extraction_latencies, 0.95),
            "details": extraction_details,
        },
        "matching": {
            "pair_count": len(annotations),
            "label_distribution": dict(Counter(expected_labels)),
            "accuracy": mean([float(a == b) for a, b in zip(expected_labels, predicted_labels)]),
            "macro_f1": macro_f1(expected_labels, predicted_labels),
            "spearman_score_vs_label": spearman(scores, label_values),
            "mean_ndcg_at_5": mean(item["ndcg_at_5"] for item in ranking_metrics),
            "mean_mrr_first_high": mean(item["mrr_first_high"] for item in ranking_metrics),
            "ablation": {
                "skill_overlap_accuracy": mean([float(a == b) for a, b in zip(expected_labels, baseline_labels)]),
                "skill_overlap_macro_f1": macro_f1(expected_labels, baseline_labels),
                "skill_overlap_spearman": spearman(baseline_scores, label_values),
                "skill_overlap_mean_ndcg_at_5": mean(item["ndcg_at_5"] for item in baseline_ranking_metrics),
                "evidence_weighted_macro_f1": macro_f1(expected_labels, predicted_labels),
                "evidence_weighted_spearman": spearman(scores, label_values),
                "evidence_weighted_mean_ndcg_at_5": mean(item["ndcg_at_5"] for item in ranking_metrics),
            },
            "calibrated_holdout": {
                "development_pair_count": len(development_indices),
                "test_pair_count": len(test_indices),
                "skill_overlap": {
                    "medium_threshold": baseline_medium,
                    "high_threshold": baseline_high,
                    "development_macro_f1": baseline_dev_f1,
                    "test_accuracy": mean(float(a == b) for a, b in zip(test_expected, baseline_test_predicted)),
                    "test_macro_f1": macro_f1(test_expected, baseline_test_predicted),
                    "test_spearman": spearman([baseline_scores[index] for index in test_indices], [LABEL_VALUE[item] for item in test_expected]),
                },
                "evidence_weighted": {
                    "medium_threshold": evidence_medium,
                    "high_threshold": evidence_high,
                    "development_macro_f1": evidence_dev_f1,
                    "test_accuracy": mean(float(a == b) for a, b in zip(test_expected, evidence_test_predicted)),
                    "test_macro_f1": macro_f1(test_expected, evidence_test_predicted),
                    "test_spearman": spearman([scores[index] for index in test_indices], [LABEL_VALUE[item] for item in test_expected]),
                },
            },
            "p50_latency_ms": percentile(match_latencies, 0.50),
            "p95_latency_ms": percentile(match_latencies, 0.95),
            "ranking_by_candidate": ranking_metrics,
            "details": match_details,
        },
        "agent_trajectory": {
            "case_count": len(trajectories),
            "exact_tool_sequence_accuracy": mean(float(item["tool_sequence_match"]) for item in trajectory_details),
            "terminal_status_accuracy": mean(float(item["terminal_status_match"]) for item in trajectory_details),
            "terminal_stage_accuracy": mean(float(item["terminal_stage_match"]) for item in trajectory_details),
            "next_action_accuracy": mean(float(item["next_action_match"]) for item in trajectory_details),
            "details": trajectory_details,
        },
        "generation_validation": {
            "case_count": len(generation_cases),
            "exact_case_accuracy": mean(float(item["exact_match"]) for item in generation_details),
            "finding_precision": generation_precision,
            "finding_recall": generation_recall,
            "finding_f1": generation_f1,
            "details": generation_details,
        },
    }
