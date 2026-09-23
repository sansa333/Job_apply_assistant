"""Re-score saved rankings against COMPLETE declared independent human reviews.

Never fills reviews or upgrades assistant labels. Original datasets/results stay
unchanged. Human relevance does not validate the assistant's quoted-span coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from app.evaluation.business import calibrate_threshold, calibration_metrics, graded_metrics, random_expectation, summarize
from app.evaluation.domestic import read_jsonl, verify_dataset


def validate_reviews(expected, rows, id_field, reviewer_a, reviewer_b):
    if not reviewer_a.strip() or not reviewer_b.strip() or reviewer_a.strip() == reviewer_b.strip():
        raise ValueError("Two distinct reviewer identifiers required")
    expected = {(r["query_id"], r[id_field]): r for r in expected}
    seen, grades, decisions = set(), {}, []
    for row in rows:
        pair = (row["query_id"], row[id_field])
        if pair not in expected or pair in seen:
            raise ValueError("Unknown or duplicate review pair")
        seen.add(pair)
        a, b = row.get("reviewer_a_grade"), row.get("reviewer_b_grade")
        if any(type(g) is not int or g not in (0, 1, 2) for g in (a, b)):
            raise ValueError("Every pair requires two completed grades: 0, 1 or 2")
        grade = a if a == b else row.get("adjudicated_grade")
        if type(grade) is not int or grade not in (0, 1, 2):
            raise ValueError("Disagreement requires an adjudicated grade")
        if a != b and (not row.get("adjudicator") or row["adjudicator"] in (reviewer_a, reviewer_b)):
            raise ValueError("Disagreement requires a third reviewer identifier")
        if not expected[pair].get("eligible", True) and grade:
            raise ValueError("Hard-filter-ineligible jobs cannot be relevant")
        grades.setdefault(pair[0], {})[pair[1]] = grade
        decisions.append((a, b))
    if seen != expected.keys():
        raise ValueError("Incomplete reviews: exhaustive frozen matrix required")
    n = len(decisions)
    ca, cb = Counter(a for a, _ in decisions), Counter(b for _, b in decisions)
    observed = sum(a == b for a, b in decisions) / n if n else 0
    chance = sum(ca[g] * cb[g] for g in (0, 1, 2)) / n**2 if n else 0
    return grades, {"pairs": n, "agreement": observed,
        "cohens_kappa": (observed - chance) / (1 - chance) if chance < 1 else None}


def review_metric_ks(run, task):
    if task == "discovery":
        return (1, 2, 3, 5, 10, 20)
    # Never report Recall@5 by padding legacy Top3-only rankings.
    return (1, 2, 3, 5) if run.get("evidence_retrieval_k", 3) >= 5 else (1, 2, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "run", "evidence-reviews", "search-reviews", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reviewer-a", required=True)
    parser.add_argument("--reviewer-b", required=True)
    args = parser.parse_args()
    manifest = verify_dataset(args.dataset)
    run = json.loads((args.run / "report.json").read_text(encoding="utf-8"))
    if run["dataset"]["files"] != manifest["files"]:
        raise ValueError("Run and dataset identity mismatch")
    original_e = read_jsonl(args.dataset / "qrels.jsonl")
    original_s = read_jsonl(args.dataset / "search_qrels.jsonl")
    eg, ea = validate_reviews(original_e, read_jsonl(args.evidence_reviews), "evidence_id", args.reviewer_a, args.reviewer_b)
    sg, sa = validate_reviews(original_s, read_jsonl(args.search_reviews), "job_id", args.reviewer_a, args.reviewer_b)
    candidates = {}
    for r in original_s:
        if r["eligible"]:
            candidates.setdefault(r["query_id"], []).append(r["job_id"])

    def rescore(item, task, random=False):
        labels, threshold = (sg, 2) if task == "discovery" else (eg, 1)
        ks = review_metric_ks(run, task)
        rows = []
        for old in item["details"]:
            qid = old["query_id"]
            grades = labels[qid]
            if random:
                metrics = random_expectation(candidates.get(qid, []) if task == "discovery" else list(grades), grades,
                    relevant_grade=threshold, ks=ks)
            else:
                metrics = graded_metrics(old["ranking"], grades, relevant_grade=threshold, ks=ks)
            rows.append({"query_id": qid, "split": old["split"], "group_id": old["group_id"], **metrics})
        return {"groups": {s: summarize([r for r in rows if r["split"] == s]) for s in ("development", "test")}, "details": rows}

    report = {"annotation_status": "declared_independent_double_review; identities not independently verified",
        "reviewers": [args.reviewer_a, args.reviewer_b], "agreement": {"evidence": ea, "discovery": sa},
        "scope": "Relevance metrics only. No human-validated span coverage or generation accuracy claim.",
        "baselines": {task: {m: rescore(item, task, m == "random") for m, item in methods.items()}
                      for task, methods in run["baselines"].items()}, "models": []}
    for m in run["models"]:
        scores = [{k: v for k, v in r.items() if k != "complete_source_support"} for r in m["gate_scores"]]
        for row in scores:
            row["answerable"] = any(g > 0 for g in eg[row["query_id"]].values())
        threshold = calibrate_threshold([r for r in scores if r["split"] == "development"])
        report["models"].append({"name": m["name"], "evidence": rescore(m["evidence"], "evidence"),
            "evidence_hybrid": rescore(m["evidence_hybrid"], "evidence"),
            "discovery": {s: rescore(item, "discovery") for s, item in m["discovery"].items()},
            "gate_threshold_development_only": threshold,
            "gate": {s: calibration_metrics([r for r in scores if r["split"] == s], threshold) for s in ("development", "test")}})
    report["input_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (args.dataset / "manifest.json", args.run / "report.json", args.evidence_reviews, args.search_reviews)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output / "report.json")


if __name__ == "__main__":
    main()
