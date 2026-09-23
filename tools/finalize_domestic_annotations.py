"""Validate completed independent reviews and create a NEW frozen dataset.

The original silver corpus is never overwritten. This command does not perform
or impersonate human annotation, and rejects missing/partial reviews.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from app.evaluation.domestic import read_jsonl, verify_dataset, write_jsonl


def finalize(source: Path, reviews: Path, output: Path, reviewer_a: str, reviewer_b: str) -> dict:
    manifest = verify_dataset(source)
    if not reviewer_a.strip() or not reviewer_b.strip() or reviewer_a.strip() == reviewer_b.strip():
        raise ValueError("Two distinct reviewer identifiers are required")
    expected = {(r["query_id"], r["evidence_id"]): r for r in read_jsonl(source / "annotation_tasks.jsonl")}
    reviewed = read_jsonl(reviews)
    seen, positives, decisions = set(), [], []
    for row in reviewed:
        pair = (row["query_id"], row["evidence_id"])
        if pair not in expected or pair in seen:
            raise ValueError("Unknown or duplicate annotation pair")
        seen.add(pair)
        left, right = row.get("annotator_1"), row.get("annotator_2")
        if type(left) is not int or type(right) is not int or left not in (0, 1) or right not in (0, 1):
            raise ValueError("All pairs require two completed binary judgments")
        grade = left if left == right else row.get("adjudicated_grade")
        if type(grade) is not int or grade not in (0, 1):
            raise ValueError("Disagreements require adjudication")
        if left != right and (not row.get("adjudicator") or row["adjudicator"] in (reviewer_a, reviewer_b)):
            raise ValueError("Disagreements require a third reviewer identifier")
        decisions.append((left, right))
        if grade:
            positives.append({"query_id": pair[0], "evidence_id": pair[1], "grade": 1,
                              "label_source": "declared_independent_human_review", "human_grade": 1})
    if seen != expected.keys():
        raise ValueError("Incomplete coverage: every frozen query/chunk pair must be reviewed")
    n = len(decisions)
    counts_a, counts_b = Counter(a for a, _ in decisions), Counter(b for _, b in decisions)
    observed = sum(a == b for a, b in decisions) / n if n else 0
    chance = sum(counts_a[i] * counts_b[i] for i in (0, 1)) / n**2 if n else 0
    agreement = {"pairs": n, "agreement": observed, "cohens_kappa": (observed - chance) / (1 - chance) if chance < 1 else None,
                 "reviewers": [reviewer_a, reviewer_b], "source_manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest["files"]:
        if name not in {"qrels.jsonl", "queries.jsonl", "annotation_tasks.jsonl"}:
            shutil.copyfile(source / name, output / name)
    write_jsonl(output / "qrels.jsonl", positives)
    write_jsonl(output / "queries.jsonl", [{**q, "annotation_status": "human_double_reviewed"} for q in read_jsonl(source / "queries.jsonl")])
    write_jsonl(output / "annotation_tasks.jsonl", reviewed)
    manifest.update(annotation_status="human_double_reviewed", annotation_agreement=agreement,
                    label_definition="Two declared independent binary relevance judgments; disagreement adjudicated by a third reviewer. Search labels remain silver.")
    manifest["files"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.jsonl")}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return agreement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer-a", required=True)
    parser.add_argument("--reviewer-b", required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.dataset, args.reviews, args.output, args.reviewer_a, args.reviewer_b), ensure_ascii=False))


if __name__ == "__main__":
    main()
