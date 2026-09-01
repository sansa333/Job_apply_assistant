from __future__ import annotations

from collections import Counter


VALID_GRADES = {0, 1, 2, 3}


def cohen_kappa(left: list[int], right: list[int]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("two non-empty equally sized label sequences are required")
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts, right_counts = Counter(left), Counter(right)
    expected = sum((left_counts[grade] / len(left)) * (right_counts[grade] / len(right)) for grade in VALID_GRADES)
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0


def finalize_annotation_tasks(tasks: list[dict], *, minimum_kappa: float = 0.70) -> tuple[list[dict], dict]:
    left: list[int] = []
    right: list[int] = []
    unresolved: list[dict] = []
    gold_qrels: list[dict] = []
    for task in tasks:
        query_relevant = 0
        for passage in task["candidate_passages"]:
            first = passage.get("annotator_1_grade")
            second = passage.get("annotator_2_grade")
            if first not in VALID_GRADES or second not in VALID_GRADES:
                unresolved.append(
                    {"query_id": task["query_id"], "evidence_id": passage["evidence_id"], "reason": "missing_label"}
                )
                continue
            left.append(first)
            right.append(second)
            if first != second:
                final = passage.get("adjudicated_grade")
                if final not in VALID_GRADES:
                    unresolved.append(
                        {"query_id": task["query_id"], "evidence_id": passage["evidence_id"], "reason": "needs_adjudication"}
                    )
                    continue
            else:
                final = first
            gold_qrels.append(
                {
                    "query_id": task["query_id"],
                    "evidence_id": passage["evidence_id"],
                    "relevance_grade": final,
                    "label_source": "two_annotators_with_adjudication",
                    "status": "gold",
                }
            )
            query_relevant += final >= 2
        if not unresolved and query_relevant == 0:
            unresolved.append({"query_id": task["query_id"], "evidence_id": None, "reason": "query_has_no_relevant_passage"})
    kappa = cohen_kappa(left, right) if left else 0.0
    report = {
        "complete": not unresolved and kappa >= minimum_kappa,
        "labelled_pairs": len(left),
        "raw_agreement": sum(a == b for a, b in zip(left, right)) / len(left) if left else 0.0,
        "cohen_kappa": kappa,
        "minimum_kappa": minimum_kappa,
        "unresolved_count": len(unresolved),
        "unresolved": unresolved[:500],
        "gold_qrel_count": len(gold_qrels),
    }
    return gold_qrels if report["complete"] else [], report
