from app.evaluation.annotation import cohen_kappa, finalize_annotation_tasks


def test_cohen_kappa_is_one_for_identical_labels() -> None:
    assert cohen_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0


def test_incomplete_human_labels_cannot_be_exported_as_gold() -> None:
    tasks = [
        {
            "query_id": "q1",
            "candidate_passages": [
                {
                    "evidence_id": "e1",
                    "annotator_1_grade": 3,
                    "annotator_2_grade": None,
                    "adjudicated_grade": None,
                }
            ],
        }
    ]

    qrels, report = finalize_annotation_tasks(tasks)

    assert qrels == []
    assert not report["complete"]
    assert report["unresolved_count"] == 1


def test_disagreement_requires_adjudication_then_exports_gold() -> None:
    tasks = [
        {
            "query_id": "q1",
            "candidate_passages": [
                {
                    "evidence_id": "e1",
                    "annotator_1_grade": 3,
                    "annotator_2_grade": 2,
                    "adjudicated_grade": 3,
                },
                {
                    "evidence_id": "e2",
                    "annotator_1_grade": 0,
                    "annotator_2_grade": 0,
                    "adjudicated_grade": None,
                },
            ],
        }
    ]

    qrels, report = finalize_annotation_tasks(tasks, minimum_kappa=-1.0)

    assert report["complete"]
    assert {item["relevance_grade"] for item in qrels} == {0, 3}
