from __future__ import annotations

import json

from app.config import settings
from app.evaluation.annotation import finalize_annotation_tasks


def main() -> None:
    dataset_dir = settings.data_dir / "eval_dataset" / "job_retrieval_v2"
    tasks = [
        json.loads(line)
        for line in (dataset_dir / "annotation_tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gold_qrels, report = finalize_annotation_tasks(tasks)
    report_path = dataset_dir / "annotation_completion_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report["complete"]:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "report": str(report_path),
                    "labelled_pairs": report["labelled_pairs"],
                    "cohen_kappa": report["cohen_kappa"],
                    "unresolved_count": report["unresolved_count"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(2)
    output = dataset_dir / "qrels_gold.jsonl"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for qrel in gold_qrels:
            handle.write(json.dumps(qrel, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "qrels": str(output), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
