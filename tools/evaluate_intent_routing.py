"""Offline admission regression report; supplied model outputs, no network or domain tools.

This measures policy behavior, NOT real-model accuracy or calibrated confidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.intent_router import recognize_intent
from app.agent.job_agent import _apply_recognized_intent
from app.agent.routing_policy import admission_response
from app.schemas import AgentRequest


def evaluate(path: Path) -> dict:
    rows = []
    for case in json.loads(path.read_text(encoding="utf-8")):
        model = Mock()
        if "model_output" in case:
            output = case["model_output"]
            model.invoke.return_value = SimpleNamespace(content=output if isinstance(output, str) else json.dumps(output))
        else:
            model.invoke.side_effect = RuntimeError("offline fixture")
        request = AgentRequest.model_validate(case["request"])
        with patch("app.agent.intent_router.get_llm", return_value=model):
            recognition = recognize_intent(request)
        routed = _apply_recognized_intent(request, recognition)
        response = admission_response(routed, recognition)
        actual = response["status"] if response else "execute:" + routed.intent.value
        rows.append({"id": case["id"], "expected": case["expected"], "actual": actual,
                     "passed": actual == case["expected"], "source": recognition.source,
                     "model_attempts": recognition.model_attempts})
    return {
        "evaluation_kind": "offline_policy_regression_with_stubbed_model",
        "real_model_accuracy": None, "confidence_calibrated": False,
        "case_count": len(rows), "passed": sum(row["passed"] for row in rows),
        "unsafe_admissions": sum(row["actual"].startswith("execute:") and row["actual"] != row["expected"] for row in rows),
        "unnecessary_blocks": sum(not row["actual"].startswith("execute:") and row["expected"].startswith("execute:") for row in rows),
        "cases": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path(__file__).resolve().parents[1] / "tests/fixtures/intent_routing_cases.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.cases)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["passed"] == report["case_count"] else 1)
