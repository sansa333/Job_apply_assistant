import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import settings
from app.schemas import EvidenceLevel, OneClickApplyRequest
from app.services.application_service import ApplicationService


class ApplicationPackageStatusTests(unittest.TestCase):
    def test_package_is_generated_when_contact_details_are_pending(self) -> None:
        service = object.__new__(ApplicationService)
        service.analyze_scoped_fit = lambda req: {
            "status": "needs_candidate_evidence",
            "fit_report": "候选人证据待补充的通用岗位准备建议",
            "job_evidence": [],
            "candidate_evidence": [],
            "evidence_level": EvidenceLevel.MISSING,
        }
        service._run_prompt_with_usage = lambda template, variables: ("待确认草稿", None)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(settings, "outputs_dir", root / "outputs"),
                patch.object(settings, "data_dir", root / "data"),
            ):
                result = service.one_click_apply(
                    OneClickApplyRequest(
                        company_name="Acme",
                        job_title="RAG Engineer",
                        jd_text="Need RAG experience",
                    )
                )

        self.assertEqual(result["status"], "generated_not_submitted")
        self.assertEqual(result["contact_status"], "pending_confirmation")
        self.assertEqual(result["candidate_email"], "待确认")
        self.assertEqual(result["next_action"], "confirm_contact_details_before_sending")


if __name__ == "__main__":
    unittest.main()
