import unittest

from app.domain.job_application import EvidenceSupport
from app.services.structured_analysis import (
    align_evidence,
    parse_candidate_profile,
    parse_job_description,
    score_evidence,
    validate_grounded_text,
)


class StructuredAnalysisTests(unittest.TestCase):
    def test_parses_concrete_arm_requirements_and_aligns_evidence(self) -> None:
        job = parse_job_description(
            company_name="Arm",
            job_title="Full Stack Data Scientist",
            description="""
Responsibilities
- Implement machine learning models into production systems.
Required Skills and Experience
- Strong proficiency in Python and React.
- Bachelor's or Master's degree in Computer Science or Data Science.
Nice to Have
- Familiarity with AWS and Docker.
""",
            location="Cambridge, United Kingdom",
            language="en",
        )
        candidate = parse_candidate_profile(
            candidate_id="syn_maya_patel_ml_platform",
            source_kind="synthetic_eval",
            sources=[
                {
                    "filename": "maya_patel.md",
                    "section": "experience",
                    "content": "MSc Data Science. Built Python and React ML products on AWS with Docker.",
                }
            ],
        )
        matrix = align_evidence(job, candidate)
        score = score_evidence(job, matrix)

        self.assertGreaterEqual(len(job.requirements), 3)
        self.assertTrue(any("python" in requirement.normalized_terms for requirement in job.requirements))
        self.assertTrue(any(item.support == EvidenceSupport.DIRECT for item in matrix))
        self.assertGreater(score.overall_score, 0)
        self.assertEqual(score.calibration_status, "uncalibrated_baseline")

    def test_validator_detects_invented_metric_and_false_submission(self) -> None:
        findings = validate_grounded_text(
            "Maya increased revenue by 73%. 已完成投递。",
            ["Maya reduced inference P95 latency by 31%."],
        )
        codes = {finding.code for finding in findings}

        self.assertEqual(codes, {"unsupported_quantified_claim", "false_submission_claim"})


if __name__ == "__main__":
    unittest.main()
