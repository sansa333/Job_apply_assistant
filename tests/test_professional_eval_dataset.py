import json
import unittest
from app.config import settings
from app.evaluation.professional_eval import evaluate_dataset


class ProfessionalEvalDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset_dir = settings.data_dir / "eval_dataset" / "job_agent_v1"

    def test_dataset_uses_concrete_public_jobs_and_explicit_synthetic_candidates(self) -> None:
        manifest = json.loads((self.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        jobs = [json.loads(line) for line in (self.dataset_dir / "job_snapshots.jsonl").read_text(encoding="utf-8").splitlines()]
        candidates = [json.loads(line) for line in (self.dataset_dir / "candidate_profiles.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertEqual(manifest["counts"]["public_historical_job_snapshots"], 8)
        self.assertIn("Arm", {job["company_name"] for job in jobs})
        self.assertIn("British Airways", {job["company_name"] for job in jobs})
        self.assertTrue(all(job["source_url"] and job["content_sha256"] for job in jobs))
        self.assertTrue(all(candidate["synthetic"] is True for candidate in candidates))
        self.assertFalse(any("某某" in job["company_name"] or "某某" in job["job_title"] for job in jobs))

    def test_offline_evaluation_has_all_four_layers(self) -> None:
        report = evaluate_dataset(self.dataset_dir)

        self.assertEqual(report["extraction"]["job_count"], 8)
        self.assertEqual(report["matching"]["pair_count"], 24)
        self.assertEqual(report["agent_trajectory"]["case_count"], 10)
        self.assertEqual(report["generation_validation"]["case_count"], 8)
        self.assertIn("calibrated_holdout", report["matching"])


if __name__ == "__main__":
    unittest.main()
