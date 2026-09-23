"""Conservative retrieval sufficiency checks, not guarantees of factual entailment."""
from __future__ import annotations

import json
import math
from pathlib import Path


def check_evidence(ingestion, job_id: str, query: str, policy_path: Path | None) -> dict:
    if policy_path is None:
        return {"status": "not_calibrated", "abstained": True, "documents": [],
                "message": "尚未配置适用于当前模型的证据充分性阈值，不能从相似片段推断已有答案。"}
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        threshold = float(policy["threshold"])
        if not math.isfinite(threshold) or not -1 <= threshold <= 1.000001:
            raise ValueError("Invalid threshold")
    except (OSError, ValueError, KeyError, TypeError):
        return {"status": "invalid_calibration", "abstained": True, "documents": []}
    if policy.get("production_approved") is not True:
        return {"status": "calibration_not_approved", "abstained": True, "documents": []}
    current = {k: v for k, v in ingestion.expected_manifest.items() if k != "collection"}
    if policy.get("embedding_identity") != current or policy.get("fit_split") != "development":
        return {"status": "calibration_mismatch", "abstained": True, "documents": []}
    pairs = ingestion.scored_source_evidence(job_id, query, k=3)
    score = pairs[0][1] if pairs else -1.0
    abstained = not pairs or score < threshold
    return {"status": "insufficient_evidence" if abstained else "candidate_evidence", "abstained": abstained,
            "score": score, "threshold": threshold, "policy_version": policy.get("version", "unknown"),
            "documents": [] if abstained else [{"chunk_id": d.metadata["chunk_id"], "job_id": d.metadata["job_id"],
                "content": d.page_content, "similarity": s} for d, s in pairs],
            "message": "相似度只能筛选候选证据；须核对否定、条件和完整支持关系，不能把候选片段当成已验证答案。"}
