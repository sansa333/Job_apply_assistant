# Diverse Job RAG Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement each task with its tests.

**Goal:** Replace single-intent job retrieval evaluation queries with a balanced, labelled set covering the distinct information needs present in real job descriptions.

**Architecture:** The evaluator remains job-scoped, so exact catalogue resolution is intentionally out of scope. It classifies each already-indexed chunk into auditable question intents, generates a natural Chinese question with content anchors, and aggregates retrieval results both overall and by intent.

**Tech Stack:** Python 3, unittest, LangChain/Chroma, JSONL, Markdown.

---

## File map

- `tests/test_job_rag_evaluation.py`: test sample diversity and type-level metrics.
- `app/knowledge/evaluation.py`: classify chunks, generate labelled samples, and aggregate metrics.
- `tools/build_job_rag_eval.py`: build 80 samples and render a type-aware report.
- `data/eval_dataset/job_rag/`: regenerated JSONL, JSON, and Markdown artifacts.

### Task 1: Specify diverse, labelled sample generation

- [x] Add a test that ingests a rich job description with duties, technical skills, qualifications, remote/location details, and benefits, then asserts generated samples contain multiple `question_type` values and target labels.
- [x] Run the test before implementation and confirm it fails because no question-type labels exist.
- [x] Implement deterministic, evidence-backed question templates for overview, responsibilities, technical skills, qualifications, experience/education, location/work mode, and benefits; select candidates round-robin for balance.
- [x] Re-run the focused test and confirm it passes.

### Task 2: Report per-type retrieval metrics

- [x] Assert that the report exposes `question_type_distribution` and `metrics_by_question_type`, with valid metric ranges.
- [x] Implement aggregation from existing retrieval details so every type receives HitRate@k, MRR@k, and keyword recall without a second retrieval pass.
- [x] Re-run the focused tests and confirm they pass.

### Task 3: Regenerate and verify the evaluation artifacts

- [x] Set the build tool’s default sample limit to 80 and render overall metrics, question-type counts, type-level metric rows, and Top-1 misses in Markdown.
- [x] Run `& .\\.venv\\Scripts\\python.exe -m tools.build_job_rag_eval` against the existing isolated `job_knowledge` index.
- [x] Run `& .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v` and verify the JSONL is non-empty, includes multiple question types, and reports all metric values in the interval [0, 1].
