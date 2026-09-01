# Real Job RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement each task with its tests.

**Goal:** Replace the unscoped JD/profile RAG path with a source-labelled, exact job-scoped knowledge base, and provide an isolated, reproducible retrieval evaluation report.

**Architecture:** SQLite is the authoritative job catalogue. `job_knowledge` and `candidate_profile` are derived Chroma collections; `eval_demo` is a separate evaluation-only collection. Every matching request resolves an exact catalogue record before retrieving chunks filtered by `job_id` and `candidate_id`.

**Tech Stack:** Python 3, FastAPI, Pydantic, SQLite, LangChain/Chroma, Streamlit, unittest.

---

## File map

- Create `app/knowledge/{models,normalize,catalog,importers,ingestion,retrieval,health}.py` for ownership of job data and retrieval.
- Create `app/routes/knowledge.py` for catalog/import/upload/rebuild/health APIs.
- Modify `app/config.py`, `app/rag.py`, `app/schemas.py`, `app/main.py`, `app/services/application_service.py`, `app/agent/tools.py`, `app/multimodal/{service,routes,reranker}.py`, and `streamlit_app.py` for integration and isolation.
- Create `tools/{import_open_source_jobs,rebuild_job_knowledge,build_job_rag_eval}.py` plus a checked-in generated evaluation dataset and report under `data/eval_dataset/job_rag/`.
- Create focused unit and API tests under `tests/test_{eval_isolation,job_catalog,open_source_import,user_job_upload,job_scoped_fit,job_index_health,job_rag_evaluation}.py`.

### Task 1: Enforce data-domain isolation

- [ ] Write failing tests proving normal job/profile retrieval cannot read `eval_demo`, and evaluation ingestion does not write the production multimodal collection.
- [ ] Add collection names and source-corpus paths to settings; route evaluation writes to a per-run `eval_demo` collection.
- [ ] Run `python -m unittest tests.test_eval_isolation -v` and the existing multimodal tests.

### Task 2: Implement authoritative job catalogue

- [ ] Test company/title normalization, content hashing, duplicate suppression, and user-upload precedence.
- [ ] Implement `NormalizedJob`, normalized keys, SQLite schema/migration, `upsert`, `lookup`, and deterministic `job_id` generation.
- [ ] Run `python -m unittest tests.test_job_catalog -v`.

### Task 3: Import only approved job sources

- [ ] Test CSV validation/reporting and Markdown real-JD metadata parsing; reject synthetic evaluation source paths.
- [ ] Implement the CSV, project Markdown, and explicit user-upload adapters; preserve source files under `source_corpus`.
- [ ] Implement job chunk creation with required metadata and stable Chroma ids.
- [ ] Run `python -m unittest tests.test_open_source_import -v`.

### Task 4: Add job management APIs and rebuild support

- [ ] Test missing company/title validation, upload persistence, exact search, detail, import, and idempotent rebuild behavior.
- [ ] Implement `/api/jobs/import/open-source`, `/api/jobs/upload`, `/api/jobs/search`, `/api/jobs/{job_id}`, and `/api/jobs/rebuild`.
- [ ] Run `python -m unittest tests.test_user_job_upload -v`.

### Task 5: Gate match generation by exact job resolution

- [ ] Test `job_not_found` is returned without an LLM call, selected job retrieval never leaks another job, and both job/profile evidence are included.
- [ ] Replace free-text `FitRequest` with candidate/company/title plus optional question; preserve legacy `jd_text` only as an upload-on-request compatibility path.
- [ ] Add candidate-scoped profile ingestion/retrieval and job-scoped retrieval; update agent tools and one-click generation to use the gate.
- [ ] Run `python -m unittest tests.test_job_scoped_fit -v`.

### Task 6: Make index health and reranker state observable

- [ ] Test a mismatched collection manifest blocks retrieval and a missing cross-encoder returns vector order with `rerank_applied=false` and a reason.
- [ ] Write collection manifests when indexing, implement `/api/knowledge/health`, and add a source-driven rebuild tool.
- [ ] Run `python -m unittest tests.test_job_index_health -v`.

### Task 7: Update the Streamlit workflow and documentation

- [ ] Add source-labelled Job Library, upload form, exact-match workspace, no-match upload guidance, and isolated evaluation view.
- [ ] Update README and write `docs/knowledge-base-operation.md` with import/rebuild/operational procedures and historical-data wording.
- [ ] Smoke-import the FastAPI app and compile edited Python modules.

### Task 8: Generate and execute a retrieval evaluation

- [ ] Build at least 20 labelled job-scoped queries from the imported open-source/project-real JDs; each label contains a target `job_id`, expected source, and terms.
- [ ] Run evaluation against a clean `job_knowledge` index, calculate HitRate@1/@3/@5, MRR@5, and keyword recall, and save JSON/Markdown reports with per-query rankings and bad cases.
- [ ] Run the full suite with `python -m unittest discover -s tests -v` and review the generated report for a non-empty dataset and metrics in [0, 1].

## Review checklist

- `job_knowledge` contains only `open_source` and `user_upload` source kinds; no `synth_*` or `eval_demo` inputs.
- Exact company+title miss returns the specified upload guidance and never invokes matching generation.
- User uploaded records are selected ahead of historical public records for the same normalized pair.
- Chunk metadata records job/profile scope, identifiers, source kind/dataset/file, and section.
- Evaluation is isolated from production collections and reports the requested retrieval metrics.
