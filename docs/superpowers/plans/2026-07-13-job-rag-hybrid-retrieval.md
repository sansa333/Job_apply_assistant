# Job RAG Hybrid Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship natural-query, job-scoped hybrid retrieval with Cross-Encoder reranking and select a production embedding model using a reproducible benchmark.

**Architecture:** `JobKnowledgeIngestion` will expose vector candidates while a new hybrid retriever adds in-job BM25, RRF fusion, and the existing Cross-Encoder as a final ranker. The evaluator will consume a retriever-compatible object and write isolated, model-specific indexes and reports before production is rebuilt with the selected model.

**Tech Stack:** Python 3, Chroma, LangChain, sentence-transformers, Hugging Face models, unittest.

---

## File map

- Modify `app/config.py` for the production model and hybrid candidate/RRF settings.
- Modify `app/embeddings.py` to instantiate named Hash or Hugging Face models for isolated experiments.
- Modify `app/knowledge/ingestion.py` to accept a collection name and embedding implementation and expose vector candidates.
- Create `app/knowledge/hybrid.py` for deterministic BM25, RRF fusion, and Cross-Encoder job retrieval.
- Modify `app/knowledge/retrieval.py` and `app/services/application_service.py` to make `hybrid_rerank` the job retrieval path.
- Modify `app/knowledge/evaluation.py` for natural queries and generic retriever evaluation metadata.
- Create `tools/compare_job_retrieval_models.py` for isolated indexing, model/strategy experiments, selection, and reports.
- Modify `tools/build_job_rag_eval.py`, `.env.example`, `docs/knowledge-base-operation.md`, and `README.md` for the selected production configuration and reproducible commands.
- Modify `tests/test_job_rag_evaluation.py`, `tests/test_job_scoped_retrieval.py`, and create `tests/test_job_hybrid_retrieval.py`.

### Task 1: Define natural query and hybrid ranking behavior

**Files:**
- Modify: `tests/test_job_rag_evaluation.py`
- Create: `tests/test_job_hybrid_retrieval.py`
- Modify: `app/knowledge/evaluation.py`
- Create: `app/knowledge/hybrid.py`

- [ ] **Step 1: Write failing tests**

```python
samples = build_job_eval_samples(catalog, ingestion, limit=12)
assert all("请重点说明" not in item["query"] for item in samples)
assert all(not set(item["anchor_terms"]).intersection(item["query"].split()) for item in samples)

docs = retriever.retrieve_for_job(target_job_id, "这个岗位需要哪些 Python 技能？", k=2)
assert docs[0].metadata["chunk_id"] == "job:target:skills"
assert all(doc.metadata["job_id"] == target_job_id for doc in docs)
```

- [ ] **Step 2: Verify RED**

Run: `& .\\.venv\\Scripts\\python.exe -m unittest tests.test_job_rag_evaluation tests.test_job_hybrid_retrieval -v`

Expected: natural-query assertion fails and the hybrid module is unavailable.

- [ ] **Step 3: Implement minimal natural-query and hybrid primitives**

```python
class JobHybridRetriever:
    def retrieve_for_job(self, job_id: str, query: str, *, k: int) -> list[Document]:
        vector = self.ingestion.retrieve_vector_for_job(job_id, query, k=self.candidate_k)
        lexical = self._bm25_rank(job_id, query, k=self.candidate_k)
        fused = reciprocal_rank_fusion(vector, lexical, rrf_k=self.rrf_k)
        return self.reranker.rerank(query, fused[:self.candidate_k], top_n=k).docs[:k]
```

`build_job_eval_samples` must keep `anchor_terms` and `expected_keywords` as labels but render template-only natural Chinese question text.

- [ ] **Step 4: Verify GREEN**

Run the focused test command again; expected: all tests pass.

### Task 2: Integrate job-scoped hybrid reranking

**Files:**
- Modify: `app/config.py`
- Modify: `app/knowledge/ingestion.py`
- Modify: `app/knowledge/retrieval.py`
- Modify: `app/services/application_service.py`
- Modify: `tests/test_job_scoped_retrieval.py`

- [ ] **Step 1: Write a failing integration test**

```python
resolution = JobScopedRetriever(catalog=catalog, job_ingestion=ingestion).resolve(
    "Acme", "RAG Engineer", "这个岗位有哪些技术要求？", k=3
)
assert resolution.retrieval_strategy == "hybrid_rerank"
assert resolution.reranker_applied is True
assert all(doc.metadata["job_id"] == target.record.job_id for doc in resolution.job_documents)
```

- [ ] **Step 2: Verify RED**

Run: `& .\\.venv\\Scripts\\python.exe -m unittest tests.test_job_scoped_retrieval -v`

Expected: `JobResolution` has no retrieval strategy/reranker fields.

- [ ] **Step 3: Implement production integration**

Add settings `job_retrieval_strategy`, `job_retrieval_candidate_k`, and `job_retrieval_rrf_k`. Inject the selected `JobHybridRetriever` into `JobScopedRetriever`, expose strategy/reranker state in `JobResolution`, and include retrieval metadata in the fit response.

- [ ] **Step 4: Verify GREEN**

Run the focused retrieval and fit tests; expected: all pass with `job_id` isolation retained.

### Task 3: Make embedding experiments isolated and reproducible

**Files:**
- Modify: `app/embeddings.py`
- Modify: `app/knowledge/ingestion.py`
- Modify: `app/knowledge/evaluation.py`
- Create: `tools/compare_job_retrieval_models.py`
- Create: `tests/test_job_embedding_experiment.py`

- [ ] **Step 1: Write a failing experiment test**

```python
result = run_model_experiment(spec, samples, catalog, root / "experiments")
assert result["collection"] != "job_knowledge"
assert result["strategies"] == {"vector", "hybrid", "hybrid_rerank"}
assert "mrr_at_3" in result["strategies"]["hybrid_rerank"]["metrics"]
```

- [ ] **Step 2: Verify RED**

Run: `& .\\.venv\\Scripts\\python.exe -m unittest tests.test_job_embedding_experiment -v`

Expected: experiment module and structured result are unavailable.

- [ ] **Step 3: Implement the experiment runner**

For each of `hash`, `BAAI/bge-small-zh-v1.5`, `intfloat/multilingual-e5-small`, and `BAAI/bge-m3`, build an independently named collection under `data/eval_dataset/job_rag/model_experiments/`, execute all three strategies, and persist model, collection, latency, reranker state, metrics, and failures. Select by `(MRR@3, HitRate@1, -latency)` from available `hybrid_rerank` rows.

- [ ] **Step 4: Verify GREEN**

Run the focused experiment test; expected: mock/hash experiment writes no production collection.

### Task 4: Execute model comparison and apply the winner

**Files:**
- Modify: `.env.example`
- Modify: `tools/build_job_rag_eval.py`
- Modify: `README.md`
- Modify: `docs/knowledge-base-operation.md`
- Generate: `data/eval_dataset/job_rag/model_experiments/{report.json,report.md}`
- Regenerate: `data/eval_dataset/job_rag/{real_job_retrieval_eval.jsonl,report.json,report.md}`

- [ ] **Step 1: Build natural query dataset**

Run: `& .\\.venv\\Scripts\\python.exe -m tools.build_job_rag_eval`

Expected: 80 natural Chinese questions with no “请重点说明”.

- [ ] **Step 2: Run model comparison**

Run: `& .\\.venv\\Scripts\\python.exe -m tools.compare_job_retrieval_models`

Expected: JSON and Markdown report lists every requested model, unavailable failures if any, three strategies per available model, and an explicit winner.

- [ ] **Step 3: Apply winner and rebuild production index**

Set `EMBEDDING_BACKEND=huggingface`, selected `HF_EMBEDDING_MODEL`, and `JOB_RETRIEVAL_STRATEGY=hybrid_rerank`; then run `& .\\.venv\\Scripts\\python.exe -m tools.rebuild_job_knowledge` and rebuild the final report.

- [ ] **Step 4: Verify full deliverable**

Run: `& .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v`

Expected: all tests pass; report has a non-empty natural query dataset, applied reranker evidence, selected model rationale, and metrics in [0, 1].
