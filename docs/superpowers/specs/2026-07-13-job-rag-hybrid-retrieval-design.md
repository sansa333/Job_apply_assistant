# Job RAG Hybrid Retrieval Design

## Objective

Improve job-scoped retrieval ranking by combining lexical and semantic candidates, applying a Cross-Encoder reranker, and selecting an embedding model through a reproducible experiment. The primary evaluation dataset must contain natural user questions only; source-derived keywords remain hidden labels for evaluation and must not appear in the query.

## Current-state findings

- The production job collection uses Chroma vector search only and the active environment uses the offline Hash embedding fallback.
- A Cross-Encoder implementation exists for multimodal retrieval but is not used by the job-scoped retriever.
- The job evaluation generator currently appends source terms through “请重点说明 …”, which leaks target evidence into the query.
- The local cache already contains `BAAI/bge-small-zh-v1.5` and `BAAI/bge-reranker-v2-m3`; sufficient disk space is available for the additional experiment models.

## Retrieval design

1. Resolve the exact job record as today; no result may cross the selected `job_id` boundary.
2. Build two candidate rankings within that job:
   - Chroma semantic vector ranking;
   - in-memory BM25 ranking over that job’s indexed chunks.
3. Fuse the rankings with Reciprocal Rank Fusion (RRF), retaining the configured candidate count.
4. Rerank fused candidates with `BAAI/bge-reranker-v2-m3`, return the configured final `k`, and report whether the reranker was actually applied. If it is unavailable, retain RRF order and expose the reason.
5. Keep the strategy configurable (`vector`, `hybrid`, `hybrid_rerank`), defaulting production job retrieval to `hybrid_rerank` after verification.

## Natural-query evaluation design

- Retain the existing labels: `job_id`, expected `chunk_id`, question type, source, and hidden expected keywords.
- Generate only natural Chinese questions such as “这个岗位需要哪些技术技能？” or “该岗位有哪些任职资格？”. The company and job title remain metadata because the upstream API resolves them separately; they are not injected into the question text.
- Continue balanced sampling across overview, responsibilities, technical skills, qualifications, experience/education, location/work mode, and benefits.
- Include the retrieval strategy, model name, candidate count, reranker state, latency, and per-type metrics in every experiment report.

## Embedding comparison

Each model is evaluated against a fresh, isolated Chroma index populated from the same job catalog and the same 80 natural-query samples:

| Role | Model |
| --- | --- |
| Offline baseline | built-in `HashEmbeddings` |
| Chinese baseline | `BAAI/bge-small-zh-v1.5` |
| Lightweight multilingual candidate | `intfloat/multilingual-e5-small` |
| Multilingual production candidate | `BAAI/bge-m3` |

For every semantic model, record vector-only, hybrid, and hybrid-plus-rerank metrics. The selected model is the best available model by `MRR@3`, then `HitRate@1`, with total evaluation latency used as a tiebreaker. The report must state failed/unavailable models explicitly rather than treating them as results.

## Safety and isolation

- Experiment collections live outside the production `job_knowledge` collection and do not write to `eval_demo`.
- The production collection is rebuilt only once with the selected model.
- Model paths and experiment configurations are recorded without API credentials.

## Verification

- Unit tests prove natural-query generation contains no anchor phrase or target keywords, hybrid fusion respects `job_id`, and a real/fake Cross-Encoder changes the final order only after candidate retrieval.
- The experiment report must include at least the configured model rows, strategy rows, metrics, latency, and model-selection rationale.
- Full test suite passes after the selected production configuration is applied.
