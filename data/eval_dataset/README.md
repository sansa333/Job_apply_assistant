# Evaluation Dataset for AI Job Apply Assistant

This directory contains a moderate-size curated dataset for the local project:

- Resume-JD matching evaluation
- RAG retrieval and rerank evaluation
- Application-material generation tests
- Synthetic Chinese multimodal job-application checks

## Contents

- `resumes/`: synthetic Markdown resumes.
- `jds/`: synthetic Markdown job descriptions.
- `pairs/synthetic_resume_jd_pairs.jsonl`: synthetic resume-JD pairs with source scores.
- `rag_queries/zh_retrieval_eval.jsonl`: Chinese resume-JD retrieval evaluation samples.
- `multimodal_zh/`: job-application-specific Chinese multimodal samples, including resume screenshots, JD posters, match dashboards, and interview feedback cards.
- `dataset_manifest.json`: counts, schema notes, and source links.

## Counts

- Synthetic resumes: 40
- Synthetic job descriptions: 40
- Synthetic resume-JD pairs: 40
- Chinese resumes: 8
- Chinese job descriptions: 6
- Chinese resume-JD pairs: 24
- Chinese retrieval evaluation queries: 24
- Chinese job-application multimodal samples: 8

## Suggested Use

1. Ingest `resumes/*.md` into the profile knowledge base.
2. Ingest `jds/*.md` into the job-description knowledge base.
3. Use `rag_queries/zh_retrieval_eval.jsonl` for `HitRate@K`, `MRR@K`, and `KeywordRecall@K`.
4. Use `pairs/*.jsonl` for resume-JD scoring and ranking experiments.
5. Use `multimodal_zh/zh_mrag_eval.jsonl` for image understanding and citation checks.
6. Prefer `zh_retrieval`, `zh_multimodal`, or `zh_all` in the Streamlit report page for Chinese job-application demos.

## Why Chinese Job-Application Images

The versioned `multimodal_zh/` subset is fully synthetic and aligned with resumes, JD posters, interview feedback, match matrices, RAG architecture, and generated application materials so the demo can show: image -> VLM OCR/semantic extraction -> Chroma unified retrieval -> RAG answer/evaluation.

## Source Notes

This dataset is for local development, demos, and educational evaluation. Keep the source license notes in `dataset_manifest.json` when reusing or redistributing.
