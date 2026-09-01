from __future__ import annotations

import ast
import csv
import json
import re
import shutil
from pathlib import Path

import docx2txt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval_dataset"
SRC = OUT / "source_cache"

REAL_SRC = SRC / "vacancy-resume-matching-dataset"
HF_SRC = SRC / "hf" / "data"

CURATED_DIRS = [
    OUT / "resumes",
    OUT / "jds",
    OUT / "pairs",
    OUT / "rag_queries",
    OUT / "multimodal" / "images",
    OUT / "multimodal" / "text",
]

TECH_KEYWORDS = [
    "Python",
    "Java",
    "C#",
    "C++",
    "JavaScript",
    "TypeScript",
    "SQL",
    "FastAPI",
    "Django",
    "Flask",
    "Spring",
    "React",
    "Vue",
    "Angular",
    "AWS",
    "Docker",
    "Linux",
    "Unix",
    "REST",
    "API",
    "PostgreSQL",
    "MySQL",
    "Oracle",
    "Redis",
    "ElasticSearch",
    "RAG",
    "LLM",
    "LangChain",
    "Chroma",
]


def reset_curated_dirs() -> None:
    for directory in CURATED_DIRS:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def slugify(value: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value[:60] or fallback


def compact_text(value: str, limit: int | None = None) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    return value[:limit].strip() if limit else value


def extract_keywords(*texts: str, limit: int = 8) -> list[str]:
    joined = "\n".join(texts)
    hits: list[str] = []
    for kw in TECH_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", joined, flags=re.IGNORECASE):
            hits.append(kw)
    return hits[:limit]


def markdown_doc(title: str, metadata: dict, body: str) -> str:
    meta = "\n".join(f"- {k}: {v}" for k, v in metadata.items())
    return f"# {title}\n\n## Metadata\n{meta}\n\n## Content\n{body}"


def parse_rankings(path: Path) -> tuple[list[list[int]], list[list[int]]]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"#.*", "", text)

    def grab(name: str) -> list[list[int]]:
        match = re.search(rf"{name}\s*=\s*(\[\[.*?\]\])", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Cannot find {name} in {path}")
        return ast.literal_eval(match.group(1))

    return grab("ANNOTATOR_1_RANKINGS"), grab("ANNOTATOR_2_RANKINGS")


def build_real_resume_jd_dataset() -> tuple[list[dict], list[dict], list[dict]]:
    vacancies_path = REAL_SRC / "5_vacancies.csv"
    annotations_path = REAL_SRC / "annotations-for-the-first-30-vacancies.txt"
    cv_dir = REAL_SRC / "CV"

    if not vacancies_path.exists() or not cv_dir.exists():
        raise FileNotFoundError("Missing vacancy-resume-matching-dataset source files.")

    with vacancies_path.open("r", encoding="utf-8", newline="") as f:
        vacancies = list(csv.DictReader(f))

    jd_rows: list[dict] = []
    for idx, row in enumerate(vacancies, start=1):
        jd_id = f"real_en_jd_{idx:02d}"
        title = compact_text(row["job_title"])
        filename = f"{jd_id}_{slugify(title, 'job')}.md"
        content = markdown_doc(
            title=title,
            metadata={
                "id": jd_id,
                "source": "NataliaVanetik/vacancy-resume-matching-dataset",
                "source_uid": row.get("uid", ""),
                "language": "en",
                "type": "job_description",
            },
            body=compact_text(row["job_description"]),
        )
        write_text(OUT / "jds" / filename, content)
        jd_rows.append(
            {
                "jd_id": jd_id,
                "title": title,
                "filename": filename,
                "text": compact_text(row["job_description"]),
            }
        )

    resume_rows: list[dict] = []
    for cv_no in range(1, 31):
        resume_id = f"real_en_resume_{cv_no:03d}"
        source_path = cv_dir / f"{cv_no}.docx"
        text = compact_text(docx2txt.process(str(source_path)))
        filename = f"{resume_id}.md"
        content = markdown_doc(
            title=f"Anonymous Resume {cv_no}",
            metadata={
                "id": resume_id,
                "source": "NataliaVanetik/vacancy-resume-matching-dataset",
                "source_file": f"CV/{cv_no}.docx",
                "language": "en",
                "type": "resume",
            },
            body=text,
        )
        write_text(OUT / "resumes" / filename, content)
        resume_rows.append({"resume_id": resume_id, "filename": filename, "text": text})

    annotator_1, annotator_2 = parse_rankings(annotations_path)
    pair_rows: list[dict] = []
    query_rows: list[dict] = []

    for cv_idx, resume in enumerate(resume_rows):
        for jd_idx, jd in enumerate(jd_rows):
            rank_1 = annotator_1[cv_idx][jd_idx]
            rank_2 = annotator_2[cv_idx][jd_idx]
            avg_rank = round((rank_1 + rank_2) / 2.0, 2)
            score = round((6.0 - avg_rank) / 5.0 * 100, 1)
            label = "high" if avg_rank <= 2.0 else "medium" if avg_rank <= 3.5 else "low"
            keywords = extract_keywords(jd["title"], jd["text"], resume["text"])
            pair_id = f"real_pair_cv{cv_idx + 1:03d}_jd{jd_idx + 1:02d}"
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "source": "NataliaVanetik/vacancy-resume-matching-dataset",
                    "resume_id": resume["resume_id"],
                    "resume_file": f"resumes/{resume['filename']}",
                    "jd_id": jd["jd_id"],
                    "jd_file": f"jds/{jd['filename']}",
                    "job_title": jd["title"],
                    "annotator_1_rank": rank_1,
                    "annotator_2_rank": rank_2,
                    "avg_rank": avg_rank,
                    "match_score": score,
                    "relevance_label": label,
                    "expected_keywords": keywords,
                }
            )

            if label in {"high", "medium"} and len(query_rows) < 80:
                query_rows.append(
                    {
                        "query_id": f"rq_{len(query_rows) + 1:04d}",
                        "source_pair_id": pair_id,
                        "scenario": "resume_job_matching",
                        "query": f"候选人是否适合 {jd['title']} 岗位？请结合简历和JD给出证据。",
                        "expected_sources": [resume["filename"], jd["filename"]],
                        "expected_keywords": keywords,
                        "relevance_label": label,
                    }
                )

    return jd_rows, resume_rows, pair_rows + query_rows


def build_synthetic_hf_subset() -> tuple[list[dict], list[dict], list[dict]]:
    path = HF_SRC / "validation-00000-of-00001.parquet"
    if not path.exists():
        return [], [], []

    df = pd.read_parquet(path).head(40)
    jd_rows: list[dict] = []
    resume_rows: list[dict] = []
    pair_rows: list[dict] = []

    for idx, row in df.iterrows():
        n = idx + 1
        jd_id = f"synth_kr_jd_{n:03d}"
        resume_id = f"synth_kr_resume_{n:03d}"
        jd_file = f"{jd_id}.md"
        resume_file = f"{resume_id}.md"
        jobpost = compact_text(row["jobpost"])
        resume = compact_text(row["resume"])
        selfintro = compact_text(row.get("selfintro", ""))
        evaluation = compact_text(row.get("evaluation", ""))
        total_score = float(row.get("total_score", 0))
        label = "high" if total_score >= 80 else "medium" if total_score >= 50 else "low"

        write_text(
            OUT / "jds" / jd_file,
            markdown_doc(
                title=f"Synthetic Korean Job {n}",
                metadata={
                    "id": jd_id,
                    "source": "Divyanandh/resume-matching-dataset-v2",
                    "language": "ko",
                    "type": "job_description",
                    "note": "synthetic data generated by GPT-4o-mini in the original dataset",
                },
                body=jobpost,
            ),
        )
        write_text(
            OUT / "resumes" / resume_file,
            markdown_doc(
                title=f"Synthetic Korean Resume {n}",
                metadata={
                    "id": resume_id,
                    "source": "Divyanandh/resume-matching-dataset-v2",
                    "language": "ko",
                    "type": "resume",
                    "note": "synthetic data generated by GPT-4o-mini in the original dataset",
                },
                body=f"{resume}\n\n## Self Introduction\n{selfintro}",
            ),
        )

        jd_rows.append({"jd_id": jd_id, "filename": jd_file})
        resume_rows.append({"resume_id": resume_id, "filename": resume_file})
        pair_rows.append(
            {
                "pair_id": f"synth_pair_{n:03d}",
                "source": "Divyanandh/resume-matching-dataset-v2",
                "resume_id": resume_id,
                "resume_file": f"resumes/{resume_file}",
                "jd_id": jd_id,
                "jd_file": f"jds/{jd_file}",
                "match_score": total_score,
                "resume_score": float(row.get("resume_score", 0)),
                "selfintro_score": float(row.get("selfintro_score", 0)),
                "relevance_label": label,
                "expected_keywords": extract_keywords(jobpost, resume),
                "evaluation_summary": evaluation[:500],
                "synthetic": True,
            }
        )

    return jd_rows, resume_rows, pair_rows


def image_bytes(value: object) -> bytes | None:
    if isinstance(value, dict):
        data = value.get("bytes")
        if isinstance(data, bytes):
            return data
    return None


def build_multimodal_subset() -> list[dict]:
    path = HF_SRC / "test-00000-of-000028.parquet"
    if not path.exists():
        return []

    df = pd.read_parquet(path).head(12)
    rows: list[dict] = []

    for idx, row in df.iterrows():
        sample_id = f"mrag_{int(row['id']):04d}"
        image_file = f"{sample_id}_query.jpg"
        data = image_bytes(row["image"])
        if data:
            (OUT / "multimodal" / "images" / image_file).write_bytes(data)

        question = compact_text(row["question"])
        choices = {
            "A": compact_text(row["A"]),
            "B": compact_text(row["B"]),
            "C": compact_text(row["C"]),
            "D": compact_text(row["D"]),
        }
        answer_choice = compact_text(row["answer_choice"])
        answer = compact_text(row["answer"])

        text_file = f"{sample_id}_qa.md"
        write_text(
            OUT / "multimodal" / "text" / text_file,
            markdown_doc(
                title=f"MRAG-Bench Sample {sample_id}",
                metadata={
                    "id": sample_id,
                    "source": "uclanlp/MRAG-Bench",
                    "scenario": compact_text(row["scenario"]),
                    "aspect": compact_text(row["aspect"]),
                    "image_type": compact_text(row["image_type"]),
                    "type": "multimodal_question_answer",
                },
                body=(
                    f"Question: {question}\n\n"
                    f"A. {choices['A']}\nB. {choices['B']}\nC. {choices['C']}\nD. {choices['D']}\n\n"
                    f"Answer: {answer_choice}. {answer}"
                ),
            ),
        )

        rows.append(
            {
                "sample_id": sample_id,
                "source": "uclanlp/MRAG-Bench",
                "scenario": compact_text(row["scenario"]),
                "aspect": compact_text(row["aspect"]),
                "image_type": compact_text(row["image_type"]),
                "image_file": f"multimodal/images/{image_file}",
                "text_file": f"multimodal/text/{text_file}",
                "question": question,
                "choices": choices,
                "answer_choice": answer_choice,
                "answer": answer,
                "expected_sources": [image_file, text_file],
                "expected_keywords": [answer],
            }
        )

    return rows


def write_readme(manifest: dict) -> None:
    readme = f"""
# Evaluation Dataset for AI Job Apply Assistant

This directory contains a moderate-size curated dataset for the local project:

- Resume-JD matching evaluation
- RAG retrieval and rerank evaluation
- Application-material generation tests
- Small multimodal image QA sanity checks

## Contents

- `resumes/`: Markdown resumes converted from public anonymous or synthetic sources.
- `jds/`: Markdown job descriptions.
- `pairs/resume_jd_pairs.jsonl`: labeled resume-JD pairs.
- `pairs/synthetic_resume_jd_pairs.jsonl`: synthetic resume-JD pairs with source scores.
- `rag_queries/retrieval_eval.jsonl`: query-level retrieval evaluation samples.
- `multimodal/images/`: sampled images from MRAG-Bench.
- `multimodal/text/`: text sidecars for multimodal QA samples.
- `multimodal/mrag_eval.jsonl`: multimodal QA evaluation records.
- `dataset_manifest.json`: counts, schema notes, and source links.

## Counts

- Real anonymous resumes: {manifest['counts']['real_resumes']}
- Synthetic resumes: {manifest['counts']['synthetic_resumes']}
- Real job descriptions: {manifest['counts']['real_jds']}
- Synthetic job descriptions: {manifest['counts']['synthetic_jds']}
- Real resume-JD pairs: {manifest['counts']['real_pairs']}
- Synthetic resume-JD pairs: {manifest['counts']['synthetic_pairs']}
- Retrieval evaluation queries: {manifest['counts']['retrieval_queries']}
- Multimodal samples: {manifest['counts']['multimodal_samples']}

## Suggested Use

1. Ingest `resumes/*.md` into the profile knowledge base.
2. Ingest `jds/*.md` into the job-description knowledge base.
3. Use `rag_queries/retrieval_eval.jsonl` for `HitRate@K`, `MRR@K`, and `KeywordRecall@K`.
4. Use `pairs/*.jsonl` for resume-JD scoring and ranking experiments.
5. Use `multimodal/mrag_eval.jsonl` for small image understanding and citation checks.

## Source Notes

This dataset is for local development, demos, and educational evaluation. Keep the source license notes in `dataset_manifest.json` when reusing or redistributing.
"""
    write_text(OUT / "README.md", readme)


def main() -> None:
    reset_curated_dirs()

    real_jds, real_resumes, mixed_real = build_real_resume_jd_dataset()
    real_pairs = [row for row in mixed_real if "query_id" not in row]
    retrieval_queries = [row for row in mixed_real if "query_id" in row]

    synth_jds, synth_resumes, synth_pairs = build_synthetic_hf_subset()
    mrag_rows = build_multimodal_subset()

    write_jsonl(OUT / "pairs" / "resume_jd_pairs.jsonl", real_pairs)
    write_jsonl(OUT / "pairs" / "synthetic_resume_jd_pairs.jsonl", synth_pairs)
    write_jsonl(OUT / "rag_queries" / "retrieval_eval.jsonl", retrieval_queries)
    write_jsonl(OUT / "multimodal" / "mrag_eval.jsonl", mrag_rows)

    manifest = {
        "name": "ai_job_apply_assistant_eval_dataset",
        "version": "2026-06-01",
        "purpose": [
            "resume_jd_matching",
            "rag_retrieval_evaluation",
            "rerank_comparison",
            "multimodal_rag_sanity_check",
        ],
        "counts": {
            "real_resumes": len(real_resumes),
            "synthetic_resumes": len(synth_resumes),
            "real_jds": len(real_jds),
            "synthetic_jds": len(synth_jds),
            "real_pairs": len(real_pairs),
            "synthetic_pairs": len(synth_pairs),
            "retrieval_queries": len(retrieval_queries),
            "multimodal_samples": len(mrag_rows),
        },
        "sources": [
            {
                "name": "vacancy-resume-matching-dataset",
                "url": "https://github.com/NataliaVanetik/vacancy-resume-matching-dataset",
                "license": "GPL-3.0",
                "used_for": "30 anonymous resumes, 5 job descriptions, human ranking labels",
            },
            {
                "name": "resume-matching-dataset-v2",
                "url": "https://huggingface.co/datasets/Divyanandh/resume-matching-dataset-v2",
                "license": "see Hugging Face dataset card",
                "used_for": "40 synthetic resume-JD scoring samples",
                "caution": "Synthetic Korean data generated by GPT-4o-mini in the original dataset.",
            },
            {
                "name": "MRAG-Bench",
                "url": "https://github.com/mragbench/MRAG-Bench",
                "license": "see upstream repository and dataset card",
                "used_for": "12 sampled multimodal image QA records",
            },
        ],
        "schemas": {
            "pairs/*.jsonl": "pair_id, source, resume_id, resume_file, jd_id, jd_file, match_score or avg_rank, relevance_label, expected_keywords",
            "rag_queries/retrieval_eval.jsonl": "query_id, scenario, query, expected_sources, expected_keywords, relevance_label",
            "multimodal/mrag_eval.jsonl": "sample_id, image_file, text_file, question, choices, answer, expected_sources, expected_keywords",
        },
    }
    (OUT / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(manifest)

    large_mrag_cache = HF_SRC / "test-00000-of-000028.parquet"
    if large_mrag_cache.exists():
        large_mrag_cache.unlink()

    for partial in [SRC / "resume_corpus", SRC / "CV-JD-Matching"]:
        if partial.exists() and not any(partial.iterdir()):
            partial.rmdir()

    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
