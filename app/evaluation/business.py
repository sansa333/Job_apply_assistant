"""Business-task metrics and frozen, source-anchored non-human evaluation data."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from app.chunking import JD_CHUNK_SCHEMA_VERSION
from app.evaluation.business_cases import DISCOVERY_CASES, EVIDENCE_CASES
from app.evaluation.domestic import digest, is_notice, rank_metrics, write_jsonl
from app.knowledge.ingestion import JobKnowledgeIngestion


def graded_metrics(ranking, grades, *, relevant_grade=1, ks=(1, 2, 3, 5, 10, 20)):
    ranking = list(dict.fromkeys(ranking))
    relevant = {key for key, grade in grades.items() if grade >= relevant_grade}
    result = rank_metrics(ranking, relevant, ks=ks)
    ideal = sorted(grades.values(), reverse=True)
    for k in ks:
        # Fixed k denominator: an unfilled result slot is not a relevant result.
        result[f"precision_at_{k}"] = len(set(ranking[:k]) & relevant) / k
        dcg = sum((2 ** grades.get(key, 0) - 1) / math.log2(i + 2) for i, key in enumerate(ranking[:k]))
        idcg = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal[:k]))
        result[f"ndcg_at_{k}"] = dcg / idcg if idcg else None
    return result


def random_expectation(candidate_ids, grades, *, relevant_grade=1, ks=(1, 2, 3, 5, 10, 20)):
    n = len(candidate_ids)
    relevant = sum(grades.get(key, 0) >= relevant_grade for key in candidate_ids)
    result = {"relevant_count": relevant}
    for k in ks:
        draws = min(k, n)
        result[f"precision_at_{k}"] = draws * relevant / (n * k) if n else 0.0
        result[f"hit_at_{k}"] = (1 - math.comb(n - relevant, draws) / math.comb(n, draws)) if relevant else None
        result[f"mrr_at_{k}"] = sum(math.comb(n - rank, relevant - 1) / math.comb(n, relevant) / rank
            for rank in range(1, min(k, n - relevant + 1) + 1)) if relevant else None
        result[f"recall_at_{k}"] = min(k, n) / n if relevant else None
        ideal = sorted((grades.get(key, 0) for key in candidate_ids), reverse=True)
        idcg = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal[:k]))
        mean_gain = sum(2 ** grade - 1 for grade in ideal) / n if n else 0
        result[f"ndcg_at_{k}"] = mean_gain * sum(1 / math.log2(i + 2) for i in range(min(k, n))) / idcg if idcg else None
    return result


def summarize(rows):
    keys = {key for row in rows for key in row if key.startswith(("mrr_at_", "recall_at_", "precision_at_", "hit_at_", "ndcg_at_", "complete_support_at_"))}
    return {"instances": len(rows), "answerable": sum(r["relevant_count"] > 0 for r in rows), **{
        key: sum(values) / len(values) if (values := [r[key] for r in rows if r.get(key) is not None]) else None for key in sorted(keys)
    }}


def fits_filters(record, filters):
    return all((
        not filters.get("company_name") or filters["company_name"] in record.company_name,
        not filters.get("location") or filters["location"] in (record.location or ""),
        not filters.get("recruitment_type") or filters["recruitment_type"] == record.recruitment_type,
        not filters.get("graduation_year") or filters["graduation_year"] == record.graduation_year,
    ))


def _grams(text):
    text = re.sub(r"\s+", "", text).lower()
    return {text[i:i + 5] for i in range(max(1, len(text) - 4))}


def build_business_dataset(directory: Path, records):
    directory.mkdir(parents=True, exist_ok=False)
    unique = {}
    for r in sorted(records, key=lambda r: r.job_id):
        if not is_notice(r):
            unique.setdefault(digest(re.sub(r"\s+", "", r.description).lower()), r)
    records = list(unique.values())
    by_id = {r.job_id: r for r in records}
    # Audit near duplicates across preassigned companies before any model run.
    near_pairs = []
    gram_sets = {r.job_id: _grams(r.description) for r in records}
    parent = {r.job_id: r.job_id for r in records}

    def group(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for i, left in enumerate(records):
        for right in records[i + 1:]:
            a, b = gram_sets[left.job_id], gram_sets[right.job_id]
            score = len(a & b) / len(a | b) if a | b else 1
            if score >= .85:
                near_pairs.append([left.job_id, right.job_id, score])
                parent[group(right.job_id)] = group(left.job_id)
    chunker = JobKnowledgeIngestion.__new__(JobKnowledgeIngestion)
    chunker.source_corpus_dir = directory / "public_sources"
    evidence, by_job = [], defaultdict(list)
    for record in records:
        for doc in chunker._chunks_for_record(record):
            row = {"evidence_id": doc.metadata["chunk_id"], "job_id": record.job_id, "text": doc.page_content, "metadata": doc.metadata}
            evidence.append(row)
            by_job[record.job_id].append(row)
    queries, qrels = [], []
    split_groups = defaultdict(set)
    for job_id, split, positives, negatives in EVIDENCE_CASES:
        if job_id not in by_id:
            raise ValueError(f"Source job missing or deduplicated: {job_id}")
        split_groups[group(job_id)].add(split)
        tasks = [(q, quotes, kind) for q, quotes, kind in positives] + [(q, [], "unanswerable") for q in negatives]
        for i, (question, quotes, kind) in enumerate(tasks):
            for quote in quotes:
                if quote not in by_id[job_id].description:
                    raise ValueError(f"Quote not present in source: {job_id}: {quote}")
            qid = f"e:{job_id}:{i}"
            matches = {e["evidence_id"]: [j for j, quote in enumerate(quotes) if quote in e["text"]]
                       for e in by_job[job_id] if e["metadata"].get("evidence_kind") == "source_text"}
            found = {j for ids in matches.values() for j in ids}
            if found != set(range(len(quotes))):
                raise ValueError(f"Production chunks lost annotated support: {qid}")
            queries.append({"query_id": qid, "job_id": job_id, "query": question, "split": split,
                            "group_id": group(job_id), "company": by_id[job_id].company_name,
                            "kind": kind, "support_quotes": quotes, "pool_size": len(by_job[job_id]),
                            "annotation_status": "assistant_source_reviewed_not_human_gold",
                            "absence_basis": "Assistant reviewed full supplied JD; requested fact is not stated" if not quotes else None})
            for e in by_job[job_id]:
                covered = matches.get(e["evidence_id"], [])
                grade = (2 if len(covered) == len(quotes) else 1) if covered else 0
                qrels.append({"query_id": qid, "evidence_id": e["evidence_id"], "grade": grade, "support_indices": covered})
    if any(len(splits) > 1 for splits in split_groups.values()):
        raise ValueError("Near-duplicate evidence tasks cross development/test")
    search, search_labels = [], []
    for qid, split, question, facets, filters in DISCOVERY_CASES:
        search.append({"query_id": qid, "split": split, "group_id": qid, "query": question,
                       "facets": facets, "filters": filters, "annotation_status": "requirement_contract_silver"})
        for r in records:
            text = r.job_title + "\n" + r.description
            hits = [bool(re.search(f, text, re.I)) for f in facets]
            eligible = fits_filters(r, filters)
            grade = (2 if all(hits) else (1 if any(hits) else 0)) if eligible else 0
            search_labels.append({"query_id": qid, "job_id": r.job_id, "grade": grade,
                                  "eligible": eligible, "facet_matches": hits})
    for name, rows in {"jobs": [asdict(r) for r in records], "evidence": evidence, "queries": queries,
                       "qrels": qrels, "search_queries": search, "search_qrels": search_labels}.items():
        write_jsonl(directory / f"{name}.jsonl", rows)
    text_by_id = {e["evidence_id"]: e["text"] for e in evidence}
    query_by_id = {q["query_id"]: q for q in queries}
    write_jsonl(directory / "human_review_evidence.jsonl", [{"query_id": r["query_id"], "evidence_id": r["evidence_id"],
        "query": query_by_id[r["query_id"]]["query"], "text": text_by_id[r["evidence_id"]],
        "reviewer_a_grade": None, "reviewer_b_grade": None, "adjudicated_grade": None} for r in qrels])
    write_jsonl(directory / "human_review_search.jsonl", [{"query_id": r["query_id"], "job_id": r["job_id"],
        "reviewer_a_grade": None, "reviewer_b_grade": None, "adjudicated_grade": None} for r in search_labels])
    manifest = {"version": "domestic_business_v4.1", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_note": "Frozen current local official-source catalog, originally synchronized 2026-09-18; no claim of fresh unseen documents",
        "jobs": len(records), "evidence": len(evidence), "evidence_queries": len(queries), "discovery_queries": len(search),
        "chunk_schema": JD_CHUNK_SCHEMA_VERSION, "near_duplicate_threshold": .85, "near_duplicate_pairs": near_pairs,
        "evidence_split": "Baidu/OPPO development; Meituan/Kylin test; near duplicate groups checked",
        "discovery_split": "10 development and 20 fixed test intents; no tuning on test; not real user logs",
        "annotation_status": "NO HUMAN GOLD: source-quoted assistant labels + exhaustive search requirement contracts",
        "label_revision": "Post-run source audit added speech synthesis aliases token2wav/TTS/ASR and inverted video-generation wording. Regression, not independent blind test. Original v4 results retained.",
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob("*.jsonl")}}
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def calibration_metrics(rows, threshold):
    positives = [r for r in rows if r["answerable"]]
    negatives = [r for r in rows if not r["answerable"]]
    accepted = [r for r in rows if r["score"] >= threshold]
    return {"n": len(rows), "answerable": len(positives), "unanswerable": len(negatives),
        "false_accept_rate": sum(r["score"] >= threshold for r in negatives) / len(negatives) if negatives else None,
        "false_reject_rate": sum(r["score"] < threshold for r in positives) / len(positives) if positives else None,
        "coverage": len(accepted) / len(rows) if rows else None,
        "accepted_unsupported_rate": sum(not r["answerable"] for r in accepted) / len(accepted) if accepted else None,
        "accepted_incomplete_support_rate": sum(not r.get("complete_source_support", False) for r in accepted) / len(accepted)
            if accepted and all("complete_source_support" in r for r in rows) else None}


def calibrate_threshold(rows, max_false_accept=.10):
    if not rows or any(r["split"] != "development" for r in rows):
        raise ValueError("Threshold may only be fitted on development rows")
    if not any(r["answerable"] for r in rows) or not any(not r["answerable"] for r in rows):
        raise ValueError("Calibration needs both answerable and unanswerable development examples")
    options = sorted({-1.0, 1.000001, *(r["score"] + 1e-7 for r in rows)})
    eligible = [(calibration_metrics(rows, t), t) for t in options]
    eligible = [(m, t) for m, t in eligible if m["false_accept_rate"] <= max_false_accept]
    # Maximize answerable recall under a predeclared false acceptance budget.
    return min(eligible, key=lambda item: (item[0]["false_reject_rate"], item[0]["false_accept_rate"], item[1]))[1]
