"""Full-catalog ranking, separate from evidence retrieval for a known job."""
from langchain_core.documents import Document

from app.knowledge.hybrid import bm25_rank, reciprocal_rank_fusion


def rank_catalog(records, query: str, strategy: str, ingestion=None):
    documents = [Document(page_content=f"{r.job_title}\n{r.description}", metadata={"chunk_id": r.job_id, "job_id": r.job_id}) for r in records]
    lexical = bm25_rank(query, documents, k=len(documents))
    if strategy == "bm25":
        return [d.metadata["job_id"] for d in lexical]
    chunks = ingestion.retrieve_across_jobs(query, [r.job_id for r in records], k=200)
    seen, dense = set(), []
    by_id = {d.metadata["job_id"]: d for d in documents}
    for chunk in chunks:
        job_id = chunk.metadata.get("job_id")
        if job_id in by_id and job_id not in seen:
            seen.add(job_id)
            dense.append(by_id[job_id])
    ranked = dense if strategy == "dense" else reciprocal_rank_fusion(dense, lexical, rrf_k=60)
    return [d.metadata["job_id"] for d in ranked]
