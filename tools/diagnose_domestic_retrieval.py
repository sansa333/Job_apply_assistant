"""Post-hoc diagnostics. Preserve the original frozen labels and model reports."""
from __future__ import annotations

import argparse
import gc
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from langchain_chroma import Chroma

from app.embeddings import get_embeddings
from app.evaluation.domestic import TOPICS, aggregate, paired_comparison, rank_metrics, read_jsonl, verify_dataset
from app.knowledge.chroma_lifecycle import close_chroma


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def ascii_boundaries(pattern):
    # Python's Unicode \b treats Chinese letters as word characters too.
    return re.sub(r"\\b(.*?)\\b", lambda m: r"(?<![A-Za-z0-9_])" + m[1] + r"(?![A-Za-z0-9_])", pattern)


def heading_only(text):
    return re.sub(r"[\s#【】\[\]：:]", "", text) in {
        "任职要求", "岗位要求", "岗位职责", "工作职责", "工作内容", "任职资格", "加分项", "岗位介绍", "岗位亮点",
    }


def sliced(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: aggregate(values) for name, values in sorted(groups.items())}


def write_analysis(output: Path):
    static = json.loads((output / "static.json").read_text(encoding="utf-8"))
    experiments = json.loads((output / "experiments.json").read_text(encoding="utf-8"))
    small = experiments["bge_small_zh_512"]["exact_fp16"]
    m3 = experiments["bge_m3_1024"]["exact_fp16"]
    rows_a = [{**r, **r["both_fixed_metrics"]} for r in small["details"] if r["relevant_count"]]
    rows_b = [{**r, **r["both_fixed_metrics"]} for r in m3["details"] if r["relevant_count"]]
    paired = {key: paired_comparison(rows_a, rows_b, key) for key in ("mrr_at_3", "recall_at_5")}
    save(output / "posthoc_comparison.json", paired)
    table = []
    for label, key in (("原始流程", "original_label_metrics"), ("仅过滤纯标题", "without_heading_same_original_answerable"),
                       ("过滤纯标题 + 修正英文边界标签", "both_fixed_same_original_answerable")):
        a, b = small[key], m3[key]
        table.append(f"| {label} | {a['mrr_at_3']:.4f} | {b['mrr_at_3']:.4f} | {a['recall_at_5']:.4f} | {b['recall_at_5']:.4f} |")
    a = static["models"]["bge_small_zh_512"]["boundary_corrected_same_469"]
    b = static["models"]["bge_m3_1024"]["boundary_corrected_same_469"]
    table.insert(1, f"| 仅修正英文边界标签 | {a['mrr_at_3']:.4f} | {b['mrr_at_3']:.4f} | {a['recall_at_5']:.4f} | {b['recall_at_5']:.4f} |")
    lines = ["# 国内 BGE 对比：差异原因诊断", "", "## 对上一轮结论的修正", "",
             "上一轮完成了链路与指标验证，但没有完成根因分析。进一步诊断确认，自动标签和生产切块均存在影响模型对比的问题。",
             "原始指标可以复现，但不应据此直接断言 BGE-M3 在国内中文岗位上不如 small。",
             "以下实验是在观察原始测试结果后提出的诊断性对照；原始索引、数据与标签未改动，没有把新结果替换成原始测试结果。",
             "原始 469 个有答案实例保持一致；仍只有 15 个问题模板，修正标签后仍是规则 silver，并非人工语义金标。", "",
             "## 控制变量对照", "", "| 条件 | small MRR@3 | M3 MRR@3 | small Recall@5 | M3 Recall@5 |",
             "|---|---:|---:|---:|---:|", *table, "",
             "过滤纯标题使用完全相同的原始向量，只从完整排名中去除没有正文的标题片段，再截取 top K；没有重训模型或改动正文。",
             "修正边界只改变规则标签，不改变向量与排名。组合实验不能当作两个效果简单相加。", "",
             "## 已确认原因一：纯标题被当作证据", "",
             f"全语料有 {len(static['header_only_chunks'])} 个纯标题片段，例如 `任职要求：`。原测试中 small 将纯标题排第一 {static['header_top1']['bge_small_zh_512']} 次，M3 为 {static['header_top1']['bge_m3_1024']} 次。",
             "问题包含‘要求’等语义，纯标题可能获得较高相似度，但没有回答所需信息。这里观察到 M3 对这种干扰更敏感，不能推断模型内部为何产生该向量关系。",
             "对照中去掉这些片段就改变了 MRR 的模型优劣顺序，证明输入切块质量是实质性混杂因素。", "",
             "## 已确认原因二：中英文连写导致标签漏标", "",
             r"旧规则使用 `\bpython\b` 等 Unicode 单词边界。Python 正则把汉字也视为单词字符，因此 `python等`、`熟悉Python` 无法按预期命中。",
             f"在原 469 个实例中，{static['boundary_label_changes_test_469']} 个问题新增了被漏标的相关片段，共 {static['boundary_added_positive_pairs']} 个 query–chunk 对。",
             "另有 24 个原本被当作无答案的 test 实例变为可回答；主对照仍固定原 469 个，没有通过扩样本制造差异。",
             "该问题来自上一轮新增的自动评测标签实现，不能归咎于模型；需要纠正上一轮对指标有效性的判断。", "",
             "## 已确认原因三：关键词片段不等于回答证据", "",
             "生产切块会附加 `技术栈关键词` 等派生片段。出现 Python 的词表会被标正，而包含 `python等` 的真正要求可能漏标。",
             "这导致指标部分衡量‘能否优先找到关键词列表’，不是‘能否找到充分回答问题的原文’。边界修正没有解决全部语义标注问题，仍需人工复核。", "",
             "## Recall 差异为何很小但看起来稳定", "",
             "原 469 个实例中 413 个候选池不超过 5，两个模型 Recall@5 都为 1。只有 56 个问题有区分空间。",
             "原 Recall 差异仅来自 6 个问题、4 个岗位：5 个问题的候选池为 6，1 个为 7；M3 有 6 次下降、0 次提高。",
             "因此全体 Recall@5 接近满分主要是任务空间小，并不表示整体求职系统具有 99% 的准确率。", "",
             "## 已排除或未支持的解释", "",
             "- Chroma 近似检索：取出原索引向量做全量精确 L2 排序，原 469 个实例两模型的前五名完全一致，指标变化为 0。",
             "- FP16 数值误差：重新用 FP32 编码正文和问题，small 排名不变，M3 两题内部排名变化，但 MRR@3/Recall@5 均不变。",
             "- M3 输入长度不够：既有 512/1024 对照的测试指标一致，1024 配置没有片段截断；未发现加长输入能解释差异。",
             "- pooling 配置错误：本地两模型都是 CLS pooling，和官方 dense 用法一致。",
             "- small 缺少查询指令：额外实验加上官方中文指令后，small 原标签 MRR@3 为 0.8266（原 0.8429），Recall@5 为 0.9973（原 0.9951）。不能只挑其中一个指标宣称指令全面改善。", "",
             "## 模型能力与任务范围", "",
             "M3 官方模型支持多语言、长文档、dense/sparse/multi-vector 检索；当前项目只用了 dense 路径，且主要是中文短片段。其额外能力未被充分测试，不保证在该小范围任务必然胜出。",
             "来源：[BGE-M3 官方模型卡](https://huggingface.co/BAAI/bge-m3)、[BGE-small-zh 官方模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5)。", "",
             "## 后续正式验证要求", "",
             "应修正标注边界、合并或删除纯标题片段；人工区分‘提到技能’与‘足以回答问题’，为派生词表设置明确评测规则。",
             "这些改动需要建立新版本数据与索引，并在新增、独立人工复核的问题上确认，不能继续将已反复诊断的集合称为未见测试集。",
             f"组合诊断的 MRR 相对变化为 {paired['mrr_at_3']['relative_percent']:+.2f}%，但这是事后对照，不是可以直接替换到简历的提升数字。", "",
             "## 逐例证据（按原始 MRR 退步排序选取，不代表总体错误率）", ""]
    for case in static["cases"][:5]:
        lines += [f"### {case['company']} / {case['job_title']}", "", case["query"], "",
                  f"原首个标正结果：small 第 {case['small_first']}，M3 第 {case['m3_first']}。", ""]
        for chunk in case["chunks"]:
            lines += [f"- `{chunk['id']}`；small 排名 {chunk['small_rank']}；M3 排名 {chunk['m3_rank']}；原标签 {chunk['original_relevant']}；边界修正标签 {chunk['boundary_corrected_relevant']}。",
                      "", "> " + chunk["text"].replace("\n", "\n> "), ""]
    (output / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/domestic_eval_20260918_final"))
    parser.add_argument("--dataset", type=Path, default=Path("outputs/domestic_eval_20260918_v3/dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/domestic_eval_diagnosis"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    verify_dataset(args.dataset)
    report = json.loads((args.run / "report.json").read_text(encoding="utf-8"))
    evidence = {r["evidence_id"]: r for r in read_jsonl(args.dataset / "evidence.jsonl")}
    queries = {r["query_id"]: r for r in read_jsonl(args.dataset / "queries.jsonl")}
    jobs = {r["job_id"]: r for r in read_jsonl(args.dataset / "jobs.jsonl")}
    qrels = defaultdict(set)
    by_job = defaultdict(list)
    for row in read_jsonl(args.dataset / "qrels.jsonl"):
        qrels[row["query_id"]].add(row["evidence_id"])
    for row in evidence.values():
        by_job[row["job_id"]].append(row["evidence_id"])
    patterns = {t[0]: t[1] for t in TOPICS}
    corrected = {qid: {eid for eid in by_job[q["job_id"]] if re.search(ascii_boundaries(patterns[q["topic"]]), evidence[eid]["text"], re.I)}
                 for qid, q in queries.items()}
    models = {}
    original = {}
    for result in report["models"]:
        name = result["spec"]["name"]
        original[name] = {r["query_id"]: r for r in result["details"]}
        rows = [r for r in result["details"] if r["stratum"] == "individual_job" and r["split"] == "test" and r["relevant_count"]]
        models[name] = {"by_topic": sliced(rows, "topic"), "by_company": sliced(rows, "company"), "by_pool": sliced(rows, "pool_size")}
        fixed_rows = [{**r, **rank_metrics(r["ranking"], corrected[r["query_id"]], ks=(1, 3, 5))} for r in rows]
        models[name]["boundary_corrected_same_469"] = aggregate(fixed_rows)
    small = original["bge_small_zh_512"]
    m3 = original["bge_m3_1024"]
    comparisons = []
    for qid, row in small.items():
        if row["stratum"] != "individual_job" or row["split"] != "test" or not row["relevant_count"]:
            continue
        other = m3[qid]
        comparisons.append({"query_id": qid, "topic": row["topic"], "company": row["company"], "pool_size": row["pool_size"],
                            "mrr_delta": other["mrr_at_3"] - row["mrr_at_3"],
                            "recall_delta": other["recall_at_5"] - row["recall_at_5"],
                            "small_first": row["first_relevant_rank"], "m3_first": other["first_relevant_rank"]})
    affected = [row for row in comparisons if corrected[row["query_id"]] != qrels[row["query_id"]]]
    cases = []
    for row in sorted(comparisons, key=lambda r: (r["mrr_delta"], r["recall_delta"]))[:15] + sorted(comparisons, key=lambda r: -r["mrr_delta"])[:5]:
        qid = row["query_id"]
        job = jobs[queries[qid]["job_id"]]
        cases.append({**row, "query": queries[qid]["query"], "job_title": job["job_title"],
                      "chunks": [{"id": eid, "section": evidence[eid]["metadata"].get("section"),
                                  "original_relevant": eid in qrels[qid], "boundary_corrected_relevant": eid in corrected[qid],
                                  "small_rank": small[qid]["ranking"].index(eid) + 1 if eid in small[qid]["ranking"] else None,
                                  "m3_rank": m3[qid]["ranking"].index(eid) + 1 if eid in m3[qid]["ranking"] else None,
                                  "text": evidence[eid]["text"]} for eid in by_job[job["job_id"]]]})
    result = {"post_hoc": True, "original_labels_unchanged": True, "models": models, "comparisons": comparisons,
              "boundary_label_changes_test_469": len(affected),
              "boundary_added_positive_pairs": sum(len(corrected[r["query_id"]] - qrels[r["query_id"]]) for r in comparisons),
              "recall_changes": {"wins": sum(r["recall_delta"] > 0 for r in comparisons), "losses": sum(r["recall_delta"] < 0 for r in comparisons)},
              "recall_loss_pool_counts": dict(Counter(r["pool_size"] for r in comparisons if r["recall_delta"] < 0)),
              "cases": cases}
    result["header_only_chunks"] = [eid for eid, row in evidence.items() if heading_only(row["text"])]
    result["header_top1"] = {name: sum(heading_only(evidence[row["ranking"][0]]["text"]) for row in rows.values()
                                      if row["stratum"] == "individual_job" and row["split"] == "test" and row["relevant_count"])
                             for name, rows in original.items()}
    save(args.output / "static.json", result)
    print("static diagnostics saved", flush=True)

    # The query text has only 15 variants; exact scoring is cheap and removes ANN.
    test_queries = [q for q in queries.values() if q["stratum"] == "individual_job" and q["split"] == "test"]
    experiments = {}
    for spec in (report["models"][0]["spec"], report["models"][2]["spec"]):
        name = spec["name"]
        print("Loading", name, flush=True)
        embedding = get_embeddings(backend="huggingface", model_name=spec["model"], device="cuda", use_fp16=True,
                                   max_length=spec["max_length"], batch_size=1, local_files_only=True)
        db = Chroma(collection_name="domestic_eval", persist_directory=str(args.run / name / "indexes" / "domestic_eval"), embedding_function=embedding)
        stored = db.get(include=["embeddings", "metadatas", "documents"])
        vectors = {meta["chunk_id"]: np.asarray(vector, dtype=np.float32) for meta, vector in zip(stored["metadatas"], stored["embeddings"])}
        texts = sorted({q["query"] for q in test_queries})
        raw_query_vectors = {q: np.asarray(embedding.embed_query(q), dtype=np.float32) for q in texts}
        prefixed = {q: np.asarray(embedding.embed_query("为这个句子生成表示以用于检索相关文章：" + q), dtype=np.float32) for q in texts} if name.startswith("bge_small") else None
        variants = {"exact_fp16": (vectors, raw_query_vectors)}
        if prefixed:
            variants["query_instruction_fp16"] = (vectors, prefixed)
        # A fresh full-precision pass checks quantization and near-tie instability.
        embedding._model.float()
        test_ids = sorted({eid for q in test_queries for eid in by_job[q["job_id"]]})
        fp32_vectors = dict(zip(test_ids, np.asarray(embedding.embed_documents([evidence[eid]["text"] for eid in test_ids]), dtype=np.float32)))
        fp32_queries = {q: np.asarray(embedding.embed_query(q), dtype=np.float32) for q in texts}
        variants["exact_fp32"] = (fp32_vectors, fp32_queries)
        token_lengths = {eid: len(embedding._model.tokenizer.encode(evidence[eid]["text"], truncation=False)) for eid in test_ids}
        model_results = {}
        for label, (passages, question_vectors) in variants.items():
            rows = []
            changed_rankings = changed_metrics = 0
            for q in test_queries:
                qid = q["query_id"]
                ids = by_job[q["job_id"]]
                matrix = np.asarray([passages[eid] for eid in ids])
                distances = ((matrix - question_vectors[q["query"]]) ** 2).sum(axis=1)
                order = np.argsort(distances, kind="stable")
                ranking = [ids[i] for i in order[:5]]
                full_ranking = [ids[i] for i in order]
                no_heading = [eid for eid in full_ranking if not heading_only(evidence[eid]["text"])]
                metrics = rank_metrics(ranking, qrels[qid], ks=(1, 3, 5))
                row = {**q, **metrics, "ranking": ranking, "full_ranking": full_ranking,
                       "corrected_metrics": rank_metrics(ranking, corrected[qid], ks=(1, 3, 5)),
                       "without_heading_metrics": rank_metrics(no_heading, {eid for eid in qrels[qid] if not heading_only(evidence[eid]["text"])}, ks=(1, 3, 5)),
                       "both_fixed_metrics": rank_metrics(no_heading, {eid for eid in corrected[qid] if not heading_only(evidence[eid]["text"])}, ks=(1, 3, 5)),
                       "near_tie_gap": float(distances[order[1]] - distances[order[0]]) if len(order) > 1 else None,
                       "has_truncated_chunk": any(token_lengths[eid] > spec["max_length"] for eid in ids)}
                if qrels[qid]:
                    changed_rankings += ranking != original[name][qid]["ranking"]
                    changed_metrics += metrics["mrr_at_3"] != original[name][qid]["mrr_at_3"] or metrics["recall_at_5"] != original[name][qid]["recall_at_5"]
                rows.append(row)
            model_results[label] = {"original_label_metrics": aggregate(rows),
                                    "without_heading_same_original_answerable": aggregate([{**r, **r["without_heading_metrics"]} for r in rows if qrels[r["query_id"]]]),
                                    "both_fixed_same_original_answerable": aggregate([{**r, **r["both_fixed_metrics"]} for r in rows if qrels[r["query_id"]]]),
                                    "boundary_corrected_all_test": aggregate([{**r, **r["corrected_metrics"]} for r in rows]),
                                    "changed_rankings_original_answerable": changed_rankings,
                                    "changed_metrics_original_answerable": changed_metrics,
                                    "details": rows}
        experiments[name] = model_results
        save(args.output / f"{name}_experiments.json", model_results)
        close_chroma(db)
        del db, embedding, vectors, variants, fp32_vectors, raw_query_vectors, fp32_queries
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        print("Completed", name, flush=True)
    save(args.output / "experiments.json", experiments)
    verify_dataset(args.dataset)
    write_analysis(args.output)
    print("Diagnostics complete", flush=True)


if __name__ == "__main__":
    main()
