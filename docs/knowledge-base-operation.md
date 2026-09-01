# 真实岗位知识库操作说明

## 数据边界

- `candidate_profile`：仅保存当前用户明确上传的简历、项目和补充经历；每个片段必须带 `candidate_id`。
- `job_knowledge`：仅保存 `open_source` 真实公开历史 JD 与 `user_upload` JD；每个片段必须带 `job_id`。
- `eval_demo`：合成资料、回归样本和多模态演示资料，只能用于离线评测，不能参与求职匹配。

公开数据是历史快照。展示或生成内容必须说明：分析基于知识库中的公开历史岗位描述，岗位当前开放状态请以官方渠道为准。

## 导入、重建与匹配

1. 将获准使用的 kyosek CSV 放到 `data/source_corpus/open_source_jobs/kyosek_jobs.csv`。
2. 执行 `python -m tools.import_open_source_jobs`，导入 CSV 与 `data/eval_dataset/jds/real_en_jd_*.md`。
3. 修改 embedding 或切块策略后，执行 `python -m tools.rebuild_job_knowledge`。SQLite `data/job_catalog.sqlite3` 与 `data/source_corpus` 是重建真源，不以 Chroma 为事实源。
4. 调用 `GET /api/knowledge/health` 检查 collection manifest、embedding 维度和 reranker 状态。
5. 调用 `/api/fit` 时必须提供 `candidate_id`、`company_name`、`job_title`。系统先精确查目录，再严格限制到该 `job_id` 检索；未命中返回 `job_not_found` 并引导上传 JD。

## 生产检索配置

- Embedding：本地 `BAAI/bge-m3`，GPU FP16，`max_length=1024`，`batch_size=4`，输出 1024 维归一化向量。
- 当前质量基线：岗位内 Chroma Dense 检索；BM25、RRF 与 `BAAI/bge-reranker-v2-m3` 作为可配置实验策略。
- 后验实验：`candidate_k=20`、`rrf_k=60`；分别评估 `0.8 × 原排名 + 0.2 × Cross-Encoder` 保守融合和 Cross-Encoder 全量排序。
- 替换门禁：固定 BGE-M3、数据、候选池和查询，仅当策略相对 Dense 的 MRR@3 配对 Bootstrap 95% CI 下界大于 0，并满足 P95 预算时才替换。

该策略始终只处理已精确匹配的一个 `job_id`，不会从其他公司或岗位引入候选片段。

## 模型对比与离线评测

执行：

```bash
python -m tools.compare_job_retrieval_models
python -m tools.build_job_rag_eval
python -m tools.evaluate_bge_m3_retrieval_strategies
```

固定 BGE-M3 的 Retrieval V2 Development 消融比较 7 条链路：BM25、Dense、Dense+BM25 RRF、Dense/Hybrid 保守重排、Dense/Hybrid 全量重排。当前 silver qrels 上 Dense 在 job-scoped 获得 MRR@3=0.7614、Recall@5=0.8750、P95=21.21ms；RRF 与保守重排均未通过相对 Dense 的质量门禁，全量重排还暴露出确定性 query construction 的 focus/intent 冲突。hard-pool 查询不含岗位身份、qrels 却把跨岗位证据记为 0，因此它只用于跨岗位污染压力测试，不作为人工 gold 质量结论。

完整策略报告：

- `data/eval_dataset/job_retrieval_v2/reports/bge_m3_strategy_ablation_report.md`

模型实验会使用同一份 80 条自然中文问题，比较：

- Hash 离线基线；
- `BAAI/bge-small-zh-v1.5`；
- `intfloat/multilingual-e5-small`；
- `BAAI/bge-m3`（若本地权重不完整则明确标为不可用）。

Embedding 选型严格使用 `vector-only` 结果，主排序键依次为 MRR@3、Recall@5、HitRate@1 和查询延迟。报告同时记录模型加载时间、索引构建时间、索引体积，以及领先模型相对第二名的查询级配对 Bootstrap 95% 置信区间。BM25/RRF 和 Cross-Encoder 只在选出候选 Embedding 后做后验验证，避免它们掩盖向量模型差异。实验报告保存在：

- `data/eval_dataset/job_rag/model_experiments/report.json`
- `data/eval_dataset/job_rag/model_experiments/report.md`

主评测集保存在：

- `data/eval_dataset/job_rag/real_job_retrieval_eval.jsonl`
- 完整 JSON 运行报告由评测脚本在本地重新生成，不提交版本库。
- `data/eval_dataset/job_rag/report.md`

主评测问题不包含公司名、岗位名、目标片段关键词或“请重点说明”；它模拟上游已精确确定岗位后的自然用户提问。报告中的 HitRate 和 MRR 衡量的是该岗位内部片段排序，并不代表公司/岗位目录检索准确率。

### 指标定义与使用边界

- `HitRate@K`：每个查询的 Top-K 中只要出现一个相关 chunk 就记 1，最后求平均。它回答“是否至少找到一条可用证据”。
- `Recall@K`：每个查询在 Top-K 找回的不同相关 chunk 数除以该查询全部相关 chunk 数，再进行宏平均。它回答“所需证据是否找全”。
- `MRR@K`：第一个相关 chunk 在 K 以内时取其排名倒数，否则为 0，再对查询求平均。它回答“首条可用证据是否足够靠前”。
- `KeywordRecall@K` 只作为词面覆盖诊断，不参与 Embedding 选型。

当前 80 条旧基线主要由少量自然问句模板生成，而且大多数查询只有一个相关 chunk，岗位内候选池也较小。评测器会在报告中显式输出 `low_query_diversity`、`single_relevant_label_recall_equals_hit_rate` 和 `candidate_pool_at_or_below_max_k_metric_saturation` 等警告。在这些警告消除前，HitRate@3/5 或 Recall@3/5 的饱和值只能用于回归检查，不能写成系统整体效果。
