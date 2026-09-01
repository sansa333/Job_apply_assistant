# 岗位 RAG 语义模型与检索策略对比

自然问句样本数：80

| 模型 | 策略 | MRR@3 | HitRate@1 | Recall@3 | Recall@5 | 平均查询延迟(ms) | Cross-Encoder |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| hash_baseline | vector | 0.5312 | 0.1875 | 0.9375 | 1.0000 | 2.44 | 0/80 |
| hash_baseline | hybrid | 0.6104 | 0.3125 | 0.9500 | 1.0000 | 3.08 | 0/80 |
| bge_small_zh | vector | 0.8125 | 0.6625 | 0.9875 | 1.0000 | 21.37 | 0/80 |
| bge_small_zh | hybrid | 0.8187 | 0.6625 | 0.9875 | 1.0000 | 19.71 | 0/80 |
| bge_small_zh | hybrid_rerank | 0.8208 | 0.6625 | 1.0000 | 1.0000 | 1294.94 | 80/80 |
| multilingual_e5_small | vector | 0.6625 | 0.4250 | 0.9375 | 1.0000 | 29.49 | 0/80 |
| multilingual_e5_small | hybrid | 0.7396 | 0.5500 | 0.9500 | 1.0000 | 36.11 | 0/80 |
| bge_m3 | - | - | - | - | - | - | 失败：OSError |

## 模型资源开销

| 模型 | 加载(ms) | 建索引(ms) | 索引大小(MiB) |
| --- | ---: | ---: | ---: |
| hash_baseline | 0.0 | 5715.7 | 6.54 |
| bge_small_zh | 135.3 | 16782.9 | 6.71 |
| multilingual_e5_small | 3423.2 | 27303.6 | 6.54 |

## 选型结果
先依据 `vector-only` 选择 `BAAI/bge-small-zh-v1.5`（bge_small_zh）：MRR@3=0.8125、HitRate@1=0.6625、Recall@5=1.0000。随后验证 `hybrid_rerank`：MRR@3=0.8208，Cross-Encoder 实际应用于全部 80 条样本。

## 配对选择证据
- `bge_small_zh` 相对 `multilingual_e5_small` 的配对 MRR@3 差值：0.1500，95% Bootstrap CI [0.0646, 0.2313]。
- 查询级胜/平/负：39/30/11；差异是否明确：True。

## 口径
- 查询为精确 `job_id` 已解析后的自然中文提问，不包含公司、岗位标题、目标片段关键词或“请重点说明”。
- `vector` 为 Chroma 向量召回；`hybrid` 为向量与 BM25 的 RRF 融合；`hybrid_rerank` 在融合候选上执行 Cross-Encoder。
- Embedding 只按 `vector-only` 的 MRR@3、Recall@5、HitRate@1 和延迟排序；混合检索与重排只用于后验验证。
- 数据诊断：唯一问句 7/80；平均相关 chunk 数 1.00；平均候选池 2.67。
- 当前警告：low_query_diversity, single_relevant_label_recall_equals_hit_rate, candidate_pool_at_or_below_max_k_metric_saturation。有警告时结果只能作为工程基线，不能作为最终效果声明。
