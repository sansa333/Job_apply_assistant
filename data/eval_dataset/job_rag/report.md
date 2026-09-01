# 岗位 RAG 检索评测报告

样本数：80

## 检索配置
- Embedding：BAAI/bge-small-zh-v1.5
- 策略：hybrid_rerank（向量召回 + BM25 + RRF + Cross-Encoder 分数融合）
- 候选数：12；RRF 常数：60；Cross-Encoder 权重：0.2
- Cross-Encoder：BAAI/bge-reranker-v2-m3；实际应用：80/80

## 总体指标
- hit_rate_at_1: 0.6625
- hit_rate_at_3: 1.0000
- hit_rate_at_5: 1.0000
- mrr_at_1: 0.6625
- recall_at_1: 0.6625
- mrr_at_3: 0.8208
- recall_at_3: 1.0000
- mrr_at_5: 0.8208
- recall_at_5: 1.0000
- keyword_recall_at_5: 1.0000

## 数据集诊断
- 唯一问句：7/80
- 平均相关 chunk 数：1.00
- 多相关查询比例：0.0000
- 平均候选池大小：2.67
- 评测警告：low_query_diversity, single_relevant_label_recall_equals_hit_rate, candidate_pool_at_or_below_max_k_metric_saturation

## 问题类型分布

| 问题类型 | 样本数 |
| --- | ---: |
| benefits | 11 |
| experience_education | 11 |
| location_work_mode | 11 |
| overview | 12 |
| qualifications | 11 |
| responsibilities | 12 |
| technical_skills | 12 |

## 分类型指标

| 问题类型 | 样本数 | HitRate@1 | HitRate@3 | Recall@3 | Recall@5 | MRR@3 | MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| benefits | 11 | 0.6364 | 1.0000 | 1.0000 | 1.0000 | 0.8030 | 0.8030 |
| experience_education | 11 | 0.8182 | 1.0000 | 1.0000 | 1.0000 | 0.9091 | 0.9091 |
| location_work_mode | 11 | 0.7273 | 1.0000 | 1.0000 | 1.0000 | 0.8636 | 0.8636 |
| overview | 12 | 0.7500 | 1.0000 | 1.0000 | 1.0000 | 0.8750 | 0.8750 |
| qualifications | 11 | 0.5455 | 1.0000 | 1.0000 | 1.0000 | 0.7424 | 0.7424 |
| responsibilities | 12 | 0.7500 | 1.0000 | 1.0000 | 1.0000 | 0.8611 | 0.8611 |
| technical_skills | 12 | 0.4167 | 1.0000 | 1.0000 | 1.0000 | 0.6944 | 0.6944 |

## Top-1 未命中样本
- 数量：27
- job_eval_003 (technical_skills): rank=3 | 这个岗位需要哪些技术技能或工具能力？
- job_eval_005 (experience_education): rank=2 | 申请这个岗位需要怎样的工作经验或学历背景？
- job_eval_008 (overview): rank=2 | 这个岗位主要是做什么的？
- job_eval_009 (responsibilities): rank=3 | 这个岗位需要承担哪些工作职责？
- job_eval_010 (technical_skills): rank=2 | 这个岗位需要哪些技术技能或工具能力？
- job_eval_011 (qualifications): rank=3 | 这个岗位有哪些任职资格或必备条件？
- job_eval_018 (qualifications): rank=2 | 这个岗位有哪些任职资格或必备条件？
- job_eval_024 (technical_skills): rank=2 | 这个岗位需要哪些技术技能或工具能力？
- job_eval_027 (location_work_mode): rank=2 | 这个岗位的工作地点和远程或混合办公安排是什么？
- job_eval_028 (benefits): rank=3 | 这个岗位提供哪些薪酬、福利或假期待遇？
