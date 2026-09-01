# Evidence-Grounded Job Agent v1 评测报告

- 数据集：`evidence_grounded_job_agent_eval` v1.0.0
- 标签状态：Match labels are first-pass annotations and must not be reported as gold until second annotation and adjudication are complete.
- 基线：`deterministic_skill_lexicon_plus_evidence_weighting` / `evidence_weighted_v1`

## 结构化 JD 解析

- 岗位数：8
- Skill Micro-F1：0.6364
- Skill Macro-F1：0.5933
- P95 延迟：42.95 ms

## 简历–岗位匹配

- 标注对数：24
- Accuracy：0.5833
- Macro-F1：0.4193
- Spearman：0.7364
- mean nDCG@5：0.9597
- mean MRR（首个 high）：0.7222
- P95 延迟：207.83 ms

## Agent 轨迹

- 场景数：10
- 工具序列准确率：1.0000
- 终态准确率：1.0000
- Next-action 准确率：1.0000

## 生成安全校验

- 场景数：8
- Case Exact Accuracy：0.8750
- Finding F1：0.8571

## 匹配 Bad Cases

- match_001: expected=high, predicted=medium, score=38.83
- match_005: expected=medium, predicted=low, score=18.20
- match_006: expected=medium, predicted=low, score=17.38
- match_009: expected=high, predicted=low, score=17.19
- match_013: expected=high, predicted=medium, score=30.89
- match_014: expected=medium, predicted=low, score=14.01
- match_017: expected=high, predicted=low, score=15.11
- match_021: expected=high, predicted=low, score=2.35
- match_022: expected=medium, predicted=low, score=15.64
- match_023: expected=medium, predicted=low, score=4.61

## 解释边界

本报告是小规模、可重复的工程回归基线。匹配标签仍为单人首轮银标，不能表述为 HR 专家金标，也不能外推为录用概率。
