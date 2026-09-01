# Evidence-Grounded Job Agent v1 评测报告

- 数据集：`evidence_grounded_job_agent_eval` v1.0.0
- 标签状态：Match labels are first-pass annotations and must not be reported as gold until second annotation and adjudication are complete.
- 基线：`deterministic_skill_lexicon_plus_evidence_weighting` / `evidence_weighted_v1`

## 结构化 JD 解析

- 岗位数：8
- Skill Micro-F1：0.7447
- Skill Macro-F1：0.7365
- 结构化切分相对全文词表 F1 变化：-0.0132
- P95 延迟：40.73 ms

## 简历–岗位匹配

- 标注对数：24
- Accuracy：0.6667
- Macro-F1：0.5784
- Spearman：0.8366
- mean nDCG@5：1.0000
- mean MRR（首个 high）：0.6667
- P95 延迟：258.10 ms

### 匹配消融

| 方法 | Macro-F1 | Spearman | mean nDCG@5 |
| --- | ---: | ---: | ---: |
| 技能重合基线 | 0.8492 | 0.7724 | 0.9489 |
| Requirement–Evidence 加权 | 0.5784 | 0.8366 | 1.0000 |

### 开发集校准 / 留出测试

- Development pairs：12；Test pairs：12
- 技能重合基线阈值：medium=21, high=57；Test Macro-F1=0.6250，Spearman=0.8159
- Evidence 加权阈值：medium=3, high=46；Test Macro-F1=0.5841，Spearman=0.8470

## Agent 轨迹

- 场景数：10
- 工具序列准确率：1.0000
- 终态准确率：1.0000
- Next-action 准确率：1.0000

## 生成安全校验

- 场景数：8
- Case Exact Accuracy：1.0000
- Finding F1：1.0000

## 匹配 Bad Cases

- match_001: expected=high, predicted=medium, score=46.17
- match_005: expected=medium, predicted=low, score=17.71
- match_009: expected=high, predicted=medium, score=51.17
- match_014: expected=medium, predicted=low, score=17.35
- match_017: expected=high, predicted=low, score=19.24
- match_021: expected=medium, predicted=low, score=3.17
- match_022: expected=medium, predicted=low, score=18.60
- match_023: expected=medium, predicted=low, score=5.22

## 解释边界

本报告是小规模、可重复的工程回归基线。匹配标签仍为单人首轮银标，不能表述为 HR 专家金标，也不能外推为录用概率。
