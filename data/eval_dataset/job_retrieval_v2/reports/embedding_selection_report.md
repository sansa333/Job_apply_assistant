# Job Retrieval V2 Embedding 选型报告

数据质量门禁：通过
标注状态：silver_expert_review_required（不得表述为人工 gold test set）

## 数据规模

- 真实历史岗位：120；岗位族：8
- 原子证据单元：1250；自然查询：480；唯一查询：480
- 多相关查询比例：0.5000；平均相关证据数：1.50
- 固定困难候选池：每个查询 50 个证据单元
- 岗位级 split：{'development': 84, 'test': 36}

## 主评测：精确岗位内 Development 模型对比

该任务与线上流程一致：上游已经解析精确 job_id，只在该 JD 的 8–20 个证据单元内排序。

| 模型 | MRR@3 | HitRate@1 | Recall@3 | Recall@5 | nDCG@5 | P95查询(ms) | 维度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hash_baseline | 0.3973 | 0.2857 | 0.4271 | 0.5878 | 0.4322 | 0.21 | 384 |
| bge_small_zh | 0.6706 | 0.5387 | 0.6994 | 0.8140 | 0.6744 | 11.73 | 512 |
| multilingual_e5_small | 0.6260 | 0.5030 | 0.6518 | 0.8125 | 0.6433 | 20.50 | 384 |
| bge_m3 | - | - | - | - | - | - | - | 失败：OSError |

## Development 集选型证据
按冻结规则选出 `bge_small_zh`。相对 `multilingual_e5_small` 的 MRR@3 配对差值为 0.0446，95% CI [0.0000, 0.0888]，替换门禁：False。

## 主评测：精确岗位内冻结 Test 结果
只运行 Development 集选出的 `bge_small_zh`，未使用 Test 集重新选型。

- MRR@3：0.6424
- HitRate@1：0.5347
- Recall@3：0.6597
- Recall@5：0.8229
- nDCG@5：0.6717

## 压力测试：50 个跨岗位困难候选
- Development：MRR@3=0.3165，HitRate@1=0.2500，Recall@3=0.3080，Recall@5=0.4122，nDCG@5=0.3296。
- Test：MRR@3=0.3576，HitRate@1=0.2917，Recall@3=0.3438，Recall@5=0.4167，nDCG@5=0.3522。

## 结论边界

- 当前 qrels 由确定性查询构造产生，状态为 silver，必须完成双人独立标注和第三人仲裁后才可升级为 gold。
- Embedding 仅在 Development 集选择；Test 集只运行一次选中模型。
- 主指标来自与生产一致的 job_scoped 检索；50 候选 hard_pool 只作为压力测试，不用于包装主效果。
- 本报告衡量检索层，不代表匹配评分、Agent 成功率或生成文本质量。
