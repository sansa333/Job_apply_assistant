# Job Retrieval V2 数据集卡

## 用途

本数据集用于评估求职 Agent 在岗位知识库中的证据检索能力，并支持 Embedding 模型选型。它只衡量
检索层，不衡量岗位匹配分数、Agent 工具规划或生成文本质量。

## 数据规模

- 120 个带真实公司名称、真实岗位标题和来源信息的公开历史 JD 快照。
- 8 个岗位族：数据科学与分析、数据工程、机器学习与 AI 研究、软件与安全工程、金融风险与量化、
  生命科学研究、业务政策与管理、运营与基础设施。
- 1,250 个原子证据单元，每个岗位 8–20 个。
- 480 条自然查询，每个岗位固定包含职责、技术技能、资格背景和工作环境四类问题。
- 720 条 silver 分级相关标注；50% 查询有两个相关证据，平均每个查询 1.5 个。
- 主评测候选为已解析 `job_id` 内的 8–20 个证据单元，与生产流程一致；压力测试固定 50 个候选，
  额外包含同岗位族跨公司困难负例。
- 岗位级 Development/Test 划分为 84/36；同一个 JD 不会跨 split。

## 文件

- `job_snapshots.jsonl`：历史岗位快照、岗位族、来源和 split。
- `evidence_units.jsonl`：可独立判断的原子证据单元。
- `queries.jsonl`：480 条唯一查询。
- `qrels.jsonl`：自动构造的 silver qrels，相关等级为 2 或 3。
- `candidate_pools.jsonl`：每条查询的固定 50 个候选和困难负例类型。
- `annotation_tasks.jsonl`：完整人工双标任务；人工字段初始均为 `null`。该文件体积较大，不纳入
  Git 仓库，可运行构建命令在本地确定性生成。
- `manifest.json`：版本、数量、分布，以及仓库内发布文件的大小与 SHA-256；本地标注任务记录在
  `local_artifacts`，不参与发布文件校验。
- `reports/embedding_selection_report.*`：Development 选型和冻结 Test 结果。

## 标注状态

当前版本为 `2.0.0-silver`。查询和初始 qrels 由确定性规则从真实 JD 证据单元构造，适合工程基线、
数据流程验证和人工标注启动，不得称为人工 gold test set。

只有以下条件全部满足后才能升级：

1. 两名标注者独立完成所有 24,000 个 query–passage 判断。
2. 分歧项经过第三人仲裁。
3. Cohen's Kappa 不低于 0.70。
4. 每条查询至少保留一个等级 2 或 3 的相关片段。
5. `python -m tools.finalize_retrieval_v2_annotations` 成功生成 `qrels_gold.jsonl`。

## 可复现构建

```bash
python -m tools.build_retrieval_v2_dataset
python -m tools.evaluate_retrieval_v2
```

构建过程固定选择公开目录中描述最完整的 120 个岗位，所有排序、split、查询变体、困难负例和文件
序列化均为确定性的。`manifest.json` 用于检测仓库内发布文件是否被修改；本地生成的
`annotation_tasks.jsonl` 不属于发布清单。

Embedding 本体选型在生产一致的 `job_scoped` 候选上执行精确余弦排序，避免 Chroma/ANN 参数干扰
模型质量对比；固定 50 个候选的 `hard_pool` 作为压力测试单独报告。选中模型仍通过原有 Chroma
隔离索引实验验证生产集成。报告记录逐查询 P50/P95 在线计算延迟，不能用批量编码吞吐冒充单请求延迟。

## 许可证与隐私边界

岗位描述为公开历史快照，来源 URL 和数据集字段保留在快照中；它们不表示岗位目前仍开放。数据集中
没有真实候选人简历。不得把私人简历加入此目录，候选人侧评测应使用明确授权或合成数据。
