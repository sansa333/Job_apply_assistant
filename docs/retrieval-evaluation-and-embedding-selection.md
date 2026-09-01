# 检索评测与 Embedding 选型

## 1. 评测层级

检索评测只评价“给定查询能否找到正确证据片段”，不评价岗位匹配分数、生成文本质量或 Agent
工具调用是否正确。项目将指标分为四层，禁止互相替代：

1. 检索层：HitRate@K、Recall@K、MRR@K。
2. 匹配层：Macro-F1、Spearman、nDCG、匹配分数校准。
3. Agent 层：工具序列、终止状态、人工确认边界。
4. 生成层：事实支持率、无依据陈述率、格式通过率。

## 2. 指标公式

对于查询 `q`，相关片段集合为 `R_q`，Top-K 结果为 `T_q(K)`：

- `HitRate@K = mean(1[|R_q ∩ T_q(K)| > 0])`
- `Recall@K = mean(|R_q ∩ T_q(K)| / |R_q|)`
- `MRR@K = mean(1 / first_relevant_rank_q)`；首个相关片段不在 K 内时记 0。

HitRate 关注“至少找到一条”，Recall 关注“是否找全”，MRR 关注“第一条正确证据是否靠前”。
当每个查询只有一个相关片段时，Recall@K 会退化为 HitRate@K，必须在报告中披露。

## 3. Embedding 选型流程

所有模型必须使用完全相同的文档快照、切块结果、查询、相关性标注、Top-K 和候选池。V2 的
Embedding 本体选型在已经解析精确 `job_id` 的岗位内候选上执行精确余弦排序，以匹配生产流程并排除
ANN 参数影响；固定 50 个跨岗位困难候选的结果作为压力测试单独报告。选出候选模型后，再使用独立
Chroma collection 进行生产集成验证，不能复用其他维度的索引。

1. 运行 `vector-only`，比较 Embedding 本体。
2. 按 MRR@3、Recall@5、HitRate@1、逐查询 P95 延迟依次排序。
3. 对第一名与第二名按相同 query ID 做配对 Bootstrap，报告 MRR@3 差值的 95% 置信区间。
4. 置信区间跨 0 时，不声称效果显著；优先选择延迟更低、索引更小的模型。
5. 只对领先模型加入 BM25/RRF，验证词法召回的增益。
6. 最后加入 Cross-Encoder，验证生产链路，但不得用重排后的结果反推 Embedding 更优。

运行命令：

```bash
python -m tools.compare_job_retrieval_models
python -m tools.build_job_rag_eval
```

## 4. 当前可复现实验

当前旧基线包含 80 条查询，实际只有 7 个唯一问句；每条查询只有一个相关 chunk，岗位内平均候选池
为 2.675，且全部候选池不超过 K=5。因此它可以做代码与索引回归，但 Recall@5、HitRate@5 会饱和，
不能代表大规模知识库效果。

纯向量结果：

| 模型 | MRR@3 | HitRate@1 | Recall@3 | Recall@5 | 平均查询延迟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hash baseline | 0.5312 | 0.1875 | 0.9375 | 1.0000 | 2.44 ms |
| BAAI/bge-small-zh-v1.5 | 0.8125 | 0.6625 | 0.9875 | 1.0000 | 21.37 ms |
| intfloat/multilingual-e5-small | 0.6625 | 0.4250 | 0.9375 | 1.0000 | 29.49 ms |
| BAAI/bge-m3 | 未完成 | 未完成 | 未完成 | 未完成 | 本地权重不可用 |

在这份工程基线上，`BAAI/bge-small-zh-v1.5` 相对
`intfloat/multilingual-e5-small` 的配对 MRR@3 差值为 0.1500，Bootstrap 95% CI 为
`[0.0646, 0.2313]`，查询级胜/平/负为 39/30/11。该证据支持当前工程配置选择，但由于查询重复、
单相关标注和候选池过小，不能作为最终论文式结论。

加入 BM25/RRF 与 Cross-Encoder 后，领先模型的 MRR@3 从 0.8125 变为 0.8208，平均查询耗时
约 1.295 秒。增益较小而延迟显著增加，线上是否默认启用重排应根据延迟预算和更完整 V2 评测决定。

完整机器可读结果见：

- `data/eval_dataset/job_rag/model_experiments/report.json`
- `data/eval_dataset/job_rag/model_experiments/report.md`
- 完整 JSON 报告由评测脚本在本地生成；版本库只保留精简的 Markdown 指标说明。

## 5. V2 检索评测集验收条件

下一版不能再由少数固定模板自动复制，至少满足：

- 120 个真实历史岗位快照，覆盖 8 个岗位族。
- 每个岗位 4 个独立自然问题，共不少于 480 个 query；相同规范问法不得直接复制。
- 每个岗位切分为 8–20 个可独立判断的职责、技能、学历、经验、工作模式和福利证据单元。
- 至少 40% 查询具有 2–4 个相关片段，使 Recall@K 不再退化为 HitRate@K。
- 每个查询加入同岗位相邻片段、同技能不同语境片段、同岗位族其他公司片段作为困难负例。
- 两名标注者独立标注 `0=不相关、1=弱相关、2=相关、3=直接回答`，第三人仲裁分歧。
- 相关集合定义为等级 2 或 3；同时保留等级用于后续 nDCG@K。
- 按岗位隔离开发集与测试集，禁止同一 JD 的改写进入两个 split。
- 冻结测试集后再选模型；若继续调整模型或切块，只能在开发集上操作。

V2 模型替换门槛建议为：测试集 MRR@3 提升的配对 Bootstrap 95% CI 下界大于 0，Recall@5
不下降超过 0.01，P95 查询延迟不超过线上预算，并通过跨岗位数据泄漏检查。若不满足这些条件，保留现有模型。

## 6. 简历表述边界

当前可以表述为：

> 建立独立的向量检索模型评测流程，在相同索引、查询和相关性标注下比较 Hash、BGE 与 E5；使用
> HitRate@K、Recall@K、MRR@K、查询延迟及配对 Bootstrap 置信区间完成 Embedding 选型，并将
> BM25/RRF 与 Cross-Encoder 作为后验消融环节。

在 V2 gold test set 完成前，不建议把 Recall@5=1.0、HitRate@5=1.0 写成项目总体性能。

## 7. V2 实现结果

`data/eval_dataset/job_retrieval_v2` 已按上述标准实现：120 个历史岗位、8 个岗位族、1,250 个原子
证据单元、480 条唯一问题、720 条 silver qrels，50% 查询具有两个相关证据，每条查询使用固定的
50 个困难候选。岗位级 Development/Test 划分为 84/36，质量门禁和 manifest SHA-256 校验均通过。

Development 集纯向量结果：

| 模型 | MRR@3 | HitRate@1 | Recall@3 | Recall@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hash baseline | 0.3973 | 0.2857 | 0.4271 | 0.5878 | 0.4322 |
| BAAI/bge-small-zh-v1.5 | **0.6706** | **0.5387** | **0.6994** | **0.8140** | **0.6744** |
| intfloat/multilingual-e5-small | 0.6260 | 0.5030 | 0.6518 | 0.8125 | 0.6433 |

BGE-small-zh 相对第二名 multilingual-E5-small 的配对 MRR@3 差值为 0.0446，Bootstrap 95% CI 为
`[0.0000, 0.0888]`，未通过要求下界严格大于 0 的替换门禁；当前继续使用 BGE 是因为其绝对指标、
P95 延迟均不差于 E5，而不是声称二者已有显著差异。人工 gold 完成后需要重新确认。
随后只在冻结 Test 集运行选中的 BGE-small-zh：MRR@3=0.6424、HitRate@1=0.5347、
Recall@3=0.6597、Recall@5=0.8229、nDCG@5=0.6717。

50 候选压力测试仍完整保留：BGE-small-zh 在 Development 上 MRR@3=0.3165、Recall@5=0.4122；
冻结 Test 上 MRR@3=0.3576、Recall@5=0.4167。主评测回答“精确岗位内能否找到证据”，压力测试
回答“混入大量相似岗位证据后是否仍能区分”，两种指标不得混写。

这些结果仍属于 silver qrels 基线。`annotation_tasks.jsonl` 中的 24,000 个人工判断字段全部为空；
只有完成双标、第三人仲裁并通过 Kappa 门禁后，`tools.finalize_retrieval_v2_annotations` 才会生成
`qrels_gold.jsonl`。
