# 国内招聘检索评测：正式修复与复现

> 历史回归记录。2026-09-21 的当前业务任务、标签和结果已迁移至 [业务评测 v4.1](domestic-business-evaluation.md)。下文旧主题指标与旧搜索实现保留审计，不作为当前主指标；新搜索已支持 BM25、dense 和 RRF。

2026-09-18。本页为正式修复后的结果；[旧基线](domestic-retrieval-evaluation-v2.md)与[根因诊断](domestic-retrieval-diagnosis.md)保留供审计。旧英文 JD 的 MRR@3 0.7614 / Recall@5 0.8750 不适用于国内版本。

## 正式代码已落实

1. `app/evaluation/domestic.py` 用 ASCII 标识符边界处理 Python、RAG、Agent、Java 等主题。`熟悉Python`、`Python等` 不再漏标，`JavaScript` 不误作 Java。
2. `app/chunking.py` 在正式切块出口过滤无正文标题，包括 fallback；保留短的尾部学历等事实。入库 fallback 不再把纯标题重新放回。
3. `app/knowledge/ingestion.py` 写入切块版本、模型、维度、截断长度、归一化、精度、索引状态和岗位正文哈希。打开旧索引不会覆盖其旧身份；不兼容索引拒绝检索，API 返回 503。切换模型或配置须离线重建。
4. 正式 `DomesticJobService.retrieve_job_evidence` 是评测与线上共同入口，先验证精确 job_id，再通过 Chroma 检索。岗位片段确实进入匹配分析提示词。
5. 搜索先提取自然语言中的技术词，经 SQLite 条件过滤后对全部候选排序，最后分页。它不使用 embedding，不能把搜索改进归因于 BGE。

## 数据与统计口径

来源为本地 `data/eval_dataset/domestic_job_search_v1/jobs.jsonl` 既有公开国内快照。290 条原记录按正文去重后保留 **288 条：260 个具体岗位、28 个公告合集**。最终 **1,162 个片段**；旧版 1,204 个片段中的 44 个纯标题删除，短事实保留又产生 2 个有效片段。

固定 15 种主题问题形成 4,320 个“岗位 × 问题”实例，不是 4,320 条独立真实用户提问。另有 528 个岗位搜索实例。标签穷举同岗位片段、由固定主题规则构造，统一标为 `silver_rule_unreviewed`，没有人工金标。

沿用正文哈希分组 development/test。具体岗位 test 含 87 个岗位，82 个岗位贡献 **494 个有答案问题实例**。新修复是在看过旧结果之后实施；这是**修正后的回归评测，不是未见数据盲测**。不能用 Bootstrap 排除自动标注错误、近似重复或同一模板相关性。

MRR@3：首个相关片段位于前三时取其名次倒数，否则为 0。Recall@5：前五个片段中的相关片段数 / 同岗位全部标注相关片段数。两者按有答案问题宏平均；无答案问题单列，不制造分母或假正例。

## 对照结果

相同快照、查询、切块、标签、job_id 过滤、Chroma vector-only 路径；模型分别重建独立索引。CUDA FP16、归一化向量、无查询前缀、不重排。512-token 组控制截断上限，tokenizer 不同意味着字符覆盖量仍不同；不是各模型独立调优后的最优比较。

| 配置 | Test MRR@3 | Test Recall@5 | Development MRR@3 | 被截断片段 |
|---|---:|---:|---:|---:|
| BGE-small-zh-v1.5 / 512 | 0.9116 | 0.9993 | 0.8877 | 116 |
| BGE-M3 / 512 | 0.9349 | 0.9963 | 0.9207 | 18 |
| BGE-M3 / 1024 | 0.9349 | 0.9963 | 0.9207 | 0 |

M3 对 small 的 MRR@3 绝对提升 **0.02328（2.33 个百分点）**，相对提升 **2.55%**；按 JD 组配对、3,000 次 Bootstrap 的 95% CI 为 **[+0.00762, +0.03968]**。Recall@5 绝对下降 **0.00304（0.30 个百分点）**，95% CI **[-0.00803, 0]**。不能写成 MRR 和 Recall 同时提升。

512 与 1024 的本轮指标相同；1024 避免正文截断，但没有证明它进一步提高本轮指标。首次最终修复运行的常驻模型/索引 P95 约 small 9.13 ms、M3-1024 24.04 ms。延迟有负载波动，不包含 HTTP、冷启动、模型生成或并发排队，不能作为上线 SLA。

最终代码复跑所有分组指标一致；该次 P95 为 small 9.06 ms、M3-1024 24.40 ms。完整运行记录均保留，未用较快一次替换全部延迟记录。

## 更严格的检查

### 2026-09-21：候选池难度与无模型基线审计

在完全不改变冻结标签、查询和原模型排名的条件下，额外核算随机排列的解析期望，以及按原始片段顺序返回的无 embedding 基线：

| 排序方法 | MRR@3 | Recall@5 |
|---|---:|---:|
| 同岗位候选均匀随机排列（理论期望） | 0.6527 | 0.9936 |
| 原始片段顺序（不使用模型或查询） | 0.7105 | 0.9934 |
| BGE-small-zh | 0.9116 | 0.9993 |
| BGE-M3 | 0.9349 | 0.9963 |

**当前高 Recall 主要反映候选池过小，不能作为强检索能力的主要证据。** 494 个有答案实例中，475 个（96.15%）候选不超过 5 个，剩余 19 个也只有 6 个。对 N 个候选随机排列，Recall@K 的期望为 min(K,N)/N；N≤5 时取前五个必然覆盖全部相关片段，与排序能力无关。46 个实例甚至全部候选都被主题规则判为相关。

当前具体岗位 test 总计 1,305 个模板实例，其中 811 个按 silver 标签无答案，未纳入 MRR/Recall。这是有条件检索指标的定义，并非计算错误，但不能代表系统处理所有用户问题的质量。候选池先按正确 job_id 限定，也意味着这里没有检验从全库找到正确岗位的能力。

M3 的 MRR 高于原顺序和随机期望，说明在本轮标签下有排序作用；不能把这个作用外推为已通过人工语义金标、真实问题分布或未见数据验证。该评测应定位为**工程可复现的弱标注回归测试，尚非严格的生产效果/泛化评测**。

审计明细：`outputs/domestic_eval_difficulty_audit_20260921.json`。随机数字为解析期望，不是新增的模型运行，也没有改写此前报告。

派生“技术栈关键词”列表具有主题相关性，但通常不足以回答岗位要求。保持检索排名不变，**仅从正例集合中排除这些词表**：

| 配置 | 正文正例 MRR@3 | 正文正例 Recall@5 |
|---|---:|---:|
| BGE-small-zh-v1.5 / 512 | 0.9072 | 1.0000 |
| BGE-M3 / 1024 | 0.9184 | 0.9980 |

此时 MRR 相对差值仅 **+1.23%**，95% CI **[-0.00611, +0.02886]** 跨零，不能证明稳定优于。它仍是主题 silver，不是人工判断“可回答”的金标。

494 个 test 有答案实例中，只有 19 个的同岗位候选数 > 5；其 Recall@5 为 small **0.9825**、M3 **0.9035**。其余短候选池使 Recall 接近饱和。该 19 题仅来自 2 个岗位，不能把它当独立的大样本，也不能隐藏这项回归。高 Recall 不等于招聘匹配准确率。

## 选型结论

质量优先、常驻 GPU 的个人部署可以暂时保留 BGE-M3：开发集与回归集主题 MRR 均更高，1024 设置避免当前语料截断。低延迟、低资源部署保留 small-zh 基线。**M3 不是已经证明全面胜出的模型**；正文正例优势尚不稳定，长候选池 Recall 有下降。下一轮应以人工双人标注、全新岗位和真实查询做决策，不能不断在这组 test 上优化后称独立验证。

当前只使用 dense embedding。BGE-M3 的 sparse/multi-vector 能力不是本项目的已实现功能；国内岗位搜索也不是 embedding 召回。

## 复现与审核文件

冻结数据：`outputs/domestic_eval_corrected_final_20260918/dataset/`。
正式修复首轮：`outputs/domestic_eval_corrected_final_20260918/report.json`、`report.md`。
最终代码复跑：`outputs/domestic_eval_release_20260918/report.json`、`report.md`。

```powershell
# 两模型均已准备到本地，CUDA PyTorch 可用；输出目录必须不存在。
.\.venv\Scripts\python.exe -m tools.evaluate_domestic_retrieval --output outputs/domestic_eval_new
# 重放相同冻结数据，分别重建索引。
.\.venv\Scripts\python.exe -m tools.evaluate_domestic_retrieval --dataset outputs/domestic_eval_corrected_final_20260918/dataset --output outputs/domestic_eval_replay
```

数据 manifest 包含逐文件 SHA-256。工具在运行前写入协议、记录代码哈希，运行结束确认代码未变化；输出模型配置、库版本、GPU、完整排名、逐题指标、分层结果和配对置信区间。不覆盖旧实验，不向生产目录回灌历史岗位。

`annotation_tasks.jsonl` 包含全部同岗位问题–片段对，包括自动判负项。两个独立复核者填 0/1；分歧由第三人仲裁。

```powershell
.\.venv\Scripts\python.exe -m tools.finalize_domestic_annotations --dataset outputs/domestic_eval_corrected_final_20260918/dataset --reviews path/to/completed_reviews.jsonl --output outputs/domestic_reviewed_dataset --reviewer-a reviewer_a --reviewer-b reviewer_b
```

导入器拒绝缺失、重复、未知样本和未仲裁分歧，保留 silver 原件，产生新冻结集并报告一致率/Kappa。不能把自动结果复制成“人工标注”。目前人工复核**未完成**；搜索标签不会因此自动升级。候选人证据召回、匹配分有效性、拒答、生成事实准确率、真实投递转化都不在本轮指标覆盖范围内。
