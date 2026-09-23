# 国内秋招检索评测与复现

> 已归档：这是正式修复前的 v2 基线，当前实现与数字以 [v3 正式修复评测](domestic-retrieval-evaluation.md) 为准。

> 后续诊断更新：本页保留原始基线。已发现中英文边界漏标和纯标题切块干扰，
> 原始模型排序不能直接用于选型；控制变量对照与具体案例见
> [差异原因诊断](domestic-retrieval-diagnosis.md)。事后修正结果不替代独立测试。

## 已补齐的应用链路

- `DomesticJobService.search`：自然语言技术词提取、真实 SQL 条件过滤、默认或候选人技能排序、最后分页。原先 SQL 先分页再排序的问题已修复。
- `DomesticJobService.retrieve_job_evidence`：精确 `job_id` 校验，通过生产 `JobHybridRetriever` 和 Chroma 返回可追溯的片段。HTTP 入口为 `/api/domestic/jobs/{job_id}/evidence`。
- `DomesticJobService.analyze_job`：岗位证据实际进入模型提示词；保留 JD 与简历全文以保证确定性评分和全文核验。没有把简历全文分析改成片段召回评分。
- 评测直接调用这些服务方法，而非重新实现一个只在评测时使用的排名器。HTTP 参数校验另外用测试验证；指标不包含网络或 LLM 生成。

## 本轮数据

使用本地 `data/eval_dataset/domestic_job_search_v1/jobs.jsonl` 的公开国内快照。本轮没有重新联网同步；运行时生产岗位目录为空，因此没有向生产目录回灌历史数据。

290 条原记录按去空白后的正文去重，保留 288 条：260 个具体岗位、28 个公告合集。实际生产切块得到 1,204 个片段。生成 4,320 个岗位–问题实例，但只有 **15 种预设主题问题正文**，不能说成 4,320 条独立真实用户查询。另有 528 个搜索测试实例，覆盖技术词、公司、城市/届别筛选和少量自然语言改写。

评测标签由固定主题正则生成，穷举同岗位所有片段，未使用被测模型排名生成标签。它衡量“主题词对应的证据检索”，不等于人工确认能充分回答问题；正文中的背景介绍、优先条件或泛化措辞也可能被误标。标注状态统一为 `silver_rule_unreviewed`。无答案问题不计入常规 MRR/Recall；单独记录覆盖率，未宣称已有拒答能力。

岗位按归一化正文哈希分组划分 development/test，同正文不跨集合；近似重复 JD 和相同问题模板仍存在，未声称完全独立。具体岗位 test 有 87 个岗位、469 个有答案问题实例。模型与配置在运行前固定，没有针对测试结果调参。

旧快照缺少部分来源类型、抓取时间和建库时间：评测目录使用 `open_source` 兼容现有 SQLite 约束，以公告标题/来源数据集区分公告，缺失创建时间使用固定占位值以稳定排序。不能把占位时间当真实抓取时间。

## 模型对比协议

1. 冻结数据、查询、切块与标签，写入 SHA-256 清单；每个模型启动和完成后校验。
2. 模型各自建立全新的 SQLite 与 Chroma 目录，不共享不同维度向量，不改动生产索引。
3. 基线 `BAAI/bge-small-zh-v1.5` 为 512 维、512-token 上限；BGE-M3 为 1024 维，分别测 512 和 1024-token 上限。
4. 三组均 CUDA、FP16、归一化向量、相同生产切块、vector-only、不重排。模型 tokenizers 不同，相同 token 上限并不意味着相同字符范围。
5. 保持生产 embedding adapter 的无查询前缀行为。这里比较现有 adapter 下替换模型的效果，不是每个模型都单独调优后的最高性能。
6. 每题调用真实 `retrieve_job_evidence`，验证没有跨岗位返回；另做公司+岗位标题搜索进入前十之后再检索的串联审计。
7. 输出逐题排名、完整 qrels、分层指标、模型配置、截断数量、加载/建索引时间、常驻索引查询延迟和代码/包版本。
8. 用同一 query ID 做配对差值，并按同 JD 组进行 3,000 次 Bootstrap；开发集和测试集分别报告。

MRR@K 取首个相关结果排名的倒数，超出 K 为 0；Recall@K 为前 K 个结果找到的相关片段数除以该题全部标注相关片段数，再进行题目宏平均。绝对变化 = M3 − small，相对变化 = 绝对变化 / small。无相关片段时不把 Recall 强制写成 0。

多数 JD 只有 3–5 个片段，因此额外报告：至少一个相关和一个不相关片段的子集，以及候选片段数 > 5 的子集。未人为把生产切块拆得更细来制造提升空间。

## 本轮结论

具体岗位 Test 的 BGE-small-zh：MRR@3 **0.8429**，Recall@5 **0.9951**；BGE-M3 两个上下文配置都是 **0.8355 / 0.9878**。

- MRR@3 绝对变化约 **−0.0075**，相对 **−0.89%**；配对 95% CI 约 **[−0.0271, +0.0124]**，不足以证明稳定差异。
- Recall@5 绝对变化约 **−0.0074**，相对 **−0.74%**；95% CI 约 **[−0.0157, −0.0005]**。只说明当前 silver 标签下的差异，不包含标注误差。
- 候选数 > 5 的 56 个有答案 test 实例上，small 的 MRR@3/Recall@5 为 **0.7857/0.9592**，M3 为 **0.7262/0.8976**。

因此本轮不能写“升级 BGE-M3 提升了国内秋招检索效果”。代码保留现有本地模型配置，没有根据弱标注单次实验自动替换生产模型。旧英文数据的 0.7614/0.8750 不能移用。

完整本地报告：`outputs/domestic_eval_20260918_final/report.md`；逐题数据与置信区间：同目录 `report.json`。最终运行复用 `outputs/domestic_eval_20260918_v3/dataset` 的冻结数据，保留首次运行记录作为审计。模型结论没有经过有利结果挑选。

搜索接口的自然语言词提取根据搜索开发问题修正，其前后结果属于回归测试，不属于未见搜索测试集上的泛化证明，也不归因于更换 embedding。

在相同的 111 个有答案具体岗位搜索实例上，仅查询词提取修复前后：MRR@10
**0.8076 → 0.9044**，Recall@50 **0.7414 → 0.8100**。这两个结果均已经采用先全量排序后分页，
不能将该差值解释成分页修复或 BGE-M3 的独立贡献。公告查询单独报告，未混入上述数字。

## 复现

需要本机准备两个模型及 CUDA PyTorch。首次从国内快照建立数据并完整运行：

```powershell
.\.venv\Scripts\python.exe -m tools.evaluate_domestic_retrieval --output outputs/domestic_eval_new
```

复用已冻结数据（不会重新构造标签）：

```powershell
.\.venv\Scripts\python.exe -m tools.evaluate_domestic_retrieval --dataset outputs/domestic_eval_20260918_v3/dataset --output outputs/domestic_eval_replay
```

每次输出必须为新目录，脚本拒绝覆盖已有实验目录和数据库。可用 `--small-model`、`--m3-model` 指定本地模型路径。所有模型加载默认 `local_files_only=True`，不会隐式下载。完整结果、快照与索引保存在本地 outputs，不进入版本库。

## 人工复核入口

冻结数据中的 `annotation_tasks.jsonl` 包含每个问题与所有同岗位片段的完整判断任务，包含自动判负项。复核者填写 `annotator_1`、`annotator_2` 为 0 或 1；分歧需要第三人填写 `adjudicated_grade` 与 `adjudicator`。不要把自动标签直接复制为人工标签。

```powershell
.\.venv\Scripts\python.exe -m tools.finalize_domestic_annotations --dataset outputs/domestic_eval_20260918_v3/dataset --reviews path/to/completed_reviews.jsonl --output outputs/domestic_reviewed_dataset --reviewer-a reviewer_a --reviewer-b reviewer_b
.\.venv\Scripts\python.exe -m tools.evaluate_domestic_retrieval --dataset outputs/domestic_reviewed_dataset --output outputs/domestic_reviewed_results
```

导入器拒绝缺失、重复、未知样本和未仲裁的分歧，保留原始 silver 文件，建立新的冻结清单并报告一致率/Kappa。它不自动完成人工标注，也不能验证填写者身份。该入口只升级证据检索标签，搜索标签仍为 silver；LLM 生成质量、候选人证据检索与实际求职收益需要另行评测。

## 验证记录

2026-09-18：全量 `pytest -q` **152 passed**；本次改动文件的 Ruff 检查通过。
覆盖超过 200 个候选的先排序后分页、自然语言查询、MRR 截断与 Recall 分母、无答案排除、
模型配对与负提升、冻结数据防篡改、人工复核导入、真实 Chroma 岗位隔离、HTTP 参数验证、
检索证据进入分析提示词。两次隔离模型运行的具体岗位 test MRR@3/Recall@5 完全一致。
延迟受本机运行负载影响，仅作为本次工程记录，不作为严格性能基准；有一条已有的
`langchain-community` 弃用警告，未影响测试通过。
