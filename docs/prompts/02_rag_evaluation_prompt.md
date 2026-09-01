# 02 RAG Evaluation Prompt

## 任务目标

为当前 RAG 系统补充更完整的评测体系，从“只评估检索命中”升级为“检索质量 + 答案质量 + 引用质量 + 性能成本”的综合评估。

## 当前项目背景

当前已有 `/api/mm/evaluate`，支持：

- `HitRate@K`
- `MRR@K`
- `KeywordRecall@K`
- 可选 `CitationHitRate`

当前评估主要集中在检索层，尚缺少答案层质量评估、评测集管理、历史结果保存和报告生成。

## 需要检查的代码位置

- `app/multimodal/service.py`
- `app/multimodal/schemas.py`
- `app/multimodal/routes.py`
- `streamlit_app.py`
- `docs/multimodal_project_guide.md`

## 推荐实施流程

1. 定义标准评测样本格式：
   - query
   - expected_sources
   - expected_keywords
   - reference_answer
   - required_facts
   - forbidden_facts
   - difficulty
   - category
2. 增加检索层指标：
   - Recall@K
   - Precision@K
   - nDCG@K
   - Context Keyword Coverage
3. 增加答案层指标：
   - Answer Relevance
   - Faithfulness
   - Completeness
   - Citation Accuracy
   - Hallucination Check
4. 支持 LLM-as-judge：
   - 使用固定 judge prompt
   - 输出结构化 JSON
   - 保存评分理由
5. 保存每次评测结果：
   - 时间
   - embedding 配置
   - reranker 配置
   - top_k / candidate_k / rerank_top_n
   - 每条样本结果
   - 聚合指标
6. 生成 Markdown 评测报告。
7. 在 Streamlit 评估看板中展示核心指标。

## 验收标准

- 可以加载一组固定评测样本运行评测。
- 可以对比 baseline 和 reranker。
- 可以输出检索层和答案层指标。
- 每次评测结果可保存、可追溯。
- 评测报告能说明当前 RAG 的优势、短板和改进建议。

## 风险点

- LLM-as-judge 本身可能不稳定，需要固定 prompt 和温度。
- 小样本评测不能代表真实效果，需要标注样本规模。
- 答案引用格式必须和 prompt 约束一致，否则 citation 检查会失真。
- 评估代码不要强依赖外部模型，否则离线环境无法运行基础指标。

## 完整提示词

```text
你是一名 RAG 评测体系专家。请为当前项目扩展 RAG 评估能力。

项目背景：
- 当前项目已有 /api/mm/evaluate。
- 现有指标包括 HitRate@K、MRR@K、KeywordRecall@K、CitationHitRate。
- 现有评测主要检查检索结果，尚未完整评估答案质量。

请完成以下任务：
1. 设计标准评测样本 schema，支持 query、expected_sources、expected_keywords、reference_answer、required_facts、forbidden_facts、category、difficulty。
2. 扩展检索层评估指标：Recall@K、Precision@K、nDCG@K、Context Keyword Coverage。
3. 扩展答案层评估指标：Answer Relevance、Faithfulness、Completeness、Citation Accuracy、Hallucination Check。
4. 增加可选 LLM-as-judge 评估流程，要求 judge 输出结构化 JSON。
5. 保存每次评测结果到 outputs 或 data/evaluations 目录。
6. 生成 Markdown 评测报告，包含配置、总体指标、逐样本结果、失败案例和改进建议。
7. 保留 baseline vs reranker 对比能力。
8. 在 Streamlit 评估看板中展示新增核心指标。

实现要求：
- 基础检索指标必须不依赖 LLM。
- LLM-as-judge 必须可选。
- 评测结果必须包含运行配置，便于复现。
- 不要破坏当前 /api/mm/evaluate 的基础用法。
- 对空样本、无 expected_sources、无 reference_answer 等情况要有明确处理。

最后请输出：
1. 新评测 schema。
2. 新增指标定义。
3. API 入参和出参示例。
4. 评测报告样例结构。
5. 验证步骤。
```
