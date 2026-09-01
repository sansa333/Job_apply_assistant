# 03 Retrieval Optimization Prompt

## 任务目标

优化当前 RAG 检索链路，从单一向量召回升级为混合检索、查询改写、metadata filter、reranker 策略可配置的检索体系。

## 当前项目背景

当前检索主要使用 Chroma 的 `similarity_search`。多模态问答中先召回 `candidate_k`，再通过 CrossEncoder reranker 或 lexical fallback 重排。

现有问题：

- 缺少 BM25 或关键词召回。
- 中文 query 没有专门分词或关键词处理。
- 缺少 metadata filter。
- query 没有改写、扩展或分类。
- reranker 参数固定，缺少实验对比。

## 需要检查的代码位置

- `app/rag.py`
- `app/multimodal/service.py`
- `app/multimodal/reranker.py`
- `app/config.py`
- `app/multimodal/schemas.py`

## 推荐实施流程

1. 抽象 Retriever 层：
   - vector retriever
   - lexical retriever
   - hybrid retriever
2. 增加 BM25 或轻量关键词召回。
3. 增加 hybrid fusion 策略：
   - weighted score
   - reciprocal rank fusion
4. 增加 metadata filter：
   - modality
   - filename
   - source
   - collection
   - created_at
5. 增加 query rewrite：
   - 简单关键词提取
   - 同义词扩展
   - 多查询扩展
   - 可选 LLM query rewrite
6. 优化 reranker：
   - reranker 是否启用
   - candidate_k
   - rerank_top_n
   - fallback 策略
7. 将检索配置写入响应，便于评测和调试。

## 验收标准

- 支持纯向量、纯关键词、混合检索三种模式。
- 支持按 modality 过滤文本或图片。
- 支持配置 candidate_k、top_k、rerank_top_n。
- 返回结果包含检索策略和候选数量。
- 可以通过评测接口比较不同检索策略效果。

## 风险点

- 混合检索分数不可直接相加，需要归一化或 rank fusion。
- 中文关键词召回如果不分词，效果可能有限。
- query rewrite 可能引入噪声，需要可关闭。
- 检索配置太复杂会影响 API 易用性，需要默认值合理。

## 完整提示词

```text
你是一名 RAG 检索优化专家。请为当前项目优化检索链路。

项目背景：
- 当前项目使用 Chroma similarity_search 做向量召回。
- 多模态问答中使用 candidate_k 召回，再进行 reranker 重排。
- 当前缺少 BM25、metadata filter、query rewrite 和可配置检索策略。

请完成以下任务：
1. 抽象统一 Retriever 层，支持 vector、lexical、hybrid 三种模式。
2. 增加轻量 BM25 或关键词召回能力，适配中文文本。
3. 增加 hybrid fusion 策略，优先考虑 Reciprocal Rank Fusion。
4. 增加 metadata filter，支持 modality、filename、source 等字段。
5. 增加 query rewrite/query expansion 能力，并允许关闭。
6. 优化 reranker 配置，使 candidate_k、rerank_top_n、top_k 可以从请求或配置传入。
7. 在 chat 和 evaluate 响应中返回检索策略、候选数量、重排状态等调试信息。
8. 增加必要测试或验证样例，对比 vector 与 hybrid 的差异。

实现要求：
- 保持当前 /api/mm/chat 默认行为兼容。
- 默认策略可以仍然是 vector + reranker。
- 新增能力必须通过配置或请求参数开启。
- 不要把 query rewrite 强制绑定 LLM，基础版本应支持离线规则。

最后请输出：
1. 检索架构说明。
2. 新增配置项。
3. API 参数变化。
4. 示例请求和响应。
5. 验证方式和评测建议。
```
