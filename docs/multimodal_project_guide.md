# 多模态智能助手项目技术说明（GLM + LangChain + RAG）

## 1. 项目定位

这是一个可直接落地的企业级大模型应用项目，目标是构建“知识助手”，支持：

1. 文本知识入库与检索增强问答（RAG）
2. 结果可追溯（返回引用来源）
3. API化对接业务系统

当前工程同时保留了原有“求职助手”链路，形成双场景能力：

- 场景A：求职材料生成（岗位匹配分析、求职信、面试题、投递邮件）
- 场景B：文本知识助手（文本 RAG 问答）

---

## 2. 技术选型与原因

- FastAPI：接口定义清晰、异步友好、易部署。
- LangChain：统一编排Prompt、模型调用与检索流程，降低业务代码复杂度。
- Chroma：本地向量库部署成本低，便于开发与演示。
- GLM（OpenAI-Compatible）：可直接走统一协议接入文本生成模型。
- HashEmbeddings / HuggingFaceEmbeddings：支持“快速演示”和“效果优化”两套模式。

---

## 3. 架构分层

### 3.1 配置层

- 文件：`app/config.py`
- 作用：统一管理模型、路径、向量库集合、API Key。
- 关键点：
  - `llm_model` 指定文本生成模型。
  - `MM_COLLECTION_NAME` 指定知识库集合。

### 3.2 模型接入层

- 文件：`app/llm.py`
- 作用：封装模型初始化。
- 方法：
  - `get_llm()`：文本生成模型

### 3.3 RAG基础层

- 文件：`app/rag.py`
- 作用：文档加载、切分、向量化、检索。
- 关键实现：
  - `load_one_file()`：支持 `pdf/docx/txt/md/csv`
  - `split_documents()`：调用语义切块模块，优先保留简历项目、JD 职责/要求、Markdown 标题路径与 CSV 表头
  - `RAGStore`：封装Chroma操作

### 3.4 知识助手业务层

- 文件：`app/multimodal/service.py`
- 作用：实现文本入库与检索增强问答。
- 核心流程：
  - 文本入库：解析 -> 切分 -> 向量化
  - 问答：问题检索 -> Reranker重排 -> 构造上下文 -> LLM生成 -> 来源引用

### 3.5 API路由层

- 文件：`app/multimodal/routes.py`
- 接口：
  - `POST /api/mm/ingest/text`
  - `POST /api/mm/chat`
  - `POST /api/mm/evaluate`

---

## 4. 端到端流程（可口述）

### 4.1 知识入库流程

1. 用户上传文本。
2. 文本经 Loader 解析为 `Document(page_content + metadata)`。
3. 按文档类型进行结构优先、语义边界优先、长度兜底的切块。
4. 写入 Chroma 向量库。

### 4.2 问答流程

1. 用户输入问题。
2. 按问题从向量库检索 TopK 片段。
3. 将“问题 + 检索上下文 + 历史对话”组装到 Prompt。
4. 调用 GLM 文本模型生成回答。
5. 返回答案与引用来源（filename/modality/source）。

---

## 5. 关键工程能力体现

1. 统一协议接入模型：通过OpenAI-compatible接口快速替换底层模型。
2. 结构化文本知识表示：不同文本格式以统一文档结构进入检索链路。
3. 二阶段检索：向量召回 + Cross-Encoder重排，提升TopK精度。
4. 可解释性设计：回答包含来源引用，利于业务审核。
5. 结构化分层：配置、模型、RAG、业务、路由解耦，便于维护和扩展。

---

## 6. 效果评估指标

1. `HitRate@K`：TopK内是否命中目标来源文档。
2. `MRR@K`：首个命中文档倒数排名，反映命中位置质量。
3. `KeywordRecall@K`：预期关键词在检索上下文中的覆盖率。
4. `CitationHitRate`（可选）：答案是否按要求标注预期来源。

---

## 6. 面试高频追问与回答要点

### Q1: 如何降低幻觉？

A: 三层策略：
1. 强制RAG上下文约束（Prompt中要求仅基于检索内容回答）
2. 返回引用来源（方便核对）
3. 未检索到时明确“不确定”，避免硬编造

### Q2: 如果数据量变大怎么办？

A: 增加三项优化：
1. Hybrid Search（向量+关键词）
2. Reranker重排（如bge-reranker）
3. 分层索引与缓存（热知识缓存、分库分collection）

### Q3: 如何走向生产？

A: 引入鉴权与审计日志、异步任务队列、对象存储、监控告警、灰度发布。

---

## 7. 快速演示步骤

1. 启动服务：`uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`
2. 打开：`http://127.0.0.1:8000` 或 `http://127.0.0.1:8000/docs`
3. 先上传文本到知识库。
4. 发起`/api/mm/chat`问答，观察返回引用来源。

---

## 8. 可扩展路线图

1. 接入Rerank模型提升召回精度。
2. 增加会话记忆管理（短期记忆+长期摘要）。
3. 增加工具调用（查询数据库/工单系统/知识库权限校验）。
4. 用LangGraph实现复杂任务流（规划-执行-反思）。
5. 接入前端会话管理与流式输出（SSE/WebSocket）。

---

## 9. 代码映射索引

- 项目入口: `app/main.py`
- 知识助手路由: `app/multimodal/routes.py`
- 知识助手服务: `app/multimodal/service.py`
- 多模态Prompt: `app/multimodal/prompts.py`
- 多模态Schema: `app/multimodal/schemas.py`
- 模型工厂: `app/llm.py`
- RAG核心: `app/rag.py`
- 全局配置: `app/config.py`
