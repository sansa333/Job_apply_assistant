# 01 Index Governance Prompt

## 任务目标

为当前 RAG 系统补充索引治理能力，解决 embedding 后端切换、向量维度不一致、重复入库、无法重建索引、无法删除或更新文档等问题。

## 当前项目背景

项目使用 FastAPI、LangChain、Chroma 构建 RAG。当前存在三个 collection：

- `profile`
- `job_description`
- `multimodal_knowledge`

默认 embedding 后端支持：

- `hash`：`HashEmbeddings(dim=384)`
- `huggingface`：默认 `BAAI/bge-small-zh-v1.5`

需要重点注意：如果 Chroma collection 已经由某个 embedding 模型创建，再切换 embedding 维度后继续查询或写入，可能出现维度不一致错误。

## 需要检查的代码位置

- `app/config.py`
- `app/rag.py`
- `app/services/document_service.py`
- `app/multimodal/service.py`
- `app/main.py`
- `app/multimodal/routes.py`

## 推荐实施流程

1. 梳理当前所有 collection 的创建、写入、查询入口。
2. 为每个 collection 增加索引元信息记录，至少包含：
   - collection name
   - embedding backend
   - embedding model
   - embedding dimension
   - created_at
   - updated_at
   - document count
3. 在系统启动或首次访问 collection 时校验 embedding 维度是否一致。
4. 如果维度不一致，返回明确错误，不允许静默失败。
5. 增加重建索引能力：
   - 重建 `profile`
   - 重建 `job_description`
   - 重建 `multimodal_knowledge`
6. 增加文档去重策略：
   - 基于文件路径、文件名、内容 hash 或业务 doc_id 去重
   - 避免同一文件多次上传产生重复 chunk
7. 增加文档删除和更新能力：
   - 删除某个文件对应的 chunks
   - 更新时先删除旧 chunks，再重新入库
8. 为 API 返回清晰状态：
   - collection name
   - chunks added
   - chunks deleted
   - embedding backend
   - index status

## 验收标准

- 切换 embedding 后，系统能明确提示需要重建索引。
- 支持一键重建至少三个 collection。
- 同一文件重复上传不会无限重复污染索引。
- 可以按文件名或 doc_id 删除已入库文档。
- 每个 collection 能查询到索引元信息。
- 所有新增接口有清晰错误信息。

## 风险点

- 直接删除 Chroma 目录可能导致运行中对象状态不一致。
- embedding 元信息不能只存在内存中，需要持久化。
- 如果历史数据没有 doc_id，需要兼容旧 metadata。
- 重建索引前需要确认源文件仍然存在。

## 完整提示词

```text
你是一名资深 Python / FastAPI / LangChain 工程师。请基于当前项目实现 RAG 索引治理能力。

项目背景：
- 项目使用 FastAPI + LangChain + Chroma。
- RAG 核心在 app/rag.py。
- 多模态 RAG 在 app/multimodal/service.py。
- 当前 collection 包括 profile、job_description、multimodal_knowledge。
- embedding 后端支持 hash 和 huggingface。
- 当前存在潜在问题：切换 embedding 后端或模型后，Chroma collection 可能出现向量维度不一致。

请完成以下任务：
1. 梳理当前索引创建、入库、查询流程。
2. 增加索引元信息持久化能力，记录 collection、embedding backend、embedding model、dimension、创建时间、更新时间、文档数量。
3. 在 collection 初始化或查询前校验 embedding 配置是否与索引元信息一致。
4. 如果发现维度或模型不一致，不要继续查询，返回明确错误，提示用户重建索引。
5. 增加重建索引能力，至少支持 profile、job_description、multimodal_knowledge。
6. 增加文档去重能力，避免同一文件重复上传导致重复 chunks。
7. 增加按文件或 doc_id 删除文档的能力。
8. 增加必要的 API 路由和响应 schema。
9. 补充最小必要测试或验证脚本。

实现要求：
- 遵循现有项目结构，不做无关重构。
- 错误信息要面向使用者清晰可读。
- 不要破坏现有 /api/ingest/profile、/api/ingest/job、/api/mm/ingest/text、/api/mm/chat 流程。
- 所有新增能力要能通过 Swagger 或 Streamlit 后续接入。

最后请输出：
1. 修改文件列表。
2. 新增接口说明。
3. 索引元信息结构。
4. 验证步骤。
5. 尚未覆盖的风险。
```
