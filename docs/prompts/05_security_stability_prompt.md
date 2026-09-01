# 05 Security And Stability Prompt

## 任务目标

补强当前项目的安全性、稳定性和生产可用性，重点覆盖文件上传限制、鉴权、审计日志、prompt injection 防护、异常处理、异步任务和服务观测。

## 当前项目背景

当前项目以本地演示和工程展示为主，核心接口包括：

- `/api/ingest/profile`
- `/api/ingest/job`
- `/api/analyze`
- `/api/one-click-apply`
- `/api/agent`
- `/api/mm/ingest/text`
- `/api/mm/chat`
- `/api/mm/evaluate`

当前缺少生产级安全控制，例如上传大小限制、用户鉴权、权限隔离、审计日志和 prompt injection 处理。

## 需要检查的代码位置

- `app/main.py`
- `app/config.py`
- `app/services/document_service.py`
- `app/multimodal/routes.py`
- `app/multimodal/service.py`
- `app/agent/job_agent.py`

## 推荐实施流程

1. 增加上传安全限制：
   - 文件大小
   - 文件类型白名单
   - 文件数量
   - 文件名清洗
2. 增加基础鉴权：
   - API Key header
   - 可配置开关
   - 本地开发可关闭
3. 增加审计日志：
   - 请求时间
   - 接口路径
   - 用户标识
   - 文件名
   - collection
   - 操作结果
4. 增加 prompt injection 防护：
   - 对入库文档标记为 untrusted context
   - system prompt 明确禁止执行文档中的指令
   - 对高风险文本做简单检测
5. 增强异常处理：
   - 统一错误响应
   - 日志记录 traceback
   - 前端展示可读错误
6. 增加异步任务设计：
   - 图片解析
   - 批量文档入库
   - 索引重建
7. 增加健康检查与观测：
   - 模型配置
   - embedding 配置
   - reranker 状态
   - collection 状态

## 验收标准

- 上传超大文件会被拒绝并返回明确错误。
- 非法文件类型会被拒绝。
- 可通过配置开启 API Key 鉴权。
- 关键操作有审计日志。
- Prompt 中明确区分系统指令和检索上下文。
- 异常不会暴露敏感 API Key 或完整内部路径。

## 风险点

- 鉴权不能影响本地演示体验，需要配置开关。
- 文件大小检查不能只依赖前端。
- 审计日志不能记录完整敏感内容。
- prompt injection 防护不能完全靠关键词，应结合 prompt 边界设计。

## 完整提示词

```text
你是一名后端安全与稳定性工程师。请为当前 FastAPI + RAG 项目补充生产可用性能力。

项目背景：
- 当前项目用于求职助手和多模态 RAG。
- 当前接口可以上传文本和图片，并调用 LLM 生成答案。
- 当前缺少上传限制、鉴权、审计日志、prompt injection 防护和统一异常处理。

请完成以下任务：
1. 增加文件上传限制，包括大小、类型、数量和文件名安全处理。
2. 增加可配置 API Key 鉴权，本地开发环境可以关闭。
3. 增加审计日志，记录关键操作但不泄露敏感内容。
4. 增强 prompt injection 防护，在 prompt 中明确检索上下文是不可信资料，不能覆盖系统指令。
5. 增加统一异常处理，返回稳定、清晰、脱敏的错误响应。
6. 设计异步任务方案，用于图片解析、批量入库和索引重建；如果不实现队列，也要预留接口边界。
7. 扩展 /health，返回模型、embedding、reranker、collection 的安全状态摘要。
8. 补充必要测试或验证步骤。

实现要求：
- 不要把 API Key 写死在代码中。
- 不要在错误响应中返回完整密钥、完整 traceback 或敏感内容。
- 不破坏现有本地演示流程。
- 所有安全限制应有合理默认值，并可通过 .env 配置。

最后请输出：
1. 新增配置项。
2. 安全策略说明。
3. 修改文件列表。
4. 错误响应示例。
5. 验证步骤。
```
