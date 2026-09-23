# LangChain 1.x 迁移

迁移日期：2026-09-15。

## 版本与选择

项目原先固定 LangChain 0.3.25；本次迁移到 1.4.0。
此前代码尝试导入 `create_agent` 后回退到 `initialize_agent`，但 0.3.25
未导出 `create_agent`，因此动态 Agent 实际使用的是旧接口。

迁移后的直接依赖：

| 组件 | 版本 |
|---|---|
| langchain | 1.4.0 |
| langchain-core | 1.6.3 |
| langchain-openai | 1.6.2 |
| langchain-community | 0.4.2 |
| langchain-chroma | 0.2.5 |
| langchain-text-splitters | 1.1.2 |
| langchain-huggingface | 1.2.2 |
| pydantic | 2.13.5 |
| pydantic-settings | 2.15.0 |

版本以 `requirements.txt` 为准。独立组件不要求版本号相同。
Chroma 保留 1.0.9；embedding 模型、索引配置与 Torch CPU/GPU 安装方式不变。
本次没有重新构建或迁移用户向量库。

## 实现变化

- 动态 Agent 仅使用 v1 `create_agent`，移除旧 Agent 的异常回退路径。
- 初始化失败返回 `agent_runtime_error`；执行失败返回 `agent_execution_error`，
  不自动重放已可能执行过工具的请求。
- LangGraph 调用设置 `recursion_limit=32`，限制反复调用工具的图执行步数。
  图步数不等于工具调用次数或旧 Agent 的迭代次数。
- 使用统一文本转换函数处理最终消息的文本内容块。
- GLM 模型明确使用 Chat Completions，不自动选择 Responses API。
- 保留意图识别、确定性工作流、候选人范围限制、上下文预算和工具结果存储逻辑。

## 安装和验证

Python >= 3.10。建议在独立虚拟环境安装，保留已有环境以便回退：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

如使用 uv，现有 requirements 的 PyTorch 附加源可能因 uv 的首索引优先策略
导致解析失败；使用 pip，或仅在信任 PyPI 与官方 PyTorch 源时指定
`--index-strategy unsafe-best-match`。

迁移前基线：140 项测试通过。迁移后共 145 项测试通过，`pip check` 和变更代码的
Ruff 检查通过，完整 CPU 依赖声明也通过解析验证。新增真实 v1 图运行的离线测试覆盖
工具执行、跨候选人调用拒绝、初始化失败、重复工具调用上限，以及 GLM
Chat Completions 请求序列化与响应解析。模型响应与 HTTP 层均为离线替身，
并未证明线上智谱服务或真实 embedding 模型在当前凭据和硬件下可用。

## 后续维护

`langchain-community` 0.4.2 发出停止积极维护的弃用提示；项目目前仍使用其中的
PDF、DOCX 和文本加载器。它还会间接安装 `langchain-classic`，但应用 Agent
不再调用 classic 接口。后续可独立迁移文档加载器并验证解析一致性。

## 官方资料

- [v1 迁移指南](https://docs.langchain.com/oss/python/migrate/langchain-v1)
- [版本支持政策](https://docs.langchain.com/oss/python/versioning)
- [发布记录](https://pypi.org/project/langchain/#history)
