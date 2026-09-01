# RAG Improvement Prompt Pack

本目录保存一组面向后续开发 Agent 或大模型协作的提示词模板，用于系统化补强当前 AI Job Apply Assistant 项目的 RAG 能力。

## 推荐使用顺序

1. `01_index_governance_prompt.md`：先修索引治理，避免 embedding 维度不一致、重复入库、无法重建等基础问题。
2. `02_rag_evaluation_prompt.md`：建立评测集与指标体系，让后续优化有可量化基线。
3. `03_retrieval_optimization_prompt.md`：在评测基线稳定后优化召回、重排、过滤和查询改写。
4. `05_security_stability_prompt.md`：补强上传限制、鉴权、审计、异常处理和异步任务。
5. `06_data_quality_prompt.md`：治理中文乱码、样例数据、编码规范和数据清洗流程。

## 使用方式

每个文件都包含：

- 任务目标
- 当前项目背景
- 需要检查的代码位置
- 推荐实施流程
- 验收标准
- 风险点
- 可直接复制使用的完整提示词

建议每次只使用一个提示词文件，完成后运行对应测试或验证，再进入下一个主题。
