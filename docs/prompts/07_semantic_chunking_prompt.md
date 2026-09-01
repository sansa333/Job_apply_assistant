# 07 Semantic Chunking Prompt

## 任务目标

将当前 RAG 系统的切块策略从“以字符长度为主的递归切分”升级为“文档结构优先、语义边界优先、token 长度兜底”的语义切块策略，尽量避免破坏简历、JD、Markdown、PDF、DOCX、CSV 和图片解析结果中的完整语义单元。

## 当前项目背景

当前项目的 RAG 切块逻辑位于 `app/rag.py`：

```python
RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", "。", "，", ";", ".", " ", ""],
)
```

该策略比简单硬切更好，但仍然主要由字符长度驱动。它无法充分理解不同文档类型中的自然结构，例如：

- 简历中的项目经历、技能栈、教育经历
- JD 中的岗位职责、任职要求、加分项
- Markdown 标题层级
- PDF/DOCX 段落
- CSV 表头与数据行
- 图片解析结果中的 OCR、摘要、关键词、图表信息

因此，当前切块可能导致一个完整项目经历被拆断、JD 职责和要求混切、图片 OCR 和语义摘要被切散，从而影响召回质量和答案可信度。

## 总体设计原则

推荐采用三阶段切块策略：

1. 结构优先
   先根据文档天然结构切分，例如标题、段落、列表、表格行、简历模块、JD 模块、图片解析字段。

2. 语义边界优先
   尽量保持一个 chunk 内包含完整语义单元，例如一个项目经历、一组岗位要求、一段完整 OCR 结果。

3. token 预算兜底
   当某个语义块过长时，再按句子或子段落进行二次切分，并保留适度 overlap。

目标不是绝对不切断任何内容，而是尽量避免在最关键的语义边界上切断内容。

## 推荐切块策略

### 1. 简历和候选人资料

推荐按照简历模块切块：

- 基本信息
- 求职意向
- 教育经历
- 技能栈
- 项目经历
- 实习经历
- 工作经历
- 竞赛/论文/证书
- 自我评价

重点要求：

- 一个项目经历优先保持为一个 chunk。
- 一个实习或工作经历优先保持为一个 chunk。
- 技能栈可以作为独立 chunk。
- 如果项目经历太长，再按“背景/职责/技术栈/成果”二级切分。

推荐 metadata：

```json
{
  "doc_type": "resume",
  "section": "project_experience",
  "section_title": "企业知识库 RAG 项目",
  "chunk_strategy": "resume_structural",
  "chunk_index": 3,
  "parent_doc_id": "..."
}
```

### 2. 岗位 JD

推荐按照招聘语义结构切块：

- 岗位名称
- 公司/业务背景
- 岗位职责
- 任职要求
- 加分项
- 技术栈关键词
- 地点/学历/经验/薪资等硬条件

重点要求：

- 职责和要求不要混切。
- 每条职责或要求较短时，可以合并成一组 chunk。
- 技术栈关键词可以作为独立增强 chunk。
- 硬条件应单独保留，便于匹配分析。

推荐 metadata：

```json
{
  "doc_type": "job_description",
  "section": "requirements",
  "chunk_strategy": "jd_structural",
  "chunk_index": 2,
  "parent_doc_id": "..."
}
```

### 3. Markdown 文档

推荐按标题层级切块：

- 优先识别 `#`、`##`、`###`
- 每个标题段落形成一个语义块
- 保留标题路径，例如 `项目说明 > RAG 架构 > 检索流程`
- 超长标题段落再按段落和句子拆分

推荐 metadata：

```json
{
  "doc_type": "markdown",
  "section_path": "多模态RAG架构说明 / 问答阶段",
  "heading_level": 3,
  "chunk_strategy": "markdown_heading",
  "chunk_index": 5
}
```

### 4. PDF 和 DOCX

推荐流程：

1. 先使用现有 loader 解析文本。
2. 以页、段落、标题样式或明显编号为候选边界。
3. 尽量按段落组合成语义块。
4. 遇到长段落再按中文句号、分号、换行拆分。

重点要求：

- 不要把一句话中间切断。
- 尽量保留页码或段落序号。
- 如果 loader 能提供 page metadata，应保留到 chunk metadata。

### 5. CSV

CSV 不建议直接按字符切。推荐按表格语义处理：

- 保留表头。
- 一行或多行数据组成一个 chunk。
- 每个 chunk 中都带上表头，避免检索到孤立数据。
- 如果 CSV 是日志或记录表，可以按固定行数或业务主键分组。

示例 chunk：

```text
表名: applications.csv
字段: application_id, company_name, job_title, candidate_name, status, note
记录:
- 20260427_205203, 某某科技有限公司, 大模型应用开发工程师, 张三, generated_not_submitted, ...
```

## 推荐实现流程

1. 新增文档类型识别函数。
   - 根据 collection、文件路径、文件名、后缀、内容特征判断 doc_type。

2. 新增结构化切块模块。
   - 可放在 `app/chunking.py` 或 `app/rag_chunking.py`。
   - 避免把所有逻辑堆在 `app/rag.py`。

3. 替换当前 `split_documents()`。
   - 保留原有 `RecursiveCharacterTextSplitter` 作为 fallback。
   - 新增 `split_documents_semantic()`。

4. 增加 token-aware 兜底切分。
   - 优先使用 tokenizer 估算 token。
   - 如果暂时不引入 tokenizer，可先使用字符长度近似，但只作为最后兜底。

5. 增强 metadata。
   - `doc_type`
   - `section`
   - `section_title`
   - `section_path`
   - `chunk_strategy`
   - `chunk_index`
   - `parent_doc_id`
   - `source`
   - `filename`
   - `modality`

6. 增加评估样例。
   - 简历项目经历不应被切碎。
   - JD 职责和要求应分开。
   - Markdown 标题路径应保留。
   - CSV chunk 应包含表头。

## 验收标准

- 简历中的一个项目经历优先保持为完整 chunk。
- JD 的职责、要求、加分项可以被明确区分。
- Markdown 文档保留标题路径。
- CSV chunk 包含表头，不出现孤立字段值。
- 超长语义块会被二次切分，但不会从句子中间硬切。
- chunk metadata 足够支持后续检索过滤、引用展示和评测分析。
- 原有入库和问答接口保持兼容。

## 风险点

- 过度细分会降低召回完整性。
- 过度合并会降低检索精度，并增加 prompt 上下文浪费。
- 文档类型识别不准确会导致错误切块策略。
- PDF/DOCX 解析结果可能丢失标题结构，需要容错。
- token-aware 切分如果引入新依赖，需要考虑安装和运行环境。
- 旧索引仍然使用旧切块方式，切换策略后需要重建索引才能生效。

## 完整提示词

```text
你是一名资深 RAG 工程师，擅长文档解析、语义切块和检索优化。请基于当前项目，将 RAG 切块策略从字符长度驱动升级为结构优先、语义边界优先、token 预算兜底的语义切块策略。

项目背景：
- 当前项目使用 FastAPI + LangChain + Chroma 构建 RAG。
- 当前 RAG 核心逻辑在 app/rag.py。
- 当前 split_documents 使用 RecursiveCharacterTextSplitter，chunk_size=800，chunk_overlap=120。
- 当前支持文本类型包括 pdf、docx、doc、txt、md、csv。
- 当前 collection 包括 profile、job_description、multimodal_knowledge。

当前问题：
- 现有切块策略主要按照字符长度切分，可能破坏语义。
- 简历中的项目经历可能被拆散。
- JD 中岗位职责和任职要求可能混切。
- Markdown 标题层级没有被充分利用。
- CSV 可能被当作普通文本切，导致表头和数据脱离。

请完成以下任务：

1. 设计并实现文档类型感知的语义切块策略。
   - resume / profile
   - job_description
   - markdown
   - pdf / docx
   - csv
   - image_analysis
   - generic_text fallback

2. 新增结构化切块模块。
   - 建议新增 app/chunking.py 或 app/rag_chunking.py。
   - 不要把所有逻辑继续堆进 app/rag.py。
   - app/rag.py 只负责调用切块接口。

3. 简历切块要求：
   - 按基本信息、教育经历、技能栈、项目经历、实习经历、工作经历、证书、自我评价等模块切。
   - 一个项目经历优先保持为一个完整 chunk。
   - 如果项目经历过长，再按背景、职责、技术栈、成果二级切分。

4. JD 切块要求：
   - 按岗位名称、业务背景、岗位职责、任职要求、加分项、技术栈、硬条件切。
   - 职责和要求不要混切。
   - 技术栈关键词和硬条件可以形成独立 chunk。

5. Markdown 切块要求：
   - 按 #、##、### 等标题层级切。
   - 每个 chunk 保留 section_path。
   - 超长标题段落再按段落和句子切分。

6. PDF/DOCX 切块要求：
   - 优先按页、段落、标题样式或编号结构切。
   - 尽量不从句子中间切断。
   - 保留 page、source、filename 等 metadata。

7. CSV 切块要求：
   - 不要直接按普通文本字符切。
   - 每个 chunk 必须包含表头。
   - 可按固定行数、业务主键或语义分组切。

8. 图片解析结果切块要求：
   - 按图片主题、关键信息、OCR 文本、图表/数值信息、检索关键词等字段组织。
   - 内容较短时保持一个完整 chunk。
   - 内容较长时按结构字段拆分。

9. 增强 chunk metadata。
   每个 chunk 尽量包含：
   - doc_type
   - section
   - section_title
   - section_path
   - chunk_strategy
   - chunk_index
   - parent_doc_id
   - filename
   - source
   - modality

10. 保留 fallback。
   - 对无法识别结构的普通文本，继续使用 RecursiveCharacterTextSplitter。
   - 但 fallback 应优先按段落、句子切，再按长度兜底。

11. 增加验证样例或测试。
   - 验证简历项目经历不被无意义拆散。
   - 验证 JD 职责和要求被分开。
   - 验证 Markdown 保留标题路径。
   - 验证 CSV chunk 包含表头。
   - 验证图片解析结果保留结构字段。

实现要求：
- 不破坏现有入库接口。
- 不破坏现有问答接口。
- 旧索引不需要自动迁移，但要说明新策略生效需要重建索引。
- 代码结构要清晰，便于后续扩展新的 doc_type。
- 对异常格式文档要有 fallback，不要让入库直接失败。

最后请输出：
1. 新切块架构说明。
2. 修改文件列表。
3. 每种 doc_type 的切块策略。
4. 新增 metadata 字段说明。
5. 验证步骤。
6. 是否需要重建索引，以及原因。
```
