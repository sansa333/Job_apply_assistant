# 工具路由与事实源

仅使用本项目受控的候选人和岗位数据路径。`data/job_catalog.sqlite3` 与 `data/source_corpus` 是岗位事实源；Chroma 是可重建索引；`eval_demo` 仅用于评测，绝不进入真实求职匹配。

## 输入与工具映射

| 目标 | 可选/必要输入 | 工具 | 成功后的证据状态 | 失败或缺失处理 |
|---|---|---|---|---|
| 候选人资料库证据 | `candidate_id` 可选 | `retrieve_profile(candidate_id, query, k)` | `verified_profile` | 无 ID 时跳过检索；有 `resume_text` 时改用 `user_provided`；两者均无则为 `missing` |
| 岗位证据 | 特定岗位需公司、岗位、检索问题 | `retrieve_job(company_name, job_title, query, k)` | 精确 JD 片段 | `job_not_found` 时请求上传/粘贴 JD；不得检索近似岗位 |
| 匹配分析 | 精确 JD；候选人证据可为资料库、当轮文本或缺失 | `analyze_job_fit` | 匹配结论与证据等级 | 个人证据缺失时仅输出通用或待补充结论，不杜撰经历 |
| 申请包 | 匹配/草稿上下文与用户明确请求 | `generate_application_package` | 本地材料与 `generated_not_submitted` | 联系信息缺失时保持 `pending_confirmation`，不得注入默认占位值 |

## 调用细节

- 对特定岗位，先调用 `retrieve_job`。岗位必须在精确命中的 `company_name + job_title` 所属 `job_id` 内检索。
- 仅在明确获得 `candidate_id` 时调用 `retrieve_profile`。不得猜测、默认或复用其他候选人 ID。
- 当轮 `resume_text` 是可使用的用户提供证据，但其等级不同于资料库检索证据；输出时必须标注。
- 优先依赖接口的 `status`、`stage`、`evidence_level`、`contact_status`、`missing_fields` 与 `next_action`。在旧接口仅返回文本时，不将“未检索到”误判为已验证证据。
- 用户提供完整 `jd_text` 时，可按现有上传路径入库后再次精确解析；公司和岗位名称仍用于建立 JD 身份。
- `generate_application_package` 只写入 `outputs/` 和本地 `applications.csv`；它不具备招聘网站、邮箱或 ATS 的提交能力。

## 查询构造

使用与当前任务相关的短查询，例如“职责、检索增强、评测、FastAPI”或“必备技能、年限、业务场景”。不要把其他候选人资料、非目标岗位内容或模型生成结论当作检索事实来源。

## 联系信息兼容处理

联系人字段缺失时，生成内容中使用“待确认”并输出缺失字段列表。若旧版工具仍会自动使用示例联系人，不得在缺失联系人时调用该工具；改为提供明确标注的本地草稿，直到接口支持 `contact_status`。
