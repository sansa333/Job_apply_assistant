# Evidence-Grounded Job Application Agent：产品、技术与评测设计 v1

## 1. 产品定位

本项目面向求职者，不面向招聘方自动筛人。核心目标是把候选人已经提供的真实经历转换成针对具体历史 JD 的可审计材料，同时避免虚构技能、业绩、联系方式和“已完成投递”状态。

产品闭环：

```text
候选人资料入库
  → 精确岗位解析
  → Requirement–Evidence Matrix
  → 透明匹配评分
  → 求职信/邮件/面试材料草稿
  → 无依据断言检查
  → 人工确认
  → 本地投递历史与审计轨迹
```

该设计吸收了 Resume Matcher 的主简历与材料生成思路、JobSync 的状态持久化与人工审批思路、AIHawk 的任务自动化边界，以及 CareerBERT 的排序评测思路，但没有复制其产品或代码。系统默认不操作招聘网站，不批量投递，也不把匹配分解释为录用概率。

## 2. 运行时状态

每次匹配请求记录以下状态：

- `request_accepted`：请求建立；
- `job_resolved`：精确定位公司和岗位；
- `requirements_parsed`：JD 转为结构化要求；
- `evidence_collected`：收集候选人和岗位证据；
- `evidence_aligned`：逐条建立要求—证据关系；
- `scored`：透明规则计算分项和总分；
- `materials_generated`：生成本地草稿；
- `output_validated`：检查无依据量化断言和虚假投递声明；
- `awaiting_human_confirmation`：等待用户核对后自行发送；
- `blocked`：缺失精确 JD、候选人证据或发现需要复核的错误。

申请包输出除 Markdown 外还包含 `evidence_matrix.json`、`score_breakdown.json`、`validation_report.json`、`workflow_trace.json` 和 `submission.json`。

## 3. 匹配评分边界

`evidence_weighted_v1` 对技术技能、职责、经验、学历、领域、语言、办公地点和软技能分别赋权。`direct/partial/missing` 对应 1.0/0.5/0.0 的证据值；缺失必备条件会降低总分并保留原始 coverage。

当前状态为 `uncalibrated_baseline`。它具备可解释性，但在第二标注者和更大数据集完成前，不用于声称“匹配准确率”或“录用概率”。接口同时返回总分、原始覆盖率、must-have 覆盖率、分类型得分、缺失条件和逐项证据引用。

## 4. 具体评测数据

`data/eval_dataset/job_agent_v1` 当前包括：

- 8 个带来源 URL 和 SHA-256 的公开历史 JD：Arm、British Airways、Visa、Veeva Systems、UiPath、Transdermal Diagnostics、Warner Bros. Discovery、The University of Edinburgh；
- 6 个明确标记为 synthetic 的完整候选人档案；
- 24 个带理由和 hard gap 的匹配银标；
- 10 个 Agent 策略轨迹场景；
- 8 个生成安全场景。

人物档案不是“候选人 A”模板，而是带教育、年限、项目、量化成果、地点和明确负面边界的合成档案。例如 Noah Williams 有八年现场广播支持、ST 2110、AES67、NMOS、UHD/HDR 和轮班证据；这使其对 Warner Bros. Discovery Broadcast Engineer 是具体 high 样本，对 Arm Full Stack Data Scientist 则是“基础设施有重合但职业主体不同”的 hard negative。

## 5. 标注协议

当前标签仅为项目作者首轮银标。升级为 gold 前必须：

1. 两名标注者独立完成 requirement-level support 和 pair-level label；
2. 计算四分类 Cohen's kappa；
3. 第三人裁决所有 pair label 与 must-have 分歧；
4. kappa 至少 0.70；
5. 每条 requirement 必须有原文、证据引文、support 和 hard-gap 说明；
6. 保留裁决前标签，禁止覆盖历史。

详细规则见 `data/eval_dataset/job_agent_v1/ANNOTATION_GUIDE.md`。

## 6. 当前可复现结果

运行：

```bash
python -m tools.build_professional_eval_dataset
python -m tools.evaluate_professional_agent
```

当前报告位于 `data/eval_dataset/job_agent_v1/reports/baseline_report.md`。当前银标基线为：

- JD 技能抽取 Micro-F1：0.7447；
- 24 对整体匹配 Macro-F1：0.5784；
- 匹配分与银标 Spearman：0.8366；
- mean nDCG@5：1.0000，但每个候选人只有 4 个被标岗位，不能外推；
- Agent 策略轨迹 10/10；
- 生成安全场景 8/8。

消融显示技能重合基线的整体 Macro-F1 更高，但 Requirement–Evidence 方法的 Spearman 和 nDCG 更高。这说明当前贡献主要体现在排序与解释性，而分档校准仍需扩大 development 数据，不能只挑更好看的单项指标。

## 7. 上线配置

公开部署必须设置：

```env
ENVIRONMENT=production
API_TOKEN=<long-random-token>
CORS_ORIGINS=https://your-ui.example.com
MAX_REQUEST_BYTES=10485760
```

API 客户端通过 `Authorization: Bearer <token>` 调用 `/api/*`。`/health` 用于存活探针，`/ready` 检查数据目录、输出目录、岗位目录和 LLM 凭据。Docker Compose 将数据与输出作为持久卷挂载。

生产上线前还需要 TLS 反向代理、PII 加密或至少磁盘加密、备份/删除策略、集中日志、真实域名、依赖漏洞扫描，以及根据实际负载决定异步队列与独立模型服务。

## 8. 下一版验收标准

- 至少 60 个公开历史 JD，覆盖 6 个职业族；
- 至少 30 个 consented/anonymized 或人工审查的高质量合成简历；
- 至少 300 个 pair labels，development/test 按候选人和职业族双重隔离；
- 每类至少 30 个 hard negatives；
- 至少 100 个 Agent 轨迹场景，包含模型超时、检索空结果、Prompt Injection、重复请求和输出校验失败；
- 至少 50 个双人评审的材料生成样本；
- 报告 bootstrap 95% CI，并同时披露效果、P95 延迟和单请求成本。
