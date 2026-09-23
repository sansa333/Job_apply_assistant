# 国内招聘版本：完整性与上线评估

初次评估：2026-09-18；业务评测和接口复核更新：2026-09-21。

本轮新增全库 BM25/向量/RRF 搜索、证据充分性检查、分任务评测及双人标签导入重算；180 项自动化测试通过（1 项已有弃用警告），相关静态检查通过。正式目录仍为 297 条，生产只读 `/ready` 检查与三种搜索方式、岗位证据隔离 HTTP 验收通过。记录见 `outputs/domestic_business_v4_20260921/production_http_smoke.json`。

新评测仍发现缺失事实识别不足；拒答策略未自动启用，生成事实性也未由检索指标证明。完整结果及选型限制见 [业务评测 v4.1](domestic-business-evaluation.md)。本轮没有重新调用付费模型或完成容器/多用户上线验收，下文历史验收中的限制依然有效。

**结论：核心业务已形成可验证闭环，适合个人求职辅助和面试展示；当前运行目录还不能直接当作已上线的正式服务。公网多用户产品尚未达到完整生产条件。** “有全部主要功能”“本机可以演示”“公网生产就绪”必须分别判断。

## 本轮落实的功能和工程修复

| 链路 | 当前实现与验证 |
|---|---|
| 国内官网岗位 → 结构化目录 | 官方源适配器、来源与状态记录、城市/届别筛选；本次额外实际获取百度官网 10 条岗位作为隔离验收输入 |
| PDF → 候选人资料 | 解析与索引、原 PDF 保留；上传先校验大小/签名并在临时目录解析，损坏 PDF 不覆盖旧文件 |
| 岗位搜索 → 精确岗位证据 | 自然语言技术词提取、全部候选排序后分页、job_id 限定召回；真实 Chroma 和 HTTP 测试验证 |
| 匹配分析 | JD/简历结构化证据与规则评分，模型输出定性建议，岗位证据注入提示；不把规则分数称预测准确率 |
| 材料生成 | 国内岗位直接生成求职信、面试准备、邮件草稿；联系方式保留待确认，不发送、不改写 PDF |
| 历史与看板 | 匹配报告及材料保存到 `outputs/domestic/{candidate_id}/{job_id}/`；保存来源哈希、生成时间，可按候选人/岗位查询回读；投递阶段持久化 |
| 检索评测 | 标签边界与切块修复，独立索引和冻结集、494 题回归、正文正例敏感性与配对 Bootstrap；两次完整运行分组指标相同 |
| 部署基础 | 生产环境启动要求至少 32 字符 API_TOKEN；两种前端支持鉴权；真实流式请求大小限制；旧索引拒绝读取；严格 readiness；独立容器配置及模型挂载 |

## 已完成的验证与证据

- 全量自动化测试：**167 passed**。其中一项真实 HTTP 链路测试涵盖 PDF 上传、损坏文件保护、岗位搜索、精确证据、分析、三类草稿、历史回读、候选人目录分离和投递看板。此测试使用 hash embedding 和模拟 LLM，验证业务接通，不证明生成质量。
- BGE 实验：真实 small-zh 与 M3 模型、真实 Chroma、隔离重建；报告位于 `outputs/domestic_eval_release_20260918/report.json`。RTX 5060 Laptop GPU，Python 3.10.9；实测 torch 2.11.0+cu130、sentence-transformers 5.4.1、Chroma 1.5.8、transformers 5.6.2。**这些不是 Docker requirements 中较旧 CPU 依赖组合的验收结果。**
- 本次所改 Python 文件 Ruff 通过；全仓 Ruff 另有既存 `tools/build_jingwei_interview_pdf.py` 未使用 TA_CENTER 导入，未修改这一无关文件。
- `pip check` 通过。已有 `langchain-community` 弃用警告需后续维护。
- `docker compose config --no-env-resolution --no-interpolate -q` 通过。Docker 客户端存在，但本机 Docker 引擎未运行，**未完成镜像构建、容器启动或容器内依赖实测**。
- 真实来源与智谱验收单独保存于 `outputs/domestic_live_acceptance_20260918/`。使用虚构候选人文本和刚获取的公开岗位，无私人简历、无实际投递。首次真实调用完成分析与三类草稿、历史回读和看板；末尾校验因 Windows CRLF 与字符串 LF 的哈希比较失败。`real_model_run/artifact_verification.json` 独立复核已生成产物、输入未改变及看板记录，全部通过，保留原始失败报告。
- 第二次真实调用在生成面试材料时出现 `OpenAITimeoutError`，详见 `verified_run/acceptance.json`；**实时服务稳定性未通过验收**，不能只选择第一次成功来声称生产可用。已添加模型超时 HTTP 504、其他供应商错误 HTTP 503 的受控响应，避免原始供应商错误泄漏；新增回归测试覆盖超时。没有隐瞒失败，也没有延长超时伪造通过。

## 当前本机的数据与索引状态

只读检测命令：

```powershell
.\.venv\Scripts\python.exe -m tools.audit_domestic_release --output outputs/domestic_release_audit.json
```

初次审计时生产 SQLite 为空：此前只完成隔离评测/验收，没有完成正式目录初始化，且 `DOMESTIC_SYNC_ENABLED=false`，启动不会自动填库。这是执行遗漏，不能把“评测应当隔离”当成一直不初始化正式库的理由。

2026-09-18 20:16（北京时间）已重新获取全部 7 个配置来源并写入正式目录，全部成功：**297 条国内记录，含 269 个具体岗位和 28 个公告合集**，覆盖 34 家公司。正式 BGE-M3 索引重建为 **1,190 个片段**。库内 open 状态依据各来源解析，公告合集不能当成单个可直接投递岗位。

使用真实配置的 HTTP 接口验收：`/ready` 返回 **200 / ready**；Python 搜索命中 **122 条**；检索返回 5 个片段，均属于指定 job_id。这里验证的是本机配置、数据和索引，不表示已经启动公网服务或通过多用户/稳定性验收。

记录及更新前备份：`outputs/domestic_production_sync_20260918/report.json`、`catalog_before.sqlite3`、`index_before/`。没有导入历史评测快照，冻结评测集与其指标保持不变。此前“生产库为空、旧索引不兼容”两项已解决，其余容器、模型波动和多用户上线问题仍按本文记录。

`/ready` 检查目录、凭据是否配置、生产鉴权、语义 embedding、可读且非空的国内岗位库、索引版本/配置/正文哈希覆盖。它不主动调用付费模型，也不保证来源日期真实、远端服务在线或负载达标。令牌存在同样不表示已经有多用户授权。

## 个人部署操作

1. 本地 Python 部署使用自己的 `.env`；选定模型和独立向量目录。small 使用 512，M3 可使用 1024。模型切换后必须重建岗位和候选人索引，不混用向量。
2. 生产方式设 `ENVIRONMENT=production`，设置至少 32 字符随机 `API_TOKEN`。原始网页手动填写令牌；Streamlit 从环境读取令牌，仅转发给 `BACKEND_URL` 对应的源，禁止带令牌重定向。
3. 初始化时调用 `/api/domestic/sources/refresh`，或者在停止 API 的维护窗口运行以下命令。该流程先同步来源目录，再统一重建国内索引。查看返回值的 `failed_sources`，不能把 partial 当全成功。

```powershell
.\.venv\Scripts\python.exe -c "from app.domestic.service import DomesticJobService; import json; print(json.dumps(DomesticJobService().refresh_all(),ensure_ascii=False,indent=2))"
```

也可使用本轮新增的维护命令，自动备份原目录和索引、同步全部来源、重建并验证 HTTP 搜索与证据接口；输出目录必须不存在，运行前停止服务并关闭定时同步：

```powershell
.\.venv\Scripts\python.exe -m tools.sync_domestic_catalog --output outputs/domestic_sync_new
```

4. 在准备好的索引上导入自己的 PDF，搜索 → 分析 → 材料草稿 → 回读历史 → 看板进行人工验收。确认 `/ready` 返回 ready 后再开放个人访问。
5. 单进程运行。当前 Chroma 重建是维护操作，并非原子热切换；**不要在并发用户查询时重建**，也不要直接开启定时全量重建后声称无停机服务。

容器配置：

```powershell
Copy-Item .env.docker.example .env.docker
# 编辑 .env.docker 填密钥和随机令牌；模型父目录包含 bge-m3 子目录。
$env:MODEL_DIR = "E:/ai/models"
docker compose config --no-env-resolution --no-interpolate -q
docker compose build
docker compose up -d
```

Compose 默认 CPU，模型挂载到 `/models`，不把 Windows 的 E: 路径直接塞进容器；API 与 Streamlit 仅发布在 127.0.0.1。CPU 指标需要另测，不能引用本轮 CUDA 延迟。两服务分别检查自身健康端点。API 使用一个 worker；挂载数据目录必须允许容器用户写入。不要把共享管理令牌注入的 Streamlit 无鉴权暴露给公网。

容器初始化可在维护时使用：

```powershell
docker compose stop api ui
docker compose run --rm api python -c "from app.domestic.service import DomesticJobService; print(DomesticJobService().refresh_all())"
docker compose up -d
```

## 公网多用户上线仍缺什么

| 必要能力 | 尚未覆盖的原因 |
|---|---|
| 用户身份与资源授权 | 目前是共享持有者令牌和 candidate_id 目录；知道其他 ID 的持有者仍有相同权限，不是租户隔离 |
| 并发与后台任务 | 索引重建非原子切换；长模型请求/源抓取未形成持久任务队列、幂等任务和跨进程锁 |
| 服务可观测性与恢复 | 尚无已验收的告警、SLO、限流策略、并发压测、备份恢复演练和滚动升级 |
| 数据与生成质量 | 人工标签、全新查询、生成事实性评价、来源长期可用率、过期岗位处理效果仍需实测 |
| 隐私与全生命周期 | 需要按真实产品需求实现用户删除、保留期限、审计访问与密钥轮换；现有单用户本地存储不是完整方案 |

这些是上线范围判断，不应写成“代码已全部补齐”。本轮完成的是正式检索修复、核心业务缺口与个人部署基础；没有虚构人工标注、HR 评价或真实求职收益。
