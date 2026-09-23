"""Source-read, assistant-authored challenge cases. NOT human gold or user logs.

Questions/quotes fixed before retrieval runs. Development: Baidu/OPPO;
test: Meituan/Kylin. Source documents have been seen in earlier diagnostics.
"""

# Each answer requires all quoted spans; partial support gets graded relevance 1.
EVIDENCE_CASES = [
    ("job_1cd95487f1ac08b65f2109ed", "development", [
        ("只做过课程里的智能助手，没有公司实习，JD是否认可这种项目来源？", ["课程 / 自发项目均可"], "qualification"),
        ("智能体除了调用外部工具，还需要怎样保存记忆和管理推理状态？", ["Tool / API 调用", "长短期记忆", "多轮推理与状态管理"], "multi_support"),
        ("这里衡量智能体落地效果时，除了任务成功率还会看哪些工程指标？", ["成功率 / 稳定性", "成本（Token / 调用）", "延迟与用户体验"], "paraphrase"),
    ], ["这个Agent岗位明确要求至少三年商业项目经验吗？", "这份Agent岗位每月基本工资具体是多少元？"]),
    ("job_2c76c6eaa1835de41a5224ea", "development", [
        ("世界模型方向对在读身份和学历层次怎么规定？", ["硕士/博士在读"], "qualification"),
        ("我做过场景三维表达和神经仿真，这些背景与岗位要求对应吗？", ["时空建模、3D 场景表示或神经仿真"], "paraphrase"),
        ("这个岗位既要做环境预测，也要搭建数据迭代链路吗？", ["复杂环境下的预测和决策能力提升", "数据闭环与数据飞轮"], "multi_support"),
    ], ["世界模型岗位规定训练集必须达到多少小时的视频吗？", "这份世界模型岗位是否提供每月固定租房补贴，金额多少？"]),
    ("job_419e3b5b236f1909022869ab", "development", [
        ("没有顶会论文就不能申请这个智能体岗位吗，原文如何表述？", ["有顶会论文或开源Agent项目贡献者优先"], "preference_not_mandatory"),
        ("Agent开发框架需要全部掌握，还是至少会一种？", ["LangChain、LlamaIndex、AutoGen等至少一种"], "alternative"),
        ("这个智能体方向同时看学历和大模型实践吗，各是什么条件？", ["硕士及以上学历", "Prompt Engineering、Fine-tuning等实践经验"], "multi_support"),
    ], ["这个智能体团队要求候选人每周到岗五天并实习六个月吗？", "JD明确规定智能体系统必须达到99.9%的成功率吗？"]),
    ("job_4feeeba409d2302ba7ddbb10", "development", [
        ("异构计算需要同时精通所有硬件平台，还是任意一种即可？", ["GPU/ARM/MIPS/DSP等任意异构计算平台"], "alternative"),
        ("底层通信优化是否涉及网络拓扑和远程直接内存访问技术？", ["不同网络拓扑结构的通信算法", "RDMA，GPU direct"], "paraphrase"),
        ("做过CUDA汇编层优化和Linux内核开发，在JD里属于什么条件？", ["具有以下条件者优先", "有SASS或PTX级别优化开发经验", "有Linux内核相关开发和优化经验"], "multi_support"),
    ], ["这份异构计算JD明确要求必须持有NVIDIA官方认证吗？", "异构计算工程师在入职后具体归谁直接管理？"]),
    ("job_00b46019f3e3cc1a0b5a7263", "development", [
        ("技术支持岗位要自己训练AI模型，还是主要把AI工具用到工作里？", ["掌握 AI 工具与产品的基础使用方法", "嵌入现有工作流程"], "distinction"),
        ("技术支持对最低学历和出差适应性分别如何要求？", ["本科及以上", "适合不定期短长期出差"], "multi_support"),
        ("除了响应用户问题，还需要持续跟踪产品生命周期里的质量数据吗？", ["建立全渠道品质数据监控体系", "对产品上市到生命周期结束进行质量跟踪管理"], "paraphrase"),
    ], ["技术支持岗位每个月最多出差几天？", "这份技术支持岗位明确要求通过大学英语六级吗？"]),
    ("job_0873c57bdeb5d0fd1fb59bf7", "development", [
        ("AI运维开发要求必须会某一款代码助手，还是列举工具中任意一款？", ["Claude Code、Codex、Gemini CLI 等任一命令行代码助手"], "alternative"),
        ("这里的研发平台后端具体服务哪个系统，要建设怎样的交付流程？", ["Android 系统 DevOps 平台的后端服务及插件", "智能化 AI CI/CD 流程"], "multi_support"),
        ("只会Python是否满足编程语言条款，还是Go和Java也都必须会？", ["Python/Go/Java 等任一编程语言"], "alternative"),
    ], ["AI运维开发的JD要求Python连续使用满五年吗？", "这个AI运维岗位是否承诺每周可以远程工作两天？"]),
    ("job_094a7a0b804e787de1224906", "development", [
        ("视觉理解方向是否把摄影知识列为所有申请者必须满足的条件？", ["具有以下经历者优先", "掌握专业摄影知识"], "preference_not_mandatory"),
        ("这个多模态岗位只负责训练模型，还是也承担数据基准和部署验证？", ["训练数据和benchmark构建", "协助算法部署与效果验证"], "multi_support"),
        ("视觉智能体准备落在哪些手机使用场景中？", ["辅助拍摄、智能管理、智慧创作"], "paraphrase"),
    ], ["这个视觉理解岗位明确规定必须有三篇CVPR论文吗？", "这份多模态岗位使用的线上评测集叫什么名字、含多少条数据？"]),
    ("job_09d4c6364c17a51a2b8e1fde", "development", [
        ("AI基础设施的第一个方向包括训练平台、推理部署和资源调度吗？", ["模型训练平台、推理部署平台、资源调度管理平台"], "paraphrase"),
        ("这份AI能力要求是否明确包含亲自训练模型？", ["不涉及模型训练"], "negation"),
        ("两个工作方向分别侧重平台架构还是大模型推理引擎？", ["机器学习平台的整体架构设计", "大模型推理框架的设计与开发"], "multi_support"),
    ], ["这个AI基础设施岗位给出了必须降低30%推理时延的考核指标吗？", "AI基础设施团队当前生产环境使用哪一型号GPU？"]),
    ("job_015adaf5a31cb6260c28b7ee", "test", [
        ("我熟悉扩散模型并能用PyTorch做实验，能对应哪些任职要求？", ["熟悉扩散模型研究进展", "熟悉PyTorch，能快速开展实验"], "multi_support"),
        ("图像生成研究中，奖励模型主要用来改善生成结果的哪些方面？", ["提升图像生成的质感与美感"], "paraphrase"),
        ("没有视觉生成论文是否一定不符合条件，JD把论文经历放在哪里？", ["加分项", "发表过视觉生成相关的论文专著"], "preference_not_mandatory"),
    ], ["图像生成实习明确要求必须有两年以上扩散模型工作经历吗？", "这个图像生成实习的日薪和每月发薪日分别是什么？"]),
    ("job_02b5838e7b91237fd6bd511b", "test", [
        ("原生多模态预训练需要依赖人工文本标签吗，原文描述的优势是什么？", ["不依赖文本标注", "直接学习无监督多模数据"], "negation"),
        ("后训练阶段要采用哪些方式，并尝试解决哪些物理世界任务？", ["通过SFT / RL等手段", "Robot/Gaming等物理世界问题"], "multi_support"),
        ("这个原生多模态课题对编程语言和开源训练框架分别有什么要求？", ["Python、C++等至少一门编程语言", "Megatron、DeepSpeed 等开源训练框架"], "multi_support"),
    ], ["原生多模态课题要求Megatron训练规模至少达到多少张显卡？", "这个原生多模态实习是否明确保证转正名额？"]),
    ("job_04772e783605a4446cfad040", "test", [
        ("我只能实习三个月、每周来三天，是否满足这里的到岗时间要求？", ["6个月以上，每周到岗4天以上"], "numeric_constraint"),
        ("大模型算法实习对学位层次和NLP经验年限分别如何规定？", ["硕士或博士学位", "具备1年以上NLP经验"], "multi_support"),
        ("模型对齐工作除了指令微调，还包含奖励模型和强化学习吗？", ["指令微调、奖励模型优化、强化学习（RL）策略迭代"], "paraphrase"),
    ], ["这个大模型实习每周四天中的具体工作日是哪几天？", "这份大模型实习明确要求有Java生产项目经验满两年吗？"]),
    ("job_07bddc30d4230c7f35cc46e4", "test", [
        ("具身智能方向除了VLA，还需要探索什么动作模型和学习方法？", ["世界动作模型（World Action Model, WAM）", "具身强化学习"], "multi_support"),
        ("没有顶会论文就不能申请LongCat具身智能研究员吗？", ["在 CVPR / NeurIPS / ICLR 等顶级会议发表过高水平论文者优先"], "preference_not_mandatory"),
        ("这个岗位需要全部掌握列出的前沿技术，还是深入理解至少一项？", ["World Model / VLA / Diffusion Policy / 强化学习等至少一项"], "alternative"),
    ], ["这个具身智能岗位是否规定必须持有机器人操作安全资格证？", "LongCat团队明确要求WAM模型上线后达到多少成功率？"]),
    ("job_06f4cc7618edc051975bc791", "test", [
        ("嵌入式校招里，Rust和汇编是必需条件还是优先条件？", ["熟练掌握C/C++编程", "掌握Rust或汇编（ARM/RISC-V/x86）者优先"], "preference_not_mandatory"),
        ("这个嵌入式岗位同时包含Linux驱动适配和实时系统内核开发吗？", ["嵌入式Linux硬件驱动的移植、适配与二次开发", "RTOS内核子系统设计与实现"], "multi_support"),
        ("JD列举了哪些实际调试手段来排查软硬件问题？", ["GDB、逻辑分析仪、示波器"], "paraphrase"),
    ], ["嵌入式岗位明确要求必须持有ISO 26262个人认证吗？", "这份嵌入式校招工作第一年提供多少天带薪年假？"]),
    ("job_0ca5f66c81c65e7eb5d0070a", "test", [
        ("只会Go、不熟Java，是否一定不满足这份后端岗位的语言要求？", ["Go或Java等至少一门后端编程语言"], "alternative"),
        ("Kubernetes是必须掌握的技能吗，Docker和数据库又如何要求？", ["了解MySQL、Redis", "了解Docker容器化部署", "有Kubernetes使用经验者优先"], "multi_support"),
        ("这里是否接受2027届本科毕业生，还是只招硕博？", ["2027届本/硕/博毕业生均可投递"], "qualification"),
    ], ["这个Go后端岗位是否规定必须有五年Spring Cloud工作经验？", "这份Go和Java后端岗位试用期工资打几折？"]),
    ("job_0fed2f1e51423ae80f3cc57c", "test", [
        ("Qt校招的日常交付主要面向哪类客户端界面？", ["系统更新客户端交互界面需求开发以及维护"], "paraphrase"),
        ("Qt开发工具、编程语言和操作系统环境各有什么要求？", ["C、C++编程语言", "Qt Creator", "Linux下常用操作命令"], "multi_support"),
        ("Qt岗位是否将研发实习经验设为没有就不能投的硬门槛？", ["有研发实习经验者优先"], "preference_not_mandatory"),
    ], ["这份Qt岗位明确要求使用Qt 6.7而不是Qt 5吗？", "Qt岗位是否承诺入职后每年可以调薪两次？"]),
    ("job_13473c6f1daff3a18cd38739", "test", [
        ("Linux内核校招是否同时接受本科生，并把Rust作为优先技能？", ["本科及以上学历", "熟悉Rust者优先考虑"], "multi_support"),
        ("内核工程师需要了解哪些网络知识，并承担怎样的驱动适配任务？", ["了解TCP/IP协议栈及各层数据通路", "网络驱动调优"], "multi_support"),
        ("参与内核社区和嵌入式课程竞赛，在JD中属于何种条件？", ["有以下经验者优先", "参与过Linux内核社区", "有嵌入式相关课程设计、学科竞赛经验并获奖"], "preference_not_mandatory"),
    ], ["内核岗位明确规定必须提交过至少十个上游补丁吗？", "这个内核研发岗位每个月发放多少元餐饮补贴？"]),
]

# Full-catalog intents, authored before runs. Graded labels are executable
# requirement contracts, NOT human recommendations or scores from rank_job.
# (id, split, query, all-of facets; each facet is an any-of regex, filters)
DISCOVERY_CASES = [
    ("d01", "development", "我想做能调用工具完成任务的智能助手，最好涉及记忆管理", [r"Agent|智能体", r"记忆|Memory"], {}),
    ("d02", "development", "找用Java或者Go开发服务端接口的校招方向", [r"Java|\bGo\b|golang", r"后端|服务端"], {"recruitment_type": "campus"}),
    ("d03", "development", "希望做手机上的多模态视觉理解，不是单纯图像生成", [r"视觉理解|视频理解|图像理解", r"手机|移动|终端"], {}),
    ("d04", "development", "想找需要对大模型训练过程做分布式加速的岗位", [r"分布式", r"训练"], {}),
    ("d05", "development", "我有Linux和C++基础，想做设备驱动或系统内核", [r"Linux", r"驱动|内核", r"C\+\+|C语言"], {}),
    ("d06", "development", "只看OPPO做自动化研发交付和AI代码工具的工作", [r"DevOps|CI/CD|研发效能", r"AI|智能"], {"company_name": "OPPO"}),
    ("d07", "development", "在上海找图像或视觉方向的实习机会", [r"视觉|图像"], {"location": "上海", "recruitment_type": "internship"}),
    ("d08", "development", "北京有没有机器人运动决策和强化学习相关方向", [r"机器人|具身", r"强化学习|\bRL\b"], {"location": "北京"}),
    ("d09", "development", "想做检索增强知识库，同时需要写Python", [r"RAG|检索增强|知识库", r"Python"], {}),
    ("d10", "development", "找渲染界面和Qt客户端开发的机会", [r"Qt", r"界面|客户端"], {}),
    ("t01", "test", "希望研究生成图片的模型，以及用奖励信号改善画面质量", [r"图像生成|文生图", r"奖励|RLHF"], {}),
    ("t02", "test", "我做过声学建模，想找语音合成或者识别方向", [r"语音合成|语音识别|声学|token2wav|语音(?:/音频)?生成|\bTTS\b|\bASR\b"], {}),
    ("t03", "test", "希望做长上下文大模型推理优化和模型压缩", [r"长上下文|长文本|长序列|long.?context", r"推理优化|推理加速|推理引擎", r"压缩|量化|蒸馏"], {}),
    ("t04", "test", "想做视觉语言动作模型，服务真实机器人控制", [r"VLA|视觉.语言.动作", r"机器人|具身|动作"], {}),
    ("t05", "test", "找处理数据清洗和训练数据质量建设的岗位", [r"数据清洗|数据质量|数据治理", r"训练|模型"], {}),
    ("t06", "test", "我熟悉CUDA，希望做底层算子和计算库加速", [r"CUDA", r"算子|计算库"], {}),
    ("t07", "test", "想做大模型对齐，关注强化学习和指令微调", [r"强化学习|\bRL\b", r"指令微调|SFT|对齐"], {}),
    ("t08", "test", "希望做无人机导航和轨迹规划相关算法", [r"无人机", r"规划|导航"], {}),
    ("t09", "test", "希望参与视频生成或视频编辑，不限具体模型框架", [r"视频生成|视频编辑|生成式视频"], {}),
    ("t10", "test", "我做过Spring Boot，想找用数据库和缓存的后端岗位", [r"Spring", r"MySQL|Redis|数据库", r"后端|服务端"], {}),
    ("t11", "test", "天津有没有使用Go或Java的后端校招岗位", [r"Go|Java", r"后端|服务端"], {"location": "天津", "recruitment_type": "campus"}),
    ("t12", "test", "深圳找Agent应用方向，希望涉及知识检索", [r"Agent|智能体", r"RAG|知识库|检索"], {"location": "深圳"}),
    ("t13", "test", "美团有什么使用Megatron或者DeepSpeed训练框架的实习", [r"Megatron|DeepSpeed"], {"company_name": "美团", "recruitment_type": "internship"}),
    ("t14", "test", "找麒麟软件做实时嵌入式操作系统的校招工作", [r"RTOS|实时操作系统"], {"company_name": "麒麟软件", "graduation_year": 2027}),
    ("t15", "test", "想在广州做有机器人硬件适配工作的岗位", [r"机器人|硬件", r"适配|驱动"], {"location": "广州"}),
    ("t16", "test", "大疆有没有面向AI应用的全栈开发机会", [r"全栈", r"AI|人工智能"], {"company_name": "大疆创新"}),
    ("t17", "test", "想研究推荐排序算法，并关注大模型的应用", [r"推荐|排序", r"大模型|LLM"], {}),
    ("t18", "test", "找数据库内核或存储引擎方向，重点关注性能优化", [r"数据库|存储", r"内核|引擎", r"性能|优化"], {}),
    ("t19", "test", "西宁有没有量子纠错算法相关校招", [r"量子纠错"], {"location": "西宁"}),
    ("t20", "test", "寻找明确研究蛋白质晶体衍射结构解析的岗位", [r"蛋白质", r"衍射"], {}),
]
