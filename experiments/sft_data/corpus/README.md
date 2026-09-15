# 分角色 SFT 候选语料

每行只监督最后一条 assistant 输出。system、user、工具观察及其中出现的其他角色内容不计算目标损失。图片路径相对于本目录，须由学生模型的多模态模板逐一对应 `<image>`。

本地图片包中可有指向原始请求图片的同卷硬链接，供不可变训练素材复用存储；不要原地编辑图片。复制本目录到其他文件系统时，图片内容随普通文件复制。具体批次的存储记录保存在运行目录 reference_review/packaged_image_storage.json。

development 文件供开发与后续训练准备；construction_validation 文件是数据构造方法的保留检查样本，不能混入开发训练。它们不是官方测试集。

origin=hi_independent 为独立 HI 真实执行，origin=gia_offline 为离线 GIA 单步决定，origin=bridge 为真实协作中的对应角色输出。离线决定中的补查提议未被执行。

origin=hi_terminal_replay 为从已记录的 HI 报告提交前状态重新生成的报告。它保留实际输入和图片，前序工具记录真实；没有新增工具执行，也没有复用原报告之后的 GIA 决定。

网页任务的 source_condition 区分来源发现与 provided_source_url。后者的候选地址已在任务中提供，仍须实际打开并核验；它训练来源阅读与取证，不能计为自主搜索发现成功。provided_source_url 保存实际提供的地址。

source_condition=recorded_plan_execution_with_provided_evidence 表示从已记录的GIA计划启动新调查，并提供先前取得的真实来源正文或媒体证据。首项计划只用于启动执行，不重复导出为新模型目标；新HI行动、报告和之后的实际GIA决定分别评审。plan_origin.json记录原始计划、证据编号对应及来源。此条件不计为从零发现来源，也不代表恢复了原调查的全部历史。

候选经过逐目标自动评审和真实输入、输出、图片对应检查；已知的任务含义、事实范围或评分分歧记录在manifest.json的curation_exclusions中，相应目标不收录。没有独立人工认证或学生权重训练。实际Qwen3.5-9B分词与图片处理批次见manifest.json的processing_checks；每批记录其覆盖行数，旧批次不代表后加样本已检查。processing_coverage.json按当前保留决策编号匹配已完成处理记录，不将已排除目标计为当前覆盖。四个完整输入在本机RTX 5090上完成前向损失计算，不能据此推定A100的长上下文训练容量。相同案例的多个运行和状态变体属于同一家族，不能按行随机拆分训练与测试。

网页取证样本的 evidence_source_policy 区分 unrestricted_retrospective_web 与 exclude_known_original_article; publication_dates_not_filtered。前者允许回顾性取证，后者在工具返回前排除已知原事实核查文章及明确登记的别名；两者均未统一限制来源必须早于声明日期，不能视为符合 AVeriTeC / AVerImaTeC 原始评测取证规则的独立调查结果。来源比对见 [来源检查](../evidence_source_audit.json)；没有匹配已知文章 URL，不表示已排除所有镜像、转载或时间泄漏。

| 文件 | 角色 | 决策行 | 来源案例家族 |
|---|---|---:|---:|
| [gia_construction_validation.jsonl](gia_construction_validation.jsonl) | GIA | 71 | 7 |
| [hi_construction_validation.jsonl](hi_construction_validation.jsonl) | HI | 415 | 15 |
| [gia_development.jsonl](gia_development.jsonl) | GIA | 37 | 8 |
| [hi_development.jsonl](hi_development.jsonl) | HI | 303 | 18 |

manifest.json 记录来源与角色构成；image_manifest.json 将原始图片路径对应到随包图片。已检查案例编号及已知素材来源不跨开发与保留划分，尚不能排除跨数据集的语义事件重叠。
