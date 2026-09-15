# ForenSearcher

ForenSearcher 使用 Qwen3.8-27B 承担 GIA 全局调度、汇总与 HI 局部调查。所有 Workspace 共享同一 HI 模型，各自维护独立上下文。`SEARCH`、`INSPECT` 和已注册的 `DETECT` 是 HI 选择的结构化行动，由 Harness 执行。

## 调查运行框架

`InvestigationRuntime`执行轮内并行、轮间串行的调查。GIA通过`INVESTIGATE`分配任务、通过`FINISH`结束调查，也可调用只读记忆动作；同一HI策略在各Workspace的独立Working Memory中运行。Workspace不能创建子Workspace，其他Workspace的Finding和报告不会进入HI上下文，但已经完整提交的Evidence会在每次HI行动前刷新。

```python
from foren_searcher import GlobalStore, InitialMedia, InvestigationRuntime

store = GlobalStore("investigation/store.sqlite3", "investigation/artifacts")
initial_video = InitialMedia(
    source_ref="I0",
    media_type="video/mp4",
    modalities=("video",),
    label="待核查视频",
    original_filename="input.mp4",
    path="input.mp4",
)
runtime = InvestigationRuntime(
    store=store,
    gia_policy=gia_policy,
    hi_policy=hi_policy,
    search_service=search,
    inspect_service=inspect,
)

async with runtime:
    result = await runtime.run_new(
        task="核查这段视频是否与所述事件有关",
        claim_materials={"C0": {"claim": "待核查说法"}},
        media=(initial_video,),
    )
```

`gia_policy`和`hi_policy`分别实现`GiaPolicy`与`HiPolicy`协议，可连接脚本化测试策略或实际模型客户端。工程并发数、轮次和超时上限由Runtime执行；GIA通过`runtime_constraints`获知是否还能创建下一轮调查。GIA和HI各自收到当前决策的`turn_index`、`max_turns`与`remaining_turns`，剩余次数不包含当前决策；值为0时仍有本次提交决定或报告的机会。达到原有次数上限后，Runtime仍会终止继续行动的策略。

HI的非终止输出为：

```json
{
  "action": "SEARCH",
  "arguments": {"input_type": "text", "query": "查询文字", "output_type": "video"},
  "focus_evidence_ids": ["E3", "E8"]
}
```

`focus_evidence_ids`是当前Workspace希望重点保留的完整列表，不表示真假或全局重要性。HI结束时直接输出Workspace Report，不调用`RETURN_REPORT`工具：

```json
{
  "task_id": "W1-1",
  "status": "COMPLETED",
  "findings": [
    {
      "statement": "该材料至少在2022年已经公开出现。",
      "evidence_ids": ["E17"],
      "limitations": ["只能确认当前找到的最早公开记录。"]
    }
  ],
  "selected_evidence_ids": ["E17"]
}
```

Harness负责分配Finding编号、验证E引用并原子保存报告。`QUERY_MEMORY`和`READ_RECORD`是只读的Harness能力，不产生Evidence。HI的`QUERY_MEMORY`只检索Candidate Source和Evidence，自身任务、Finding与报告可通过`READ_RECORD`读取；GIA还可查询Finding、已完成报告及尚未提交报告的Workspace任务。

GIA的`latest_reports`将原分配任务`task/materials`与报告或运行状态一起提供。`COMPLETED`只针对该局部任务。历史报告的任务文字参与记忆检索；视图中的`task_context_truncated`表示任务上下文需要按W编号分页读取。尚未提交报告的W编号返回任务和运行状态，提交后返回报告及其原任务。GIA可读取各工作区记录，HI只可读取自身任务和报告。

最新报告的 `supporting_records` 直接提供所引用 Evidence 的正文首页、来源和分页位置，优先保留工具分析结果。`image_inputs` 分别列出该 HI 成功模型请求实际附带的图片 E、来源和定位；它不表示工具处理过的全部媒体，也不证明模型理解正确。这些报告附件只进入 GIA 视图，受原有条数及总上下文限制，截断会显式标记；完整正文仍可按 E 编号和 `next_offset` 读取。工具处理范围、候选对应范围与模型附图位置分别保留，不用采样间隔推定可排除的编辑长度。

超时、取消或运行失败但未提交报告的 HI，也提供已成功接收的图片记录与观察统计。INSPECT 动作摘要保留实际响应的 E 卡片，包括没有单独结果正文编号的帧和页面；工具分析正文优先于媒体属性与文件描述。GIA 可从这些图片以及 `supporting_records` 中的图片 Evidence 取得实际像素，沿用同一附图容量和来源规则；工作区终态仍保持失败或超时，已有材料不自动成为已核实结论。

GIA 汇总请求在不同 HI 报告之间、各报告的 Finding 之间轮流选取所引图片，结合记忆动作选中的图片与 GIA 上次实际查看的图片，在同一次请求中附上最多8张原图，沿用 HI 的方向、尺寸和 PNG 传输设置。候选图片超出容量且有未完成报告时，先附 `READ_RECORD` 点选图片，再在超时、失败、取消或 `INCOMPLETE` 的 HI 之间轮流附图；已完成报告的文字仍保留，未附图片显式列出并可继续读取。`global_view.image_inputs` 表示本次 GIA 实际附图，报告内的同名字段表示该 HI 的历史附图；两者分别保存。GIA 可用 `READ_RECORD` 选择已有图片复核，需要新取帧、裁剪或转写时交给 HI。私有分析不会随图片传递。

`visual_observation_summary` 汇总本 HI 成功请求及当前附图；报告附件中的统计只包括该 HI 已完成的请求。`global_view.visual_observation_summary` 按持久化记录合并本调查所有 HI 历轮成功请求的实际附图，不包含 GIA 附图或尚未返回的请求，也不传递 HI 私有判断。图片 E 数量包含裁剪和同一时刻的不同 E，视频完整帧数按源视频与真实帧时刻去重；重复查看不重复计数，局部裁剪不计为看过整个视频帧。

`observed_ranges_s` 根据实际帧时长合并覆盖区间，`unconfirmed_ranges_s` 列出 `coverage_span_s` 内尚未完整覆盖的区间；已知媒体时长时包含全长及未查看的尾段，否则范围只到已观察末端。未知帧时长只记帧时刻。覆盖不代表场景连续，也不代表模型理解正确。范围明细超出上下文预算时保留计数并标记 `coverage_truncated`；连计数也无法容纳时省略整份统计并标记 `visual_observation_summary_truncated`。缺失的范围字段不表示没有未查区间。完整观察记录仍在 Store 中，源视频总帧数、工具处理帧数及实际附图帧数分别解释。

尚未提交报告的旧任务进入`history_index`，GIA也可通过`record_types: ["workspace_task"]`按原任务、运行状态或错误文字查询；紧凑视图的截断不限制检索范围。已提交报告的任务统一按`report`检索，同一W编号不重复返回两类卡片。兄弟HI不能查询或读取其他工作区的任务与错误。

GIA 可复用证据卡片 `derived_from` 中已展示的来源 E。材料引用仍须存在于当前调查；正文中提及的编号不自动进入可用材料集合，R、Finding 和报告不能作为新 Workspace 的材料。

`source_ref` 用于定位入库来源。Demo 的 `I0`、`I1` 等编号按输入材料分配，不代表不同拍摄者、发布者或独立真实来源；不同 E、R、I 编号也不能证明独立佐证或材料之间没有关联。来源关系须由实际出处与材料内容支持。

读取网页 Evidence 正文时，`READ_RECORD` 在权限允许的情况下同时提供该来源 R 的 `source_url` 和 `final_url`，便于将正文、引用与实际出处核对；来源标题不代表机构身份。此信息属于读取结果，Evidence 卡片仍保持六字段，候选检索上下文和兄弟 Workspace 的判断不会随正文附带。

完整的模型输入、结构化输出、Observation、错误和运行事件保存在`trajectory_events`中，可导出为JSONL：

```python
store.export_trajectory("investigation/trajectory.jsonl")
```

专门媒体取证`DETECT`只有注册实际处理器后才会出现在HI可用行动中；未配置时不会暴露占位或模拟检测结果。

## 主框架工程边界

编排、Workspace/Harness、持久化 Store、SEARCH/INSPECT 门面和本地模型适配器采用“一个调查对应一个持久化 Store、同一时刻由一个进程内 Runtime 推进”的部署方式。Runtime 会串行化同一实例上的重复调用；同一轮的 Workspace 按 `max_concurrency` 并行执行。多个进程不得同时推进同一个 Store，横向扩展时应按调查分片。

回归测试覆盖控制流、数据一致性、资源边界、失败恢复、关闭语义和审计。模型在实际任务中的准确率与第三方搜索服务的可用性须通过目标任务实测评估，不能由接口测试推定。

同一个 `hi_policy` 实例会收到并发的 Workspace 调用，因此实际模型适配器必须支持异步并发；不支持并发的客户端应把 `max_concurrency` 设为 `1`。GIA 策略只会串行调用。

多 Workspace 运行时，首份报告提交后仍有任务持续运行超过 `report_review_delay_s`（默认 30 秒），GIA 可评估已有证据并选择 `WAIT` 或 `FINISH`。只有原问题已被充分回答、剩余调查已无必要时才能提前结束；若评估期间到达新报告，停止等待旧评估，并将新报告纳入后续决策。设置 `report_review_delay_s=None` 可关闭中途评估。未结束的 HI 只在 GIA 决定结束后取消，已发送请求仍按原规则结算费用；正常结束时取消的剩余调查不计为轮次错误，实际运行错误仍保留。

JSON 解析、角色校验和外层动作校验共享连续输出修复次数。连续四次输出失败且没有完成有效动作时，当前角色停止修复并保留错误与已有证据；完成有效动作后重新计数。各 HI 的次数独立，不能通过切换修复层次重新取得次数。

HI 动作未给出 `focus_evidence_ids` 时，模型适配器保留当前焦点；该字段仅被误放入 `arguments` 时移回动作外层。显式空列表仍表示清空焦点，内外同时提供时不猜测应采用哪一份。动作、工具参数和证据权限继续由原有解析器与 Harness 校验。

默认每次 GIA 策略调用最多运行 120 秒，每个 Workspace 最多运行 900 秒。可通过 `gia_turn_timeout_s` 和 `workspace_timeout_s` 调整；传入 `None` 可以关闭对应截止时间，但生产环境应保留有限值。轮次数和模型轮次数仍分别由 `max_rounds`、`max_gia_turns` 与 `max_workspace_turns` 限制。

模型输出还受资源上限保护：默认每轮最多 32 个 Workspace、每个 Workspace 最多携带 64 个材料、一个调查最多 64 个原始材料。对应参数为 `max_workspaces_per_round`、`max_materials_per_workspace` 和 `max_original_materials`。`MemoryPolicy` 另外限制焦点证据数、报告 Finding 数、记忆查询长度、单条 Observation 大小及总模型视图大小；超限的模型输出会作为可修复的 Harness rejection 返回，不会创建部分轮次。

`run_new` 对相同的 C 材料和初始媒体是幂等的。客户端在响应丢失后可以安全重试：框架会逐字节核对已保存媒体并复用原有 E；同一 `source_ref` 的内容发生变化时会拒绝继续，避免把新文件误当作旧证据。恢复已有调查也可以直接调用 `run`，运行中的旧轮次会从最后一个完整持久化边界继续，已经提交的 E、Workspace Report 和已完成轮次不会重复生成。

开放轮次中未完成的 Workspace 会恢复自身已提交的行动历史、最新 Observation、焦点、候选来源和已查看证据状态。只有模型请求而没有 Observation 的步骤不计为已执行；恢复过程不重放工具动作。兄弟 HI 的历史与局部判断仍保持隔离，模型 KV cache 和在途网络请求不恢复。

原 Harness 工具批次取消在适配层转为标准 `asyncio.CancelledError`，兼容外层 `asyncio.timeout`；原始批次异常保留在异常链中，取消仍向调用方传播。

Runtime 关闭时先停止接收新调查，等待当前调查到达终态，再依次关闭拥有的 INSPECT 与 SEARCH 服务。即使一个服务关闭失败，其他服务仍会得到关闭机会：

```python
async with InvestigationRuntime(
    store=store,
    gia_policy=gia_policy,
    hi_policy=hi_policy,
    search_service=search,
    inspect_service=inspect,
) as runtime:
    result = await runtime.run(
        task=task,
        original_material_ids=original_material_ids,
    )
```

每个 Artifact 在提交时保存 SHA-256 摘要，默认在读取前核对文件大小和摘要；文件被替换或损坏时会抛出 `ArtifactIntegrityError`，不会把变化后的字节交给 INSPECT。受控的只读存储若需要减少大文件的重复读取成本，可以用 `GlobalStore(..., verify_artifacts_on_read=False)` 关闭逐次核对，但仍应定期运行健康检查。

进程存活检查可以使用 Store 的只读健康检查。应在没有材料正在写入时执行；`degraded` 会列出数据库外键问题、缺失、大小或摘要不符的 Artifact、孤立文件以及未完成的暂存文件：

```python
health = store.healthcheck()
if health.status != "ok":
    raise RuntimeError(health.to_dict())
```

标准 Python 日志只记录轮次、Workspace 状态、耗时和错误类型。完整模型输入、可见输出与 Observation 位于 `trajectory_events`；其中可能包含调查材料，生产环境应按证据数据的敏感级别设置数据库访问权限、保留期和备份策略。SQLite 与 Artifact 目录必须作为同一数据集备份。

`SEARCH` 的稳定调用形式：

```python
{"input_type": "text", "query": "桥梁事故 官方通报", "output_type": "text"}
{"input_type": "image", "query": "E27", "output_type": "video"}
{"input_type": "text_image", "query": {"evidence_id": "E27", "text": "桥梁 坍塌"}, "output_type": "video"}
{"source": "https://example.org/report", "output_type": "text"}
{"source": "R12", "output_type": "video", "include": ["subtitles"], "subtitle_language": "en"}
```

候选来源使用 `R` 编号；只有实际取得并提交完成的材料使用 `E` 编号。SQLite 保存元数据、执行记录及关联，原始供应商响应和来源材料保存在 Artifact Store。网页、PDF、图片、视频来源信息和字幕保持各自材料身份。

网页正文与候选索引中的相对链接均按重定向后的页面地址和第一个 `<base href>` 解析。`text/plain` 来源保留字面文本、空行和原始文件，按行编号供引用。

`INSPECT` 对一个或多个 E 执行一个明确操作：

```python
{"targets": ["E12"], "request": {"operation": "get_frames", "params": {"times_s": [2.5, 8.0]}}}
{"targets": ["E21"], "request": {"operation": "ocr", "params": {}}}
{"targets": ["E12", "E14"], "request": {"operation": "align_media", "params": {"mode": "visual", "top_k": 5}}}
```

统一模型 Demo 启用 `get_frames`、`get_pages`、`crop`、`locate_regions`、`ocr`、`asr`、`locate_frames` 和 `align_media`。HI 直接读取图片像素、理解材料并判断证据。底层兼容适配器仍支持 `audio_qa` 和 `visual_qa`，这两个操作不在统一模型 Demo 的可用行动中。返回统一使用 `ready`、`partial` 或 `failed`，所有结果中的 E 引用都已提交到同一个 Global Store。图片、视频帧、PDF 页面和裁剪结果可继续作为后续 INSPECT 或 SEARCH 图片查询输入。

分析结果的 E 正文保存后端提供的实际 `backend`、`model` 和 `provider`，并随工具观察展示。其他工作区读取或复用这条证据时，可查看同一份识别来源及适用范围。

连续媒体范围采用左闭右开区间 `[start_s, end_s)`。区间取帧按等分中点选择区间内最近的真实帧，保留实际时间戳和去重后的数量；区间外的邻近帧不补入结果。视频 OCR、帧定位与视觉对齐同样只解码各自范围内的帧。区间内没有可用帧时返回失败，保留执行反馈，不发布空分析作为共享 Evidence；成功处理但未发现匹配仍可作为有效结果保存。

`locate_frames` 的语义或外观相似度用于排列单帧候选，不表示来自同一具体片段，也不是匹配正确率。结果 Evidence 保存采样间隔、实际后端和适用限制。未配置 CLIP 时，图片查询可用本地外观特征，含红/蓝颜色的文本查询仅测量颜色比例；其他语义查询会返回明确的能力反馈，可改用 `get_frames` 直接查看画面。跨视频的连续对应可用 `align_media` 提出时序候选，再结合实际细节核查。

视频帧时间先按容器的原始时间基减去主视频起点，再转换为秒；压缩音频切片使用同一媒体时间轴，纯音频以第一条音轨为参考。音频重采样保留末尾输出；请求范围为空、包含时间戳断点或缺少覆盖时返回解码失败，可选择实际有音频的连续范围重试。ASR 接收按媒体时间裁切的 16 kHz 音频，转写片段时间还原到原媒体时间轴。

`align_media` 的 `matches` 是候选对应关系，可重叠，不要求全部同时成立；候选数量和重叠本身不证明删减或拼接。范围、分数与锚点应结合实际内容和采样精度解释。每个锚点的 `time_b_minus_a_s` 为该点 B 时间减去 A 时间，a/b 按本次 `targets` 顺序定义；含采样误差的区间端点之差不代表精确偏移。

视觉对齐在工具内部采样最长边320像素的工作图，通过 SIFT 局部特征、FLANN 候选检索、RANSAC 几何验证和连续时间路径匹配定位片段；近乎相同的低纹理画面辅以全局外观比较。密集采样顺序解码，工作图与特征不生成 Evidence，也不送入 HI。两侧使用同一时间步长，在总计512帧、最多65,536对比较及配置的帧预算内采样。采样因预算变稀时以 `COUNT_REDUCED` 警告注明实际精度；有效完成仍为 `ready`，解码等实际失败按原规则返回 `partial` 或 `failed`。预算不足以形成片段时明确失败。

视觉与频谱序列的分数按连续路径的累计匹配收益和时间步长惩罚计算，并按本次采样规模归一化；它是候选排序分数，不是匹配正确率。次级路径的每个采样点都已有更强对应时，该次级路径不再返回；保留支持程度相当的重复位置和含新增对应的路径。视觉结果保留各候选范围、分数及最多4个数值锚点，按候选顺序保存首末完整 A/B 原帧对，整次调用最多8张且受图片预算约束。其他锚点的图片 E 引用可以为空，HI 可按源 E 与时间调用 `get_frames` 查看；数值锚点本身不代表模型已看过画面。相同来源、时间与内容的已保存原帧可复用，保留原始出处与完整性检查。边界误差保持未知，不能从采样间隔推定；低纹理、镜像、倒放或大幅编辑可能无法对齐。

已完成的相同 INSPECT 请求可复用保存的结果，执行主键关联本次审计，并在 HI 观察中明确标出复用状态。get_frames、locate_frames 和 align_media 生成的原帧在来源、定位、生成方式及图片字节相同时沿用已有 E 编号；各次定位查询的分数与结果分别保存。PDF 页面按相同规则复用，页定位包含页码和渲染 DPI。工具返回的输入材料、属性卡与处理结果分别表达。

相同参数、焦点、数据和反馈的 INSPECT 历史，在模型视图中保留最近一次及执行次数，适用于完整成功，以及有有效结果且反馈均不可重试的 `partial` 返回。有效数据和错误反馈均保留；不同结果、参数或焦点、整体失败及可重试错误仍分别展示。原始审计历史完整。

HI 的多模态输入最多8张图片，`image_input_limit` 向模型提供当前容量。一般观察按当前工作区的焦点与最新工具观察交替选取，再以分配的材料补充；重复 E 只附一次。成功的 INSPECT 复用已有结果且未产生新 E 时，剩余容量保留本 HI 上次实际附图，当前焦点和新观察优先。对齐观察按完整 A/B 锚点帧对提供图片，在容量内保留候选的首末帧对；`alignment_image_pairs` 只列本次实际附上的完整帧对。HI 显式选择的 `focus_evidence_ids` 仍受8张总上限约束。共享 Evidence 索引中的图片可由 HI 通过焦点或读取动作选择，证据库大小不等于每次模型的附图数量。原图、PDF 渲染页、视频帧和裁剪结果直接附在同一次 HI 决策请求中，保留 E 编号、来源、派生关系、页码或时间戳。每张图按 EXIF 方向显示，小图按当前附图数量分配目标分辨率并按比例插值，所有尺寸调整均受150万像素上限约束；调整后以 PNG 传输，插值不增加原图信息，原始文件保持完整。超出批次上限或无法读取的 E 会显式列出。HI 可用 `focus_evidence_ids` 保留另一侧待比较的帧或选择遗漏图片，并通过 `INSPECT` 取页、取帧或裁剪。

按 `count` 请求时，已知总帧数不超过预算的视频会按实际 PTS 返回所选范围内的全部帧；帧数未知、超过预算或经有界扫描发现声明不符时使用均匀取样。`times_s` 按指定顺序取最近帧。`COUNT_REDUCED` 说明本次选择实际返回的不同帧数，不代表源视频只有这些帧。

`get_frames` 的 `time_s` 为所取帧的真实起点，`frame_duration_s` 为容器声明的单帧持续时长，缺失或无效时为 `null`。该时长同时保存在帧 E 的定位字段中，不能用平均帧率或抽样间隔补造，也不能据此填满未观察的时间空档。目标场景的起止需结合实际帧内容与切换位置判断。帧的起点和时长有效时，GIA/HI 附图说明中的定位字段采用结构化对象，并提供 `frame_presentation_interval_s`，明确区间起止与端点包含关系。

Finding 的内容证据必须列入该条 `evidence_ids`，并通过可见性与已读内容检查。正文和限定条件可使用原件 E 说明所引内容的来源，但该条 Finding 必须同时引用实际观察过的派生证据，且来源关系必须能从真实 Evidence 的 `derived_from` 核对。原件仅作为来源时不会登记为已观察的内容，也不能扩大结论范围。编号一致和读过材料只说明引用过程有效，不能代替答案事实核验。

GIA 最终答复中的明确 F 引用会交回原模型按现有次数限制修正：核对 Finding 的 `evidence_ids` 和原始正文，再引用支持该句的 E；程序不自动替换编号或改写事实。原文摘录、代码和 F1 分数等普通术语不作为局部判断引用处理。

模型正文已明确引用且内容已查看的 E，会自动补齐到该 Finding 的 `evidence_ids` 和报告选中列表；不添加正文未引用的证据，不改写陈述。有已引用派生内容支持的原件编号会作为来源保留在正文，不放入内容证据列表；仅引用原件、未知编号或未查看的独立内容仍须修正或补读。OpenRouter 适配器使用持久化来源关系校验已观察证据，当前卡片截断不会丢失这些关系。原始模型输出保留在轨迹中。

模型决策按完整 JSON 对象解析。外层损坏、截断或同时输出多个对象时，使用原有次数与预算限制请求重新输出；不会把内部 task、Finding 或 arguments 单独当作完整决定执行。原始输出保存在轨迹中。

HI 已执行动作对应的原生推理随本 Workspace 的历史保留，并在后续请求中作为该次 assistant 消息的 `reasoning` 传入。完整推理块与图片清单、工具历史共用现有上下文限额，优先较新的块，放不下的整块省略；中断后只从本 Workspace 的审计事件恢复。它是 HI 自身可修订的分析，不构成共享 Evidence，也不进入其他 HI 或 GIA 的上下文。供应商未返回的隐藏内容及 KV cache 无法恢复。

GIA 在同一次记忆读取会话中保留自己对已见图片的原生分析，并与对应的已执行动作一同提供给后续决策。换图后仍可结合先前观察继续核查；这些分析可以修订，不是新的 Evidence。会话结束后不向其他会话或 HI 传递；被拒绝的动作不携带分析进入行动历史。分析与工具正文共用原有上下文预算，超限时先按整块省略较旧分析，并标记 `reasoning_context_omitted`，保留实际证据和报告。

GIA 的新增调查轮次与记忆读取次数分别限制。`can_create_round=false` 时仍可在剩余决策次数内读取已有记录；已存在但尚未展开的正文、后续页或本次未附的图片，可按 E 继续核查。有限结论保留实际无法补齐的缺口，图片附件容量本身不表示记录不可读取。

图片记录的正文是元数据；其读取说明与实际视觉能力一致：支持直接视觉输入时，`READ_RECORD` 可请求该 E 的图片，实际附图以 `image_inputs` 为准。PDF、视频和音频仍通过相应处理操作取得页面、帧、文字或转写。

小图附图按 `max(768, 1536 / √n)` 像素的目标长边做普通插值，`n` 为当前附图数，仍受单图像素上限约束；原始证据和定位坐标使用原图尺寸。插值不会恢复原图中不存在的细节。

仅在模型请求成功返回后，实际附图的 E 才进入已查看集合；原始模型输出的审计事件保存这些 E 编号，工作区恢复时只读取自己的事件。图中可读文字由 HI 直接理解，精确转写可使用 OCR；语音使用 ASR。专用工具输出按原有证据协议提交，HI 自身的材料解释与判断保存在其局部报告中。

跨材料取图且 HI 未指定图片焦点时，输入会保留本 HI 最近查看的另一份材料的部分图片，与新图片交替安排在现有容量内。已保存的派生关系用于区分材料，其他 HI 的图片历史不参与选择；显式焦点仍优先，未附图的 E 在输入中列明。时序对齐的锚点继续按完整 A/B 帧对提供。

格式或参数被 Harness 拒绝、或 INSPECT 返回失败时，纠错请求在当前焦点与材料之外保留该 HI 最近一次模型响应实际附带的图片，受同一图片容量限制。上次输出格式无效不影响这些像素已实际提供的事实；其他 HI 的图片历史不参与纠错输入。

`READ_RECORD` 返回媒体文件或派生图片的描述时会标明内容范围。这类元数据及空正文页不计入已查看内容，不能替代图像输入、PDF 页面读取或音视频处理。该区别在工作区恢复后仍然保留。

ASR 与音频对齐默认选择标记为默认的音轨，否则选择第一条音轨。音视频使用主视频零点，纯音频使用第一条音轨零点；默认处理实际可解码的音轨覆盖，并返回该范围，ASR 记录实际音轨编号。明确指定的时间区间必须完整覆盖，真实时间缺口不会通过补静音消除。M4A、Matroska 与 Ogg 根据实际流区分音频和视频，封面图不作为视频轨。

## 安装

```powershell
python -m pip install -e ".[test]"
python -m pip install -e ".[ocr,asr,semantic]"
python -m pip install -e ".[local-models]"
python -m pip install -e ".[free-search,dynamic]"
python -m playwright install chromium
```

## 论文能力 Demo

### 调查网站

```powershell
python -m pip install -e ".[web,free-search]"
python -m foren_searcher.web_demo --harness-root "C:/path/to/harness-1" --open
```

网站地址为 `http://127.0.0.1:8765`，仅监听本机。使用同一 OpenRouter 设置、累计费用账本、SEARCH/INSPECT 和 Harness-1 运行时。入口也可使用 `foren-searcher-web`；`--project` 指定项目根目录，`--config` 指定该项目下的设置文件，`--port` 指定端口。

案例库提供已有真实记录和实时 API 调查两个入口；查看记录不调用模型。内置目录包含 VideoSham 1004、VidDiffBench surgery_1、ChartQA 验证集样本，以及 NASA 网页和自建 PDF 并行调查。数据集来源与样本编号显示在案例卡片中；原始媒体位于 `outputs/web-demo/catalog`，已有项目案例引用位于 `web/cases.json`。未准备材料的案例明确显示不可运行，参考标注不进入模型输入。

脑图展示 GIA 决策、独立 HI、工具行动与共享证据。点击 HI 展开步骤，点击节点查看供应商实际返回的分析、实际附图、工具参数、Observation 和完整事件。节点保持可读尺寸，拖动平移、滚轮缩放；“适应画布”查看全图。已有记录可按事件回放，实时调查按实际模型响应和工具完成事件更新，不模拟 token 输出。

结果页保留模型原始结论，独立核验分别标明流程是否完成、答案是否正确。错误答案与中断记录同样可查看证据和失败原因。

自定义调查支持最多8份图片、PDF、音视频材料，每份64 MB。一次只运行一项网站调查，内部 Workspace 仍可并行。停止会取消后续工作并保存已取得的记录；已发出的请求可能仍计费，未知费用继续预留。刷新页面可回到当前调查，服务重启后的未完成运行标记为中断，不自动重发请求。

中断记录若已保存调查数据库和结果文件，可点击“继续调查”复用证据、已完成的 HI 报告和行动历史。继续会产生新的模型请求，已结算费用和未知费用预留沿用原账本。命令行使用 `python -m foren_searcher.demo run --resume --task "原调查问题" --output-dir "原调查目录" --harness-root "C:/path/to/harness-1"`，不再传入材料；完成的调查直接返回已保存结果。同一调查只允许一个运行进程，退出后可继续。

证据库可预览图片、视频、音频和 PDF，并查看来源、派生关系与正文；不可信 HTML 只作为文件下载。密钥留在本机配置中，不返回浏览器。网站读取全部轨迹用于调查拥有者查看，不改变模型角色的共享/私有边界。`outputs/web-demo/jobs` 保存独立调查；媒体及数据库须一起备份。 脑图节点显示处理模式、输入 E、时间范围及新增/复用结果；HI 的工具行动与输出修正分别计数。历史记录的复用状态来自原执行审计，原始轨迹保持完整。

当前 Demo 默认通过 OpenRouter 调用 `qwen/qwen3.8-27b`，使用 Harness-1、现有工具接口和工作区访问权限。

### 邀请试用

在项目目录生成每位试用者独立的邀请码，并启动本机试用网站：

```powershell
python -m foren_searcher.web_demo --invite "试用者姓名" --invite-days 7
python -m foren_searcher.web_demo --trial --port 8766 --harness-root "C:/path/to/harness-1"
```

访问 `http://127.0.0.1:8766`，输入命令返回的 `invite_code`。网站只监听本机；邀请码不提供外网访问。邀请码默认有效7天，登录会话最长8小时。撤销邀请使用 `python -m foren_searcher.web_demo --revoke USER_ID`，其中 `USER_ID` 为创建邀请时返回的 `id`；撤销立即阻止该用户的后续访问，已开始的调查仍保存结果。

每位试用者只能访问自己的上传材料、调查记录、事件和证据；案例库中明确提供的示例记录共用。每人24小时内默认最多开始3次调查，可用 `--daily-runs` 调整；失败、停止和手动继续的每次运行均计入次数。每份上传材料最多64 MB，每位试用者累计上传空间512 MB。网站一次运行一项调查；请保持单个网站进程运行。

试用版沿用现有模型和累计费用账本，材料会提交给已配置的模型服务。邀请码和会话只在 `.local/trial-access.sqlite3` 保存随机凭证摘要；试用材料和记录位于 `outputs/web-trial`。备份时保留访问数据库及对应调查目录。邀请码由邀请人通过适合的私密渠道分发。

### OpenRouter

在项目目录打开密钥设置窗口：

```powershell
python -m foren_searcher.openrouter_settings
```

填写 API Key 并保存。Windows 使用当前用户的 DPAPI 加密保存到 `.local/openrouter.json`；该目录不进入 Git。也支持从环境变量 `OPENROUTER_API_KEY` 读取密钥，环境变量优先。不要把密钥放进任务文本。其他操作系统使用环境变量提供密钥。

设置窗口中的模型 ID 默认为 `qwen/qwen3.8-27b`。GIA 负责全局规划和汇总，全部 HI 共用同一个模型客户端；角色提示、任务、行动历史与每次多模态输入分别构造。HI 请求直接包含所选图片像素，PDF 通过 `get_pages` 渲染，视频通过 `get_frames` 或 `locate_frames` 取帧。GIA 依据全局可见记录及 HI 报告汇总；每次供应商请求独立，没有跨 Workspace 的会话缓存。

```powershell
$env:FOREN_HARNESS_ROOT = "C:\path\to\harness-1"
python -m foren_searcher.demo check --model-backend openrouter --search ddgs
python -m foren_searcher.demo run --model-backend openrouter --search ddgs --task "查找 NASA 官方原始来源，核实 Ingenuity 首飞日期与飞行时长，提供支持各项事实的原文短摘。"
```

图片、音频、视频和 PDF 可通过 `--media 文件路径` 输入。`--openrouter-config` 指定设置文件，默认 `.local/openrouter.json`。首次连接读取模型目录并校验 Key，保存设置和 `check` 不调用付费推理。

新配置的项目累计预算初始值为 **1 美元**，可在设置窗口修改。GIA、全部 HI、相同配置目录下的重复运行和并发调用共用 `openrouter-usage.sqlite3`。请求前按完整上下文和输出上限预留费用，成功后按 `usage.cost` 结算；超时、缺失账单和中断的未知费用保留预留金额，不自动重发。预算不足时停止新增模型请求，保存已取得的证据。不要删除费用数据库来重试。建议在 OpenRouter 同时设置 Key 的账户额度上限；本地记录不涵盖其他应用消费。

推理接口明确返回 HTTP 400（参数拒绝）或 HTTP 404（路由拒绝）时，该请求记为 `rejected`，推理费用为零，不继续占用预留。400 不自动重发，调用方可修正参数后发起新请求；404 会停止本次客户端的后续模型调用，检查供应商、精度及输入支持后重新运行。只读查询的错误不改变推理费用；HTTP 200 正文内的错误仍按实际账单结算，未知费用继续保留预留。已有账单和其他请求不被覆盖。[OpenRouter 错误说明](https://openrouter.ai/docs/api_reference/errors-and-debugging)、[失败请求计费说明](https://openrouter.zendesk.com/hc/en-us/articles/51693138951451-Was-I-charged-for-a-failed-errored-or-empty-response-Zero-Completion-Insurance)。

响应开始前的 `ConnectError`、`ConnectTimeout` 或 `PoolTimeout` 表示未能建立或取得发送所需的连接，使用同一请求内容和费用预留重试一次；两次均未发送时按零费用结束并释放预留。发送或接收阶段的错误、响应开始后的异常仍保留未知费用；历史未知记录不根据错误名称批量改写。[HTTPX 异常分类](https://www.python-httpx.org/exceptions/)。

响应头提供有效的 X-Generation-Id 时，客户端在读取正文前将其写入本次费用记录的 generation_id。随后即使超时或正文损坏，该编号仍可用于[查询单次请求账单](https://openrouter.ai/docs/api/api-reference/generations/get-generation)。编号本身不证明请求免费；费用未确认时继续占用原预留，已有账单和其他请求记录保持独立。

新配置采用 FP8，限定 Parasail、AkashML 供应商并按价格排序，输入与输出价格上限分别为每百万 token **0.25 / 2.55 美元**，不允许额外按请求或按图收费；无符合条件的端点就返回错误。每次调用输出上限为8,192 token，包含思考 token。`input_price_cap_per_million`、`output_price_cap_per_million`、`quantizations` 与 `providers` 可显式配置，更换模型不会自动提高这些限制。`result.json` 的 `model_usage` 记录本次实际费用、未知费用预留、累计预算及供应商，不包含密钥。GIA 与 HI 的请求按同一账本主键统计，总预算不会因角色或 Workspace 数量增加。

显式指定 `providers` 时，客户端核验符合价格和精度要求的端点能力。只有同一可用端点支持所需推理参数及输出格式时才发送结构化输出参数；其余情况使用文本 JSON，并由既有解析器与 Harness 校验。默认的 `temperature`、`top_p` 也必须与所选输出格式和推理参数在同一端点上兼容，否则省略该采样参数。路由始终保留 `require_parameters`、价格和精度限制。

将输入、输出单价上限同时设为 `0` 可限定免费路由；模型目录和请求路由均核验价格，已有付费请求的费用预留继续保留。付费与免费调用收到 HTTP 429 或响应正文中的429错误后，同一步模型决策自动重试最多2次，累计等待最多180秒。采用可解析的 `Retry-After` 与 `X-RateLimit-Reset` 中较晚的时刻，没有有效提示时等待30秒；供应商要求的等待时间超出剩余等待时限时结束重试，不提前请求。GIA与HI在同一客户端中共享冷却时间；等待不新增费用预留，可被停止操作或工作区时限取消。每次实际重发独立检查和预留预算，保留此前请求的已知费用及未知预留。网页显示限流等待、重试和终止事件。超时及无法确认结果的网络故障保持原有处理，不纳入429重试。只读查询不受此等待状态影响；错误信息不包含供应商正文。[限流重试说明](https://openrouter.ai/docs/api_reference/errors-and-debugging#retry-after-header)。

模型决策的 JSON 语法错误会返回具体行列和字符位置。输出达到长度上限、正文为空或原生决策格式无效时，也交回同一模型修正一次；仍不可执行则返回原有的可恢复输出反馈，受调查轮次和预算限制。每次输出及已知费用分别记录，不提高输出上限，不执行被截断的动作，即使截断文本碰巧构成合法 JSON。网络故障和费用不明的请求不纳入格式修正。

完整 HI 报告若仅在解析器指向的对象结尾前缺少一个数组闭合符 `]`，可补齐该标点后继续原有角色及证据校验。该恢复要求报告字段完整且无歧义，不补写文字、数字或引用，不恢复未结束的输出，也不用于动作参数。原始模型文本始终保留。

`structured_outputs` 默认为 `false`：模型按既定接口生成 JSON，供应商不施加 JSON 解码约束；解析后的动作、引用与报告仍须通过运行时校验。完整 FINISH 文本中的漏转义引号可按原字恢复，原始输出仍保存在轨迹中；缺失结尾、混入额外字段或行动参数不使用这项恢复。显式设为 `true` 时，优先采用允许端点支持的 JSON Schema；不支持该格式而支持工具调用与函数选择时，通过一个原生函数提交原有决策对象；其余端点采用可用的普通 JSON 或文本输出。初始规划、HI和GIA汇总遵循相同设置。原生函数仅负责传输决策，不会增加HI行动或改变报告字段。

HI 的最后一次决策（`remaining_turns=0`）只接受 Workspace Report。格式定义和角色校验共同遵守这一轮次限制；证据不足时以 `INCOMPLETE` 说明缺口。报告仍须通过原有引用与可见性检查，不能引用尚未读取的内容；校验修正次数用尽后保存失败状态。

API解码schema保留完整工具参数、互斥条件和批次大小。视觉目标必须互不重复：该要求保留在模型可见工具定义中，由原有参数解析器在执行前强制校验；供应商解码层采用文字说明，以兼容不能编译`uniqueItems`的端点。所有输出模式的角色字段、证据引用、完整工具参数和访问权限均由Policy与Harness校验。完整文本响应无法解析为JSON时，允许一次格式纠正调用，两次响应与费用均记录；网络失败、空响应、截断响应或无效原生函数不自动重发。

工作区已经执行但未能正常提交报告时，GIA可以依据已保存证据给出有限结论或说明无法核实。未开始调查时仍须先分配工作区，失败状态不会使不存在或未展示的证据成为可用引用。

仅向模型目录声明支持推理参数的模型发送 `reasoning`；新配置的 `reasoning_effort` 为 `medium`，支持关闭推理的模型可设为 `none`，实际设置记入费用报告。单次 HTTP 响应在数据批次间检查 180 秒期限，网络读取间隔上限为 30 秒。确认请求尚未发出的建连或连接池失败重试一次，使用同一请求内容和费用预留；两次均未发出时释放该笔预留。写入、读取、已开始响应或费用不明的失败不自动重发，未知费用保留预留。模型目录和 Key 状态的只读查询遇到网络传输失败时重试一次。

[Qwen3.8-27B 官方权重](https://huggingface.co/Qwen/Qwen3.8-27B)为27B稠密视觉语言模型，采用 Apache-2.0 许可，支持图像和视频理解。当前工程通过托管 API 运行，没有训练，也没有验证服务器上的权重、LoRA 加载、分布式显存或吞吐。8×80 GB A100 的部署与训练仍需在服务器上单独测量；托管端点的量化和推理实现可能与服务器不同。当前验证结果见[代码与框架审查报告](代码与框架审查报告.md)。

### 本地兼容适配器

显式 `--model-backend local` 连接现有本地适配器，使用 Qwen3-14B 的 4-bit 量化权重；GIA 与 HI 共享权重实例，各自使用独立输入和工作记忆。下载命令只在显式执行时联网，该适配器的调查推理使用本地模型，不属于当前 Qwen3.8-27B Demo 配置。

本地生成器的 `LocalModelConfig.max_batch_size` 默认为1。设为2时，同一模型队列中已就绪的HI请求可一起生成；每个HI保留独立图片、JSON约束、纠错次数与取消状态。成功完成多 HI 批次后，后续不足一批的请求可等待上一批推理耗时的5%，最多2秒；已排队时间计入等待，批次满额即执行。首次调用、单条生成或生成失败后不主动等待收批，独立GIA调用和服务关闭会结束收批等待；可选的轮内复核不打断HI收批。等待时间扣入各HI原有预算。GIA继续单独执行，整批生成结束前模型锁保持占用。每个HI在自身输出结束并通过解析后即可返回，不等待同批其他HI继续生成或JSON重试；批量显存不足或输入超长时逐项回退，单个请求仍受原有上下文、队列和超时限制。批量生成可能改变输出长度及采样结果，实际收益需按任务测量。

本地代理使用 CUDA 时，GIA、HI 与需要 GPU 模型的 INSPECT 操作共用模型锁。工具调用结束或失败后释放闲置 GPU 模型缓存，为后续代理推理留出显存；再次调用时重新加载工具模型。CPU 模型、取帧、取页及其他不使用 GPU 模型的操作可独立执行。重复 GPU 工具调用包含模型重新加载的耗时。

思考模式使用模型 `generation_config.json` 中的采样参数，JSON 外壳受格式约束。默认单次推理最多 1,024 个思考 token、总输出最多 4,096 个 token；GIA 单次调用时限为 300 秒，Workspace 时限为 900 秒。实际耗时受任务拆分、查询数量与页面可用性影响。

Qwen3.5 在 CUDA 上使用 PyTorch 后备线性注意力时，单 token 增量解码直接从已有缓存状态计算输出，省去 CPU 输出缓冲及其设备拷贝。多 token、无缓存、预填充和已有融合内核路径使用模型原实现；缓存状态与归一化沿用原始计算精度。

INSPECT 解码约束使用既有的 `targets`、`request.operation`、`request.params` 层级；操作的具体参数仍由完整工具 schema 校验。视觉问答的每张图片观察附带其 Evidence 来源和位置，视频帧保留实际时间；这些信息不等于模型已正确判断时间顺序。

视觉对齐根据两侧帧率选择共同采样频率，两侧总计最多处理 512 帧、最多比较 65,536 对帧，并受 `maximum_frames` 限制。短视频可利用原始帧率；长视频与低帧率素材仍可能漏掉很短的片段。每个候选的锚点数随视觉问答批次上限调整，默认最多4对，底层匹配仍使用全部采样帧。采样间隔、边界不确定性和算法限制保存在分析 Evidence 正文中，并呈现在执行 HI 的 Observation 中。相似静态画面的对应位置可能不唯一，未匹配空档不能直接当作完整删减区间。

```powershell
foren-searcher-models download --profile demo --model-root .models --include-audio
```

配置 Harness-1 源码目录。每个 Workspace 使用原始 Harness-1 的 `EpisodeEnv`、`ToolExecutor` 和独立任务状态；模型仍直接提交 Workspace Report。

```powershell
$env:FOREN_HARNESS_ROOT = "C:\path\to\harness-1"
foren-searcher-demo check --search ddgs
```

默认搜索不需要 API 密钥：文本通过 DDGS 优先访问 Brave，无结果或访问失败时在同一请求时限内尝试 Bing、Yandex；图片使用 Bing，视频使用 DuckDuckGo。网页正文仍需实际下载后形成 Evidence。免费服务可能限流或无结果，失败会作为工具反馈保留，模型须据此调整查询或说明缺口。

自建 SearXNG 可以通过 `SEARXNG_BASE_URL` 与 `--search searxng` 指定。`--search serpapi` 需要单独配置 API 密钥。反向图片搜索默认使用 Yandex；`FOREN_REVERSE_IMAGE=none` 可关闭图片上传。

`check` 仅检查本地部署配置，不能证明自然任务质量或实时搜索可用性。每次 `run` 需要独立的空输出目录。

纯网页调查：

```powershell
foren-searcher-demo run `
  --model-root .models `
  --search ddgs `
  --task "搜索官方来源，核实某一公开事件的日期与关键事实，并说明证据" `
  --output-dir outputs/demo/web-case
```

图片或音频调查：

```powershell
foren-searcher-demo run `
  --model-root .models `
  --search ddgs `
  --task "检查这张图片，判断人物可能的职业并说明可见依据" `
  --media C:\path\to\public-image.png `
  --output-dir outputs/demo/image-case

foren-searcher-demo run `
  --model-root .models `
  --search ddgs `
  --task "检查这段音频，提取关键句并概括内容" `
  --media C:\path\to\public-audio.wav `
  --output-dir outputs/demo/audio-case
```

每次运行使用独立目录，交付文件包括 `report.md`、`result.json`、`trajectory.jsonl`、SQLite Store 和 Artifact 目录。`report.md` 适合演示结果与引用；`result.json` 包含 Workspace 报告、候选来源、Evidence 索引、健康检查和信息边界；`trajectory.jsonl` 含完整模型输入、工具观察和错误，属于敏感审计材料，不应公开展示。

信息可见性按以下边界执行：已经提交的 Evidence 卡片、E 编号与可按需读取的完整记录在 Workspace 之间共享，未被报告选中的 E 也保留；E 的来源 R 可沿出处关系读取。Working Memory、焦点、未打开的 R、自己的 Finding 与 Report 对同级 Workspace 私有。GIA 可以读取全部 Workspace 的 Finding 与 Report，HI 只能读取自己的判断记录。当前论文能力 Demo 不包含 `DETECT`。

`QUERY_MEMORY` 对 Evidence 全文、Finding 的完整陈述和限制、报告完整内容，以及来源的完整检索说明与最终地址进行词面匹配，返回简短卡片。连续短语匹配优先于分散关键词；短语忽略大小写和连续空白，词内片段不作为独立英文词命中。中文等连续文字使用双字符片段辅助检索，关键词顺序不同的相关记录仍可返回。标题和正文不会拼成一个短语命中。检索分数不是事实置信度，也不证明引文准确或来源可靠。

卡片中的省略号表示内容被截断。`READ_RECORD` 可取得完整记录；较长的 R、Finding 和报告以 JSON 文本放在 `content` 中分页，使用返回的 `next_offset` 继续读取。R 的 `final_url` 是每页返回的当前最终跳转地址。图片随 HI 请求直接输入；PDF 页面和视频帧先由 `INSPECT` 提取，再进入 HI 的多模态输入。

HI 的记忆动作在 `items` 中提示读取位置或查询页状态，完整结果保存在 `details` 中。正文与分页指针只传递一份；恢复已有工作区时采用相同表示，原始轨迹保持不变。超出配置大小上限的观察仍会明确标记截断。

OpenRouter Demo 对上下文容量至少为131,072 token的控制模型采用每页12,000字符、每次观察20,000字符、总视图80,000字符的容量；较小或未知上下文采用6,000、10,000和40,000字符。较长正文通过 `next_offset` 连续读取。

## 本地开发 Smoke Test

受控 Smoke Test 配置使用 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) 作为 GIA，使用 [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) 作为 HI。前者负责单轮规划和报告汇总，后者负责 Workspace 内的动作选择与证据报告。模型均通过 `GiaPolicy`、`HiPolicy` 接口注入，主框架不依赖 Qwen 专用类型。

权重只在显式执行下载命令时联网；调查运行始终使用本地目录。下载器固定到已经通过本仓库端到端验收的 Hugging Face 提交版本，避免远端 `main` 更新导致同一份代码取得不同权重：

```powershell
foren-searcher-models download --profile smoke --model-root .models
```

受控端到端检查会实际加载两个模型，并依次验证 GIA、HI、SEARCH、INSPECT、READ_RECORD、报告提交、最终引用与 Store 健康状态：

```powershell
python -m foren_searcher.local_smoke --model-root .models --suite

$env:RUN_LOCAL_MODEL_SMOKE = "1"
$env:FOREN_LOCAL_MODEL_ROOT = ".models"
python -m pytest tests/test_local_models_live.py -q -m "live and gpu"
```

每次命令运行使用新的 `outputs/local-model-smoke/<时间>` 目录，并保存 `summary.json`、每个场景的 SQLite Store 和 Artifact。`--suite` 依次运行多媒体工具链和纯文本检索链，并在同一对已加载模型上验证实例复用、Store 隔离、报告提交、最终引用与模型运行指标。该受控检查证明两个本地模型与完整控制流能够协作，不代表 0.6B/1.7B 模型已经达到真实取证研究所需的结论质量。用于真实任务前，应在目标语言、来源类型和证据难度的数据集上评估任务分解、工具调用成功率、引用正确率和最终答案准确率。

代码中可以这样组装：

```python
from foren_searcher import InvestigationRuntime, MemoryPolicy, create_qwen_local_policies

policies = create_qwen_local_policies(".models")
await policies.preload()
try:
    runtime = InvestigationRuntime(
        store=store,
        gia_policy=policies.gia_policy,
        hi_policy=policies.hi_policy,
        search_service=search,
        inspect_service=inspect,
        memory_policy=MemoryPolicy(
            max_model_view_chars=20_000,
            max_observation_chars=6_000,
        ),
        max_concurrency=1,
    )
    result = await runtime.run_new(
        task=task,
        claim_materials=claim_materials,
        media=media,
    )
finally:
    await policies.aclose()
```

同一个本地 HI 模型会服务所有 Workspace。生成器内部会在异步边界串行化 GPU 推理，默认最多接受 4 个在途请求，超过时抛出 `LocalModelBusyError`，不会无限创建等待线程。调用方取消时，在模型层之间和解码步之间检查停止信号，并等待工作线程退出后释放推理槽；已开始的单个底层算子仍需返回，不能承诺任意设备负载下的固定取消时延。`aclose()` 会停止接收新请求、等待当前推理结束并卸载权重，关闭后的实例不能重新加载。本地参考配置应保持 `max_concurrency=1`；若换成支持批处理或多个独立副本的推理服务器，再提高 Workspace 并发数。输入超过本地模型 token 预算时会抛出 `LocalModelContextError`，不会静默截断证据。

GIA/HI 输出在进入 Harness 前会先执行角色级状态校验。非法接口结构、使用不可见的材料或 Evidence、引用未读取的正文、错误 Workspace ID 等情况先在适配器内纠正一次；仍失败时，`LocalModelOutputError` 作为 `PolicyOutputError` 交回原有调查循环，模型可依据 Harness rejection 修正参数或补读材料。循环始终受轮数与超时限制；加载、上下文溢出和其他运行错误仍按失败处理。材料和网页中的指令被明确视为不可信数据，不能改变角色、接口或可用动作。`gia_model.metrics()` 与 `hi_model.metrics()` 提供不含提示词和生成内容的请求数、失败数、取消数、排队拒绝数、token 数、推理耗时和排队耗时。`queue_seconds` 累计各请求等待推理槽的时间，包含在队列中取消的请求；多个请求的等待可能重叠，不能把累计排队时间直接加到整次调查的墙钟时间上。

GIA 和 HI 共用同一本地生成器时，轮内 GIA 检查会在常规模型请求到达后让出资源。排队中的检查直接取消，已开始的检查先停止工作线程，再让后续请求进入推理；重复取消不会提前释放推理槽。调查轨迹以 `round_review_deferred` 记录这种调度，不生成 WAIT 或 FINISH 决定，也不发布已停止生成的输出或附图完成记录。新报告可再次触发检查，本轮结束后仍进行常规汇总。GIA 首次分配和最终汇总按常规模型请求执行。

本地生成器在 CUDA 可用、PyTorch 未编译 Flash Attention 且模型使用 SDPA 时，自动展开分组 K/V 后沿用 SDPA 的注意力与遮罩处理，减少长上下文退回高内存计算路径的情况。该兼容处理不截断输入，保留逐请求生成缓存；实际显存与速度仍取决于模型、输入长度及设备负载。

## 组装 SEARCH

```python
import asyncio
import os

from foren_searcher.search import create_serpapi_search_service
from foren_searcher.store import GlobalStore


async def main() -> None:
    store = GlobalStore("investigation/store.sqlite3", "investigation/artifacts")
    search = create_serpapi_search_service(
        store=store,
        api_key=os.environ["SERPAPI_API_KEY"],
        enable_dynamic_pages=True,
        allow_image_uploads=True,
    )
    try:
        response = await search.execute(
            {"input_type": "text", "query": "site:iana.org example domains", "output_type": "text"},
            workspace_id="W1-1",
        )
        print(response.to_dict())
    finally:
        await search.aclose()


asyncio.run(main())
```

## 组装 INSPECT

```python
import asyncio

from foren_searcher.inspect import create_inspect_service
from foren_searcher.store import GlobalStore


async def inspect_example() -> None:
    store = GlobalStore("investigation/store.sqlite3", "investigation/artifacts")
    image = store.commit_ingested_evidence(
        source_ref="I0",
        media_type="image/png",
        modalities=("image",),
        label="待检查图片",
        original_filename="input.png",
        path="input.png",
    )
    inspect = create_inspect_service(store=store)
    try:
        response = await inspect.execute(
            {"targets": [image.evidence_id], "request": {"operation": "crop", "params": {"box": [0.25, 0.25, 0.75, 0.75]}}},
            workspace_id="W1-1",
        )
        print(response.to_dict())
    finally:
        await inspect.aclose()


asyncio.run(inspect_example())
```

默认本地适配器使用 Pillow、OpenCV、PyMuPDF、PyAV、NumPy 和 SciPy。OCR 使用 RapidOCR；ASR、开放式问答与语义定位使用本地模型，并通过下列环境变量启用：

```powershell
$env:FOREN_WHISPER_MODEL = "C:\models\faster-whisper-small"
$env:FOREN_CLIP_MODEL = ".models\inspect\clip-vit-base-patch32"
$env:FOREN_GROUNDING_MODEL = ".models\inspect\grounding-dino-tiny"
$env:FOREN_VLM_MODEL = ".models\inspect\Qwen3-VL-2B-Instruct"
$env:FOREN_AUDIO_MODEL = ".models\inspect\Qwen2.5-Omni-3B"
```

这些模型只从本地 Hugging Face 缓存加载，不在处理请求时联网下载。未配置语义模型时，视觉和音频操作只返回其可验证的颜色、结构、能量和频率能力，并在 `limitations` 中标明范围。生产部署若需要开放式问题、任意物体区域定位或自然语言视频帧检索，应配置相应模型。

本地视觉问答达到输出长度上限且未生成结束标记时，返回 `RESOURCE_LIMIT`，不保存未完成的分析结果。HI 可依据反馈缩小问题范围或在报告中说明缺口；完整生成的回答仍需核对事实与证据是否一致。

本地模型分析结果中的 `model` 保留仓库标识或模型目录名，不包含本机父目录。已保存的历史分析与轨迹保留原内容。

INSPECT 可从已保存的视频来源页取得实际媒体。直接视频 URL 由受控 HTTP 客户端流式下载；YouTube 和 HTML5MediaEmbed 来源可使用 SEARCH 保存的 yt-dlp 元数据，优先选择同时含音视频的 HTTPS 渐进式格式，也支持完整、未加密的 HLS 点播流。HLS 支持 TS、fMP4、字节范围和独立音轨；最多四个资源并发下载，共用实际字节总预算，在本地子进程中流复制封装。每个远程请求仍经过地址及重定向检查，解码器只读取本次已下载的文件。

取得的视频、来源清单、媒体属性和派生帧保留对应记录；帧定位使用实际解码时间，清单声明时长与容器探测时长分别保存。超时或失败会清理本次暂存文件。私网地址、登录资源、直播、加密 HLS、DASH、跨分段编码参数变化或未支持的平台返回明确失败。获取覆盖范围与识别质量需分别核验；找到来源页不表示视频已经取得。

`InspectPolicy` 默认允许处理至多 2 GiB 的已有文件；单次网络获取总预算为 384 MiB，获取时限为 180 秒，整次操作时限为 300 秒。`maximum_download_bytes` 与 `maximum_file_bytes` 分别控制下载量和可处理文件大小；下载同时遵守两者的较小值。HLS 清晰度按来源声明的带宽和时长估算选择，实际下载量逐字节累计。

yt-dlp 的视频元数据与可选文字搜视频请求也使用受限 HTTP 客户端，包括嵌入媒体和跳转。提取器按需读取响应，大视频只读取元数据所需的部分；读取字节、响应头和总时限均受限制，取消后关闭响应并停止后续请求。公开接口需要的 POST、Cookie 和 HTTP 错误响应可由提取器处理，跨域跳转移除显式凭据。

不需要付费搜索 API 时，可连接自己部署的 SearXNG。实例必须在 `settings.yml` 的
`search.formats` 中启用 `json`：

```python
from foren_searcher.search import create_searxng_search_service

search = create_searxng_search_service(
    store=store,
    base_url="http://127.0.0.1:8080/",
    language="zh-CN",
    safe_search=1,
    enable_dynamic_pages=True,
)
```

该免费路线支持文字查询得到网页、图片或视频候选。SearXNG 没有统一的上传图片反向搜索接口，
因此单独使用该工厂时，`image` 和 `text_image` 输入会明确返回 `ACCESS_RESTRICTED`；直接打开已知图片 URL 不受影响。

需要同时支持免费文字搜索和上传图片反向搜索时，使用组合工厂：

```python
from foren_searcher.search import create_free_search_service

search = create_free_search_service(
    store=store,
    searxng_base_url="http://127.0.0.1:8080/",
    reverse_engine="yandex",
    allow_image_uploads=True,
)
```

`yandex` 适合通用网页来源，并支持 `image` 与带文字提示的 `text_image` 输入；
`iqdb-artwork` 面向动漫和插画，`iqdb-3d` 面向其独立的 3D/实拍索引，两者仅接受纯图片输入。
Yandex 路线使用其公开网页上传流程，不需要密钥但没有正式 API 的稳定性承诺。所有上传仍会写入
`search_image_uploads` 审计记录；不允许外传调查图片的部署必须设置 `allow_image_uploads=False`。

默认网络策略只允许公网 HTTP/HTTPS，逐次检查重定向，限制响应大小，不复用日常浏览器登录状态。动态页面处理需要显式启用。
禁止向第三方发送调查图片的部署应设置 `allow_image_uploads=False`；图片查询会返回明确的 `ACCESS_RESTRICTED`，不会退化成文字描述查询。

## SEARCH 生产部署

SEARCH 的回归测试覆盖门面、来源取得、R/E/Artifact 持久化、超时与取消、并发复用、安全边界和审计记录。外部搜索引擎的排名、覆盖率和可用性由所选供应商决定，实际任务仍需检查检索结果的相关性。

SEARCH 服务支持异步上下文管理；进程停止时会拒绝新调用、等待在途调用完成，再关闭 HTTP 与供应商客户端：

```python
async with create_free_search_service(
    store=store,
    searxng_base_url="https://search.internal.example/",
    allow_image_uploads=False,
) as search:
    response = await search.execute(request, workspace_id="W1-1")
```

`create_search_service` 默认取得传入 provider 的生命周期所有权；共享 provider 的调用方设置 `close_provider=False`，并在应用退出时自行关闭它。

生产环境需要满足以下运行条件：

- SQLite 文件和 Artifact 目录使用持久化本地磁盘，并作为同一个调查数据集备份；备份期间暂停写入，或使用 SQLite 在线备份配合同一时点的 Artifact 快照。不要把 SQLite 放在不保证文件锁语义的网络文件系统上。
- 一个数据库由一个长生命周期 `SearchService` 负责来源取得。调用开始时复用已经保存完整的材料；尚未取得结果的同一 R 请求各自执行，可能产生少量重复材料，后续调用沿用已保存的 E。多进程扩展按调查分片数据库。
- 当前 Store schema 版本为 3；旧 Store 会在首次由 v3 打开时按当时已有的 Artifact 字节补充 SHA-256 摘要，新于当前程序的 schema 会在启动时明确拒绝。升级前先从可信快照核对并备份数据库与 Artifact 目录。
- 默认公网 URL 检查、重定向检查和响应上限保持启用。启用 Playwright 时，将运行进程放在禁止访问云元数据地址和内网管理面的网络沙箱中，以覆盖 DNS 重绑定等浏览器网络栈无法完全钉住的风险。
- API 密钥只从部署环境注入，不写入请求、日志或 Store。标准日志只记录 execution ID、停止原因、数量和耗时；查询正文与上传图片不写日志。
- 自建 SearXNG 或具备正式 SLA 的供应商用于稳定文字检索。Yandex 与 IQDB 免费反向搜图适配器基于公开网页流程，适合作为 best-effort 路线，但第三方页面变化、限流和可用性不属于本项目可保证的生产 SLA。
- 对外层调用施加身份认证、Workspace 授权、速率限制和磁盘配额；这些属于承载 SEARCH 的应用边界，不由模型侧 SEARCH 接口新增参数。

每次执行会持久化总耗时、供应商调用、候选发现、来源打开及重试结果。调用方取消仍会结束执行记录并保留已经提交的材料；Artifact、E 和 R→E 关系继续以事务方式提交。

查询的成功来源数量按最终 R 去重。多个候选网址跳转到同一来源时只计一次；尚未达到来源目标且仍有候选预算时继续获取其他来源。

静态响应和浏览器渲染结果均检查软失败。已识别的人机验证、登录限制或不存在页面不登记为正文 Evidence，也不计入成功来源数；HTTP 200 本身不表示已经取得所需正文。

来源打开器的异常按对应 R 返回，其他来源取得的 E 仍在响应中保留。原生 TLS 证书校验失败返回 `fetch/AUTH_ERROR`；证书验证、重定向和公网地址检查始终生效。

HTML 正文抽取过滤显式隐藏的元素和未渲染的 template 内容。只含隐藏导航的页面不计为取得正文；配置动态页面支持时，仍使用原始页面进行渲染。
普通段落的源码折行按 HTML 文本规则合并，避免把一句引文拆成多个编号段落；预格式化内容与显式换行保留。

发布检查：

```powershell
python -m mypy src
python -m pytest -q
python -m pytest tests/test_search_sources.py -q -m "dynamic"
python -m build
$env:RUN_LOCAL_MODEL_SMOKE = "1"
python -m pytest tests/test_local_models_live.py -q -s -o addopts=""
```

仓库的 `release-checks` CI 会在 Windows 和 Linux、Python 3.11 和 3.12 上执行离线测试、类型检查和构建。正式部署还应在目标 GPU 上运行本地模型 suite，并在目标部署网络运行下方对应的真实服务 Smoke Test。Playwright 路线还需在镜像构建阶段执行 `python -m playwright install chromium`。

## 测试

```powershell
python -m pytest -q

$env:RUN_LIVE_SEARCH_TESTS = "1"
python -m pytest tests/test_search_live.py -q -m "live and not searxng and not serpapi"

$env:SEARXNG_BASE_URL = "http://127.0.0.1:8080/"
python -m pytest tests/test_search_live.py -q -m "searxng"

$env:SERPAPI_API_KEY = "..."
python -m pytest tests/test_search_live.py -q -m "serpapi"

python -m pytest tests/test_search_live.py -q -m "live and reverse_image"

$env:FOREN_ASR_SAMPLE = "C:\path\to\public-speech.wav"
$env:FOREN_WHISPER_MODEL = "C:\path\to\faster-whisper-model"
$env:FOREN_PUBLIC_IMAGE = "C:\path\to\public-image.png"
$env:FOREN_PUBLIC_PDF = "C:\path\to\public-document.pdf"
$env:FOREN_PUBLIC_VIDEO = "C:\path\to\public-video.avi"
$env:FOREN_PUBLIC_OCR_IMAGE = "C:\path\to\public-ocr-image.jpg"
$env:FOREN_AUDIO_EVENT_SAMPLE = "C:\path\to\public-audio-event.wav"
$env:FOREN_AUDIO_EVENT_EXPECTED = "trumpet,horn,brass"
$env:FOREN_AUDIO_EVENT_QUESTION = "Which musical instrument is playing? Answer with one word."
$env:FOREN_CLIP_MODEL = ".models\inspect\clip-vit-base-patch32"
$env:FOREN_GROUNDING_MODEL = ".models\inspect\grounding-dino-tiny"
$env:FOREN_VLM_MODEL = ".models\inspect\Qwen3-VL-2B-Instruct"
$env:FOREN_AUDIO_MODEL = ".models\inspect\Qwen2.5-Omni-3B"
python -m pytest tests/test_inspect_live.py tests/test_inspect_semantic_live.py tests/test_inspect_acquisition.py -q -m "live"
```

普通测试默认排除标记为`live`的外部网络检查，因此不会把缺少外部服务配置显示成测试跳过。普通测试全部离线可重复，
并通过本地真实HTTP服务器覆盖SearXNG JSON请求到来源保存的完整链路。真实网络测试覆盖公开 HTML、PDF、图片、JavaScript 页面、视频来源和文字搜视频；
SearXNG Smoke Test 需要启用 JSON 的实例，SerpApi Smoke Test 仅在提供 `SERPAPI_API_KEY` 时运行。
反向搜图 Smoke Test 只会上传仓库中声明的公开测试图片。

INSPECT 普通回归使用实际生成的 PNG、PDF、WAV 和 AVI 文件，覆盖取帧、取页、裁剪、OCR、区域/帧定位、视觉/音频问答、视觉/音频对齐、跨 Workspace E 复用、损坏材料、越界、部分结果和超时。外部语音样本与本地大模型只在显式运行 live 能力测试时使用。
