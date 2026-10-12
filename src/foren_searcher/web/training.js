"use strict";
(() => {
  const page = document.querySelector('#training-page');
  const T = {data:null, section:'flow', stage:2, walkerCase:'vs1007', filter:'all', query:'', dataset:'all', sample:null, step:null, stepFilter:'all', tab:'overview', archNode:'gia', archZoom:1, archX:0, archY:0, cache:new Map(), serial:0, playing:false, timer:null, map:null, mobilePane:'map'};
  const labels = {SEARCH:'搜索与打开来源',INSPECT:'检查原始材料',READ_RECORD:'读取已有证据',QUERY_MEMORY:'查找调查记忆',SEARCH_MEMORY:'查找调查记忆',INVESTIGATE:'安排下一轮调查',FINISH:'结束并回答',REPORT:'提交局部报告'};
  const operations = {get_frames:'提取视频帧',align_media:'对齐视频片段',get_pages:'读取文档页面',crop:'裁剪局部画面',ocr:'读取图中文字',locate_frames:'定位视频片段'};
  const dispositions = {accept_candidate:'可作为候选',repair_reexecute:'需要重新执行',context_only:'仅保留为历史',exclude:'排除目标',unknown:'尚未判定'};
  const modes = {bridge:'GIA × HI 真实协作',hi_independent:'HI 独立调查',gia_offline:'GIA 离线决策'};
  const stageData = [
    {title:'案例选取与依据准备',short:'案例与依据',subtitle:'先界定问题边界，再准备核验材料。',input:'真实问题、原始媒体、来源记录与数据集标注',actor:'数据构造',output:'明确的任务 + 训练侧参考材料',points:['按原始案例、事件或视频来源划分开发与保留检查数据，同一案例的变体放在同一侧。','核对原始材料、时间轴和来源条件。给定候选视频、给定网页与自主搜索分别记录。','参考标注辅助核验；不把缺失证据补写成已经观察过的事实。'],example:'video-correspondence',exampleLabel:'查看双视频对应任务',note:'对应片段可证明内容匹配，但不足以单独判定原创归属。',upgrade:'案例按数据集标注统一入库，每个案例附带来源政策与必需证据；开发集与保留检查集按案例家族划分，并由程序自动核查两侧是否重叠。'},
    {title:'决策输入构造',short:'决策输入构造',subtitle:'每次决策均有明确的信息边界。',input:'当前任务、已取得证据、已有报告与历史',actor:'上下文组织',output:'本轮角色输入 + 实际图片附件',points:['HI 获得自己的局部任务和可见材料；GIA 获得总任务、报告及当前可用证据。','标准答案、评分记录和未来调查结果留在核验侧，不能提前进入决策输入。','独立 GIA 样本可用已核实的真实报告构造状态。报告可信，仍可能没有查完整。'],example:'gia-evidence-states',exampleLabel:'对比同一证据下的四个问题',note:'问题改变时，“已有证据是否足够”的答案也可能改变。',upgrade:'新增状态渲染器：长上下文分两档压缩，实测将调查员一侧的输入字符中位数降至原先的一半以下，证据编号无缺失；超出预算时明确标注已截断，并保留读取全文的入口。'},
    {title:'教师生成与工具真实执行',short:'生成与执行',subtitle:'完整记录调查过程，支持逐步回放。',input:'当轮可见状态',actor:'GIA / HI 教师',output:'一个决定 → 实际返回 → 下一轮状态',points:['一次教师请求只生成一个角色的输出。工具调用由现有 Harness 实际执行。','保存原始输入、图片、模型输出、工具观察和执行对应关系。失败也按实际结果记录。','GIA 可以独立生成下一步决定；只有接入 HI 并真实执行，才计为协作轨迹。'],example:'video-intervals',exampleLabel:'展开 22 次真实角色决策',note:'检索、抽帧、转写等工具由 Harness 真实执行，非模拟返回。',upgrade:'新增版本化规格：面向学生的角色提示与面向教师的生成指令分开管理，各自带版本号，四项内容统一计算哈希并写入每一行数据，任一样本均可追溯到其规格版本。'},
    {title:'逐单元核验与问题定位',short:'逐单元核验',subtitle:'每一步动作、每份报告、每次派工与最终回答，分别判定。',input:'真实输入、输出、执行记录 + 案卷要求',actor:'三道工序 · 核验模型独立于生成模型',output:'问题类型 + 严重度 + 每行处置与权重',points:['四种判定单元分开走：HI 的每一步动作、HI 的局部报告、GIA 的派工、GIA 的最终回答。','三道工序依次执行——程序检查、逐条证据锚定、任务充分性；前一道已判定的不再重复判定。','判定不给分数，只贴统一定义的问题类型；类型自带严重度与是否污染后续结论。'],example:'video-local-intervals',exampleLabel:'查看真实案例的逐步判定',note:'判定针对每一个单元单独进行，不以整份报告的结论替代单步判定。',upgrade:'核验器升到 0.2.0：判定单元由「报告」扩到动作、派工与最终回答；新增两条校准规则，修掉第 0 轮抽样发现的两类误报。'},
    {title:'筛选、修复与处置',short:'筛选与修复',subtitle:'逐目标判定，不以整体平均分替代单步结论。',input:'逐目标评分 + 实际执行核对',actor:'筛选与重新执行',output:'候选目标 / 重新执行 / 仅作历史 / 排除',points:['逐目标判为可收录、评分 3 或 4、无关键评审未知，并通过执行及输入对应检查，才进入候选。','有实质错误的动作，从对应状态重新执行并复评，不把改写答案接到不相容的旧后续。','错误前缀可以作为之后的真实上下文；它本身不因此成为正向学习目标。'],example:'video-intervals',exampleLabel:'查看 19 行收录与 3 行未收录',note:'如实声明「尚未查完」的报告同样具备示范价值，但不代表该项调查已经完成。',upgrade:'处置结果与统一的问题类型枚举绑定：每一条被退回的输出均归入具体类型，使改进方向可按类型统计，而非仅留一段自由文本说明。'},
    {title:'按角色导出与输入核对',short:'导出与核对',subtitle:'模型仅学习当前角色的合格输出。',input:'通过核验的 GIA 与 HI 决策',actor:'分角色 SFT 数据',output:'当前上下文 → 下一次正确输出',points:['GIA 与 HI 分开导出；开发数据和构造方法的保留检查数据分开保存。','系统提示、用户输入、工具观察及其他角色内容作为上下文；只有最后一条目标输出计算训练损失。','用 Qwen3.5-9B 的实际模板、分词和图片处理检查输入、附件与监督位置，核对是否截断。'],example:'dog-source',exampleLabel:'打开一条实际 SFT 样本',note:'导出后逐条核对监督位置与截断情况，确认模型实际接收的输入与设计一致。',upgrade:'每一行数据均附带规格版本、核验器版本、问题类型与长度估算，训练、评估与后续改进可直接依这些字段切分，无需回溯运行目录。'}
  ];
  const format = n => Number(n||0).toLocaleString('zh-CN');
  const text = value => esc(value).replace(/\*\*([^*\n]+)\*\*/g,'<strong>$1</strong>');
  const json = value => `<pre class="tr-code">${esc(JSON.stringify(value,null,2))}</pre>`;
  const raw = (title,value) => `<details class="tr-raw"><summary>${esc(title)}</summary>${json(value)}</details>`;
  const pill = (label,cls='') => `<span class="tr-pill ${cls}">${esc(label)}</span>`;
  const actionTitle = s => operations[s.operation] || labels[s.action] || s.action;
  const routeButton = (hash,label,cls='tr-link') => `<a class="${cls}" href="${esc(hash)}" data-training-route="${esc(hash)}">${label}</a>`;
  function stop(){T.playing=false;clearTimeout(T.timer);const b=page.querySelector('[data-tr="play"]');if(b){b.textContent='▷ 自动回放';b.setAttribute('aria-pressed','false');}}
  window.stopTrainingPlayback = stop;
  async function get(path){if(T.cache.has(path))return T.cache.get(path);const value=await api(path);T.cache.set(path,value);return value;}
  function loading(){page.innerHTML='<div class="tr-loading" role="status"><span class="tr-loader"></span>正在读取实际训练记录…</div>';}
  function top(section){return `<div class="tr-top"><div><div class="eyebrow muted">METHOD &amp; PROGRESS</div><h1>方法与进展</h1></div><span class="tr-readonly"><i class="small-dot"></i> 已保存记录 · 浏览不调用模型</span></div><nav class="tr-tabs" aria-label="训练数据栏目">${[['roadmap','项目进展'],['architecture','系统架构'],['flow','生成流程'],['walker','轨迹生成'],['rubric','质量核验'],['samples','轨迹示例'],['paper','论文草稿']].map(([id,label])=>routeButton(`#training${id==='roadmap'?'':'/'+id}`,label,`tr-tab ${section===id?'active':''}`)).join('')}</nav>${page.classList.contains('tr-detail-page')||section==='paper'?'':statusStrip(section)}`;}
  function vs(){return (T.data&&T.data.verification)||{};}
  function num(v){return v===undefined||v===null?'—':format(v);}
  function statusStrip(section){return `<div class="tr-status-strip"><span class="tr-status-now">11 个阶段 · 1 个完成 · 5 个进行中</span><span>框架、流水线、核验、环境固化的代码已齐；等正式教师模型与人工金标</span><span>已完成三层核验：调查 <b>${num(vs().runs)}</b> 次 · 报告 <b>${num(vs().reports)}</b> 份 · 数据来源 <b>${num(vs().datasets)}</b> 个</span><span>下一步：正式第 0 轮与首次训练</span>${section==='roadmap'?'':routeButton('#training','看项目进展 →')}</div>`;}
  function renderFlow(){
    const v=vs(),d=stageData[T.stage];
    page.innerHTML=top('flow')+`<section class="tr-hero"><div><div class="eyebrow">FROM INVESTIGATION TO LEARNING</div><h2>把每一次真实调查，<br>沉淀为可复用的训练示范<span class="accent">。</span></h2><p>真实执行留下完整过程，逐单元核验决定去留。<br>GIA 学习判断与分工，HI 学习取证与报告。</p></div><div class="tr-hero-card"><span>已核验的工作区报告</span><strong>${num(v.reports)}<small>份</small></strong><div class="tr-role-split"><span class="tr-role-gia">调查 <b>${num(v.runs)}</b></span><span class="tr-role-hi">证据对照 <b>${num(v.pairs)}</b></span></div><div class="tr-stat-note">第 0 轮三层核验结果</div></div></section><div class="tr-stats"><div><b>${num(v.runs)}</b><span>真实调查</span></div><div><b>${num(v.reports)}</b><span>工作区报告</span></div><div><b>${num(v.pairs)}</b><span>结论–证据逐对核验</span></div><div><b>${num(v.datasets)}</b><span>数据来源</span></div></div><section class="tr-flow-section"><div class="tr-section-title"><div><div class="eyebrow muted">DATA GENERATION PIPELINE</div><h2>一条训练样本的产生过程</h2></div><span>选择步骤，查看输入与产物</span></div><div class="tr-flow-layout"><div class="tr-stages" aria-label="数据生成步骤">${stageData.map((v,i)=>`<button class="tr-stage ${i===T.stage?'active':''}" data-tr-stage="${i}" aria-pressed="${i===T.stage}"><span>${String(i+1).padStart(2,'0')}</span><div><strong>${v.short}</strong><small>${i===2?'模型与工具实际运行':i===3?'逐个决策独立判定':i===5?'GIA / HI 分开保存':i===0?'任务、来源与参考材料':i===1?'隔离参考答案与未来信息':'保留有据可依的示范'}</small></div><i>↗</i></button>`).join('')}</div><article class="tr-stage-detail" aria-live="polite"><div class="tr-panel-kicker">步骤 ${String(T.stage+1).padStart(2,'0')} / 06</div><h3>${d.title}</h3><p class="tr-subtitle">${d.subtitle}</p><div class="tr-transformation"><div><small>输入</small><strong>${d.input}</strong></div><span>→</span><div class="tr-transform-actor"><small>${d.actor}</small><strong>${d.output}</strong></div></div><ol class="tr-points">${d.points.map(p=>`<li>${p}</li>`).join('')}</ol><p class="tr-callout">${d.note}</p><p class="tr-upgrade"><b>本期升级</b>${d.upgrade}</p><div class="tr-stage-footer">${routeButton(`#training/samples/${d.example}`,`${d.exampleLabel} <span>→</span>`)}<div><button class="tr-round" data-tr-stage="${T.stage-1}" ${T.stage===0?'disabled':''} aria-label="上一步">←</button><button class="tr-round" data-tr-stage="${T.stage+1}" ${T.stage===5?'disabled':''} aria-label="下一步">→</button></div></div></article></div><div class="tr-return"><span>↶</span> 需要修复的动作回到对应状态重新执行，保留原来的真实历史。</div></section><section class="tr-supervision"><div><div class="eyebrow muted">WHAT THE MODEL LEARNS</div><h2>上下文用于理解，<br>目标输出才计入监督。</h2><p>一段完整调查被拆分为逐次决策的训练样本。</p></div><div class="tr-message-stack"><div><span>输入上下文</span><p>角色提示 · 当前任务 · 已有证据与报告</p><small>不计算目标损失</small></div><i>↓</i><div class="tr-target-message"><span>当前角色输出</span><p>GIA 的下一次决定 / HI 的下一次行动或报告</p><small>合格输出参与监督</small></div></div></section><div class="tr-footnote">第 0 轮数据已完成三层核验与学生模型处理器的逐行核对；首次模型训练正在准备中。</div><div class="tr-doc-links">${docLink('method','完整构造方法')} ${docLink('results','实际测量与结果')}</div>`;
  }
  const planStates={done:['已完成','done'],doing:['进行中','doing'],todo:['未开始','todo'],later:['延后 / 备选','later']};
  // 11 个阶段，按 2026-09-24 拍板后的口径与 10-01 总清单重排；状态截至 2026-10-01。
  const roadmap=[
    {id:'framework',kicker:'阶段 01 · 已完成',title:'调查框架与工具链',hint:'真实任务端到端验证',items:[
      {state:'done',name:'两级调查角色',text:'GIA 负责全局：拆解问题、并行派发调查任务、判断证据是否足以形成结论。HI 在各自独立的工作区内执行单项任务，取证后提交局部报告。'},
      {state:'done',name:'动态调查轮次',text:'轮内并行、轮间串行。证据全局共享，结论按工作区隔离，避免各工作区之间的判断相互干扰。'},
      {state:'done',name:'检索与检查工具',text:'检索侧分为来源、打开、抽取三层；材料检查侧支持视频抽帧、片段对齐、文档读页、区域裁剪、图中文字识别、语音转写与片段定位。语音转写、按文字找帧与图内定位自 9 月 30 日起在数据生成中启用。'},
      {state:'done',name:'可追溯的证据库',text:'全局证据持久化存储，具备读取审计与断点恢复。结论链条固定为：最终回答 → 发现 → 证据 → 来源。'},
      {state:'done',name:'两个角色都直接看图',text:'HI 直接接收图片像素；GIA 看到原始材料的关键帧，需要时按编号调取证据图片。'},
      {state:'done',name:'真实任务验证',text:'网页事实核查与视频比对任务均已通过真实接口端到端验证，全过程留有可回放的记录。'},
      {state:'later',name:'工作记忆的三项增强',text:'焦点证据分级淘汰、证据指纹去重、长文本句级压缩。暂不进入首次训练前的必做清单，移植后再纳入可调参数。'}]},
    {id:'pipeline',kicker:'阶段 02 · 进行中',title:'训练数据流水线与案卷',hint:'33 个案例 · 五份划分',items:[
      {state:'done',name:'运行记录 → 决策序列',text:'每一步保存当时的输入、附图、模型输出与工具真实返回，以及三者之间的对应关系，支持逐步回放。'},
      {state:'done',name:'版本化规格',text:'角色提示、生成指令与运行参数分开管理，各自带版本号与内容哈希，并写入每一行数据；运行时提示与规格逐字一致由测试守卫。'},
      {state:'done',name:'统一的状态渲染档位',text:'全项目统一使用 16K 档，实测含附图后最长输入远低于 32K 训练上限。'},
      {state:'done',name:'按角色导出与核对',text:'两个角色分别导出，仅当前角色的目标输出参与训练；导出后用学生模型的处理器逐条核对监督位置与长度。'},
      {state:'done',name:'案卷扩到 33 个',text:'每个案例一份案卷：正确的调查必须建立什么、必需的证据、程序化检查。新增 22 个覆盖文本声明、图文声明、片段定位与拷贝检测，经独立审稿后入库。'},
      {state:'done',name:'五份数据划分',text:'训练 / 门控 / 确认 / 评测 / 测试五份，按事件家族一次性分配并校验无泄漏；评测与测试划分从不参与任何决策。'},
      {state:'done',name:'批量调查驱动',text:'从案卷出发批量运行、断点续跑、自动记录每次调查的成本（墙钟、教师调用、token、附图数）。'},
      {state:'doing',name:'离线换题的调度数据',text:'在已有调查状态上替换问题，为调度角色补充带标准答案的训练样本；45 道题已经独立审稿入库，等待重跑。'},
      {state:'doing',name:'新增案例的首批轨迹',text:'新增案例中属于训练划分的 10 个已备好启动脚本，等待运行。'},
      {state:'todo',name:'成本实测',text:'驱动与记录已就位；正式的单案例成本等正式教师模型部署后测定。'}]},
    {id:'quality',kicker:'阶段 03 · 进行中',title:'核验体系（循环的地基）',hint:'核验器 0.3.3 · 等人工金标',items:[
      {state:'done',name:'三层核验',text:'第一层以程序核查引用与必需项；第二层逐条结论对照其引用的每一份证据原文或像素，并由程序回查引语；第三层依据案卷核查覆盖度与终止条件。'},
      {state:'done',name:'四类判定单元',text:'HI 的每一步动作、每一份报告，GIA 的每一次派工、每一个最终答案，都单独判定并归因到负责的角色。'},
      {state:'done',name:'统一的问题类型枚举',text:'核验结果归入统一定义的问题类型，共五个层级，每类标注角色、严重度与是否影响后续结论。'},
      {state:'done',name:'核验器逐版修正',text:'0.3.1 区分可选证据；0.3.2 修正覆盖汇总的记错规则；0.3.3 让核验模型看到帧时间与视频总长、多图比较附图、答案标签语义判定。重核后 8 份报告由不通过翻为通过。'},
      {state:'done',name:'召回率校准工具',text:'向已核验的报告、派工与答案注入已知错误再核验，统计抓到的比例；试点的报告级注入召回 0.96。'},
      {state:'done',name:'钻空子探针与钻空子率',text:'专测"少写含糊结论、堆引用、过度报未完成、派简单任务"四条路径；另有训练用核验器与参考核验器分歧的统计脚本。'},
      {state:'doing',name:'人工金标集',text:'抽样脚本与离线标注页已备好（第一批 80 条，标注页看不到核验器结论）；目标 300 对结论–证据加 100 个单元、两人标注。等待人工标注。'},
      {state:'doing',name:'核验模型选定',text:'本机可接任意 OpenAI 兼容服务或 Claude 命令行；服务器前先用 Sonnet 看数据。正式核验模型与冻结的参考核验器等服务器到位。'},
      {state:'todo',name:'核验分层方案',text:'模型只回答两类原子问题（证据是否支持这句话、这段文字对准哪项要求），其余由程序推导；待拍板。'}]},
    {id:'environment',kicker:'阶段 04 · 代码完成',title:'环境固化',hint:'回放 · 缓存 · 黑名单',items:[
      {state:'done',name:'检索录制回放',text:'每个案例首次运行时录下搜索结果、打开的页面与下载的媒体；之后所有对照组与轮次回放同一份快照，跨轮曲线不再混入网络漂移。'},
      {state:'done',name:'工具结果缓存',text:'文字识别、语音转写、媒体对齐的结果缓存，并可统计重跑方差。'},
      {state:'done',name:'核查站点黑名单',text:'检索与打开两层拦截核查类站点（含二手核查栏目），拒绝与打标均留痕；对照 33 份案卷后，必需证据不可达的案例为 0。实测：教师打开的核查页面从 18 个降到 0。'},
      {state:'done',name:'运行时按规格读取参数',text:'图片上限、读页长度等参数由规格驱动运行时，并记录生效值；这是后续自动修改规格的前提。'},
      {state:'doing',name:'按声明日期截断检索',text:'机制已生效，但多数声明类案卷的必需证据发表于声明之后，是否启用、如何改案卷待定。'}]},
    {id:'training',kicker:'阶段 05 · 进行中',title:'正式第 0 轮与首次训练',hint:'等正式教师模型',items:[
      {state:'done',name:'模型口径已确定',text:'学生模型 Qwen3.5-9B（原生支持图片与视频），教师模型 Qwen3.5-122B-A10B，本地调试采用同系列 4B，确保本地与服务器使用同一条多模态路径。'},
      {state:'done',name:'训练链路冒烟',text:'本地打通「取数据 → 训练 → 保存 LoRA → 加载推理 → 前向」，并用真实导出的数据验证。'},
      {state:'done',name:'流水线验证用的第 0 轮',text:'12 次试点调查完整走过核验与导出；按决定只作流水线验证，不进入正式训练。'},
      {state:'done',name:'三轮对照改进',text:'同 11 个案例、同一教师、三版规格：核查站点沾染归零；每步输出带一句话理由，监督信号中位长度 91 → 132；声明类每次调查的检索查询 2 → 6.5，读过的独立来源 3.0 → 5.7。'},
      {state:'todo',name:'部署正式教师模型',text:'122B FP8 占 4 张卡，与核验模型、学生采样分卡；接入代码已备好。'},
      {state:'todo',name:'用正式教师重做第 0 轮',text:'在训练划分的 21 个案例上运行、核验、导出。'},
      {state:'todo',name:'服务器首次 SFT',text:'HI、GIA 各一次，至少 2 个随机种子；在评测划分上与未训练底座比较，得到所有后续实验的起点。'},
      {state:'todo',name:'留出数据集评估',text:'每个能力维度留出一个训练中完全不出现的数据集，检验学到的是通用调查能力还是固定模式。'}]},
    {id:'loop',kicker:'阶段 06 · 未开始',title:'自我改进循环的骨架',hint:'诊断 · 路由 · 门控',items:[
      {state:'todo',name:'错误账本与诊断表',text:'核验结果按「角色 × 单元 × 错误类型」汇总，附教师率、学生率、核验器在该类型上的召回与精确率、上游归因占比。'},
      {state:'todo',name:'两问路由',text:'只对核验器已校准的错误类型做判断：这错是哪个角色的？教师也犯吗？教师也犯→改规格，只有学生犯→训权重。'},
      {state:'todo',name:'规格修改的提议与门控',text:'按错误格逐个演化：每个被判为规格诱发的错误格给出最多四个候选，每个只改一个对象，按频次顺序对最新规格过门 A；每轮子步数设上限并按预先登记的条件停止，学生提示每轮最多改一处。门 B 在冻结规格下按角色比较权重。'},
      {state:'doing',name:'教师侧的第一批规格改动',text:'已有人工版：搜索指令（不搜核查文章、找一手来源）、图片比容量多时的查看策略，并配同案例对照。自动化的 A/A 测试与修订日志未做。'},
      {state:'todo',name:'双干预与剂量–反应工具',text:'对高频错误同时做一个规格候选与一次小规模增量训练，实测哪根杠杆更有效；冻结权重下批量生成扰动规格并评分。'}]},
    {id:'inner',kicker:'阶段 07 · 未开始',title:'规格演化单独试跑',hint:'权重冻结',items:[
      {state:'todo',name:'冻结第 0 轮学生，只改规格 2–3 轮',text:'不训练；记录每轮接受的改动数、门控效果、确认划分上的方向一致率。'},
      {state:'todo',name:'判断点',text:'方向一致率明显低于文献报告的 53% 或几乎没有改动过门，则收窄到只改运行参数。'}]},
    {id:'outer',kicker:'阶段 08 · 未开始',title:'两个角色交替训练',hint:'伙伴同步数据',items:[
      {state:'todo',name:'伙伴同步数据',text:'训练 HI 时任务来自当前的学生 GIA；训练 GIA 时离线状态用当前学生 HI 的真实报告（含有缺陷的报告），并固定混入教师报告。'},
      {state:'todo',name:'学生自采与教师桥接',text:'每个案例学生跑 4 次，通过核验的直接入选；没通过的由教师从首个错误步桥接一步、学生续跑。'},
      {state:'todo',name:'教师依赖率',text:'入选数据中由教师产出的比例，逐轮下降。硬指标：第 2 轮起低于一半且留出集不降；达不到则论文改称"自我改进"。'},
      {state:'todo',name:'交替更新与门 B',text:'固定 GIA 训 HI、过门；再固定新 HI 训 GIA、过门；不过只回滚那一个角色。'},
      {state:'todo',name:'防钻空子监控',text:'必需要求被支持结论覆盖的比例、可回答案例的未完成率上限、校准规则反例的假接受率。'},
      {state:'todo',name:'第一次交叉搭配',text:'新旧 GIA × 新旧 HI 四种组合，得到两个角色互相帮助的协同项第一个点。'}]},
    {id:'experiments',kicker:'阶段 09 · 未开始',title:'完整循环与主实验',hint:'r2–r4 · 四臂同预算',items:[
      {state:'todo',name:'每轮协议',text:'运行 → 核验 → 诊断 → 路由 → 逐格改规格（每格一次门 A，至停止条件）→ 训权重（门 B）→ 回滚 → 在评测划分上测一次 → 写修订日志。核验器只在轮间升版。'},
      {state:'todo',name:'主曲线：四臂同预算',text:'完整循环 / 只训练 / 只改规格 / 纯蒸馏，至少四个轮次点，曲线只画在从不参与决策的评测划分上。'},
      {state:'todo',name:'路由先验的反事实检验',text:'三轮合并约 9 格错误，加项目外人员盲设计的植入缺陷，报告先验预测最优杠杆的准确率。'},
      {state:'todo',name:'核验分与提升的剂量–反应',text:'冻结权重生成 100–200 个随机或刻意变坏的规格改动，看核验分变化能否预测留出集变化。'},
      {state:'later',name:'强化学习',text:'仅在 SFT 曲线走平且有剩余算力时作为附加组。'}]},
    {id:'evidence',kicker:'阶段 10 · 未开始',title:'收尾证据',hint:'迁移 · 消融 · 测试集',items:[
      {state:'todo',name:'规格改动的方向一致率',text:'从门控日志直接统计；另测改动在不同执行模型间是否保持方向。'},
      {state:'todo',name:'完整交叉搭配矩阵',text:'GIA、HI 各取底座 / 训练后 / 教师三档，多种子，带置信区间。'},
      {state:'todo',name:'核验器报告',text:'各错误类型的注入召回、人工精确率、标注一致性、钻空子率，逐轮报告。'},
      {state:'todo',name:'迁移与快速适应',text:'留出数据集上用 0 / 20 / 50 条样例做短训练，比较循环后模型、只做第 0 轮训练的模型与原始底座。'},
      {state:'todo',name:'架构消融',text:'去掉 GIA 只留单 HI / 两角色共用一套权重 / 只跑一轮 / 朴素单智能体基线。'},
      {state:'todo',name:'测试集最终一次',text:'所有对照组的最终检查点在锁定的测试集上各测一次，带 / 不带核查站点黑名单两组。'}]},
    {id:'paper',kicker:'阶段 11 · 进行中',title:'论文与演示',hint:'方法成稿 · 实验待写',items:[
      {state:'done',name:'论文方法部分',text:'摘要、引言、相关工作（含定位表）、方法七节已成稿，中英双语，可在「论文草稿」页签查看；已按精简方案改为两根杠杆、两问路由。'},
      {state:'done',name:'两张图',text:'系统架构图与核验路由循环图。'},
      {state:'doing',name:'演示网页同步',text:'本页已按 10 月 1 日状态重写；质量核验页与样例页的版本号、链接随核验器升版更新。'},
      {state:'todo',name:'实验章节',text:'等阶段 09、10 出数；实验开始前把各假设的通过阈值预先登记进文档。'},
      {state:'todo',name:'参考文献与查新',text:'2026 年的 arXiv 条目逐条补全；两处引文回原文核对；架构层面补一次查新。'},
      {state:'todo',name:'第二支点：数据集',text:'案卷 + 核验结果 + 轨迹，整理成带过程级核验标注的视频核查轨迹数据集。'}]}
  ];
  function planCount(state){return roadmap.reduce((n,b)=>n+b.items.filter(i=>i.state===state).length,0);}
  const archMap = {width:1600, height:1164,
    caption:'三块合在一张图上：上方是一次调查的整体运行结构，左下是单个工作区内部的一步，右下是持久化的全局库与证据链。拖动可以平移，点击任意方块查看它的职责与边界。',
    nodes:[
      {id:'panel-a', x:20, y:20, w:1540, h:560, kind:'panel', title:'a · 全局调查',
       lines:['GIA 决定查什么 · 轮内并行 · 轮间串行'],
       detail:['一次调查由若干轮组成。每一轮 GIA 看一份紧凑的全局视图，决定再派几个局部任务，还是证据已经够了直接作答。',
               '工作区不能再派生子工作区，轮与轮之间严格串行。']},
      {id:'input', x:48, y:200, w:160, h:112, kind:'input', title:'任务与原始材料',
       lines:['问题 + 视频 / 图片 / PDF'],
       detail:['用户提交的每一份材料都会登记成一条证据，带上模态、来源与处理链，后续所有引用都指回它。',
               '图片原件，以及视频、PDF 已经取得的画面，会随全局视图一起送进 GIA——调度方不看画面就只能派出模板化的任务。']},
      {id:'gv', x:252, y:68, w:240, h:156, kind:'view', title:'全局视图',
       lines:['总任务','原始材料','本轮读到的证据','最新一轮报告','更早结果索引'],
       detail:['固定五块，由 Harness 构造，GIA 不需要自己维护状态，也不输出"待办清单"。',
               '看不到的历史可以用 QUERY_MEMORY 检索、用 READ_RECORD 按编号读全文或调取图片。',
               '超出预算时按"可恢复优先"裁剪，并标记内容已截断。']},
      {id:'render', x:252, y:246, w:240, h:74, kind:'render', title:'状态渲染器',
       lines:['legacy / compact_8k / compact_16k'],
       detail:['模型调用前把历史与报告正文压到预算之内，证据编号一个不丢，装不下的正文标注已截断并保留读取全文的入口。',
               '在 340 行试点语料实测：调查员一侧的字符中位数从 23.1K 降到 11.3K（16K 档），调度一侧从 24.9K 降到 18.5K。']},
      {id:'gia', x:252, y:344, w:240, h:114, kind:'gia', title:'GIA 全局调度',
       lines:['决定：INVESTIGATE / FINISH','动作：QUERY_MEMORY · READ_RECORD'],
       detail:['只做两件事：判断现有证据够不够回答总问题，以及把总问题拆成若干条互不依赖的局部任务。',
               '它不直接调用检索或检查工具——那是工作区的事。',
               '布置任务时不能把尚未证实的判断当作任务前提。']},
      {id:'finish', x:252, y:490, w:240, h:72, kind:'final', title:'最终回答',
       lines:['每句结论后跟支持它的证据编号'],
       detail:['Harness 只机械检查这些编号是否存在、能否解析回来源，不替 GIA 判断答案对不对。',
               '"结论是否真的被证据支持"由离线核验回答，不在运行时这条链路上。']},
      {id:'round', x:602, y:68, w:400, h:396, kind:'group', title:'第 t 轮',
       lines:['轮内并行 · 轮间串行'],
       detail:['同一轮里的工作区并行执行，轮与轮之间串行。任务编号由 Harness 统一分配。',
               '某个工作区超时或出错不会取消其他工作区，它的错误本身会成为下一轮的输入。']},
      {id:'ws1', x:624, y:136, w:356, h:74, kind:'ws', title:'工作区 W t,1',
       lines:['HI θ · 独立局部状态'],
       detail:['每个工作区拿到自己的任务、材料，和一份按本任务排序的证据视图。',
               '所有工作区共享同一套 HI 参数 θ，但不共享上下文与判断。']},
      {id:'ws2', x:624, y:222, w:356, h:74, kind:'ws', title:'工作区 W t,2',
       lines:['HI θ · 独立局部状态'],
       detail:['兄弟工作区之间看不到彼此的实时状态、工具历史与发现。',
               '它们能共享的只有已经保存完整的证据。']},
      {id:'more', x:624, y:308, w:356, h:40, kind:'note', title:'⋮',
       lines:['同一轮内可以有更多工作区'],
       detail:['一轮里派出几个工作区由 GIA 决定，受每轮工作区数量上限约束。']},
      {id:'wsm', x:624, y:360, w:356, h:74, kind:'ws', title:'工作区 W t,m',
       lines:['HI θ · 独立局部状态'],
       detail:['已完成轮次的证据全局可见，当前轮的证据在聚合前只有产出它的工作区能看到。',
               '下方 b 区展开的就是这样一个工作区的内部。']},
      {id:'report', x:1042, y:178, w:184, h:168, kind:'report', title:'工作区报告',
       lines:['status: COMPLETED / INCOMPLETE','findings: 陈述 + 证据编号','selected_evidence_ids','status_reason'],
       detail:['HI 自审：所派的这项任务完成了没有。这是第一层审核。',
               '如实声明"尚未查完"同样是合格报告——它准确描述了边界。']},
      {id:'agg', x:1256, y:68, w:300, h:88, kind:'harness', title:'Harness 轮聚合',
       lines:['收齐 · 落库 · 构造下一轮视图'],
       detail:['等这一轮全部工作区到达终态，把报告与本轮实际用到的证据打包成 GIA 的新视图。',
               'GIA 随后做第二层审核：这些结果能否成为总任务的有效信息，继续还是结束。']},

      {id:'panel-b', x:20, y:600, w:920, h:540, kind:'panel', title:'b · 工作区内部',
       lines:['HI 每一步只做一个决定'],
       detail:['这是上方任意一个工作区的内部。HI 每次要么选一个动作，要么提交报告。',
               '工具由 Harness 真实执行，返回的观察再进入下一步。']},
      {id:'state', x:46, y:720, w:186, h:196, kind:'view', title:'局部工作记忆',
       lines:['当前任务与材料','候选来源','按本任务排序的证据视图','自己保留的重点证据','自己的历史 · 最新一次返回'],
       detail:['明确看不到：兄弟工作区的实时状态、整个证据库的完整内容、其他工作区的工具历史、数据集标签、GIA 的推理。',
               '证据只保存一份，每个工作区各自决定排序与展示，完整内容按需展开。']},
      {id:'pixels', x:46, y:940, w:186, h:84, kind:'render', title:'画面直接进入 HI',
       lines:['每次最多 8 张图片像素'],
       detail:['原图、抽出的视频帧、渲染的文档页面以像素形式随决策请求送进模型，长边缩放并受总像素上限约束。',
               '正因为 HI 自己看得见画面，把"看图"外包给工具的做法才被取消。']},
      {id:'submit', x:268, y:650, w:180, h:114, kind:'report', title:'提交局部报告',
       lines:['COMPLETED / INCOMPLETE','格式与引用检查'],
       detail:['HI 自审这项任务完成了没有，Harness 只检查格式与引用编号是否存在。',
               '报告交给 GIA，兄弟工作区读不到。']},
      {id:'hi', x:268, y:786, w:180, h:112, kind:'gia', title:'HI 局部调查员',
       lines:['选择一个动作','或提交局部报告'],
       detail:['一次请求只产出一个决定：SEARCH、INSPECT、QUERY_MEMORY、READ_RECORD 之一，或者一份报告。',
               '输出是结构化 JSON，解析或角色校验不通过会进入有限次数的修复。']},
      {id:'search', x:486, y:650, w:226, h:92, kind:'tool', title:'SEARCH',
       lines:['检索 · 打开来源 · 登记材料','网页 / 图片 / 视频 / 以图搜图'],
       detail:['两种互斥调用：用文字或图片去搜，或者直接打开一个已知地址、已有的候选来源编号。',
               '候选来源不会原地升级成证据——看到搜索摘要不等于读过正文。',
               '视频类结果默认只拿到来源页信息，下载是 INSPECT 内部的事。']},
      {id:'inspect', x:486, y:748, w:226, h:132, kind:'tool', title:'INSPECT',
       lines:['抽帧 · 读文档页 · 裁剪','定位区域 · 图中文字 · 语音转写','定位片段 · 片段对齐'],
       detail:['当前实际注册了这 8 个处理器，全部作用于已经登记的证据编号。',
               '画面与声音的问答不再作为工具操作——HI 直接看画面，不需要把"看图"当成一次工具调用。']},
      {id:'detect', x:486, y:900, w:226, h:74, kind:'reserved', title:'DETECT',
       lines:['待接入'],
       detail:['设计里保留给第三个工具的位置，方向是生成与伪造检测。',
               '当前还没有接入任何检测器，HI 的动作表里也还没有它——接入之后这一格才会有内容。']},
      {id:'mem', x:486, y:990, w:226, h:112, kind:'tool', title:'QUERY_MEMORY · READ_RECORD',
       lines:['检索已有记录','按编号读全文或调取图片'],
       detail:['查已保存的证据、发现、候选来源、报告与工作区任务卡片。',
               '新的抽帧、裁剪、转写不走这里，那是 INSPECT 的事。']},
      {id:'exec', x:750, y:748, w:166, h:132, kind:'harness', title:'Harness 执行与登记',
       lines:['真实执行','登记新的证据','写入执行审计'],
       detail:['工具调用全部真实执行，失败也按实际结果记录，不伪造返回。',
               '每条新证据带上来源、处理链和取得方式，这是后面能回溯的前提。']},
      {id:'obs', x:750, y:936, w:166, h:96, kind:'report', title:'观察 Observation',
       lines:['编号 · 来源 · 范围'],
       detail:['一次工具执行后返回给 HI 看的紧凑文字，本身不是一种证据类型，只是"这次工具说了什么"。',
               '它里面带的编号指向真正的候选来源或证据。']},

      {id:'panel-c', x:968, y:600, w:592, h:540, kind:'panel', title:'c · 全局库与证据链',
       lines:['证据共享 · 结论隔离 · 全程可回溯'],
       detail:['这一侧是整个调查的持久层，以及四层记录之间的关系。',
               '四层记录的边界是整套设计里最容易混淆、也最要紧的一处：读到了什么，和由此判断了什么，必须分开。']},
      {id:'store', x:992, y:648, w:250, h:140, kind:'store', title:'持久化全局库',
       lines:['证据：跨工作区共享','发现：只交给 GIA','报告 · 轮次 · 读取审计'],
       detail:['SQLite 持久化，支持断点恢复。工作区的证据视图就是从这里读出来并按本任务排序的。',
               '共享的是保存完整的证据，不是判断。']},
      {id:'share', x:992, y:812, w:250, h:96, kind:'store', title:'共享与隔离',
       lines:['证据跨工作区共享','发现只交给 GIA'],
       detail:['兄弟工作区不接收、也检索不到别人的发现。',
               '每个工作区应当根据任务、原始材料和证据自己形成判断。']},
      {id:'verify', x:992, y:932, w:250, h:124, kind:'render', title:'离线三层核验',
       lines:['程序检查','逐条证据锚定','任务充分性'],
       detail:['"这条结论是否真的被它引用的证据支持"，运行时的两层自审都不管，由离线核验回答。',
               '第二层把每条结论与它引用的每一份证据配对，单独判定并要求给出引语，引语须能由程序在原文中回查到。',
               '核验所用模型与生成数据的模型相互独立，避免自评。']},
      {id:'answer', x:1288, y:648, w:252, h:78, kind:'final', title:'回答中的一句结论',
       lines:['后面跟着支持它的证据编号'],
       detail:['GIA 使用发现时必须能沿它的证据引用回到原始来源。',
               '证据链的形状是固定的：最终回答 → 发现 → 证据 → 来源。']},
      {id:'finding', x:1288, y:754, w:252, h:78, kind:'gia', title:'发现 F',
       lines:['HI 对局部任务得出的判断'],
       detail:['发现不是证据，也不能自动转成证据。它属于产生它的工作区，随报告交给 GIA。',
               '例：证据是"页面显示上传日期为 2022-04-18"，发现是"该视频至少在那天已经公开出现"。']},
      {id:'evidence', x:1288, y:860, w:252, h:78, kind:'view', title:'证据 E',
       lines:['实际读到的材料，或工具对它的可追溯结果'],
       detail:['六字段卡片：编号、实际模态、简短标签、来源引用、派生自哪条、用什么行动取得。',
               '登记为证据只表示能追到来源和处理过程，不表示结果已被证实。证据可以互相矛盾。']},
      {id:'source', x:1288, y:966, w:252, h:78, kind:'input', title:'来源 R',
       lines:['搜索结果里的候选：标题 · 摘要 · 链接'],
       detail:['只是线索。候选来源不会原地升级成证据——没读正文就不能当成读过。']}],
    links:[
      {from:'input', to:'gv', dir:'h'}, {from:'gv', to:'render', dir:'v'}, {from:'render', to:'gia', dir:'v'},
      {from:'gia', to:'round', dir:'h', label:'INVESTIGATE'}, {from:'gia', to:'finish', dir:'v', label:'FINISH'},
      {from:'ws2', to:'report', dir:'h'}, {from:'report', to:'agg', dir:'vup'},
      {from:'agg', to:'gv', dir:'over', label:'下一轮'},
      {from:'wsm', to:'panel-b', dir:'v', dash:true, label:'展开任一工作区'},
      {from:'report', to:'store', dir:'v', dash:true, label:'归档'},
      {from:'state', to:'hi', dir:'h'}, {from:'pixels', to:'hi', dir:'h'},
      {from:'hi', to:'search', dir:'h'}, {from:'hi', to:'inspect', dir:'h'},
      {from:'hi', to:'detect', dir:'h', dash:true}, {from:'hi', to:'mem', dir:'h'},
      {from:'hi', to:'submit', dir:'v-up'},
      {from:'search', to:'exec', dir:'h'}, {from:'inspect', to:'exec', dir:'h'},
      {from:'detect', to:'exec', dir:'h', dash:true}, {from:'mem', to:'exec', dir:'h', dash:true},
      {from:'exec', to:'obs', dir:'v'}, {from:'obs', to:'state', dir:'under', viaY:1114, label:'进入下一步'},
      {from:'exec', to:'store', dir:'h', dash:true, label:'登记新证据'},
      {from:'answer', to:'finding', dir:'v'}, {from:'finding', to:'evidence', dir:'v'},
      {from:'evidence', to:'source', dir:'v'}]};
  const archKinds = {input:'原始输入', view:'上下文', render:'机制', gia:'角色决策', panel:'分区', group:'轮次',
                     ws:'工作区', note:'说明', tool:'工具', reserved:'位置保留', harness:'Harness 执行', report:'产出记录',
                     final:'最终产物', store:'持久化'};
  function archBox(id) { return archMap.nodes.find(n => n.id === id); }
  function archPath(link) {
    const a = archBox(link.from), b = archBox(link.to);
    if (!a || !b) return null;
    const ac = {x:a.x + a.w / 2, y:a.y + a.h / 2}, bc = {x:b.x + b.w / 2, y:b.y + b.h / 2};
    switch (link.dir) {
      case 'v': return {d:`M ${ac.x} ${a.y + a.h} C ${ac.x} ${a.y + a.h + 26}, ${bc.x} ${b.y - 26}, ${bc.x} ${b.y}`,
                        at:{x:(ac.x + bc.x) / 2, y:(a.y + a.h + b.y) / 2}};
      case 'v-up': case 'vup':
        return {d:`M ${ac.x} ${a.y} C ${ac.x} ${a.y - 28}, ${bc.x} ${b.y + b.h + 28}, ${bc.x} ${b.y + b.h}`,
                at:{x:(ac.x + bc.x) / 2, y:(a.y + b.y + b.h) / 2}};
      case 'over': {
        const top = Math.min(a.y, b.y) - 26;
        return {d:`M ${ac.x} ${a.y} C ${ac.x} ${top}, ${bc.x} ${top}, ${bc.x} ${b.y}`, at:{x:(ac.x + bc.x) / 2, y:top + 15}};
      }
      case 'under': {
        const bottom = link.viaY || Math.max(a.y + a.h, b.y + b.h) + 30;
        return {d:`M ${ac.x} ${a.y + a.h} C ${ac.x} ${bottom}, ${bc.x} ${bottom}, ${bc.x} ${b.y + b.h}`,
                at:{x:(ac.x + bc.x) / 2, y:bottom + 4}};
      }
      default: return {d:`M ${a.x + a.w} ${ac.y} C ${a.x + a.w + 34} ${ac.y}, ${b.x - 34} ${bc.y}, ${b.x} ${bc.y}`,
                       at:{x:(a.x + a.w + b.x) / 2, y:(ac.y + bc.y) / 2 - 11}};
    }
  }
  function archDetail() {
    const node = archBox(T.archNode) || archMap.nodes[0];
    return `<div class="tr-panel-kicker">${esc(archKinds[node.kind] || '节点')}</div><h3>${esc(node.title)}</h3>
      <ul class="tr-arch-facts">${node.lines.map(l => `<li>${esc(l)}</li>`).join('')}</ul>
      ${node.detail.map(d => `<p>${esc(d)}</p>`).join('')}`;
  }
  function archTransform() {
    const stage = page.querySelector('#tr-arch-stage'), zoom = page.querySelector('#tr-arch-zoom');
    if (!stage) return;
    stage.style.transform = `translate(${T.archX}px,${T.archY}px) scale(${T.archZoom})`;
    if (zoom) zoom.textContent = `${Math.round(T.archZoom * 100)}%`;
  }
  function archFit() {
    const frame = page.querySelector('#tr-arch-frame');
    if (!frame || !frame.clientWidth) return;
    const fitWidth = (frame.clientWidth - 24) / archMap.width;
    // 窄屏不再一路缩到看不清：低于保底比例就改为拖动浏览，并从左上角开始。
    T.archZoom = Math.min(1, Math.max(Math.min(fitWidth, (frame.clientHeight - 24) / archMap.height), .5));
    const scaled = archMap.width * T.archZoom;
    T.archX = scaled > frame.clientWidth ? 12 : (frame.clientWidth - scaled) / 2;
    T.archY = Math.max(12, (frame.clientHeight - archMap.height * T.archZoom) / 2);
    archTransform();
  }
  function archZoomAt(value, x, y) {
    const frame = page.querySelector('#tr-arch-frame');
    if (!frame) return;
    const next = Math.min(2, Math.max(.2, value)), ratio = next / T.archZoom;
    const cx = x == null ? frame.clientWidth / 2 : x, cy = y == null ? frame.clientHeight / 2 : y;
    T.archX = cx - (cx - T.archX) * ratio; T.archY = cy - (cy - T.archY) * ratio; T.archZoom = next;
    archTransform();
  }
  function archFocus(id) {
    const node = archBox(id), frame = page.querySelector('#tr-arch-frame');
    if (!node || !frame || !frame.clientWidth) return;
    // 只在看不见目标时才移动画面，避免每次点击都把图整个甩一下。
    const left = T.archX + node.x * T.archZoom, top = T.archY + node.y * T.archZoom;
    const right = left + node.w * T.archZoom, bottom = top + node.h * T.archZoom;
    if (left >= 8 && top >= 8 && right <= frame.clientWidth - 8 && bottom <= frame.clientHeight - 8) return;
    T.archX = frame.clientWidth / 2 - (node.x + node.w / 2) * T.archZoom;
    T.archY = frame.clientHeight / 2 - (node.y + node.h / 2) * T.archZoom;
    archTransform();
  }
  function archBindPan() {
    const frame = page.querySelector('#tr-arch-frame');
    if (!frame || frame.dataset.bound) return;
    frame.dataset.bound = '1';
    let drag = null;
    frame.addEventListener('pointerdown', e => {
      if (e.target.closest('button') || e.button !== 0) return;
      drag = {id:e.pointerId, x:e.clientX, y:e.clientY, originX:T.archX, originY:T.archY};
      frame.setPointerCapture(e.pointerId); frame.classList.add('dragging');
    });
    frame.addEventListener('pointermove', e => {
      if (!drag || e.pointerId !== drag.id) return;
      T.archX = drag.originX + (e.clientX - drag.x); T.archY = drag.originY + (e.clientY - drag.y); archTransform();
    });
    const end = () => { drag = null; frame.classList.remove('dragging'); };
    frame.addEventListener('pointerup', end); frame.addEventListener('pointercancel', end);
    frame.addEventListener('wheel', e => {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) {
        const rect = frame.getBoundingClientRect();
        archZoomAt(T.archZoom * Math.exp(-e.deltaY * .005), e.clientX - rect.left, e.clientY - rect.top);
      } else { T.archX -= e.deltaX; T.archY -= e.deltaY; archTransform(); }
    }, {passive:false});
  }
  function renderArchStage() {
    const stage = page.querySelector('#tr-arch-stage'), detail = page.querySelector('#tr-arch-detail');
    if (!stage) return;
    const paths = archMap.links.map(link => ({link, path:archPath(link)})).filter(row => row.path);
    stage.style.setProperty('--arch-w', archMap.width);
    stage.style.setProperty('--arch-h', archMap.height);
    stage.innerHTML = `<svg class="tr-arch-links" viewBox="0 0 ${archMap.width} ${archMap.height}" aria-hidden="true">
        <defs><marker id="arch-tip" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
          <path d="M0 0 L7 3.5 L0 7 Z" fill="#9db387"></path></marker></defs>
        ${paths.map(({link, path}) => `<path d="${path.d}" class="${link.dash ? 'dashed' : ''}" marker-end="url(#arch-tip)"></path>`).join('')}
      </svg>
      <div class="tr-arch-nodes">${archMap.nodes.map(n => `<button class="tr-arch-node ${n.kind} ${n.id === T.archNode ? 'selected' : ''}"
        style="left:${n.x}px;top:${n.y}px;width:${n.w}px;height:${n.h}px" data-arch-node="${esc(n.id)}"
        aria-pressed="${n.id === T.archNode}"><strong>${esc(n.title)}</strong>
        <span>${n.lines.map(l => esc(l)).join('<br>')}</span></button>`).join('')}</div>
      <svg class="tr-arch-labels" viewBox="0 0 ${archMap.width} ${archMap.height}" aria-hidden="true">${paths.filter(row => row.link.label)
        .map(({link, path}) => `<text x="${path.at.x}" y="${path.at.y}" class="tr-arch-edge-label">${esc(link.label)}</text>`).join('')}</svg>`;
    if (detail) detail.innerHTML = archDetail();
    archTransform();
  }
  window.addEventListener('resize', () => { if (T.section === 'architecture') archFit(); });
  function renderArchitecture() {
    if (!archBox(T.archNode)) T.archNode = 'gia';
    page.innerHTML = top('architecture') + `<div class="tr-section-title tr-arch-title"><div><div class="eyebrow muted">SYSTEM ARCHITECTURE</div>
        <h2>调查是怎样被组织起来的</h2><p>${esc(archMap.caption)}</p></div></div>
      <div class="tr-arch-frame" id="tr-arch-frame" tabindex="0" aria-label="系统架构图，可拖动平移与缩放"><div class="tr-arch-stage" id="tr-arch-stage"></div></div>
      <div class="tr-arch-bottom"><span>拖动平移 · 点击方块查看详情</span><div class="tr-arch-zoomers">
        <button data-arch-zoom="out" aria-label="缩小">−</button><output class="tr-arch-zoom" id="tr-arch-zoom">100%</output>
        <button data-arch-zoom="in" aria-label="放大">＋</button><button data-arch-zoom="fit" aria-label="适应画布" title="适应画布">⊡</button></div></div>
      <aside class="tr-arch-detail" id="tr-arch-detail" aria-live="polite"></aside>
      <div class="tr-arch-legend">${[['input','原始输入'],['view','上下文'],['gia','角色决策'],['tool','工具'],['reserved','位置保留'],['harness','Harness 执行'],['report','产出记录'],['store','持久化'],['render','机制']]
        .map(([k, n]) => `<span class="lg-${k}">${n}</span>`).join('')}</div>
      <div class="tr-footnote">图中标注的能力均以当前代码为准：HI 的动作表为 SEARCH / INSPECT / QUERY_MEMORY / READ_RECORD，INSPECT 实际注册 8 个处理器。DETECT 作为第三个工具的位置保留，尚未接入检测器，动作表里还没有它。</div>`;
    renderArchStage();
    archBindPan();
    archFit();
  }
  function renderRoadmap(){
    page.innerHTML=top('roadmap')+`<div class="tr-section-title tr-plan-title"><div><div class="eyebrow muted">PROJECT PROGRESS</div><h2>阶段进展与交付内容</h2><p>项目按最终方案划分为 11 个阶段：前四个阶段的代码已就位，正式训练等教师模型部署，循环与实验尚未开始。展开任一阶段可查看具体交付内容与状态。</p></div></div>
      <div class="tr-plan"><div class="tr-plan-root"><div class="tr-panel-kicker">PROJECT GOAL</div><strong>面向异构核查任务的<br>自主调查 Agent</strong><p>覆盖视频、图像与网页混合的核查场景：模型自主拆解问题、决定检索方向并完成取证，最终给出的每一条结论均可沿证据链回溯至原始来源。</p><div class="tr-plan-tally">${[['done','已完成'],['doing','进行中'],['todo','未开始'],['later','延后']].map(([k,l])=>`<span class="${k}"><b>${planCount(k)}</b>${l}</span>`).join('')}</div></div>
      <div class="tr-plan-branches">${roadmap.map(b=>`<details class="tr-plan-branch" ${b.id==='quality'||b.id==='training'||b.id==='loop'?'open':''}><summary class="tr-plan-head"><span class="tr-panel-kicker">${esc(b.kicker)}</span><strong>${esc(b.title)}</strong><small>${esc(b.hint)}</small><i class="tr-plan-chevron" aria-hidden="true"></i></summary><ul class="tr-plan-leaves">${b.items.map(i=>`<li class="${planStates[i.state][1]}"><div class="tr-plan-leaf-top"><b>${esc(i.name)}</b><span class="tr-plan-tag ${planStates[i.state][1]}">${planStates[i.state][0]}</span></div><p>${esc(i.text)}</p></li>`).join('')}</ul></details>`).join('')}</div></div>
      <div class="tr-return"><span>⟲</span> 每完成一轮，核验结果将指出下一轮的改进方向，流程本身随之迭代。</div>
      <div class="tr-plan-links">${routeButton('#training/architecture','系统架构 →')}${routeButton('#training/flow','数据生成流程 →')}${routeButton('#training/walker','轨迹生成实测 →')}${routeButton('#training/rubric','质量核验体系 →')}${routeButton('#training/samples','真实样例库 →')}</div>
      <div class="tr-footnote">本页反映 2026 年 10 月 1 日的项目状态，随各阶段进展更新。阶段 06–10 的实验设计以项目设计文档为准。</div>`;
  }
  function docLink(name,label){return `<a href="/api/training/documents/${name}" target="_blank" rel="noopener noreferrer">${label} ↗</a>`;}
  const verifyLayers=[
    {name:'程序检查',tag:'不用模型 · 结果确定',text:'核查所引证据编号是否存在、提交时是否可见、报告是否为空、案卷要求的证据是否被引用、来源排除名单是否被违反。结论中出现的数字若不在所引证据文本中，先标记为待查。',errors:['fabricated_evidence_id','empty_report','source_restriction_miss']},
    {name:'逐条证据锚定',tag:'核验模型 ≠ 生成模型',text:'每一条结论与其所引的每一份证据配对，单独调用一次，仅提供该句与证据全文或图片，判定支持 / 反驳 / 未找到，并须给出引语。引语须可由程序在原文中回查到，否则降为未找到。',errors:['unsupported_assertion','wrong_citation','attribution_drift','scope_overreach','misread_fine_text']},
    {name:'充分性',tag:'对着任务本身查',text:'以任务全文与案卷中的必需证据对照整份报告，判定要求是否已覆盖、当前是否已可作答，以及状态应为「完成」还是「未完成」。',errors:['omitted_required_fact','completed_without_finishing','incomplete_though_answerable']}
  ];
  const errorGroups=[
    {layer:'结论级',items:['unsupported_assertion','wrong_citation','fabricated_evidence_id','source_restriction_miss','attribution_drift','scope_overreach','misread_fine_text','sloppy_numeric','omitted_required_fact','incomplete_citation_set'],note:'结论与其所引证据之间的问题。典型表现为将「来源声称某事」写作「某事成立」，或将「批准拨付」写作「直接拨付」这类阶段互换，也是人工最难逐条查出的一类。校准后新增「引用集合不全」，用于区分「漏引」与「无据」。'},
    {layer:'状态级',items:['completed_without_finishing','incomplete_though_answerable','empty_report'],note:'报告整体层面的问题：尚未查完即报完成、已可作答却报未完成、提交空报告。该类由程序层直接拦截，不依赖模型判断。'},
    {layer:'动作级',items:['illegal_action','redundant_action','evidence_dropped','wrong_tool','ignored_feedback'],note:'行动本身的问题。重复查看同一材料不必然是错误，关键在于该次复查是否针对一个具体的未决问题。'},
    {layer:'调度规划级',items:['task_not_gap_driven','premature_finish','unnecessary_round','dependent_sibling_tasks','overlapping_sibling_tasks'],note:'分工与终止判断的问题。同一轮内相互依赖的任务不应并行派发；局部报告标注「完成」，亦不代表总问题已可结束。同轮任务范围重叠单列一类，只计数、不影响收录。'},
    {layer:'环境级',items:['env_contamination','env_tool_failure'],note:'不计入模型问题，但须单独记录，否则易将网络或工具故障误判为模型能力不足。'}
  ];
  const verifyUnits=[
    ['step','HI 的每一步动作','HI','动作合法吗、目标当时可见吗、是不是原样重复了上一步、工具报错后有没有调整。再算这一步带来了几条新线索——这一行的权重由它决定。','只走第一道'],
    ['report','HI 的局部报告','HI','每一条结论逐句对着它所引的每一份证据核；再看所派的任务是不是真的做完了，还是没查完就报了完成。','三道全走'],
    ['plan','GIA 的派工','GIA','任务可解析吗、材料存在吗、同一轮的任务互相依赖吗、有没有把尚未核实的说法当成前提写进任务里。','第一道 + 模型另判'],
    ['answer','GIA 的最终回答','GIA','引用的证据当时都在视图里吗、结论逐句有依据吗、该停的时候停了吗、有没有照抄上游被判有问题的结论。','三道全走']
  ];
  const unitLayers=[
    ['HI 的局部报告','report','full','full','full','每一条结论 × 每一份所引证据，逐对回到原文重读。'],
    ['GIA 的最终回答','answer','full','full','full','答案拆成句子后，由核验模型逐句对照所引证据锚定。'],
    ['GIA 的派工','plan','full','judge','gap','第二道换成「只读任务文字判它对准哪些要求、有没有把未核实的说法当前提」，再与事后反推的缺口对照。'],
    ['HI 的每一步动作','step','full','none','none','一个动作没有「结论」可以锚定。它的价值不用对错衡量，用带来了几条后来被证实的新线索衡量。']
  ];
  const layerMark={full:['✓','这一道按原样执行'],judge:['另判','这一道换成另一种模型判定'],gap:['反推','用事后缺口反推代替'],none:['—','对这种单元不适用']};
  const severityRows=[
    ['critical','一票否决','结论不可信，整个单元判为未通过。','unsupported_assertion · completed_without_finishing · premature_finish'],
    ['major','实质缺陷','结论可能仍然成立，但过程有实质问题，计入统计。','attribution_drift · scope_overreach · incomplete_citation_set'],
    ['minor','表述或效率','记录并计数，不影响这一行的去留。','redundant_action · unnecessary_round · overlapping_sibling_tasks']
  ];
  const calibrationRules=[
    ['R1 · 多证据范围豁免','一条结论引了多份证据、且每一份都被判为支持时，不再按单份证据的覆盖范围判它「过度外推」。'],
    ['R2 · 漏引降级','「与……相容」这类关系性断言，若支持其实就在本工作区已经引用过的材料里，从「没有支持」降为「引用集合不全」，并且不再污染后续结论。'],
    ['R4 · 同句证据覆盖','逐对核验一次只看一份证据，可能把「更早的状态」误读成「相反」。同一句所引的另一份证据若已直接支持被反驳的那部分，撤销这条反驳。']
  ];
  const dispositionRows=[
    ['accepted','进入候选','判定通过，且没有被上游污染。权重按这一步带来的新线索定，1 到 3。'],
    ['held','暂缓','这一步本身没有问题，但所在工作区未通过、而且它没有带来新线索。先留着，等核验器进一步校准后再定。'],
    ['rejected','暂不使用','报告级判定未通过。不进正例；在召回率校准完成之前，也不当反例训练。']
  ];
  const verifySteps=[
    ['01','这一步的输出','HI 的一次动作或一份报告，GIA 的一次派工或最终回答'],
    ['02','三道工序','程序检查 → 逐条证据锚定 → 任务充分性，前一道判过的不再重复判'],
    ['03','贴问题类型','从统一定义的类型表里选，每个类型自带严重度与是否污染后续结论'],
    ['04','定去留','进入候选 / 暂缓 / 暂不使用，并给出这一行的权重']
  ];
  // ---------------------------------------------------------------- 轨迹生成（walker）
  const walkerFacts=[
    ['9 个案例','videosham 5 · vcsl 3 · qvhighlights 1'],
    ['2 次','每条轨迹的教师调用（r0e 是 26 次）'],
    ['1.2 分钟','平均墙钟（r0e 是 9.4 分钟）'],
    ['69 行','HI 进入训练候选（wk1 是 52 行）']
  ];
  const walkerChain=[
    {name:'两段视频比对',who:'HI',steps:['抽 8 帧','视觉对齐','逐帧差分','回看差分图','未覆盖区间补帧','小框放大','音轨对齐 + 转写','提交报告'],note:'主干分支。本批 9 个案例里 4 条走到逐帧差分。'},
    {name:'对齐太弱时',who:'HI',steps:['A 侧均匀取帧','逐帧以图在 B 定位','成对回看','提交报告'],note:'对齐信号不足时自动改走这条，vcsl_pair_1 / 5 用的就是它。'},
    {name:'单段按描述定位',who:'HI',steps:['抽帧','按描述定位','两轮边界帧','提交报告'],note:'qvhighlights 这类"找片段"任务，3 步动作就能交报告。'},
    {name:'全局调度',who:'GIA',steps:['第 1 轮派原任务','报告引用图不在附图就读记录','仅预算类缺口才派第 2 轮','结束并回答'],note:'派工与终止都由可见状态决定，不依赖任何答案信息。'}
  ];
  const walkerSlots=[
    ['槽 1 · 检索词','工作区第一次要检索时','任务 + 8 帧 + 图中文字','3–4 条按优先序的检索词'],
    ['槽 2 · 任务文本','第 1 轮 GIA 派工','总任务 + 原始材料卡片与关键帧','两个任务的文本'],
    ['槽 3 · 报告','工作区走完链条','本工作区的全部证据，含差分图','findings[]（状态由程序定）'],
    ['槽 4 · 答案','GIA 结束调查时','各工作区报告 + 证据','带证据编号的最终答案']
  ];
  const walkerSlotNote='教师拿到的输入就是学生在那一步看到的状态，案卷与标准答案都不传给它——这是代码层的无泄漏保证，不靠事后检查。报告状态由程序判定，教师只能把「完成」降为「未完成」，不能反过来。';
  const walkerMetrics=[
    ['每条轨迹的决策数','26.0','10.2','11.4',''],
    ['教师（LLM）调用 / 条','26.0（最多 44）','2.0','2.0','win'],
    ['每条输入 token','116 万','7.7 万','7.8 万','win'],
    ['墙钟（平均 / 最长，分钟）','9.4 / 20.6','1.7 / 3.1','1.2 / 1.7','win'],
    ['HI 动作步','16.8','6.2','7.4','','比 wk1 多出来的步，是补看缺口与转写音轨'],
    ['重复步 / 格式拒','2 / 0','0 / 0','0 / 0','win'],
    ['链条走到（对齐→差分→读差分，或定位→边界帧）','1 / 9','9 / 9','9 / 9','win'],
    ['报告核验通过','19/22（86%）','7/9（78%）','7/9（78%）','miss','唯一没有达标的一项'],
    ['报告错误 / 份','2.1','1.3','1.6',''],
    ['答案核验通过','3/9','4/9','6/9','win'],
    ['答案错误 / 条','6.8','3.2','3.1','win'],
    ['导出 HI 进候选 / 暂缓 / 暂不用','—','52 / 11 / 2','69 / 5 / 2','win'],
    ['可见性扣下的行（动作 / 报告）','—','7 / 4','2 / 3','win']
  ];
  const walkerCases=[
    {id:'vs1007',name:'videosham_1007',kind:'两段时长差一倍',tag:'gain',
     tagText:'报告转为通过',
     rows:[['r0e',33,33,'11.7',23,'部分','COMPLETED · INCOMPLETE · COMPLETED','3/3（错 6）','0/1（错 9）','get_frames×10; READ_RECORD×6; diff_media×3; asr×2; align(visual)×1; align(audio)×1'],
           ['wk1',12,2,'1.5',8,'全部','COMPLETED','0/1（错 2）','0/1（错 5）','get_frames×4; align(visual)×1; diff_media×1; READ_RECORD×1; align(audio)×1'],
           ['wk2',13,2,'1.5',9,'全部','INCOMPLETE','1/1（错 3）','0/1（错 7）','get_frames×4; asr×2; align(visual)×1; diff_media×1; align(audio)×1']],
     text:'wk1 的报告被判未通过：B 侧 20 秒、A 侧 8 秒，walker 只核了差分段，较长那一侧没有对应的部分根本没看。wk2 按 F1 补看了 B 的 0–2.71 秒与 10.68–20.0 秒，又按 F2 把两侧音轨都转写了；程序据此把状态定为 INCOMPLETE，报告转为通过。答案层还欠一句对音轨结果的交代。'},
    {id:'vs1028',name:'videosham_1028',kind:'标题文字被替换',tag:'gain',
     tagText:'答案转为通过 · 报告 0 错误',
     rows:[['r0e',10,10,'3.7',6,'部分','COMPLETED · COMPLETED','2/2（错 6）','0/1（错 5）','asr×2; align(visual)×1; diff_media×1; align(audio)×1; READ_RECORD×1'],
           ['wk1',11,2,'1.2',7,'全部','COMPLETED','1/1（错 1）','0/1（错 4）','get_frames×2; crop×2; align(visual)×1; diff_media×1; align(audio)×1'],
           ['wk2',13,2,'1.3',9,'全部','COMPLETED','1/1（错 0）','1/1（错 2）','get_frames×2; asr×2; crop×2; align(visual)×1; diff_media×1; align(audio)×1']],
     text:'放大标题区之后，两轮都准确指出 A 的蓝框里是 "lorem ipsum"、B 是 "A conversation about CRPS"。wk1 的答案只差音轨这一项没有被核实的结论覆盖；wk2 补了两次转写，答案不再有重要错误，报告做到 0 错误。'},
    {id:'vs4126',name:'videosham_4126',kind:'画面中的人被移除',tag:'gain',
     tagText:'答案转为通过 · 报告连续 0 错误',
     rows:[['r0e',31,31,'9.5',19,'部分','COMPLETED · INCOMPLETE','1/2（错 4）','0/1（错 6）','get_frames×8; READ_RECORD×6; diff_media×2; crop×2; align(visual)×1'],
           ['wk1',12,2,'1.4',8,'全部','INCOMPLETE','1/1（错 0）','0/1（错 2）','get_frames×4; align(visual)×1; diff_media×1; READ_RECORD×1; align(audio)×1'],
           ['wk2',14,2,'1.2',10,'全部','INCOMPLETE','1/1（错 0）','1/1（错 3）','get_frames×4; asr×2; align(visual)×1; diff_media×1; READ_RECORD×1; align(audio)×1']],
     text:'差异区间 [0.200, 1.367) 与 [2.300, 6.000) 以及边界帧的夹逼，两轮都对。教师两轮都自己把状态降为 INCOMPLETE——第二段入画的是不是人物没能确认，它如实说了。wk2 补上音轨后答案转为通过。'},
    {id:'qv9167',name:'qvhighlights_9167',kind:'单段按描述定位',tag:'gain',
     tagText:'答案转为通过 · 全批最快',
     rows:[['r0e',18,18,'5.0',9,'部分','COMPLETED · COMPLETED','2/2（错 3）','1/1（错 2）','get_frames×7; locate_frames×1; READ_RECORD×1'],
           ['wk1',7,2,'1.0',3,'全部','COMPLETED','1/1（错 2）','0/1（错 5）','get_frames×2; locate_frames×1'],
           ['wk2',7,2,'0.6',3,'全部','COMPLETED','1/1（错 1）','1/1（错 2）','get_frames×2; locate_frames×1']],
     text:'报告给出 84–119 秒，标注是 84–120 秒，交并比 0.89。wk1 的答案被扣在一句没带证据编号的方法说明上（"定位依据采样图像"）；wk2 按 F4 要求每句都带引用后转为通过。3 步动作、0.6 分钟，是这一批里最快的一条。'},
    {id:'vcsl1',name:'vcsl_pair_1',kind:'对齐信号太弱',tag:'gain',
     tagText:'答案 0 错误 · 墙钟减半',
     rows:[['r0e',44,44,'20.6',33,'全部','INCOMPLETE ×3','3/3（错 5）','0/1（错 10）','get_frames×12; align(visual)×10; READ_RECORD×5; locate_frames×4; diff_media×2'],
           ['wk1',11,2,'3.1',7,'全部','INCOMPLETE','1/1（错 0）','1/1（错 2）','locate_frames×4; align(visual)×1; get_frames×1; READ_RECORD×1'],
           ['wk2',11,2,'1.2',7,'全部','INCOMPLETE','1/1（错 0）','1/1（错 0）','locate_frames×4; align(visual)×1; get_frames×1; READ_RECORD×1']],
     text:'对齐信号不足，自动改走「以图定位 + 成对回看」分支。两轮报告与答案都通过，wk2 的答案做到 0 错误。r0e 在同一个案例上花了 20.6 分钟、44 次教师调用，答案仍未通过。'},
    {id:'vcsl5',name:'vcsl_pair_5',kind:'对齐信号太弱',tag:'same',
     tagText:'两轮均通过',
     rows:[['r0e',34,34,'12.5',20,'部分','COMPLETED ×3','3/3（错 1）','1/1（错 6）','get_frames×10; READ_RECORD×7; align(visual)×3'],
           ['wk1',11,2,'2.5',7,'全部','INCOMPLETE','1/1（错 0）','1/1（错 2）','locate_frames×4; align(visual)×1; get_frames×1; READ_RECORD×1'],
           ['wk2',11,2,'1.1',7,'全部','INCOMPLETE','1/1（错 1）','1/1（错 2）','locate_frames×4; align(visual)×1; get_frames×1; READ_RECORD×1']],
     text:'同样走以图定位分支，两轮报告与答案都通过。只给出采样时刻的对应、没有给完整区间，所以状态是 INCOMPLETE——这是如实的克制，不是缺陷。r0e 在这个案例上报了 COMPLETED。'},
    {id:'vcsl6',name:'vcsl_pair_6',kind:'整段偏移约 5.1 秒',tag:'loss',
     tagText:'答案由通过转为未通过',
     rows:[['r0e',26,26,'9.1',16,'部分','COMPLETED · INCOMPLETE','2/2（错 5）','1/1（错 6）','READ_RECORD×8; get_frames×7; align(visual)×1'],
           ['wk1',11,2,'1.7',7,'全部','INCOMPLETE','1/1（错 1）','1/1（错 1）','get_frames×4; align(visual)×1; diff_media×1; READ_RECORD×1'],
           ['wk2',13,2,'1.7',9,'全部','INCOMPLETE','1/1（错 1）','0/1（错 2）','get_frames×6; align(visual)×1; diff_media×1; READ_RECORD×1']],
     text:'A 与 B 的偏移约 5.1 秒，与标注（A 1–19 ↔ B 5–24）一致。但答案由通过转为未通过：两处数字口径错误——答案里写的 B 18.354–24.225 秒是由 mapping 换算出来的，与工具返回的原值 18.314–24.253 对不上。下一步要求区间端点一律引工具原值，就是冲着这一处去的。'},
    {id:'vs2010',name:'videosham_2010',kind:'实拍换成插画背景',tag:'mixed',
     tagText:'答案转为通过 · 报告转为未通过',
     rows:[['r0e',13,13,'3.9',9,'部分','COMPLETED · INCOMPLETE','2/2（错 4）','0/1（错 2）','get_frames×2; asr×2; align(visual)×1; diff_media×1; ocr×1; align(audio)×1; READ_RECORD×1'],
           ['wk1',9,2,'1.1',5,'全部','COMPLETED','1/1（错 4）','0/1（错 6）','get_frames×2; align(visual)×1; diff_media×1; align(audio)×1'],
           ['wk2',11,2,'1.2',7,'全部','COMPLETED','0/1（错 4）','1/1（错 4）','get_frames×2; asr×2; align(visual)×1; diff_media×1; align(audio)×1']],
     text:'答案转为通过：wk1 里有一句把 B 的实拍帧引作「动画驾驶场景」的依据，F4 之后不再出现。但报告反过来由通过转为未通过——程序在链条走完时把状态定成了 COMPLETED，而任务里「差异对画面含义的影响」这类解释性要求没有被逐条回应。这正是下一步 S4 要修的。'},
    {id:'vs0069',name:'videosham_0069',kind:'片段重复 / 慢放',tag:'loss',
     tagText:'报告仍未通过 · 答案由通过转为未通过',
     rows:[['r0e',25,25,'8.6',16,'否','COMPLETED · INCOMPLETE ×2','1/3（错 13）','0/1（错 15）','diff_media×5; READ_RECORD×4; get_frames×3; asr×2; align(visual)×1; locate_frames×1'],
           ['wk1',8,2,'1.4',4,'全部','COMPLETED','0/1（错 2）','1/1（错 2）','get_frames×2; align(visual)×1; diff_media×1'],
           ['wk2',10,2,'1.3',6,'全部','COMPLETED','0/1（错 4）','0/1（错 6）','get_frames×3; align(visual)×1; diff_media×1; asr×1']],
     text:'wk2 按 F1 补看了 A 的 4.04–10.15 秒，但报告仍报 COMPLETED，「差异对画面含义的影响」没有回应；答案也没点出「重复帧 / 慢放」，由通过转为未通过。根因是能力缺口而非语言槽：walker 目前没有「检查同一视频内相邻帧是否重复」这个动作，这一项本来就做不到，已记入待补的动作清单。'}
  ];
  const walkerFixes=[
    ['F1 · 没覆盖的区间','按可见的时长和 mapping 算出两侧没被比对到的区间（≥ 1 秒且 ≥ 该侧时长 10%，最多 2 段），均匀取帧。仍没看完的，程序把状态定为 INCOMPLETE。','done'],
    ['F2 · 语音','有音轨的一侧各做一次转写，不再只做音轨对齐；没转写的同样记为缺口。','done'],
    ['F3 · 可见性','抄进参数的数按渲染视图的精度取 3 位小数；A 侧时刻由 history 里可见的 B 侧请求和 mapping 换算；后面段的起止看不见时分页重读结果记录（最多 3 页），仍看不见就放弃这一段。','done'],
    ['F4 · 答案槽','每句单独带证据引用，局限句也要引；图只能引它自己画面里的内容；任务的每一部分（含声音）都要回应。','done'],
    ['F5 · 第 2 轮','本轮只在报告写明「预算用尽」时才派。其他缺口派出去，第 2 轮工作区会把第 1 轮整条链重走一遍——要有针对性的第 2 轮规则才值得做。','todo']
  ];
  const walkerAcceptance=[
    ['重复步 0','0','pass'],['格式拒 0','0','pass'],
    ['9 条全部走到链条','9 / 9','pass'],['教师调用 ≤ 6 次','最多 2','pass'],
    ['墙钟 ≤ 5 分钟','最长 1.7','pass'],['报告核验不低于 r0e','78% vs 86%','fail']
  ];
  const walkerNext=[
    ['按任务逐条回应','程序从任务文字里拆出子要求（差异在哪、影响什么含义、什么不能确定），交给报告槽逐条写结论；程序判定「完成」的前提再加一条：每个子要求都要有对应的结论。'],
    ['时刻一律引工具原值','区间端点直接引工具结果里的原数，不用换算值——vcsl_pair_6 的回退就出在这里。'],
    ['同一视频内的重复帧检查','videosham_0069 要判断「重复帧 / 慢放」，需要一个新的检查动作（相邻帧差分）。这是能力缺口，已记入待补的动作清单。']
  ];
  const walkerNextNote='前两处不改动作，只改语言槽与状态判定，可以直接用 wk2 的状态重答，不必重跑轨迹。之后进入 D2：找原件族与声明核查族。';
  const walkerHeldNote='被扣下的 2 行动作，都是换算出来的时刻恰好与工具原值相同（12.086、10.678），属于检查器的巧合误报；3 行报告里有 "This" 这类词触发的误报，另有一处确实把教师写的 "29.970" 当成了不可见内容。';
  const walkerTags={gain:['改善','gain'],same:['保持','same'],mixed:['一升一降','mixed'],loss:['回退','loss']};
  function walkerCase(){ return walkerCases.find(c=>c.id===T.walkerCase) || walkerCases[0]; }
  function walkerCaseDetail(c){
    const [tagLabel,tagCls]=walkerTags[c.tag];
    return `<div class="tr-step-heading"><div><span class="tr-panel-kicker">${esc(c.kind)}</span><h3>${esc(c.name)}</h3></div>
        <span class="tr-wk-tag ${tagCls}">${esc(tagLabel)}</span></div>
      <p class="tr-wk-tagline">${esc(c.tagText)}</p>
      <div class="tr-matrix tr-wk-caseruns"><table>
        <thead><tr><th>轮次</th><th>决策</th><th>教师调用</th><th>墙钟</th><th>HI 动作</th><th>链条</th><th>报告状态</th><th>报告核验</th><th>答案核验</th></tr></thead>
        <tbody>${c.rows.map(r=>`<tr class="run-${r[0]}"><th scope="row">${esc(r[0])}</th><td>${r[1]}</td><td>${r[2]}</td><td>${esc(r[3])}</td><td>${r[4]}</td>
          <td class="${r[5]==='全部'?'m-full':'m-none'}">${esc(r[5])}</td><td class="m-note">${esc(r[6])}</td>
          <td class="${r[7].startsWith('0/')?'m-bad':'m-ok'}">${esc(r[7])}</td><td class="${r[8].startsWith('0/')?'m-bad':'m-ok'}">${esc(r[8])}</td></tr>`).join('')}</tbody></table></div>
      <p class="tr-prose">${esc(c.text)}</p>
      <div class="tr-wk-actions"><h4>这一轮实际走的动作</h4>
        ${c.rows.map(r=>`<div><b>${esc(r[0])}</b><code>${esc(r[9])}</code></div>`).join('')}</div>`;
  }
  function renderWalker(){
    const c=walkerCase();
    page.innerHTML=top('walker')+`
      <div class="tr-section-title"><div><div class="eyebrow muted">TRAJECTORY GENERATION</div>
        <h2>程序走链条，教师只填语言</h2>
        <p>把一条调查轨迹拆成两部分：结构（抽帧 → 对齐 → 差分 → 回看 → 交报告）由一个只读学生可见状态的程序策略走完，语言（任务文本、结论、最终答案）由教师模型分别填入。教师因此退出串行环，每条轨迹的调用从 26 次降到 2 次。</p></div>${pill('第二轮 wk2 · 2026-10-12')}</div>
      <div class="tr-traj-facts">${walkerFacts.map(([n,t])=>`<div><b>${esc(n)}</b><span>${esc(t)}</span></div>`).join('')}</div>

      <section class="tr-wk-section"><div class="tr-section-title"><div><h2>程序走的四条链</h2>
        <p>按学生当前可见的状态自上而下匹配，第一条命中的就是这一步的决定。停在哪里由比对结果决定，不依赖任何标准答案，所以这是一条学生能够模仿的合法路径。</p></div></div>
        <div class="tr-wk-chains">${walkerChain.map(b=>`<article><div class="tr-wk-chain-head"><strong>${esc(b.name)}</strong>${pill(b.who,b.who.toLowerCase())}</div>
          <ol class="tr-wk-steps">${b.steps.map(s=>`<li>${esc(s)}</li>`).join('')}</ol>
          <p>${esc(b.note)}</p></article>`).join('')}</div></section>

      <section class="tr-wk-section"><div class="tr-section-title"><div><h2>教师只填四个语言槽</h2>
        <p>四次调用互不依赖，可以并行、可以批量。</p></div></div>
        <div class="tr-matrix"><table>
          <thead><tr><th>语言槽</th><th>何时调用</th><th>输入</th><th>输出</th></tr></thead>
          <tbody>${walkerSlots.map(([a,b,d,e])=>`<tr><th scope="row">${esc(a)}</th><td class="m-note">${esc(b)}</td><td class="m-note">${esc(d)}</td><td class="m-note">${esc(e)}</td></tr>`).join('')}</tbody></table></div>
        <p class="tr-callout">${esc(walkerSlotNote)}</p></section>

      <section class="tr-wk-section"><div class="tr-section-title"><div><h2>三轮同案例对照</h2>
        <p>同样 9 个案例、同一套核验器。r0e 是教师模型走完全程的旧做法，wk1 是 walker 第一轮，wk2 是补完规则后的第二轮。</p></div><span>绿色为本轮最好</span></div>
        <div class="tr-matrix tr-wk-metrics"><table>
          <thead><tr><th>指标</th><th>r0e<small>教师走全程</small></th><th>wk1<small>walker 第一轮</small></th><th>wk2<small>walker 第二轮</small></th></tr></thead>
          <tbody>${walkerMetrics.map(([name,a,b,d,flag,note])=>`<tr><th scope="row">${esc(name)}${note?`<small>${esc(note)}</small>`:''}</th>
            <td>${esc(a)}</td><td>${esc(b)}</td><td class="${flag==='win'?'m-win':flag==='miss'?'m-bad':''}">${esc(d)}</td></tr>`).join('')}</tbody></table></div>
        <p class="tr-footnote">${esc(walkerHeldNote)}</p></section>

      <section class="tr-wk-section"><div class="tr-section-title"><div><h2>逐案看这一轮的变化</h2>
        <p>九个案例全部列出，包括两个回退的。点击左侧任一案例查看它三轮的数字、实际走过的动作，以及这一轮为什么变好或变差。</p></div></div>
        <div class="tr-flow-layout"><div class="tr-cases" aria-label="案例列表">${walkerCases.map(v=>{
          const [tl,tc]=walkerTags[v.tag];
          return `<button class="tr-case ${v.id===c.id?'active':''}" data-wk-case="${esc(v.id)}" aria-pressed="${v.id===c.id}">
            <span><strong>${esc(v.name)}</strong><small>${esc(v.kind)}</small></span><i class="tr-wk-dot ${tc}" title="${esc(tl)}"></i></button>`;}).join('')}</div>
          <article class="tr-stage-detail" aria-live="polite">${walkerCaseDetail(c)}</article></div></section>

      <section class="tr-wk-section"><div class="tr-section-title"><div><h2>这一轮修了什么</h2>
        <p>第一轮暴露出的五处，四处已改，一处判断为暂时不值得做。</p></div></div>
        <ul class="tr-traj-req tr-wk-fixes">${walkerFixes.map(([n,t,s])=>`<li class="${s}"><b>${esc(n)}</b><span class="tr-pill ${s==='done'?'':'amber'}">${s==='done'?'已改':'本轮未做'}</span><p>${esc(t)}</p></li>`).join('')}</ul></section>

      <section class="tr-wk-section"><div class="tr-section-title"><div><h2>验收与下一步</h2>
        <p>第一阶段的六条验收标准，过了五条。</p></div></div>
        <ul class="tr-wk-checks">${walkerAcceptance.map(([n,v,s])=>`<li class="${s}"><i aria-hidden="true">${s==='pass'?'✓':'✗'}</i><b>${esc(n)}</b><span>${esc(v)}</span></li>`).join('')}</ul>
        <ul class="tr-traj-req">${walkerNext.map(([n,t])=>`<li><b>${esc(n)}</b><p>${esc(t)}</p></li>`).join('')}</ul>
        <p class="tr-callout">${esc(walkerNextNote)}</p></section>

      <div class="tr-footnote">数据来自 2026 年 10 月 12 日的两次实际运行（wk1、wk2），规格 hi@0.7.0 / gia@0.8.0，核验器与 r0e 同一套。本轮跳过了用学生模型处理器逐行核对这一步（该模型未在本机）。</div>`;
  }
  function renderRubric(){
    page.innerHTML=top('rubric')+`
      <div class="tr-section-title tr-rubric-title"><div><div class="eyebrow muted">VERIFICATION AT EVERY DECISION</div>
        <h2>每一步输出，单独判定能否进入训练</h2>
        <p>判定不给分数，只贴统一定义的问题类型。类型自带严重度与是否影响后续结论，一行数据的去留由此得出。</p></div>${pill('核验器 0.2.0')}</div>
      <p class="tr-upgrade"><b>为什么逐步判定</b>训练数据的最小单位是一次决策，不是一整条调查。以第 0 轮实际跑通的案例为例：两份局部报告都被判未通过，按「整条收或整条不收」这 20 个决策一行都留不下；按逐步判定，其中 17 行是干净可学的。</p>
      <div class="tr-pipeline">${verifySteps.map(([n,t,d],i)=>
        `<div><span>${n}</span><strong>${esc(t)}</strong><p>${esc(d)}</p></div>${i<3?'<i>→</i>':''}`).join('')}</div>

      <section class="tr-units-section"><div class="tr-section-title"><div><h2>四种判定单元</h2>
        <p>一次调查会产生四类输出，各自单独判定，互不替代。某一份报告未通过，不代表它里面每一步动作都不能学。</p></div></div>
        <div class="tr-units">${verifyUnits.map(([id,name,role,text,goes])=>
          `<article class="tr-unit"><div class="tr-unit-head"><strong>${esc(name)}</strong>${pill(role,role.toLowerCase())}</div>
            <code>${esc(id)}</code><p>${esc(text)}</p><small>${esc(goes)}</small></article>`).join('')}</div></section>

      <section class="tr-verify"><div class="tr-section-title"><div><div class="eyebrow muted">EVIDENCE-ANCHORED VERIFICATION</div>
        <h2>三道工序：谁来判</h2>
        <p>不是三套标准，是同一条流水线上的三道工序。能用程序判的先用程序判，判不了的才调核验模型，最后再对着任务本身查一遍覆盖。前一道已判定的，后一道不再重复判定。</p></div></div>
        <div class="tr-verify-layers">${verifyLayers.map((v,i)=>`<article><div class="tr-verify-head"><span class="tr-verify-index">${String(i+1).padStart(2,'0')}</span><div><strong>${esc(v.name)}</strong><small>${esc(v.tag)}</small></div></div><p>${esc(v.text)}</p><div class="tr-verify-errors">${v.errors.map(e=>`<code>${esc(e)}</code>`).join('')}</div></article>`).join('')}</div>
        <div class="tr-callout">第二道要求核验模型给出引语，再由程序回到证据原文核查该引语是否存在，未查到即降级为「找不到依据」。核验所用模型与生成数据的模型相互独立，这是「结论经过实际核验」与「模型自评」之间的分界。</div>
        <div class="tr-matrix"><h3>四种单元各走哪几道</h3>
          <table><thead><tr><th>判定单元</th><th>程序检查</th><th>逐条证据锚定</th><th>充分性</th><th>说明</th></tr></thead>
            <tbody>${unitLayers.map(([name,id,a,b,c,note])=>`<tr><th scope="row">${esc(name)}<code>${esc(id)}</code></th>${
              [a,b,c].map(k=>`<td class="m-${k}" title="${esc(layerMark[k][1])}">${layerMark[k][0]}</td>`).join('')
            }<td class="m-note">${esc(note)}</td></tr>`).join('')}</tbody></table>
          <p class="tr-footnote">「另判」「反推」表示这一道对该单元不按原样执行，另有对应的判定方式。三道工序最初是为局部报告设计的，扩到其余三种单元时按各自的形态取舍——动作没有结论，就没有可锚定的对象。</p></div></section>

      <section class="tr-errors"><div class="tr-section-title"><div><h2>问题类型：判成什么</h2>
        <p>判定不许写「这里不太对」，必须从下面的类型表里选一个名字。类型统一定义于一处，每一次流程改进均须说明其拟消除的类型，改进因此可归因、可统计。</p></div><span>按发生层级分组</span></div>
        <div class="tr-error-groups">${errorGroups.map(g=>`<div><h3>${esc(g.layer)}<span>${g.items.length}</span></h3><div>${g.items.map(e=>`<code>${esc(e)}</code>`).join('')}</div><p>${esc(g.note)}</p></div>`).join('')}</div>
        <div class="tr-severity"><h3>每个类型自带两个属性，后果由它们算出</h3>
          <ul>${severityRows.map(([id,name,text,examples])=>
            `<li class="${id}"><b>${esc(name)}</b><code>${esc(id)}</code><p>${esc(text)}</p><small>${esc(examples)}</small></li>`).join('')}</ul>
          <p class="tr-callout">第二个属性是「是否污染后续结论」：引证类问题会连带影响所有依赖它的步骤，效率类问题不会。经校准后，轻微问题一律不再向下传播。</p></div></section>

      <section class="tr-calib"><div class="tr-section-title"><div><h2>核验器本身也要被核验</h2>
        <p>第 0 轮抽样复核发现几类误报——把本来正确的判成了错的。核验器随之升到 0.2.0，并陆续新增三条校准规则；规则在更多运行上的误放行率校验正在进行。</p></div></div>
        <ul class="tr-traj-req">${calibrationRules.map(([name,text])=>`<li><b>${esc(name)}</b><p>${esc(text)}</p></li>`).join('')}</ul>
        <p class="tr-callout">这几条修的是「误判为错」。反方向的校准程序也已落成：把判定干净的结论按六种方式故意改坏——换成别的证据、加一句编的、改掉数字、去掉「据……称」、引一个不存在的证据、删掉一条必需结论却仍报完成——每种改法事先就知道应该报什么错，改坏的报告用与正式核验完全相同的一遍流程再跑，抓到算命中，由此得出每种错误类型的召回率。第一轮结果：内容类的四种改法共 15 处全部抓到、没有误报；「删掉一条必需结论却仍报完成」在不提供案卷时抓不到——没有案卷就没有「该拿的清单」，下一版核验器会按每个工作区所对准的要求来查。在这一类补上之前，被判未通过的报告仍只做「不进正例」，不作为反例训练。</p></section>

      <section class="tr-admission"><div class="tr-section-title"><div><h2>三种处置</h2>
        <p>判定结果直接决定这一行数据的去向，不经过任何总分。</p></div></div>
        <div class="tr-dispositions tr-dispositions-3">${dispositionRows.map(([id,name,text])=>
          `<div class="tr-disposition ${id}"><span class="tr-disposition-dot"></span><h3>${esc(name)}</h3><code>${esc(id)}</code><p>${esc(text)}</p></div>`).join('')}</div>
        <div class="tr-admission-rule"><strong>进入 SFT 候选</strong><span>单元判定通过</span><i>＋</i><span>无致命问题且未被污染</span><i>＋</i><span>学生模型处理器逐行核对通过</span></div></section>

      <div class="tr-footnote">问题类型的唯一定义处是代码中的类型表，页面与核验器共用同一份。历史运行保留它当时实际使用的评审版本；新一轮起统一由三道工序重核，以核验结果为准。</div>
      <div class="tr-doc-links">${docLink('method','完整构造方法')}</div>`;
  }
  const trajectory = {
    case_id:'true:4124903', dataset:'true', run:'true_4124903_r1', mode:'bridge',
    task:'核查以下声明，必要时检查给定媒体并搜索外部证据。依据实际证据回答，说明结论范围与不确定性：Fire and smoke billowing from the vicinity of the Eiffel Tower were evidence of a Bastille Day terrorist attack in Paris.',
    description:'给一段 25 秒视频，核查"埃菲尔铁塔附近的火焰与浓烟是巴黎国庆日遭遇恐袭的证据"这一声明。实际事件是 2016 年 7 月 14 日耶拿桥上一辆烟花卡车意外起火。',
    facts:[['1 轮','GIA 派出两个并行工作区'],['22 次','角色决策（HI 20 · GIA 2）'],['24 条','登记的证据'],['19 行','进入训练候选']],
    steps:[
      {id:'step_1', role:'GIA', actor:'GIA', action:'INVESTIGATE', round_index:1, execution:'gia_decision_accepted',
       task:'核查声明：埃菲尔铁塔附近的火焰与浓烟是否是巴黎国庆日恐袭的证据。', task_count:2,
       included:true, weight:1, image_count:0, state:'accepted',
       produced:'两个互不依赖的局部任务',
       user:'[总任务] 核查以下声明……Fire and smoke billowing from the vicinity of the Eiffel Tower were evidence of a Bastille Day terrorist attack in Paris.\n[原始材料] C0 任务卡 | E1 video "材料 A" src=I0\n[本轮新证据] 无   [最新报告] 无   [历史索引] 无\n[约束] 第 1 轮 · turn 1/8 · 可开新轮 · 附图 0\n[可用动作] INVESTIGATE(tasks[{task, materials[]}]) · FINISH(answer) · QUERY_MEMORY · READ_RECORD',
       output:'{"decision": "INVESTIGATE",\n "tasks": [\n  {"task": "核查材料 A 视频的来源与事件语境……", "materials": ["C0", "E1"]},\n  {"task": "核查声明……重点查证外部可靠来源对相关火情原因及是否涉及恐怖袭击的记载。", "materials": ["C0", "E1"]}\n ]}',
       meta:'"verdict": "positive",\n"errors": [{"type": "overlapping_sibling_tasks", "severity": "minor"}],\n"gap": {"task_targets": {"T1": ["REQ1","REQ2","REQ3"], "T2": ["REQ1","REQ2"]}, "must_cite_untargeted": []},\n"round_yield": {"REQ1": "covered", "REQ2": "covered", "REQ3": "covered"},\n"premises": "pending_judge", "sample_weight": 1.0',
       note:'核验：两个任务都可解析、材料存在、没有"等 W1-1 确认后"这类依赖；声明被放在引号里当核查对象，不是当事实，没有未核实前提走私。缺口评判上 REQ1、REQ2 都有任务对准。\n但两个任务都写了"视频本身能否支持恐袭说法"，于是两个工作区各写了一条帧 Finding、各读了一遍 Snopes 和警方推文——这不是任何一方 HI 的重复动作，是派工层的重叠。这一类已经作为 overlapping_sibling_tasks（轻微）补进错误枚举，只计数、不影响收录。\n另外要说明：任务对准了哪几项要求，是用该工作区最终经核实的结论反推出来的，不是读任务文字判的，所以它只进元信息，不作为筛选门槛。'},

      {id:'step_2', role:'HI', actor:'W1-1', action:'INSPECT', operation:'get_frames', round_index:1,
       task:'核查材料 A 视频的来源与事件语境，判断其是否对应"法国国庆日期间埃菲尔铁塔附近出现火焰和浓烟"的场景。',
       preview:'均匀抽 8 帧看清场景 → E3–E10', included:true, weight:3, image_count:0, state:'accepted',
       produced:'E3 E4 E5 E6 E7 E8 E9 E10（8 帧）', gain:{a:8, o:0, q:0, t:8},
       stills:['e3','e4','e5','e6','e7','e8','e9','e10'],
       user:'[任务 W1-1] 核查材料 A 视频的来源与事件语境……\n[材料] C0 任务卡 | E1 video "材料 A" src=I0 by=HARNESS/ingest\n[候选来源] 无\n[证据视图] E1 | E2 text "媒体属性：video，25.100秒，1280×720，30fps，1条音轨" by=INSPECT/media_profile\n[焦点] 无   [已看正文] 无   [历史] 无   [最新观察] 无\n[约束] turn 1/12 · 直接视觉输入 · 图片上限 8 · 附图 0',
       output:'{"action": "INSPECT",\n "arguments": {"targets": ["E1"], "request": {"operation": "get_frames", "params": {"count": 8}}},\n "focus_evidence_ids": []}',
       meta:'"clue_gain": {"acquired": 8, "acquired_ids": ["E3","E4","E5","E6","E7","E8","E9","E10"], "observed": 0, "quote_page": 0, "total": 8},\n"sample_weight": 3.0, "images": [], "loss_scope": "last_assistant"',
       note:'"第一步先抽 8 帧"是最容易被当成套路的动作。这一步不删，但它的权重来自回溯：案卷把视频帧列为必需证据，这 8 帧全部落进有用证据集合，所以算高收益步，权重按上限封到 3。换一个帧根本没用上的案子，同样的动作权重就是 1。'},
      {id:'step_3', role:'HI', actor:'W1-1', action:'SEARCH', round_index:1, preview:'英文关键词检索 → 10 个候选，读到 5 个页面',
       included:true, weight:2, image_count:8, state:'accepted', produced:'R1–R10；E11 Snopes、E12–E15 报道', gain:{a:1, o:0, q:0, t:1}},
      {id:'step_4', role:'HI', actor:'W1-1', action:'READ_RECORD', round_index:1, preview:'读 Snopes 全文',
       included:true, weight:3, image_count:8, state:'accepted', produced:'E11 正文', gain:{a:0, o:1, q:1, t:2}},
      {id:'step_5', role:'HI', actor:'W1-1', action:'SEARCH', round_index:1, preview:'顺着 Snopes 的链接打开 YouTube 来源',
       included:true, weight:3, image_count:8, state:'accepted', produced:'E16 页面、E17 封面', gain:{a:2, o:0, q:0, t:2},
       stills:['e17'],
       user:'[候选来源] R1–R10 一行卡\n[证据视图] E1–E15 一行卡\n[焦点] E6 E10   [已看正文] E11\n[历史] t1 INSPECT→E3–E10；t2 SEARCH→10 候选 5 页；t3 READ E11\n[最新观察] E11 Snopes 全文摘要：…"seized upon photos and video of the event"…"No fire at the #TourEiffel. An accidental truck fire took place at Jena Bridge"…\n[约束] 附图 8',
       output:'{"action": "SEARCH",\n "arguments": {"source": "https://www.youtube.com/watch?v=cv_F3UkbdbI", "output_type": "video"},\n "focus_evidence_ids": ["E6", "E10"]}',
       meta:'"turn": 4, "clue_gain": {"acquired": 2, "acquired_ids": ["E16","E17"], "total": 2},\n"sample_weight": 3.0, "images": ["E3","E4","E5","E6","E7","E8","E9","E10"]'},
      {id:'step_6', role:'HI', actor:'W1-1', action:'READ_RECORD', round_index:1, preview:'读 YouTube 元数据：上传日期 2016-07-14',
       included:true, weight:3, image_count:8, state:'accepted', produced:'E16 元数据', gain:{a:0, o:1, q:1, t:2}},
      {id:'step_7', role:'HI', actor:'W1-1', action:'READ_RECORD', round_index:1, preview:'读 Mirror 报道',
       included:false, held:true, image_count:8, state:'held', produced:'E12 正文', gain:{a:0, o:0, q:0, t:0},
       stateNote:'线索收益为 0，且所在工作区被判 negative，暂缓入库。'},
      {id:'step_8', role:'HI', actor:'W1-1', action:'SEARCH', round_index:1, preview:'换法语关键词检索 → 三家法媒',
       included:true, weight:2, image_count:8, state:'accepted', produced:'E18 E19 E20', gain:{a:1, o:0, q:0, t:1}},
      {id:'step_9', role:'HI', actor:'W1-1', action:'READ_RECORD', round_index:1, preview:'读 franceinfo 全文',
       included:true, weight:3, image_count:8, state:'accepted', produced:'E19 正文', gain:{a:0, o:1, q:1, t:2}},
      {id:'step_10', role:'HI', actor:'W1-1', action:'SEARCH', round_index:1, preview:'打开巴黎警方推文 ②',
       included:true, weight:2, image_count:8, state:'accepted', produced:'E21', gain:{a:1, o:0, q:0, t:1}},
      {id:'step_11', role:'HI', actor:'W1-1', action:'READ_RECORD', round_index:1, preview:'读警方推文 ②：铁塔没着火，耶拿桥卡车意外起火',
       included:true, weight:3, image_count:8, state:'accepted', produced:'E21 正文', gain:{a:0, o:1, q:1, t:2}},
      {id:'step_12', role:'HI', actor:'W1-1', action:'READ_RECORD', round_index:1, preview:'再看一帧像素',
       included:false, held:true, image_count:8, state:'held', produced:'E3 像素（t2 起已随请求附图）', gain:{a:0, o:0, q:0, t:0},
       stateNote:'线索收益为 0，且所在工作区被判 negative，暂缓入库。'},
      {id:'step_13', role:'HI', actor:'W1-1', action:'REPORT', report_status:'COMPLETED', round_index:1,
       preview:'提交报告：F1 F2 F3 F4', included:false, image_count:8, state:'rejected',
       produced:'四条 Finding', stateNote:'仍判未通过；这类「未做完却报完成」的错误召回率还测不出来，这一行暂不作为反例训练。',
       meta:'"verdict": "negative",\n"errors": [{"type": "completed_without_finishing", "severity": "critical", "status_should_be": "INCOMPLETE"}],\n"tainted_findings": []（校准后 F3 的过度外推按 R1 豁免）,\n"bucket_reason": "negative_report_until_recall_calibrated", "sample_weight": null',
       note:'GIA 派的任务第一句就是"核查材料 A 视频的来源"；F2 自己写着"不能确认它就是 A 的直接来源"，报告却报了 COMPLETED。这是 HI 相对它自己收到的任务的错，不是 GIA 的错。\nF3 那条"用一帧推整段"的过度外推，校准后判定豁免——它实际引了 7 条证据且全部支持。所以这份报告现在没有被污染的 Finding，但"该报未完成却报了完成"这一条保留。\n下一步是配上同状态重生成的正例（同一桌面、状态改成 INCOMPLETE、其余 Finding 不动），凑成一对对照样本。'},

      {id:'step_14', role:'HI', actor:'W1-2', action:'READ_RECORD', round_index:1, preview:'读警方推文 ②（本工作区首次）',
       task:'核查声明"埃菲尔铁塔附近升腾的火焰和浓烟是巴黎在法国国庆日遭遇恐怖袭击的证据"，重点查证外部可靠来源对火情原因及是否涉及恐袭的记载。',
       included:true, weight:3, image_count:8, state:'accepted', produced:'E21 正文', gain:{a:0, o:1, q:1, t:2}},
      {id:'step_15', role:'HI', actor:'W1-2', action:'READ_RECORD', round_index:1, preview:'读 Var-Matin',
       included:true, weight:1, image_count:8, state:'accepted', produced:'E20 正文', gain:{a:0, o:0, q:0, t:0},
       stateNote:'线索收益为 0，但所在工作区经校准后判为通过，按规则以权重 1 收录——没有收益不等于不该走这一步。'},
      {id:'step_16', role:'HI', actor:'W1-2', action:'SEARCH', round_index:1, preview:'英文关键词检索 → NYDN、Express',
       included:true, weight:2, image_count:8, state:'accepted', produced:'E22 E23', gain:{a:1, o:0, q:0, t:1}},
      {id:'step_17', role:'HI', actor:'W1-2', action:'READ_RECORD', round_index:1, preview:'读 NYDN 全文',
       included:true, weight:3, image_count:8, state:'accepted', produced:'E22 正文', gain:{a:0, o:1, q:1, t:2}},
      {id:'step_18', role:'HI', actor:'W1-2', action:'READ_RECORD', round_index:1, preview:'读 Snopes（本工作区首次）',
       included:true, weight:3, image_count:8, state:'accepted', produced:'E11 正文', gain:{a:0, o:1, q:1, t:2}},
      {id:'step_19', role:'HI', actor:'W1-2', action:'SEARCH', round_index:1, preview:'打开巴黎警方推文 ①',
       included:true, weight:2, image_count:8, state:'accepted', produced:'E24', gain:{a:1, o:0, q:0, t:1}},
      {id:'step_20', role:'HI', actor:'W1-2', action:'READ_RECORD', round_index:1, preview:'读警方推文 ①：烟花卡车技术事故起火',
       included:true, weight:3, image_count:8, state:'accepted', produced:'E24 正文', gain:{a:0, o:1, q:1, t:2}},
      {id:'step_21', role:'HI', actor:'W1-2', action:'REPORT', report_status:'COMPLETED', round_index:1,
       preview:'提交报告：F5 F6 F7', included:true, image_count:8, state:'accepted',
       produced:'三条 Finding', stateNote:'第 0 轮原判未通过；核验器校准后转为通过，整份报告进入候选。',
       meta:'"verdict": "positive"（校准前为 negative）,\n"errors": [{"type": "incomplete_citation_set", "severity": "major", "propagates": false},\n           {"type": "scope_overreach", "severity": "minor"}],\n"tainted_findings": [],\n"extra": {"calibration": {"before": "unsupported_assertion / critical", "applied": "R2"}}',
       note:'F7 写了"这与报道描述相容"，却只引了四帧画面、没引任何报道。按逐条证据锚定的隔离规则判"无支持"没错，但把它定成致命错会让整个工作区翻成未通过——支持其实就在本工作区可见、而且已经被 F5、F6 引用过的报道里，只是这条 Finding 没带上。\n校准规则把这一类"关系句引用不全"单独降级为 incomplete_citation_set（重要，但不向后传播污染），于是这份报告转为通过。这正是要先校准核验器、再用它筛数据的理由：核验器偏严，会把本来干净的一整批数据扔掉。'},

      {id:'step_22', role:'GIA', actor:'GIA', action:'FINISH', round_index:2,
       task:'综合两份报告，给出最终回答。', preview:'声明不实；视频来源仍未独立核实',
       included:true, weight:2, image_count:8, state:'accepted', produced:'最终答案，引用 E24 E21 E11 E22 E3 E7 E16',
       user:'[本轮新证据] E3–E10 帧 | E11 Snopes | E12–E15 报道 | E16 YouTube 页 | E17 封面 | E18–E20 法媒 | E21 警方推文② | E22 NYDN | E23 Express | E24 警方推文①\n[最新报告]\n  W1-1 COMPLETED · F1 帧显示铁塔附近大量烟雾、低处明火 ←[E3 E4 E5 E7 E8 E9] · F2 YouTube 上传 2016-07-14 ←[E16 E17 E11]，限制：未确认是 A 的直接来源 · F3 与 2016-07-14 火情相符 ←[…] · F4 画面不足以定性恐袭 ←[…]\n  W1-2 COMPLETED · F5 声明不实；警方通报耶拿桥烟花卡车技术事故起火 ←[E24 E21] · F6 NYDN 与 Snopes 均将巴黎火灾与尼斯袭击区分 ←[E22 E11] · F7 帧显示烟雾火光，与报道相容但不能证明恐袭 ←[E3 E5 E7 E10]\n[约束] 第 2 轮 · 附图 8：E3 E17 E5 E4 E7 E10 E8 E9',
       output:'{"decision": "FINISH",\n "answer": "结论：针对2016年7月14日埃菲尔铁塔附近的已报道火情，该声明不实。巴黎警方当日通报，耶拿桥上一辆装载烟花的卡车因技术事故起火；随后明确确认这是意外火灾，已经扑灭，埃菲尔铁塔本身没有着火。（E24、E21）同期发生在尼斯的恐怖袭击与巴黎这起火灾应予区分；相关火情影像被误传为巴黎遭袭的证据。（E11、E22）\\n\\n给定视频画面确实显示铁塔附近有烟雾、低处有明火，但画面本身不能确定起火原因或证明恐怖袭击。（E3、E7）目前找到相关公开视频的上传日期为2016年7月14日，但尚未确认它与给定视频的完整画面对应，因此给定视频的拍摄日期及确切来源仍未独立核实。（E16）"}',
       meta:'"verdict": "positive", "error_types": [{"type": "scope_overreach", "severity": "minor"}],\n"tainted_input": false, "recovered_from_tainted": true,\n"exec": {"all_visible": true, "label_expected": "false", "label_match": true, "date_check": "2016-07-14 in_range"},\n"anchored": {"method": "judge_sentences", "sentences": 7, "pairs": 13, "unsupported_sentences": [], "calibration": "R4 撤销 1 处反驳"},\n"sufficiency": {"REQ1": "covered", "REQ2": "covered", "REQ3": "declared_unverified", "premature_finish": false},\n"sample_weight": 2.0',
       note:'结论与数据集标签一致。核验模型把答案拆成 7 句、13 组句子–证据对逐一核对，7 句全部有依据（下方"最终答案逐句核对"按引用归为四组展示）。最见功力的是第四句：W1-1 已经把"核查视频来源"报成完成，答案却写"仍未独立核实"——整个调查确实没做过帧对齐，GIA 没有跟着抄，反而把结论收窄了。\n上游被判未通过、下游把它纠正回来，这是最有价值的一类正例，权重给到 2。'}],
    checks:[
      ['label_expected', '答案判定应为"不实"'],
      ['entity_in_evidence', '所引证据中须出现 Iéna 或 artifices'],
      ['date_in_range', '事件日期 2016-07-14（±1 天）']],
    requirements:[
      ['REQ1', '必须引用', '官方或可靠来源对铁塔附近火情原因的记载，以及是否涉及恐怖袭击'],
      ['REQ2', '必须引用', '给定视频画面实际显示了什么，画面本身能否支持恐袭说法'],
      ['REQ3', '可不引用', '给定视频的公开来源与日期（可确认则给出，不能确认则说明）']],
    verdicts:[
      ['W1-1 工作区', 'negative', '原判：报了完成，但任务要求的"核查视频来源"并没有做完（致命）；另有 F3 用一帧推整段（轻微）。校准后过度外推获豁免——它实际引了 7 条证据且全部支持——"该报未完成却报完成"这一条保留，工作区仍判未通过。'],
      ['W1-2 工作区', 'positive', '原判未通过：F7 的"这与报道描述相容"只引了四帧画面、没引任何报道，被判无支持（致命）。校准后降级为"引用集合不全"（重要，不传播污染）——支持其实就在本工作区已经引用过的报道里。工作区转为通过。'],
      ['GIA 派工', 'positive', '两个任务可解析、材料存在、无兄弟依赖、无未核实前提；三项案卷要求都有任务对准。唯一问题是两个任务范围重叠，记为轻微。'],
      ['GIA 回答', 'positive', '引用的 7 条证据都在视图里，案卷标签与日期都命中；拆成 7 句逐句核对，全部有依据（一处被较早的警方推文判为矛盾，同句所引的后一条推文已直接支持，按 R4 撤销），另有一处轻微的范围问题；并且纠正了上游过早声明完成的问题。']],
    timeline:[
      ['当晚', '巴黎国庆烟花放完后，耶拿桥上一辆装烟花的卡车起火，铁塔周围冒出大量烟。'],
      ['21:51 · E24', '巴黎警察局：因技术事故，一辆装烟花的卡车在耶拿桥起火，请避开该区域。'],
      ['22:33 · E21', '巴黎警察局：不要传播假消息，铁塔没有起火，是一起意外卡车火灾，已扑灭。'],
      ['同一晚', '尼斯发生卡车冲撞袭击，于是"巴黎也遭袭、铁塔被点燃"的说法在社交媒体和一些媒体上传开。'],
      ['事后', 'Snopes、纽约每日新闻、franceinfo 的核查都指向同一结论：这条声明不实。']],
    sentences:[
      ['第一句', '火情原因、意外定性、已扑灭、铁塔未起火', '两条警方推文原文里逐字可查（E24、E21）'],
      ['第二句', '与尼斯袭击区分，相关影像被误传', 'Snopes 与纽约每日新闻原文支持（E11、E22）'],
      ['第三句', '画面显示烟和火，但不能确定起火原因', '帧像素支持，而且没有多说（E3、E7）', ['e3','e7']],
      ['第四句', '找到的公开视频上传于 2016-07-14，但与给定视频的对应关系仍未独立核实', '元数据支持上传日期（E16）；"尚未确认"是如实限定', ['e17']]],
    sentenceNote:'这里按引用把答案归为四组展示；核验器实际拆成 7 句逐句核对，结论相同。唯一可以挑的地方在第一句的限定语"针对 2016 年 7 月 14 日埃菲尔铁塔附近的已报道火情"——它证明了那天的火是意外，却没有证明给定视频拍的就是那天那场火。答案把这个缺口放在第四句如实交代了，所以整体是诚实的；更严格的写法会把结论再收一档："若给定视频确为该次火情，则声明不实；视频与该事件的对应关系本次未独立确认。"这不影响它进正例，属于可以在规格里提醒的写法问题。',
    tokens:[
      ['进入候选 · HI', '17 行', '输入 2.6K – 13.6K，中位 10.1K', '监督中位 35'],
      ['进入候选 · GIA', '2 行', '输入 1.7K 与 16.7K', '监督 161 / 196'],
      ['暂缓', '2 行', '输入 9.2K – 10.0K', '监督 39 / 43'],
      ['待校准', '1 行', '输入 14.5K', '监督 593']],
    tokenNote:'用学生模型的真实处理器逐行核对：零失败、零超长（上限 32 768）。一张 1280×720 的图折合 880 token，最长的一行是带 8 张图的最终回答。监督 token 仍然很薄——这一批数据没有留下教师的推理过程，补不上。',
    boundaries:[
      ['逐对核验有固有盲区', '答案逐句对证据的核验一次只看一份证据，会把「更早的状态」误读成「相反」。本例的一处就是这样，已由校准规则 R4 处理；这类规则仍需在更多运行上校验。'],
      ['缺口评判有两种口径', '任务对准了哪几项要求，一种由核验模型读任务文字判定，一种用该工作区最终经核实的结论反推；两者只在一个任务上有出入，以前者为准。缺口只记录，不作为筛选门槛。'],
      ['召回率只测了一轮', '内容类错误全部抓到，但「未做完却报完成」这类状态级错误在不提供案卷时抓不到；在补上之前，未通过的报告暂不作为反例训练。'],
      ['一个案例不是数据集', '19 行训不出任何东西。这是流水线跑通的验证，接下来要把同模式的其余运行逐个写案卷，用同一条命令导出。']],
    stills:[
      ['e3','E3','1.57 秒','镜头对着铁塔：塔身亮灯、结构完好，右侧已经有烟飘过。'],
      ['e4','E4','4.70 秒','镜头下移到河岸，浓烟成团升起，岸边出现一处橙色火光。'],
      ['e5','E5','7.83 秒','火点在岸边的建筑与树木之间，不在塔上。'],
      ['e6','E6','10.97 秒','铁塔基座与浓烟同框，塔身结构完整、照明未断。'],
      ['e7','E7','14.13 秒','火光贴近岸边地面，浓烟向上扩散，塔身只露在画面左缘。'],
      ['e8','E8','17.27 秒','岸边游船与路灯清晰可辨，火点始终在岸上。'],
      ['e9','E9','20.40 秒','水面出现蓝色闪光反射，与应急车辆一致。'],
      ['e10','E10','23.53 秒','镜头拉远，铁塔完整，火点在塔外的岸边。']],
    stillNote:'这八帧不是插图，是模型当时真正看进去的画面——抽帧那一步产出它们，之后每一步都带着它们做决定，最终回答里"画面显示烟和火、但看不出起火原因"这一句，核验时就是回到这些像素上逐条比对的。',
    sourceStill:['e17','E17','来源封面','Snopes 正文里链接的 YouTube 视频「Smoke rises around Eiffel Tower」的封面，元数据上传日期 2016-07-14。它的取景与材料 A 的八帧不同（更靠近塔身），本次没有做逐帧比对，所以最终回答第四句如实写了来源"仍未独立核实"。'],
    yieldNote:'第 0 轮原判两份报告都未通过。按旧的"整条轨迹要么全收要么全不收"的办法，这 20 个决策一行都留不下；按逐步判定加上核验器校准，其中 17 行是干净可学的。'
  };
  const trajStates = {accepted:['进入候选','accepted'], held:['暂缓','held'], rejected:['待校准后入负例','rejected']};
  const TRAJ_STILL_DIR = '/assets/traj/';
  function trajStillMap(){
    if (!trajectory._stills) {
      const map = new Map();
      trajectory.stills.forEach(([k, id, time, caption]) => map.set(k, {id, time, caption}));
      const [k, id, time, caption] = trajectory.sourceStill;
      map.set(k, {id, time, caption});
      trajectory._stills = map;
    }
    return trajectory._stills;
  }
  function trajStill(key){
    const s = trajStillMap().get(key); if (!s) return '';
    return `<button type="button" class="tr-still" data-still="${esc(key)}" title="${esc(s.caption)}">
      <img src="${TRAJ_STILL_DIR}${esc(key)}-t.jpg" width="320" height="180" loading="lazy" decoding="async"
        alt="${esc(s.id + ' ' + s.time + '：' + s.caption)}">
      <span><b>${esc(s.id)}</b><i>${esc(s.time)}</i></span></button>`;
  }
  function trajStillStrip(keys, kind){
    return `<div class="tr-stills${kind ? ' ' + kind : ''}">${keys.map(trajStill).join('')}</div>`;
  }
  function openStill(key){
    const s = trajStillMap().get(key); if (!s) return;
    const order = trajectory.stills.map(r => r[0]).concat([trajectory.sourceStill[0]]);
    const at = order.indexOf(key);
    let box = document.getElementById('tr-lightbox');
    if (!box) {
      box = document.createElement('div');
      box.id = 'tr-lightbox'; box.className = 'tr-lightbox';
      box.setAttribute('role', 'dialog'); box.setAttribute('aria-modal', 'true');
      document.body.appendChild(box);
    }
    box.dataset.at = String(at);
    box.innerHTML = `<div class="tr-lightbox-inner" role="document">
      <img src="${TRAJ_STILL_DIR}${esc(key)}.jpg" alt="${esc(s.id + ' ' + s.time + '：' + s.caption)}">
      <figcaption><b>${esc(s.id)} · ${esc(s.time)}</b><p>${esc(s.caption)}</p></figcaption>
      <nav><button type="button" data-still-step="-1" aria-label="上一张">←</button>
        <span>${at + 1} / ${order.length}</span>
        <button type="button" data-still-step="1" aria-label="下一张">→</button></nav>
      <button type="button" class="tr-lightbox-close" data-still-close aria-label="关闭">✕</button></div>`;
    box.onclick = e => { if (e.target === box) closeStill(); };
    box.classList.add('open');
    box.querySelector('.tr-lightbox-close')?.focus({preventScroll:true});
  }
  function stepStill(delta){
    const box = document.getElementById('tr-lightbox'); if (!box) return;
    const order = trajectory.stills.map(r => r[0]).concat([trajectory.sourceStill[0]]);
    const at = (Number(box.dataset.at || 0) + delta + order.length) % order.length;
    openStill(order[at]);
  }
  function closeStill(){
    const box = document.getElementById('tr-lightbox');
    if (box) { box.classList.remove('open'); box.innerHTML = ''; }
  }
  document.addEventListener('keydown', e => {
    const box = document.getElementById('tr-lightbox');
    if (!box || !box.classList.contains('open')) return;
    if (e.key === 'Escape') { closeStill(); }
    else if (e.key === 'ArrowLeft') { stepStill(-1); }
    else if (e.key === 'ArrowRight') { stepStill(1); }
  });

  function trajCounts(){
    const hi = trajectory.steps.filter(s => s.role === 'HI');
    return {hi:hi.length, gia:trajectory.steps.length - hi.length,
            accepted:hi.filter(s => s.state === 'accepted').length,
            held:hi.filter(s => s.state === 'held').length,
            rejected:hi.filter(s => s.state === 'rejected').length};
  }
  function trajBlock(title, body){ return `<section class="tr-detail-section"><h3>${title}</h3>${body}</section>`; }
  function trajStepDetail(step){
    const [stateLabel, stateClass] = trajStates[step.state] || ['—',''];
    const gain = step.gain ? `<ul class="tr-arch-facts"><li>新取得证据 ${step.gain.a} 条</li><li>首次读到正文 ${step.gain.o} 条</li><li>可引用的页面 ${step.gain.q} 条</li><li>合计 ${step.gain.t}</li></ul>` : '';
    return `<div class="tr-step-heading"><div><span class="tr-panel-kicker">${esc(step.actor)} · ${step.role === 'GIA' ? '第 ' + step.round_index + ' 轮' : '第 ' + trajTurn(step) + ' 步'}</span>
        <h3>${esc(actionTitle(step))}</h3></div>${pill(step.role, step.role.toLowerCase())}</div>
      <div class="tr-traj-verdict ${stateClass}"><b>${esc(stateLabel)}</b>${step.weight == null ? '' : `<span>权重 ${step.weight}</span>`}
        ${step.stateNote ? `<p>${esc(step.stateNote)}</p>` : ''}</div>
      ${step.preview ? trajBlock('这一步做了什么', `<p class="tr-prose">${esc(step.preview)}</p>`) : ''}
      ${step.produced ? trajBlock('产出 / 读取', `<p class="tr-prose">${esc(step.produced)}</p>`) : ''}
      ${step.stills ? trajBlock('这一步得到的画面', trajStillStrip(step.stills) + '<p class="tr-footnote">点开任意一张查看大图。</p>') : ''}
      ${gain ? trajBlock('线索收益', gain) : ''}
      ${step.user ? trajBlock('当时看到的状态（compact_8k 示意）', `<pre class="tr-code">${esc(step.user)}</pre>`) : ''}
      ${step.output ? trajBlock('模型这一步的输出', `<pre class="tr-code">${esc(step.output)}</pre>`) : ''}
      ${step.meta ? trajBlock('数据行元信息', `<pre class="tr-code">${esc(step.meta)}</pre>`) : ''}
      ${step.note ? trajBlock('为什么这样判', step.note.split('\n').map(p => `<p class="tr-prose">${esc(p)}</p>`).join('')) : ''}`;
  }
  function trajTurn(step){
    return trajectory.steps.filter(s => s.actor === step.actor && s.role === 'HI').indexOf(step) + 1;
  }
  function trajGroupDetail(node){
    const steps = (node.steps || []).filter(s => s);
    const counted = steps.reduce((acc, s) => { acc[s.state] = (acc[s.state] || 0) + 1; return acc; }, {});
    const rows = Object.entries(counted).map(([k, v]) => `<li>${esc((trajStates[k] || [k])[0])} ${v} 条</li>`).join('');
    return `<div class="tr-step-heading"><div><span class="tr-panel-kicker">${esc(node.badge || '分组')}</span><h3>${esc(node.label)}</h3></div></div>
      <p class="tr-prose">${esc(node.subtitle || '')}</p>
      ${rows ? trajBlock('这一组的处置', `<ul class="tr-arch-facts">${rows}</ul>`) : ''}
      <p class="tr-footnote">点击脑图里的任意一步，查看它当时看到什么、输出了什么、为什么这样判。</p>`;
  }
  function setPane(pane){
    T.mobilePane=pane;
    const layout=page.querySelector('.tr-record-layout');if(!layout)return;
    layout.dataset.pane=pane;
    page.querySelectorAll('[data-tr-pane]').forEach(b=>{const active=b.dataset.trPane===pane;
      b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});
  }
  function selectTrajNode(id){
    const node = T.map?.nodes.get(id); if (!node) return;
    const main = page.querySelector('#tr-step-main'); if (!main) return;
    main.innerHTML = node.step ? trajStepDetail(node.step) : trajGroupDetail(node);
    setPane('detail');
  }
  function renderTrajectory(){
    const c = trajCounts();
    trajectory.step_count = trajectory.steps.length;
    T.map?.destroy();
    page.classList.add('tr-detail-page');
    page.innerHTML = top('samples') + `<div class="tr-section-title tr-samples-title"><div><div class="eyebrow muted">ONE CASE, END TO END</div>
        <h2>一个案例，从调查到训练数据</h2><p>${esc(trajectory.description)}</p></div></div>
      <div class="tr-traj-facts">${trajectory.facts.map(([n, t]) => `<div><b>${esc(n)}</b><span>${esc(t)}</span></div>`).join('')}</div>
      <div class="tr-condition">${esc(trajectory.task)}</div>
      <div class="tr-reader-tabs" aria-label="阅读视图"><button data-tr-pane="map" aria-pressed="true" class="active">调查脑图</button><button data-tr-pane="detail" aria-pressed="false">节点详情</button></div>
      <div class="tr-record-layout" data-pane="map"><section class="tr-map-panel" id="tr-map-panel" aria-label="调查脑图"></section>
        <section class="tr-step-main" id="tr-step-main" aria-live="polite"></section></div>
      <section class="tr-traj-stills"><h3>模型看到的画面</h3>
        <p class="tr-prose">${esc(trajectory.stillNote)}</p>
        ${trajStillStrip(trajectory.stills.map(r => r[0]))}
        <div class="tr-still-source">${trajStill(trajectory.sourceStill[0])}
          <p class="tr-prose">${esc(trajectory.sourceStill[3])}</p></div></section>
      <section class="tr-traj-chain"><h3>最终答案逐句核对</h3>
        <ol class="tr-traj-sentences">${trajectory.sentences.map(([n, what, basis, keys]) =>
          `<li><b>${esc(n)}</b><p>${esc(what)}</p><small>${esc(basis)}</small>${keys ? trajStillStrip(keys, 'mini') : ''}</li>`).join('')}</ol>
        <p class="tr-callout">${esc(trajectory.sentenceNote)}</p></section>
      <section class="tr-traj-extra">
        <div><h3>案卷要求</h3><ul class="tr-traj-req">${trajectory.requirements.map(([id, must, txt]) =>
          `<li><b>${esc(id)}</b><span class="tr-pill ${must === '必须引用' ? '' : 'amber'}">${esc(must)}</span><p>${esc(txt)}</p></li>`).join('')}</ul>
          <ul class="tr-arch-facts">${trajectory.checks.map(([k, txt]) => `<li><code>${esc(k)}</code> ${esc(txt)}</li>`).join('')}</ul></div>
        <div><h3>核验结果</h3><ul class="tr-traj-verdicts">${trajectory.verdicts.map(([who, v, why]) =>
          `<li class="${v}"><b>${esc(who)}</b><span>${v === 'positive' ? '通过' : '未通过'}</span><p>${esc(why)}</p></li>`).join('')}</ul></div>
        <div><h3>这一个案例产出了什么</h3><ul class="tr-arch-facts"><li>HI 决策 ${c.hi} 次：${c.accepted} 行进入候选、${c.held} 行暂缓、${c.rejected} 行待校准</li>
          <li>GIA 决策 ${c.gia} 次：派工与最终回答各 1 条，都进入候选</li>
          <li>合计 ${c.accepted + c.gia} 行可训练数据，全部通过学生模型处理器的逐行核对</li>
          <li>GIA 的数据天然比 HI 少一个数量级，所以离线换题是 GIA 的主要扩样手段</li></ul>
          <p class="tr-prose">${esc(trajectory.yieldNote)}</p></div>
      </section>
      <section class="tr-traj-extra">
        <div><h3>事件本身</h3><ul class="tr-traj-timeline">${trajectory.timeline.map(([when, what]) =>
          `<li><b>${esc(when)}</b><p>${esc(what)}</p></li>`).join('')}</ul></div>
        <div><h3>这批数据能不能训</h3><ul class="tr-traj-tokens">${trajectory.tokens.map(([name, rows, inp, sup]) =>
          `<li><b>${esc(name)}</b><span>${esc(rows)}</span><p>${esc(inp)} · ${esc(sup)}</p></li>`).join('')}</ul>
          <p class="tr-prose">${esc(trajectory.tokenNote)}</p></div>
        <div><h3>本例的适用边界</h3><ul class="tr-traj-req">${trajectory.boundaries.map(([t, d]) =>
          `<li><b>${esc(t)}</b><p>${esc(d)}</p></li>`).join('')}</ul></div>
      </section>
      <div class="tr-footnote">本例取自真实运行 ${esc(trajectory.run)}，GIA 与 HI 均由教师模型实际决策；核验结论取自第 0 轮实际核验记录。状态示意按 compact_8k 档手写，未保留教师推理的步骤不补写推理。</div>`;
    T.map = new TrainingMindMap(trajectory, {titleForStep:actionTitle, onStep:selectTrajNode,
      onGroup:node => selectTrajNode(node.id), onFilter:value => { T.map.setFilter(value); }});
    T.map.filter = 'all';
    // 默认把整条轨迹摊开：这一页的重点就是"一个案例产出哪些数据行"，收起来反而要点好几下才看得到。
    T.map.expanded = new Set(['root', 'round-1', 'round-2']);
    T.map.mount(page.querySelector('#tr-map-panel'));
    selectTrajNode('root');
    setPane('map');
  }
  // 论文草稿：内容与公式、图在 paper.js（window.ForenPaper），这里只负责挂载、语言切换与目录定位。
  function paperLang(){try{const v=localStorage.getItem('fs-paper-lang');if(v==='en'||v==='zh')return v;}catch(e){}return 'zh';}
  function renderPaper(){
    const P=window.ForenPaper;
    if(!P||!P.render){page.innerHTML=top('paper')+'<div class="tr-empty"><strong>论文草稿未加载</strong><p>缺少 /assets/paper.js。</p></div>';return;}
    const lang=paperLang(),r=P.render(lang),L=r.L;
    page.innerHTML=top('paper')+`<div class="pp-shell pp-lang-${lang}" lang="${lang==='en'?'en':'zh-CN'}">
      <aside class="pp-side"><div class="pp-switch" role="group" aria-label="语言 / Language">${[['zh','中文'],['en','English']].map(([k,l])=>`<button type="button" data-paper-lang="${k}" aria-pressed="${k===lang}">${l}</button>`).join('')}</div>${r.toc}</aside>
      <article class="pp-paper"><div class="pp-draft">${L.draft}</div>
        <header class="pp-head"><h1>${L.title}</h1><div class="pp-authors">${L.authors}</div></header>
        <section class="pp-abs"><h2>${L.abstractTitle}</h2><p>${L.abstract}</p></section>
        ${r.body}
      </article></div>`;
    paperSpy();
  }
  let paperTicking=false;
  function paperSpy(){
    const btns=[...page.querySelectorAll('.pp-toc button')];if(!btns.length)return;
    let cur=btns[0].dataset.paperJump;
    for(const b of btns){const el=document.getElementById(b.dataset.paperJump);if(el&&el.getBoundingClientRect().top<140)cur=b.dataset.paperJump;}
    btns.forEach(b=>b.classList.toggle('active',b.dataset.paperJump===cur));
  }
  window.addEventListener('scroll',()=>{if(T.section!=='paper'||paperTicking)return;paperTicking=true;requestAnimationFrame(()=>{paperTicking=false;paperSpy();});},{passive:true});
  async function open(hash='#training'){
    stop();T.map?.destroy();closeStill();page.classList.remove('tr-detail-page');clearTimeout(S.poll);stopPlayback();const serial=++T.serial;showPage('training-page');navigate(hash);
    if(!T.data){loading();try{T.data=await get('/api/training');}catch(e){if(serial===T.serial)page.innerHTML=`<div class="tr-empty"><strong>训练记录暂时不可用</strong><p>${esc(e.message)}</p><button class="secondary" data-tr="retry">重新读取</button></div>`;return;}}
    if(serial!==T.serial)return;
    const parts=hash.split('/');
    T.section=['rubric','samples','flow','architecture','walker','paper'].includes(parts[1])?parts[1]:'roadmap';
    if(T.section==='rubric')renderRubric();else if(T.section==='samples')renderTrajectory();else if(T.section==='flow')renderFlow();else if(T.section==='walker')renderWalker();
    else if(T.section==='architecture')renderArchitecture();else if(T.section==='paper')renderPaper();else renderRoadmap();window.scrollTo({top:0});
  }
  window.openTraining=open;
  document.addEventListener('click',async e=>{
    const b=e.target.closest('button,a');if(!b)return;
    if(b.dataset.trainingRoute){e.preventDefault();await open(b.dataset.trainingRoute);return;}
    if(b.dataset.archZoom){const m=b.dataset.archZoom;if(m==='fit')archFit();else archZoomAt(T.archZoom*(m==='in'?1.2:1/1.2));return;}
    if(b.dataset.archNode){T.archNode=b.dataset.archNode;renderArchStage();archFocus(T.archNode);page.querySelector(`[data-arch-node="${T.archNode}"]`)?.focus({preventScroll:true});return;}
    if(b.dataset.wkCase){T.walkerCase=b.dataset.wkCase;renderWalker();page.querySelector(`[data-wk-case="${T.walkerCase}"]`)?.focus({preventScroll:true});return;}
    if(b.dataset.trStage!==undefined){T.stage=Number(b.dataset.trStage);renderFlow();page.querySelector(`[data-tr-stage="${T.stage}"]`)?.focus({preventScroll:true});return;}
    if(b.dataset.still){openStill(b.dataset.still);return;}
    if(b.dataset.stillStep){stepStill(Number(b.dataset.stillStep));return;}
    if(b.dataset.stillClose!==undefined){closeStill();return;}
    if(b.dataset.trPane){setPane(b.dataset.trPane);return;}
    if(b.dataset.paperLang){try{localStorage.setItem('fs-paper-lang',b.dataset.paperLang);}catch(err){}const y=window.scrollY;renderPaper();window.scrollTo({top:y});return;}
    if(b.dataset.paperJump){document.getElementById(b.dataset.paperJump)?.scrollIntoView({behavior:'smooth',block:'start'});return;}
    if(b.dataset.tr==='retry'){T.data=null;T.cache.delete('/api/training');open(location.hash);}
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
})();
