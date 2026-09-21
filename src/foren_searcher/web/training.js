"use strict";
(() => {
  const page = document.querySelector('#training-page');
  const T = {data:null, section:'flow', stage:2, role:'HI', dimension:'grounding', filter:'all', query:'', dataset:'all', sample:null, step:null, stepFilter:'all', tab:'overview', archView:'overview', archNode:'gia', cache:new Map(), serial:0, playing:false, timer:null, map:null, mobilePane:'map'};
  const labels = {SEARCH:'搜索与打开来源',INSPECT:'检查原始材料',READ_RECORD:'读取已有证据',QUERY_MEMORY:'查找调查记忆',SEARCH_MEMORY:'查找调查记忆',INVESTIGATE:'安排下一轮调查',FINISH:'结束并回答',REPORT:'提交局部报告'};
  const operations = {get_frames:'提取视频帧',align_media:'对齐视频片段',get_pages:'读取文档页面',crop:'裁剪局部画面',ocr:'读取图中文字',locate_frames:'定位视频片段'};
  const dispositions = {accept_candidate:'可作为候选',repair_reexecute:'需要重新执行',context_only:'仅保留为历史',exclude:'排除目标',unknown:'尚未判定'};
  const modes = {bridge:'GIA × HI 真实协作',hi_independent:'HI 独立调查',gia_offline:'GIA 离线决策'};
  const stageData = [
    {title:'案例选取与依据准备',short:'案例与依据',subtitle:'先界定问题边界，再准备核验材料。',input:'真实问题、原始媒体、来源记录与数据集标注',actor:'数据构造',output:'明确的任务 + 训练侧参考材料',points:['按原始案例、事件或视频来源划分开发与保留检查数据，同一案例的变体放在同一侧。','核对原始材料、时间轴和来源条件。给定候选视频、给定网页与自主搜索分别记录。','参考标注辅助核验；不把缺失证据补写成已经观察过的事实。'],example:'video-correspondence',exampleLabel:'查看双视频对应任务',note:'对应片段可证明内容匹配，但不足以单独判定原创归属。',upgrade:'案例按数据集标注统一入库，每个案例附带来源政策与必需证据；开发集与保留检查集按案例家族划分，并由程序自动核查两侧是否重叠。'},
    {title:'决策输入构造',short:'决策输入构造',subtitle:'每次决策均有明确的信息边界。',input:'当前任务、已取得证据、已有报告与历史',actor:'上下文组织',output:'本轮角色输入 + 实际图片附件',points:['HI 获得自己的局部任务和可见材料；GIA 获得总任务、报告及当前可用证据。','标准答案、评分记录和未来调查结果留在核验侧，不能提前进入决策输入。','独立 GIA 样本可用已核实的真实报告构造状态。报告可信，仍可能没有查完整。'],example:'gia-evidence-states',exampleLabel:'对比同一证据下的四个问题',note:'问题改变时，“已有证据是否足够”的答案也可能改变。',upgrade:'新增状态渲染器：长上下文分两档压缩，实测将调查员一侧的输入字符中位数降至原先的一半以下，证据编号无缺失；超出预算时明确标注已截断，并保留读取全文的入口。'},
    {title:'教师生成与工具真实执行',short:'生成与执行',subtitle:'完整记录调查过程，支持逐步回放。',input:'当轮可见状态',actor:'GIA / HI 教师',output:'一个决定 → 实际返回 → 下一轮状态',points:['一次教师请求只生成一个角色的输出。工具调用由现有 Harness 实际执行。','保存原始输入、图片、模型输出、工具观察和执行对应关系。失败也按实际结果记录。','GIA 可以独立生成下一步决定；只有接入 HI 并真实执行，才计为协作轨迹。'],example:'video-intervals',exampleLabel:'展开 106 次真实模型输出',note:'检索、抽帧、转写等工具由 Harness 真实执行，非模拟返回。',upgrade:'新增版本化规格：面向学生的角色提示与面向教师的生成指令分开管理，各自带版本号，四项内容统一计算哈希并写入每一行数据，任一样本均可追溯到其规格版本。'},
    {title:'逐步评分与问题定位',short:'逐步评分',subtitle:'同时评估调查完成度与该步输出的示范价值。',input:'真实输入、输出、执行记录 + 核验依据',actor:'独立评审会话',output:'0–4 分 + 理由 + 每步处置',points:['按七个维度检查执行、可见性、事实依据、任务、决策、不确定性与效率。','行动按当时可知信息评价；报告按实际取得的证据核验。合理搜索未命中，不自动算错误。','用明确错误与合法替代样本检验评分标准。评审缺少关键原文或图片时，保留未知。'],example:'video-local-intervals',exampleLabel:'查看逐步评分依据',note:'评分针对每一步单独进行，不以整条调查的平均分替代单步判定。',upgrade:'新增三层核验器：程序检查、逐条证据锚定与任务充分性依次执行。核验模型与生成模型相互独立，核验时回到证据原文与像素，引语须可由程序在原文中回查到。'},
    {title:'筛选、修复与处置',short:'筛选与修复',subtitle:'逐目标判定，不以整体平均分替代单步结论。',input:'逐目标评分 + 实际执行核对',actor:'筛选与重新执行',output:'候选目标 / 重新执行 / 仅作历史 / 排除',points:['逐目标判为可收录、评分 3 或 4、无关键评审未知，并通过执行及输入对应检查，才进入候选。','有实质错误的动作，从对应状态重新执行并复评，不把改写答案接到不相容的旧后续。','错误前缀可以作为之后的真实上下文；它本身不因此成为正向学习目标。'],example:'video-intervals',exampleLabel:'查看 86 条收录与 20 条未收录',note:'如实声明「尚未查完」的报告同样具备示范价值，但不代表该项调查已经完成。',upgrade:'处置结果与统一的问题类型枚举绑定：每一条被退回的输出均归入具体类型，使改进方向可按类型统计，而非仅留一段自由文本说明。'},
    {title:'按角色导出与输入核对',short:'导出与核对',subtitle:'模型仅学习当前角色的合格输出。',input:'通过核验的 GIA 与 HI 决策',actor:'分角色 SFT 数据',output:'当前上下文 → 下一次正确输出',points:['GIA 与 HI 分开导出；开发数据和构造方法的保留检查数据分开保存。','系统提示、用户输入、工具观察及其他角色内容作为上下文；只有最后一条目标输出计算训练损失。','用 Qwen3.5-9B 的实际模板、分词和图片处理检查输入、附件与监督位置，核对是否截断。'],example:'dog-source',exampleLabel:'打开一条实际 SFT 样本',note:'导出后逐条核对监督位置与截断情况，确认模型实际接收的输入与设计一致。',upgrade:'每一行数据均附带规格版本、核验器版本、问题类型与长度估算，训练、评估与后续改进可直接依这些字段切分，无需回溯运行目录。'}
  ];
  const dimensions = [
    {id:'execution',name:'执行对应',weight:10,hi:'原始请求、接受输出、工具观察或提交报告能够对应。',gia:'下发任务、报告与决定能够对应；脚本安排不当作模型输出。',question:'该步骤是否确实执行',detail:'调用成功与内容正确分别检查。离线 GIA 决定尚未执行，执行维度可以不适用。',example:'video-correspondence'},
    {id:'visibility',name:'信息可见性',weight:15,hi:'只使用当时可见的材料和合法引用，不借用标注、未来结果或其他工作区的报告。',gia:'只依据当前证据和报告，不能提前知道后续调查结果。',question:'当时是否可见这些信息',detail:'曾经取得的图片、当前附带的图片、报告保留的事实，是不同的信息状态。',example:'dog-source'},
    {id:'grounding',name:'事实与证据',weight:25,hi:'重要断言有直接依据，引文、来源与位置准确，表达没有扩大事实。',gia:'从局部事实到全局结论的推断成立，不扩大时间、对象和适用范围。',question:'所引证据是否支持该结论',detail:'“来源声称某事”不等于某事已被独立证实；批准预算也不等于实际花费。',example:'university-budget'},
    {id:'task',name:'任务完成',weight:15,hi:'按接到的局部问题检查必要部分，并准确报告哪些部分尚未完成。',gia:'按总问题检查必要信息，不能因为局部报告标为完成就直接结束。',question:'任务要求是否全部覆盖',detail:'看到一些差异时刻，可能足以回答“有没有差异”，还不足以回答“全部区间在哪里”。',example:'gia-evidence-states'},
    {id:'decision',name:'决策合理性',weight:15,hi:'行动针对合理的未决问题，工具与参数合适，失败后能够调整。',gia:'分工目标清楚、粒度可执行、依赖合理，补查和停止符合当前证据。',question:'该步骤是否推进当前问题',detail:'允许存在多条合理路径，不要求每一步直接得到最终答案，也不与唯一参考计划逐字比对。',example:'gia-evidence-states'},
    {id:'uncertainty',name:'不确定性',weight:15,hi:'区分采样观察、工具处理范围、连续区间与没有检查的内容。',gia:'理解可靠报告仍可能不完整，考虑来源范围与实际执行能力。',question:'已确定与未确定的边界是否清晰',detail:'如实说明未完成部分同样是合格报告；预算耗尽后提交的报告不能用于证明自主停止能力。',example:'video-local-intervals'},
    {id:'efficiency',name:'调查效率',weight:5,hi:'避免没有信息价值的重复，允许必要复看与没有结果的合理探索。',gia:'不重复分工，不继续与原问题无关的调查。',question:'是否在填补具体缺口',detail:'重复查看同一材料不必然是错误，关键在于该次复查是否针对一个具体的未决问题。',example:'video-intervals'}
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
  function top(section){return `<div class="tr-top"><div><div class="eyebrow muted">METHOD &amp; PROGRESS</div><h1>方法与进展</h1></div><span class="tr-readonly"><i class="small-dot"></i> 已保存记录 · 浏览不调用模型</span></div><nav class="tr-tabs" aria-label="训练数据栏目">${[['roadmap','项目进展'],['architecture','系统架构'],['flow','生成流程'],['rubric','质量核验'],['samples','样本库']].map(([id,label])=>routeButton(`#training${id==='roadmap'?'':'/'+id}`,`${label}${id==='samples'?`<span>${T.data.samples.length}</span>`:''}`,`tr-tab ${section===id?'active':''}`)).join('')}</nav>${page.classList.contains('tr-detail-page')?'':statusStrip(section)}`;}
  function statusStrip(section){return `<div class="tr-status-strip"><span class="tr-status-now">阶段 3 / 5 已完成</span><span>调查框架、数据流水线与质量核验体系均已就位</span><span>候选决策 <b>${format(T.data.stats.total)}</b> 条 · 数据来源 <b>${T.data.stats.datasets}</b> 个</span><span>正在进入首次模型训练</span>${section==='roadmap'?'':routeButton('#training','看项目进展 →')}</div>`;}
  function renderFlow(){
    const s=T.data.stats,d=stageData[T.stage];
    page.innerHTML=top('flow')+`<section class="tr-hero"><div><div class="eyebrow">FROM INVESTIGATION TO LEARNING</div><h2>把每一次真实调查，<br>沉淀为可复用的训练示范<span class="accent">。</span></h2><p>真实执行留下完整过程，逐步评审选出示范样本。<br>GIA 学习判断与分工，HI 学习取证与报告。</p></div><div class="tr-hero-card"><span>当前候选决策</span><strong>${format(s.total)}<small>条</small></strong><div class="tr-role-split"><span class="tr-role-gia">GIA <b>${s.gia}</b></span><span class="tr-role-hi">HI <b>${s.hi}</b></span></div><div class="tr-stat-note">逐步决策数，非独立案例数</div></div></section><div class="tr-stats"><div><b>${format(s.development)}</b><span>开发与训练准备</span></div><div><b>${format(s.validation)}</b><span>构造方法保留检查</span></div><div><b>${s.datasets}</b><span>数据来源</span></div><div><b>${format(s.images)}</b><span>实际附件文件 · 含重复画面</span></div></div><section class="tr-flow-section"><div class="tr-section-title"><div><div class="eyebrow muted">DATA GENERATION PIPELINE</div><h2>一条训练样本的产生过程</h2></div><span>选择步骤，查看输入与产物</span></div><div class="tr-flow-layout"><div class="tr-stages" aria-label="数据生成步骤">${stageData.map((v,i)=>`<button class="tr-stage ${i===T.stage?'active':''}" data-tr-stage="${i}" aria-pressed="${i===T.stage}"><span>${String(i+1).padStart(2,'0')}</span><div><strong>${v.short}</strong><small>${i===2?'模型与工具实际运行':i===3?'逐个决策独立判定':i===5?'GIA / HI 分开保存':i===0?'任务、来源与参考材料':i===1?'隔离参考答案与未来信息':'保留有据可依的示范'}</small></div><i>↗</i></button>`).join('')}</div><article class="tr-stage-detail" aria-live="polite"><div class="tr-panel-kicker">步骤 ${String(T.stage+1).padStart(2,'0')} / 06</div><h3>${d.title}</h3><p class="tr-subtitle">${d.subtitle}</p><div class="tr-transformation"><div><small>输入</small><strong>${d.input}</strong></div><span>→</span><div class="tr-transform-actor"><small>${d.actor}</small><strong>${d.output}</strong></div></div><ol class="tr-points">${d.points.map(p=>`<li>${p}</li>`).join('')}</ol><p class="tr-callout">${d.note}</p><p class="tr-upgrade"><b>本期升级 · 2026-09</b>${d.upgrade}</p><div class="tr-stage-footer">${routeButton(`#training/samples/${d.example}`,`${d.exampleLabel} <span>→</span>`)}<div><button class="tr-round" data-tr-stage="${T.stage-1}" ${T.stage===0?'disabled':''} aria-label="上一步">←</button><button class="tr-round" data-tr-stage="${T.stage+1}" ${T.stage===5?'disabled':''} aria-label="下一步">→</button></div></div></article></div><div class="tr-return"><span>↶</span> 需要修复的动作回到对应状态重新执行，保留原来的真实历史。</div></section><section class="tr-supervision"><div><div class="eyebrow muted">WHAT THE MODEL LEARNS</div><h2>上下文用于理解，<br>目标输出才计入监督。</h2><p>一段完整调查被拆分为逐次决策的训练样本。</p></div><div class="tr-message-stack"><div><span>输入上下文</span><p>角色提示 · 当前任务 · 已有证据与报告</p><small>不计算目标损失</small></div><i>↓</i><div class="tr-target-message"><span>当前角色输出</span><p>GIA 的下一次决定 / HI 的下一次行动或报告</p><small>合格输出参与监督</small></div></div></section><div class="tr-footnote">候选数据已完成自动核验与实际多模态输入检查；首次模型训练正在准备中。保留检查数据用于构造方法自检，不作为公开测试集。</div><div class="tr-doc-links">${docLink('method','完整构造方法')} ${docLink('results','实际测量与结果')}</div>`;
  }
  const planStates={done:['已完成','done'],doing:['进行中','doing'],todo:['下一阶段','todo']};
  const roadmap=[
    {id:'framework',kicker:'阶段 01 · 已完成',title:'调查框架与工具链',hint:'真实任务端到端验证',items:[
      {state:'done',name:'两级调查角色',text:'GIA 负责全局：拆解问题、并行派发调查任务、判断证据是否足以形成结论。HI 在各自独立的工作区内执行单项任务，取证后提交局部报告。'},
      {state:'done',name:'动态调查轮次',text:'轮内并行、轮间串行。证据全局共享，结论按工作区隔离，避免各工作区之间的判断相互干扰。'},
      {state:'done',name:'检索与检查工具',text:'检索侧分为来源、打开、抽取三层；材料检查侧支持视频抽帧、片段对齐、文档读页、区域裁剪、图中文字识别、语音转写与片段定位。'},
      {state:'done',name:'可追溯的证据库',text:'全局证据持久化存储，具备读取审计与断点恢复。结论链条固定为：最终回答 → 发现 → 证据 → 来源。'},
      {state:'done',name:'多模态输入',text:'GIA 与 HI 均可直接读取原始画面。视频类任务中，调度方若不接触原始材料则只能派发模板化任务，该路径现已打通。'},
      {state:'done',name:'真实任务验证',text:'网页事实核查与视频比对任务均已通过真实接口端到端验证，全过程留有可回放的记录。'}]},
    {id:'pipeline',kicker:'阶段 02 · 已完成',title:'训练数据流水线',hint:'826 条候选决策 · 6 个数据来源',items:[
      {state:'done',name:'逐步决策语料',text:'共 826 条候选决策，其中全局调度 108 条、局部调查 718 条，覆盖网页核查、视频比对、图表问答等多类任务。'},
      {state:'done',name:'完整过程留存',text:'每一步均保存当时的输入、图片附件、模型输出与工具真实返回，以及三者之间的对应关系，支持逐步回放。'},
      {state:'done',name:'2 926 份实际附件',text:'训练样本中出现的每一张图，均为调查过程中真实取得的材料，而非事后补配的示意图。'},
      {state:'done',name:'划分与防泄漏',text:'开发集与保留检查集按案例家族划分，由程序核查两侧是否重叠；保留数据一旦被用于调整流程即自动降级。'},
      {state:'done',name:'按角色导出',text:'两个角色分别导出，仅当前角色的目标输出参与训练，其余内容作为上下文；导出后逐条核对监督位置与截断情况。'}]},
    {id:'quality',kicker:'阶段 03 · 已完成',title:'质量与核验体系',hint:'结论可回溯至原始证据',items:[
      {state:'done',name:'版本化规格',text:'角色提示与生成指令分开管理，各自带版本号与内容哈希，并写入每一行数据，任一样本均可追溯到产生它的规格版本。'},
      {state:'done',name:'状态渲染器',text:'长上下文分两档压缩，输入字符中位数降至原先的一半以下，证据编号全部保留；超出预算时明确标注，并保留读取全文的入口。'},
      {state:'done',name:'三层核验',text:'第一层以程序核查引用与必需项，第二层逐条结论对照其引用的每一份证据原文或像素，第三层依据任务本身核查覆盖度与终止条件。'},
      {state:'done',name:'统一的问题类型枚举',text:'核验结果归入统一定义的问题类型，共五个层级，每类标注角色、严重度与是否影响后续结论，使改进方向可量化。'},
      {state:'done',name:'核验与生成分离',text:'核验所用模型与生成数据的模型相互独立，并要求给出引语、由程序回到证据原文中回查，以避免自评。'}]},
    {id:'training',kicker:'阶段 04 · 进行中',title:'首次模型训练',hint:'本地链路 → 服务器 SFT',items:[
      {state:'done',name:'模型口径已确定',text:'学生模型 Qwen3.5-9B（原生支持图片与视频），教师模型 Qwen3.5-122B-A10B，本地调试采用同系列 4B，确保本地与服务器使用同一条多模态路径。'},
      {state:'doing',name:'第 0 轮数据核验',text:'以三层核验器对已有语料统一重核，形成第一份按问题类型统计的质量基线。'},
      {state:'doing',name:'本地训练链路',text:'在本地打通「取数据 → 训练一步 → 保存 LoRA → 加载推理 → 打分」的完整链路，确保上服务器前各环节均已验证。'},
      {state:'todo',name:'服务器首次 SFT',text:'在 8×A100 上以开发集训练学生模型，标志项目由数据构造阶段进入模型能力验证阶段。'},
      {state:'todo',name:'保留集对照评估',text:'在训练中未出现的数据集上做对照评估，检验所学为通用调查能力还是特定任务的固定模式。'}]},
    {id:'loop',kicker:'阶段 05 · 下一阶段',title:'自我发展闭环',hint:'核验结果驱动迭代改进',items:[
      {state:'todo',name:'质量诊断',text:'将核验结果汇总为问题类型分布与跨轮次曲线，明确下一轮的优先改进方向。'},
      {state:'todo',name:'规格迭代',text:'每次修订绑定其拟消除的问题类型并记录变更，使改进可归因、可回滚。'},
      {state:'todo',name:'对照轨迹',text:'自同一状态重新生成，得到改进前后成对的样本，用于验证本次修订是否确实消除了目标问题。'},
      {state:'todo',name:'留出门控',text:'每一轮改动须在留出集上通过门控方予采纳，并记录核验分数与实际提升之间的相关性。'},
      {state:'todo',name:'消融与迁移',text:'依据六次训练的比较计划，逐项验证架构中各项设计的实际收益，以及能力向新任务的迁移情况。'}]}
  ];
  function planCount(state){return roadmap.reduce((n,b)=>n+b.items.filter(i=>i.state===state).length,0);}
  const archViews = [
    {id:'overview', name:'一次调查怎么跑', width:1180, height:614,
     caption:'轮内并行、轮间串行。GIA 决定查什么，HI 在各自独立的工作区里查，Harness 负责真实执行并登记证据。',
     nodes:[
      {id:'input', x:20, y:214, w:150, h:104, kind:'input', title:'任务与原始材料',
       lines:['问题 + 视频 / 图片 / PDF'], badge:'2026-09 更新',
       detail:['用户提交的每一份材料都会登记成一条证据 E，带上模态、来源与处理链，后续所有引用都指回它。',
               '2026-09 起，图片原件以及视频、PDF 已经取得的画面会随全局视图一起送进 GIA——在此之前调度方只拿得到文字卡片。']},
      {id:'gv', x:212, y:36, w:236, h:168, kind:'view', title:'全局视图',
       lines:['总任务','原始材料','本轮读到的证据','最新一轮报告','更早结果索引'],
       detail:['固定五块，由 Harness 构造，GIA 不需要自己维护状态，也不输出"待办清单"。',
               '看不到的历史可以用 QUERY_MEMORY 检索、用 READ_RECORD 按编号读全文或调取图片。',
               '超出预算时按"可恢复优先"裁剪，并置 context_truncated 标记。']},
      {id:'render', x:212, y:232, w:236, h:78, kind:'render', title:'状态渲染器', badge:'2026-09 新增',
       lines:['legacy / compact_8k / compact_16k'],
       detail:['模型调用前把历史与报告正文压到预算之内，证据编号一个不丢，装不下的正文标注已截断并保留读取全文的入口。',
               '在 340 行试点语料实测：HI 侧字符中位从 23.1K 降到 11.3K（16K 档），GIA 侧从 24.9K 降到 18.5K。',
               '本地与服务器各一份配置，同一份代码两个档位。']},
      {id:'gia', x:212, y:338, w:236, h:118, kind:'gia', title:'GIA 全局调度', badge:'2026-09 更新',
       lines:['决定：INVESTIGATE / FINISH','动作：QUERY_MEMORY · READ_RECORD'],
       detail:['只做两件事：判断现有证据够不够回答总问题，以及把总问题拆成若干条互不依赖的局部任务。',
               '它不直接调用检索或检查工具——那是工作区的事。',
               '现在能直接看到原始材料的画面：视频类任务的文字里没有线索，不看画面只能派出模板化的任务。']},
      {id:'finish', x:212, y:492, w:236, h:96, kind:'final', title:'最终回答',
       lines:['每句结论后跟支持它的 E 编号'],
       detail:['Harness 只机械检查这些编号是否存在、能否解析回来源，不替 GIA 判断答案对不对。',
               '"结论是否真的被证据支持"由离线的三层核验器回答，不在运行时这条链路上。']},
      {id:'round', x:496, y:36, w:346, h:396, kind:'group', title:'第 t 轮',
       lines:['轮内并行 · 轮间串行'],
       detail:['同一轮里的工作区并行执行，轮与轮之间串行。工作区不能再派生子工作区。',
               '任务编号由 Harness 统一分配（W1-1、W1-2 …），GIA 只给任务文本和材料。']},
      {id:'ws1', x:516, y:108, w:306, h:78, kind:'ws', title:'工作区 W t,1',
       lines:['HI θ · 独立局部状态'],
       detail:['每个工作区拿到自己的任务、材料和一份按本任务排序的证据视图。',
               '所有工作区共享同一套 HI 参数 θ，但不共享上下文与判断。']},
      {id:'ws2', x:516, y:198, w:306, h:78, kind:'ws', title:'工作区 W t,2',
       lines:['HI θ · 独立局部状态'],
       detail:['兄弟工作区之间看不到彼此的实时状态、工具历史与 Finding。',
               '它们能共享的只有已经保存完整的证据。']},
      {id:'wsm', x:516, y:310, w:306, h:78, kind:'ws', title:'工作区 W t,m',
       lines:['HI θ · 独立局部状态'],
       detail:['一轮里派出几个工作区由 GIA 决定，受每轮上限约束。',
               '某个工作区超时或出错不会取消其他工作区，它的错误本身会成为下一轮的输入。']},
      {id:'report', x:888, y:196, w:174, h:150, kind:'report', title:'工作区报告',
       lines:['status: COMPLETED / INCOMPLETE','findings: 陈述 + 证据编号','selected_evidence_ids','status_reason'],
       detail:['HI 自审：所派的这项任务完成了没有。这是第一层审核。',
               '如实声明"尚未查完"同样是合格报告——它准确描述了边界。']},
      {id:'agg', x:888, y:36, w:262, h:96, kind:'harness', title:'Harness 轮聚合',
       lines:['收齐 · 落库 · 构造下一轮视图'],
       detail:['等这一轮全部工作区到达终态，把报告与本轮实际用到的证据打包成 GIA 的新视图。',
               'GIA 随后做第二层审核：这些结果能否成为总任务的有效信息，继续还是结束。']},
      {id:'store', x:888, y:392, w:262, h:158, kind:'store', title:'持久化全局库',
       lines:['证据 E：跨工作区共享','发现 F：只交给 GIA','报告 · 轮次 · 读取审计'],
       detail:['SQLite 持久化，支持断点恢复。已完成轮次的证据全局可见，当前轮的证据在聚合前只有产出它的工作区能看到。',
               '共享的是保存完整的证据，不是判断——兄弟工作区不接收也检索不到别人的 Finding。']}],
     links:[
      {from:'input', to:'gv', dir:'h'}, {from:'gv', to:'render', dir:'v'}, {from:'render', to:'gia', dir:'v'},
      {from:'gia', to:'round', dir:'h', label:'INVESTIGATE'}, {from:'gia', to:'finish', dir:'v', label:'FINISH'},
      {from:'ws2', to:'report', dir:'h'}, {from:'report', to:'agg', dir:'vup'},
      {from:'agg', to:'gv', dir:'over', label:'下一轮'},
      {from:'store', to:'wsm', dir:'hback', dash:true, label:'读取 · 复用编号'},
      {from:'report', to:'store', dir:'v', dash:true, label:'归档'}]},

    {id:'workspace', name:'工作区内部', width:1180, height:600,
     caption:'HI 每一步只做一个决定：选一个动作，或者提交报告。工具由 Harness 真实执行，返回的观察再进入下一步。',
     nodes:[
      {id:'state', x:20, y:150, w:196, h:210, kind:'view', title:'局部工作记忆',
       lines:['当前任务与材料','候选来源 R','按本任务排序的证据视图','自己保留的重点证据','自己的历史 · 最新一次返回'],
       detail:['明确看不到：兄弟工作区的实时状态、整个证据库的完整内容、其他工作区的工具历史、数据集标签、GIA 的推理。',
               '证据只保存一份，每个工作区各自决定排序与展示，完整内容按需展开。']},
      {id:'pixels', x:20, y:392, w:196, h:96, kind:'render', title:'画面直接进入 HI', badge:'已转正',
       lines:['每次最多 8 张图片像素'],
       detail:['原图、抽出的视频帧、渲染的 PDF 页面以像素形式随决策请求送进模型，长边缩放并受总像素上限约束。',
               '正因为 HI 自己看得见画面，早期设计里把"看图"外包给工具的 visual_qa / audio_qa 才被取消。']},
      {id:'hi', x:266, y:236, w:190, h:124, kind:'gia', title:'HI 局部调查员',
       lines:['选择一个动作','或提交局部报告'],
       detail:['一次请求只产出一个决定：SEARCH、INSPECT、QUERY_MEMORY、READ_RECORD 之一，或者一份报告。',
               '输出是结构化 JSON，解析或角色校验不通过会进入有限次数的修复，仍不通过则记为失败。']},
      {id:'search', x:512, y:40, w:250, h:104, kind:'tool', title:'SEARCH',
       lines:['检索 · 打开来源 · 登记材料','网页 / 图片 / 视频 / 以图搜图'],
       detail:['两种互斥调用：用文字或图片去搜，或者直接打开一个已知 URL / 已有的 R 编号。',
               '候选来源 R 不会原地升级成证据 E——看到搜索摘要不等于读过正文。',
               '视频类结果默认只拿到来源页信息，下载是 INSPECT 内部的事。']},
      {id:'inspect', x:512, y:164, w:250, h:158, kind:'tool', title:'INSPECT', badge:'2026-09 更新',
       lines:['抽帧 · 读文档页 · 裁剪','定位区域 · 图中文字 · 语音转写','定位片段 · 片段对齐'],
       detail:['当前实际注册了这 8 个处理器，全部作用于已经登记的证据编号。',
               'visual_qa 与 audio_qa 已从主线移除：HI 直接看画面，不需要把"看图"当成一次工具调用。']},
      {id:'detect', x:512, y:342, w:250, h:74, kind:'reserved', title:'DETECT', badge:'位置保留',
       lines:['待接入'],
       detail:['设计里保留给第三个工具的位置，方向是生成与伪造检测。',
               '当前还没有接入任何检测器，HI 的动作表里也还没有它——接入之后这一格才会有内容。']},
      {id:'mem', x:512, y:436, w:250, h:104, kind:'tool', title:'QUERY_MEMORY · READ_RECORD',
       lines:['检索已有记录','按编号读全文或调取图片'],
       detail:['查已保存的证据、发现、候选来源、报告与工作区任务卡片。',
               'READ_RECORD 分页读长文本，也能把某条证据的图片重新调出来看。',
               '新的抽帧、裁剪、转写不走这里，那是 INSPECT 的事。']},
      {id:'exec', x:816, y:200, w:170, h:166, kind:'harness', title:'Harness 执行与登记',
       lines:['真实执行','登记新的证据 E','写入执行审计'],
       detail:['工具调用全部真实执行，失败也按实际结果记录，不伪造返回。',
               '每条新证据带上来源、处理链和取得方式，这是后面能回溯的前提。']},
      {id:'obs', x:816, y:400, w:170, h:112, kind:'report', title:'观察 Observation',
       lines:['编号 · 来源 · 范围'],
       detail:['一次工具执行后返回给 HI 看的紧凑文字，本身不是一种证据类型，只是"这次工具说了什么"。',
               '它里面带的编号指向真正的 R 或 E。']},
      {id:'submit', x:266, y:40, w:190, h:112, kind:'report', title:'提交局部报告',
       lines:['COMPLETED / INCOMPLETE','格式与引用检查'],
       detail:['HI 自审这项任务完成了没有，Harness 只检查格式与引用编号是否存在。',
               '报告交给 GIA，兄弟工作区读不到。']}],
     links:[
      {from:'state', to:'hi', dir:'h'}, {from:'pixels', to:'hi', dir:'h'},
      {from:'hi', to:'search', dir:'h'}, {from:'hi', to:'inspect', dir:'h'},
      {from:'hi', to:'detect', dir:'h', dash:true}, {from:'hi', to:'mem', dir:'h'},
      {from:'search', to:'exec', dir:'h'}, {from:'inspect', to:'exec', dir:'h'},
      {from:'detect', to:'exec', dir:'h', dash:true}, {from:'mem', to:'exec', dir:'h', dash:true},
      {from:'exec', to:'obs', dir:'v'},
      {from:'obs', to:'state', dir:'under', label:'进入下一步'},
      {from:'hi', to:'submit', dir:'v-up'}]},

    {id:'records', width:1180, height:374, name:'记录与证据链',
     caption:'四层记录的边界是整套设计里最容易混淆、也最要紧的一处：读到了什么，和由此判断了什么，必须分开。',
     nodes:[
      {id:'source', x:40, y:56, w:214, h:128, kind:'input', title:'来源 R',
       lines:['搜索结果里的候选','标题 · 摘要 · 链接'],
       detail:['只是线索。候选来源不会原地升级成证据——没读正文就不能当成读过。']},
      {id:'evidence', x:326, y:56, w:214, h:128, kind:'view', title:'证据 E',
       lines:['实际读到的材料','或工具对它的可追溯结果'],
       detail:['六字段卡片：编号、实际模态、简短标签、来源引用、派生自哪条、用什么行动取得。',
               '登记为证据只表示能追到来源和处理过程，不表示结果已被证实。证据之间可以互相矛盾。']},
      {id:'finding', x:612, y:56, w:214, h:128, kind:'gia', title:'发现 F',
       lines:['HI 结合证据对局部任务','得出的判断'],
       detail:['发现不是证据，也不能自动转成证据。它属于产生它的工作区，随报告交给 GIA。',
               '例：证据是"页面显示上传日期为 2022-04-18"，发现是"该视频至少在那天已经公开出现"。',
               '不能从上传日期直接生成"视频拍摄于那天"这样的证据——那已经是推断。']},
      {id:'answer', x:898, y:56, w:214, h:128, kind:'final', title:'最终回答',
       lines:['GIA 综合全部报告','证据编号跟在它支持的句子后'],
       detail:['GIA 用 Finding 时必须能沿它的证据引用回到原始来源。',
               '证据链的形状是固定的：最终回答 → 发现 → 证据 → 来源。']},
      {id:'share', x:40, y:240, w:500, h:104, kind:'store', title:'共享与隔离',
       lines:['证据 E 跨工作区共享 · 发现 F 只交给 GIA'],
       detail:['共享的是保存完整的证据，不是判断。兄弟工作区不接收、也检索不到别人的 Finding。',
               '每个工作区应当根据任务、原始材料和证据自己形成判断。']},
      {id:'verify', x:612, y:240, w:500, h:104, kind:'render', title:'离线三层核验', badge:'2026-09 新增',
       lines:['程序检查 → 逐条证据锚定 → 任务充分性'],
       detail:['"这条结论是否真的被它引用的证据支持"，运行时的两层自审都不管，由离线核验器回答。',
               '第二层把每条结论与它引用的每一份证据配对，单独判定并要求给出引语，引语须能由程序在原文中回查到。',
               '核验所用模型与生成数据的模型相互独立，避免自评。']}],
     links:[
      {from:'answer', to:'finding', dir:'hback', label:'支持'},
      {from:'finding', to:'evidence', dir:'hback', label:'引用'},
      {from:'evidence', to:'source', dir:'hback', label:'取自'}]}
  ];
  const archKinds = {input:'原始输入', view:'上下文', render:'新增机制', gia:'角色决策', group:'轮次', ws:'工作区',
                     tool:'工具', reserved:'位置保留', harness:'Harness 执行', report:'产出记录', final:'最终产物', store:'持久化'};
  function archBox(view, id) { return view.nodes.find(n => n.id === id); }
  function archPath(view, link) {
    const a = archBox(view, link.from), b = archBox(view, link.to);
    if (!a || !b) return null;
    const ac = {x:a.x + a.w / 2, y:a.y + a.h / 2}, bc = {x:b.x + b.w / 2, y:b.y + b.h / 2};
    switch (link.dir) {
      case 'v': return {d:`M ${ac.x} ${a.y + a.h} C ${ac.x} ${a.y + a.h + 26}, ${bc.x} ${b.y - 26}, ${bc.x} ${b.y}`,
                        at:{x:(ac.x + bc.x) / 2, y:(a.y + a.h + b.y) / 2}};
      case 'v-up': return {d:`M ${ac.x} ${a.y} C ${ac.x} ${a.y - 26}, ${bc.x} ${b.y + b.h + 26}, ${bc.x} ${b.y + b.h}`,
                           at:{x:(ac.x + bc.x) / 2, y:(a.y + b.y + b.h) / 2}};
      case 'vup': return {d:`M ${ac.x} ${a.y} C ${ac.x} ${a.y - 30}, ${bc.x} ${b.y + b.h + 30}, ${bc.x} ${b.y + b.h}`,
                          at:{x:(ac.x + bc.x) / 2, y:(a.y + b.y + b.h) / 2}};
      case 'hback': return {d:`M ${a.x} ${ac.y} C ${a.x - 30} ${ac.y}, ${b.x + b.w + 30} ${bc.y}, ${b.x + b.w} ${bc.y}`,
                            at:{x:(a.x + b.x + b.w) / 2, y:(ac.y + bc.y) / 2 - 12}};
      case 'over': {
        const top = Math.min(a.y, b.y) - 26;
        return {d:`M ${ac.x} ${a.y} C ${ac.x} ${top}, ${bc.x} ${top}, ${bc.x} ${b.y}`, at:{x:(ac.x + bc.x) / 2, y:top + 15}};
      }
      case 'under': {
        const bottom = Math.max(a.y + a.h, b.y + b.h) + 30;
        return {d:`M ${ac.x} ${a.y + a.h} C ${ac.x} ${bottom}, ${bc.x} ${bottom}, ${bc.x} ${b.y + b.h}`,
                at:{x:(ac.x + bc.x) / 2, y:bottom + 4}};
      }
      default: return {d:`M ${a.x + a.w} ${ac.y} C ${a.x + a.w + 34} ${ac.y}, ${b.x - 34} ${bc.y}, ${b.x} ${bc.y}`,
                       at:{x:(a.x + a.w + b.x) / 2, y:(ac.y + bc.y) / 2 - 11}};
    }
  }
  function archDetail(view) {
    const node = archBox(view, T.archNode) || view.nodes[0];
    return `<div class="tr-panel-kicker">${esc(archKinds[node.kind] || '节点')}</div><h3>${esc(node.title)}</h3>
      ${node.badge ? `<span class="tr-arch-badge">${esc(node.badge)}</span>` : ''}
      <ul class="tr-arch-facts">${node.lines.map(l => `<li>${esc(l)}</li>`).join('')}</ul>
      ${node.detail.map(d => `<p>${esc(d)}</p>`).join('')}`;
  }
  function renderArchStage() {
    const view = archViews.find(v => v.id === T.archView) || archViews[0];
    const stage = page.querySelector('#tr-arch-stage'), detail = page.querySelector('#tr-arch-detail');
    if (!stage) return;
    const paths = view.links.map(link => ({link, path:archPath(view, link)})).filter(row => row.path);
    stage.style.setProperty('--arch-w', view.width);
    stage.style.setProperty('--arch-h', view.height);
    stage.innerHTML = `<svg class="tr-arch-links" viewBox="0 0 ${view.width} ${view.height}" aria-hidden="true">
        <defs><marker id="arch-tip" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
          <path d="M0 0 L7 3.5 L0 7 Z" fill="#9db387"></path></marker></defs>
        ${paths.map(({link, path}) => `<path d="${path.d}" class="${link.dash ? 'dashed' : ''}" marker-end="url(#arch-tip)"></path>`).join('')}
      </svg>
      <div class="tr-arch-nodes">${view.nodes.map(n => `<button class="tr-arch-node ${n.kind} ${n.id === T.archNode ? 'selected' : ''}"
        style="left:${n.x}px;top:${n.y}px;width:${n.w}px;height:${n.h}px" data-arch-node="${esc(n.id)}"
        aria-pressed="${n.id === T.archNode}"><strong>${esc(n.title)}</strong>
        ${n.badge ? `<i class="tr-arch-flag">${esc(n.badge)}</i>` : ''}
        <span>${n.lines.map(l => esc(l)).join('<br>')}</span></button>`).join('')}</div>
      <svg class="tr-arch-labels" viewBox="0 0 ${view.width} ${view.height}" aria-hidden="true">${paths.filter(row => row.link.label)
        .map(({link, path}) => `<text x="${path.at.x}" y="${path.at.y}" class="tr-arch-edge-label">${esc(link.label)}</text>`).join('')}</svg>`;
    if (detail) detail.innerHTML = archDetail(view);
    archFit();
  }
  function archFit() {
    const frame = page.querySelector('#tr-arch-frame'), stage = page.querySelector('#tr-arch-stage'),
          sizer = page.querySelector('#tr-arch-sizer'), hint = page.querySelector('#tr-arch-hint');
    if (!frame || !stage || !sizer) return;
    const view = archViews.find(v => v.id === T.archView) || archViews[0];
    // 窄屏不再无限缩小：保底比例之下改为横向滚动，宁可拖动也不要缩成看不清的一团。
    const scale = Math.max(Math.min(1, (frame.clientWidth - 2) / view.width), .68);
    stage.style.transform = `scale(${scale})`;
    sizer.style.width = `${Math.ceil(view.width * scale)}px`;
    sizer.style.height = `${Math.ceil(view.height * scale)}px`;
    if (hint) hint.hidden = Math.ceil(view.width * scale) <= frame.clientWidth;
  }
  window.addEventListener('resize', () => { if (T.section === 'architecture') archFit(); });
  function renderArchitecture() {
    const view = archViews.find(v => v.id === T.archView) || archViews[0];
    if (!archBox(view, T.archNode)) T.archNode = view.nodes[0].id;
    page.innerHTML = top('architecture') + `<div class="tr-section-title tr-arch-title"><div><div class="eyebrow muted">SYSTEM ARCHITECTURE</div>
        <h2>调查是怎样被组织起来的</h2><p>三张图分别看：一次调查的整体运行结构、单个工作区内部的一步，以及四层记录之间的关系。点击任意方块查看它的职责与边界。</p></div></div>
      <div class="tr-arch-switch" role="tablist" aria-label="架构视图">${archViews.map(v =>
        `<button role="tab" aria-selected="${v.id === T.archView}" data-arch-view="${esc(v.id)}" class="${v.id === T.archView ? 'active' : ''}">${esc(v.name)}</button>`).join('')}</div>
      <p class="tr-arch-caption">${esc(view.caption)}</p>
      <div class="tr-arch-frame" id="tr-arch-frame"><div class="tr-arch-sizer" id="tr-arch-sizer"><div class="tr-arch-stage" id="tr-arch-stage"></div></div></div>
      <p class="tr-arch-hint" id="tr-arch-hint" hidden>横向拖动可以看到完整结构</p>
      <aside class="tr-arch-detail" id="tr-arch-detail" aria-live="polite"></aside>
      <div class="tr-arch-legend">${[['input','原始输入'],['view','上下文'],['gia','角色决策'],['tool','工具'],['harness','Harness 执行'],['report','产出记录'],['store','持久化'],['reserved','位置保留'],['render','2026-09 新增或更新']]
        .map(([k, n]) => `<span class="lg-${k}">${n}</span>`).join('')}</div>
      <div class="tr-footnote">图中标注的能力均以当前代码为准：HI 的动作表为 SEARCH / INSPECT / QUERY_MEMORY / READ_RECORD，INSPECT 实际注册 8 个处理器；visual_qa / audio_qa 已从主线移除。DETECT 作为第三个工具的位置保留，尚未接入检测器，动作表里还没有它。</div>`;
    renderArchStage();
  }
  function renderRoadmap(){
    page.innerHTML=top('roadmap')+`<div class="tr-section-title tr-plan-title"><div><div class="eyebrow muted">PROJECT PROGRESS</div><h2>阶段进展与交付内容</h2><p>项目划分为五个阶段：前三个阶段已完成，第四阶段进行中。展开任一阶段可查看该阶段的具体交付内容。</p></div></div>
      <div class="tr-plan"><div class="tr-plan-root"><div class="tr-panel-kicker">PROJECT GOAL</div><strong>面向异构核查任务的<br>自主调查 Agent</strong><p>覆盖视频、图像与网页混合的核查场景：模型自主拆解问题、决定检索方向并完成取证，最终给出的每一条结论均可沿证据链回溯至原始来源。</p><div class="tr-plan-tally">${[['done','已完成'],['doing','进行中'],['todo','下一阶段']].map(([k,l])=>`<span class="${k}"><b>${planCount(k)}</b>${l}</span>`).join('')}</div></div>
      <div class="tr-plan-branches">${roadmap.map(b=>`<details class="tr-plan-branch" ${b.id==='quality'||b.id==='training'?'open':''}><summary class="tr-plan-head"><span class="tr-panel-kicker">${esc(b.kicker)}</span><strong>${esc(b.title)}</strong><small>${esc(b.hint)}</small><i class="tr-plan-chevron" aria-hidden="true"></i></summary><ul class="tr-plan-leaves">${b.items.map(i=>`<li class="${planStates[i.state][1]}"><div class="tr-plan-leaf-top"><b>${esc(i.name)}</b><span class="tr-plan-tag ${planStates[i.state][1]}">${planStates[i.state][0]}</span></div><p>${esc(i.text)}</p></li>`).join('')}</ul></details>`).join('')}</div></div>
      <div class="tr-return"><span>⟲</span> 每完成一轮，核验结果将指出下一轮的改进方向，流程本身随之迭代。</div>
      <div class="tr-plan-links">${routeButton('#training/architecture','系统架构 →')}${routeButton('#training/flow','数据生成流程 →')}${routeButton('#training/rubric','质量核验体系 →')}${routeButton('#training/samples','真实样例库 →')}</div>
      <div class="tr-footnote">本页反映 2026 年 9 月的项目状态，随各阶段进展更新。</div>`;
  }
  function docLink(name,label){return `<a href="/api/training/documents/${name}" target="_blank" rel="noopener noreferrer">${label} ↗</a>`;}
  const verifyLayers=[
    {name:'程序检查',tag:'不用模型 · 结果确定',text:'核查所引证据编号是否存在、提交时是否可见、报告是否为空、案卷要求的证据是否被引用、来源排除名单是否被违反。结论中出现的数字若不在所引证据文本中，先标记为待查。',errors:['fabricated_evidence_id','empty_report','source_restriction_miss']},
    {name:'逐条证据锚定',tag:'核验模型 ≠ 生成模型',text:'每一条结论与其所引的每一份证据配对，单独调用一次，仅提供该句与证据全文或图片，判定支持 / 反驳 / 未找到，并须给出引语。引语须可由程序在原文中回查到，否则降为未找到。',errors:['unsupported_assertion','wrong_citation','attribution_drift','scope_overreach','misread_fine_text']},
    {name:'充分性',tag:'对着任务本身查',text:'以任务全文与案卷中的必需证据对照整份报告，判定要求是否已覆盖、当前是否已可作答，以及状态应为「完成」还是「未完成」。',errors:['omitted_required_fact','completed_without_finishing','incomplete_though_answerable']}
  ];
  const errorGroups=[
    {layer:'结论级',items:['unsupported_assertion','wrong_citation','fabricated_evidence_id','source_restriction_miss','attribution_drift','scope_overreach','misread_fine_text','sloppy_numeric','omitted_required_fact'],note:'结论与其所引证据之间的问题。典型表现为将「来源声称某事」写作「某事成立」，或将「批准拨付」写作「直接拨付」这类阶段互换，也是人工最难逐条查出的一类。'},
    {layer:'状态级',items:['completed_without_finishing','incomplete_though_answerable','empty_report'],note:'报告整体层面的问题：尚未查完即报完成、已可作答却报未完成、提交空报告。该类由程序层直接拦截，不依赖模型判断。'},
    {layer:'动作级',items:['illegal_action','redundant_action','evidence_dropped','wrong_tool','ignored_feedback'],note:'行动本身的问题。重复查看同一材料不必然是错误，关键在于该次复查是否针对一个具体的未决问题。'},
    {layer:'调度规划级',items:['task_not_gap_driven','premature_finish','unnecessary_round','dependent_sibling_tasks'],note:'分工与终止判断的问题。同一轮内相互依赖的任务不应并行派发；局部报告标注「完成」，亦不代表总问题已可结束。'},
    {layer:'环境级',items:['env_contamination','env_tool_failure'],note:'不计入模型问题，但须单独记录，否则易将网络或工具故障误判为模型能力不足。'}
  ];
  function renderRubric(){
    const d=dimensions.find(d=>d.id===T.dimension);
    page.innerHTML=top('rubric')+`<div class="tr-section-title tr-rubric-title"><div><div class="eyebrow muted">QUALITY AT EVERY DECISION</div><h2>每一步输出的质量判定</h2><p>每次输出单独判定，报告准确性、任务完成度与行动有效性分别核查。</p></div>${pill('评分标准 r6')}</div><p class="tr-upgrade"><b>两道关口</b>先按下列七个维度对每一步单独评分，定位问题所在环节；再由页面下半部分的三层核验器回到证据原文逐条核对。两道均通过，该步输出方可成为模型的学习对象。</p><div class="tr-role-switch" aria-label="评分角色">${['HI','GIA'].map(r=>`<button data-tr-role="${r}" class="${T.role===r?'active':''}" aria-pressed="${T.role===r}"><b>${r}</b><span>${r==='HI'?'局部调查员':'全局调查组长'}</span></button>`).join('')}<p>${T.role==='HI'?'取得证据 → 读懂材料 → 准确报告':'发现缺口 → 安排分工 → 综合与停止'}</p></div><div class="tr-rubric-layout"><div class="tr-dimension-list" aria-label="评分维度">${dimensions.map(v=>`<button class="tr-dimension ${d.id===v.id?'active':''}" data-tr-dimension="${v.id}" aria-pressed="${d.id===v.id}"><span>${v.name}<small>${v.id}</small></span><b>${v.weight}<small>%</small></b><i style="--weight:${v.weight*4}%"></i></button>`).join('')}</div><article class="tr-rubric-detail" aria-live="polite"><div class="tr-panel-kicker">${T.role} · ${d.name} · 权重 ${d.weight}%</div><h3>${d.question}</h3><p class="tr-rubric-definition">${d[T.role.toLowerCase()]}</p><div class="tr-callout">${d.detail}</div><div class="tr-score-scale">${[['0','明确错误'],['1','主要要求未满足'],['2','存在实质缺陷'],['3','仅有小缺陷'],['4','充分满足']].map(([n,l])=>`<div class="score-${n}"><b>${n}</b><span>${l}</span></div>`).join('')}</div><div class="tr-unknowns"><div><strong>未知 unknown</strong><p>缺少评审依据，不能按通过处理。</p></div><div><strong>不适用 N/A</strong><p>该维度不适合本次输出，不强行打分。</p></div></div>${routeButton(`#training/samples/${d.example}`,'查看对应的真实样例 →')}</article></div><section class="tr-admission"><div class="tr-section-title"><div><h2>四类处置结果</h2><p>总分用于概括，收录依据为每个目标的具体判定与检查结果。</p></div></div><div class="tr-dispositions">${[['accept_candidate','候选收录','当前状态下合理，重要事实有依据。'],['repair_reexecute','重新执行','有实质缺陷，从对应状态修复并复评。'],['context_only','仅作历史','真实发生，但不作为正向模仿目标。'],['exclude','排除目标','存在明确不可执行、捏造或信息越界。']].map(([id,n,p])=>`<div class="tr-disposition ${id}"><span class="tr-disposition-dot"></span><h3>${n}</h3><code>${id}</code><p>${p}</p></div>`).join('')}</div><div class="tr-admission-rule"><strong>进入 SFT 候选</strong><span>判为可收录</span><i>＋</i><span>3 或 4 分</span><i>＋</i><span>无关键评审未知</span><i>＋</i><span>执行、输入与附件对应检查通过</span></div></section><section class="tr-verify"><div class="tr-section-title"><div><div class="eyebrow muted">EVIDENCE-ANCHORED VERIFICATION</div><h2>每一条结论，<br>均回溯至证据原文逐条核对。</h2><p>三层依次执行，前一层已判定的不再重复判定。核验结果归入统一的问题类型，质量因此可按类型统计与跟踪。</p></div></div><div class="tr-verify-layers">${verifyLayers.map((v,i)=>`<article><div class="tr-verify-head"><span class="tr-verify-index">${String(i+1).padStart(2,'0')}</span><div><strong>${esc(v.name)}</strong><small>${esc(v.tag)}</small></div></div><p>${esc(v.text)}</p><div class="tr-verify-errors">${v.errors.map(e=>`<code>${esc(e)}</code>`).join('')}</div></article>`).join('')}</div><div class="tr-callout">第二层要求核验模型给出引语，再由程序回到证据原文核查该引语是否存在，未查到即降级为「找不到依据」。核验所用模型与生成数据的模型相互独立，这是「结论经过实际核验」与「模型自评」之间的分界。</div></section>
      <section class="tr-errors"><div class="tr-section-title"><div><h2>核验覆盖的问题类型</h2><p>类型统一定义于一处，每一次流程改进均须说明其拟消除的类型，使改进可归因。</p></div><span>带角色 · 严重度 · 是否影响后续结论</span></div><div class="tr-error-groups">${errorGroups.map(g=>`<div><h3>${esc(g.layer)}<span>${g.items.length}</span></h3><div>${g.items.map(e=>`<code>${esc(e)}</code>`).join('')}</div><p>${esc(g.note)}</p></div>`).join('')}</div></section><div class="tr-footnote">加权总分按已评分的适用维度归一化并报告覆盖率。历史样本保留它当时实际使用的评分版本；新一轮起统一由三层核验器重核，以核验结果为准。</div><div class="tr-doc-links">${docLink('rubric','阅读完整 r6 标准')}</div>`;
  }
  function sampleCard(s){return `<article class="tr-sample-card"><a class="tr-sample-visual ${s.kind}" href="#training/samples/${s.id}" data-training-route="#training/samples/${s.id}" aria-label="打开 ${esc(s.title)}">${s.thumbnail?`<img src="${s.thumbnail}" alt="${esc(s.dataset)} ${esc(s.case)} 实际输入图片" loading="lazy">`:`<div class="tr-document-art"><span>原始来源</span><b>${s.kind==='web'?'SOURCE':'EVIDENCE'}</b><i></i><i></i><i></i></div>`}<span class="tr-dataset">${esc(s.dataset)} · ${esc(s.case)}</span>${s.kind==='contrast'?'<span class="tr-contrast-mark">同证据 × 4 个问题</span>':''}</a><div class="tr-sample-body"><div class="tr-card-mode">${esc(modes[s.mode])}</div><h3>${routeButton(`#training/samples/${s.id}`,esc(s.title))}</h3><p>${esc(s.description)}</p><div class="tr-tags">${s.tags.map(t=>pill(t)).join('')}</div><div class="tr-card-count"><span><b>${s.step_count}</b> ${s.mode==='gia_offline'?'次决定':'次输出'}</span><span><i class="tr-accept-dot"></i><b>${s.included_count}</b> 条已收录</span>${s.excluded_count?`<span class="tr-amber">${s.excluded_count} 条未收录</span>`:''}</div><div class="tr-card-footer"><small>${s.usage_split==='development'?'开发与训练准备':'构造方法保留检查'} · ${esc(s.rubric_version)}</small>${routeButton(`#training/samples/${s.id}`,'查看步骤 ↗')}</div></div></article>`;}
  function filteredSamples(){return T.data.samples.filter(s=>(T.filter==='all'||(T.filter==='GIA'||T.filter==='HI'?s.roles.includes(T.filter):s.mode===T.filter))&&(T.dataset==='all'||s.dataset===T.dataset)&&`${s.title} ${s.description} ${s.dataset} ${s.case} ${s.tags.join(' ')}`.toLowerCase().includes(T.query.toLowerCase()));}
  function renderSampleGrid(){const samples=filteredSamples();page.querySelector('#tr-sample-count').textContent=`${samples.length} 组样例`;page.querySelector('#tr-sample-grid').innerHTML=samples.map(sampleCard).join('')||'<div class="tr-empty"><strong>没有匹配的样例</strong><p>试试其他问题关键词，或清除筛选。</p><button class="secondary" data-tr="clear">清除筛选</button></div>';}
  function renderSamples(){page.innerHTML=top('samples')+`<div class="tr-section-title tr-samples-title"><div><div class="eyebrow muted">REAL EXAMPLES</div><h2>真实记录中的完整样例</h2><p>覆盖多类任务与两个角色，每一步均可回到当时的输入、输出与评分依据。</p></div></div><div class="tr-library-note"><strong>${format(T.data.stats.total)} 条</strong>全量候选决策<span>精选 ${T.data.samples.length} 组实际样例 · 保留未收录的真实步骤</span></div><div class="tr-search-row"><label class="tr-search"><span aria-hidden="true">⌕</span><input id="tr-search" type="search" placeholder="搜索问题、数据集或案例编号…" value="${esc(T.query)}" aria-label="搜索训练样例"></label><select id="tr-dataset" aria-label="按数据集筛选"><option value="all">全部数据来源</option>${[...new Set(T.data.samples.map(s=>s.dataset))].map(d=>`<option ${T.dataset===d?'selected':''}>${esc(d)}</option>`).join('')}</select></div><div class="tr-filter-row"><div class="tr-filters">${[['all','全部'],['GIA','包含 GIA'],['HI','包含 HI'],['bridge','真实协作'],['gia_offline','离线决策']].map(([id,n])=>`<button data-tr-filter="${id}" class="${T.filter===id?'active':''}" aria-pressed="${T.filter===id}">${n}</button>`).join('')}</div><span id="tr-sample-count" aria-live="polite"></span></div><div class="tr-sample-grid" id="tr-sample-grid"></div><p class="tr-footnote">「已收录」表示该输出通过逐步评分与核验，进入训练候选；同一案例的多次运行按一个案例计。</p>`;renderSampleGrid();}
  function visibleSteps(){return T.sample.steps.filter(s=>T.stepFilter==='all'||T.stepFilter==='excluded'&&!s.included||s.role===T.stepFilter);}
  function stepDot(s){return s.included?'accepted':s.disposition==='repair_reexecute'?'repair':'excluded';}
  function condition(sample){const c=sample.source_condition;if(sample.mode==='gia_offline')return '离线决策状态：依据真实调查报告构造问题，本组补查提议未在这些状态中执行。';if(c==='provided_source_url')return '任务已提供来源地址；实际执行打开与阅读，不计为自主发现来源。';if(c==='recorded_plan_execution_with_provided_evidence')return '从已有计划和已取得材料启动真实协作；首项记录计划不重复计为新模型输出。';if(sample.kind==='video')return '给定视频材料下的调查；片段对应不自动证明最早来源或原创归属。';return '依据实际提供的网页与媒体条件取证；具体来源限制以本次任务输入为准。';}
  function renderSampleShell(){
    const s=T.sample;
    T.map?.destroy();
    page.classList.add('tr-detail-page');
    page.innerHTML=top('samples')+`<div class="tr-sample-heading">${routeButton('#training/samples','← 返回样本库','tr-back')}<div class="tr-sample-title"><div><div class="tr-tags">${pill(s.dataset)}${pill(s.mode==='gia_offline'?'离线决策':'真实执行')}${pill('Rubric '+s.rubric_version)}</div><h2>${esc(s.title)}</h2><p>${esc(s.description)}</p></div><div class="tr-sample-totals"><strong>${s.included_count}<span> / ${s.step_count}</span></strong><small>条输出进入候选</small></div></div><div class="tr-condition">${condition(s)}</div></div>
      <div class="tr-reader-tabs" aria-label="样例阅读视图"><button data-tr-pane="map" aria-pressed="true" class="active">调查脑图</button><button data-tr-pane="detail" aria-pressed="false">节点详情</button></div>
      <div class="tr-record-layout" data-pane="map"><section class="tr-map-panel" id="tr-map-panel" aria-label="真实样例脑图"></section><section class="tr-step-main" id="tr-step-main" aria-live="polite"></section></div>`;
    T.map=new TrainingMindMap(s,{titleForStep:actionTitle,onStep:id=>selectStep(id),onGroup:node=>selectGroup(node),onFilter:setStepFilter});
    T.map.filter=T.stepFilter;
    T.map.mount(page.querySelector('#tr-map-panel'));
    setPane(T.mobilePane);
  }
  function setPane(pane){
    T.mobilePane=pane;
    const layout=page.querySelector('.tr-record-layout');if(!layout)return;
    layout.dataset.pane=pane;
    page.querySelectorAll('[data-tr-pane]').forEach(b=>{const active=b.dataset.trPane===pane;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});
    if(pane==='map')requestAnimationFrame(()=>{if(T.map?.selected==='root')T.map.fit();else T.map?.focus(T.map.selected,false);});
  }
  function breadcrumbs(id){
    const nodes=[];let node=T.map?.nodes.get(id);
    while(node){nodes.unshift(node);node=T.map.nodes.get(node.parent);}
    return `<nav class="tr-map-breadcrumbs" aria-label="当前节点位置">${nodes.map((n,i)=>`${i?'<span>›</span>':''}<button data-tr-node="${esc(n.id)}" ${n.id===id?'aria-current="location"':''}>${esc(n.id==='root'?'调查问题':n.kind==='round'?n.label.split(' · ')[0]:n.kind==='hi'?n.label:n.step?`${n.step.actor} #${Number(n.step.id.split('_')[1])}`:n.label)}</button>`).join('')}</nav>`;
  }
  function selectGroup(node,{push=true}={}){
    stop();++T.serial;T.step=null;
    if(push)navigate(`#training/samples/${T.sample.id}`);
    renderGroup(node);
  }
  function renderGroup(node){
    const matching=node.steps.filter(s=>T.map.matches(s));
    const children=node.children.map(id=>T.map.nodes.get(id)).filter(n=>n.steps.some(s=>T.map.matches(s)));
    const main=page.querySelector('#tr-step-main');if(!main)return;
    main.innerHTML=breadcrumbs(node.id)+`<div class="tr-step-heading"><div><span class="tr-panel-kicker">${esc(node.badge)}</span><h3>${esc(node.label)}</h3></div>${pill(node.steps.length+' 次输出')}</div><div class="tr-step-content tr-group-content"><div class="tr-task-box"><span>${node.id==='root'?(T.sample.mode==='gia_offline'?'共同证据':'总问题'):node.kind==='hi'?'局部任务':'本轮调查'}</span><p>${text(node.subtitle)}</p></div>
      <div class="tr-branch-stats"><div><b>${matching.length}</b><span>当前输出</span></div><div><b>${matching.filter(s=>s.included).length}</b><span>已收录</span></div><div><b>${matching.filter(s=>!s.included).length}</b><span>未收录</span></div></div>
      ${node.status?`<div class="tr-callout">${esc(node.status)} · ${node.steps.filter(s=>s.action!=='REPORT').length} 次行动</div>`:''}
      <div class="tr-branch-heading"><h3>${node.id==='root'?(T.sample.mode==='gia_offline'?'四个独立问题':T.sample.mode==='hi_independent'?'局部调查':'调查轮次'):node.kind==='hi'?'行动与报告':'分工与决策'}</h3><span>点击展开查看</span></div>
      <div class="tr-branch-list">${children.map(n=>`<button data-tr-node="${esc(n.id)}"><div><span class="tr-branch-kind ${n.kind}">${esc(n.step?n.step.role:n.kind==='hi'?'HI':n.kind==='round'?'轮次':'调查')}</span><strong>${esc(n.label)}</strong><i>↗</i></div><p>${esc(n.subtitle)}</p><small>${n.step?`${n.step.included?'✓ 已收录':'○ 未收录'}${n.step.rating==null?'':' · '+n.step.rating+' / 4 分'}`:`${n.steps.filter(s=>T.map.matches(s)).length} 次输出${n.status?' · '+n.status:''}`}</small></button>`).join('')||'<div class="tr-empty"><strong>当前分类没有输出</strong><p>切换筛选，查看其他角色或全部记录。</p></div>'}</div></div>`;
  }
  function setStepFilter(value){
    stop();T.stepFilter=value;T.map.setFilter(value);
    if(T.step&&T.map.matches(T.step)){
      T.map.selectStep(T.step.id);if(T.step.output)renderStep();
    }else selectGroup(T.map.nodes.get(T.map.selected));
  }
  function recordList(value,title){const records=Array.isArray(value)?value:Object.values(value||{});if(!records.length)return '';return `<section class="tr-detail-section"><h3>${title}<span>${records.length}</span></h3>${records.map(r=>typeof r==='object'&&r!==null?`<details class="tr-record"><summary>${pill(r.evidence_id||r.task_id||r.report_id||r.record_id||'记录')}<span>${esc(r.label||r.task||r.statement||r.record_type||'展开记录')}</span></summary><div class="tr-record-content">${r.content?`<div class="tr-prose">${text(r.content)}</div>`:''}${r.source_ref&&urlSafe(r.source_ref)?sourceLink(r.source_ref,'打开来源 ↗'):''}${raw('完整记录',r)}</div></details>`:`<div class="tr-prose">${text(r)}</div>`).join('')}</section>`;}
  function imagesBlock(images){if(!images.length)return '<p class="tr-no-images">本轮没有图片附件，依据文字与已有报告作出决定。</p>';return `<div class="tr-detail-section"><h3>本次实际图片<span>${images.length} 张</span></h3><div class="tr-image-grid">${images.map((im,i)=>{const loc=im.note.locator||{},time=loc.time_s;return `<button class="tr-image" data-tr-image="${i}" aria-label="放大 ${esc(im.evidence_id)} 原始输入图片"><div><img src="${im.url}" alt="${esc(im.evidence_id)} · ${time==null?'实际输入材料':Number(time).toFixed(3)+' 秒'}" loading="lazy"></div><span><b>${esc(im.evidence_id)}</b><small>${time==null?(im.note.derived_from||[]).join(' · '):`${Number(time).toFixed(3)} 秒 · ${(im.note.derived_from||[]).join(' / ')}`}</small><i>⤢</i></span></button>`;}).join('')}</div></div>`;}
  function outputBlock(o){if(o.answer)return `<div class="tr-output-answer tr-prose">${text(o.answer)}</div>`;if(o.tasks)return `<div class="tr-task-list">${o.tasks.map((v,i)=>`<article><span class="tr-task-index">${String(i+1).padStart(2,'0')}</span><div><h4>局部调查 ${i+1}</h4><p class="tr-prose">${text(v.task)}</p><div class="tr-tags">${(v.materials||[]).map(x=>pill(x)).join('')}</div></div></article>`).join('')}</div>`;if(o.findings)return `${o.status?`<div class="tr-report-status">${pill(o.status==='COMPLETED'?'局部任务完成':'局部任务未完成',o.status==='COMPLETED'?'good':'amber')}<p>${text(o.status_reason||'')}</p></div>`:''}${o.findings.map(f=>`<article class="tr-finding"><div class="tr-prose">${text(f.statement)}</div><div class="tr-tags">${(f.evidence_ids||[]).map(id=>pill(id)).join('')}</div>${f.limitations?.length?`<ul>${f.limitations.map(l=>`<li>${text(l)}</li>`).join('')}</ul>`:''}</article>`).join('')}`;const a=o.arguments||{};return `<div class="tr-action-card"><div>${pill(o.action,T.step.role==='GIA'?'gia':'hi')}<strong>${esc(actionTitle(T.step))}</strong></div>${Object.entries(a).map(([k,v])=>`<div class="tr-action-param"><span>${({query:'搜索内容',source:'来源地址',targets:'目标材料',record_id:'读取记录',request:'操作与参数',offset:'读取起点',record_types:'记录类型'})[k]||esc(k)}</span><div>${typeof v==='string'?(urlSafe(v)?sourceLink(v):`<p class="tr-prose">${text(v)}</p>`):json(v)}</div></div>`).join('')}${o.focus_evidence_ids?.length?`<div class="tr-focus-evidence"><span>保留到后续输入</span>${o.focus_evidence_ids.map(id=>pill(id)).join('')}</div>`:''}</div>`;}
  function ratingBlock(s){return `<div class="tr-verdict ${s.included?'accepted':'not-included'}"><div><span>${s.included?'✓ 候选已收录':'○ 未作为正向目标收录'}</span><b>${s.rating==null?'—':s.rating}<small> / 4</small></b></div><p>${text(s.reason)}</p><small>${esc(dispositions[s.disposition]||s.disposition)}${s.critical_unknown?' · 存在关键评审未知':''}${s.disposition==='accept_candidate'&&!s.included?' · 以最终候选包收录状态为准':''}</small></div>`;}
  function workspaceLinks(){
    const node=T.map.nodes.get(T.step.id),children=(node?.children||[]).map(id=>T.map.nodes.get(id));
    if(!children.length)return '';
    return `<section class="tr-detail-section"><h3>接到分工的 HI</h3><div class="tr-branch-list">${children.map(n=>`<button data-tr-node="${esc(n.id)}"><div><span class="tr-branch-kind hi">HI</span><strong>${esc(n.label)}</strong><i>↗</i></div><p>${esc(n.subtitle)}</p><small>${n.steps.length} 次真实输出 · ${esc(n.status)}</small></button>`).join('')}</div></section>`;
  }
  function overviewContent(){const s=T.step;return `<div class="tr-task-box"><span>${s.role==='GIA'?'当前总问题':'当前局部任务'}</span><p>${text(s.task)}</p></div><section class="tr-detail-section"><h3>模型这一步输出了什么</h3>${outputBlock(s.output)}</section>${workspaceLinks()}${ratingBlock(s)}${imagesBlock(s.images)}${raw('查看完整原始输出',s.output)}`;}
  function inputContent(){const s=T.step,v=s.payload.model_view||s.payload.global_view||{};return `<div class="tr-context-label">决策发生前的实际输入 · ${s.images.length} 张图片附件</div><div class="tr-task-box"><span>当前任务</span><p>${text(s.task)}</p></div>${imagesBlock(s.images)}${recordList(v.latest_reports||v.reports,'已有局部报告')}${recordList(v.materials,'任务材料')}${recordList(v.evidence_view||v.supporting_records,'当前证据记录')}${v.latest_observation?raw('上一步工具观察',v.latest_observation):''}${raw('完整决策前状态',s.payload)}<details class="tr-raw"><summary>角色提示与图片说明</summary><pre class="tr-code">${esc(s.system_prompt)}\n\n${esc(s.image_notes.join('\n'))}</pre></details>`;}
  function observationContent(){const o=T.step.observation;if(!o)return `<div class="tr-empty"><strong>${T.sample.mode==='gia_offline'?'本条是离线决定，尚无实际工具返回':T.step.action==='REPORT'?'本步提交了局部报告':T.step.action==='FINISH'?'本步提交了最终回答':'本步没有独立工具观察'}</strong><p>${T.sample.mode==='gia_offline'?'拟议调查任务保留在模型输出中，不把计划视为已经执行。':'可在本步概览查看提交内容与质量判断。'}</p><button class="secondary" data-tr-tab="overview">查看模型输出</button></div>`;return `<div class="tr-context-label">行动发生后的实际返回 · 不属于本步决策前输入</div>${o.status?pill(o.status==='failed'?'执行失败':o.status,o.status==='failed'?'amber':'good'):''}<div class="tr-observation-items">${(o.items||[]).map(i=>`<p class="tr-prose">${text(i)}</p>`).join('')}</div>${o.details?.content?`<div class="tr-task-box"><span>${esc(o.details.label||'工具返回正文')}</span><div class="tr-prose">${text(o.details.content)}</div></div>`:''}${raw('完整工具返回',o)}`;}
  function reviewContent(){const s=T.step;return `${ratingBlock(s)}<div class="tr-review-heading"><h3>${s.assessment_scope==='state'?'本条决策的维度评分':'该角色 / 工作区的整体维度评分'}</h3><span>Rubric ${esc(s.rubric_version)}</span></div><p class="tr-footnote">${s.assessment_scope==='state'?'这组离线决定分别评审；未执行的工具效果不计为已验证。':'以下评分概括该工作区的完整过程；本条输出的收录以顶部逐目标判断为准。'}</p><div class="tr-review-dimensions">${s.dimensions.map(d=>`<details><summary><span>${esc(dimensions.find(v=>v.id===d.id)?.name||d.id)}</span><div class="tr-rating-dots">${[0,1,2,3].map(n=>`<i class="${Number.isInteger(d.rating)&&n<d.rating?'filled':''}"></i>`).join('')}</div><b>${d.rating==null?(d.status==='not_applicable'?'N/A':'未知'):d.rating+' / 4'}</b></summary><p>${text(d.reason)}</p>${d.request_ids?`<small>关联 ${d.request_ids.map(id=>'#'+Number(id.split('_')[1])).join('、')}</small>`:''}</details>`).join('')}</div>${raw('完整逐目标评审',s.assessment)}<div class="tr-doc-links">${docLink('rubric','当前完整 r6 标准')}</div>`;}
  function sftContent(){const s=T.step;return `<div class="tr-context-label">${s.included?'当前候选包中的监督目标':'本输出未进入当前候选包'}</div><div class="tr-sft-stack"><div><div><b>system</b><span>角色提示 · 作为上下文</span></div><details><summary>展开实际角色提示</summary><pre class="tr-code">${esc(s.system_prompt)}</pre></details></div><div><div><b>user</b><span>当时状态 + ${s.images.length} 张实际图片 · 作为上下文</span></div>${raw('展开实际输入', {payload:s.payload,image_notes:s.image_notes})}</div><div class="${s.included?'tr-supervised':'tr-unsupervised'}"><div><b>assistant</b><span>${s.included?'仅此输出计算监督损失':'不作为正向监督目标'}</span></div>${json(s.output)}</div></div><div class="tr-callout">${s.included?'只学习当前角色的这一次输出。另一角色的报告、工具原文和此前历史帮助模型理解当前状态。':'这一步仍是实际发生的历史，可以成为后续决策的上下文；当前输出本身不作为正确示范。'}</div>${s.candidate?raw('候选元数据',s.candidate):''}`;}
  function renderStep(){const s=T.step;if(!s)return;const index=T.sample.steps.findIndex(v=>v.id===s.id),sequence=visibleSteps(),position=sequence.findIndex(v=>v.id===s.id);const main=page.querySelector('#tr-step-main');if(!main)return;main.innerHTML=breadcrumbs(s.id)+`<div class="tr-step-heading"><div><span class="tr-panel-kicker">${esc(s.actor)} · ${s.id.startsWith('state')?'问题':'请求'} ${Number(s.id.split('_')[1])}</span><h3>${esc(actionTitle(s))}</h3></div>${pill(s.role,s.role.toLowerCase())}</div><div class="tr-step-controls"><span>${index+1} / ${T.sample.steps.length}</span><div><button class="tr-control" data-tr="previous" ${position<=0?'disabled':''} aria-label="上一个决策">← 上一步</button><button class="tr-control" data-tr="play" aria-pressed="${T.playing}">${T.playing?'Ⅱ 暂停回放':'▷ 自动回放'}</button><button class="tr-control" data-tr="next" ${position===sequence.length-1?'disabled':''}>下一步 →</button></div></div><div class="tr-detail-tabs" role="tablist" aria-label="本步详细记录">${[['overview','本步概览'],['input','当时输入'],['observation','工具返回'],['review','质量评分'],['sft','SFT 样本']].map(([id,n])=>`<button role="tab" id="tr-tab-${id}" aria-controls="tr-step-content" aria-selected="${T.tab===id}" tabindex="${T.tab===id?0:-1}" data-tr-tab="${id}" class="${T.tab===id?'active':''}">${n}</button>`).join('')}</div><div class="tr-step-content" id="tr-step-content" role="tabpanel" aria-labelledby="tr-tab-${T.tab}">${T.tab==='input'?inputContent():T.tab==='observation'?observationContent():T.tab==='review'?reviewContent():T.tab==='sft'?sftContent():overviewContent()}</div>`;}
  async function selectStep(id,{push=true,keepPlaying=false}={}){if(!keepPlaying)stop();const serial=++T.serial,sampleId=T.sample.id;const summary=T.sample.steps.find(v=>v.id===id);if(!summary)return;T.step={...summary};T.map?.selectStep(id);setPane('detail');const main=page.querySelector('#tr-step-main');if(main)main.innerHTML='<div class="tr-loading" role="status"><span class="tr-loader"></span>正在读取本步输入与评分…</div>';if(push)navigate(`#training/samples/${sampleId}/${id}`);try{const detail=await get(`/api/training/samples/${sampleId}/steps/${id}`);if(T.serial!==serial||T.sample?.id!==sampleId)return;T.step=detail;renderStep();}catch(e){if(serial===T.serial&&main)main.innerHTML=`<div class="tr-empty"><strong>暂时无法读取这一步</strong><p>${esc(e.message)}</p><button class="secondary" data-tr-step="${id}">重试</button></div>`;stop();}}
  async function open(hash='#training'){
    stop();T.map?.destroy();page.classList.remove('tr-detail-page');clearTimeout(S.poll);stopPlayback();const serial=++T.serial;showPage('training-page');navigate(hash);
    if(!T.data){loading();try{T.data=await get('/api/training');}catch(e){if(serial===T.serial)page.innerHTML=`<div class="tr-empty"><strong>训练记录暂时不可用</strong><p>${esc(e.message)}</p><button class="secondary" data-tr="retry">重新读取</button></div>`;return;}}
    if(serial!==T.serial)return;
    const parts=hash.split('/');
    if(parts[1]==='samples'&&parts[2]){
      try{
        const same=T.sample?.id===parts[2];if(!same){T.stepFilter='all';T.tab='overview';loading();}
        const sample=await get(`/api/training/samples/${encodeURIComponent(parts[2])}`);if(serial!==T.serial)return;
        T.sample=sample;T.section='detail';T.step=null;T.mobilePane=parts[3]?'detail':'map';
        const target=sample.steps.find(s=>s.id===parts[3]);
        if(target&&!visibleSteps().some(s=>s.id===target.id))T.stepFilter='all';
        renderSampleShell();
        if(target)await selectStep(target.id,{push:false});else selectGroup(T.map.nodes.get('root'),{push:false});
        if(!same)window.scrollTo({top:0});
      }catch(e){page.innerHTML=top('samples')+`<div class="tr-empty"><strong>无法打开这个样例</strong><p>${esc(e.message)}</p>${routeButton('#training/samples','返回样本库')}</div>`;}
      return;
    }
    T.section=['rubric','samples','flow','architecture'].includes(parts[1])?parts[1]:'roadmap';
    if(T.section==='rubric')renderRubric();else if(T.section==='samples')renderSamples();else if(T.section==='flow')renderFlow();
    else if(T.section==='architecture')renderArchitecture();else renderRoadmap();window.scrollTo({top:0});
  }
  window.openTraining=open;
  function showImage(index){const images=T.step?.images||[],im=images[index];if(!im)return;stop();let dialog=document.querySelector('#tr-image-dialog');if(!dialog){dialog=document.createElement('dialog');dialog.id='tr-image-dialog';dialog.className='tr-image-dialog';document.body.append(dialog);}dialog.innerHTML=`<div class="tr-image-modal-heading"><span>${esc(im.evidence_id)} · 本轮实际输入图片 · ${index+1} / ${images.length}</span><button data-tr="close-image" aria-label="关闭图片">×</button></div><div class="tr-image-canvas"><img src="${im.url}" alt="${esc(im.evidence_id)} 原始输入图片"></div><div class="tr-image-modal-footer"><button class="tr-control" data-tr-image="${index-1}" ${index===0?'disabled':''}>← 上一张</button><span>${esc((im.note.derived_from||[]).join(' / '))}${im.note.locator?.time_s!=null?' · '+Number(im.note.locator.time_s).toFixed(6)+' 秒':''}</span><button class="tr-control" data-tr-image="${index+1}" ${index===images.length-1?'disabled':''}>下一张 →</button></div>`;dialog.dataset.index=String(index);if(!dialog.open)dialog.showModal();}
  async function advance(direction,playing=false){const sequence=visibleSteps(),index=sequence.findIndex(s=>s.id===T.step?.id),step=sequence[index+direction];if(!step){stop();return;}await selectStep(step.id,{keepPlaying:playing});}
  function play(){T.playing=true;const b=page.querySelector('[data-tr="play"]');if(b){b.textContent='Ⅱ 暂停回放';b.setAttribute('aria-pressed','true');}async function tick(){if(!T.playing)return;await advance(1,true);if(T.playing)T.timer=setTimeout(tick,2600);}T.timer=setTimeout(tick,2600);}
  document.addEventListener('click',async e=>{
    const b=e.target.closest('button,a');if(!b)return;
    if(b.dataset.trainingRoute){e.preventDefault();await open(b.dataset.trainingRoute);return;}
    if(b.dataset.archView){T.archView=b.dataset.archView;T.archNode=null;renderArchitecture();page.querySelector(`[data-arch-view="${T.archView}"]`)?.focus({preventScroll:true});return;}
    if(b.dataset.archNode){T.archNode=b.dataset.archNode;renderArchStage();page.querySelector(`[data-arch-node="${T.archNode}"]`)?.focus({preventScroll:true});return;}
    if(b.dataset.trStage!==undefined){T.stage=Number(b.dataset.trStage);renderFlow();page.querySelector(`[data-tr-stage="${T.stage}"]`)?.focus({preventScroll:true});return;}
    if(b.dataset.trRole){T.role=b.dataset.trRole;renderRubric();page.querySelector(`[data-tr-role="${T.role}"]`)?.focus({preventScroll:true});return;}
    if(b.dataset.trDimension){T.dimension=b.dataset.trDimension;renderRubric();page.querySelector(`[data-tr-dimension="${T.dimension}"]`)?.focus({preventScroll:true});return;}
    if(b.dataset.trFilter){T.filter=b.dataset.trFilter;page.querySelectorAll('[data-tr-filter]').forEach(v=>{v.classList.toggle('active',v===b);v.setAttribute('aria-pressed',String(v===b));});renderSampleGrid();return;}
    if(b.dataset.trPane){setPane(b.dataset.trPane);return;}
    if(b.dataset.trNode){T.map.select(b.dataset.trNode);return;}
    if(b.dataset.trStep){await selectStep(b.dataset.trStep);return;}
    if(b.dataset.trTab){stop();T.tab=b.dataset.trTab;renderStep();page.querySelector(`#tr-tab-${T.tab}`)?.focus({preventScroll:true});return;}
    if(b.dataset.trImage!==undefined){showImage(Number(b.dataset.trImage));return;}
    switch(b.dataset.tr){case 'retry':T.data=null;T.cache.delete('/api/training');open(location.hash);break;case 'clear':T.filter='all';T.query='';T.dataset='all';renderSamples();break;case 'previous':await advance(-1);break;case 'next':await advance(1);break;case 'play':if(T.playing)stop();else play();break;case 'close-image':document.querySelector('#tr-image-dialog')?.close();break;}
  });
  document.addEventListener('input',e=>{if(e.target.id==='tr-search'){T.query=e.target.value;renderSampleGrid();}});
  document.addEventListener('change',e=>{if(e.target.id==='tr-dataset'){T.dataset=e.target.value;renderSampleGrid();}});
  document.addEventListener('keydown',e=>{const tab=e.target.closest('.tr-detail-tabs [data-tr-tab]');if(tab&&['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();const tabs=[...page.querySelectorAll('.tr-detail-tabs [data-tr-tab]')];let index=tabs.indexOf(tab);index=e.key==='Home'?0:e.key==='End'?tabs.length-1:(index+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;tabs[index].click();}const dialog=document.querySelector('#tr-image-dialog');if(dialog?.open&&['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();showImage(Number(dialog.dataset.index)+(e.key==='ArrowRight'?1:-1));}});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
})();
