"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const iconPaths = {grid:'<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',history:'<path d="M3 11a9 9 0 1 1 2.6 7M3 4v7h7M12 7v5l3 2"/>',upload:'<path d="M12 16V3m-5 5 5-5 5 5M4 15v5h16v-5"/>'};
$$('[data-icon]').forEach(e => e.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">${iconPaths[e.dataset.icon] || ''}</svg>`);
const S = {cases:[], history:[], status:null, run:null, events:new Map(), nodes:new Map(), expanded:new Set(), showEvidence:false, selected:null, tab:'detail', filter:'all', cursor:0, cutoff:Infinity, zoom:1, x:30, y:50, graphWidth:800, graphHeight:400, follow:true, poll:null, playing:null, uploads:[], loadingDetail:0, evidenceSelection:null};
const statusNames = {running:'调查进行中', stopping:'正在停止', completed:'调查完成', failed:'运行未完成', stopped:'已停止', interrupted:'运行中断'};
const actionNames = {SEARCH:'搜索与打开来源', INSPECT:'检查原始材料', DETECT:'检测工具', READ_RECORD:'读取证据记录', SEARCH_MEMORY:'检索调查记忆', PIN_EVIDENCE:'保留重点证据'};
const operationNames = {get_frames:'提取视频帧', get_pages:'读取 PDF 页面', crop:'裁剪局部画面', ocr:'识别文字', asr:'转写音频', align_media:'对齐媒体片段', locate_frames:'定位视频片段', locate_regions:'定位图像区域'};
const paramNames={source:'来源地址',query:'搜索内容',targets:'目标证据',record_id:'记录编号',offset:'起始位置',count:'最多数量',times_s:'时间点（秒）',start_s:'起点（秒）',end_s:'终点（秒）',page_numbers:'页码',mode:'处理模式',top_k:'候选数量',output_type:'返回类型',input_type:'输入类型',operation:'具体操作',language:'语言'};
Object.assign(paramNames,{range_s:'处理范围（秒）',range_a_s:'A侧范围（秒）',range_b_s:'B侧范围（秒）',box:'裁剪范围'});
function seconds(value){return Number.isFinite(value)?String(Number(value.toFixed(3))):String(value);}
function actionSubtitle(output){
  const a=output.arguments||{},p=a.request?.params||{},targets=(a.targets||[]).join(' / '),range=v=>Array.isArray(v)?`${v.map(seconds).join('–')}秒`:'全片';
  if(output.action==='INSPECT'){
    switch(a.request?.operation){
      case 'align_media':return `输入 ${a.targets?.[0]||'A'}：${range(p.range_a_s)}\n输入 ${a.targets?.[1]||'B'}：${range(p.range_b_s)} · 最多${p.top_k||5}段`;
      case 'get_frames':return `输入 ${targets}\n${p.times_s?`${p.times_s.map(seconds).join('、')}秒`:`${range(p.range_s)} · ${p.count||'默认数量'}帧`}`;
      case 'get_pages':return `输入 ${targets} · 第${(p.page_numbers||[]).join('、')||'指定'}页`;
      case 'locate_frames':return `输入 ${targets} · ${range(p.range_s)}\n查找 ${p.query?.evidence_id||p.query?.text||'指定内容'}`;
      case 'crop':return `输入 ${targets}\n裁剪 ${(p.box||[]).join('、')}`;
      default:return `输入 ${targets}${p.range_s?` · ${range(p.range_s)}`:''}`;
    }
  }
  if(output.action==='READ_RECORD')return `读取 ${a.record_id} · 从第${a.offset||0}字符开始`;
  const value=a.query||a.source||a.record_id||'由 HI 选择并执行';return typeof value==='string'?value:JSON.stringify(value);
}
function resultSummary(observation){
  if(!observation)return '';
  if(observation.status==='failed')return '未取得处理结果';
  const ids=observation.result_evidence_ids||[],execution=observation.execution;
  if(!execution)return ids.length?`返回 ${ids.length} 条结果`:'';
  const created=new Set(execution.created_evidence_ids||[]),reused=new Set(execution.reused_evidence_ids||[]);
  const fresh=ids.filter(id=>created.has(id)).length,existing=ids.filter(id=>reused.has(id)||(execution.reused&&!created.has(id))).length;
  return [fresh?`新增 ${fresh} 条结果`:'',existing?`复用 ${existing} 条结果`:'',observation.match_count!=null?`${observation.match_count} 个匹配候选`:''].filter(Boolean).join(' · ')||'没有新增处理结果';
}
function readableParams(argumentsValue){const rows=[];function add(key,value){if(key==='request'){if(value?.operation)rows.push(['具体操作',operationNames[value.operation]||value.operation]);for(const [k,v] of Object.entries(value?.params||{}))add(k,v);}else rows.push([paramNames[key]||key,Array.isArray(value)?value.join(' · '):typeof value==='object'?JSON.stringify(value):value]);}Object.entries(argumentsValue||{}).forEach(([k,v])=>add(k,v));return rows.map(([k,v])=>`<div class="param-row"><dt>${esc(k)}</dt><dd>${typeof v==='string'?sourceLink(v):esc(v)}</dd></div>`).join('');}
const urlSafe = value => /^https?:\/\//i.test(String(value || ''));
const sourceLink = (url, label) => urlSafe(url) ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(label || url)}</a>` : esc(label || url);
function rich(text) {
  let html = esc(text);
  const links = [];
  html = html.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, (_, label, url) => {links.push(`<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`);return `\u0001${links.length - 1}\u0001`;});
  html = html.replace(/https?:\/\/[^\s<>]+/g, url => `<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`);
  html = html.replace(/\u0001(\d+)\u0001/g, (_, i) => links[Number(i)]);
  html = html.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>').replace(/^#{1,4} (.+)$/gm, '<h3>$1</h3>');
  // 引用按钮只处理文字节点，避免修改 URL 或标签属性。
  const template = document.createElement('template'); template.innerHTML = html;
  const walker = document.createTreeWalker(template.content, NodeFilter.SHOW_TEXT);
  const texts = []; while (walker.nextNode()) { if (!walker.currentNode.parentElement?.closest('a')) texts.push(walker.currentNode); }
  for (const node of texts) {
    const parts = node.textContent.split(/\b(E[1-9][0-9]*)\b/);
    if (parts.length < 2) continue;
    const fragment = document.createDocumentFragment();
    parts.forEach((p, i) => { if (i % 2) { const b = document.createElement('button'); b.className='evidence-id'; b.dataset.evidence=p; b.textContent=p; fragment.append(b); } else fragment.append(document.createTextNode(p)); });
    node.replaceWith(fragment);
  }
  return template.innerHTML;
}
async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers:{'X-Foren-Request':'1', ...(options.body && typeof options.body === 'string' ? {'Content-Type':'application/json'} : {}), ...options.headers}});
  if (response.status === 401) { clearTimeout(S.poll); location.reload(); throw new Error('请重新登录'); }
  if (!response.ok) {const data = await response.json().catch(() => ({}));throw Object.assign(new Error(typeof data.detail === 'string' ? data.detail : `请求未成功（${response.status}）`), {status:response.status});}
  return response.json();
}
let toastTimer;
function toast(message) {$('#toast').textContent=message;$('#toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(() => $('#toast').hidden=true, 5000);}
function showPage(page) {['library','workspace','history-page','training-page'].forEach(id => $(`#${id}`).hidden=id!==page);$('.nav-item[data-action="library"]').classList.toggle('active',page==='library');$('.nav-item[data-action="history"]').classList.toggle('active',page==='history-page');$('.nav-item[data-action="training"]').classList.toggle('active',page==='training-page');if(page!=='training-page')window.stopTrainingPlayback?.();$('#breadcrumb').textContent=page==='library'?'案例库':page==='history-page'?'调查记录':page==='training-page'?'方法与进展':'调查脑图';}
function navigate(hash){if(location.hash!==hash)history.pushState(null,'',hash);}
async function refreshStatus() {
  try { S.status=await api('/api/status'); $('#api-status').textContent=S.status.configured?'API 已就绪':'需要配置';$('#api-status').classList.toggle('ready',S.status.configured);$('#api-dot').style.background=S.status.configured?'#a9c97a':'#c99c62';$('#budget').textContent=S.status.trial?`每24小时 ${S.status.daily_runs} 次`:`可用 $${S.status.budget.remaining_usd.toFixed(2)}`; }
  catch {$('#api-status').textContent='服务连接中断';$('#api-status').classList.remove('ready');}
}
async function refreshLibrary() {
  [S.cases,S.history]=await Promise.all([api('/api/cases'),api('/api/runs')]);
  $('#case-count').textContent=S.cases.length;$('#history-count').textContent=S.history.length;
  $('#recent-runs').innerHTML=S.history.slice(0,8).map(j=>`<button class="recent-run" data-run="${esc(j.id)}"><i class="small-dot ${esc(j.status)}"></i><span>${esc(j.title)}</span></button>`).join('')||'<p class="sidebar-empty">从一个案例开始探索</p>';
  $('#history-list').innerHTML=S.history.map(j=>`<button class="history-row" data-run="${esc(j.id)}"><span><strong>${esc(j.title)}</strong><small>${new Date(j.started_at*1000).toLocaleString('zh-CN')} · 真实 API 调用</small></span><span>${statusNames[j.status]||j.status} →</span></button>`).join('')||'<div class="empty-library">还没有调查记录。选择案例或新建调查，即可开始。</div>';
  renderCases();
}
function renderCases() {
  $('#case-grid').innerHTML=S.cases.filter(c=>S.filter==='all'||c.kind===S.filter).map(c=>`<article class="case-card" data-kind="${esc(c.kind)}"><div class="case-visual">${c.thumbnail?`<img src="${esc(c.thumbnail)}" alt="${esc(c.dataset)} 案例材料预览" loading="lazy">`:`<div class="fallback-visual">${c.kind==='web'?'<div class="planet"></div><div class="fallback-title">MARS<br>FIRST FLIGHT</div>':'<div class="parallel-icons">PDF</div><span>＋</span><div class="parallel-icons">WEB</div>'}</div>`}<span class="dataset-tag">${esc(c.dataset)}</span><span class="media-count">${c.media_count?`${c.media_count} 份原始材料`:'实时网络检索'}</span></div><div class="card-content"><h3>${esc(c.title)}</h3><p>${esc(c.description)}</p><div class="case-tags">${c.tags.map(t=>`<span>${esc(t)}</span>`).join('')}</div><div class="case-actions">${c.record_id?`<button class="text-button" data-run="${esc(c.record_id)}">查看真实记录 →</button>`:c.latest_run_id?`<button class="text-button" data-run="${esc(c.latest_run_id)}">查看运行记录 →</button>`:'<span class="muted" style="font-size:9px">等待首次运行</span>'}<button class="run-case" data-start-case="${esc(c.id)}" ${!c.available?'disabled':''}>${c.available?'实时运行 ↗':'材料未就绪'}</button></div><div class="source-caption"><span title="${esc(c.license)}">${esc(c.sample)}</span>${sourceLink(c.source_url,'数据来源 ↗')}</div></div></article>`).join('')||'<div class="empty-library">这一分类还没有案例。</div>';
}
function openNew() {$('#form-error').textContent='';$('#new-dialog').showModal();setTimeout(()=>$('#task-input').focus(),50);}
async function startCase(id) {
  const button=$(`[data-start-case="${id}"]`);if(button)button.disabled=true;
  try {const run=await api('/api/runs',{method:'POST',body:JSON.stringify({case_id:id})});await openRun(run.id);refreshLibrary().catch(()=>{});}
  catch(e){toast(e.message);}
  finally {if(button)button.disabled=false;}
}
async function openRun(id) {
  clearTimeout(S.poll);stopPlayback();S.run={id};S.events=new Map();S.cursor=0;S.cutoff=Infinity;S.selected=null;S.nodes=new Map();S.expanded=new Set();S.follow=true;S.tab='detail';S.loadingDetail++;
  showPage('workspace');$('#graph-nodes').innerHTML='';$('#graph-lines').innerHTML='';$('#graph-empty').hidden=false;$('#graph-empty').textContent='正在读取真实调查记录…';setTab('detail');
  navigate(`#run/${id}`);
  await pollRun(id,true);
}
async function pollRun(id, initial=false) {
  try {
    const data=await api(`/api/runs/${encodeURIComponent(id)}?after=${S.cursor}`);if(S.run?.id!==id)return;
    const previousSelectedEvents=S.nodes.get(S.selected)?.eventIds?.join(',');const oldCursor=S.cursor;data.events.forEach(e=>S.events.set(e.id,e));S.cursor=data.cursor;S.run=data;
    renderRunHeading();buildGraph();renderGraph(initial || (S.follow && oldCursor!==S.cursor));
    if(initial) {selectNode('root'); if(data.mode==='live'&&data.status==='running')toast('调查已开始，模型每次响应后更新脑图');}
    if(S.tab==='evidence'&&!S.evidenceSelection)renderEvidenceList();if(S.tab==='answer')renderAnswer();
    if(!initial&&S.selected&&S.tab==='detail'&&(previousSelectedEvents!==S.nodes.get(S.selected)?.eventIds?.join(',')||(oldCursor!==S.cursor&&(S.selected==='root'||S.nodes.get(S.selected)?.workspace))))renderDetail(S.selected);
    const active=['running','stopping'].includes(data.status);
    $('#replay-bar').hidden=active;
    $('#replay-slider').max=Math.max(S.events.size,1);if(S.cutoff===Infinity)$('#replay-slider').value=S.events.size;
    if(active)S.poll=setTimeout(()=>pollRun(id),1500);
    else {refreshStatus();refreshLibrary().catch(()=>{});}
  } catch(e) {if(S.run?.id!==id)return;$('#graph-empty').textContent=e.message;$('#run-notice').hidden=false;if(e.status>=400&&e.status<500&&e.status!==429){$('#run-notice').textContent=e.message;$('#run-status').textContent='无法打开';$('#run-status').className='status-pill failed';$('#stop-run').hidden=true;$('#rerun').hidden=true;$('#replay-bar').hidden=true;return;}$('#run-notice').textContent='连接暂时中断，正在重新连接；后台调查状态以恢复后的记录为准。';S.poll=setTimeout(()=>pollRun(id),4000);}
}
function renderRunHeading() {
  const r=S.run;$('#run-title').textContent=r.title;$('#run-mode').textContent=r.mode==='record'?'已有真实记录':'真实 API 调查';$('#run-status').textContent=statusNames[r.status]||r.status;$('#run-status').className=`status-pill ${r.status}`;
  const active=['running','stopping'].includes(r.status);$('#stop-run').hidden=!active;$('#stop-run').disabled=r.status==='stopping';$('#rerun').hidden=active;$('#rerun').textContent=r.can_resume?'继续调查':'重新运行';$('#run-cost').textContent=r.usage?.actual_cost_usd!=null?`已结算 $${r.usage.actual_cost_usd.toFixed(4)}`:'费用结算中';
  if(r.mode==='record'&&!r.usage)$('#run-cost').textContent='已有记录 · 不计费';
  const error=r.error||S.run.error;$('#run-notice').hidden=!(error||['stopped','interrupted','stopping'].includes(r.status));
  if(!$('#run-notice').hidden)$('#run-notice').textContent=r.status==='stopping'?'正在停止后续调查并保存记录；已发出的 API 请求可能仍会计费。':r.status==='stopped'?'调查已停止，已取得的证据与记录已保留。未知费用仍按原有账本预留。':r.status==='interrupted'?'服务曾中断。已取得的证据与记录保留，本记录不会自动重新调用 API。':`运行未完成：${error?.message||error?.type||'请查看相关步骤'}。已取得的证据与记录仍可查看。`;
  if(r.can_resume){$('#run-notice').hidden=false;$('#run-notice').textContent+=' 点击“继续调查”可复用已保存的调查进度，会产生新的 API 调用并占用一次试用次数；未知费用仍保留预留。';}
  $('#evidence-count').textContent=r.evidence?.length||0;updateClock();showRetryWait();
}
function updateClock(){const r=S.run;if(!r)return;const start=r.attempt_starts?.at(-1)||r.started_at;$('#run-time').textContent=start?`${r.attempt_starts?.length>1?'本次 ':''}${Math.max(0,(r.ended_at||Date.now()/1000)-start).toFixed(0)} 秒`:'历史记录';}
function showRetryWait(){
  if(!S.run||!['running','stopping'].includes(S.run.status)||S.run.status==='stopping')return;
  const waiting=new Map();for(const e of S.events.values()){if(e.event_type==='model_retry'){if(e.output?.state==='waiting')waiting.set(e.actor_id,e.output);else waiting.delete(e.actor_id);}else if(['raw_model_output','runtime_error'].includes(e.event_type))waiting.delete(e.actor_id);}
  if(waiting.size){$('#run-notice').hidden=false;$('#run-notice').textContent=[...waiting].map(([actor,w])=>`${actor}：API 暂时限流，约 ${Math.max(0,Math.ceil(w.resume_at-Date.now()/1000))} 秒后继续请求${w.retry_number?`（第 ${w.retry_number}/${w.maximum_retries} 次重试）`:''}`).join('；')+'。等待期间可停止调查。';}
}
setInterval(()=>{updateClock();showRetryWait();},1000);
function buildGraph() {
  const nodes=new Map();const root={id:'root',kind:'root',label:'调查问题',subtitle:S.run.task,badge:'起点',parent:null,eventIds:[]};nodes.set(root.id,root);
  const events=[...S.events.values()].filter(e=>e.id<=S.cutoff);
  const latest=new Map(), roundGias=new Map(), workspaceById=new Map((S.run.workspaces||[]).map(w=>[w.task_id,w]));let currentGia='root';
  for(const e of events){
    if(e.event_type==='model_input'){
      if(e.actor_role==='GIA') {const n={id:`g-${e.id}`,kind:'gia',label:'GIA · 全局调度',subtitle:'读取原始材料与已有证据，决定下一步调查',parent:'root',badge:'全局',eventIds:[e.id],pending:true};nodes.set(n.id,n);currentGia=n.id;latest.set('GIA',n);if(e.round_index)roundGias.set(e.round_index,n.id);}
      else if(e.actor_role==='HI') {
        let w=nodes.get(e.actor_id);const ws=workspaceById.get(e.actor_id);
        if(!w){const round=ws?.round_index||Number(e.actor_id.match(/^W(\d+)/)?.[1]||1);w={id:e.actor_id,kind:'hi',label:`HI · ${e.actor_id}`,subtitle:ws?.task||e.task||'独立局部调查',parent:roundGias.get(round)||currentGia,badge:'独立上下文',eventIds:[],workspace:ws};nodes.set(w.id,w);}
        const n={id:`h-${e.id}`,kind:'hi',label:'HI 正在分析',subtitle:'读取局部任务与可见证据',parent:e.actor_id,owner:e.actor_id,badge:'局部',eventIds:[e.id],pending:true};nodes.set(n.id,n);latest.set(e.actor_id,n);
      }
    } else if(e.event_type==='raw_model_output') {
      const n=latest.get(e.actor_id);if(n){n.eventIds.push(e.id);n.retry=null;n.badge=e.actor_role==='GIA'?'全局':'局部';n.pending=false;n.hasReasoning=e.has_reasoning;n.reasoningPreview=e.reasoning_preview;if(e.image_evidence_ids?.length)n.images=e.image_evidence_ids;}
    } else if(e.event_type==='model_retry') {
      const n=latest.get(e.actor_id);if(n){n.eventIds.push(e.id);n.retry=e.output;n.pending=['waiting','retrying'].includes(e.output?.state);n.badge=e.output?.state==='waiting'?'限流等待':e.output?.state==='retrying'?'正在重试':'重试结束';}
    } else if(e.event_type==='model_output') {
      const n=latest.get(e.actor_id);if(!n)continue;n.eventIds.push(e.id);n.output=e.output;n.pending=false;
      if(e.actor_role==='GIA'){
        if(e.output?.decision==='INVESTIGATE'){n.label='GIA · 派发调查任务';n.subtitle=`派发 ${e.output.tasks?.length||0} 项相互独立的调查任务`;if(e.round_index)roundGias.set(e.round_index,n.id);}
        else if(e.output?.decision==='FINISH'){n.kind='final';n.label='GIA · 最终结论';n.subtitle=e.output.answer||'综合证据与工作区报告';}
        else if(e.output?.decision==='WAIT'){n.label='GIA · 等待报告';n.badge='等待报告';n.subtitle='局部调查继续进行，等待更多证据与报告';}
        else {n.label='GIA · 读取全局记录';n.action=e.output?.action;n.badge=n.action||'全局';n.subtitle=e.output?.arguments?.record_id||e.output?.action||'综合调查进展';}
      } else if(e.output?.action){n.kind='tool';n.action=e.output.action;n.label=e.output.arguments?.request?.operation==='align_media'?(e.output.arguments.request.params?.mode==='audio'?'音频片段对齐':'视觉片段对齐'):(operationNames[e.output.arguments?.request?.operation]||actionNames[n.action]||n.action);n.subtitle=actionSubtitle(e.output);n.badge=n.action;n.pending=true;}
      else if(e.output?.status){n.kind='hi';n.label='HI · 局部报告';n.subtitle=e.output.status==='COMPLETED'?'所派任务已完成，提交局部报告':e.output.status_reason||'所派任务未完成，提交已确认部分与未决边界';n.isReport=true;}
    } else if(e.event_type==='observation'){
      const n=latest.get(e.actor_id);if(n){n.eventIds.push(e.id);n.pending=false;n.observation=e.observation;const es=e.observation?.evidence||[];n.evidenceIds=e.observation?.action==='INSPECT'?(e.observation.result_evidence_ids||[]):es.map(x=>x.evidence_id).filter(Boolean);n.outcome=resultSummary(e.observation);n.toolStatus=e.observation?.status;if(n.toolStatus==='failed')n.badge='工具失败';else if(n.toolStatus==='partial')n.badge='部分完成';else if(e.observation?.execution?.reused)n.badge='INSPECT · 结果复用';if(e.observation?.action==='HARNESS_REJECTION'){n.isCorrection=true;n.label='HI · 输出校验反馈';n.badge='需要修正';n.subtitle='本次输出未被接受，反馈交回同一 HI';}}
    } else if(['round_review_deferred','round_review_superseded','round_review_failed'].includes(e.event_type)) {
      const n=latest.get('GIA');if(n){n.eventIds.push(e.id);n.pending=false;n.retry=null;const deferred=e.event_type==='round_review_deferred',superseded=e.event_type==='round_review_superseded';n.label=deferred?'GIA · 检查延后':superseded?'GIA · 检查已结束':'GIA · 检查未完成';n.badge=deferred?'已延后':superseded?'报告已更新':'检查失败';n.subtitle=deferred?'先继续局部取证，后续依据新报告再检查':superseded?'已有新报告，后续依据更新后的调查记录判断':'局部调查继续，错误记录可在调用详情中查看';}
    } else if(e.event_type==='workspace_report') {const n=latest.get(e.actor_id);if(n){n.eventIds.push(e.id);n.report=e.output;n.isReport=true;n.pending=false;}}
    else if(e.event_type==='workspace_outcome'){const w=nodes.get(e.actor_id);if(w)w.eventIds.push(e.id);}
  }
  for(const n of nodes.values()){const ws=n.owner?workspaceById.get(n.owner):null;if(n.pending&&(!['running','stopping'].includes(S.run.status)||(ws&&['failed','cancelled','timed_out'].includes(ws.run_status)))){n.pending=false;n.badge='未完成';n.subtitle='本步骤未取得完整响应，记录已保留';}}
  const visible=new Map();
  for(const [id,n] of nodes){if(n.owner&&!S.expanded.has(n.owner)&&!n.isReport)continue;visible.set(id,n);}
  if(S.showEvidence){const attached=new Set();for(const n of visible.values()){for(const eid of n.evidenceIds||[]){if(attached.has(eid))continue;const e=S.run.evidence.find(e=>e.evidence_id===eid);if(!e)continue;attached.add(eid);visible.set(`e-${eid}`,{id:`e-${eid}`,kind:'evidence',label:`${eid} · ${e.label}`,subtitle:e.locator||e.acquired_by,parent:n.id,evidenceId:eid,badge:'共享',eventIds:[]});}}}
  for(const n of visible.values()){n.children=[];if(n.kind==='hi'&&workspaceById.has(n.id)){const steps=[...nodes.values()].filter(x=>x.owner===n.id&&!x.isReport),actions=steps.filter(x=>x.action&&!x.isCorrection).length,corrections=steps.filter(x=>x.isCorrection).length;n.badge=`${actions} 次行动${corrections?` · ${corrections} 次修正`:''}`;n.expansion=S.expanded.has(n.id)?'已展开':'点击展开';}}
  for(const n of visible.values()){if(n.parent&&visible.has(n.parent))visible.get(n.parent).children.push(n.id);}
  S.nodes=visible;$('#map-count').textContent=`${events.filter(e=>e.event_type==='model_output').length} 次决策`;
}
function renderGraph(fit=false) {
  const nodes=S.nodes;if(!nodes.size)return;let leaf=0,maxDepth=0;
  function layout(id,depth){const n=nodes.get(id);n.x=40+depth*290;maxDepth=Math.max(maxDepth,depth);if(!n.children.length){n.y=leaf*160+45;leaf++;}else{n.children.forEach(c=>layout(c,depth+1));n.y=(nodes.get(n.children[0]).y+nodes.get(n.children.at(-1)).y)/2;}}
  layout('root',0);S.graphWidth=40+maxDepth*290+276;S.graphHeight=Math.max(230,leaf*160+65);
  $('#graph-stage').style.width=`${S.graphWidth}px`;$('#graph-stage').style.height=`${S.graphHeight}px`;
  $('#graph-nodes').innerHTML=[...nodes.values()].map(n=>`<button class="graph-node ${n.kind} ${S.selected===n.id?'selected':''}" style="left:${n.x}px;top:${n.y}px" data-node="${esc(n.id)}" aria-label="${esc(n.label)}，${esc(n.subtitle)}，${esc(n.outcome||'')}"><div class="node-header"><span>${esc(n.badge||n.kind)}</span><span class="node-state ${n.pending?'pending':''}">${n.pending?'进行中':n.kind==='root'?'TASK':n.kind==='evidence'?'EVIDENCE':n.expansion|| (n.eventIds.length?'●':'')}</span></div><strong>${esc(n.label)}</strong><p title="${esc(n.subtitle)}">${esc(n.subtitle)}</p>${n.outcome?`<div class="node-outcome">${esc(n.outcome)}</div>`:''}</button>`).join('');
  $('#graph-lines').setAttribute('width',S.graphWidth);$('#graph-lines').setAttribute('height',S.graphHeight);
  $('#graph-lines').innerHTML=[...nodes.values()].filter(n=>n.parent&&nodes.has(n.parent)).map(n=>{const p=nodes.get(n.parent),x=p.x+236,y=p.y+64,end=n.x,ey=n.y+64;return `<path class="${n.kind==='evidence'?'evidence-line':''}" d="M ${x} ${y} C ${x+28} ${y}, ${end-28} ${ey}, ${end} ${ey}"/>`;}).join('');
  $('#graph-empty').hidden=true;if(fit)fitGraph();else transformGraph();
}
function transformGraph(){$('#graph-stage').style.transform=`translate(${S.x}px, ${S.y}px) scale(${S.zoom})`;$('#zoom-level').textContent=`${Math.round(S.zoom*100)}%`;}
function fitGraph(overview=false){const r=$('#graph-viewport').getBoundingClientRect();if(!r.width||!r.height)return;S.zoom=Math.max(overview?.15:.75,Math.min(1,(r.width-45)/S.graphWidth,(r.height-75)/S.graphHeight));const n=S.nodes.get(S.selected)||S.nodes.get('root');S.x=(r.width-S.graphWidth*S.zoom)/2;S.y=35+(r.height-40-S.graphHeight*S.zoom)/2;if(!overview&&S.graphHeight*S.zoom>r.height&&n)S.y=r.height/2-(n.y+64)*S.zoom;if(!overview&&S.graphWidth*S.zoom>r.width&&n)S.x=r.width/2-(n.x+118)*S.zoom;transformGraph();}
function zoomBy(factor){const r=$('#graph-viewport').getBoundingClientRect(),z=Math.min(2.2,Math.max(.15,S.zoom*factor)),f=z/S.zoom;S.x=r.width/2-(r.width/2-S.x)*f;S.y=r.height/2-(r.height/2-S.y)*f;S.zoom=z;S.follow=false;transformGraph();}
function setTab(tab){S.evidenceSelection=null;S.tab=tab;$$('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===tab));if(tab==='evidence')renderEvidenceList();else if(tab==='answer')renderAnswer();else if(S.selected)renderDetail(S.selected);else $('#inspector-content').innerHTML='<div class="detail-empty"><div>⌁</div><h3>沿着线索，深入一步</h3><p>点击脑图中的任意节点，查看分析、工具和证据。</p></div>';}
function mobilePane(pane){$('.investigation-layout').classList.toggle('inspector-visible',pane==='inspector');$$('[data-pane]').forEach(b=>b.classList.toggle('active',b.dataset.pane===pane));}
function selectNode(id){const n=S.nodes.get(id);if(!n)return;S.selected=id;if(id!=='root'){S.follow=false;const r=$('#graph-viewport').getBoundingClientRect();if(r.width){S.zoom=Math.max(.85,S.zoom);S.x=r.width/2-(n.x+118)*S.zoom;S.y=r.height/2-(n.y+64)*S.zoom;transformGraph();}}$$('.graph-node').forEach(b=>b.classList.toggle('selected',b.dataset.node===id));if(n.evidenceId){showEvidence(n.evidenceId);return;}setTab('detail');if(matchMedia('(max-width:760px)').matches)mobilePane('inspector');}
async function renderDetail(id) {
  const n=S.nodes.get(id);if(!n)return;const serial=++S.loadingDetail,runId=S.run.id;
  const el=$('#inspector-content');const ws=n.workspace;
  let html=`<div class="detail-type">${n.kind==='root'?'INVESTIGATION':n.kind==='gia'||n.kind==='final'?'GLOBAL INVESTIGATOR':n.kind==='tool'?'HI · STRUCTURED ACTION':'HARNESS INVESTIGATOR'}<span>${esc(n.badge||'')}</span></div><h2>${esc(n.label)}</h2>`;
  if(n.kind==='root'){
    html+=`<div class="rendered">${rich(S.run.task)}</div><div class="detail-section"><h3>如何阅读这次调查</h3><div class="detail-description">GIA 读取原始材料与已有证据，拆分调查目标并派发到多个 HI 工作区，轮内并行、轮间串行。每个 HI 在独立工作区内选择工具、获取证据并提交局部报告。点击节点可查看该步的分析、工具参数与返回证据。</div><div class="role-explainer">HI 自审所派任务是否完成，GIA 自审证据是否足以结束调查。证据在全局共享，局部结论按工作区隔离；两个角色均可直接读取画面。演示推理由 GIA 与 HI 共用 Qwen3.8-27B。</div></div>`;
    const branches=(S.run.workspaces||[]).filter(w=>S.nodes.has(w.task_id));if(branches.length)html+=`<div class="detail-section"><h3>局部调查</h3>${branches.map(w=>`<button class="step-link" data-node="${esc(w.task_id)}"><span>${esc(w.task_id)}</span><strong>${esc(w.task)}</strong><i>→</i></button>`).join('')}</div>`;
    const initial=S.run.evidence.filter(e=>e.acquired_by==='HARNESS/ingest');if(initial.length)html+=`<div class="detail-section"><h3>输入材料</h3>${initial.map(e=>evidenceCard(e)).join('')}</div>`;
    html+=`<button class="secondary" data-action="answer">查看最终结果 →</button><details><summary>全部运行事件（${S.events.size}）</summary>${[...S.events.values()].map(e=>`<button class="event-link" data-show-event="${e.id}">#${e.id} · ${esc(e.actor_id)} · ${esc(e.event_type)}</button>`).join('')}</details>`;el.innerHTML=html;return;
  }
  if(ws){const count=[...S.events.values()].filter(e=>e.actor_id===id&&e.event_type==='model_output').length;html+=`<div class="rendered">${rich(ws.task)}</div><div class="role-explainer">${esc(id)} 有独立的任务、行动历史与局部报告。它可以读取已共享的证据，其他 HI 的判断不会进入它的上下文。</div><div class="detail-summary"><span class="chip">第 ${ws.round_index} 轮</span><span class="chip">${count} 次决策</span><span class="chip">${esc(({reported:'已提交报告',failed:'运行失败',cancelled:'已取消',timed_out:'超出工作区时限',running:'调查中',pending:'待开始'})[ws.run_status]||ws.run_status)}</span></div>${ws.error?`<div class="verification fail"><strong>工作区未正常结束</strong>${esc(ws.error.message||ws.error.type||'运行中断')}<details><summary>原始失败记录</summary><pre>${esc(JSON.stringify(ws.error,null,2))}</pre></details></div>`:''}<p><button class="secondary" data-expand-workspace="${esc(id)}">${S.expanded.has(id)?'收起':'展开'}该 HI 的全部步骤</button></p><div class="step-list">${[...S.nodes.values()].filter(x=>x.owner===id).map((x,i)=>`<button class="step-link" data-node="${esc(x.id)}"><span>${String(i+1).padStart(2,'0')}</span><strong>${esc(x.label)}</strong><i>→</i></button>`).join('')}</div>`;el.innerHTML=html;return;}
  el.innerHTML=html+'<div class="detail-description">正在读取该步骤的真实记录…</div>';
  try {
    const events=await Promise.all(n.eventIds.map(i=>api(`/api/runs/${runId}/events/${i}`)));if(serial!==S.loadingDetail||S.tab!=='detail'||S.run.id!==runId)return;
    const raws=events.filter(e=>e.event_type==='raw_model_output'),obs=events.find(e=>e.event_type==='observation'),output=n.output||{},input=events.find(e=>e.event_type==='model_input');
    if(output.decision==='INVESTIGATE')html+=`<h3>分配的调查目标</h3>${(output.tasks||[]).map((t,i)=>`<div class="role-explainer"><strong>调查 ${i+1}</strong><div>${rich(t.task)}</div></div>`).join('')}`;
    if(output.decision==='FINISH')html+=`<div class="rendered">${rich(output.answer)}</div>`;
    if(n.action){html+=`<div class="detail-description">${n.owner?`由 ${esc(n.owner)} 选择该行动；工具负责执行，材料含义与证据判断由 HI 完成。`:'GIA 读取全局调查记录，为后续调度或汇总提供依据。'}</div><h3>工具参数</h3><dl>${readableParams(output.arguments)}</dl>`;}
    if(n.report||output.findings){const r=n.report||output;html+=`<div class="detail-summary"><span class="chip">${r.status==='COMPLETED'?'局部任务已完成':'局部任务未完成'}</span></div>${r.status_reason?`<p class="detail-description">${esc(r.status_reason)}</p>`:''}${(r.findings||[]).map(f=>`<div class="detail-section"><div class="rendered">${rich(f.statement)}</div><p>${(f.evidence_ids||[]).map(e=>`<button class="evidence-id" data-evidence="${esc(e)}">${esc(e)}</button>`).join(' ')}</p>${(f.limitations||[]).map(x=>`<div class="role-explainer">${rich(x)}</div>`).join('')}</div>`).join('')}`;}
    if(n.retry)html+=`<div class="role-explainer">${n.retry.state==='waiting'?`API 暂时限流，等待约 ${Math.ceil(n.retry.wait_seconds)} 秒后继续；本步骤尚未结束。`:n.retry.state==='retrying'?'等待已结束，正在重新请求同一步模型决策。':n.retry.state==='cancelled'?'等待已取消。':'已达到自动重试上限，失败记录已保留。'}</div>`;
    html+='<div class="detail-section"><h3>模型分析</h3><div class="detail-description">供应商实际返回的分析原文。它是模型的判断，不等同于已核实事实。</div>';
    html+=raws.length?raws.map((e,i)=>`<div class="detail-type" style="margin-top:12px">${raws.length>1?`第 ${i+1} 次响应 · `:''}${esc(e.payload.model||'')}<span>${esc(e.payload.provider||'')}</span></div>${e.payload.reasoning?`<div class="thought">${esc(e.payload.reasoning)}</div>`:'<div class="thought-empty">这次响应没有返回独立的分析文字。</div>'}`).join(''):'<div class="thought-empty">模型尚未返回这一步的响应。脑图在实际响应后更新。</div>';html+='</div>';
    const imageIds=[...new Set(raws.flatMap(e=>e.payload.image_evidence_ids||[]))];if(imageIds.length)html+=`<div class="detail-section"><h3>本次实际送入 ${raws[0]?.actor_role==='GIA'?'GIA':'HI'} 的图片</h3><div class="input-images">${imageIds.map(e=>`<button data-evidence="${esc(e)}"><img src="/api/runs/${runId}/evidence/${e}/file" alt="${e} 模型实际输入">${e} · 查看原始证据</button>`).join('')}</div></div>`;
    if(obs){const observation=obs.payload.observation;html+=`<div class="detail-section"><h3>工具返回</h3>${n.outcome?`<div class="detail-summary">${esc(n.outcome)}</div>`:''}${n.observation?.execution?.reused?'<div class="role-explainer">本次复用已保存的处理结果，没有再次执行媒体处理后端。原始输入编号保持不变。</div>':''}<div class="trace-text">${rich((observation.items||[]).join('\n'))}</div>${(n.evidenceIds||[]).length?`<p>${n.evidenceIds.map(e=>`<button class="evidence-id" data-evidence="${esc(e)}">${esc(e)}</button>`).join(' ')}</p>`:''}<details><summary>展开完整工具输出</summary><pre>${esc(JSON.stringify(observation,null,2))}</pre></details></div>`;}
    html+=`<div class="detail-section"><h3>完整调用记录</h3><div class="detail-description">含本步骤的模型输入、实际输出及执行反馈。</div>${events.map(e=>`<details><summary>#${e.id} · ${esc(e.event_type)} · ${esc(e.created_at)} UTC</summary><pre>${esc(JSON.stringify(e.payload,null,2))}</pre></details>`).join('')}</div>`;
    el.innerHTML=html;
  } catch(e) {if(serial===S.loadingDetail)el.innerHTML=html+`<p class="detail-description">${esc(e.message)}</p>`;}
}
function evidenceCard(e){const type=e.media_type||e.modalities?.join(' / ')||'材料';return `<button class="evidence-card" data-evidence="${esc(e.evidence_id)}"><header><span class="evidence-id">${esc(e.evidence_id)}</span><span>共享 · ${esc(type)}</span></header><strong>${esc(e.label)}</strong>${e.media_type?.startsWith('image/')?`<img loading="lazy" src="/api/runs/${S.run.id}/evidence/${e.evidence_id}/file" alt="${esc(e.label)}">`:''}<small>${esc(e.discovered_by)} · ${esc(e.acquired_by)}</small></button>`;}
function renderEvidenceList(){S.loadingDetail++;$('#inspector-content').innerHTML=`<div class="detail-type">SHARED EVIDENCE</div><h2>共享证据库</h2><div class="detail-description">${S.run.evidence.length} 份已提交材料。每一项都可追溯到原始来源或处理工具，编号本身不代表内容已被证实。</div><div class="detail-section">${S.run.evidence.map(evidenceCard).join('')||'<p class="detail-description">尚未取得证据。材料保存完成后会在这里出现。</p>'}</div>`;}
async function showEvidence(id){setTab('evidence');S.evidenceSelection=id;mobilePane('inspector');const serial=++S.loadingDetail,runId=S.run.id;$('#inspector-content').innerHTML='<p class="detail-description">正在读取证据…</p>';try{const e=await api(`/api/runs/${runId}/evidence/${id}`);if(serial!==S.loadingDetail||S.run.id!==runId)return;const file=`/api/runs/${runId}/evidence/${id}/file`;let media='';if(e.has_artifact){if(e.media_type?.startsWith('image/')&&!e.media_type.includes('svg'))media=`<a href="${file}" target="_blank"><img class="media-preview" src="${file}" alt="${esc(e.label)}"></a>`;else if(e.media_type?.startsWith('video/'))media=`<video class="media-preview video" controls preload="metadata" src="${file}"></video>`;else if(e.media_type?.startsWith('audio/'))media=`<audio class="media-preview" controls preload="metadata" src="${file}"></audio>`;else if(e.media_type==='application/pdf')media=`<iframe class="media-preview pdf" src="${file}" title="${esc(e.label)}"></iframe>`;}
$('#inspector-content').innerHTML=`<button class="text-button" data-tab="evidence">← 全部证据</button><div class="detail-type" style="margin-top:16px">${esc(id)}<span>同一调查内共享</span></div><h2>${esc(e.label)}</h2>${media}<div class="evidence-tools">${e.has_artifact?`<a href="${file}" target="_blank" rel="noopener">打开原始文件 ↗</a>`:''}${e.source_url?sourceLink(e.source_url,'访问来源 ↗'):''}</div><dl>${[['位置',e.locator],['取得方式',e.acquired_by],['产生者',e.discovered_by],['材料性质',e.production_method],['来源记录',e.source_ref]].map(([k,v])=>`<div class="param-row"><dt>${k}</dt><dd>${esc(v||'—')}</dd></div>`).join('')}</dl>${e.derived_from?.length?`<h3>派生自</h3>${e.derived_from.map(e=>`<button class="evidence-id" data-evidence="${esc(e)}">${esc(e)}</button>`).join(' ')}`:''}<div class="detail-section"><h3>材料正文</h3><div class="rendered">${rich(e.content)}</div></div>`;
}catch(e){if(serial===S.loadingDetail)$('#inspector-content').innerHTML=`<p>${esc(e.message)}</p>`;}}
function renderAnswer(){
  S.loadingDetail++;const r=S.run,v=r.verification;
  const verdict=v?.content_verdict;
  const label=verdict==='pass'?'独立核验：所问结论通过':verdict==='fail'?'独立核验：存在答案错误':verdict==='partial'?'独立核验：部分内容已确认':verdict==='not_completed'?'独立核验：未形成最终答案':'答案尚未完成核验';
  const review=v?`<div class="verification ${['fail','not_completed'].includes(verdict)?'fail':''}"><strong>${label}</strong>${typeof v.flow_passed==='boolean'?`<p>完整流程核验：${v.flow_passed?'通过':'未完成'}</p>`:''}${(v.remaining_issues||[]).map(x=>esc(x)).join('<br>')||'核验针对该次运行，不能代替其他任务的评估。'}${v.verified_points?.length?`<details><summary>查看核验依据</summary>${v.verified_points.map(x=>`<p>${esc(x)}</p>`).join('')}</details>`:''}</div>`:'<div class="verification"><strong>模型原始结论</strong>本次答案尚未独立核验。可点击 E 引用检查依据。</div>';
  $('#inspector-content').innerHTML=`<div class="detail-type">GLOBAL SYNTHESIS<span>GIA 汇总</span></div><h2>调查结果</h2>${review}${r.answer?`<div class="rendered">${rich(r.answer)}</div>`:`<div class="detail-empty"><div>⌁</div><h3>${['running','stopping'].includes(r.status)?'调查仍在进行':'没有形成最终答案'}</h3><p>已取得的材料和处理步骤均保留在脑图与证据库。</p></div>`}${r.usage?`<div class="detail-section"><div class="detail-description">${r.usage.request_count} 次模型调用 · 已结算 $${(r.usage.actual_cost_usd||0).toFixed(4)}${r.usage.unsettled_reserved_usd?` · 未结算预留 $${r.usage.unsettled_reserved_usd.toFixed(4)}`:''}</div></div>`:''}`;
}
function toggleWorkspace(id){S.expanded.has(id)?S.expanded.delete(id):S.expanded.add(id);buildGraph();renderGraph(true);if(S.tab==='detail')renderDetail(id);if(matchMedia('(max-width:760px)').matches)mobilePane('map');}
function stopPlayback(){clearInterval(S.playing);S.playing=null;$('#play-button').textContent='▷';}
function replayTo(count){const all=[...S.events.keys()];S.cutoff=count>=all.length?Infinity:count>0?all[count-1]:0;$('#replay-position').textContent=S.cutoff===Infinity?'全部':`${count}/${all.length}`;$('#replay-slider').value=count;buildGraph();renderGraph(true);}
$('#replay-slider').addEventListener('input',e=>{stopPlayback();replayTo(Number(e.target.value));});
const viewport=$('#graph-viewport');let drag=null;
viewport.addEventListener('pointerdown',e=>{if(e.target.closest('button'))return;drag={x:e.clientX,y:e.clientY,sx:S.x,sy:S.y};viewport.setPointerCapture(e.pointerId);viewport.classList.add('dragging');});
viewport.addEventListener('pointermove',e=>{if(!drag)return;S.x=drag.sx+e.clientX-drag.x;S.y=drag.sy+e.clientY-drag.y;S.follow=false;transformGraph();});
function endDrag(){drag=null;viewport.classList.remove('dragging');}viewport.addEventListener('pointerup',endDrag);viewport.addEventListener('pointercancel',endDrag);
viewport.addEventListener('wheel',e=>{e.preventDefault();zoomBy(e.deltaY>0?.9:1.1);},{passive:false});
new ResizeObserver(()=>{if(S.nodes.size)fitGraph();}).observe(viewport);
document.addEventListener('click',async e=>{
  const b=e.target.closest('button,a');if(!b)return;
  if(b.dataset.run){await openRun(b.dataset.run);return;}
  if(b.dataset.startCase){await startCase(b.dataset.startCase);return;}
  if(b.dataset.node){const n=S.nodes.get(b.dataset.node);selectNode(b.dataset.node);if(n?.workspace&&!S.expanded.has(n.id)){S.expanded.add(n.id);buildGraph();renderGraph(true);renderDetail(n.id);}return;}
  if(b.dataset.evidence){showEvidence(b.dataset.evidence);return;}
  if(b.dataset.showEvent){const runId=S.run.id,serial=++S.loadingDetail;try{const e=await api(`/api/runs/${runId}/events/${b.dataset.showEvent}`);if(serial===S.loadingDetail&&S.run.id===runId)$('#inspector-content').innerHTML=`<div class="detail-type">完整运行事件 · #${e.id}</div><h2>${esc(e.actor_id)} · ${esc(e.event_type)}</h2><div class="detail-description">${esc(e.created_at)} UTC · 保存的原始内容</div><pre>${esc(JSON.stringify(e.payload,null,2))}</pre>`;}catch(e){toast(e.message);}return;}
  if(b.dataset.expandWorkspace){toggleWorkspace(b.dataset.expandWorkspace);return;}
  if(b.dataset.tab){setTab(b.dataset.tab);return;}
  if(b.dataset.pane){mobilePane(b.dataset.pane);return;}
  if(b.dataset.filter){S.filter=b.dataset.filter;$$('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));renderCases();return;}
  if(b.dataset.removeUpload){S.uploads=S.uploads.filter(x=>x.id!==b.dataset.removeUpload);renderUploads();return;}
  switch(b.dataset.action){
    case 'logout':await api('/api/session',{method:'DELETE'});location.replace('/');break;
    case 'new':openNew();break;
    case 'close-dialog':$('#new-dialog').close();break;
    case 'library':clearTimeout(S.poll);stopPlayback();showPage('library');navigate('#');refreshLibrary().catch(e=>toast(e.message));break;
    case 'history':clearTimeout(S.poll);stopPlayback();showPage('history-page');navigate('#history');refreshLibrary().catch(e=>toast(e.message));break;
    case 'training':window.openTraining?.('#training');break;
    case 'zoom-in':zoomBy(1.2);break;
    case 'zoom-out':zoomBy(1/1.2);break;
    case 'fit':S.follow=false;fitGraph(true);$('#map-hint').textContent='全图概览 · 点击节点放大阅读';break;
    case 'expand':{const all=(S.run.workspaces||[]).map(w=>w.task_id),expanded=all.every(id=>S.expanded.has(id));S.expanded=new Set(expanded?[]:all);b.textContent=expanded?'展开步骤':'收起步骤';buildGraph();renderGraph(true);break;}
    case 'evidence-nodes':S.showEvidence=!S.showEvidence;b.textContent=S.showEvidence?'隐藏证据':'显示证据';buildGraph();renderGraph(true);break;
    case 'answer':setTab('answer');mobilePane('inspector');break;
    case 'stop':try{await api(`/api/runs/${S.run.id}/stop`,{method:'POST'});b.disabled=true;toast('正在停止后续调查并保存记录');}catch(e){toast(e.message);}break;
    case 'rerun':if(S.run.can_resume){b.disabled=true;try{const r=await api(`/api/runs/${S.run.id}/resume`,{method:'POST'});await openRun(r.id);refreshLibrary().catch(()=>{});}catch(e){toast(e.message);}finally{b.disabled=false;}}else if(S.run.case_id)startCase(S.run.case_id);else{try{const r=await api('/api/runs',{method:'POST',body:JSON.stringify({task:S.run.task,upload_ids:S.run.upload_ids||[]})});await openRun(r.id);}catch(e){toast(e.message);}}break;
    case 'latest':stopPlayback();replayTo(S.events.size);break;
    case 'play':if(S.playing){stopPlayback();break;}if(S.cutoff===Infinity)replayTo(0);$('#play-button').textContent='Ⅱ';S.playing=setInterval(()=>{const next=Number($('#replay-slider').value)+1;replayTo(next);if(next>=S.events.size)stopPlayback();},700);break;
  }
});
function renderUploads(){$('#upload-list').innerHTML=S.uploads.map(u=>`<div class="upload-item"><span>${esc(u.name)} · ${(u.size/1024/1024).toFixed(1)} MB</span><button type="button" data-remove-upload="${esc(u.id)}" aria-label="移除 ${esc(u.name)}">×</button></div>`).join('');}
$('#file-input').addEventListener('change',async e=>{const files=[...e.target.files];if(S.uploads.length+files.length>8){$('#form-error').textContent='每次最多添加8份材料';return;}$('#submit-run').disabled=true;try{for(const file of files){if(file.size>64*1024*1024)throw new Error(`${file.name} 超过64 MB`);$('#form-error').textContent=`正在上传 ${file.name}…`;const uploaded=await api(`/api/uploads?name=${encodeURIComponent(file.name)}`,{method:'POST',body:file,headers:{'Content-Type':'application/octet-stream'}});S.uploads.push(uploaded);renderUploads();}$('#form-error').textContent='';}catch(err){$('#form-error').textContent=err.message;}finally{$('#submit-run').disabled=false;e.target.value='';}});
$('#new-form').addEventListener('submit',async e=>{e.preventDefault();$('#submit-run').disabled=true;$('#form-error').textContent='';try{const r=await api('/api/runs',{method:'POST',body:JSON.stringify({task:$('#task-input').value,upload_ids:S.uploads.map(u=>u.id)})});$('#new-dialog').close();S.uploads=[];renderUploads();await openRun(r.id);refreshLibrary().catch(()=>{});}catch(err){$('#form-error').textContent=err.message;}finally{$('#submit-run').disabled=false;}});
document.addEventListener('keydown',e=>{if(e.key.toLowerCase()==='n'&&!e.ctrlKey&&!e.metaKey&&!e.altKey&&!['INPUT','TEXTAREA'].includes(document.activeElement.tagName)&&!document.querySelector('dialog[open]')){e.preventDefault();openNew();}});
document.addEventListener('error',e=>{const media=e.target;if(media instanceof HTMLMediaElement&&media.classList.contains('media-preview')&&!media.dataset.errorShown){media.dataset.errorShown='1';const note=document.createElement('p');note.className='role-explainer';note.textContent='浏览器无法直接播放此文件。可打开原始文件，或在证据库查看已提取的帧和转写结果。';media.insertAdjacentElement('afterend',note);}},true);
async function init(){const session=await api('/api/session');$('#logout').hidden=!session.trial;if(session.user)$('#logout').title=`${session.user.name} · 退出登录`;await Promise.all([refreshStatus(),refreshLibrary()]);if(S.status?.trial){$('#budget').textContent=`每24小时 ${S.status.daily_runs} 次`;$('#budget').title='试用次数包含已停止或未完成的调查';$('#new-form .dialog-footer span').innerHTML=`演示推理 Qwen3.8-27B · 每24小时最多 ${S.status.daily_runs} 次<br><small>材料会发送至模型服务进行分析</small>`;}const match=location.hash.match(/^#run\/([a-z0-9-]+)$/);if(location.hash.startsWith('#training'))await window.openTraining?.(location.hash);else if(match)await openRun(match[1]);else if(location.hash==='#history')showPage('history-page');}
function routeLocation(){if(location.hash.startsWith('#training')){window.openTraining?.(location.hash);return;}const match=location.hash.match(/^#run\/([a-z0-9-]+)$/);if(match){if(S.run?.id!==match[1]||$('#workspace').hidden)openRun(match[1]);}else{clearTimeout(S.poll);stopPlayback();showPage(location.hash==='#history'?'history-page':'library');}}
addEventListener('hashchange',routeLocation);addEventListener('popstate',routeLocation);
init().catch(e=>{toast(e.message);$('#case-grid').innerHTML='<div class="empty-library">暂时无法读取案例，请刷新页面重试。</div>';});
