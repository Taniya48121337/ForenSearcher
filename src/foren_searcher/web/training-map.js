"use strict";
(() => {
  const W = 244, H = 144, DX = 306, DY = 174;
  const stateTitles = ['已核对画面是否一致', '完整差异区间在哪里', '能否确定实际编辑方向', '哪个版本更早公开'];

  class TrainingMindMap {
    constructor(sample, callbacks) {
      this.sample = sample;
      this.callbacks = callbacks;
      this.nodes = new Map();
      this.expanded = new Set(['root']);
      this.selected = 'root';
      this.filter = 'all';
      this.zoom = 1;
      this.x = this.y = 0;
      this.build();
    }

    add(node) {
      node.children = [];
      this.nodes.set(node.id, node);
      if (node.parent) this.nodes.get(node.parent).children.push(node.id);
      return node;
    }

    build() {
      const s = this.sample, offline = s.mode === 'gia_offline';
      this.add({id:'root', kind:'root', label:offline?'同一份真实证据':'调查问题', subtitle:offline?s.description:s.task, badge:offline?'四个独立问题':'调查起点', parent:null});
      const rounds = new Map();
      if (s.mode === 'bridge') {
        for (const step of s.steps) {
          const index = step.round_index;
          if (index == null || rounds.has(index)) continue;
          const steps = s.steps.filter(v => v.round_index === index);
          const workers = new Set(steps.filter(v => v.role === 'HI').map(v => v.actor));
          const gia = steps.filter(v => v.role === 'GIA');
          const node = this.add({id:`round-${index}`, kind:'round', label:`第 ${index} 轮 · ${workers.size?'分工与取证':'汇总与回答'}`,
            subtitle:workers.size?`${workers.size} 个 HI 工作区${gia.length?` · ${gia.length} 次 GIA 输出`:' · 从已有计划启动'}`:`GIA 读取证据与报告，${steps.some(v=>v.action==='FINISH')?'形成最终回答':'检查调查进展'}`,
            badge:`ROUND ${String(index).padStart(2,'0')}`, parent:'root', collapsible:true});
          rounds.set(index, node.id);
        }
      }
      for (const step of s.steps.filter(v => v.role === 'GIA')) {
        const index = s.steps.indexOf(step);
        this.addStep(step, offline?'root':rounds.get(step.round_index)||'root', offline?`问题 ${index+1} · ${stateTitles[index]||'GIA 决策'}`:null);
      }
      for (const actor of new Set(s.steps.filter(v => v.role === 'HI').map(v => v.actor))) {
        const steps = s.steps.filter(v => v.actor === actor && v.role === 'HI');
        const first = steps[0];
        const plan = s.mode === 'bridge' && s.steps.find(v => v.role === 'GIA' && v.round_index != null && v.round_index === first.round_index && v.action === 'INVESTIGATE' && v.execution === 'gia_decision_accepted');
        const parent = plan?plan.id:rounds.get(first.round_index)||'root';
        const report = steps.findLast(v => v.action === 'REPORT');
        this.add({id:`workspace-${actor}`, kind:'hi', label:`HI · ${actor}`, subtitle:first.task,
          badge:'局部调查工作区', parent, collapsible:true,
          status:report?(report.report_status==='COMPLETED'?'局部任务完成':'局部任务未完成'):'未见提交报告'});
        for (const step of steps) this.addStep(step, `workspace-${actor}`);
      }
      const collect = id => {
        const node = this.nodes.get(id);
        node.steps = node.step?[node.step]:[];
        for (const child of node.children) node.steps.push(...collect(child));
        node.steps.sort((a,b) => s.steps.indexOf(a)-s.steps.indexOf(b));
        return node.steps;
      };
      collect('root');
    }

    addStep(step, parent, label) {
      const report = step.action === 'REPORT', final = step.action === 'FINISH';
      const subtitle = this.sample.mode==='gia_offline'?step.task:step.preview || (step.task_count?`安排 ${step.task_count} 个局部调查目标`:report?step.task:step.operation||step.task);
      this.add({id:step.id, kind:final?'final':report?'report':step.role==='GIA'?'gia':'tool',
        label:label || `${step.role === 'GIA'?'GIA · ':''}${this.callbacks.titleForStep(step)}`,
        subtitle, badge:`${step.actor} · ${step.id.startsWith('state')?'问题':'#'}${Number(step.id.split('_')[1])}`,
        step, parent});
    }

    matches(step) {
      return this.filter==='all' || (this.filter==='excluded'?!step.included:step.role===this.filter);
    }

    mount(container) {
      this.container = container;
      container.innerHTML = `<div class="tr-map-toolbar"><div><strong>调查脑图</strong><span id="tr-map-count"></span></div><button class="tr-control" data-map-action="overview">收起分支</button></div>
        <div class="tr-map-filters" aria-label="筛选脑图输出">${[['all','全部'],['GIA','GIA'],['HI','HI'],['excluded','未收录']].map(([id,label])=>`<button data-map-filter="${id}" aria-pressed="${this.filter===id}" class="${this.filter===id?'active':''}">${label}</button>`).join('')}</div>
        <div class="tr-map-legend"><span class="gia">GIA 调度</span><span class="hi">HI 取证</span><span class="report">报告 / 回答</span><span class="unselected">○ 未收录</span></div>
        <div class="tr-map-viewport" id="tr-map-viewport" tabindex="0" aria-label="调查脑图。拖动平移，使用加减按钮缩放，按方向键平移。"><div class="tr-map-stage"><svg class="tr-map-lines" aria-hidden="true"></svg><div class="tr-map-nodes"></div></div><div class="tr-map-empty" hidden>当前分类没有输出</div></div>
        <div class="tr-map-bottom"><span>拖动平移 · 点击节点查看详情</span><div><button data-map-action="out" aria-label="缩小脑图">−</button><output class="tr-map-zoom" aria-label="脑图缩放比例">100%</output><button data-map-action="in" aria-label="放大脑图">＋</button><button data-map-action="fit" aria-label="适应全部可见节点" title="适应画布">⊡</button><button data-map-action="locate" aria-label="定位选中节点" title="定位选中节点">◎</button></div></div>`;
      this.viewport = container.querySelector('.tr-map-viewport');
      container.addEventListener('click', e => {
        const button = e.target.closest('button');
        if (!button || !container.contains(button)) return;
        if (button.dataset.mapFilter) { this.callbacks.onFilter(button.dataset.mapFilter); return; }
        if (button.dataset.mapToggle) { this.toggle(button.dataset.mapToggle); return; }
        if (button.dataset.mapSelect) { this.select(button.dataset.mapSelect); return; }
        switch (button.dataset.mapAction) {
          case 'overview': this.expanded = new Set(['root']); this.selected='root'; this.render(); this.fit(); this.callbacks.onGroup(this.nodes.get('root')); break;
          case 'fit': this.fit(null,true); break;
          case 'locate': this.focus(this.selected); break;
          case 'in': this.zoomAt(this.zoom*1.2); break;
          case 'out': this.zoomAt(this.zoom/1.2); break;
        }
      });
      this.bindPan();
      this.resizeObserver = new ResizeObserver(() => {
        if (!this.viewport.clientWidth) return;
        if ((this.viewport.clientWidth<480)!==this.vertical) this.render();
        if (this.selected==='root') this.fit(); else this.focus(this.selected, false);
      });
      this.resizeObserver.observe(this.viewport);
      this.render();
      this.fit();
    }

    destroy() { this.resizeObserver?.disconnect(); }

    render() {
      if (!this.container) return;
      this.visible = new Map();
      const visit = id => {
        const node = this.nodes.get(id);
        if (id!=='root' && !node.steps.some(s=>this.matches(s))) return;
        this.visible.set(id, node);
        if (!node.collapsible || this.expanded.has(id)) node.children.forEach(visit);
      };
      visit('root');
      let leaf = 0, maxDepth = 0;
      let row = 0;
      this.vertical=(this.viewport.clientWidth || this.container.parentElement.clientWidth)<480;
      const layout = (id, depth) => {
        const node = this.visible.get(id), children=node.children.filter(c=>this.visible.has(c));
        if(this.vertical){node.x=25+depth*24;node.y=25+row++*DY;maxDepth=Math.max(depth,maxDepth);children.forEach(c=>layout(c,depth+1));return;}
        node.x=30+depth*DX; maxDepth=Math.max(depth,maxDepth);
        children.forEach(c=>layout(c,depth+1));
        node.y=children.length?(this.visible.get(children[0]).y+this.visible.get(children.at(-1)).y)/2:30+leaf++*DY;
      };
      layout('root',0);
      this.width=60+maxDepth*DX+W; this.height=60+Math.max(0,leaf-1)*DY+H;
      if(this.vertical){this.width=50+maxDepth*24+W;this.height=50+Math.max(0,row-1)*DY+H;}
      const stage=this.container.querySelector('.tr-map-stage');
      stage.style.width=`${this.width}px`; stage.style.height=`${this.height}px`;
      const svg=this.container.querySelector('svg'); svg.setAttribute('width',this.width); svg.setAttribute('height',this.height);
      svg.innerHTML=[...this.visible.values()].filter(n=>n.parent && this.visible.has(n.parent)).map(n=>{
        const parent=this.visible.get(n.parent), x=parent.x+W, y=parent.y+H/2, endY=n.y+H/2;
        if(this.vertical)return `<path class="${n.id===this.selected?'selected':''}" d="M ${parent.x+12} ${parent.y+H} V ${endY-10} Q ${parent.x+12} ${endY}, ${n.x} ${endY}"></path>`;
        return `<path class="${n.id===this.selected?'selected':''}" d="M ${x} ${y} C ${x+31} ${y}, ${n.x-31} ${endY}, ${n.x} ${endY}"></path>`;
      }).join('');
      this.container.querySelector('.tr-map-nodes').innerHTML=[...this.visible.values()].map(n=>{
        const step=n.step, matching=n.steps.filter(s=>this.matches(s));
        const footer=step?`${step.included?'✓ 已收录':'○ 未收录'}${step.rating==null?'':` · ${step.rating} / 4 分`}${step.image_count?` · ${step.image_count} 张图`:''}`:n.status||`${matching.length} 次输出 · ${matching.filter(s=>s.included).length} 条收录`;
        return `<div class="tr-map-node ${n.kind} ${n.id===this.selected?'selected':''} ${step&&!step.included?'not-included':''}" style="left:${n.x}px;top:${n.y}px" data-map-id="${esc(n.id)}">
          <button class="tr-map-select" data-map-select="${esc(n.id)}" aria-pressed="${n.id===this.selected}" title="${esc(n.subtitle)}"><span class="tr-node-badge">${esc(n.badge)}</span><strong>${esc(n.label)}</strong><p>${esc(n.subtitle)}</p><span class="tr-node-footer">${esc(footer)}</span></button>
          ${n.collapsible?`<button class="tr-map-toggle" data-map-toggle="${esc(n.id)}" aria-expanded="${this.expanded.has(n.id)}" aria-label="${this.expanded.has(n.id)?'收起':'展开'} ${esc(n.label)}">${this.expanded.has(n.id)?'−':'+'}</button>`:''}</div>`;
      }).join('');
      const count=this.sample.steps.filter(s=>this.matches(s)).length;
      this.container.querySelector('#tr-map-count').textContent=`${count} / ${this.sample.step_count} 次输出`;
      this.container.querySelector('.tr-map-empty').hidden=count>0;
      this.transform();
    }

    transform() {
      this.container.querySelector('.tr-map-stage').style.transform=`translate(${this.x}px,${this.y}px) scale(${this.zoom})`;
      this.container.querySelector('.tr-map-zoom').textContent=`${Math.round(this.zoom*100)}%`;
    }

    fit(id, overview=false) {
      if (!this.viewport?.clientWidth) return;
      let nodes=[...this.visible.values()];
      if (id && id!=='root') {
        const children=new Set();
        const collect=key=>{if(!this.visible.has(key))return;children.add(key);this.nodes.get(key).children.forEach(collect);};
        collect(id); nodes=nodes.filter(n=>children.has(n.id));
      }
      const left=Math.min(...nodes.map(n=>n.x))-25, top=Math.min(...nodes.map(n=>n.y))-25;
      const width=Math.max(...nodes.map(n=>n.x+W))-left+25, height=Math.max(...nodes.map(n=>n.y+H))-top+25;
      const z=Math.min(1,this.viewport.clientWidth/width,this.viewport.clientHeight/height);
      this.zoom=Math.max(id ? Math.min(.85,this.viewport.clientWidth/width) : .25,z);
      if(this.vertical&&!overview)this.zoom=Math.min(1,this.viewport.clientWidth/width);
      this.x=(this.viewport.clientWidth-width*this.zoom)/2-left*this.zoom;
      this.y=(this.viewport.clientHeight-height*this.zoom)/2-top*this.zoom;
      if (z<this.zoom && id) this.y=this.viewport.clientHeight/2-(this.nodes.get(id).y+H/2)*this.zoom;
      if(this.vertical&&!overview)this.y=16-top*this.zoom;
      this.transform();
    }

    focus(id, readable=true) {
      const node=this.visible.get(id); if(!node || !this.viewport.clientWidth)return;
      if(readable)this.zoom=Math.max(.85,this.zoom);
      this.x=this.viewport.clientWidth/2-(node.x+W/2)*this.zoom;
      this.y=this.viewport.clientHeight/2-(node.y+H/2)*this.zoom;
      this.transform();
    }

    zoomAt(value, x=this.viewport.clientWidth/2, y=this.viewport.clientHeight/2) {
      const zoom=Math.min(2,Math.max(.2,value)), ratio=zoom/this.zoom;
      this.x=x-(x-this.x)*ratio; this.y=y-(y-this.y)*ratio; this.zoom=zoom; this.transform();
    }

    reveal(id) {
      let node=this.nodes.get(id);
      while(node?.parent){this.expanded.add(node.parent);node=this.nodes.get(node.parent);}
    }

    selectStep(id, focus=true) {
      this.reveal(id); this.selected=id; this.render(); if(focus)this.focus(id);
    }

    select(id) {
      const node=this.nodes.get(id); if(!node)return;
      if(node.step){this.selectStep(id);this.callbacks.onStep(id);return;}
      this.selected=id;
      if(node.collapsible)this.expanded.add(id);
      this.render();this.fit(id);this.callbacks.onGroup(node);
    }

    toggle(id) {
      const closing=this.expanded.has(id);
      if(closing)this.expanded.delete(id);else this.expanded.add(id);
      this.selected=id;this.render();this.fit(closing?null:id);this.callbacks.onGroup(this.nodes.get(id));
      this.container.querySelector(`[data-map-toggle="${id}"]`)?.focus({preventScroll:true});
    }

    setFilter(value) {
      this.filter=value;
      this.container.querySelectorAll('[data-map-filter]').forEach(b=>{
        const active=b.dataset.mapFilter===value;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));
      });
      if (!this.nodes.get(this.selected)?.steps.some(s=>this.matches(s))) this.selected='root';
      this.render();this.fit();
    }

    bindPan() {
      const v=this.viewport;
      v.addEventListener('pointerdown',e=>{
        if(e.target.closest('button') || e.button!==0)return;
        this.drag={id:e.pointerId,x:e.clientX,y:e.clientY,originX:this.x,originY:this.y};
        v.setPointerCapture(e.pointerId);v.classList.add('dragging');
      });
      v.addEventListener('pointermove',e=>{
        if(!this.drag || e.pointerId!==this.drag.id)return;
        const dx=e.clientX-this.drag.x,dy=e.clientY-this.drag.y;
        this.x=this.drag.originX+dx;this.y=this.drag.originY+dy;this.transform();
      });
      const end=()=>{this.drag=null;v.classList.remove('dragging');};
      v.addEventListener('pointerup',end);v.addEventListener('pointercancel',end);
      v.addEventListener('wheel',e=>{
        e.preventDefault();
        if(e.ctrlKey || e.metaKey){const r=v.getBoundingClientRect();this.zoomAt(this.zoom*Math.exp(-e.deltaY*.005),e.clientX-r.left,e.clientY-r.top);}
        else {this.x-=e.deltaX;this.y-=e.deltaY;this.transform();}
      },{passive:false});
      v.addEventListener('keydown',e=>{
        if(e.target!==v)return;
        if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)){
          e.preventDefault();this.x+=e.key==='ArrowLeft'?50:e.key==='ArrowRight'?-50:0;this.y+=e.key==='ArrowUp'?50:e.key==='ArrowDown'?-50:0;this.transform();
        }else if(['+','=','-','0'].includes(e.key)){
          e.preventDefault();if(e.key==='0')this.fit();else this.zoomAt(this.zoom*(e.key==='-'?1/1.2:1.2));
        }
      });
    }
  }
  window.TrainingMindMap=TrainingMindMap;
})();
