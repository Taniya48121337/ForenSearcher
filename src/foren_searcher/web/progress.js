/* ForenSearcher 项目进展页（2026-10-13 重做）。无外部依赖；CSP script-src 'self' 兼容（无内联脚本）。 */
(function () {
  'use strict';

  var IMG = '/assets/progress/';
  var ST_TEXT = { done: '已完成', doing: '进行中', warn: '有问题', todo: '没开始 / 在等', drop: '已放弃' };

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }
  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  /* ================= 数据：全景脑图 ================= */
  var OVERVIEW = {
    t: 'ForenSearcher', s: '会查视频的 AI 调查员', root: true,
    d: '<p>整个项目分 6 块。前 3 块（任务、智能体、考试）已经有能用的东西；后 3 块（造教材、判对错、训练与进化）还在半路，<b>第一次真实训练还没跑</b>。</p><p>最大的瓶颈是服务器还没开跑。</p>',
    kids: [
      { t: '① 任务', s: '要解决什么问题', st: 'done', go: '#task',
        d: '<p>给一段被做过手脚的视频，要求找到原片、对比出改了哪里、说清楚想骗人相信什么，每句话都要带证据编号。</p><p>考试用 EVID-Bench（222 题），练习用 6 个数据集的 33 个案例。</p>',
        kids: [
          { t: 'EVID-Bench 接入', s: '222 题，正式报分用 91 题', st: 'done' },
          { t: '练习案例 33 个', s: '五份划分互不泄漏', st: 'done' }
        ] },
      { t: '② 智能体', s: '它怎么查', st: 'doing', go: '#agent',
        d: '<p>项目经理（GIA）+ 调查员（HI）的双角色框架 9 月就完成了。10-11 加上"找原片并逐帧比对"（DETECT v2.4）后成绩翻倍。</p><p>10-12 把 DETECT 并进了搜索和看材料两个工具，小测掉分（23 → 13/44）。10-13 查清是三处衔接问题，修了两轮，正在复测。</p>',
        kids: [
          { t: '双角色框架', s: 'GIA 派活、HI 动手、共享证据库', st: 'done',
            d: '<p>同一轮的多个工作区并行、轮与轮之间串行；两个角色都能直接看图。9 月完成并在真实任务上跑通。</p>' },
          { t: '找原片 + 逐帧比对', s: 'DETECT v2.4 · AI 类 1.9% → 24.4%', st: 'done', go: '#score',
            d: '<p>搜到候选视频后，程序自动排序、下载、和材料逐帧比对，找出换脸、改物体、插入片段，再请模型给每张对照图写一句描述。</p><p>依据：《DETECT_v2全量结果_20261011》。</p>' },
          { t: '工具合并（10-12）', s: '掉分 23 → 13/44', st: 'warn', go: '#agent-merge',
            d: '<p>新的工具目录：SEARCH 搜到视频自动标"像不像同源"；INSPECT 6 项操作（含 compare_media、video_structure）；以图搜图 4 引擎默认开。</p><p>掉分原因（10-13 查清）：文章页里的原片不让下载；图文搜索丢了文字那半；比对摘要把重要的行排到看不见的位置。上一版说的"零调用"是计数 bug。</p>' },
          { t: '合并后修复两轮', s: '修复 1：12/44；修复 2 复测中', st: 'doing', go: '#agent-merge',
            d: '<p>修复 1 让原片被比对到 0/14 → 6/14，但另 3 题在别处失分，总分 12/44。修复 2 加了收尾检查（答案不是编号列表就退回重写一次）和图文搜索名额放宽，evidM27c / d 连跑两次取平均，验收 ≥ 17.5/44。</p><p>依据：《合并后掉分诊断与修复_evidM27a》《合并目录修复2》。</p>' },
          { t: '新工具：日期查询', s: 'SEARCH 查发布 / 上传 / 最早存档', st: 'done',
            d: '<p>回答"谁先发布 / 是否早于声明"用。返回页面声明日期、视频上传日期、Wayback 最早存档，每个日期都写明含义（存档只说明"不晚于"）。免费、不要 key。hi@0.9.0 起开放；真实网络在 wk8 里第一次用。</p>' },
          { t: '以图搜图多引擎', s: '代码好了，缺 3 个 key', st: 'todo',
            d: '<p>Google Cloud Vision、Google Lens（SerpAPI）、TinEye、Yandex 同时搜、合并去重。目前只有免 key 的 Yandex 能用，而它的召回很低。</p>' }
        ] },
      { t: '③ 成绩', s: '考多少分', st: 'doing', go: '#score',
        d: '<p>EVID 正式 91 题：8B 8.5%，27B 15.9%，27B + 找原片 v2.4 <b>33.6%</b>（当前最好）。</p><p>服务器基线和另外两个数据集还没跑。</p>',
        kids: [
          { t: 'EVID 本机基线', s: '27B 15.9% → 33.6%', st: 'done' },
          { t: 'EVID 服务器基线', s: '已打包，没上机', st: 'todo' },
          { t: 'FMNV 视频', s: '已下 281 个（YouTube 上还在的）', st: 'doing' },
          { t: 'GroundLie360 视频', s: '解压出 0 个，要再看', st: 'todo' }
        ] },
      { t: '④ 训练数据', s: '造示范教材', st: 'doing', go: '#data',
        d: '<p>现行方法：程序按规则走每一步，教师大模型只填报告和答案（walker）。"两段视频比对"这类已稳定（wk6，答案核验后重写达 8/9）；wk8 扩到 5 类题、29 个运行，正在跑。</p>',
        kids: [
          { t: '教师自己走（r0–r0e）', s: '慢、乱；数据留作混合', st: 'done',
            d: '<p>每个案例 26–40 次大模型调用、9 分多钟，学生学到重复调用。现在不再用它造主数据，但 r0e 的 361 行调查员 / 42 行项目经理数据会混入训练。</p>' },
          { t: '分叉续跑 tp0–tp2', s: '噪声比信号大', st: 'drop' },
          { t: 'walker 两段视频比对（B）', s: 'wk1–wk6，已稳定', st: 'done', go: '#data',
            d: '<p>9 个 VideoSham / VCSL / QVHighlights 案例。wk6：平均 2.6 次大模型调用、1.6 分钟；报告核验 8/10；答案 4/9，核验后重写到 8/9。可程序核对的标准答案要点全部对上。</p>' },
          { t: '新 4 类题（A / C / E / L）', s: 'wk8 首跑，运行中', st: 'doing', go: '#data-wk8',
            d: '<p>找原片 3、核实说法 5、图片最早出处 6、照片拍摄地 4，加 2 条视图压缩变体。目的是让每个工具都在合理的情形下被用到。</p>' },
          { t: '程序化造题（几百题）', s: '提案，待试点', st: 'todo' }
        ] },
      { t: '⑤ 判对错', s: '核验器', st: 'doing', go: '#verify',
        d: '<p>三层核验能用（verifier 0.3.6），但核验模型暂时借 Sonnet，人工金标没人标，参考核验器没冻结。</p>',
        kids: [
          { t: '三层核验', s: 'verifier 0.3.6', st: 'done' },
          { t: '核验后重写答案', s: '4/9 → 8/9，默认开启', st: 'done' },
          { t: '正式核验模型', s: '等服务器；暂用 Sonnet', st: 'doing' },
          { t: '人工金标 80 条', s: '标注页已备好', st: 'todo' }
        ] },
      { t: '⑥ 训练与进化', s: '论文主线', st: 'todo', go: '#loop',
        d: '<p>设计已定稿（7 个零件），8 卡脚本写好；但<b>一次真实训练都没跑</b>，诊断 / 路由 / 门控代码还没写。</p>',
        kids: [
          { t: '训练脚本', s: '本机冒烟通过', st: 'done' },
          { t: '122B 教师部署', s: '没开始', st: 'todo' },
          { t: '第一次 SFT', s: '等数据包和服务器', st: 'todo' },
          { t: '诊断 / 路由 / 门控', s: '一行没写', st: 'todo' },
          { t: '论文', s: '方法写完，实验为零', st: 'doing' }
        ] }
    ]
  };

  /* ================= 数据：案例 A 的调查脑图 ================= */
  var CASE205 = {
    t: '总任务：找出视频 A 的伪造点', s: 'EVID 第 205 题 · 525 秒舞蹈合集', root: true,
    d: '<p>题目要求上网找原片、对比，列出 3–5 个伪造点。标准答案 3 点：背包→白纸箱、冰箱→售货机、零食→灭火器。</p>',
    kids: [
      { t: 'GIA 项目经理 · 第 1 轮', st: 'gia', b: ['gia', 'GIA'],
        d: '<p>项目经理只做两件事：决定这一轮派哪些任务；所有报告回来后判断证据够不够、写最终答案。</p>',
        kids: [
          { t: '连查 5 次记忆', s: '证据库还是空的，白查', st: 'warn', b: ['bad', '弯路'],
            d: '<p>开始时证据库里只有原始视频，项目经理却用不同关键词连续 5 次 QUERY_MEMORY，什么也没查到。</p><p>这是 27B 常见的"重复调用"毛病：本机基线里重复调用率 20%（8B 是 61%）。训练数据要教的第一件事就是不要这样。</p>' },
          { t: '派出 1 个工作区 W1-1', s: '任务 = 原题', st: 'gia', b: ['gia', 'GIA'],
            d: '<p>这题只有一个材料、一个目标，所以只开了一个工作区。复杂题会同时开多个，比如"找来源"和"核实事件"分开查。</p>',
            kids: [
              { t: 'HI 调查员 · W1-1', s: '11 个动作 → 交报告', st: 'hi', b: ['hi', 'HI'],
                d: '<p>调查员每一步只做一件事：选一个工具、填好参数。下面是它的真实动作顺序。</p>',
                kids: [
                  { t: '1. 听音频', s: '只有音乐，没有旁白', st: 'hi',
                    d: '<p>INSPECT asr（语音转写）。结果 E35：没有可识别的说话声。所以伪造只可能在画面上。</p>' },
                  { t: '2. 搜「"ONE Challenge" dance … split screen」', s: '自动比对 R1–R5：最高 3.8%，都不是', st: 'hi',
                    d: '<p>SEARCH 按视频画面两侧的台标写了查询词。搜到 5 个同频道的合集，程序自动下载并逐帧比对（E24）：和材料重合的画面只有 3.3%–3.8%，都不是原片。</p><p>这个"同频道、标题很像、却不是同一期"的情况很常见，所以必须靠比对确认，不能看标题。</p>' },
                  { t: '3. 换词再搜', s: 'R6–R8 也不是（≤ 3.6%）', st: 'hi',
                    d: '<p>换了查询词再搜，又比了 3 个候选（E34），还是对不上。中间回头翻读了搜索结果 E8 和比对记录 E24。</p>' },
                  { t: '4. 读画面文字、细听前 120 秒', s: 'OCR + ASR', st: 'hi',
                    d: '<p>想从画面文字里找更多线索，并重新转写前 120 秒（E35）。没有新线索。</p>' },
                  { t: '5. 扫描"10 秒镜头"', s: '没发现；没发现 ≠ 是真的', st: 'hi',
                    d: '<p>很多 AI 视频生成器一次出 10 秒，所以剪辑点之间正好 10 秒的镜头可疑。扫描 525 秒、59 个镜头，没有发现（E36）。</p><p>工具输出里明确写着：<b>没发现不能说明视频是真的</b>。这条是 10-11 修的——之前有一题因为"没找到原片"就写成"未见改动"，从 4/5 掉到 0/5。</p>' },
                  { t: '6. 第 3 次搜索 → 找到原片 R11', s: '97% 画面重合', st: 'res', b: ['ok', '关键'],
                    d: '<p>R11：YouTube「Tiktok Dance Battle Compilation – November 2024」，525 秒对 526 秒、同分辨率，A 的半秒采样 97% 都能在 R11 里找到（E48）。确认是原片。</p>',
                    kids: [
                      { t: '7. 程序逐帧比对', s: '找出 4 处局部改动区域', st: 'res',
                        d: '<p>其余画面逐帧一致、48 条人脸轨迹都对得上（没有换脸），只有 4 个小区域不同。程序给每处生成一张对照图，并请模型写一句"原片是什么、A 里是什么"。</p>',
                        kids: [
                          { t: '冰箱 → 售货机', s: '308–320 秒 · 3.8%', st: 'res', img: 'c205_fridge.jpg', th: 'c205_fridge_t.jpg', cap: '对照图 E44：原片（右）是白色冰箱，A（左）变成饮料售货机。' },
                          { t: '黑包 → 白纸箱', s: '189–208 秒 · 0.5%', st: 'res', img: 'c205_bag.jpg', th: 'c205_bag_t.jpg', cap: '对照图 E45：原片桌上是黑色背包，A 里变成白纸箱。' },
                          { t: '"红盒 → 蓝盒"', s: '469–479 秒 · 0.1% · 实为零食→灭火器', st: 'warn', img: 'c205_shelf.jpg', th: 'c205_shelf_t.jpg', cap: '对照图 E47：位置找对了，但描述写成"红盒 → 蓝盒"；实际是原片的零食被换成了红色灭火器。' }
                        ] }
                    ] },
                  { t: '8. 交报告', s: '状态：已完成', st: 'hi',
                    d: '<p>报告把"重新上传的旧合集"和 3 处物体替换写成 Finding，每条都引用了证据编号。中间有一次输出格式不合规被拒，重写后通过。</p>' }
                ] }
            ] },
          { t: '回看比对记录 E48', s: '翻了 2 页', st: 'gia', b: ['gia', 'GIA'],
            d: '<p>项目经理写答案前自己去翻比对结果，确认报告没有夸大。设计上要求两个角色都能直接看原始证据，不只听报告。</p>' },
          { t: '最终答案：5 个伪造点', s: '评委 3/3', st: 'res', b: ['ok', '得分'],
            d: '<p>① 这是一期旧合集的重新上传；② 308–320 秒冰箱→售货机；③ 189–208 秒黑包→白盒；④ 469–479 秒红盒→蓝盒；⑤ 总结。</p><p>每句都带编号（E44、E45、E47、E48…）。评委判 3 个标准点全部说中。</p><p class="muted">同一题在工具合并后：M27a 0/3（文字搜索被丢，原片没进池）→ 修复 1 后 M27b 1/3（原片回来了，报出冰箱，另两处没写具体）。</p>' }
        ] }
    ]
  };

  /* ================= 数据：案例 B 的 walker 轨迹 ================= */
  var P = ['prog', '程序'], T = ['llm', '教师'];
  var CASE4126 = {
    t: '题目：黑衣人在 B 可见、A 不可见的全部时间段', s: '标准答案 0.200–1.367 秒、2.300–6.000 秒', root: true,
    d: '<p>两段 6 秒视频，B 是原版，A 把从镜头前走过的黑衣人抹掉了。题目还要求核对每段的起止边界、说明精度。</p>',
    kids: [
      { t: 'GIA 第 1 轮：派 1 个工作区', s: '任务文字＝原题', st: 'gia', b: P,
        d: '<p>规则：给定两段视频的题，第 1 轮原样派出。不需要大模型。</p>',
        kids: [
          { t: 'HI 调查员 · W1-1', s: '10 个动作 + 报告', st: 'hi',
            kids: [
              { t: '1. 对齐两段视频', s: 'align_media：时间一一对应', st: 'hi', b: P,
                d: '<p>规则：两段视频都在、还没对齐 → 先对齐。结果：A 的每一刻都对应 B 的同一刻（没有剪掉或加速）。</p>' },
              { t: '2. 逐帧找不同', s: '两段：0.200–1.367 秒、2.300–5.99 秒', st: 'res', b: ['ok', '与标准一致到帧'], img: 'c4126_diff.jpg',
                cap: '差异图 E25（0.667 秒）：左 A、中 B、右边红色是不同的像素。',
                d: '<p>diff_media 逐帧比对 A、B，报出两段差异，起止和标准答案一致到帧（30 帧/秒）。</p><p>这一步就已经把题做对了，后面的问题都出在"怎么写"上。</p>' },
              { t: '3–4. 第一段边界两侧取帧', s: 'A、B 各取 0.167 / 0.200 / 1.333 / 1.367 秒', st: 'hi', b: P,
                d: '<p>规则：有了差异区间，就在起点和终点两侧各取一帧，A、B 同一时刻成对看，确认进出画面的时刻。</p>' },
              { t: '5. 翻回对比记录', s: '被截断的数要先读出来', st: 'hi', b: P,
                d: '<p>第二段的终点在学生视图里被截断了。规则（wk2 加的）：要用的数在视图里看不见，就先 READ_RECORD 读出来，不能直接用原始数据里的数——否则学生会学成"凭空写出看不见的数"。</p>' },
              { t: '6–7. 第二段边界取帧', s: '2.267 / 2.300 / 5.967 秒', st: 'hi', b: P },
              { t: '8–10. 转写两段语音 + 音轨对齐', s: '没有说话', st: 'hi', b: P,
                d: '<p>规则（wk2 加的）：两段都有音轨、题目没说只看画面 → 各转写一次再对齐音轨，避免漏掉"声音被换"。</p>' },
              { t: '11. 交报告', s: '状态被降为"未完成"', st: 'warn', b: T,
                d: '<p>程序判这个工作区走完了（状态 COMPLETED），教师写报告时降为 INCOMPLETE，理由是"像素不同不等于确认这个人一直可见""2.300 秒边缘那一小条认不出是不是他"。</p><p>教师只能把"完成"降为"未完成"，不能反过来。</p>' }
            ] }
        ] },
      { t: 'GIA：报告不完整 → 开第 2 轮', s: '先回看帧 E31，再派补查任务', st: 'gia', b: T,
        d: '<p>规则（wk3/wk4 加的）：报告有"未完成"且缺口是可补的（比如边界没定），就开第 2 轮。任务文字由教师按缺口写：确认 1.333–1.367 秒和 2.267–2.300 秒处是不是这个人。</p>',
        kids: [
          { t: 'HI 调查员 · W2-1', s: '6 个动作 + 报告', st: 'hi',
            kids: [
              { t: '逐帧细看 1.333–1.367 秒', s: 'A、B 各 6 帧', st: 'hi', b: P },
              { t: '2.267 / 2.300 秒再取帧', s: 'A、B 各一组', st: 'hi', b: P },
              { t: '只对 2.3–6.0 秒逐帧对比', s: 'diff_media 带范围', st: 'hi', b: P },
              { t: '回看全部边界帧', s: 'READ_RECORD 带 8 张图', st: 'hi', b: P },
              { t: '交报告', s: '仍是"未完成"', st: 'warn', b: T,
                d: '<p>第一个出点（1.333→1.367 秒）确认了；第二段"人物级"的区间教师仍不认，因为逐帧差异给的是"像素不同"的区间，不是"人可见"的区间。</p>' }
            ] }
        ] },
      { t: 'GIA：回看、查记忆 → 写答案', s: '答案核验未通过', st: 'warn', b: T,
        d: '<p>答案（教师写）：</p><div class="quote">Every person-specific interval could not be verified: the first candidate interval is [0.200, 1.367) s … The first exit is localized between 1.333 s, when part of the person remains visible in B …</div><p>两个问题：<b>① 太保守</b>，区间明明对了却说"无法确认"；<b>② 指代不清</b>，写成"the person"，核验模型以为说的是滑板少年 → 判引用相反。</p><p><b>后来：</b>wk5 规定边界写成"位于 t1 与 t2 之间"就算回答、答案逐字用"the person in black"；wk6 程序只用看得见的数、补一次逐帧比对。wk6 这道题报告 0 错、答案通过，一轮完成。</p>' }
    ]
  };

  /* ================= 脑图渲染 ================= */
  function nodeHTML(n) {
    var h = '';
    if (n.b) h += '<span class="nb ' + n.b[0] + '">' + esc(n.b[1]) + '</span>';
    h += '<span class="nt">' + esc(n.t) + '</span>';
    if (n.s) h += '<span class="ns">' + esc(n.s) + '</span>';
    if (n.th || n.img) h += '<img class="thumb" loading="lazy" src="' + IMG + (n.th || n.img) + '" alt="' + esc(n.t) + '">';
    return h;
  }

  function makeNode(n) {
    var node = el('div', 'mm-node ' + (n.root ? 'root ' : '') + (n.st || ''), nodeHTML(n));
    node.setAttribute('role', 'button');
    node.setAttribute('tabindex', '0');
    node.setAttribute('aria-label', n.t + (n.s ? '：' + n.s : ''));
    node._data = n;
    return node;
  }

  // side: '' = 向右展开；'l' = 向左展开（全景脑图左半边）
  function buildBranch(n, side) {
    var br = el('div', 'mm-branch' + (side ? ' ' + side : ''));
    var node = makeNode(n);
    br.appendChild(node);
    if (n.kids && n.kids.length) {
      var tog = el('button', 'tog', '−');
      tog.type = 'button';
      tog.setAttribute('aria-label', '收起或展开');
      node.appendChild(tog);
      var kids = el('div', 'mm-kids' + (side ? ' ' + side : ''));
      n.kids.forEach(function (k) { kids.appendChild(buildBranch(k, side)); });
      br.appendChild(kids);
    }
    return br;
  }

  function buildTwoSided(data) {
    var tree = el('div', 'mm mm-2');
    var half = Math.ceil(data.kids.length / 2);
    var left = el('div', 'mm-kids l side-root');
    var right = el('div', 'mm-kids side-root');
    data.kids.slice(0, half).forEach(function (k) { left.appendChild(buildBranch(k, 'l')); });
    data.kids.slice(half).forEach(function (k) { right.appendChild(buildBranch(k, '')); });
    tree.appendChild(left);
    tree.appendChild(makeNode(data));
    tree.appendChild(right);
    return tree;
  }

  function showDetail(panel, n) {
    var h = '';
    if (ST_TEXT[n.st]) h += '<span class="st ' + n.st + '">' + ST_TEXT[n.st] + '</span>';
    h += '<h4>' + esc(n.t) + '</h4>';
    if (n.s) h += '<p class="muted small">' + esc(n.s) + '</p>';
    if (n.d) h += n.d;
    if (n.img) h += '<img src="' + IMG + n.img + '" alt="' + esc(n.t) + '" data-cap="' + esc(n.cap || n.t) + '"><div class="cap">' + esc(n.cap || '') + ' 点图放大。</div>';
    if (!n.d && !n.img) h += '<p class="muted small">详细内容见对应章节。</p>';
    if (n.go) h += '<a class="jump go" href="' + n.go + '">跳到这一章 →</a>';
    panel.innerHTML = h;
  }

  function mountMap(shell, data, opts) {
    var scroll = shell.querySelector('.mm-scroll');
    var panel = shell.querySelector('.mm-detail');
    var tree;
    if (opts.layout === 'two') {
      tree = buildTwoSided(data);
    } else {
      tree = el('div', 'mm mm-v');
      tree.appendChild(buildBranch(data, ''));
    }
    scroll.appendChild(tree);

    function setCollapsed(br, on) {
      br.classList.toggle('collapsed', on);
      var t = br.querySelector(':scope > .mm-node > .tog');
      if (t) t.textContent = on ? '+' : '−';
    }

    scroll.addEventListener('click', function (ev) {
      var tog = ev.target.closest('.tog');
      if (tog) {
        ev.stopPropagation();
        var br = tog.closest('.mm-branch');
        setCollapsed(br, !br.classList.contains('collapsed'));
        return;
      }
      var node = ev.target.closest('.mm-node');
      if (!node || !node._data) return;
      scroll.querySelectorAll('.mm-node.sel').forEach(function (x) { x.classList.remove('sel'); });
      node.classList.add('sel');
      showDetail(panel, node._data);
      var r = panel.getBoundingClientRect();
      if (r.top > window.innerHeight - 80 || r.bottom < 60) {
        panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    });
    scroll.addEventListener('keydown', function (ev) {
      if ((ev.key === 'Enter' || ev.key === ' ') && ev.target.classList.contains('mm-node')) {
        ev.preventDefault(); ev.target.click();
      }
    });
    shell.querySelectorAll('[data-mm]').forEach(function (b) {
      b.addEventListener('click', function () {
        var expand = b.getAttribute('data-mm') === 'expand';
        scroll.querySelectorAll('.mm-branch').forEach(function (br) {
          if (!br.querySelector(':scope > .mm-kids')) return;
          // “收起”只保留前两层
          var depth = 0, p = br.parentElement;
          while (p && p !== scroll) { if (p.classList && p.classList.contains('mm-kids')) depth++; p = p.parentElement; }
          setCollapsed(br, !expand && depth >= (opts.collapseKeep || 1));
        });
      });
    });
    if (opts.select) {
      var first = scroll.querySelector('.mm-node.root') || scroll.querySelector('.mm-node');
      if (first) { first.classList.add('sel'); showDetail(panel, first._data); }
    }
  }

  /* ================= 进度条 ================= */
  var PIPE = [
    { n: '①', t: '任务', go: '#task', pct: 100, st: 'done', s: '题目与评分已定' },
    { n: '②', t: '智能体', go: '#agent', pct: 80, st: 'doing', s: '合并后掉分已修，复测中' },
    { n: '③', t: '成绩', go: '#score', pct: 55, st: 'doing', s: '本机 33.6%；服务器未跑' },
    { n: '④', t: '训练数据', go: '#data', pct: 45, st: 'doing', s: 'B 族稳定；新 4 类在跑' },
    { n: '⑤', t: '判对错', go: '#verify', pct: 55, st: 'doing', s: '可用；正式模型与金标缺' },
    { n: '⑥', t: '训练与进化', go: '#loop', pct: 10, st: 'todo', s: '只有设计和脚本' }
  ];
  function mountPipeline() {
    var box = document.getElementById('pipeline');
    if (!box) return;
    PIPE.forEach(function (p) {
      var a = el('a', 'pipe');
      a.href = p.go;
      a.innerHTML = '<span class="n">' + p.n + ' <span class="st ' + p.st + '" style="font-size:11px;padding:0 7px 0 5px">' + ST_TEXT[p.st] + '</span></span>' +
        '<span class="t">' + esc(p.t) + '</span>' +
        '<span class="bar"><i style="width:' + p.pct + '%;background:var(--' + p.st + ')"></i></span>' +
        '<span class="s">' + esc(p.s) + '</span>';
      box.appendChild(a);
    });
  }

  /* ================= 柱状图 ================= */
  function mountBars(id, rows) {
    var box = document.getElementById(id);
    if (!box) return;
    rows.forEach(function (r) {
      var row = el('div', 'brow');
      row.innerHTML = '<div class="lbl">' + esc(r.l) + (r.sub ? '<small>' + esc(r.sub) + '</small>' : '') + '</div>' +
        '<div class="track"><div class="fill ' + (r.k || '') + '" style="width:' + Math.max(r.v, 0.8) + '%"></div></div>' +
        '<div class="val">' + (r.approx ? '≈' : '') + r.v.toFixed(1) + '%</div>';
      box.appendChild(row);
    });
  }

  /* ================= 循环图 ================= */
  var LOOP = [
    { t: '① 学生跑调查', s: '每题采样 4 次', st: 'todo',
      d: '<p>用当前的学生模型（第一课之后的 9B）在练习题上自己查，每题跑 4 次；教师再跑 1 次，用来算"教师也犯不犯"。</p><p><b>现状：</b>还没有训练过的学生，所以这一环还没法开始。现在的数据都是教师或程序走的。</p>' },
    { t: '② 核验 → 记账', s: '每句、每步判对错', st: 'doing',
      d: '<p>核验器逐单元判断，结果记进"账本"：哪个角色（项目经理 / 调查员）、哪类单元（步骤 / 报告 / 答案）、哪种错误。</p><p><b>现状：</b>核验器可用（0.3.6）；账本还缺"谁产出的"（教师 / 学生第几轮）和"错误源头在哪个角色"两个字段。</p>' },
    { t: '③ 诊断', s: '谁错？什么错？', st: 'todo',
      d: '<p>把账本汇总成一张表：每一格（角色 × 单元 × 错误类型）的教师错误率、学生错误率、底座错误率，以及有多少是被上游带坏的。</p><p><b>现状：</b>diagnose.py 没写。</p>' },
    { t: '④ 路由', s: '改规则还是改权重？', st: 'todo',
      d: '<p>只看核验器可靠的错误类型。问两个问题：<b>是谁的错</b>（上游角色带坏的就改上游）；<b>教师也犯吗</b>（教师也犯 → 规则没写清，改规则；只有学生犯 → 学生没学会，改权重）。</p><p><b>现状：</b>route.py 没写。</p>' },
    { t: '⑤ 内环：改规则', s: '过门 A 才保留', st: 'doing',
      d: '<p>教师针对一格提出 ≤ 4 个规则改动（提示、参数），冻结模型权重，在选择题上跑，最好的那个再在确认题上复核。逐格进行，每轮最多约 5 格。</p><p><b>现状：</b>自动版没写；但 hi@0.2 → 0.10、gia@0.3 → 0.9 这些版本就是人工在做同样的事。</p>' },
    { t: '⑥ 外环：重新训练', s: '过门 B 才换模型', st: 'todo',
      d: '<p>用核验通过的学生自采数据（不够时教师从第一个错步接手一步）训练 LoRA。先固定项目经理训调查员，再固定调查员训项目经理。每个角色单独过门 B，过不了只回滚那个角色。</p><p>硬指标：第 2 轮起，训练数据里由教师产出的比例要低于一半。</p><p><b>现状：</b>训练脚本和 8 卡调度脚本已写好，一次真实训练都没跑。</p>' }
  ];
  var SVGNS = 'http://www.w3.org/2000/svg';
  function svg(tag, attrs, text) {
    var e = document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text != null) e.textContent = text;
    return e;
  }
  function mountLoop() {
    var root = document.getElementById('loop-svg');
    var panel = document.getElementById('loop-detail');
    if (!root) return;
    var cx = 300, cy = 290, R = 210, W = 196, H = 78;
    var defs = svg('defs');
    var mk = svg('marker', { id: 'arr', viewBox: '0 0 10 10', refX: '8', refY: '5', markerWidth: '7', markerHeight: '7', orient: 'auto-start-reverse' });
    mk.appendChild(svg('path', { d: 'M0,0 L10,5 L0,10 z', style: 'fill:var(--line-2)' }));
    defs.appendChild(mk);
    root.appendChild(defs);
    var pts = LOOP.map(function (_, i) {
      var a = (-90 + i * 60) * Math.PI / 180;
      return { x: cx + R * Math.cos(a), y: cy + R * Math.sin(a) };
    });
    // 弧线箭头
    pts.forEach(function (p, i) {
      var a0 = (-90 + i * 60 + 25) * Math.PI / 180, a1 = (-90 + (i + 1) * 60 - 25) * Math.PI / 180;
      var x0 = cx + R * Math.cos(a0), y0 = cy + R * Math.sin(a0), x1 = cx + R * Math.cos(a1), y1 = cy + R * Math.sin(a1);
      root.appendChild(svg('path', { d: 'M' + x0.toFixed(1) + ',' + y0.toFixed(1) + ' A' + R + ',' + R + ' 0 0 1 ' + x1.toFixed(1) + ',' + y1.toFixed(1), 'class': 'arrowline', 'marker-end': 'url(#arr)' }));
    });
    root.appendChild(svg('text', { x: cx, y: cy - 14, 'text-anchor': 'middle', 'class': 'lt' }, '每一轮'));
    root.appendChild(svg('text', { x: cx, y: cy + 8, 'text-anchor': 'middle', 'class': 'ctr' }, '只保留真的变好的改动'));
    root.appendChild(svg('text', { x: cx, y: cy + 28, 'text-anchor': 'middle', 'class': 'ctr' }, '教师占比逐轮下降'));
    var colors = { done: 'done', doing: 'doing', todo: 'todo', warn: 'warn' };
    pts.forEach(function (p, i) {
      var n = LOOP[i];
      var g = svg('g', { 'class': 'lnode', tabindex: '0', role: 'button', 'aria-label': n.t });
      g.appendChild(svg('rect', { x: (p.x - W / 2).toFixed(1), y: (p.y - H / 2).toFixed(1), width: W, height: H, rx: 12, style: 'fill:var(--surface);stroke:var(--' + colors[n.st] + ')' }));
      g.appendChild(svg('text', { x: p.x.toFixed(1), y: (p.y - 12).toFixed(1), 'text-anchor': 'middle', 'class': 'lt' }, n.t));
      g.appendChild(svg('text', { x: p.x.toFixed(1), y: (p.y + 8).toFixed(1), 'text-anchor': 'middle', 'class': 'ls' }, n.s));
      g.appendChild(svg('text', { x: p.x.toFixed(1), y: (p.y + 27).toFixed(1), 'text-anchor': 'middle', 'class': 'lst', style: 'fill:var(--' + colors[n.st] + ')' }, '● ' + ST_TEXT[n.st]));
      var pick = function () {
        root.querySelectorAll('.lnode.sel').forEach(function (x) { x.classList.remove('sel'); });
        g.classList.add('sel');
        panel.innerHTML = '<span class="st ' + n.st + '">' + ST_TEXT[n.st] + '</span><h4>' + esc(n.t) + '</h4>' + n.d;
      };
      g.addEventListener('click', pick);
      g.addEventListener('keydown', function (ev) { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); pick(); } });
      root.appendChild(g);
    });
  }

  /* ================= 名词 ================= */
  var GLOSS = [
    ['GIA（项目经理）', '决定这一轮查什么、开几个工作区；报告回来后判断证据够不够、写最终答案。'],
    ['HI（调查员）', '坐在某个工作区里一步步用工具查，最后交一份报告。所有工作区共用同一个模型。'],
    ['工作区 W1-1', '一个独立的小调查任务。W1-1 = 第 1 轮第 1 个。同一轮的工作区并行，轮与轮串行。'],
    ['E 编号 / R 编号', 'E 是证据库里的一条证据（帧、网页、比对结果……）；R 是搜索得到、还没打开的候选。答案里每句话都要带 E。'],
    ['EVID-Bench', '222 个被篡改视频的考卷，每题 3–5 个标准伪造点。正式报分用第 131–222 题。'],
    ['点级得分', '说中的标准点 ÷ 全部标准点。视频级 = 所有点都说中的视频占比。'],
    ['DETECT', '原来单独的"找原片并比对"工具（v2.4 最好）。10-12 起并进 SEARCH 和 INSPECT，不再单独存在。'],
    ['compare_media', '合并后的两段对比工具：对齐、找删减 / 插入 / 变速 / 镜像、区域差异、人脸是否被换、音轨比对。'],
    ['walker（程序走、教师填）', '造训练数据的方法：每一步由程序按规则决定，只有报告、答案、任务文字请教师大模型写。'],
    ['教师 / 学生', '教师是大模型（正式定为 Qwen3.5-122B；现在临时用 Codex / 27B），学生是要训练的 Qwen3.5-9B。'],
    ['r0e、wk1–wk8、evidC27G', '轮次名。r0* 是教师走的数据轮；wk* 是 walker 的轮；evid* 是 EVID 考试轮（F=原版，C=加找原片，M=合并后）。'],
    ['hi@0.8.0 / gia@0.9.0', '两个角色的"规格"版本：提示词、工具目录、运行参数打包在一起，有改动就升版本。'],
    ['核验器 verifier', '自动判对错的程序 + 模型。三层：程序检查、逐句对证据、要求覆盖够不够。当前 0.3.6。'],
    ['金标', '人工标注的"正确判断"，用来量核验器到底准不准。'],
    ['SFT / LoRA', 'SFT＝用示范数据做监督训练；LoRA＝只训练一小块附加参数，两个角色各一套。'],
    ['内环 / 外环', '内环改规则（提示、参数），模型不动；外环重新训练模型权重，规则不动。'],
    ['门 A / 门 B', '内环和外环各自的"验收门"：在没见过的题上真的变好，改动才保留。'],
    ['A / B / C / E / L 族', '造训练数据时的题型：A 上网找原片，B 给两段视频找不同，C 核实一句说法，E 图片最早出处，L 照片拍摄地。'],
    ['日期查询（SEARCH dates）', '对某个结果查页面发布日期、视频上传日期和最早存档日期，用来回答"谁先发布 / 是否早于声明"。'],
    ['核验后重写', '答案被核验判错时，把错误清单交给教师改一次再核验，只留通过的。'],
    ['content_match', 'SEARCH 搜到视频后程序自动和材料粗对，标出 source（≥30% 画面重合）/ partial / none。'],
    ['evidM27a / b / c / d', '工具合并后的 EVID 小测轮次：a 合并后，b 修复 1，c / d 修复 2（连跑两次取平均）。'],
    ['dev_train / dev_select / dev_confirm / dev_eval', '练习题的四份划分：训练用、挑改动用、复核用、最终评估用，互不泄漏。']
  ];
  function mountGloss() {
    var box = document.getElementById('gloss');
    if (!box) return;
    GLOSS.forEach(function (g) { box.appendChild(el('div', '', '<b>' + esc(g[0]) + '</b>' + esc(g[1]))); });
  }

  /* ================= 灯箱 ================= */
  function mountLightbox() {
    var lb = document.getElementById('lb'), img = document.getElementById('lb-img'), cap = document.getElementById('lb-cap');
    function close() { lb.classList.remove('open'); img.removeAttribute('src'); }
    document.addEventListener('click', function (ev) {
      var t = ev.target;
      if (t.tagName === 'IMG' && (t.closest('.fig') || t.closest('.mm-detail'))) {
        img.src = t.currentSrc || t.src;
        img.alt = t.alt || '';
        cap.textContent = t.getAttribute('data-cap') || t.alt || '';
        lb.classList.add('open');
      } else if (t === lb || t === img || t.id === 'lb-close') {
        close();
      }
    });
    document.addEventListener('keydown', function (ev) { if (ev.key === 'Escape' && lb.classList.contains('open')) close(); });
  }

  /* ================= 导航高亮 ================= */
  function mountScrollSpy() {
    var links = Array.prototype.slice.call(document.querySelectorAll('#nav a'));
    var map = {};
    links.forEach(function (a) { map[a.getAttribute('href').slice(1)] = a; });
    var secs = Array.prototype.slice.call(document.querySelectorAll('section.ch[id]'));
    if (!('IntersectionObserver' in window)) return;
    var visible = {};
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { visible[e.target.id] = e.isIntersecting ? e.intersectionRect.height : 0; });
      var best = null, bh = 0;
      secs.forEach(function (s) { if ((visible[s.id] || 0) > bh) { bh = visible[s.id]; best = s.id; } });
      links.forEach(function (a) { a.classList.remove('on'); });
      if (best && map[best]) {
        map[best].classList.add('on');
        var nav = document.getElementById('nav');
        var a = map[best];
        if (a.offsetLeft < nav.scrollLeft || a.offsetLeft + a.offsetWidth > nav.scrollLeft + nav.clientWidth) {
          nav.scrollTo({ left: a.offsetLeft - 20, behavior: 'smooth' });
        }
      }
    }, { rootMargin: '-70px 0px -35% 0px', threshold: [0, .1, .25, .5, .75, 1] });
    secs.forEach(function (s) { io.observe(s); });
  }

  /* ================= 深浅色 ================= */
  function mountTheme() {
    var btn = document.getElementById('theme-btn');
    var saved = null;
    try { saved = localStorage.getItem('fs-progress-theme'); } catch (e) { saved = null; }
    if (saved === 'light' || saved === 'dark') document.documentElement.setAttribute('data-theme', saved);
    if (!btn) return;
    btn.addEventListener('click', function () {
      var cur = document.documentElement.getAttribute('data-theme');
      var dark = cur ? cur === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
      var next = dark ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      try { localStorage.setItem('fs-progress-theme', next); } catch (e) { /* 忽略 */ }
    });
  }

  /* ================= 启动 ================= */
  function init() {
    mountTheme();
    mountPipeline();
    var shells = document.querySelectorAll('.mm-shell[data-map]');
    shells.forEach(function (sh) {
      var key = sh.getAttribute('data-map');
      if (key === 'overview') mountMap(sh, OVERVIEW, { layout: window.matchMedia('(max-width: 900px)').matches ? 'v' : 'two', select: true, collapseKeep: 1 });
      if (key === 'case205') mountMap(sh, CASE205, { layout: 'v', select: false, collapseKeep: 3 });
      if (key === 'case4126') mountMap(sh, CASE4126, { layout: 'v', select: false, collapseKeep: 2 });
    });
    mountBars('bars-total', [
      { l: '8B 原版', v: 8.5 },
      { l: '27B 原版', v: 15.9 },
      { l: '27B + 找原片比对', sub: 'DETECT v2.4 · 当前最好', v: 33.6, k: 'best' },
      { l: '论文 GPT-5.5', sub: '按这 91 题构成折算', v: 33.0, k: 'ref', approx: true },
      { l: '论文 Qwen3-VL-235B', sub: '按这 91 题构成折算', v: 24.2, k: 'ref', approx: true }
    ]);
    mountBars('bars-cat', [
      { l: 'AI 类 · 27B 原版', sub: '71 题', v: 1.9 },
      { l: 'AI 类 · + 找原片', v: 24.4, k: 'best' },
      { l: 'AI 类 · 论文最高', sub: 'Gemini-3.1-Pro', v: 18.1, k: 'ref' },
      { l: '剪辑类 · 27B 原版', sub: '20 题', v: 52.4 },
      { l: '剪辑类 · + 找原片', v: 57.3, k: 'best' },
      { l: '剪辑类 · 论文 GPT-5.5', sub: '按构成加权', v: 82.5, k: 'ref', approx: true }
    ]);
    mountLoop();
    mountGloss();
    mountLightbox();
    mountScrollSpy();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
