"use strict";
/* ForenSearcher 论文草稿页（方法与进展 → 论文草稿）。
   纯前端静态内容：不调用后端接口。中英两份正文共用同一套公式、图、参考文献。
   引用按正文首次出现的顺序自动编号；公式用 MathML，无外部依赖。 */
(() => {
  const P = window.ForenPaper = window.ForenPaper || {};
  P.lang = P.lang || {};

  /* ---------- 参考文献（统一英文著录） ---------- */
  P.refs = {
    newsclip:  'G. Luo, T. Darrell, A. Rohrbach. NewsCLIPpings: Automatic Generation of Out-of-Context Multimodal Media. EMNLP, 2021.',
    verite:    'S. Papadopoulos, C. Koutlis, S. Papadopoulos, P. Petrantonakis. VERITE: A Robust Benchmark for Multimodal Misinformation Detection Accounting for Unimodal Bias. IJMIR, 2024.',
    fakesv:    'P. Qi, Y. Bu, J. Cao, et al. FakeSV: A Multimodal Benchmark with Rich Social Context for Fake News Detection on Short Video Platforms. AAAI, 2023.',
    fakett:    'Y. Bu, Q. Sheng, J. Cao, et al. FakingRecipe: Detecting Fake News on Short Video Platforms from the Perspective of Creative Process. ACM MM, 2024.',
    sniffer:   'P. Qi, Z. Yan, W. Hsu, M. L. Lee. SNIFFER: Multimodal Large Language Model for Explainable Out-of-Context Misinformation Detection. CVPR, 2024.',
    averitec:  'M. Schlichtkrull, Z. Guo, A. Vlachos. AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web. NeurIPS Datasets and Benchmarks, 2023.',
    averimatec: 'R. Cao, et al. AVerImaTeC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web. 2025.',
    defame:    'T. Braun, M. Rothermel, M. Rohrbach, A. Rohrbach. DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts. ICML, 2025.',
    evid: 'EVID-Bench. arXiv:2606.04098, 2026.',
    groundlie: 'GroundLie360. ACM MM Datasets Track, 2025.',
    videosham: 'T. Mittal, R. Sinha, V. Swaminathan, J. Collomosse, D. Manocha. Video Manipulations Beyond Faces: A Dataset with Human-Machine Analysis. WACV Workshops, 2023.',
    searchr1:  'B. Jin, H. Zeng, Z. Yue, et al. Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning. arXiv:2503.09516, 2025.',
    webthinker:'X. Li, J. Jin, G. Dong, et al. WebThinker: Empowering Large Reasoning Models with Deep Research Capability. arXiv:2504.21776, 2025.',
    mmsearch: 'J. Wu, et al. MMSearch-R1: Incentivizing LMMs to Search. arXiv:2506.20670, 2025.',
    videoagent:'X. Wang, Y. Zhang, O. Zohar, S. Yeung-Levy. VideoAgent: Long-form Video Understanding with Large Language Model as Agent. ECCV, 2024.',
    harness1: 'Harness-1. arXiv:2606.02373, 2026.',
    planact:   'L. E. Erdogan, N. Lee, S. Kim, et al. Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks. ICML, 2025.',
    magentic:  'A. Fourney, G. Bansal, H. Mozannar, et al. Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks. arXiv:2411.04468, 2024.',
    mgrpo: 'M-GRPO. arXiv:2511.13288, 2025.',
    lightman:  'H. Lightman, V. Kosaraju, Y. Burda, et al. Let’s Verify Step by Step. ICLR, 2024.',
    mathshep:  'P. Wang, L. Li, Z. Shao, et al. Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations. ACL, 2024.',
    alce:      'T. Gao, H. Yen, J. Yu, D. Chen. Enabling Large Language Models to Generate Text with Citations. EMNLP, 2023.',
    carr: 'CaRR. arXiv:2601.06021, 2026.',
    rar:       'A. Gunjal, et al. Rubrics as Rewards: Reinforcement Learning beyond Verifiable Domains. arXiv:2507.17746, 2025.',
    rubicon: 'Reinforcement Learning with Rubric Anchors (Rubicon). arXiv:2508.12790, 2025.',
    arco: 'ARCO. arXiv:2606.21262, 2026.',
    reveal: 'REVEAL. arXiv:2608.08612, 2026.',
    whowhen:   'S. Zhang, M. Yin, J. Zhang, et al. Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems. ICML, 2025.',
    blame: 'Who is the Agent to Blame? arXiv:2608.24306, 2026.',
    mappa: 'MAPPA. arXiv:2601.23228, 2026.',
    dspy:      'O. Khattab, A. Singhvi, P. Maheshwari, et al. DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines. ICLR, 2024.',
    gepa:      'L. A. Agrawal, S. Tan, D. Soylu, et al. GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning. arXiv:2507.19457, 2025.',
    adas:      'S. Hu, C. Lu, J. Clune. Automated Design of Agentic Systems. ICLR, 2025.',
    dgm:       'J. Zhang, S. Hu, C. Lu, R. Lange, J. Clune. Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents. arXiv:2505.22954, 2025.',
    ace: 'Q. Zhang, et al. Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models. arXiv:2510.04618, 2025.',
    rbank: 'S. Ouyang, et al. ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory. arXiv:2509.25140, 2025.',
    metah: 'Meta-Harness. arXiv:2603.28052, 2026.',
    selfh: 'Self-Harness. arXiv:2606.09498, 2026.',
    rhi: 'Recursive Harness Self-Improvement (RHI). arXiv:2607.15524, 2026.',
    modrsi: 'ModularRSI. arXiv:2609.14857, 2026.',
    rrsi:      'RRSI: Regularized Recursive Self-Improvement of Agent Harnesses. arXiv:2609.24972, 2026.',
    star:      'E. Zelikman, Y. Wu, J. Mu, N. D. Goodman. STaR: Bootstrapping Reasoning with Reasoning. NeurIPS, 2022.',
    rest:      'C. Gulcehre, T. L. Paine, S. Srinivasan, et al. Reinforced Self-Training (ReST) for Language Modeling. arXiv:2308.08998, 2023.',
    restem:    'A. Singh, J. D. Co-Reyes, R. Agarwal, et al. Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models. TMLR, 2024.',
    seal:      'A. Zweiger, J. Pari, H. Guo, E. Akyürek, Y. Kim, P. Agrawal. Self-Adapting Language Models. arXiv:2506.10943, 2025.',
    coharness: 'Co-Harness. arXiv:2607.22688, 2026.',
    hase: 'HASE: Harness-Aware Self-Evolving. arXiv:2607.03935, 2026.',
    sia:       'SIA: Self-Improving AI with Harness and Weight Updates. arXiv:2605.27276, 2026.',
    hzero: 'Harness-Zero. arXiv:2609.24974, 2026.',
    agent0: 'P. Xia, et al. Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning. arXiv:2511.16043, 2025.',
    sage: 'SAGE. arXiv:2603.15255, 2026.',
    mae: 'Multi-Agent Evolve. arXiv:2510.23595, 2025.',
    futureb: 'FutureBridge. arXiv:2608.01953, 2026.',
    absk: 'ABSeeker. arXiv:2608.05102, 2026.',
    rsisurvey: 'Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops. arXiv:2607.07663, 2026.',
    aspire:    'Aspire: Can Models Self-Evolve from Vague Goals? arXiv:2608.31111, 2026.',
    s3gym:     'S³Gym: Can LLMs Turn Self-Testing and Self-Judging into Self-Improvement? arXiv:2608.31100, 2026.',
    harnessdev:'HarnessDev: Can LLMs Create and Evolve Their Own Agent Harness? arXiv:2609.01437, 2026.',
    inertia: 'Inertia. arXiv:2608.23045, 2026.',
    real: 'REAL. arXiv:2609.08943, 2026.',
    qwen35: 'Qwen Team. Qwen3.5 model family. 2026.',
    grpo:      'Z. Shao, P. Wang, Q. Zhu, et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.',
    dpo:       'R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, C. Finn. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. NeurIPS, 2023.',
    lora:      'E. J. Hu, Y. Shen, P. Wallis, et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022.'
  };

  /* ---------- MathML 小工具 ---------- */
  const mi = x => `<mi>${x}</mi>`, mo = x => `<mo>${x}</mo>`, mn = x => `<mn>${x}</mn>`, mt = x => `<mtext>${x}</mtext>`;
  const row = (...a) => `<mrow>${a.join('')}</mrow>`;
  const sub = (a, b) => `<msub>${a}${b}</msub>`, sup = (a, b) => `<msup>${a}${b}</msup>`, subsup = (a, b, c) => `<msubsup>${a}${b}${c}</msubsup>`;
  const frac = (a, b) => `<mfrac>${a}${b}</mfrac>`, under = (a, b) => `<munder>${a}${b}</munder>`;
  const cal = x => `<mi mathvariant="script">${x}</mi>`, rm = x => `<mi mathvariant="normal">${x}</mi>`;
  const sc = x => `<mtext class="pp-sc">${x}</mtext>`;
  const par = (...a) => row(mo('('), ...a, mo(')'));
  const block = (inner, n) => `<div class="pp-eq"><math display="block">${inner}</math><span class="pp-eqno">(${n})</span></div>`;
  const inl = inner => `<math>${inner}</math>\u2060`; // 词连接符：避免中文标点被挤到行首
  const cases = rows => row(mo('{'), `<mtable columnalign="left left">${rows.map(([a, b]) => `<mtr><mtd>${a}</mtd><mtd>${b}</mtd></mtr>`).join('')}</mtable>`);

  const piG = sub(mi('π'), rm('G')), piH = sub(mi('π'), rm('H'));
  const thG = sub(mi('θ'), rm('G')), thH = sub(mi('θ'), rm('H'));
  const sG = sub(mi('s'), rm('G')), sH = sub(mi('s'), rm('H'));
  const E = cal('E'), F = cal('F'), R = cal('R'), S = cal('S'), V = cal('V'), C = cal('C'), U = cal('U');

  P.m = { mi, mo, mn, mt, sub, sup, inl, cal, rm, row, par, frac,
    piG, piH, thG, thH, sG, sH, E, F, R, S, V };

  const EQT_EN = {chain:'chain', if:'if ', or:' or ', and:'  or  ', pending:'if any judgment is pending', otherwise:'otherwise', reroute:'re-route to ', origin:'origin', cost:'cost', accept:'accept ', on:'  on ', then:', then confirmed on ', end:'', teacher:' target of x produced by the teacher', sev:' sev'};
  P.makeEq = (T0 = {}) => { const T = Object.assign({}, EQT_EN, T0); return {
    gia: n => block(row(sub(mi('d'), mi('r')), mo('∼'), piG, par(mo('·'), mo('|'), sub(mi('V'), mi('r')), mo(';'), sG, mo(','), thG), mo(','), mspace(),
      sub(mi('d'), mi('r')), mo('∈'), row(mo('{'), sc('Investigate'), par(sub(mi('T'), mi('r'))), mo(','), sc('Finish'), par(mi('y')), mo('}'))), n),
    hi: n => block(row(sub(mi('a'), row(mi('r'), mo(','), mi('k'), mo(','), mi('t'))), mo('∼'), piH, par(mo('·'), mo('|'), sub(mi('L'), row(mi('r'), mo(','), mi('k'), mo(','), mi('t'))), mo(';'), sH, mo(','), thH), mo(','), mspace(),
      sub(mi('a'), row(mi('r'), mo(','), mi('k'), mo(','), mi('t'))), mo('∈'), row(mo('{'), sc('Search'), mo(','), sc('Inspect'), mo(','), sc('Read'), mo(','), sc('Query'), mo(','), sc('Report'), mo('}'))), n),
    answer: n => block(row(mi('y'), mo('='), subsup(row(mo('{'), par(sub(mi('σ'), mi('j')), mo(','), sub(C, mi('j'))), mo('}')), row(mi('j'), mo('='), mn('1')), mi('m')), mo(','), mspace(),
      sub(C, mi('j')), mo('⊆'), F, mo('∪'), E, mo(','), mspace(),
      mt(T.chain), mo(':'), mi('y'), mo('→'), F, mo('→'), E, mo('→'), R), n),
    disp: n => block(row(mi('δ'), par(mi('u')), mo('='), cases([
      [sc('neg'), row(mt(T.if), mo('∃'), mi('e'), mo('∈'), cal('Err'), par(mi('u')), mo(':'), mt(T.sev), par(mi('e')), mo('='), mt('critical'), mt(T.and), mo('|'), row(mo('{'), mi('e'), mo(':'), mt(T.sev), par(mi('e')), mo('='), mt('major'), mo('}')), mo('|'), mo('≥'), mn('2'))],
      [sc('unc'), mt(T.pending)],
      [sc('pos'), mt(T.otherwise)]
    ])), n),
    clue: n => block(row(sub(mi('g'), mi('t')), mo('='), mo('|'), par(subsup(E, mi('t'), mt('vis')), mo('∖'), subsup(E, row(mi('t'), mo('−'), mn('1')), mt('vis'))), mo('∩'), sup(U, mo('⋆')), mo('|'), mo(','), mspace(),
      sup(U, mo('⋆')), mo('='), sub(mt('cl'), mt('derived')), par(sub(E, mt('req')), mo('∪'), under(mo('⋃'), row(mi('f'), mo(':'), mi('δ'), par(mi('f')), mo('='), sc('pos'))), mt('cite'), par(mi('f'))), mo(','), mspace(),
      sub(mi('w'), mi('t')), mo('='), mi('min'), par(mn('1'), mo('+'), sub(mi('g'), mi('t')), mo(','), mn('3'))), n),
    route: n => block(row(mi('λ'), par(mi('c')), mo('='), cases([
      [row(mt(T.reroute), mt(T.origin), par(mi('c'))), row(mt(T.if), sub(mi('u'), mi('c')), mo('≥'), mn('0.5'))],
      [sc('outer'), row(mt(T.if), sub(mi('s'), mi('c')), mo('−'), sub(mi('t'), mi('c')), mo('≥'), mi('Δ'))],
      [sc('inner'), row(mt(T.if), sub(mi('t'), mi('c')), mo('≥'), mn('0.5'), sub(mi('s'), mi('c')), mt(T.or), sub(mi('t'), mi('c')), mo('≥'), mi('τ'))],
      [sc('observe'), mt(T.otherwise)]
    ])), n),
    cf: n => block(row(subsup(mi('λ'), mi('c'), mo('⋆')), mo('='), under(mi('arg max'), row(mi('ℓ'), mo('∈'), row(mo('{'), sc('inner'), mo(','), sc('outer'), mo('}')))),
      frac(row(sub(mi('e'), mi('c')), par(mi('before')), mo('−'), sub(mi('e'), mi('c')), par(mi('ℓ'))), row(mt(T.cost), par(mi('ℓ')))), mo(','), mspace(),
      mt('Acc'), mo('='), frac(mn('1'), row(mo('|'), C, mo('|'))), under(mo('∑'), row(mi('c'), mo('∈'), C)), mn('𝟙'), row(mo('['), mi('λ'), par(mi('c')), mo('='), subsup(mi('λ'), mi('c'), mo('⋆')), mo(']'))), n),
    gateA: n => block(row(mt(T.accept), sup(mi('s'), mo('′')), mspace(), mo('⟺'), mspace(),
      sub(mi('Δ'), mt('tgt')), mo('≤'), mo('−'), sub(mi('m'), mi('α')), mo(','), mspace(), sub(mi('Δ'), mt('other')), mo('≤'), mi('ε'), mo(','), mspace(), sub(mi('Δ'), mt('native')), mo('≥'), mo('−'), mi('ε'),
      mt(T.on), sc('dev_select'), mt(T.then), sc('dev_confirm'), mt(T.end)), n),
    dep: n => block(row(sub(mi('D'), mi('n')), mo('='), frac(row(mo('|'), row(mo('{'), mi('x'), mo('∈'), sub(cal('D'), mi('n')), mo(':'), mt(T.teacher), mo('}')), mo('|')), row(mo('|'), sub(cal('D'), mi('n')), mo('|')))), n),
    syn: n => block(row(mi('I'), mo('='), mi('P'), par(sub(mi('G'), mi('k')), mo(','), sub(mi('H'), mi('k'))), mo('−'), mi('P'), par(sub(mi('G'), mi('k')), mo(','), sub(mi('H'), mn('0'))), mo('−'), mi('P'), par(sub(mi('G'), mn('0')), mo(','), sub(mi('H'), mi('k'))), mo('+'), mi('P'), par(sub(mi('G'), mn('0')), mo(','), sub(mi('H'), mn('0')))), n),
    obj: n => block(row(under(mi('max'), row(mi('s'), mo(','), mi('θ'))), mspace(), mi('J'), par(mi('s'), mo(','), mi('θ')), mo('='), sub(mi('𝔼'), row(mi('x'), mo('∼'), sub(cal('D'), mt('eval')))), row(mo('['), mt('Task'), par(mi('y'), mo(','), mi('x')), mo('+'), mi('β'), mo('·'), mt('Chain'), par(mi('y'), mo(','), S), mo(']'))), n)
  }; };
  function mspace() { return '<mspace width="1em"></mspace>'; }

  /* ---------- 图 ---------- */
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  function tx(x, y, s, cls = '', anchor = 'start') { return `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`; }
  function box(x, y, w, h, cls, rx = 10) { return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" class="${cls}"/>`; }
  function arrow(d, cls = 'pf-arr', label = '', lx = 0, ly = 0, anchor = 'middle') {
    return `<path d="${d}" class="${cls}" marker-end="url(#pf-ah)"/>${label ? `<text x="${lx}" y="${ly}" class="pf-alabel" text-anchor="${anchor}">${esc(label)}</text>` : ''}`;
  }
  const defs = `<defs><marker id="pf-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" class="pf-ahead"/></marker></defs>`;

  P.figArch = L => `<svg viewBox="0 0 906 452" class="pp-svg" role="img" aria-label="${esc(L.aria)}">${defs}
    ${box(10, 44, 160, 152, 'pf-in')}${tx(24, 70, L.caseT, 'pf-h')}${tx(24, 96, L.case1, 'pf-s')}${tx(24, 118, L.case2, 'pf-s')}${tx(24, 140, L.case3, 'pf-s')}${tx(24, 180, L.case4, 'pf-m')}
    ${box(196, 44, 214, 152, 'pf-gia')}${tx(210, 70, 'GIA', 'pf-h pf-gia-t')}${tx(250, 70, L.giaT, 'pf-s')}${tx(210, 96, L.gia1, 'pf-s')}${tx(210, 118, L.gia2, 'pf-s')}${tx(210, 140, L.gia3, 'pf-s')}${tx(210, 180, L.gia4, 'pf-m')}
    ${box(446, 36, 440, 206, 'pf-round', 14)}${tx(462, 60, L.round, 'pf-h')}${tx(462, 79, L.roundR, 'pf-m')}
    ${[0, 1, 2].map(i => { const x = 460 + i * 142; return `${box(x, 88, 132, 124, 'pf-hi')}${tx(x + 12, 110, `W r-${i + 1} · HI`, 'pf-h pf-hi-t')}${tx(x + 12, 132, L.hi1, 'pf-s')}${tx(x + 12, 150, L.hi2, 'pf-s')}${tx(x + 12, 172, 'SEARCH', 'pf-tool')}${tx(x + 12, 188, 'INSPECT ×8', 'pf-tool')}${tx(x + 12, 204, L.hi4, 'pf-m')}`; }).join('')}
    ${[0, 1].map(i => { const x = 598 + i * 142; return `<line x1="${x}" y1="92" x2="${x}" y2="208" class="pf-wall"/>`; }).join('')}
    ${tx(872, 79, L.wall, 'pf-alabel pf-wall-t', 'end')}
    ${box(196, 300, 690, 140, 'pf-store', 14)}${tx(212, 324, L.store, 'pf-h')}
    ${box(212, 338, 196, 88, 'pf-r')}${tx(226, 362, L.rT, 'pf-s pf-b')}${tx(226, 384, L.r1, 'pf-m')}${tx(226, 404, L.r2, 'pf-m')}
    ${box(422, 338, 240, 88, 'pf-e')}${tx(436, 362, L.eT, 'pf-s pf-b')}${tx(436, 384, L.e1, 'pf-m')}${tx(436, 404, L.e2, 'pf-m')}
    ${box(676, 338, 196, 88, 'pf-f')}${tx(690, 362, L.fT, 'pf-s pf-b')}${tx(690, 384, L.f1, 'pf-m')}${tx(690, 404, L.f2, 'pf-m')}
    ${box(10, 300, 160, 140, 'pf-out')}${tx(24, 326, L.ansT, 'pf-h')}${tx(24, 352, L.ans1, 'pf-s')}${tx(24, 372, L.ans2, 'pf-s')}${tx(24, 414, 'y → F → E → R', 'pf-s pf-b')}
    ${arrow('M170 120 H192')}
    ${arrow('M410 120 H442', 'pf-arr pf-arr-gia', L.aTasks, 426, 112)}
    ${arrow('M560 212 V334', 'pf-arr', L.aWrite, 566, 290, 'start')}
    ${arrow('M520 334 V216', 'pf-arr pf-arr-read', L.aRead, 514, 290, 'end')}
    ${arrow('M818 212 V334', 'pf-arr pf-arr-hi', L.aRep, 824, 290, 'start')}
    ${arrow('M872 382 H896 V18 H303 V40', 'pf-arr pf-arr-gia', L.aF, 600, 31)}
    ${arrow('M250 196 V262 H90 V296', 'pf-arr pf-arr-gia', L.aFin, 256, 240, 'start')}
  </svg>`;

  P.figLoop = L => `<svg viewBox="0 0 926 572" class="pp-svg" role="img" aria-label="${esc(L.aria)}">${defs}
    ${box(32, 12, 872, 88, 'pf-anchor', 12)}${tx(48, 38, L.anc, 'pf-h')}${tx(48, 62, L.anc1, 'pf-s')}${tx(48, 84, L.ancR + '  ·  ' + L.ancR2, 'pf-m')}
    ${box(32, 124, 190, 118, 'pf-in')}${tx(46, 150, L.run, 'pf-h')}${tx(46, 176, L.run1, 'pf-s')}${tx(46, 198, L.run2, 'pf-s')}
    ${box(246, 124, 190, 118, 'pf-ver')}${tx(260, 150, L.ver, 'pf-h')}${tx(260, 176, L.ver1, 'pf-s')}${tx(260, 198, L.ver2, 'pf-s')}
    ${box(460, 124, 190, 118, 'pf-led')}${tx(474, 150, L.led, 'pf-h')}${tx(474, 176, L.led1, 'pf-s')}${tx(474, 198, L.led2, 'pf-s')}
    ${box(674, 124, 230, 118, 'pf-rou')}${tx(688, 150, L.rou, 'pf-h')}${tx(688, 172, L.rou0, 'pf-m')}${tx(688, 190, L.rou1, 'pf-m')}${tx(688, 208, L.rou2, 'pf-m')}${tx(688, 230, L.rou4, 'pf-m pf-b')}
    ${arrow('M222 183 H242')}${arrow('M436 183 H456')}${arrow('M650 183 H670')}
    ${arrow('M341 100 V120', 'pf-arr pf-arr-anc')}
    <path d="M789 242 V262 M248 262 H789" class="pf-arr pf-arr-dash"/>
    ${arrow('M248 262 V286', 'pf-arr pf-arr-dash', L.aIn, 256, 280, 'start')}
    ${arrow('M688 262 V286', 'pf-arr pf-arr-dash', L.aOut, 696, 280, 'start')}
    ${box(32, 290, 424, 150, 'pf-l2')}${tx(46, 316, L.l2, 'pf-h')}${tx(46, 342, L.l2a, 'pf-s')}${tx(46, 364, L.l2b, 'pf-s')}${tx(46, 386, L.l2c, 'pf-s')}${tx(46, 424, L.l2d, 'pf-m')}
    ${box(480, 290, 424, 150, 'pf-l3')}${tx(494, 316, L.l3, 'pf-h')}${tx(494, 342, L.l3a, 'pf-s')}${tx(494, 364, L.l3b, 'pf-s')}${tx(494, 386, L.l3c, 'pf-s')}${tx(494, 424, L.l3d, 'pf-m')}
    ${arrow('M248 440 V472')}${arrow('M688 440 V472')}
    ${box(32, 476, 872, 84, 'pf-eval')}${tx(48, 502, L.ev, 'pf-h')}${tx(48, 526, L.ev1, 'pf-s')}${tx(48, 548, L.ev2, 'pf-m')}
    ${arrow('M32 518 H14 V183 H28', 'pf-arr pf-arr-next')}
    <text transform="translate(10 350) rotate(-90)" class="pf-alabel" text-anchor="middle">${esc(L.aNext)}</text>
  </svg>`;

  /* ---------- 渲染 ---------- */
  P.render = function (lang) {
    const L = P.lang[lang] || P.lang.zh;
    const order = [], toc = [];
    const cite = (...keys) => '<span class="pp-cite">[' + keys.map(k => {
      if (!P.refs[k]) return '?';
      let i = order.indexOf(k); if (i < 0) { order.push(k); i = order.length - 1; }
      return `<span title="${esc(P.refs[k])}">${i + 1}</span>`;
    }).join(', ') + ']</span>';
    const sec = (num, title, id) => { toc.push({ num, title, id, lvl: 2 }); return `<h2 class="pp-h2" id="pp-${id}"><span>${num}</span>${title}</h2>`; };
    const subsec = (num, title, id) => { toc.push({ num, title, id, lvl: 3 }); return `<h3 class="pp-h3" id="pp-${id}"><span>${num}</span>${title}</h3>`; };
    const ctx = { c: cite, sec, sub: subsec, eq: P.makeEq(L.eqText), m: P.m, figArch: P.figArch, figLoop: P.figLoop };
    const body = L.body(ctx);
    const refs = `<h2 class="pp-h2" id="pp-refs">${L.refsTitle}</h2><ol class="pp-refs">${order.map(k => `<li>${esc(P.refs[k])}</li>`).join('')}</ol>`;
    toc.push({ num: '', title: L.refsTitle, id: 'refs', lvl: 2 });
    const tocHtml = `<nav class="pp-toc" aria-label="${L.tocLabel}"><div class="pp-toc-k">${L.tocLabel}</div>${toc.map(t => `<button type="button" class="pp-toc-${t.lvl}" data-paper-jump="pp-${t.id}"><span>${t.num}</span>${t.title}</button>`).join('')}</nav>`;
    return { toc: tocHtml, body: body + refs, L };
  };
})();
/* 论文草稿 · English */
(() => {
  const P = window.ForenPaper;
  const FA = {
    aria: 'ForenSearcher architecture: GIA plans dynamic rounds of parallel HI workspaces over a shared evidence store; findings flow only to GIA.',
    caseT: 'Case input', case1: 'claim C0 · task q', case2: 'video / image / page', case3: 'registered as E', case4: 'original materials M',
    giaT: 'global planner', gia1: 'Global View (5 blocks)', gia2: '+ key frames of M', gia3: 'READ_RECORD on demand', gia4: 'INVESTIGATE {τ} | FINISH y',
    round: 'Dynamic Investigation Round r', roundR: 'parallel within · sequential across',
    hi1: 'task τ only', hi2: 'local memory L', hi3: '+ pixels of E', hi4: 'REPORT: status + F',
    wall: 'no sibling findings or reports',
    store: 'Global Store  (one copy of every record)',
    rT: 'R · candidate sources', r1: 'titles, snippets, links', r2: 'leads only, never evidence',
    eT: 'E · evidence', e1: 'material actually read / tool output', e2: 'shared across workspaces',
    fT: 'F · findings', f1: 'judgments citing E', f2: 'returned to GIA only',
    ansT: 'Final answer', ans1: 'every sentence cites', ans2: 'F / E ids',
    aTasks: 'tasks', aWrite: 'write E', aRead: 'read E', aRep: 'report', aF: 'findings → GIA', aFin: 'FINISH'
  };
  const FL = {
    aria: 'Verification-Routed Self-Improvement: run, verify, ledger, route to inner loop or outer loop, gate, evaluate.',
    anc: 'External anchors', anc1: 'dataset labels · original sources · per-case dossier (requirements, necessary evidence)',
    ancR: 'verifier 𝒱 (local, ≠ teacher) · 𝒱ref frozen, other family', ancR2: 'precision: human gold · recall: error injection',
    run: 'Run', run1: 'student system on dev_train', run2: 'N per case + teacher probe',
    ver: 'Verify every unit', ver1: 'HI step · HI report', ver2: 'GIA plan · GIA answer',
    led: 'Error ledger', led1: 'role × unit × error type', led2: 'teacher / student rate, origin',
    rou: 'Router (two-question prior)', rou0: 'eligible only if 𝒱 is calibrated for the type', rou1: '① whose error is it?', rou2: '② does the teacher fail too?', rou4: 'prior, scored counterfactually',
    l2: 'Lever 1 · Inner loop (θ frozen)', l2a: 'edit one spec object: prompt · params · teacher instr.', l2b: 'K ≤ 4 candidates → regression check → Gate A', l2c: 'best one confirmed on dev_confirm', l2d: 'one accepted change per round',
    l3: 'Lever 2 · Outer loop (s frozen)', l3a: 'GIA@n tasks → train HI → Gate B-HI', l3b: 'HI@n+1 reports → train GIA → Gate B-GIA', l3c: 'self-sampling + teacher bridging (partner-consistent data)', l3d: 'teacher dependence Dn ↓',
    aIn: 'teacher fails too → spec', aOut: 'capability gap → weights',
    ev: 'Gates → next round', ev1: 'Gate A: spec only · Gate B-HI / B-GIA: one role\u2019s weights · a failed gate rolls back only that change', ev2: 'curves only on dev_eval (never used for decisions) · test locked · recorded retrieval replay',
    aNext: 'next round  (s, θ, 𝒱)'
  };

  P.lang.en = {
    label: 'English',
    title: 'ForenSearcher: Evidence-Anchored, Verification-Routed Self-Improvement of a Two-Role Multimodal Investigation Agent',
    authors: 'Anonymous CVPR submission',
    draft: 'Working draft · Sections 1–3 (Introduction, Related Work, Method). Experiments are being run and will be added; numbers marked † come from the round-0 pilot and serve only as illustration.',
    abstractTitle: 'Abstract',
    tocLabel: 'Contents',
    refsTitle: 'References',
    abstract: `Verifying a misleading video is an investigation, not a classification: an analyst breaks the claim into checkable questions, retrieves candidate sources, inspects frames and pages, compares media, and writes a conclusion in which every sentence can be traced back to material that was actually read. We present <b>ForenSearcher</b>, a multimodal investigation agent built around two learned roles. A <i>Global Investigation Agent</i> (GIA) plans dynamic rounds of parallel sub-investigations and decides when the evidence suffices; a <i>Harness Investigator</i> (HI) carries out each sub-investigation with search and media-inspection tools inside an isolated workspace. A persistent store shares what has been read across workspaces while keeping each workspace’s conclusions private, so every answer forms an auditable chain <i>answer → finding → evidence → source</i>. On top of this architecture we propose <b>Verification-Routed Self-Improvement</b> (VRSI), a bounded self-improvement loop designed around three recent negative results on recursive self-improvement. Instead of self-assessment, every decision unit—each HI action, each workspace report, each GIA plan and each final answer—is verified against external anchors (dataset labels, original sources and a per-case dossier) and labeled with one error taxonomy attributed to the responsible role. The resulting error ledger is routed by a two-question prior—which role caused the error, and does the teacher make it too—to one of two levers: evolving the harness specification with weights frozen, or retraining one role with the specification frozen; in the training loop each role is trained on data produced with the other role as its current partner. The routing rule is treated as a falsifiable prior and scored against counterfactual double interventions, and a guarded protocol—five-way splits, recorded retrieval replay, one accepted change per gate—specifies how to tell whether the loop improves the system rather than merely changing it.`,
    body: h => `
${h.sec('1', 'Introduction', 'intro')}
<p>A clip of a collapsing bridge circulates with the caption “filmed in Chengdu last week”. Whether the caption is true seldom hinges on pixels alone. A professional fact-checker extracts key frames, runs a reverse image search, opens the earliest upload she can find, aligns it frame by frame with the circulating copy, looks up reports of the claimed event, and only then writes a verdict that cites the pages and frames supporting each statement. The same repertoire of actions—retrieving sources, localizing segments, comparing media, attributing edits, checking claims—recurs across tasks that benchmarks treat separately. This paper asks two questions: how to build an agent that performs such investigations end to end with auditable evidence, and how such an agent can improve itself without drifting away from what is actually true.</p>

<p>Existing work covers pieces of the problem. Multimodal misinformation detectors predict out-of-context or fake labels for image–caption pairs and short videos ${h.c('newsclip', 'verite', 'fakesv', 'fakett')}, and recent multimodal LLMs add explanations ${h.c('sniffer')}, but none of them leaves the given inputs to establish provenance. Evidence-based fact-checking ${h.c('averitec', 'averimatec', 'defame')} retrieves web evidence, mostly for text or image claims. Deep-research agents ${h.c('searchr1', 'webthinker', 'mmsearch')} search the open web but keep one monolithic context that cannot carry video frames through dozens of steps, and they seldom check whether the sources they cite support what they write. New benchmarks for video misinformation ${h.c('evid', 'groundlie')} make the gap explicit: solving them requires external retrieval, cross-video comparison and grounding of the manipulated segments.</p>

<p>Two obstacles stand in the way. <b>The first is architectural.</b> An investigation over video overflows any single context. Splitting it into a planner and executors helps, but naïve splitting lets an unverified conclusion from one branch enter another branch as a premise, and it blurs responsibility: when the final answer is wrong, one cannot tell whether the plan or the execution failed. <b>The second is learning.</b> There is no large corpus of expert investigation trajectories, so the agent must start from teacher demonstrations and then improve on its own. Recursive self-improvement (RSI) has become the natural framing, either by evolving the harness around frozen weights ${h.c('metah', 'rhi', 'modrsi')}, by training on self-generated and filtered data ${h.c('star', 'rest')}, or by combining both ${h.c('coharness', 'hase', 'sia')}. Three recent studies, however, temper the optimism: models allowed to choose their own training seldom surpass the base model and often mistake recovery from a self-inflicted regression for progress ${h.c('aspire')}; the ability to test and judge oneself does not turn into improvement ${h.c('s3gym')}; and only 34 of 64 self-evolved harness edits kept their direction on held-out tasks, while many evolved components were never exercised ${h.c('harnessdev')}. We read these failures as sharing two root causes. The improvement signal is not anchored to anything outside the model, and nothing decides <i>what</i> should change in response to a given failure—a prompt, the weights of some component, or the judge itself.</p>

<p><b>Fact-checking supplies both missing pieces.</b> First, it has external anchors. Dataset labels, the original sources, and a per-case dossier listing what must be established and which evidence is necessary allow intermediate claims to be checked against material the agent actually retrieved, rather than against the model’s opinion of its own output. Only labels are ground truth; process judgments are made by a verifier model that is anchored to these materials, and we measure its precision and recall on human annotations instead of assuming them. Second, investigation has an attributable structure. When planning and execution are two roles with explicit interfaces, each decision can be judged relative to the role’s own input and each error can be charged to the role that caused it, which in turn indicates what should be changed.</p>

<p>ForenSearcher builds on these two observations (Fig. 1). The GIA sees a compact global view together with key frames of the original media, dispatches independent tasks for the current round, and either finishes with a cited answer or opens another round. Each task runs in its own workspace, where the HI—one shared policy across workspaces—chooses one action per step among <span class="pp-tt">SEARCH</span>, <span class="pp-tt">INSPECT</span> and two read-only memory actions, and finally submits a report whose findings cite evidence identifiers. Evidence is stored once and shared; findings are returned only to the GIA. On top of this system, VRSI (Fig. 2) closes the loop. An evidence-anchored verifier judges four decision units, records typed errors in a ledger, and a two-question router assigns each frequent error cell either to a harness-only inner loop or to a weight-only outer loop in which the GIA and the HI take turns being trained on data produced with the other role as partner. The verifier is not a lever inside this loop: it is calibrated out of band, and only error types it is calibrated for may be routed. Every change passes a gate that isolates it, and all curves are reported on a split that never informs a decision.</p>

<p>Our contributions are:</p>
<ul class="pp-contrib">
<li><b>A two-role multimodal investigation architecture</b> with dynamic rounds and record semantics—candidate source, evidence, finding—that share what was read but isolate what was concluded. Every answer sentence is traceable to source material, and every error is attributable to a role (Sec. 3.2).</li>
<li><b>Evidence-anchored, unit-level verification.</b> A three-tier verifier (executable checks, quote-grounded partial-support judgments with provenance scoping, and dossier-based sufficiency) covers all four decision units of both roles under a single error taxonomy, with role-relative attribution and hindsight credit, and is calibrated for precision on human gold labels and for recall by programmatic error injection (Sec. 3.3).</li>
<li><b>Verification-Routed Self-Improvement</b>, a dual-loop, dual-role, bounded RSI procedure. A role-attributed error ledger is routed to the harness or to the weights by an explicit two-question prior; the two roles serve as each other’s on-policy data generators; and the prior is scored against counterfactual double interventions rather than assumed (Sec. 3.4–3.6).</li>
<li><b>A guarded evaluation protocol</b> that turns the three negative findings into measured quantities: a dose–response relation between verifier score and held-out gain, the direction consistency of accepted edits, a teacher-dependence curve, and a cross-pairing synergy term, all under five-way data splits and recorded retrieval replay (Sec. 3.7).</li>
</ul>

${h.sec('2', 'Related Work', 'related')}
<p><b>Multimodal misinformation and fact-checking.</b> Out-of-context misinformation was first cast as classification of image–caption pairs, either synthesized by caption swapping ${h.c('newsclip')} or curated from fact-checking articles ${h.c('verite')}; short-video datasets add audio, comments and publisher metadata ${h.c('fakesv', 'fakett')}. Detectors evolved from multimodal fusion to MLLM explainers ${h.c('sniffer')}. Evidence-based verification was formalized for real-world text claims with web evidence ${h.c('averitec')} and extended to image–text claims ${h.c('averimatec')}; DEFAME ${h.c('defame')} plans retrieval and calls multimodal experts. Most recently, EVID-Bench ${h.c('evid')} and GroundLie360 ${h.c('groundlie')} require search-grounded analysis of videos, cross-video comparison and grounding of forged segments, and manipulation datasets such as VideoSham ${h.c('videosham')} target edits beyond faces. We share this evidence-seeking view but differ in two ways: one agent covers heterogeneous video verification tasks, and every answer sentence must be traceable through persistent records. The second property is what makes process-level verification, and therefore self-improvement, possible. Forensic detectors for synthetic content are complementary; our tool interface reserves a slot for them but does not rely on one.</p>

<p><b>Deep-research and multimodal search agents.</b> Search agents trained with reinforcement learning interleave reasoning and retrieval ${h.c('searchr1', 'webthinker')}, and multimodal variants decide when and how to search with images ${h.c('mmsearch')}. Video agents iteratively query frames to answer questions about a given video ${h.c('videoagent')}, but do not leave that video to establish where it came from. Harness-1 ${h.c('harness1')} shows the value of an environment that owns working memory (deduplication, compression, curated evidence) while the model makes semantic decisions; we adopt this principle and re-implement it for multiple workspaces, a shared multimodal store and video tools. Planner–executor systems ${h.c('planact', 'magentic')} and hierarchical multi-agent RL for deep research ${h.c('mgrpo')} also separate global planning from local execution. Our GIA/HI split differs in its information contract. Evidence persists and is shared across workspaces, whereas findings return only to the planner and never to sibling workspaces; rounds are decided on the fly instead of being planned once; and tasks may not carry unverified conclusions as premises. These constraints are what make role-level error attribution well defined.</p>

<p><b>Process verification and credit assignment.</b> Process reward models and step-level verifiers ${h.c('lightman', 'mathshep')} improve reasoning where correctness can be checked. For attributed generation and search agents, citation quality ${h.c('alce')}, citation-aware rubric rewards ${h.c('carr')}, instance-specific rubrics ${h.c('rar', 'rubicon')}, prefix-conditioned step criteria ${h.c('arco')} and sufficiency gates ${h.c('reveal')} test whether claims are supported and whether search should stop. In multi-agent systems, failure attribution asks which agent and which step caused a failure ${h.c('whowhen', 'blame')}, and per-role local verification enables critic-free process RL ${h.c('mappa')}. We merge these lines into one verifier. Its judgments are anchored to the exact evidence records a claim cites (text or pixels) and checked by programmatic quote matching; it covers four decision units of two roles with one taxonomy; and it is calibrated for both precision and recall. Calibration is the prerequisite for using verification as the learning signal of a self-improvement loop.</p>

<p><b>Recursive self-improvement of agents.</b> Training-free approaches optimize prompts and programs ${h.c('dspy', 'gepa')}, search over agent designs ${h.c('adas', 'dgm')}, grow experience playbooks ${h.c('ace', 'rbank')}, or evolve the harness itself ${h.c('metah', 'selfh', 'rhi', 'modrsi', 'rrsi')}; among the lessons are that module-wise edits beat joint edits ${h.c('modrsi')} and that edit budgets and pruning curb overfitting ${h.c('rrsi')}. Training-based approaches filter self-generated trajectories and fine-tune on them ${h.c('star', 'rest', 'restem')} or let the model write its own fine-tuning data ${h.c('seal')}. Joint approaches alternate harness diffs and supervised fine-tuning with failure attribution by type ${h.c('coharness')}, co-train harness edits and solutions with RL ${h.c('hase')}, update both through a feedback agent ${h.c('sia')}, or distill an evolved harness back into the weights ${h.c('hzero')}. Multi-model co-evolution pairs a curriculum or challenger agent with a solver ${h.c('agent0', 'sage', 'mae')}. A recent survey ${h.c('rsisurvey')} separates <i>bounded</i> RSI, which converges toward a fixed external standard, from open-ended RSI, and names self-evaluation as the load-bearing pillar; three studies report where current RSI fails ${h.c('aspire', 's3gym', 'harnessdev')}. We position VRSI as bounded RSI and differ from prior work in four respects (Tab. 1): the signal is verification anchored to external material rather than self-judgment or pairwise preference; two roles and two levers sit in one loop, joined by an explicit and falsifiable routing rule; the two roles generate each other’s training data; and the protocol measures the quantities the negative results call into question.</p>

<div class="pp-table-wrap"><table class="pp-table">
<caption><b>Table 1.</b> Positioning among self-improving agent methods. <i>Signal</i>: what decides whether a change is kept. <i>Lever choice</i>: how the method decides whether a failure is fixed in the harness or in the weights.</caption>
<thead><tr><th>Method</th><th>Harness</th><th>Weights</th><th>Roles</th><th>Signal</th><th>Lever choice</th></tr></thead>
<tbody>
<tr><td>ModularRSI / RRSI ${h.c('modrsi', 'rrsi')}</td><td>✓</td><td>–</td><td>1</td><td>benchmark score, critic</td><td>–</td></tr>
<tr><td>RHI ${h.c('rhi')}</td><td>✓</td><td>–</td><td>many</td><td>pairwise LLM judgment</td><td>–</td></tr>
<tr><td>STaR / ReST ${h.c('star', 'rest')}</td><td>–</td><td>✓</td><td>1</td><td>answer correctness</td><td>–</td></tr>
<tr><td>Co-Harness ${h.c('coharness')}</td><td>✓</td><td>✓</td><td>1</td><td>LLM critic on failures</td><td>failure-type attribution</td></tr>
<tr><td>SIA / HASE ${h.c('sia', 'hase')}</td><td>✓</td><td>✓</td><td>1</td><td>feedback agent / RL reward</td><td>implicit</td></tr>
<tr><td>Agent0 / SAGE / MAE ${h.c('agent0', 'sage', 'mae')}</td><td>–</td><td>✓</td><td>2–4</td><td>self-play, verifier reward</td><td>–</td></tr>
<tr class="pp-ours"><td><b>VRSI (ours)</b></td><td>✓</td><td>✓</td><td>2, role-attributed</td><td>externally anchored verification, human-calibrated</td><td>explicit prior, tested counterfactually</td></tr>
</tbody></table></div>

${h.sec('3', 'Method', 'method')}
${h.sub('3.1', 'Problem setup', 'setup')}
<p>A case ${h.m.inl(h.m.row(h.m.mi('x'), h.m.mo('='), h.m.par(h.m.mi('q'), h.m.mo(','), h.m.mi('M'))))} consists of a task or claim ${h.m.inl(h.m.mi('q'))} in natural language and original materials ${h.m.inl(h.m.mi('M'))} (videos, images, documents). The agent must return an answer ${h.m.inl(h.m.mi('y'))} whose sentences cite the records that support them. Tasks are heterogeneous: locating the earliest source of a clip, localizing a described moment, comparing two versions of a video, attributing a manipulation, or verifying a factual claim. The method fixes no task type and no relation between pieces of evidence (Sec. 3.2).</p>
<p>Three things can change over the life of the system: the <i>harness specification</i> ${h.m.inl(h.m.row(h.m.mi('s'), h.m.mo('='), h.m.par(h.m.sG, h.m.mo(','), h.m.sH)))} (role prompts, harness parameters and teacher instructions, all versioned and hashed); the role weights ${h.m.inl(h.m.row(h.m.mi('θ'), h.m.mo('='), h.m.par(h.m.thG, h.m.mo(','), h.m.thH)))}, two LoRA adapters ${h.c('lora')} on one frozen multimodal base; and the verifier ${h.m.inl(h.m.V)}. The goal is</p>
${h.eq.obj(1)}
<p>where <i>Task</i> is the dataset-native metric, <i>Chain</i> scores the evidence chain (share of findings supported by their cited evidence, citation correctness, rate of unsupported assertions), and ${h.m.inl(h.m.sub(h.m.cal('D'), h.m.mo('eval')))} is a split that is never used to make a decision. Self-improvement is a sequence ${h.m.inl(h.m.par(h.m.sub(h.m.mi('s'), h.m.mi('n')), h.m.mo(','), h.m.sub(h.m.mi('θ'), h.m.mi('n')), h.m.mo(','), h.m.sub(h.m.V, h.m.mi('n'))))} produced by the system’s own runs, verification and training. We call it <i>bounded</i> because the standard it improves toward—labels, sources, dossiers and a frozen reference verifier—stays fixed throughout.</p>

${h.sub('3.2', 'Two-role investigation architecture', 'arch')}
<figure class="pp-fig">${h.figArch(FA)}<figcaption><b>Figure 1.</b> ForenSearcher. The GIA reads a five-block global view plus key frames of the original media and either dispatches mutually independent tasks for round <i>r</i> or finishes. Each task runs in its own workspace, where the shared HI policy acts with SEARCH and INSPECT. Evidence (E) is stored once and visible to every workspace; findings (F) return only to the GIA; candidate sources (R) remain leads. The answer cites F/E identifiers and can be resolved to sources.</figcaption></figure>

<p><b>Records.</b> The store separates three record types with deliberately narrow meanings. A <i>candidate source</i> R is a search hit (title, snippet, link) and is only a lead; it never becomes evidence in place. An <i>evidence</i> record E is material that was actually read, or the output of a tool applied to such material (a page body, a frame at a timestamp, a crop, an OCR result, an alignment between two videos). It is shown to the model as a six-field card (<span class="pp-tt">id, modalities, label, source_ref, derived_from, acquired_by</span>), with full content retrievable on demand. Being registered as E means <i>traceable</i>, not <i>true</i>: an OCR misread or a spurious alignment remains E and is labeled as model-produced. A <i>finding</i> F is an HI’s judgment about its local task that cites one or more E. Findings are not evidence and are never converted into evidence. Every answer is therefore resolved along</p>
${h.eq.answer(2)}

<p><b>GIA.</b> In round ${h.m.inl(h.m.mi('r'))} the GIA reads a compact global view ${h.m.inl(h.m.sub(h.m.mi('V'), h.m.mi('r')))} with five fixed blocks: the task; cards of the original materials together with their key frames; the evidence cited in the round that just ended; all reports of that round; and a short index of older results. It may call <span class="pp-tt">READ_RECORD</span> or <span class="pp-tt">QUERY_MEMORY</span> to see anything else, including a particular frame. It then decides</p>
${h.eq.gia(3)}
<p>where ${h.m.inl(h.m.sub(h.m.mi('T'), h.m.mi('r')))} is a set of tasks that must be startable independently. A task that needs another task’s result is deferred to a later round. Tasks must also be free of unverified premises: “verify whether the clip matches candidate R12” is allowed, “the clip is from Turkey; find the date” is not. The GIA emits no state updates or to-do lists; the harness builds the next view from the reports. Letting the GIA see pixels is deliberate. For manipulation tasks the text contains no clue, and a planner without pixels can only issue template tasks.</p>

<p><b>HI.</b> Each task opens a workspace ${h.m.inl(h.m.sub(h.m.mi('W'), h.m.row(h.m.mi('r'), h.m.mo(','), h.m.mi('k'))))} with local working memory ${h.m.inl(h.m.mi('L'))}: the task, the materials, candidate sources, a task-ranked view of the shared evidence, a small set of focus evidence the HI keeps, its own history, the latest observation, and up to eight attached images. At each step,</p>
${h.eq.hi(4)}
<p><span class="pp-tt">SEARCH</span> retrieves from the open web by text, image or both, opens a batch of candidates and registers what was actually read as E; a second form opens a known URL or R. <span class="pp-tt">INSPECT</span> applies one of eight operations to registered E (frames, pages, crop, region localization, OCR, ASR, frame localization, media alignment). It returns a uniform result in which an empty result (“no text found”) is distinct from a failure. Visual and audio question answering are deliberately <i>not</i> tools: the HI reads pixels directly, so no intermediate model paraphrases the image. <span class="pp-tt">REPORT</span> ends the workspace with a status (<span class="pp-tt">COMPLETED</span>/<span class="pp-tt">INCOMPLETE</span>, answering only whether the local task is done), findings with evidence citations, the selected evidence and a reason. All workspaces share one HI policy.</p>

<p><b>Dynamic Investigation Rounds and isolation.</b> Workspaces of a round run in parallel and rounds run in sequence; the number of rounds and of workspaces per round are the GIA’s decisions, not constants of the method. Evidence saved by any workspace is immediately visible to all later decisions of the investigation, while an HI never sees sibling findings, reports, tool histories or dataset labels. This asymmetry—<i>share what was read, isolate what was concluded</i>—has three purposes. It prevents a sibling’s unverified conclusion from contaminating another branch. It forces each workspace to form its judgment from evidence. And it makes each role’s decision a function of an input that can be written down, which is exactly what role-relative verification needs (Sec. 3.3). Following Harness-1 ${h.c('harness1')}, the environment rather than the model owns memory management, but we re-implement it for multiple workspaces, a shared multimodal store and video tools.</p>

${h.sub('3.3', 'Evidence-anchored verification of every decision', 'verify')}
<p>Training data and self-improvement signals both come from a verifier that judges <i>decision units</i>, not whole trajectories. There are four units: an HI <b>step</b> (one tool action), an HI <b>report</b>, a GIA <b>plan</b> (one <span class="pp-tt">INVESTIGATE</span>) and a GIA <b>answer</b> (one <span class="pp-tt">FINISH</span>). Unit-level judgment matters in practice. In one pilot case both workspace reports failed verification, so whole-trajectory filtering would have discarded all 20 HI decisions, whereas unit-level judgment kept 17 of them as clean training rows<sup>†</sup>.</p>

<p><b>Three tiers.</b> <i>Executable checks</i> cost nothing and veto: cited identifiers must exist and must have been visible when the unit was produced; completed reports must be non-empty; numbers in a finding must be locatable in the cited text; dossier-required evidence must be cited; tasks within a round must not depend on each other. <i>Anchored judgment</i> pairs every claim with every evidence record it cites. The verifier model sees only that sentence and that record (full text up to a length cap, or the image) and lists which <i>parts</i> of the sentence the record supports or contradicts, each with a quotation. Quotations from text records are matched back to the source programmatically, and unmatched ones are discarded. Judging parts rather than whole sentences is essential because investigative sentences are compound. A pilot with whole-sentence three-way labels marked 401 of 474 pairs “not found”, almost all of them false alarms; with part-level judgment, 589 of 641 pairs were found supported<sup>†</sup>. We also scope each record by <i>provenance</i>: tracing <span class="pp-tt">derived_from</span> tells the verifier that a frame belongs to video B, so it does not “refute” a statement about video A with it—a failure we observed on video-comparison answers before adding provenance<sup>†</sup>. <i>Sufficiency</i> compares the unit against the case dossier: are the required items covered, can the local task already be answered, and does the status (or the decision to finish) match?</p>

<p><b>One taxonomy, five levels.</b> All tiers emit labels from a single enumeration grouped into five levels. <i>Claim</i> level: unsupported assertion, wrong citation, attribution drift, scope overreach, misread fine text, sloppy numerics, omitted required fact, incomplete citation set. <i>State</i> level: completed without finishing, incomplete though answerable, empty report. <i>Action</i> level: illegal, redundant, evidence dropped, wrong tool, ignored feedback. <i>Planning</i> level: task not gap-driven (including premise smuggling), premature finish, unnecessary round, dependent sibling tasks. <i>Environment</i> level: tool failure and contamination, which are recorded but charged to no role. Each type carries a severity and a flag saying whether it propagates to downstream claims. A unit’s disposition is</p>
${h.eq.disp(5)}

<p><b>Role-relative attribution.</b> Each role is judged only against its own input ${h.c('blame', 'mappa')}. If the GIA writes a task that smuggles a premise and the HI executes it faithfully with findings consistent with evidence, the HI’s units are positive and the error is charged to the GIA’s plan. If an HI finding is tainted and the GIA copies it into the answer, the error is charged to the HI’s report, and the GIA’s answer is marked <span class="pp-tt">tainted_input</span>: it is neither a positive nor a negative GIA example. If instead the GIA refuses the tainted finding, re-dispatches a check, or explicitly excludes it, the unit is marked <span class="pp-tt">recovered_from_tainted</span> and up-weighted. These markers record how errors propagate across roles, which the router uses (Sec. 3.4).</p>

<p><b>Hindsight credit.</b> A step has no claim to anchor, so we value it by what it contributed in hindsight ${h.c('absk')}. Let ${h.m.inl(h.m.sup(h.m.cal('U'), h.m.mo('⋆')))} be the dossier-required evidence together with everything cited by supported findings, closed under <span class="pp-tt">derived_from</span>. The clue gain of step ${h.m.inl(h.m.mi('t'))} and its loss weight are</p>
${h.eq.clue(6)}
<p>Clue gain weights the loss and is never a filter, since exploratory steps legitimately gain nothing. The planning analogue, <i>round yield</i>, counts the new requirements covered by supported findings after a round; it is logged but not used for filtering, because a well-aimed task that an HI failed to complete is not a planning error.</p>

<p><b>Calibration and separation.</b> We treat the verifier as a measured instrument, not an oracle. Precision is estimated on a human gold set (at least 300 (finding, evidence) pairs plus 100 step/plan/answer units, double-annotated, with agreement reported). Recall is estimated by injecting known errors into verified reports—swapping a citation, fabricating an identifier, appending an unsupported clause, altering a number, dropping an attribution or dropping a required finding—and counting detections. In the pilot, 0.96 of 202 scoped injections were caught, with the weakest type being dropped required findings (7/10)<sup>†</sup>. Any calibration rule that relaxes a judgment is registered first and its false-acceptance rate is measured on runs other than the one it was derived from. The verifier is a locally hosted model with fixed weights and differs from the teacher ${h.c('inertia')}. A reference verifier ${h.m.inl(h.m.sub(h.m.V, h.m.mo('ref')))} from yet another model family is frozen at the start of the experiments and used only for gating and reporting. Its disagreements with the training verifier, adjudicated by humans, define the exploitation rate.</p>

${h.sub('3.4', 'Error ledger and lever routing', 'route')}
<figure class="pp-fig">${h.figLoop(FL)}<figcaption><b>Figure 2.</b> Verification-Routed Self-Improvement. Runs of the current system are verified unit by unit, and the typed errors form a ledger indexed by role × unit × error type. Only cells whose error type the verifier is calibrated for are eligible; a two-question routing prior sends each eligible frequent cell to one of two levers, the specification or the weights. Each lever is isolated by its own gate (A for the specification, B for one role’s weights at a time). Counterfactual double interventions score the router, and curves are drawn only on a split that never informs a decision.</figcaption></figure>

<p><b>Ledger.</b> Every verified unit adds rows with its role, unit, error type, the <i>actor kind</i> that produced it (teacher, student at round <i>n</i>, or base model), the <i>origin role</i> after attribution, the specification hash and the verifier version. The diagnosis table aggregates these rows into cells ${h.m.inl(h.m.row(h.m.mi('c'), h.m.mo('='), h.m.par(h.m.mt('role'), h.m.mo(','), h.m.mt('unit'), h.m.mo(','), h.m.mt('type'))))}. For each cell it records the teacher rate ${h.m.inl(h.m.sub(h.m.mi('t'), h.m.mi('c')))}, the student rate ${h.m.inl(h.m.sub(h.m.mi('s'), h.m.mi('c')))}, the verifier’s recall ${h.m.inl(h.m.sub(h.m.mi('ρ'), h.m.mi('c')))} and precision ${h.m.inl(h.m.sub(h.m.mi('p'), h.m.mi('c')))} for that type, and the fraction ${h.m.inl(h.m.sub(h.m.mi('u'), h.m.mi('c')))} of occurrences whose origin is an upstream role. Because the teacher normally receives extra instructions, ${h.m.inl(h.m.sub(h.m.mi('t'), h.m.mi('c')))} is measured with a probe in which the teacher receives exactly the student prompt, so that both rates refer to the same input.</p>

<p><b>Eligibility, then two questions.</b> The verifier is not a lever inside the loop but the instrument the loop reads. It is upgraded out of band, only in response to recall injections and human precision audits, never in the same round as a specification change, and the ledger is re-scored after every upgrade. A cell is <i>eligible</i> for routing only if the verifier is calibrated for its type (${h.m.inl(h.m.row(h.m.sub(h.m.mi('ρ'), h.m.mi('c')), h.m.mo('≥'), h.m.mn('0.8')))} and ${h.m.inl(h.m.row(h.m.sub(h.m.mi('p'), h.m.mi('c')), h.m.mo('≥'), h.m.mn('0.85')))}); all other cells are observed and reported but never acted on. For each eligible top-${h.m.inl(h.m.mi('k'))} cell (${h.m.inl(h.m.row(h.m.mi('k'), h.m.mo('='), h.m.mn('3')))}) the prior asks two questions. <i>Whose error is it?</i> If most occurrences originate upstream, the cell is re-routed to the upstream role. <i>Does the teacher make it too?</i> If the student fails far more often than the teacher on the same input, the gap is one of capability and the cell goes to the <i>weights</i>; if the teacher fails as well, the <i>specification</i> is the only thing that can change the outcome:</p>
${h.eq.route(7)}
<p>evaluated top to bottom; re-routed cells are re-evaluated for the upstream role, the capability-gap test precedes the specification test, and cells matching neither are observed. The rationale for the specification branch is not that training would copy teacher errors—teacher rows that fail verification never enter training. Rather, if the teacher also fails in a state, the pipeline yields no positive example for that state, and only a specification change can create one.</p>

<p><b>Counterfactual scoring of the router.</b> A rule that merely sounds reasonable is not a contribution, so the prior is tested. In rounds 1–3, every routed top cell receives a small-budget <i>double intervention</i>: one inner-loop candidate targeted at the cell and one targeted LoRA increment on data for that cell. The lever that reduces the cell’s error rate more per unit cost is the counterfactual best lever, and we report the prior’s accuracy against it:</p>
${h.eq.cf(8)}
<p>Pooling three rounds yields on the order of nine routed cells, so a second test adds defects planted by people outside the project, including ambiguous defects that either lever could fix, with the counterfactual result—not the designer’s intent—as the reference.</p>

${h.sub('3.5', 'Inner loop: harness evolution with weights frozen', 'inner')}
<p><b>What may evolve.</b> Edits are restricted to a whitelist of specification objects and never touch code. The whitelist has three entries. <i>Teacher instructions</i> affect only data production. <i>Harness parameters</i>, such as image limits and page-read sizes, become evolvable once the runtime reads them from the specification. The <i>student prompt</i> includes the role description, the output format and the GIA’s guidance on how to write tasks. Three things are excluded on purpose. Interface contracts stay fixed, because adding fields to tasks or reports would reintroduce designs we rejected (split objectives, unresolved-issue lists). The state renderer stays fixed for the entire study. Tool interfaces and the meaning of E and F stay fixed as well. We keep no separate experience playbook ${h.c('ace', 'rbank')}: an entry of that kind is simply a prompt edit, and treating it as one lets it pass the same gate and be reverted by the same mechanism. Any prompt edit that introduces entities, numbers, URLs or dates is rejected programmatically, so that edits encode investigative practice rather than facts about development cases.</p>

<p><b>Proposal.</b> A strong proposer receives, in the manner of ${h.c('metah')}, the current and past specifications, the diagnosis table, contrasting successful and failed trajectories of the same case for the target cell ${h.c('modrsi')}, and previously rejected edits with their reasons. It returns a structured diff: object, field, old value, new value and the targeted cells. Each candidate edits exactly one object for exactly one cell, and at most ${h.m.inl(h.m.row(h.m.mi('K'), h.m.mo('='), h.m.mn('4')))} candidates are produced per round. The proposer is the teacher throughout; letting the trained student propose is left to future work.</p>

<p><b>Gate A.</b> Candidates first pass regression verification: no case that passed on <span class="pp-tt">dev_select</span> may fail. They are then compared with the current specification with <i>weights frozen</i> and with the current student as executor (two seeds). The executor must be the student rather than the teacher, since the effect of a harness edit depends on the model that executes it ${h.c('harnessdev')}.</p>
${h.eq.gateA(9)}
<p>The single best candidate is then confirmed on <span class="pp-tt">dev_confirm</span>, with ${h.m.inl(h.m.mi('α'))} pre-registered and Holm correction. Only one specification change enters the next version per round. An A/A test (the specification against itself) measures the gate’s false-acceptance rate before any run. Teacher-instruction edits, which change only the data, are judged on data yield and error rates with the teacher as executor. Student-prompt edits are only <i>provisionally</i> accepted and are decided by Gate B after the next retraining, because a student trained under the old prompt is evaluated out of distribution under the new one. Reverting an earlier edit is itself a candidate diff and passes the same gate.</p>

${h.sub('3.6', 'Outer loop: the two roles train each other', 'outer')}
<p>The weights are trained by rejection-sampling fine-tuning in the style of ReST<sup>EM</sup> ${h.c('star', 'restem')}: decisions are sampled, the verifier keeps those that pass, and each role’s LoRA is fit to the kept decisions by maximum likelihood. The loss is that of supervised fine-tuning; the verifier enters only through selection, as a binary reward, which makes the update a form of filtered behavior cloning—the operator SEAL uses as its RL algorithm ${h.c('seal')}. There is one LoRA per role on a shared base. Each training row is one decision of one role; the other role’s output appears only as input and is masked from the loss. The contribution of the outer loop lies in <i>where the data come from</i>, and one principle governs it: <b>partner-consistent data</b>. Each role is trained on data produced with the <i>current</i> other role as its partner, never with the teacher standing in for the partner.</p>
<ul class="pp-list">
<li><b>GIA → HI.</b> In round <i>n</i> the HI is trained on tasks written by the current student GIA, not by the teacher GIA, so its training distribution matches the tasks it will receive at deployment.</li>
<li><b>HI → GIA.</b> Offline GIA states are built from the current student HI’s real reports, flawed ones included. These are the material from which the GIA learns not to trust and to re-check (<span class="pp-tt">recovered_from_tainted</span>). A fixed share of teacher and older-HI reports is mixed in so that this material does not disappear as the HI improves.</li>
</ul>
<p>We deliberately add no curriculum channel in which the GIA is told the HI’s weaknesses ${h.c('agent0')}. It would give the GIA an input that does not exist at deployment, and it would confound the synergy measurement of Sec. 3.7, which is meant to isolate the effect of partner-consistent data alone.</p>

<p><b>Alternation and Gate B.</b> Each round first fixes GIA@<i>n</i> and trains HI@<i>n</i>+1 on GIA→HI data, then fixes HI@<i>n</i>+1 and trains GIA@<i>n</i>+1 on HI→GIA data. Only one role changes at a time. Each update passes its own gate, B-HI or B-GIA, which compares the new weights with the previous ones under the frozen specification. A failed gate rolls back only that role, and its data are kept with a <span class="pp-tt">gate_failed</span> flag.</p>

<p><b>From teacher to self.</b> Round 0 is pure teacher data. From round 1 on, the student system runs ${h.m.inl(h.m.row(h.m.mi('N'), h.m.mo('='), h.m.mn('4')))} times per case, and verified decisions are selected directly, as in rejection sampling ${h.c('star', 'restem')}. For a failed trajectory, the teacher bridges the first erroneous step and the student continues; the bridged segment is kept only if the continuation verifies ${h.c('futureb')}. We track the teacher dependence of the selected data,</p>
${h.eq.dep(10)}
<p>and set a hard criterion: from round 2 on, ${h.m.inl(h.m.row(h.m.sub(h.m.mi('D'), h.m.mi('n')), h.m.mo('&lt;'), h.m.mn('0.5')))} while held-out performance does not drop. If this fails by round 3, the procedure is reported as self-improvement, not RSI. Replay data from earlier rounds are re-exported from the run logs under the current specification, keeping only rows that remain positive under the current verifier, so no training batch mixes specification hashes.</p>

<p><b>Why not online RL.</b> The closest systems that update harness and weights together fine-tune on verified trajectories ${h.c('coharness', 'hzero')}, whereas systems in which several roles co-evolve from one backbone use online policy-gradient RL such as GRPO or Task-Relative REINFORCE++ ${h.c('grpo', 'agent0', 'sage', 'mae')}. We follow the former for three reasons. First, the unit of such RL is an episode per prompt, and a development split of a few dozen cases is too small for group-based estimates without overfitting, whereas decision-level rows turn each case into tens of training examples. Second, an episode here involves open-web search, page reading and video processing and costs tens of teacher-scale calls; sampling a group of episodes per case at every update would dominate the budget, while rejection sampling needs ${h.m.inl(h.m.row(h.m.mi('N'), h.m.mo('='), h.m.mn('4')))} episodes per case per round and verifies them offline. Third, optimizing directly against a model verifier opens the hacking paths below far wider than using it only to select data, and it would stack the instability of RL on that of the loop, so that a change in the curves could no longer be attributed. RL remains a pre-registered add-on, reported as a separate arm: if the curve flattens by rounds 2–3, or teacher dependence stays above 0.5, we add offline DPO ${h.c('dpo')} on preference pairs the procedure already produces—at the same state, a student decision that failed verification against the bridged or resampled decision that passed—which needs no extra rollouts.</p>

<p><b>Guarding against verifier hacking.</b> Selecting self-generated data with a model verifier pushes toward writing that pleases the verifier: fewer and vaguer findings, piled-up citations that trigger lenient rules, excess <span class="pp-tt">INCOMPLETE</span>, and GIA tasks that are easy to complete. Each round reports, with bounds, the coverage of required items by supported findings, the <span class="pp-tt">INCOMPLETE</span> rate on answerable cases, the false-acceptance rate of every calibration rule on counter-examples that should not be exempted, and the exploitation rate against ${h.m.inl(h.m.sub(h.m.V, h.m.mo('ref')))}.</p>

${h.sub('3.7', 'Round protocol and safeguards', 'protocol')}
<div class="pp-algo"><div class="pp-algo-h"><b>Algorithm 1</b> One round of VRSI. Identity: (spec <i>s</i><sub>n</sub>, LoRA<sub>HI</sub>@<i>n</i>, LoRA<sub>GIA</sub>@<i>n</i>, verifier 𝒱<sub>n</sub>)</div><ol>
<li><b>Run</b> the student system on <span class="pp-tt">dev_train</span> (<i>N</i> samples per case) and the teacher probe on the same cases.</li>
<li><b>Verify</b> all step, report, plan and answer units → ledger <span class="pp-tt">verdicts_r<i>n</i></span>.</li>
<li><b>Diagnose</b> cells (role × unit × type) with rates, calibration and origin → <span class="pp-tt">diagnosis_r<i>n</i></span>.</li>
<li><b>Route</b> the eligible top-<i>k</i> cells with Eq. 7; in rounds 1–3, run double interventions to score the router.</li>
<li><b>Inner loop</b>: ≤ <i>K</i> single-object candidates → regression check → Gate A → confirm → <i>s</i><sub>n+1</sub>.</li>
<li><b>Outer loop</b> under <i>s</i><sub>n+1</sub>: produce partner-consistent data (self-sampling, teacher bridging) → train HI → Gate B-HI → train GIA → Gate B-GIA.</li>
<li><b>Roll back</b> whatever failed its gate; keep data flagged <span class="pp-tt">gate_failed</span>.</li>
<li><b>Evaluate</b> once on <span class="pp-tt">dev_eval</span> with recorded retrieval; append to <span class="pp-tt">revision_log</span> and <span class="pp-tt">budget</span>.</li>
</ol></div>

<p><b>Attribution by construction.</b> Each gate evaluates one kind of change. Gate A holds the weights fixed; Gates B-HI and B-GIA hold the specification fixed and change one role each. Verifier upgrades happen between rounds, never in a round with a specification change, and the ledger is re-scored so that error curves stay comparable. Every change carries its target cells, gate results and confirmation result in the revision log.</p>

<p><b>Five splits and a fixed environment.</b> Cases are grouped by event or source video and split into <span class="pp-tt">dev_train</span> (data), <span class="pp-tt">dev_select</span> (gates), <span class="pp-tt">dev_confirm</span> (confirmation), <span class="pp-tt">dev_eval</span> (curves only) and a locked <span class="pp-tt">test</span>. Gating rolls back failures, so a curve drawn on the gating split rises by construction; we therefore draw curves only on splits that never inform a decision. Open-web retrieval drifts over time and can leak published fact-checks. We therefore record search results, opened pages and downloaded media on first contact and replay them for all arms and rounds, truncate retrieval at the claim date, cache tool outputs, and report test results with and without a blocklist of fact-checking domains.</p>

<p><b>What the protocol measures.</b> The design lets each negative result be checked directly. <i>Against self-judgment failing to predict improvement</i> ${h.c('s3gym')}, we estimate a dose–response relation between the change in verifier score and the change in held-out metrics. Beyond the few points produced by gating, the frozen-weight inner loop generates 100–200 random or deliberately harmful specification edits, analysed with a hierarchical model clustered by round and seed. <i>Against edits that do not hold</i> ${h.c('harnessdev')}, we report the direction consistency of every accepted edit on <span class="pp-tt">dev_confirm</span> and whether its direction holds across executors (4B, 9B, 27B students and the teacher). <i>Against progress that is really recovery</i> ${h.c('aspire')}, the training recipe is fixed; models influence only which verified data are selected, and every checkpoint is compared with both the previous round and the base model. To show that the two roles help each other, we evaluate cross-pairings of GIA and HI checkpoints and report the synergy</p>
${h.eq.syn(11)}
<p>with confidence intervals over seeds, together with pairings against teacher partners, which separate genuine improvement from co-adaptation. Main comparisons run at equal budget (GPU hours, teacher tokens, verifier calls): full loop, training only, harness only, and pure distillation, in which the same teacher budget produces more data without a loop.</p>

${h.sub('3.8', 'Instantiation', 'impl')}
<p>Both roles share Qwen3.5-9B ${h.c('qwen35')}, a natively multimodal base that accepts images and video, with one LoRA adapter per role; local debugging uses the 4B model of the same family. The teacher is Qwen3.5-122B-A10B in FP8 on four of eight A100-80GB GPUs, leaving the rest for student sampling and training. The verifier is a local fixed-weight model distinct from the teacher, and ${h.m.inl(h.m.sub(h.m.V, h.m.mo('ref')))} comes from a third family. States are rendered by one renderer at a single 16K-token budget, with images accounted separately under a 32K training limit, fixed for the whole study. Development cases are drawn from six sources spanning web and image–text claim verification, video manipulation, moment retrieval and copy localization; EVID-Bench ${h.c('evid')} is locked as the main test set. Held-out datasets from each capability dimension are never seen in training and are used to measure transfer with 0/20/50-example adaptation curves.</p>
`
  };
})();
/* 论文草稿 · 中文 */
(() => {
  const P = window.ForenPaper;
  const FA = {
    aria: 'ForenSearcher 架构：GIA 按动态轮次派发并行的 HI 工作区，证据库全局共享，发现只回交 GIA。',
    caseT: '案例输入', case1: '说法 C0 · 任务 q', case2: '视频 / 图片 / 网页', case3: '初始化时登记为 E', case4: '原始材料 M',
    giaT: '全局调查智能体', gia1: '全局视图（固定五块）', gia2: '+ 原始材料关键帧', gia3: '按需 READ_RECORD', gia4: 'INVESTIGATE {τ} | FINISH y',
    round: '动态调查轮次 r', roundR: '轮内并行 · 轮间串行',
    hi1: '只拿到任务 τ', hi2: '局部工作记忆 L', hi3: '+ 证据图像像素', hi4: '报告：状态 + F',
    wall: '看不到兄弟工作区的发现与报告',
    store: '全局存储（每条记录只存一份）',
    rT: 'R · 候选来源', r1: '标题、摘要、链接', r2: '只是线索，不升级为证据',
    eT: 'E · 证据', e1: '实际读到的材料 / 工具结果', e2: '所有工作区共享',
    fT: 'F · 发现', f1: '引用 E 的局部判断', f2: '只回交 GIA',
    ansT: '最终答案', ans1: '每句话都带', ans2: 'F / E 编号',
    aTasks: '派任务', aWrite: '写入 E', aRead: '读取 E', aRep: '报告', aF: '发现 → GIA', aFin: 'FINISH'
  };
  const FL = {
    aria: '核验路由的自我改进：运行、核验、错误账本、路由到内环或外环、门控、评测。',
    anc: '外部参照（锚）', anc1: '数据集标签 · 原始来源 · 每案例的案卷（必须查明的要求、必需证据）',
    ancR: '核验器 𝒱（本地固定权重，≠ 教师）· 𝒱ref 冻结、另一模型家族', ancR2: '精确率：人工金标 · 召回率：错误注入',
    run: '运行', run1: '学生系统跑 dev_train', run2: '每案例 N 条 + 教师探针',
    ver: '逐单元核验', ver1: 'HI 动作 · HI 报告', ver2: 'GIA 派工 · GIA 答案',
    led: '错误账本', led1: '角色 × 单元 × 错误类型', led2: '教师率 / 学生率、责任角色',
    rou: '路由器（两问先验）', rou0: '入场条件：该错误类型的核验已校准', rou1: '① 是哪个角色的错？', rou2: '② 教师也犯吗？', rou4: '先验，由反事实双干预打分',
    l2: '杠杆 1 · 内环（权重冻结）', l2a: '只改一个规格对象：提示 · 参数 · 教师指令', l2b: '≤ 4 个候选 → 回归验证 → 门 A', l2c: '最优者再到 dev_confirm 确认', l2d: '每轮只接受一个改动',
    l3: '杠杆 2 · 外环（规格冻结）', l3a: 'GIA@n 派任务 → 训 HI → 门 B-HI', l3b: 'HI@n+1 报告 → 训 GIA → 门 B-GIA', l3c: '学生自采 + 教师桥接（伙伴同步数据）', l3d: '教师依赖率 Dn ↓',
    aIn: '教师也犯 → 改规格', aOut: '能力差距 → 改权重',
    ev: '门控 → 下一轮', ev1: '门 A 只评规格 · 门 B-HI / B-GIA 各评一个角色的权重 · 门不过只回滚那一处改动', ev2: '曲线只画在 dev_eval（从不参与决策）· test 锁定 · 检索录制回放',
    aNext: '下一轮  (s, θ, 𝒱)'
  };

  P.lang.zh = {
    label: '中文',
    eqText: { chain: '证据链', if: '若 ', or: ' 或 ', and: '  或  ', pending: '若存在未决判定', otherwise: '其余情况', reroute: '改道至 ', origin: '上游角色', cost: '成本', accept: '接受 ', on: '  （在 ', then: ' 上成立，再经 ', end: ' 确认）', teacher: ' 目标输出由教师产生', sev: ' 严重度' },
    title: 'ForenSearcher：以外部证据为锚、按核验路由的双角色多模态调查智能体自我改进',
    authors: '匿名 CVPR 投稿',
    draft: '工作草稿 · 第 1–3 节（引言、相关工作、方法）。实验正在进行，结果后补；标 † 的数字来自第 0 轮试点，仅作说明。',
    abstractTitle: '摘要',
    tocLabel: '目录',
    refsTitle: '参考文献',
    abstract: `核查一段误导性视频是一次调查，而不是一次分类：调查员把说法拆成可以查的小问题，检索候选来源，查看帧和页面，比对媒体，最后写出结论，结论里的每一句话都能追溯到实际读过的材料。本文提出 <b>ForenSearcher</b>，一个由两个可学习角色构成的多模态调查智能体。<i>全局调查智能体</i>（GIA）按动态轮次派发并行的子调查，并判断证据是否已经足够；<i>工作区调查员</i>（HI）在相互隔离的工作区里，用检索与媒体检查工具完成各自的子调查。一个持久化的证据库让所有工作区共享"读到了什么"，同时让每个工作区的结论保持私有，于是每个答案都构成一条可审计的链：<i>答案 → 发现 → 证据 → 来源</i>。在这一架构之上，我们提出<b>核验路由的自我改进</b>（Verification-Routed Self-Improvement，VRSI）。这是一个有界的自我改进循环，针对近期关于递归自我改进的三项负面结论而设计。VRSI 不依赖模型自评：每一个决策单元——HI 的每一步动作、每一份工作区报告、GIA 的每一次派工与每一个最终答案——都对照外部参照（数据集标签、原始来源、每案例的案卷）核验，并按同一套错误类型标注、归因到负责的角色。由此形成的错误账本经一条两问先验——这错是哪个角色的？教师也犯吗？——路由到两根杠杆之一：冻结权重、只改调查规格；或冻结规格、只重训其中一个角色。在训练环里，每个角色都用当前搭档产出的数据训练。路由规则被当作可证伪的先验，用反事实双干预检验；一套防自欺的协议——五份数据划分、检索录制回放、每道门只接受一个改动——规定了如何判断循环是在改进系统，而不只是在改变系统。`,
    body: h => `
${h.sec('1', '引言', 'intro')}
<p>一段桥梁坍塌的视频在网上流传，配文写着"上周拍摄于成都"。这句配文是真是假，很少能只靠像素判断。专业的事实核查员会先抽出关键帧，做以图搜图，打开能找到的最早上传版本，把它与流传版本逐帧对齐，再去查所说事件的新闻报道，最后才写下结论，并为每一句陈述注明支撑它的页面和画面。检索来源、定位片段、比对媒体、归因篡改、核实说法——这同一组动作反复出现在各类任务里，而现有基准通常把这些任务分开处理。本文关心两个问题：如何构建一个能端到端完成此类调查、且证据可审计的智能体；以及这样的智能体如何自我改进，而不偏离事实本身。</p>

<p>现有工作各自覆盖了问题的一部分。多模态虚假信息检测器对图文对和短视频预测"脱离语境"或"虚假"标签 ${h.c('newsclip', 'verite', 'fakesv', 'fakett')}，近来的多模态大模型还能给出解释 ${h.c('sniffer')}，但它们都不会离开给定输入去查证来源。基于证据的事实核查 ${h.c('averitec', 'averimatec', 'defame')} 会从网上检索证据，但主要面向文本或图文说法。深度研究智能体 ${h.c('searchr1', 'webthinker', 'mmsearch')} 能在开放网络上搜索，却把全部过程塞进一个上下文，装不下几十步里的视频帧，也很少检查自己引用的来源是否真的支持自己写下的话。新近的视频虚假信息基准 ${h.c('evid', 'groundlie')} 把这一缺口摆到了台面上：要解这些题，必须做外部检索、跨视频比对，并指出被篡改的片段。</p>

<p>要做成这件事，有两道坎。<b>第一道在架构。</b>一次涉及视频的调查会撑爆任何单一上下文。拆成规划者和执行者能缓解问题，但简单的拆分会让某个分支尚未核实的结论，作为前提混进另一个分支；责任也随之变得模糊——最终答案错了，说不清是规划错了还是执行错了。<b>第二道在学习。</b>专家调查轨迹没有大规模语料，智能体只能先从教师示范起步，再自己改进。递归自我改进（RSI）自然成为框架：冻结权重、演化运行框架（harness）${h.c('metah', 'rhi', 'modrsi')}；在自生成并经筛选的数据上训练 ${h.c('star', 'rest')}；或者两者结合 ${h.c('coharness', 'hase', 'sia')}。但近期三项研究给这种乐观泼了冷水：让模型自己决定训练什么，最终很少超过底座，而且常把"从自己造成的退化里爬回来"误当成进步 ${h.c('aspire')}；会自测、会自评，并不等于会改进 ${h.c('s3gym')}；自我演化出的 64 次 harness 修改只有 34 次在留出任务上方向一致，还有大量演化出的组件从未被触发 ${h.c('harnessdev')}。我们认为这些失败有两个共同的根源：改进信号没有锚定在模型之外的任何东西上；面对一次具体的失败，也没有机制决定<i>该改什么</i>——是改提示，改某个组件的权重，还是改评判者自己。</p>

<p><b>事实核查恰好补上了这两块。</b>其一，它有外部参照。数据集标签、原始来源，以及每个案例的案卷（写明必须查明什么、哪些证据是必需的），使我们可以拿智能体实际取回的材料去核对中间结论，而不是拿模型对自身输出的看法去核对。需要说明，只有标签是真值；过程判定由一个锚定在这些材料上的核验模型完成，它的精确率与召回率在人工标注上实测，而不是默认成立。其二，调查天然可以归因。规划与执行分成两个接口明确的角色之后，每个决策都能只相对该角色自己的输入来评判，每个错误都能记到造成它的角色头上，从而指明该改哪里。</p>

<p>ForenSearcher 正是建立在这两点之上（图 1）。GIA 读取一份紧凑的全局视图和原始媒体的关键帧，为本轮派出相互独立的任务，然后要么给出带引用的答案，要么再开一轮。每个任务在自己的工作区里执行：HI（所有工作区共用同一套参数）每步在 <span class="pp-tt">SEARCH</span>、<span class="pp-tt">INSPECT</span> 和两个只读记忆动作中选一个，最后提交一份报告，报告里的每条发现都引用证据编号。证据只存一份、全局共享；发现只回交 GIA。在这个系统之上，VRSI（图 2）把循环闭合起来：以证据为锚的核验器评判四类决策单元，把带类型的错误记入账本；一个两问路由器把每个高频错误格分配给只改 harness 的内环或只改权重的外环——在外环里，GIA 与 HI 轮流接受训练，训练数据由对方作为搭档参与生成。核验器不是循环里的一根杠杆：它在循环之外校准，只有它已校准的错误类型才允许被路由。每个改动都要过一道把它单独隔离出来的门，所有曲线都画在从不参与决策的数据划分上。</p>

<p>本文的贡献如下：</p>
<ul class="pp-contrib">
<li><b>一个双角色的多模态调查架构。</b>它采用动态调查轮次，并规定了三类记录——候选来源、证据、发现——的语义：共享读到的材料，隔离得出的结论。于是答案的每一句都能追溯到原始材料，每一个错误都能归到一个角色（第 3.2 节）。</li>
<li><b>以证据为锚、逐单元的核验。</b>三层核验器（可执行检查；带引语回查与来源限定的部分支持判定；基于案卷的充分性检查）用同一套错误类型覆盖两个角色的全部四类决策单元，配有相对输入的角色归因和事后信用分配；精确率在人工金标上校准，召回率用程序化错误注入校准（第 3.3 节）。</li>
<li><b>核验路由的自我改进（VRSI）。</b>这是一个双环、双角色、有界的 RSI 过程：带角色归因的错误账本经一条显式的两问先验路由到 harness 或权重；两个角色互为对方的在策略数据生成者；先验不被当作前提，而是用反事实双干预打分（第 3.4–3.6 节）。</li>
<li><b>一套防自欺的评测协议。</b>它把三项负面结论变成可测的量：核验分变化与留出集提升之间的剂量–反应关系、被接受改动的方向一致率、教师依赖率曲线，以及交叉搭配的协同项；全部在五份数据划分与检索录制回放下进行（第 3.7 节）。</li>
</ul>

${h.sec('2', '相关工作', 'related')}
<p><b>多模态虚假信息与事实核查。</b>脱离语境的虚假信息最早被表述为图文对分类，数据或由互换配文自动合成 ${h.c('newsclip')}，或取自核查文章 ${h.c('verite')}；短视频数据集进一步加入了音频、评论与发布者信息 ${h.c('fakesv', 'fakett')}。检测方法从多模态融合发展到能给出解释的多模态大模型 ${h.c('sniffer')}。基于证据的核验先在带网络证据的真实文本说法上成形 ${h.c('averitec')}，随后扩展到图文说法 ${h.c('averimatec')}；DEFAME ${h.c('defame')} 会规划检索并调用多模态专家。最近，EVID-Bench ${h.c('evid')} 与 GroundLie360 ${h.c('groundlie')} 要求以检索为依据分析视频、跨视频比对并定位伪造片段，VideoSham ${h.c('videosham')} 等篡改数据集则关注人脸以外的编辑。我们认同这种"求证"的视角，但有两点不同：一个智能体覆盖多种异构的视频核查任务；答案的每一句都必须能沿持久化记录追溯。正是后一点，使过程级核验——进而使自我改进——成为可能。检测合成内容的取证工具与本文互补，我们的工具接口为其预留了位置，但方法不依赖它。</p>

<p><b>深度研究与多模态检索智能体。</b>用强化学习训练的检索智能体让推理与检索交替进行 ${h.c('searchr1', 'webthinker')}，多模态版本还学习何时、如何以图搜索 ${h.c('mmsearch')}。视频智能体通过反复取帧来回答关于给定视频的问题 ${h.c('videoagent')}，但不会离开这段视频去查它从哪里来。Harness-1 ${h.c('harness1')} 表明，让环境持有工作记忆（去重、压缩、精选证据）、让模型只做语义决策是有益的；我们采纳这一思想，并针对多工作区、共享多模态证据库和视频工具重新实现。规划–执行分离的系统 ${h.c('planact', 'magentic')} 以及面向深度研究的分层多智能体强化学习 ${h.c('mgrpo')} 同样把全局规划与局部执行分开。我们的 GIA/HI 划分的不同之处在于信息契约：证据持久化并在工作区之间共享，而发现只回交规划者、从不交给兄弟工作区；轮次当场决定，而不是一次规划到底；任务里不得把尚未核实的结论当作前提。正是这些约束，让"错误归属于哪个角色"有了明确定义。</p>

<p><b>过程核验与信用分配。</b>在正确性可以检查的领域，过程奖励模型和步级核验器 ${h.c('lightman', 'mathshep')} 能改进推理。面向带引用的生成与检索智能体，已有工作分别检查引用质量 ${h.c('alce')}、引用感知的 rubric 奖励 ${h.c('carr')}、实例级 rubric ${h.c('rar', 'rubicon')}、以轨迹前缀为条件的步级标准 ${h.c('arco')} 和决定是否停止检索的充分性门 ${h.c('reveal')}。在多智能体系统中，失败归因研究"哪个智能体、哪一步导致了失败" ${h.c('whowhen', 'blame')}，按角色的局部核验则支撑了无 Critic 的过程强化学习 ${h.c('mappa')}。我们把这几条线合成一个核验器：判定锚定在断言所引用的那条证据记录上（文本或像素），并由程序回查引语；用同一套错误类型覆盖两个角色的四类单元；精确率与召回率都经过校准。只有校准过的核验，才能拿来当自我改进循环的学习信号。</p>

<p><b>智能体的递归自我改进。</b>免训练的做法包括优化提示与程序 ${h.c('dspy', 'gepa')}、搜索智能体设计 ${h.c('adas', 'dgm')}、积累经验手册 ${h.c('ace', 'rbank')}，以及直接演化 harness ${h.c('metah', 'selfh', 'rhi', 'modrsi', 'rrsi')}；其中的经验包括：逐模块修改优于联合修改 ${h.c('modrsi')}，编辑预算与修剪能抑制过拟合 ${h.c('rrsi')}。需训练的做法筛选自生成轨迹再微调 ${h.c('star', 'rest', 'restem')}，或让模型自己写微调数据 ${h.c('seal')}。两者结合的做法有：交替产出 harness 修改与监督微调、并按失败类型归因 ${h.c('coharness')}；在强化学习中同时学习解题和修改 harness ${h.c('hase')}；由一个反馈智能体同时更新两者 ${h.c('sia')}；或把演化好的 harness 蒸馏回权重 ${h.c('hzero')}。多模型共同演化让课程/挑战者智能体与求解者配对 ${h.c('agent0', 'sage', 'mae')}。一篇近期综述 ${h.c('rsisurvey')} 区分了朝固定外部标准收敛的<i>有界</i> RSI 与开放式 RSI，并把自我评估称为承重支柱；三项研究指出了当前 RSI 的失效之处 ${h.c('aspire', 's3gym', 'harnessdev')}。我们把 VRSI 定位为有界 RSI，与已有工作有四点不同（表 1）：改进信号是锚定在外部材料上的核验，而不是自评或两两偏好；两个角色与两根杠杆处在同一个循环里，由一条显式、可证伪的路由规则连接；两个角色互相生成训练数据；评测协议直接测量负面结论所质疑的那些量。</p>

<div class="pp-table-wrap"><table class="pp-table">
<caption><b>表 1.</b> 与自我改进类方法的比较。<i>信号</i>：决定一个改动是否保留的依据。<i>杠杆选择</i>：方法如何决定一次失败该在 harness 里修还是在权重里修。</caption>
<thead><tr><th>方法</th><th>改 harness</th><th>改权重</th><th>角色数</th><th>信号</th><th>杠杆选择</th></tr></thead>
<tbody>
<tr><td>ModularRSI / RRSI ${h.c('modrsi', 'rrsi')}</td><td>✓</td><td>–</td><td>1</td><td>基准分、评论器</td><td>–</td></tr>
<tr><td>RHI ${h.c('rhi')}</td><td>✓</td><td>–</td><td>多个</td><td>LLM 两两评判</td><td>–</td></tr>
<tr><td>STaR / ReST ${h.c('star', 'rest')}</td><td>–</td><td>✓</td><td>1</td><td>答案正确性</td><td>–</td></tr>
<tr><td>Co-Harness ${h.c('coharness')}</td><td>✓</td><td>✓</td><td>1</td><td>LLM 评论器看失败轨迹</td><td>按失败类型归因</td></tr>
<tr><td>SIA / HASE ${h.c('sia', 'hase')}</td><td>✓</td><td>✓</td><td>1</td><td>反馈智能体 / RL 奖励</td><td>隐式</td></tr>
<tr><td>Agent0 / SAGE / MAE ${h.c('agent0', 'sage', 'mae')}</td><td>–</td><td>✓</td><td>2–4</td><td>自博弈、核验器奖励</td><td>–</td></tr>
<tr class="pp-ours"><td><b>VRSI（本文）</b></td><td>✓</td><td>✓</td><td>2，按角色归因</td><td>以外部材料为锚的核验，经人工校准</td><td>显式先验，反事实检验</td></tr>
</tbody></table></div>

${h.sec('3', '方法', 'method')}
${h.sub('3.1', '问题设定', 'setup')}
<p>一个案例 ${h.m.inl(h.m.row(h.m.mi('x'), h.m.mo('='), h.m.par(h.m.mi('q'), h.m.mo(','), h.m.mi('M'))))} 由自然语言的任务或说法 ${h.m.inl(h.m.mi('q'))} 和原始材料 ${h.m.inl(h.m.mi('M'))}（视频、图片、文档）组成。智能体要返回答案 ${h.m.inl(h.m.mi('y'))}，其中每一句都引用支撑它的记录。任务是异构的：找一段视频的最早来源、定位一段描述对应的时刻、比较同一视频的两个版本、归因一处篡改，或者核实一条事实说法。方法本身不预设任务类型，也不预设证据之间的关系（第 3.2 节）。</p>
<p>在系统的生命周期里，有三样东西可以变：<i>调查规格</i> ${h.m.inl(h.m.row(h.m.mi('s'), h.m.mo('='), h.m.par(h.m.sG, h.m.mo(','), h.m.sH)))}（角色提示、harness 参数和教师指令，全部带版本号与哈希）；两个角色的权重 ${h.m.inl(h.m.row(h.m.mi('θ'), h.m.mo('='), h.m.par(h.m.thG, h.m.mo(','), h.m.thH)))}，即挂在同一个冻结多模态底座上的两套 LoRA ${h.c('lora')}；以及核验器 ${h.m.inl(h.m.V)}。目标是</p>
${h.eq.obj(1)}
<p>其中 <i>Task</i> 是数据集原生指标；<i>Chain</i> 评价证据链，包括发现被所引证据支持的比例、引用归属正确率和无据断言率；${h.m.inl(h.m.sub(h.m.cal('D'), h.m.mt('eval')))} 是从不用于任何决策的数据划分。自我改进，就是由系统自己的运行、核验和训练产生的一串 ${h.m.inl(h.m.par(h.m.sub(h.m.mi('s'), h.m.mi('n')), h.m.mo(','), h.m.sub(h.m.mi('θ'), h.m.mi('n')), h.m.mo(','), h.m.sub(h.m.V, h.m.mi('n'))))}。我们称之为<i>有界</i>的，是因为它所趋近的标准——标签、来源、案卷和一个冻结的参考核验器——始终不变。</p>

${h.sub('3.2', '双角色调查架构', 'arch')}
<figure class="pp-fig">${h.figArch(FA)}<figcaption><b>图 1.</b> ForenSearcher。GIA 读取固定五块的全局视图和原始媒体的关键帧，要么为第 <i>r</i> 轮派出相互独立的任务，要么结束调查。每个任务在自己的工作区里执行，由共享参数的 HI 调用 SEARCH 与 INSPECT。证据（E）只存一份、对所有工作区可见；发现（F）只回交 GIA；候选来源（R）始终只是线索。答案引用 F/E 编号，可以一路解析回来源。</figcaption></figure>

<p><b>记录。</b>证据库区分三类记录，每类的含义都刻意收窄。<i>候选来源</i> R 是一条搜索结果（标题、摘要、链接），只是线索，永远不会原地升级为证据。<i>证据</i> E 是实际读到的材料，或工具作用于这些材料的结果（网页正文、某一时刻的帧、裁剪图、OCR 结果、两段视频的对齐结果）。它以六字段卡片呈现给模型（<span class="pp-tt">id、modalities、label、source_ref、derived_from、acquired_by</span>），完整内容按需读取。登记为 E 只表示<i>可追溯</i>，不表示<i>已证实</i>：OCR 认错的字、似是而非的对齐结果照样是 E，只是标明由模型产生。<i>发现</i> F 是 HI 针对自己的局部任务、引用一条或多条 E 得出的判断。发现不是证据，也永远不会被转成证据。于是每个答案都沿下式解析：</p>
${h.eq.answer(2)}

<p><b>GIA。</b>第 ${h.m.inl(h.m.mi('r'))} 轮，GIA 读取一份紧凑的全局视图 ${h.m.inl(h.m.sub(h.m.mi('V'), h.m.mi('r')))}，固定五块：总任务；原始材料卡片及其关键帧；刚结束这一轮实际引用的证据；这一轮的全部报告；更早结果的简短索引。其余内容（包括某一张具体的帧）可以通过 <span class="pp-tt">READ_RECORD</span> 或 <span class="pp-tt">QUERY_MEMORY</span> 调取。随后它做出决策：</p>
${h.eq.gia(3)}
<p>其中 ${h.m.inl(h.m.sub(h.m.mi('T'), h.m.mi('r')))} 是一组必须能各自独立启动的任务；需要等别的任务出结果才能确定的任务，推到下一轮。任务里也不能夹带未经核实的前提："核实待查视频与候选 R12 是否一致"可以，"该视频来自土耳其，请查日期"不行。GIA 不输出状态更新或待办清单，下一轮的视图由 harness 根据报告构造。让 GIA 看到像素是有意为之：篡改类任务的文字里没有任何线索，看不到画面的规划者只能派出模板化的任务。</p>

<p><b>HI。</b>每个任务开启一个工作区 ${h.m.inl(h.m.sub(h.m.mi('W'), h.m.row(h.m.mi('r'), h.m.mo(','), h.m.mi('k'))))}，配有局部工作记忆 ${h.m.inl(h.m.mi('L'))}：任务、材料、候选来源、按本任务排序的共享证据视图、HI 自己保留的少量焦点证据、自己的历史、最新一次观察，以及至多八张附图。每一步，</p>
${h.eq.hi(4)}
<p><span class="pp-tt">SEARCH</span> 用文字、图片或图文组合在开放网络上检索，自动打开一批候选，把实际读到的内容登记为 E；它的另一种形式用于打开已知的 URL 或 R。<span class="pp-tt">INSPECT</span> 对已登记的 E 执行八种操作之一（取帧、取页、裁剪、区域定位、OCR、语音转写、片段定位、媒体对齐），返回统一结构，并严格区分空结果（"没有识别到文字"）与失败。视觉问答和音频问答被刻意<i>排除</i>在工具之外：HI 直接读像素，不需要中间模型替它转述画面。<span class="pp-tt">REPORT</span> 结束工作区，给出状态（<span class="pp-tt">COMPLETED</span>/<span class="pp-tt">INCOMPLETE</span>，只回答局部任务是否完成）、带证据引用的发现、选定的证据和理由。所有工作区共用一套 HI 参数。</p>

<p><b>动态调查轮次与隔离。</b>同一轮的工作区并行，不同轮之间串行；轮数和每轮的工作区数由 GIA 决定，而不是方法里的常数。任何工作区保存的证据，对本次调查之后的所有决策立即可见；但 HI 看不到兄弟工作区的发现、报告、工具历史，也看不到数据集标签。这种不对称——<i>共享读到的，隔离得出的</i>——有三个目的：防止兄弟分支未经核实的结论污染本分支；迫使每个工作区从证据出发形成自己的判断；使每个角色的决策都成为一个可以写下来的输入的函数，而这正是按角色相对输入核验所需要的（第 3.3 节）。我们沿用 Harness-1 ${h.c('harness1')} 的思想，由环境而非模型负责管理记忆，但针对多工作区、共享多模态证据库和视频工具重新实现。</p>

${h.sub('3.3', '以证据为锚、逐个决策的核验', 'verify')}
<p>训练数据和自我改进信号都来自同一个核验器，它评判的是<i>决策单元</i>，而不是整条轨迹。单元共四类：HI 的<b>动作</b>（一次工具调用）、HI 的<b>报告</b>、GIA 的<b>派工</b>（一次 <span class="pp-tt">INVESTIGATE</span>）和 GIA 的<b>答案</b>（一次 <span class="pp-tt">FINISH</span>）。逐单元评判在实践中很关键：试点中有一个案例，两份工作区报告都未通过核验，若按整条轨迹筛选，20 个 HI 决策一行都留不下；逐单元评判则保留了其中 17 行干净可学的数据<sup>†</sup>。</p>

<p><b>三层核验。</b><i>可执行检查</i>不花钱，且一票否决：被引编号必须存在，且在该单元产生时可见；报告完成时不能为空；发现里的数字必须能在所引文本中找到；案卷要求的证据必须被引用；同一轮的任务不能相互依赖。<i>锚定判定</i>把每条断言与它引用的每一条证据逐对配对。核验模型只看这一句话和这一条记录（文本全文，有长度上限；或图片），列出记录支持了句子的哪些<i>部分</i>、反驳了哪些部分，每部分都附引语；文本记录的引语由程序回原文匹配，匹配不上的一律不采信。按部分而不是按整句判定至关重要，因为调查里的句子几乎都是复合句。试点中，按整句三态判定时，474 对里有 401 对被判"未找到"，几乎全是误报；改为按部分判定后，641 对中有 589 对被判为有支持<sup>†</sup>。我们还按<i>来源</i>限定每条记录的管辖范围：沿 <span class="pp-tt">derived_from</span> 回溯，核验器就知道某张帧属于视频 B，不会拿它去"反驳"关于视频 A 的陈述。加入这一限定之前，我们在视频比对类答案上确实观察到了这种误判<sup>†</sup>。<i>充分性</i>检查把单元与案卷对照：要求项是否都已覆盖？局部任务是否已经可以回答？状态（或结束调查的决定）是否与之相符？</p>

<p><b>一套错误类型，五个层级。</b>三层核验都从同一份枚举中取标签，分五个层级。<i>结论级</i>：无据断言、引用错误、转述当事实、以偏概全、细字误读、数值不严、遗漏必需事实、引用集合不全。<i>状态级</i>：未查完即报完成、已可回答仍报未完成、空报告。<i>动作级</i>：非法动作、重复动作、丢弃证据、选错工具、忽视反馈。<i>调度规划级</i>：任务不针对缺口（含前提走私）、过早结束、多余轮次、同轮任务相互依赖。<i>环境级</i>：工具故障与污染，只记录，不归咎任何角色。每种类型自带严重度，以及是否会污染下游结论的标记。一个单元的处置为</p>
${h.eq.disp(5)}

<p><b>相对输入的角色归因。</b>每个角色只相对它自己的输入评判 ${h.c('blame', 'mappa')}。若 GIA 派出的任务夹带了前提，HI 照任务忠实执行，得到的发现与证据一致，那么 HI 的单元是正例，错误记在 GIA 的派工上。若某条 HI 发现已被污染，而 GIA 把它照抄进答案，错误记在 HI 的报告上，GIA 的答案标为 <span class="pp-tt">tainted_input</span>：既不作 GIA 的正例，也不作反例。反过来，若 GIA 没有采信被污染的发现，而是改派复核或在答案中明确排除，该单元标为 <span class="pp-tt">recovered_from_tainted</span>，并提高权重。这些标记记录了错误如何在角色之间传播，供路由器使用（第 3.4 节）。</p>

<p><b>事后信用分配。</b>一步动作没有断言可以锚定，我们就看它事后贡献了什么 ${h.c('absk')}。设 ${h.m.inl(h.m.sup(h.m.cal('U'), h.m.mo('⋆')))} 为案卷要求的证据与所有被判支持的发现所引用证据的并集，并沿 <span class="pp-tt">derived_from</span> 取闭包。第 ${h.m.inl(h.m.mi('t'))} 步的线索增益及其损失权重为</p>
${h.eq.clue(6)}
<p>线索增益只用于给损失加权，从不用于筛选，因为探索性的步骤本来就可能一无所获。规划层面的对应量是<i>轮次收益</i>，即一轮结束后被支持的发现新覆盖了多少项要求。它同样只记录、不用于筛选：派得对、但 HI 没办成，不是规划的错。</p>

<p><b>校准与分离。</b>我们把核验器当作需要标定的测量仪器，而不是神谕。精确率在人工金标集上估计（至少 300 对（发现，证据），另加 100 个动作/派工/答案单元，两人标注并报告一致性）。召回率通过向已核验报告注入已知错误来估计：替换引用、捏造编号、追加无据分句、改动数字、删去转述标记、删去必需发现，再统计被抓到的比例。试点中，202 处限定范围的注入被抓到 0.96，最弱的一类是删去必需发现（7/10）<sup>†</sup>。任何放宽判定的校准规则都先登记，再在推导出它的运行<i>之外</i>测假接受率。核验器是本地部署、权重固定的模型，并且不同于教师 ${h.c('inertia')}；另有一个来自第三个模型家族的参考核验器 ${h.m.inl(h.m.sub(h.m.V, h.m.mt('ref')))}，在实验开始时冻结，只用于门控和报告。它与训练用核验器的分歧经人工裁定后，定义钻空子率（exploitation rate）。</p>

${h.sub('3.4', '错误账本与杠杆路由', 'route')}
<figure class="pp-fig">${h.figLoop(FL)}<figcaption><b>图 2.</b> 核验路由的自我改进。当前系统的运行结果逐单元核验，带类型的错误汇成按"角色 × 单元 × 错误类型"索引的账本；只有核验器已对其错误类型校准过的格才有资格进入路由；两问先验把每个有资格的高频格分配给两根杠杆之一——规格或权重。每根杠杆都有自己的门（规格用门 A，权重用门 B，每次只评一个角色）把改动隔离出来。反事实双干预为路由器打分；曲线只画在从不参与决策的数据划分上。</figcaption></figure>

<p><b>账本。</b>每个核验过的单元都会写入若干行：角色、单元、错误类型、产出它的<i>执行者类别</i>（教师、第 <i>n</i> 轮学生或底座）、归因后的<i>责任角色</i>、规格哈希和核验器版本。诊断表把这些行汇总为格 ${h.m.inl(h.m.row(h.m.mi('c'), h.m.mo('='), h.m.par(h.m.mt('角色'), h.m.mo(','), h.m.mt('单元'), h.m.mo(','), h.m.mt('类型'))))}。每一格记录：教师率 ${h.m.inl(h.m.sub(h.m.mi('t'), h.m.mi('c')))}、学生率 ${h.m.inl(h.m.sub(h.m.mi('s'), h.m.mi('c')))}、核验器对该类型的召回率 ${h.m.inl(h.m.sub(h.m.mi('ρ'), h.m.mi('c')))} 与精确率 ${h.m.inl(h.m.sub(h.m.mi('p'), h.m.mi('c')))}，以及责任在上游角色的比例 ${h.m.inl(h.m.sub(h.m.mi('u'), h.m.mi('c')))}。由于教师平时会多拿一段教师指令，${h.m.inl(h.m.sub(h.m.mi('t'), h.m.mi('c')))} 用一个探针来测：让教师只拿与学生完全相同的提示，保证两个比率针对的是同一份输入。</p>

<p><b>先看资格，再问两个问题。</b>核验器不是循环里的一根杠杆，而是循环读数用的仪器。它在循环之外升版，理由只能来自召回注入的漏报和人工精确率抽检的误报，从不与规格改动发生在同一轮，升版后重打账本。一个格只有在核验器对其错误类型已经校准（${h.m.inl(h.m.row(h.m.sub(h.m.mi('ρ'), h.m.mi('c')), h.m.mo('≥'), h.m.mn('0.8')))} 且 ${h.m.inl(h.m.row(h.m.sub(h.m.mi('p'), h.m.mi('c')), h.m.mo('≥'), h.m.mn('0.85')))}）时才<i>有资格</i>进入路由；其余格只观察、只报告，不采取任何动作。对每个有资格的、频次前 ${h.m.inl(h.m.mi('k'))} 的格（${h.m.inl(h.m.row(h.m.mi('k'), h.m.mo('='), h.m.mn('3')))}），先验问两个问题。<i>这是谁的错？</i>若多数出错的责任在上游角色，改道交给上游角色。<i>教师也犯吗？</i>若在同一份输入上学生比教师错得多得多，那是能力差距，交给<i>权重</i>；若教师也犯，那就只有改<i>规格</i>才能改变结果：</p>
${h.eq.route(7)}
<p>自上而下判断：改道的格按上游角色重新判断；"能力差距"先于"规格"判断；两条都不满足的格记为观察。走规格这一支的理由并不是训练会把教师的错误抄给学生——未通过核验的教师行本来就进不了训练。真正的原因是：如果教师在某个状态上也会犯错，流水线就产不出这个状态上的正例，只有改规格才能造出正例。</p>

<p><b>用反事实检验路由器。</b>一条只是听起来合理的规则算不上贡献，所以先验必须接受检验。在第 1–3 轮，对每个被路由的高频格做一次小预算的<i>双干预</i>：一个针对该格的内环候选，加一次用该格数据做的针对性 LoRA 增量训练。单位成本下把该格错误率降得更多的杠杆，就是反事实最优杠杆。我们报告先验对它的预测准确率：</p>
${h.eq.cf(8)}
<p>三轮合并大约只有九个被路由的格，因此第二项检验补充由项目外人员设计的植入缺陷，其中包括两根杠杆都能修的模糊缺陷；以反事实结果、而不是设计者的意图作为参照答案。</p>

${h.sub('3.5', '内环：冻结权重，演化 harness', 'inner')}
<p><b>哪些东西可以演化。</b>修改只允许发生在一份规格对象白名单内，从不改代码。白名单有三项：<i>教师指令</i>，只影响数据生产；<i>harness 参数</i>，如图片上限、读页长度，在运行时改为从规格读取之后才可演化；以及<i>学生提示</i>，包括角色说明、输出格式和 GIA"如何写任务"的文字指导。有三样东西刻意排除在外：接口契约不变，因为给任务或报告加字段会重新引入我们已经否定的设计（拆分目标字段、未决问题列表）；状态渲染器在整个研究期间不变；工具接口以及 E、F 的含义也不变。我们不另设经验手册 ${h.c('ace', 'rbank')}：那种条目本质上就是一段提示改动，把它当作提示改动，就能走同一道门、用同一套机制回滚。任何引入实体、数字、URL 或日期的提示改动都由程序拒绝，确保改动记录的是调查方法，而不是开发案例里的具体事实。</p>

<p><b>提议。</b>仿照 ${h.c('metah')} 的组织方式，一个强提议者会拿到：当前和历史各版规格、诊断表、目标格下同一案例成功与失败轨迹的对照 ${h.c('modrsi')}，以及此前被拒绝的改动及其原因。它返回一个结构化的差异：对象、字段、旧值、新值和目标格。每个候选只针对一个格、只改一个对象；每轮最多产生 ${h.m.inl(h.m.row(h.m.mi('K'), h.m.mo('='), h.m.mn('4')))} 个候选。提议者始终是教师；让训练过的学生自己提议留作未来工作。</p>

<p><b>门 A。</b>候选先做回归验证：在 <span class="pp-tt">dev_select</span> 上已经通过的案例不能掉。然后在<i>冻结权重</i>的条件下，以当前学生为执行者（两个随机种子），与当前规格比较。执行者必须是学生而不是教师，因为 harness 改动的效果取决于执行它的模型 ${h.c('harnessdev')}。</p>
${h.eq.gateA(9)}
<p>最优的那一个候选再到 <span class="pp-tt">dev_confirm</span> 上确认，显著性水平 ${h.m.inl(h.m.mi('α'))} 预先登记，并做 Holm 校正；每轮只允许一个规格改动进入下一版。任何运行之前，先做一次 A/A 测试（规格与自身比较），量出门本身的假接受率。教师指令的改动只影响数据，以教师为执行者，按数据产出与错误率评判。面向学生的提示改动只能<i>暂定</i>接受，等下一轮重训后由门 B 定夺，因为在旧提示下训出的学生面对新提示属于分布外评测。撤销一条早先的改动本身也是一个候选差异，走同一道门。</p>

${h.sub('3.6', '外环：两个角色互相训练', 'outer')}
<p>权重用 ReST<sup>EM</sup> 式的拒绝采样微调训练 ${h.c('star', 'restem')}：先采样决策，核验器留下通过的，再对留下的决策做最大似然，拟合各角色的 LoRA。损失函数就是监督微调的损失；核验器只通过筛选起作用，相当于一个二值奖励，因此这种更新是一种过滤后的行为克隆——SEAL 正是把它当作 RL 算法使用的 ${h.c('seal')}。共享底座上每个角色一套 LoRA。每条训练数据是一个角色的一次决策；另一个角色的输出只作为输入出现，不计入损失。外环的贡献在于<i>数据从哪里来</i>，而这由一条原则决定：<b>伙伴同步数据</b>。每个角色都用<i>当前</i>的另一个角色作为搭档产出的数据训练，从不让教师顶替搭档。</p>
<ul class="pp-list">
<li><b>GIA → HI。</b>第 <i>n</i> 轮训练 HI 时，任务由当前的学生 GIA 撰写，而不是教师 GIA。这样 HI 的训练分布与它部署时实际收到的任务一致。</li>
<li><b>HI → GIA。</b>离线的 GIA 状态用当前学生 HI 的真实报告搭建，包括有缺陷的报告。GIA 正是从这些材料中学会"不轻信、改派复核"（<span class="pp-tt">recovered_from_tainted</span>）。其中固定混入一定比例的教师报告与旧版 HI 报告，以免这类材料随 HI 变好而消失。</li>
</ul>
<p>我们有意不加"把 HI 的弱点告诉 GIA、让它出针对性课程"这样的通道 ${h.c('agent0')}：它会给 GIA 一项部署时不存在的输入，也会和第 3.7 节的协同项测量混在一起——那项测量的目的正是单独分离出伙伴同步数据的作用。</p>

<p><b>交替更新与门 B。</b>每一轮先固定 GIA@<i>n</i>，用 GIA→HI 数据训练得到 HI@<i>n</i>+1；再固定 HI@<i>n</i>+1，用 HI→GIA 数据训练得到 GIA@<i>n</i>+1。每次只更新一个角色。两次更新各过自己的门：门 B-HI、门 B-GIA 在冻结规格下比较新旧权重。门不过，只回滚对应的角色，它的数据保留并标记 <span class="pp-tt">gate_failed</span>。</p>

<p><b>从教师到自己。</b>第 0 轮全部是教师数据。从第 1 轮起，学生系统对每个案例跑 ${h.m.inl(h.m.row(h.m.mi('N'), h.m.mo('='), h.m.mn('4')))} 次，通过核验的决策直接入选，即拒绝采样 ${h.c('star', 'restem')}。对失败的轨迹，由教师从第一个出错的步骤桥接一步，再由学生续跑；续跑通过核验，桥接片段才入库 ${h.c('futureb')}。我们追踪入选数据的教师依赖率</p>
${h.eq.dep(10)}
<p>并设一条硬指标：从第 2 轮起 ${h.m.inl(h.m.row(h.m.sub(h.m.mi('D'), h.m.mi('n')), h.m.mo('&lt;'), h.m.mn('0.5')))}，且留出集性能不下降。若到第 3 轮仍未达到，我们就把这一过程称为"自我改进"而不是 RSI。前几轮的复习数据一律用当前规格从运行日志重新导出，只保留在当前核验器下仍为正例的行，因此任何一批训练数据都不会混用不同的规格哈希。</p>

<p><b>为什么不用在线 RL。</b>与我们最接近、同时更新 harness 与权重的工作，都在经核验的轨迹上做微调 ${h.c('coharness', 'hzero')}；而多个角色从同一底座共同演化的工作，用的是在线策略梯度 RL，如 GRPO 或 Task-Relative REINFORCE++ ${h.c('grpo', 'agent0', 'sage', 'mae')}。我们选前者，理由有三。第一，这类 RL 以"每道题一条完整轨迹"为单位，几十个开发案例不足以支撑按组估计而不过拟合；按决策切分的数据则能把每个案例变成几十条训练样本。第二，这里的一条轨迹要检索开放网络、读网页、处理视频，需要几十次教师规模的调用；每次更新都为每个案例采一组轨迹，预算会被它占满，而拒绝采样每轮每个案例只需 ${h.m.inl(h.m.row(h.m.mi('N'), h.m.mo('='), h.m.mn('4')))} 条，且可以离线核验。第三，直接对模型核验器做优化，比只用它筛数据更容易打开下文所列的钻空子路径；RL 的不稳定还会叠加在循环本身的不稳定之上，曲线变了就无法归因。RL 作为预先登记的附加项保留，单独成臂报告：若到第 2–3 轮曲线走平，或教师依赖率一直高于 0.5，就加一组离线 DPO ${h.c('dpo')}，偏好对来自流程里现成的材料——同一状态下，未通过核验的学生决策对比通过核验的桥接或重采样决策——不需要额外采轨迹。</p>

<p><b>防止钻核验器的空子。</b>用模型核验器筛选自生成数据，会把学生推向"讨核验器喜欢"的写法：发现写得更少、更含糊，堆砌引用以触发宽松规则，过多地报 <span class="pp-tt">INCOMPLETE</span>，GIA 专派容易完成的任务。每一轮都报告以下各项并设上下限：必需项被已支持发现覆盖的比例；可回答案例上的 <span class="pp-tt">INCOMPLETE</span> 率；每条校准规则在"不该被豁免"的反例上的假接受率；以及相对 ${h.m.inl(h.m.sub(h.m.V, h.m.mt('ref')))} 的钻空子率。</p>

${h.sub('3.7', '每轮协议与防自欺措施', 'protocol')}
<div class="pp-algo"><div class="pp-algo-h"><b>算法 1</b> VRSI 的一轮。身份：（规格 <i>s</i><sub>n</sub>，LoRA<sub>HI</sub>@<i>n</i>，LoRA<sub>GIA</sub>@<i>n</i>，核验器 𝒱<sub>n</sub>）</div><ol>
<li><b>运行</b>：学生系统在 <span class="pp-tt">dev_train</span> 上运行（每案例 <i>N</i> 条），教师探针在同一批案例上运行。</li>
<li><b>核验</b>：全部动作、报告、派工、答案单元 → 账本 <span class="pp-tt">verdicts_r<i>n</i></span>。</li>
<li><b>诊断</b>：按"角色 × 单元 × 类型"汇总，附比率、校准结果和责任角色 → <span class="pp-tt">diagnosis_r<i>n</i></span>。</li>
<li><b>路由</b>：对有资格的前 <i>k</i> 格按式 (7) 判断；第 1–3 轮做双干预，为路由器打分。</li>
<li><b>内环</b>：≤ <i>K</i> 个单对象候选 → 回归验证 → 门 A → 确认 → <i>s</i><sub>n+1</sub>。</li>
<li><b>外环</b>（在 <i>s</i><sub>n+1</sub> 下）：产伙伴同步数据（学生自采、教师桥接）→ 训 HI → 门 B-HI → 训 GIA → 门 B-GIA。</li>
<li><b>回滚</b>未过门的部分；数据保留并标记 <span class="pp-tt">gate_failed</span>。</li>
<li><b>评测</b>：用录制回放的检索在 <span class="pp-tt">dev_eval</span> 上跑一次；写入 <span class="pp-tt">revision_log</span> 与 <span class="pp-tt">budget</span>。</li>
</ol></div>

<p><b>在构造上保证可归因。</b>每道门只评一类改动：门 A 固定权重；门 B-HI 与门 B-GIA 固定规格，每次只改一个角色。核验器的升版发生在轮与轮之间，从不与规格改动同轮，升版后重打账本，使错误曲线前后可比。每个改动的目标格、门控结果和确认结果都写入修订日志。</p>

<p><b>五份划分与固定的环境。</b>案例按事件或源视频分组，再划为 <span class="pp-tt">dev_train</span>（产数据）、<span class="pp-tt">dev_select</span>（门控）、<span class="pp-tt">dev_confirm</span>（确认）、<span class="pp-tt">dev_eval</span>（只画曲线）和锁定的 <span class="pp-tt">test</span>。门控会回滚失败的改动，画在门控集上的曲线天然只升不降，所以曲线只画在从不参与决策的划分上。开放网络的检索结果会随时间漂移，还可能搜到已发表的核查文章。为此，我们在首次运行时录下搜索结果、打开的页面和下载的媒体，所有对照组和轮次都回放同一份录制；检索按说法日期截断；工具结果做缓存；测试结果分别报告加与不加核查网站黑名单两组。</p>

<p><b>协议测量什么。</b>这套设计让每项负面结论都能被直接检验。<i>针对"自评不能预测提升"</i> ${h.c('s3gym')}，我们估计核验分变化与留出指标变化之间的剂量–反应关系：除门控本身产生的少数点之外，再用冻结权重的内环生成 100–200 个随机或刻意变坏的规格改动，用按轮次和种子聚类的分层模型分析。<i>针对"改动站不住"</i> ${h.c('harnessdev')}，我们报告每个被接受改动在 <span class="pp-tt">dev_confirm</span> 上的方向一致率，以及它在不同执行者（4B、9B、27B 学生和教师）之间方向是否一致。<i>针对"进步其实是恢复"</i> ${h.c('aspire')}，训练配方固定不变，模型能影响的只有哪些经核验的数据入选；每个检查点都同时与上一轮和底座比较。为了说明两个角色确实在互相帮助，我们评测 GIA 与 HI 检查点的交叉搭配，并报告协同项</p>
${h.eq.syn(11)}
<p>及其跨种子的置信区间，同时报告与教师搭档的搭配结果，以区分"真的变强"与"只是互相适应"。主要比较在同等预算（GPU 时、教师 token、核验调用）下进行：完整循环、只训练、只改 harness，以及纯蒸馏——同样的教师预算直接多产数据，不跑循环。</p>

${h.sub('3.8', '实现', 'impl')}
<p>两个角色共用 Qwen3.5-9B ${h.c('qwen35')}，这是一个原生支持图片和视频的多模态底座，每个角色各挂一套 LoRA；本地调试使用同系列 4B 模型。教师为 Qwen3.5-122B-A10B（FP8），占 8 张 A100-80GB 中的 4 张，其余留给学生采样与训练。核验器是本地部署、权重固定且不同于教师的模型；${h.m.inl(h.m.sub(h.m.V, h.m.mt('ref')))} 来自第三个模型家族。所有状态由同一个渲染器按统一的 16K token 档渲染，图片另计，训练长度上限 32K，整个研究期间不变。开发案例来自六个数据来源，覆盖网页与图文说法核验、视频篡改、片段检索和拷贝定位；EVID-Bench ${h.c('evid')} 作为主测试集锁定。每个能力维度另留一个在训练中完全不出现的数据集，用 0/20/50 条样例的适应曲线衡量迁移。</p>
`
  };
})();
