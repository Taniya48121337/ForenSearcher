# Targeted Paper Search — Full Results

Timestamp: 2026-08-16 Asia/Shanghai

## Search errors surfaced

- OpenReview connector: `openreview not installed. pip install openreview-py`
- OpenAlex: `504 Server Error: Gateway Timeout`
- 其余 CLI 学术 API 在沙箱中最初发生 Windows socket 权限错误；批准访问后出现持续 rate limiting。没有把这些失败返回的 0 篇视作负面证据。
- 最终结果由 arXiv/OpenReview 公开网页的原始摘要、HTML 正文和 PDF 文本补齐。

## arXiv (13 papers)

| # | Title | Date | Venue | Citations |
|---|---|---|---|---:|
| 1 | [OpenVisTool](https://arxiv.org/abs/2608.08557) | 2026-08 | arXiv preprint | N/A |
| 2 | [CausalFlow](https://arxiv.org/abs/2605.25338) | 2026-05 | arXiv preprint | N/A |
| 3 | [From Faulty Memories to Corrected Actions](https://arxiv.org/abs/2608.10502) | 2026-08 | arXiv preprint | N/A |
| 4 | [Bridge Evidence](https://arxiv.org/abs/2607.15253) | 2026-07 | arXiv preprint | N/A |
| 5 | [Causal Agent Replay](https://arxiv.org/abs/2606.08275) | 2026-06 | arXiv preprint | N/A |
| 6 | [STAMP](https://arxiv.org/abs/2607.11172) | 2026-07 | arXiv preprint | N/A |
| 7 | [COVERT](https://arxiv.org/abs/2604.09813) | 2026-04 | arXiv preprint | N/A |
| 8 | [MetaRAG](https://arxiv.org/abs/2509.09360) | 2025-11 | ECAI 2025 workshop / arXiv | N/A |
| 9 | [From Agent Traces to Trust](https://arxiv.org/abs/2606.04990) | 2026-06 | arXiv preprint | N/A |
| 10 | [From Fluent to Verifiable](https://arxiv.org/abs/2602.13855) | 2026-02 | arXiv preprint | N/A |
| 11 | [VidForensics-M1](https://arxiv.org/abs/2608.11201) | 2026-08 | arXiv preprint | N/A |
| 12 | [VIGIL](https://arxiv.org/abs/2603.21526) | 2026-03 | arXiv preprint | N/A |
| 13 | [Learning From Failure](https://arxiv.org/abs/2402.11651) | 2024-02 | arXiv preprint | N/A |

## OpenReview (3 papers)

| # | Title | Date | Venue | Citations |
|---|---|---|---|---:|
| 1 | [EviPath](https://openreview.net/forum?id=alNQNzaWhI) | 2025-09 | NeurIPS 2025 Efficient Reasoning Workshop / ICLR 2026 submission record | N/A |
| 2 | [HardGen](https://openreview.net/forum?id=WLmL6YWM7t) | 2026-01 | ACL ARR 2026 submission | N/A |
| 3 | [InfCycle](https://openreview.net/forum?id=CD2wgg9RQD) | 2024-09 | ICLR 2025 withdrawn submission | N/A |

## Other API sources

- Semantic Scholar: API search failed/rate-limited; no independent rows were accepted from its failed response.
- OpenAlex: 504; no independent rows.
- Crossref: first broad query returned mostly off-topic results; all were removed in method-signature dedup/triage.
- DBLP: initial socket failure; no independent rows.
- Model Knowledge: 0 unverified additions; no citation was added solely from memory.

# Summary

## Overview

Queries covered source-grounded video-forensic agent trajectories, verified tool-use trajectory synthesis, and claim/evidence mutation with counterfactual replay. Sixteen deduplicated papers from 2024–2026 survived relevance triage.

## Trends

Work shifted rapidly in 2026 from outcome-only filtering to: whole-trace causal utility, executed counterfactual attribution, provenance-guided step credit, controlled tool-output perturbations, and selective state repair. The closest results are very recent—several appeared in July–August 2026—so novelty claims are unusually time-sensitive.

## Key themes

1. **Causal trajectory quality**：OpenVisTool、Bridge Evidence、CAR。
2. **Failure-to-repair supervision**：CausalFlow、Learning From Failure、HardGen。
3. **Provenance and local credit**：STAMP、From Faulty Memories、From Agent Traces to Trust。
4. **Controlled perturbation / metamorphic testing**：COVERT、MetaRAG。
5. **Evidence-to-trajectory synthesis**：EviPath、InfCycle。
6. **Video forensic grounding**：VidForensics-M1、VIGIL。

## Keywords frequency

| Keyword | Count |
|---|---:|
| evidence / provenance | 12 |
| trajectory / trace | 11 |
| verification / verifier | 10 |
| intervention / mutation / perturbation | 7 |
| repair / recovery / failure | 7 |

## Most cited by accepted paper

当前检索接口未可靠返回 citation counts；为避免伪造，不做数值排名。

| Rank | Title | Year | Citations |
|---:|---|---:|---:|
| — | Citation metadata unavailable from successful primary-source retrieval | — | N/A |

## Most cited by first author

当前检索接口未可靠返回 citation counts；为避免伪造，不做作者排名。

| Rank | Author | Papers in set | Total citations |
|---:|---|---:|---:|
| — | N/A | N/A | N/A |

## Recommendations for reading

1. **OpenVisTool**：最接近 visual trajectory quality 的问题设置。
2. **CausalFlow**：最接近 intervention + replay + repair supervision 的机制。
3. **MetaRAG**：最接近 atomic claim mutation 与 metamorphic oracle。
4. **From Faulty Memories to Corrected Actions**：最接近 provenance graph、independent support 与 selective replay。
5. **STAMP**：最接近 evidence graph 到 action-level training signal 的映射。
