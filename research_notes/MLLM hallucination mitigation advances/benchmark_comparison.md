# Training-Free MLLM Hallucination Mitigation: Benchmark Comparison (2024-2026)

## What are the latest SOTA results on POPE, CHAIR, AMBER, MME-Hal benchmarks for training-free methods?

### Takeaway
No single method dominates all benchmarks. Results vary significantly by backbone model, decoding strategy, and token budget. The field has exploded from ~5 methods in early 2024 to 79+ training-free methods by late 2026, with the strongest recent contenders being CRoPS (TMLR 2026), AttnReal (IJCV 2026), DeCo (ICLR 2025), and AGLA (CVPR 2025) on CHAIR/POPE, though direct cross-paper comparison is unreliable due to differing evaluation setups.

### Cited Findings

**CHAIR Results (LLaVA-1.5-7B, lower is better)**

| Method | Venue | CHAIR_S | CHAIR_I | Recall | Decoding |
|--------|-------|---------|---------|--------|----------|
| Regular (baseline) | — | 51.0 | 15.2 | 75.2 | Multinomial |
| VCD | CVPR 2024 | 51.0 | 14.9 | 77.2 | Multinomial |
| OPERA | CVPR 2024 | 47.0 | 14.6 | 78.5 | Beam |
| DoLa | — | 57.0 | 15.9 | 78.2 | Multinomial |
| AGLA | CVPR 2025 | 43.0 | 14.1 | 78.9 | Multinomial |
| Greedy baseline | — | 46.2-46.6 | 13.4-13.8 | 75.9 | Greedy |
| PAI (greedy) | ECCV 2024 | 23.4-24.8 | 6.6-6.9 | 75.7 | Greedy |
| PAI (beam) | ECCV 2024 | 21.8 | 5.6 | — | Beam |
| DeCo (greedy) | ICLR 2025 | 37.8 | 11.1 | — | Greedy |

Sources:
- AGLA CHAIR numbers from Table 3 — [AGLA paper](https://arxiv.org/html/2406.12718v3)
- PAI CHAIR numbers from Tables 1, S3 — [PAI paper](https://arxiv.org/html/2407.21771v1)
- DeCo CHAIR numbers from GitHub reproduction — [DeCo paper](https://arxiv.org/pdf/2410.11779); note these are from a third-party README, not the paper's own table
- CRoPS claims "lowest CHAIR scores across all LVLMs" with 15-25% relative reduction vs M3ID, but absolute numbers were not extracted from the snippets — [CRoPS paper](https://arxiv.org/pdf/2601.00659)
- AttnReal claims best hallucination performance across most model/decoding settings on CHAIR — [AttnReal paper](https://arxiv.org/pdf/2503.08342)

**POPE Results (LLaVA-1.5-7B, Accuracy / F1, higher is better)**

| Method | Venue | Random Acc/F1 | Popular Acc/F1 | Adversarial Acc/F1 |
|--------|-------|---------------|----------------|---------------------|
| Regular | — | 83.49 / 82.28 | 79.98 / 79.34 | 76.03 / 76.26 |
| DoLa | — | 84.78 / 84.19 | 79.75 / 80.61 | 76.32 / 76.16 |
| VCD | CVPR 2024 | 86.84 / 86.83 | 82.65 / 83.37 | 77.31 / 79.28 |
| OPERA | CVPR 2024 | 87.53 / 86.45 | 84.21 / 83.50 | 80.88 / 80.69 |
| AGLA | CVPR 2025 | 88.54 / 87.71 | 85.14 / 84.68 | 81.13 / 81.36 |
| PAI (greedy, avg) | ECCV 2024 | — | — | — |
| PAI (greedy, overall avg) | ECCV 2024 | 86.13 (avg) | — | — / 86.42 (avg F1) |
| DeCo (greedy) | ICLR 2025 | — | — | — / 86.7 (F1) |

Sources:
- AGLA POPE from Table 1 — [AGLA paper](https://arxiv.org/html/2406.12718v3)
- PAI POPE from Table S3 (3-split average: Acc 86.13, F1 86.42 for 7B) — [PAI paper](https://arxiv.org/html/2407.21771v1)
- DeCo POPE F1 of 86.7 from paper table — [DeCo paper](https://arxiv.org/pdf/2410.11779)

**AMBER Results (LLaVA-1.5, greedy, from LISA paper reproduction)**

| Method | CHAIR | Hal. | Cog. |
|--------|-------|------|------|
| Vanilla | 8.2 | 34.3 | 4.0 |
| DeCo | 6.6 | 28.1 | 2.8 |
| CAAC (2025) | claims ~4% lower Hal rate vs best baseline | — | — |

Source: DeCo AMBER numbers reproduced by LISA (arXiv 2507.19110) — [LISA paper](https://arxiv.org/pdf/2507.19110); CAAC claim — [CAAC paper](https://arxiv.org/pdf/2505.21472)

**MME Perception Results (LLaVA-1.5-7B)**

| Method | Perception Total |
|--------|-----------------|
| Regular | 1279.19 |
| DoLa | 1232.54 |
| OPERA | 1332.79 |
| VCD | 1363.96 |
| AGLA | 1472.66 |

Source: AGLA Table A6 — [AGLA paper](https://arxiv.org/html/2406.12718v3)

### Inferences
- PAI shows the strongest CHAIR improvements (greedy: 46.6 to 24.8 CHAIR_S, a ~47% relative reduction), but its POPE gains are more modest.
- AGLA is the strongest on POPE among methods with verified per-split numbers, gaining ~5% accuracy on the adversarial split.
- On MME perception, AGLA gains ~193 points over the baseline, the largest jump among training-free methods with verified numbers.
- DeCo's numbers are competitive but partially reproduced by third parties rather than the original paper.
- CRoPS and AttnReal claim SOTA on CHAIR but their exact LLaVA-1.5 numbers were not extractable from search snippets.

### Gaps
- CRoPS absolute CHAIR/POPE numbers on LLaVA-1.5 could not be confirmed from search results; need to check Table 1 and Table 3 of arXiv 2601.00659.
- AttnReal absolute CHAIR/POPE numbers not available from snippets; need to check the PDF tables of arXiv 2503.08342.
- AMBER results are sparse across the field; most papers do not report AMBER.
- MME-Hal (hallucination-specific MME subset) numbers were not found for most methods.
- No unified leaderboard exists for training-free methods; all comparisons are from individual papers.


## How do methods published after CVG (Fu et al., IEEE TIP 2025) compare on these benchmarks?

### Takeaway
The CVG paper could not be located in web searches despite multiple query variations. The paper may use a different title or acronym in public databases. All 2025-2026 methods listed below represent the post-2024 landscape that would surround CVG's publication window.

### Cited Findings

**CVG paper search results:**
- Searches for "complementary visual grounding" + hallucination, "CVG" + Fu + IEEE TIP 2025, and variations returned no matching paper — [Search attempts on arxiv.org and Google Scholar]
- The closest conceptual match is AdaVBoost (arXiv 2602.13600), which uses "Visual Grounding Entropy" as a complementary signal for hallucination risk estimation — [AdaVBoost](https://arxiv.org/html/2602.13600)
- Another nearby match is VDGD (arXiv 2405.15683), which grounds LVLM responses in generated visual descriptions — [VDGD](https://arxiv.org/html/2405.15683v3)

**Key 2025-2026 methods that would compare against CVG:**

| Method | Venue | Type | Key Claim |
|--------|-------|------|-----------|
| DeCo | ICLR 2025 | Contrastive Decoding | Dynamic correction at each decoding step |
| AGLA | CVPR 2025 | Attention Intervention | Assembly of global and local attention |
| Nullu | CVPR 2025 | Weight Editing | HalluSpace projection (not strictly training-free) |
| VISTA | ICML 2025 | Logit Steering | Hidden token visual info steering |
| MARINE | ICML 2025 | Classifier-Free Guidance | Image-grounded guidance (uses auxiliary model) |
| ICT | CVPR 2025 | Attention Intervention | Image-object cross-level intervention |
| CRoPS | TMLR 2026 | Prompt Steering | Prompt-based hallucination mitigation |
| AttnReal | IJCV 2026 | Attention Reallocation | Zero-cost attention redistribution to visual tokens |
| ResDec | CVPR 2026 | Contrastive Decoding | History-aware residual guidance |
| VGA | CVPR 2026 | Attention Intervention | Vision-guided attention steering |
| FLB | CVPR 2026 | Logit Boosting | First logit boosting via visual grounding |

Sources: [Awesome Train-Free MLLM Hallucination Mitigation](https://github.com/zhongjingyu1/Awesome-Train-Free-MLLM-LVLM-Hallucination-Mitigation)

### Inferences
- The field has grown rapidly: 24 training-free methods in 2024, 22 in 2025, and 33 in 2026 (through October).
- Methods have shifted from pure contrastive decoding (VCD, M3ID) toward attention intervention and representation steering, which can be more targeted.
- Multiple CVPR 2026 papers exist in this space (at least 15), indicating it remains a very active research area.

### Gaps
- The CVG paper itself could not be found. Recommend checking IEEE Xplore directly, or verifying the exact title and arXiv ID. The paper may appear under a different abbreviation.
- Without CVG's specific benchmark numbers, direct comparison is not possible. The user should provide the paper's arXiv ID or exact title.


## What methods have been proposed specifically for LLaVA-1.5, LLaVA-Next, Qwen2-VL, InternVL architectures?

### Takeaway
Most training-free methods (2024-2025) were primarily developed and tested on LLaVA-1.5 (7B/13B). By 2026, newer methods are expanding to Qwen2-VL, Qwen2.5-VL, and InternVL, reflecting the shift in the MLLM ecosystem.

### Cited Findings

**LLaVA-1.5 (most extensively tested):**
- Nearly every method in the field uses LLaVA-1.5-7B as the primary testbed: OPERA, VCD, AGLA, PAI, DeCo, CRoPS, AttnReal, Nullu, HALC, M3ID, IBD, CODE, ICD, LCD, etc. — [Awesome Train-Free list](https://github.com/zhongjingyu1/Awesome-Train-Free-MLLM-LVLM-Hallucination-Mitigation)
- LLaVA-1.5-13B is a common secondary evaluation: PAI reports CHAIR_S 33.0 / CHAIR_I 9.2 on 13B greedy (vs 44.0/12.7 baseline) — [PAI paper](https://arxiv.org/html/2407.21771v1)
- AGLA reports 13B POPE on COCO adversarial: 84.56 / 83.11 (Acc/F1) vs 80.00 / 78.62 baseline — [AGLA paper](https://arxiv.org/html/2406.12718v3)

**LLaVA-Next/LLaVA-NeXT:**
- ASCD (AAAI 2026) tested on LLaVA-NeXT 7B, claiming up to 38.2% reduction on POPE/CHAIR/MMHal — [ASCD, arXiv 2506.14766](https://arxiv.org/abs/2506.14766)
- AdaVBoost/SAVAA tested on LLaVA-NeXT-7B — [AdaVBoost](https://arxiv.org/html/2602.13600)

**Qwen2-VL / Qwen2.5-VL:**
- ILVAD (arXiv 2605.20965, May 2026): training-free method using inter-layer attention discrepancy, tested on Qwen2-VL-7B — [ILVAD](https://www.alphaxiv.org/abs/2605.20965)
- AIMS (arXiv 2610.11907, October 2026): adaptive multi-context steering tested on Qwen2.5-VL-3B, reports CHAIR_S 40.8 vs 56.0 baseline — [AIMS](https://alphaxiv.org/abs/2610.11907v1)
- ASCD (AAAI 2026) also tested on Qwen2.5-VL-Instruct — [via web search results]
- SAKED (arXiv 2602.09825): training-free decoding method tested on Qwen2.5-VL — [SAKED](https://www.alphaxiv.org/abs/2602.09825)
- A saliency-based method (ICLR 2026) uses Qwen2-VL-7B for analysis — [via web search results]

**InternVL:**
- AdaVBoost/SAVAA tested on InternVL3.5-8B alongside LLaVA-NeXT and Qwen3-VL — [AdaVBoost](https://arxiv.org/html/2602.13600)
- No other training-free methods were found with explicit InternVL evaluation in search results.

### Inferences
- LLaVA-1.5 remains the standard benchmark model. Any new method should report LLaVA-1.5-7B results for comparability.
- The transition to Qwen2-VL and InternVL is happening in 2026 papers but is not yet standard. Papers submitted in mid-2026 are more likely to include these models.
- Methods designed around LLaVA's attention architecture (like OPERA's attention-sink detection) may need adaptation for different architectures.

### Gaps
- InternVL-specific training-free results are extremely sparse (only AdaVBoost found).
- No benchmark comparison table was found that includes the same method across LLaVA-1.5, LLaVA-Next, Qwen2-VL, and InternVL simultaneously.


## What is the current landscape: VCD, OPERA, PAI, CVG, and what came after?

### Takeaway
The training-free MLLM hallucination mitigation field has evolved through three generations: (1) foundational contrastive decoding (VCD, OPERA, M3ID, 2024), (2) attention intervention and grounding methods (PAI, AGLA, DeCo, 2024-2025), and (3) a diverse explosion of methods combining representation steering, attention reallocation, token manipulation, and prompt engineering (2025-2026, 79+ methods total).

### Cited Findings

**Taxonomy of training-free methods** (from curated lists and surveys):

1. **Contrastive Decoding (CD):** Modify output logits by contrasting with a "corrupted" or "text-only" distribution.
   - VCD (CVPR 2024): Visual contrastive decoding with distorted image — [VCD](https://arxiv.org/abs/2311.16922)
   - M3ID (CVPR 2024): Multi-modal hallucination control via visual information grounding — [M3ID](https://arxiv.org/abs/2403.14003)
   - ICD (Findings ACL 2024): Instruction contrastive decoding
   - LCD (Findings ACL 2024): Language-contrastive decoding
   - IBD (CVPR 2025): Image-biased decoding — [IBD](https://arxiv.org/abs/2402.18476)
   - DeCo (ICLR 2025): Dynamic correction decoding — [DeCo](https://arxiv.org/abs/2410.11779)
   - ResDec (CVPR 2026): History-aware residual guidance — [ResDec, arXiv 2602.01047]
   - ASCD (AAAI 2026): Attention-steerable contrastive decoding — [ASCD, arXiv 2506.14766]

2. **Inference-Time Intervention (IT):** Directly modify attention patterns, hidden states, or activations during inference.
   - OPERA (CVPR 2024): Over-trust penalty and retrospection-allocation on beam search — [OPERA](https://arxiv.org/abs/2311.17911)
   - PAI (ECCV 2024): Scale up image attention weights + subtract text-only logits — [PAI](https://arxiv.org/abs/2407.21771)
   - AGLA (CVPR 2025): Assembly of global and local attention — [AGLA](https://arxiv.org/abs/2406.12718)
   - Nullu (CVPR 2025): HalluSpace weight projection — [Nullu, arXiv 2412.13817]
   - AttnReal (IJCV 2026): Zero-cost attention reallocation from sink to visual tokens — [AttnReal](https://arxiv.org/abs/2503.08342)
   - CRoPS (TMLR 2026): Prompt-based steering — [CRoPS](https://arxiv.org/abs/2601.00659)
   - VGA (CVPR 2026): Vision-guided attention
   - FLB (CVPR 2026): First logit boosting via visual grounding — [FLB, arXiv 2604.00455]

3. **Post-hoc Verification:** Generate response, then verify/correct against visual evidence.
   - Woodpecker (SCIS 2024): Uses visual expert tools to correct hallucinations — [Woodpecker, arXiv 2310.16045]
   - LogicCheckGPT (Findings ACL 2024): Logical closed-loop verification
   - SelfVal (arXiv 2026): Self-validation for object hallucination — [SelfVal, arXiv 2601.22451]

4. **Auxiliary Model Methods:** Use external models (diffusion, CLIP, retrieval) alongside the MLLM.
   - MARINE (ICML 2025): Classifier-free guidance with image grounding
   - HALC (ICML 2024): Adaptive focal-contrast decoding with grounding models
   - EnAR (CVPR 2026): Frozen diffusion prior + contrastive decoding

Source: Comprehensive list from — [Awesome Train-Free MLLM/LVLM Hallucination Mitigation](https://github.com/zhongjingyu1/Awesome-Train-Free-MLLM-LVLM-Hallucination-Mitigation) (last updated 2026-10-12, 79 entries)

**Growth timeline:**
- 2024: 24 methods (VCD, OPERA, PAI, HALC, M3ID, ICD, LCD, CODE, ESREAL, GTHM, DAMRO, etc.)
- 2025: 22 methods (DeCo, AGLA, Nullu, VISTA, MARINE, ICT, IBD, DeGF, MAI, etc.)
- 2026: 33 methods through October (AttnReal, CRoPS, ResDec, VGA, FLB, ASCD, etc.)

Source: Count from — [Awesome Train-Free list](https://github.com/zhongjingyu1/Awesome-Train-Free-MLLM-LVLM-Hallucination-Mitigation)

**Key trends in 2025-2026:**
- A 2025 paper (arXiv 2503.08342) splits mitigation into training-based vs training-free, noting that training-free methods incur inference overhead from tracking multiple candidates or decoding steps — [AttnReal](https://arxiv.org/pdf/2503.08342)
- A 2026 paper (arXiv 2603.10360) groups training-free methods into "vision enhancement" and "text-inertia suppression" — [One Token Two Fates](https://arxiv.org/pdf/2603.10360)
- Findings of EMNLP 2026 notes that training-free inference-time intervention has emerged as an attractive route given the cost of retraining — [CAS paper](https://aclanthology.org/2026.findings-emnlp.1156.pdf)

### Inferences
- The field has tripled in size from 2024 to 2026, with no signs of slowing.
- Attention intervention (IT) has become the dominant approach, overtaking contrastive decoding.
- The trend is toward zero or near-zero additional inference cost (AttnReal, FLB), moving away from methods that require multiple forward passes (VCD, M3ID).
- CVPR 2026 alone has at least 15 training-free hallucination papers, making it the most active venue.

### Gaps
- No head-to-head comparison exists that covers all major methods under identical conditions.
- Inference cost/speed comparison across methods is rarely reported; most papers only compare hallucination rates.


## Are there survey papers or benchmark papers that compare multiple methods?

### Takeaway
Two broad surveys exist (arXiv 2404.18930 covers MLLM hallucination generally; EMNLP 2024 Findings covers LLM+MLLM hallucination), plus one curated GitHub list focused specifically on training-free methods. No single benchmark paper systematically compares all training-free methods under identical conditions.

### Cited Findings

**Surveys:**
- "Hallucination of Multimodal Large Language Models: A Survey" (arXiv 2404.18930) — covers causes, benchmarks, and mitigation approaches including training-free methods like MARINE (classifier-free guidance). Background-level coverage, not a benchmark comparison — [Survey](https://arxiv.org/pdf/2404.18930)
- "A Comprehensive Survey of Hallucination in Large Language, Vision, and Language Models" (EMNLP 2024 Findings) — broader scope covering both LLMs and MLLMs — [EMNLP Survey](https://aclanthology.org/2024.findings-emnlp.685.pdf)

**Curated lists:**
- Awesome Train-Free MLLM/LVLM Hallucination Mitigation (GitHub) — the most comprehensive resource for training-free methods specifically, listing 79 papers from 2024-2026 with venue, type classification, and common benchmarks — [GitHub](https://github.com/zhongjingyu1/Awesome-Train-Free-MLLM-LVLM-Hallucination-Mitigation)
- Awesome-MLLM-Hallucination (GitHub, showlab) — broader list covering both training-based and training-free methods — [GitHub](https://github.com/showlab/Awesome-MLLM-Hallucination)

**Papers with multi-method comparison tables:**
- The "Magnifying Glass" paper (arXiv 2503.10183) compares greedy, VCD, VDD, M3ID, IBD, PAI, and its own method on MME and POPE, though gains on these benchmarks are marginal (POPE: 84.40-84.60 range) — [Magnifying Glass](https://arxiv.org/pdf/2503.10183)
- AGLA (arXiv 2406.12718) provides one of the more thorough comparison tables: Regular, DoLa, VCD, OPERA, and AGLA on POPE (3 splits), CHAIR, and MME for LLaVA-1.5-7B and InstructBLIP — [AGLA](https://arxiv.org/html/2406.12718v3)
- LISA (arXiv 2507.19110) provides a reproduction of multiple training-free methods on AMBER for LLaVA-1.5 — [LISA](https://arxiv.org/pdf/2507.19110)

### Inferences
- The lack of a unified benchmark comparison is a significant gap in the field. Each paper compares against a subset of prior work, making cross-paper comparison unreliable.
- A standardized evaluation protocol (fixed backbone, decoding settings, token budget, splits) would be extremely valuable and could be a contribution in itself.
- The curated GitHub lists serve as the de facto directory of the field, though they do not include benchmark numbers.

### Gaps
- No dedicated benchmark paper was found that runs all major training-free methods under identical conditions.
- No official leaderboard exists for POPE, CHAIR, or AMBER that tracks training-free methods specifically.
- Survey coverage of 2025-2026 methods is limited; the existing surveys are from 2024.


## What are the latest results on model-based hallucination evaluation (MM-Hal, HallusionBench)?

### Takeaway
MMHal-Bench and HallusionBench are widely used but have no centralized 2025-2026 leaderboard for training-free decoding methods. Available numbers come from aggregator sites with inconsistent metrics, and most training-free papers do not report these benchmarks.

### Cited Findings

**MMHal-Bench:**
- Original design: 96 curated image-question pairs across 8 semantic categories, scored by GPT-4 and human evaluation. Open-ended format, unlike POPE's yes/no — [MMHal-Bench paper](https://arxiv.org/pdf/2309.14525)
- PAI paper includes MMHal-Bench results only as a bar chart (Figure 6), with no numeric table. The paper states PAI improves the overall average across 8 categories but gains are absent for comparison and relation questions — [PAI paper](https://arxiv.org/html/2407.21771v1)
- ASCD (AAAI 2026) claims hallucination reduction on MMHal-Bench of up to 38.2% (combined with POPE and CHAIR) — [via web search results]
- A WizWand aggregator (updated May 2026) shows a "BERTScore" leaderboard with TIGER at top, but this metric is different from the standard MMHal score — [WizWand MMHal](https://www.wizwand.com/sota/image-plus-text-to-text-hallucination-evaluation-on-mmhal-bench)

**HallusionBench:**
- Tests two failure modes: language hallucination (relying on prior knowledge instead of image) and visual illusion (misreading image content) — [benchmarklist](https://benchmarklist.com/benchmarks/hallusionbench/)
- One aggregator (sampled 2026-05-06) lists MiMo-VL-7B at ~63.8% overall, BlueLM-2.6-3B at ~63.1%. Top-8 models span only 7.9 percentage points — [WizWand HallusionBench](https://www.wizwand.com/sota/multimodal-hallucination-evaluation-on-hallusionbench)
- A separate leaderboard shows ThinkLite-VL at 70.7 "Hallucination Score" (December 2025), but this metric differs from overall accuracy — [WizWand](https://www.wizwand.com/sota/multimodal-hallucination-evaluation-on-hallusionbench)
- These leaderboard entries are for model-level comparisons (different MLLMs), not training-free decoding methods applied to a fixed backbone.

**AMBER:**
- Covers both generative and discriminative tasks, does not rely on LLM judge. Derives a combined AMBER Score blending CHAIR and F1 — [via search results on benchmark descriptions]
- CAAC (arXiv 2505.21472) reports reducing hallucination rate by ~4% (CHAIR) and ~1.8% (AMBER) relative to best baseline — [CAAC paper](https://arxiv.org/pdf/2505.21472)
- DeCo reproduced on AMBER by LISA paper: Vanilla 8.2/34.3/4.0 (CHAIR/Hal/Cog) vs DeCo 6.6/28.1/2.8 — [LISA paper](https://arxiv.org/pdf/2507.19110)

### Inferences
- Model-based evaluation (MMHal, HallusionBench) is less commonly reported for training-free decoding methods than discriminative benchmarks (POPE) or captioning benchmarks (CHAIR).
- The leaderboards that exist track different MLLMs (GPT-4V, Gemini, etc.), not decoding interventions on a fixed model, making them less relevant for training-free method comparison.
- AMBER is gaining traction as a more comprehensive benchmark that avoids LLM-judge dependency.

### Gaps
- No consolidated table of MMHal-Bench scores exists for training-free methods across VCD, OPERA, PAI, AGLA, DeCo, etc.
- HallusionBench numbers for training-free methods (as opposed to model-level) were not found.
- MMHal-Bench's reliance on GPT-4 scoring introduces its own variability across time (GPT-4 versions change).


## Appendix: Complete List of Training-Free Methods (2024-2026)

The following 79 methods are cataloged in the [Awesome Train-Free MLLM/LVLM Hallucination Mitigation](https://github.com/zhongjingyu1/Awesome-Train-Free-MLLM-LVLM-Hallucination-Mitigation) repository (last updated 2026-10-12). Types: CD = Contrastive Decoding, IT = Inference-Time Intervention, Post-hoc = Post-generation correction, Aux = Uses auxiliary model/tool.

### 2026 (33 methods)
- ResDec (CVPR 2026, CD), PND (CVPR 2026, CD), CausalLens (CVPR 2026, IT), VGA (CVPR 2026, IT), LAD (ACL 2026, IT), VLI (ACL 2026, IT), CEI (Findings ACL 2026, IT), HAVAE (ACL 2026, IT), ACG (CVPR 2026, IT), DPA (CVPR 2026, IT), HARM-VEED (CVPR 2026, IT), SVC-CRC (CVPR 2026, IT), SISS (CVPR 2026, IT), EnAR (CVPR 2026, Aux), CEC (ICMR 2026, CD), PAR (ACL 2026, IT/OCR), SEASON (CVPR 2026, CD/video), GEASS (arXiv, IT), CRoPS (TMLR 2026, IT), ASCD (AAAI 2026, CD), IRI (AAAI 2026, IT), SAVER (AAAI 2026, IT), TAF (AAAI 2026, IT), CAST (arXiv, IT), SelfVal (arXiv, Post-hoc), AttnReal (IJCV 2026, IT), AQAH (IEEE TCSVT 2026, Aux), KVSmooth (CVPR 2026), FLB (CVPR 2026, Decoding), MAD (CVPR 2026, CD), VECD (IEEE TMM 2026, CD), AIMS (arXiv, IT), FADE (arXiv, IT)

### 2025 (22 methods)
- IMCCD (arXiv, CD), MINT (arXiv, IT), IFCD (arXiv, CD), DAC (arXiv, IT), DeGF (ICLR, IT), MAI (ICLR, IT), VISTA (ICML, IT), MARINE (ICML, Aux), MemVR (ICML, IT), ICT (CVPR, IT), Nullu (CVPR, IT), VASparse (CVPR, IT), AGLA (CVPR, IT), FarSight (CVPR, IT), INTER (ICCV, IT), ONLY (ICCV, CD), MFCD (EMNLP, CD), SPIN (EMNLP, IT), ASD (ACL, IT), CLAIM (ACL, IT), RVCD (Findings ACL, CD), RUR (IEEE TMM, Post-hoc)

### 2024 (24 methods)
- VCD (CVPR, CD), OPERA (CVPR, Decoding), M3ID (CVPR, CD), HALC (ICML, CD), PAI (ECCV, IT), ESREAL (ECCV, IT), ICD (Findings ACL, CD), LCD (Findings ACL, CD), Less is More (ACL, Decoding), GTHM (EMNLP, Decoding), DAMRO (EMNLP, IT), CODE (NeurIPS, CD), CGD (ICLR Workshop, CD), SID (arXiv/ICLR 2025, CD), VTI (arXiv/ICLR 2025, IT), CausalMM (arXiv/ICLR 2025, IT), DeCo (arXiv/ICLR 2025, CD), SumGD (arXiv/NAACL 2025, CD), IBD (arXiv/CVPR 2025, CD), Pensieve (arXiv, IT), VaLiD (arXiv, CD), ConVis (arXiv/AAAI 2025, CD), Woodpecker (SCIS, Post-hoc), LogicCheckGPT (Findings ACL, Post-hoc)
