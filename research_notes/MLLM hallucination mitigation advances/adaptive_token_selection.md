# Adaptive Visual Token Selection Methods for MLLM Hallucination Mitigation (2024-2026)

## What methods use entropy-based or adaptive alpha for visual token selection in MLLMs?

### Takeaway
Entropy-based token budgeting has emerged primarily in video MLLMs (AdaptToken), while image-domain methods use attention-score-based adaptive thresholds (AdaptPrune, AgilePruner) or learned per-layer pruning ratios (ATP-LLaVA). No published method yet combines entropy-based alpha with the CVG-style cross+self attention scoring formula for hallucination mitigation specifically.

### Cited Findings
- **AdaptToken** (arXiv 2603.28696, 2026) is a training-free method for long video that uses response entropy as a global signal for token budgeting. Each frame group's budget is B_g = B * Softmax({C_1..C_G}/tau), where C_g is the negative mean token entropy over the lowest-10% tokens and tau=2. More confident (lower-entropy) groups receive more tokens. Individual tokens are scored with r_v = max_t sum_h Attn_h(q_t, k_v), summing over attention heads and taking the max over text queries. Top-B_g tokens per group are kept, then 10% most mutually similar tokens are removed. On Qwen2.5-VL 7B: VideoMME 70.5, MLVU 76.8, LongVideoBench 65.2, LVBench 54.8 (vs base 65.4/70.2/59.5/45.3). — [arXiv](https://arxiv.org/abs/2603.28696)
- **QTSplus** (arXiv 2511.11910) scores vision tokens with cross-attention against the text query, then uses the entropy of the normalized relevance distribution (along with other statistics) to set how many tokens to keep per frame. — [arXiv](https://arxiv.org/abs/2511.11910)
- **AdaptPrune** (arXiv 2503.08019) is training-free and reframes visual token pruning as Non-Maximum Suppression (NMS). It starts from attention scores and adds spatial distance and token similarity through an adaptive suppression step. — [arXiv](https://arxiv.org/abs/2503.08019)
- **AgilePruner** (arXiv 2603.01236, 2026) proposes an adaptive similarity threshold tied to image complexity. Tokens are visited in descending attention order, and all candidate tokens whose cosine distance to the selected token is smaller than the adaptive threshold are pruned. — [arXiv](https://arxiv.org/abs/2603.01236)
- **ATP-LLaVA** (CVPR 2025) inserts a learned module between decoder layers that scores visual tokens from the previous layer's self-attention. Two lightweight heads learn separate thresholds for redundancy and spatial pruning, adapting per layer and per input. This requires training the pruning heads (not fully training-free). — [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Ye_ATP-LLaVA_Adaptive_Token_Pruning_for_Large_Vision_Language_Models_CVPR_2025_paper.pdf)
- **VisionDrop** (AAAI 2025) scores tokens using only visual self-attention rather than text-guided relevance. Reports that visual-only scoring generally beats text-guided scoring, especially under aggressive pruning. — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/39949/43910)

### Inferences
- Entropy-based methods are currently oriented toward video (where frame-level budgeting is natural). Adapting AdaptToken's entropy-based budgeting to image-domain hallucination mitigation (e.g., using response entropy to set the pruning aggressiveness per image) is a plausible and unexplored extension.
- The separation between "how many tokens to keep" (budget) and "which tokens to keep" (scoring) is a useful decomposition. Entropy addresses the budget question; attention scores address the selection question.
- VisionDrop's finding that visual-only self-attention outperforms text-guided cross-attention for token importance is counterintuitive and directly relevant to CVG's a = a_cross + a_self formula.

### Gaps
- No published method combines entropy-based adaptive alpha with CVG's specific cross+self attention scoring for image-domain hallucination mitigation.
- AdaptToken's benchmarks are video understanding tasks, not hallucination benchmarks (POPE, CHAIR). Whether its entropy signal transfers to hallucination settings is unknown.


## Are there approaches that use attention pattern analysis to identify text-containing patches vs object patches?

### Takeaway
Recent work has identified specialized "OCR heads" in MLLMs that attend specifically to text-containing image patches, distinct from object-attending heads. ASCD exploits this by amplifying text-centric heads and dampening others, achieving hallucination reduction while improving TextVQA performance.

### Cited Findings
- **OCR Heads** (arXiv 2505.15865, also EMNLP 2025): The authors identify attention heads that specifically attend to image patches containing text and read them out. Their OCR score measures how consistently a head retrieves text from image regions. Manipulating attention distributions of these heads can improve OCR-VQA performance and reduce hallucination in OCR-centric tasks. — [arXiv](https://arxiv.org/abs/2505.15865); [EMNLP](https://aclanthology.org/anthology-files/pdf/emnlp/2025.emnlp-main.1032v1.pdf)
- **ASCD** (AAAI 2025/2026): Amplifies automatically mined text-centric heads and dampens critical visual tokens during decoding. Reduces hallucination on POPE, CHAIR, and MMHAL-BENCH by up to 38.2% while improving accuracy on TextVQA. — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/download/38000/41962)
- **What Do Visual Tokens Really Encode?** (arXiv 2603.00510, 2026): Uncovers sparsity and redundancy in visual token representations, finding that different tokens encode different types of information (spatial, semantic, textual). — [arXiv](https://arxiv.org/abs/2603.00510)
- **Text Overlay-Induced Hallucination** (arXiv 2604.17375): Identifies a failure mode where on-screen text overrides visual evidence in video MLLMs, suggesting text-containing patches can dominate attention and mislead generation. — [arXiv](https://arxiv.org/abs/2604.17375)

### Inferences
- The existence of specialized OCR heads suggests that a scoring formula like CVG's a = a_cross + a_self could be decomposed further: different heads attend to different patch types, and weighting them uniformly may lose information.
- An extension that identifies text-centric vs object-centric heads and weights their attention contributions differently could improve both TextVQA and hallucination benchmarks simultaneously.

### Gaps
- No method explicitly classifies visual tokens into "text-containing" vs "object" categories at the patch level and uses this classification for hallucination-specific token selection.
- The OCR heads paper's hallucination reduction claims are for OCR-centric tasks; generalization to object hallucination benchmarks (POPE) is not shown.


## What alternatives to the fixed a = a_cross + a_self scoring formula have been proposed?

### Takeaway
Post-CVG methods have moved in several directions: contrastive decoding (VCD, IBD, MARINE), attention assembly with external matching models (AGLA), vision token manipulation at intermediate layers (OTT, VISTA), and learned adaptive pruning (ATP-LLaVA). None uses CVG's exact formula but proposes a different weighting; instead, they replace the scoring paradigm entirely.

### Cited Findings
- **CVG** (Fu et al., IEEE TIP 2025) scores visual tokens as a = a_cross + a_self, splits them into complementary/supplementary branches by relevance, and contrasts output distributions. Training-free, no external tools. — [IEEE TIP](https://doi.org/10.1109/tip.2025.3644140)
- **AGLA** (CVPR 2025, arXiv 2406.12718) replaces internal attention scoring with an external BLIP-ITM matching model. GradCAM over its cross-attention gives per-patch relevance scores. Low-relevance regions are masked to build an augmented view. The final logit is softmax[logit(y|v,t) + alpha * logit(y|v_aug,t)], where alpha weights the local branch. The masking ratio is adaptive: sim(v,t)/2. POPE adversarial (LLaVA-1.5 7B): accuracy 81.13, F1 81.36 (vs vanilla 76.03/76.26). CHAIR_S: 43.0 (vs vanilla 51.0), recall 78.9. — [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/An_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_with_Assembly_of_CVPR_2025_paper.html); [arXiv](https://arxiv.org/abs/2406.12718)
- **IBD** (CVPR 2025 Workshop, arXiv 2402.18476) modifies the attention weight matrix to create an image-biased model variant, then contrasts standard vs image-biased logits. Training-free. Includes a statistical analysis of vocabulary types feeding an adaptive adjustment strategy. — [CVF](https://openaccess.thecvf.com/content/CVPR2025W/TMM-OpenWorld/html/Zhu_IBD_Alleviating_Hallucinations_in_Large_Vision-Language_Models_via_Image-Biased_Decoding_CVPRW_2025_paper.html); [arXiv](https://arxiv.org/abs/2402.18476)
- **VCD** (Leng et al., CVPR 2024, arXiv 2311.16922) constructs a distorted image (Gaussian noise) and subtracts its logit distribution from the original: p = softmax[(1+alpha)*logit(v) - alpha*logit(v_distorted)]. POPE accuracy (LLaVA-1.5 7B, random): 87.6 (vs vanilla 84.7, per follow-up evaluation). Training-free. — [arXiv](https://arxiv.org/abs/2311.16922); numbers from [arXiv 2602.11737](https://arxiv.org/abs/2602.11737)
- **MARINE** (ICML 2025, arXiv 2402.08680) uses object detectors (DETR threshold 0.95, RAM++ threshold 0.68) to list detected objects, constructs a guidance prompt, and combines logits as l = gamma * l_cond + (1-gamma) * l_uncond (classifier-free guidance style). gamma = 0.7. POPE adversarial accuracy (5-LVLM average): 79.9 (vs vanilla 73.2). CHAIR_S: 8.4 (vs vanilla 11.0). Overhead: 1.98x decoding time. — [arXiv](https://arxiv.org/abs/2402.08680)
- **OTT (One Token, Two Fates)** (arXiv 2603.10360, 2026) manipulates vision tokens at intermediate layers. SVC (enhancement): concatenates augmented image tokens (flip + blur + noise) with original, blends via attention at layer 16 with lambda_s=0.06. CRC (suppression): creates K=3 latent negatives by randomly pruning to 5 vision tokens, computes a correction vector, adds it with lambda_c=0.1. No token scoring or selection; uses all tokens with softmax weights. POPE F1 (MSCOCO, LLaVA-1.5): 87.04 (vs vanilla 85.61). CHAIR_S (64 tokens): 18.1 (vs vanilla 25.4). Overhead: 1.06x. — [arXiv](https://arxiv.org/abs/2603.10360)
- **VISTA** (ICML 2025, arXiv 2502.03628) uses a Visual Steering Vector to restore visual cues in activation space, plus Self-Logits Augmentation drawing on early-layer activations. Training-free. Reports ~40% hallucination reduction on average across four benchmarks and four architectures. — [ICML 2025](https://proceedings.mlr.press/v267/li25ca.html); [arXiv](https://arxiv.org/abs/2502.03628)
- **VisionTrim** (arXiv 2601.22674) selects tokens from the [CLS] token's attention in the vision encoder, then combines with a second signal using variance-based adaptive weighting. Training-free. — [arXiv](https://arxiv.org/abs/2601.22674)
- **VisiPruner** (EMNLP 2025) decodes discontinuous cross-modal attention patterns for visual token pruning. — [EMNLP](https://aclanthology.org/2025.emnlp-main.955.pdf)

### Inferences
- The field has largely moved away from explicit token scoring formulas toward logit-level interventions (contrastive decoding, classifier-free guidance) or activation-space manipulations (VISTA, OTT).
- CVG's a_cross + a_self formula remains relatively unique in directly combining two attention signals into a token importance score. No published work has proposed a weighted variant (e.g., w1*a_cross + w2*a_self) or a normalized version.
- AGLA's use of an external matching model (BLIP-ITM) for relevance scoring is a fundamentally different approach than using the MLLM's own attention, raising questions about which signal is more reliable.
- OTT's random token pruning for creating negatives (vs CVG's attention-guided split) is a surprising design choice that apparently works, suggesting that the specific token selection may matter less than the contrastive training signal.

### Gaps
- No direct ablation study compares weighted vs unweighted combinations of cross-attention and self-attention for token scoring in the hallucination mitigation context.
- Exact TextVQA numbers for most methods above are not available from search results. AGLA and MARINE papers may contain them but were not fully extracted.


## How do RITUAL, AGLA, IBD, MARINE, and other post-CVG works handle token selection?

### Takeaway
These methods handle visual information at fundamentally different granularities: RITUAL operates on whole images (no token selection), AGLA uses external model attention for region masking, IBD and VCD modify attention matrices or logits without selecting tokens, and MARINE uses object detectors for text-prompt-level guidance. Only TAVEN and SeeMe perform explicit visual token selection comparable to CVG.

### Cited Findings
- **RITUAL** (arXiv 2405.17821, 2024) is a training-free method that feeds transformed copies of the entire image (flips, rotations) as complementary inputs during decoding. It adjusts the output distribution using alternative whole-image views. No token-level selection occurs. — [arXiv](https://arxiv.org/abs/2405.17821)
- **AGLA** (CVPR 2025): Uses GradCAM over BLIP-ITM cross-attention to score patches. Low-relevance regions are masked (adaptive ratio = sim(v,t)/2), creating an augmented local view. The masking is at the image region level, not individual token level within the MLLM. A plausibility cutoff beta (fixed hyperparameter) truncates tokens below beta * max(p) in the logit space. — [arXiv](https://arxiv.org/abs/2406.12718)
- **IBD** (CVPR 2025 Workshop): Modifies the MLLM's internal attention weight matrix to create an image-biased variant. No explicit token selection; the entire visual token set is processed, but the attention structure is altered to bias toward image information. — [arXiv](https://arxiv.org/abs/2402.18476)
- **MARINE** (ICML 2025): Token selection is replaced by object detection. DETR (threshold 0.95) and RAM++ (threshold 0.68) detect objects; their intersection forms the guidance set. This is object-level, not token-level. Fixed thresholds; an appendix variant maps gamma from detector confidence but was not adopted as default. — [arXiv](https://arxiv.org/abs/2402.08680)
- **TAVEN** (ScienceDirect, 2025): Training-free and plug-and-play. Built on the observation that not all visual tokens contribute equally to hallucination mitigation. Boosts informative tokens while avoiding over-enhancement of distracting ones. (Mechanism details not fully extracted from search results.) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320326021060)
- **SeeMe** (arXiv 2607.04163, 2026): Training-free pipeline that (1) prunes visual tokens at an intermediate decoder layer using cross-modal attention, (2) fuses pruned tokens with embedding-layer tokens by similarity, (3) makes a final text-aligned selection. One review notes the fusion stage is nearly interchangeable with naive re-injection of original embeddings. — [Pith](https://pith.science/paper/2607.04163)
- **V-ITI** (arXiv 2512.03542, 2025): Visual Inference-Time Intervention. Intervenes on hidden states at specific layers to steer attention back toward visual evidence. — [arXiv](https://arxiv.org/abs/2512.03542)

### Inferences
- The trend is away from explicit token selection (CVG-style) toward either logit-level contrastive methods or activation-space interventions. This may reflect a belief that selecting individual visual tokens is too noisy or that the LLM's later layers can compensate for missing visual information.
- SeeMe's multi-stage pipeline (prune, fuse, re-select) is the closest to an improved version of CVG's approach, adding a recovery step after initial pruning.
- MARINE's reliance on external object detectors introduces a dependency that CVG avoids. The detector thresholds (0.95, 0.68) are aggressive, suggesting that precision matters more than recall for the guidance signal.

### Gaps
- TAVEN's exact mechanism could not be fully extracted from search results (behind ScienceDirect paywall). It is described as token-aware but the specific scoring function is unknown.
- No published comparison directly evaluates CVG's token selection against AGLA's GradCAM-based region masking on the same benchmarks.


## Are there methods that normalize or weight cross-attention and self-attention signals differently?

### Takeaway
Most recent methods either use cross-attention alone (FastV, AdaptToken, VisiPruner), self-attention alone (VisionDrop, ATP-LLaVA), or bypass attention scoring entirely (contrastive decoding, activation steering). The specific combination and relative weighting of cross-attention and self-attention, as in CVG, appears underexplored.

### Cited Findings
- **VisionDrop** (AAAI 2025): Uses only visual self-attention (from the vision encoder) for scoring. Reports that visual-only scoring outperforms text-guided (cross-attention) scoring, especially under aggressive pruning. This directly contradicts the assumption that cross-attention to the query is the best relevance signal. — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/39949/43910)
- **VisionTrim** (arXiv 2601.22674): Combines [CLS] attention from the vision encoder with a variance-based adaptive weighting as a second signal. The two signals are combined with adaptive (not fixed) weights. Training-free. — [arXiv](https://arxiv.org/abs/2601.22674)
- **DyVTE** (arXiv 2411.19628): Analyzes entropy of cross-attention vs text self-attention in LLaVA-style models. Finds that removing visual tokens after a certain layer barely changes text self-attention entropy. The hyper-network that decides the exit layer is trained (not training-free). POPE accuracy (LLaVA-1.5 7B): drops from 85.9 to 81.6 with early exit, suggesting visual tokens remain important even when self-attention entropy is stable. — [arXiv](https://arxiv.org/abs/2411.19628)
- **SPIDER** (arXiv 2609.34977, 2026): Uses visual-encoder attention scores to pick anchor tokens, with a dynamic threshold set to meet a token budget. Adds adaptive sub-layer skipping in the decoder. — [arXiv](https://arxiv.org/abs/2609.34977)
- **FastV** (reference in multiple papers): Uses attention from the second LLM decoder layer to guide pruning. Fixed schedule, not adaptive. Cross-attention only.
- **PDrop** (reference in multiple papers): Prunes at stage boundaries based on attention to the last text token. Fixed schedule.

### Inferences
- The absence of work exploring weighted combinations of cross-attention and self-attention (e.g., a = w_cross * a_cross + w_self * a_self with learned or adaptive w) represents a clear research gap.
- VisionDrop's finding that self-attention alone outperforms cross-attention challenges the rationale for including cross-attention in the scoring formula. However, this was tested for efficiency (token pruning), not specifically for hallucination mitigation, where the relationship between token importance and hallucination may differ.
- An entropy-based adaptive weighting scheme (using the entropy of attention distributions to set the relative contribution of cross-attention vs self-attention) is a natural extension that no paper has proposed.

### Gaps
- No ablation study exists that varies the relative weight of cross-attention vs self-attention in a CVG-style scoring formula and measures the effect on POPE/CHAIR.
- VisionTrim's "variance-based adaptive weighting" is the closest to weighted attention combination, but its exact formula was not extracted from search results.
- Whether normalizing attention scores (e.g., by layer, by head count, or by distribution statistics) before combining them affects hallucination outcomes is unexplored.


## Critical Reassessment: Do POPE Gains Reflect Real Faithfulness?

### Takeaway
A 2026 reassessment paper argues that many inference-time hallucination mitigation methods achieve POPE gains through "risk suppression" (generating fewer objects) rather than genuine faithfulness improvement. This calls into question the reliability of POPE as the sole evaluation metric.

### Cited Findings
- **"Does Playing it Safe Count as Faithfulness?"** (Fazli et al., arXiv 2609.01888, GMU, September 2026) evaluates 6 inference-time methods (VCD, M3ID, AGLA, CAAC, CEI, AFTER) across 3 LVLMs (LLaVA-1.5, LLaVA-NeXT, InstructBLIP 7B) on CHAIR, AMBER, and MMStar (note: not POPE). Main findings: (1) Methods that cut hallucination also cut object recall (CHAIR) or visual coverage (AMBER). LLaVA-1.5 CHAIR_S / recall: vanilla 46.6/79.6, CEI 31.0/68.6, CAAC 38.2/74.4. (2) VCD and M3ID produce "hyper-loquacity" that raises both hallucination and informativeness: VCD CHAIR_S 53.8 (worse than vanilla 46.6) but recall 80.0. (3) No method consistently beats vanilla on MMStar. — [arXiv](https://arxiv.org/abs/2609.01888)
- **"The Mirage of Performance Gains"** (arXiv 2504.10020): Reports that spurious methods achieve POPE performance gains comparable to contrastive decoding. The contrastive inputs push outputs toward "No," so gains mainly reflect suppressing the "No" bias. — [Moonlight Review](https://www.themoonlight.io/review/the-mirage-of-performance-gains-why-contrastive-decoding-fails-to-address-multimodal-hallucination)

### Inferences
- Evaluation of any new token selection method must include both precision metrics (CHAIR, POPE) and recall/coverage metrics (object recall, AMBER coverage, MMStar) to demonstrate genuine improvement.
- The VCD critique is particularly relevant to CVG's contrastive approach: contrasting complementary vs supplementary token branches may also introduce a bias toward conservative generation.
- Methods that maintain high recall while reducing hallucination (AGLA: recall 78.9 vs vanilla 75.2, with lower CHAIR) may be more robust than those that trade recall for precision.

### Gaps
- CVG has not been evaluated in the reassessment paper's framework. Whether its complementary/supplementary split suffers from the same risk-suppression effect is unknown.


## Summary Table: Training-Free Inference-Time Methods

| Method | Venue | Token-Level Selection? | Scoring Mechanism | Adaptive? | Key Benchmark |
|--------|-------|----------------------|-------------------|-----------|---------------|
| CVG | IEEE TIP 2025 | Yes (a_cross + a_self) | MLLM internal attention | Fixed split | POPE, CHAIR |
| VCD | CVPR 2024 | No (whole image) | Distorted image contrast | Fixed alpha | POPE 87.6 (random) |
| AGLA | CVPR 2025 | Region-level masking | External BLIP-ITM GradCAM | Adaptive ratio | POPE adv 81.13 |
| IBD | CVPRW 2025 | No (attention matrix mod) | Modified attention weights | Partial (vocab-adaptive) | POPE, MME |
| MARINE | ICML 2025 | Object-level (detector) | DETR + RAM++ confidence | Fixed thresholds | POPE adv 79.9 |
| RITUAL | arXiv 2024 | No (whole image transforms) | Image augmentation | N/A | Multiple |
| OTT | arXiv 2026 | No (all tokens, intermediate layers) | Random pruning for negatives | Fixed hyperparams | POPE F1 87.04 |
| VISTA | ICML 2025 | No (activation steering) | Steering vector + early logits | N/A | ~40% reduction |
| AdaptToken | arXiv 2026 | Yes (entropy-based budget) | Cross-attention max over queries | Adaptive (entropy) | Video benchmarks |
| SeeMe | arXiv 2026 | Yes (multi-stage prune/fuse) | Cross-modal attention | Adaptive pipeline | Hallucination benchmarks |
| VisionDrop | AAAI 2025 | Yes (self-attention only) | Visual self-attention | N/A | Efficiency benchmarks |
| VisionTrim | arXiv 2026 | Yes ([CLS] + variance) | Vision encoder attention | Adaptive weighting | Efficiency benchmarks |
| TAVEN | Pattern Recognition 2025 | Yes (token-aware) | Token contribution analysis | Unknown | Hallucination benchmarks |
| ASCD | AAAI 2025/2026 | Head-level (text-centric heads) | Head specialization mining | Automatic | POPE, CHAIR, TextVQA |
