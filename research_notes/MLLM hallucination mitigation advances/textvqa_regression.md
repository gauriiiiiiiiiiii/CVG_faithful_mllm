# MLLM Hallucination Mitigation on Text-Rich Images and the TextVQA Regression Problem

## Has anyone specifically addressed CVG's TextVQA regression (where text patches are incorrectly rejected)?

### Takeaway
No paper found that directly addresses CVG's TextVQA regression by name. However, the visual token pruning literature has thoroughly documented the phenomenon that text-containing patches are disproportionately harmed by attention-based token selection, which is the same underlying mechanism CVG's visual grounding uses to separate relevant from irrelevant patches. The closest work is FEATHER (2025), which identifies positional bias from RoPE as the root cause of text-patch rejection, and "Seeing is Believing?" (NeurIPS 2025), which addresses OCR hallucination specifically but through fine-tuning rather than training-free methods.

### Cited Findings
- CVG ("Boosting Faithful Multi-Modal LLMs via Complementary Visual Grounding") is published in IEEE TIP 2025. It disentangles visual context into two branches based on query relevance and addresses Insufficient Visual Context and Progressive Textual Drift as hallucination factors. The paper claims SOTA on hallucination benchmarks but its TextVQA-specific numbers were not accessible from search results alone. — [CVG (IEEE TIP)](https://doi.org/10.1109/tip.2025.3644140)
- FEATHER found that at 75% pruning (R=0.75), TextVQA accuracy drops 42.0% when using FastV's attention-based selection criterion, far more than other benchmarks. The root cause is RoPE's long-term decay making tokens later in raster-scan order (bottom of image) score higher, disproportionately retaining bottom image tokens (average retained position at 80.7% down the image). — [FEATHER (arXiv 2412.13180)](https://arxiv.org/html/2412.13180v2)
- FEATHER also found that prior works report smaller TextVQA declines because they "incorporate OCR-parsed input tokens into prompts, reducing the reliance on visual information." When OCR tokens are excluded from the prompt (the fair evaluation protocol), pruning's TextVQA impact is severe. — [FEATHER (arXiv 2412.13180)](https://arxiv.org/html/2412.13180v2)
- FEATHER's fix: removing RoPE from the token-selection criterion (phi-R) and ensembling with uniform spatial sampling recovers TextVQA from 31.8 to 51.7 at K=3 and from 45.0 to 51.7 at K=8, against a baseline of 54.9 (Table 1 and Table A2). — [FEATHER (arXiv 2412.13180)](https://arxiv.org/html/2412.13180v2)

### Inferences
- CVG's visual grounding likely suffers from the same positional bias FEATHER identifies: text patches, often small and spatially distributed, receive lower attention scores than large coherent object regions, causing them to be classified as "irrelevant" visual context.
- The fact that OCR-token injection into prompts masks the pruning damage suggests that any visual grounding method that discards text patches could be partially compensated by an OCR-aware fallback, but this defeats the purpose of a training-free, external-tool-free approach.

### Gaps
- No paper was found that explicitly cites CVG and discusses its TextVQA regression. A citation search of CVG's IEEE TIP publication would be needed.
- CVG's specific TextVQA numbers were not accessible to confirm the magnitude of the regression.

---

## What methods handle OCR-heavy or text-rich images in the context of hallucination mitigation?

### Takeaway
"Seeing is Believing?" (NeurIPS 2025) is the first work to directly target OCR hallucination in MLLMs, introducing both a benchmark (KIE-HVQA) and a GRPO-based mitigation method. However, it requires fine-tuning. Training-free methods that specifically preserve text-rich content during hallucination mitigation remain an open gap, though several token pruning methods (FEATHER, PoRe, VisionDrop) offer partial solutions by improving spatial coverage during token selection.

### Cited Findings
- "Seeing is Believing?" introduces KIE-HVQA, the first benchmark for OCR hallucination under degraded document conditions (blur, occlusion, low contrast) covering identity cards, invoices, and prescriptions. Their Qwen2.5-VL-7B model achieves ~28 points above GPT-4o on KIE-HVQA (58.05 vs 30.21 average "Final" score) after cold-start SFT followed by GRPO with a rule-based OCR reward. — [Seeing is Believing? (arXiv 2506.20168)](https://arxiv.org/html/2506.20168v2); [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/117155)
- The paper identifies three pipeline gaps causing OCR hallucinations: pretraining lacks degraded-scene KIE data, instruction tuning assumes non-degraded OCR inputs, and evaluation lacks OCR hallucination benchmarks. Models "default to linguistic priors rather than anchoring decisions to observable visual evidence." — [Seeing is Believing? (arXiv 2506.20168)](https://arxiv.org/html/2506.20168v2)
- Their method is NOT training-free: it requires SFT (5 epochs, LR 1e-6 on LLaMA-Factory) followed by GRPO reinforcement learning. The reward function credits keeping legible characters, including partly occluded but human-readable characters, and emitting a space for unrecognizable ones. — [Seeing is Believing? (arXiv 2506.20168)](https://arxiv.org/html/2506.20168v2)
- On OCRBench subsets, their fine-tuned model scores 180/179/183 (Scene/Doc/Info), comparable to base Qwen2.5-VL-7B at 181/181/182, suggesting no degradation on standard OCR tasks from the hallucination mitigation. — [Seeing is Believing? (arXiv 2506.20168)](https://arxiv.org/html/2506.20168v2)
- Models evaluated include GPT-4o, Claude 3.5 Sonnet, Claude 3.7 Sonnet, Gemini 2.5 Pro, InternVL3-8B/38B/78B, and Qwen2.5-VL-8B/32B/72B. Qwen2-VL does not appear. — [Seeing is Believing? (arXiv 2506.20168)](https://arxiv.org/html/2506.20168v2)
- iTaD (NAACL 2025) is a training-free, plug-and-play decoding method that identifies when output tokens pay insufficient attention to image tokens and contrasts layers to emphasize image understanding. It does not specifically address text-rich images. — [iTaD (ACL Anthology)](https://aclanthology.org/2025.naacl-long.75/)
- TGIF (arXiv 2601.03100) is a 2026 method using text-guided layer fusion for hallucination mitigation. Its reference list cites LLaVAR (a text-rich image instruction tuning method), suggesting it may evaluate on text-rich benchmarks, but its full results were not accessible. — [TGIF (arXiv 2601.03100)](https://arxiv.org/pdf/2601.03100)

### Inferences
- The gap between training-free hallucination mitigation and OCR preservation is real: no existing training-free method was designed with text patches in mind. "Seeing is Believing?" solves OCR hallucination but requires model fine-tuning, which is incompatible with CVG's training-free philosophy.
- A training-free approach that adapts CVG's visual grounding to be text-patch-aware (e.g., using spatial attention patterns to identify text regions before filtering) would address an unoccupied niche in the literature.

### Gaps
- No training-free method was found that specifically targets text-rich image hallucination mitigation.
- TGIF's full evaluation results on text-rich benchmarks could not be verified.

---

## Are there approaches that distinguish text patches from object patches using attention patterns alone (no external OCR)?

### Takeaway
No paper was found that explicitly uses attention patterns to distinguish text patches from object patches without external OCR. However, several papers document distinctive properties of text patches in attention: they tend to have high local connectivity (text characters attend to neighboring characters), they are spatially distributed rather than forming large contiguous regions, and they receive lower global attention scores than object patches. These properties could theoretically be exploited for text-region identification.

### Cited Findings
- Vision transformers process patches in two distinct stages: an early "perceptual" stage where tokens mostly attend within the same object, followed by a later "relational" stage where cross-object attention emerges. Some models route object information to background register tokens. — [Beyond the Doors of Perception (arXiv 2406.15955)](https://arxiv.org/pdf/2406.15955)
- Patch pruning studies show that attention weights separate object patches from background, with background patches mostly pruned and object-region patches retained through the final layer. — [Patch Pruning Strategy (arXiv 2507.19175)](https://arxiv.org/pdf/2507.19175)
- VisionDrop identifies three types of cross-modal misalignment in token pruning: causal misalignment (from autoregressive decoding), semantic misalignment (visual/textual token entanglement in LLM layers), and spatial misalignment (from positional embedding flattening). Their fix uses visual-to-visual attention only, bypassing text-guided scoring entirely. Training-free, retaining 91.46% of original performance at 94.4% token reduction on LLaVA-1.5-7B. — [VisionDrop (arXiv 2506.22283)](https://arxiv.org/html/2506.22283v2)
- TokenCarve notes that "VQAText and VQAV2 contain a large number of OCR tasks that are sensitive to the loss of informativeness" but does not provide analysis of why text patches specifically lose information under attention-based scoring. Its information contribution score (from SVD of layer-2 attention output) performs worse than raw attention on TextVQA at 64 tokens (54.1 vs 55.7), suggesting the SVD-based score is even less text-aware. — [TokenCarve (arXiv 2503.10501)](https://arxiv.org/html/2503.10501v1)

### Inferences
- Text patches likely differ from object patches in their attention patterns in three ways: (1) they exhibit high local connectivity (characters attend to adjacent characters in a word), (2) they form spatially distributed clusters rather than large contiguous regions, and (3) they receive lower aggregate global attention because their individual information contribution is small relative to large objects, even though their collective contribution is essential for understanding.
- VisionDrop's visual-to-visual attention approach may implicitly help text patches by avoiding the cross-modal misalignment that hurts them, but this is not tested.
- An explicit text-vs-object patch classifier based on local attention connectivity patterns (without OCR) appears to be an unexplored research direction.

### Gaps
- No empirical study directly compares attention patterns on text patches vs. object patches in a vision transformer or MLLM vision encoder.
- Whether local connectivity in self-attention is sufficient to distinguish text regions from fine-grained textures (e.g., brick patterns) is unknown.

---

## How do newer MLLMs (Qwen2-VL, InternVL-2.5) handle text-rich content differently?

### Takeaway
Newer MLLMs achieve near-saturation on standard text-rich benchmarks (Qwen2.5-VL reaches 96.4% on DocVQA), but their OCR hallucination rates under degraded conditions remain high. Dynamic resolution and native aspect ratio handling are key architectural differences. InternVL3 reports OCRBench 866 for Qwen2-VL-7B via its comparison tables.

### Cited Findings
- Qwen2.5-VL has achieved 96.4% accuracy on DocVQA, suggesting near-saturation on standard document understanding benchmarks. — [OCRBench v2 (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/file/8c2e6bb15be1894b8fb4e0f9bcad1739-Paper-Datasets_and_Benchmarks_Track.pdf)
- OCRBench v2 was created because existing OCR benchmarks including the original OCRBench show rapid performance saturation with newer models. — [OCRBench v2 (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/file/8c2e6bb15be1894b8fb4e0f9bcad1739-Paper-Datasets_and_Benchmarks_Track.pdf)
- InternVL3 blog reports OCRBench 866 for Qwen2-VL-7B in its comparison. — [InternVL3 blog](https://internvl.github.io/blog/2025-04-11-InternVL-3.0/)
- CC-OCR (ICCV 2025) is a comprehensive OCR benchmark that includes Qwen2-VL-72B (Total score 71.14) and InternVL3-78B in its evaluations. — [CC-OCR (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_CC-OCR_A_Comprehensive_and_Challenging_OCR_Benchmark_for_Evaluating_Large_ICCV_2025_paper.pdf)
- Despite high standard benchmark scores, "Seeing is Believing?" shows that Qwen2.5-VL-8B scores only 21.53 average on KIE-HVQA (degraded document OCR hallucination), while GPT-4o scores 30.21. These are far below the 58.05 achieved after targeted fine-tuning. — [Seeing is Believing? (arXiv 2506.20168)](https://arxiv.org/html/2506.20168v2)
- OCR-Reasoning Benchmark (arXiv 2505.17163) targets complex text-rich image reasoning and includes Qwen2-VL in its reference models. — [OCR-Reasoning Benchmark (arXiv 2505.17163)](https://arxiv.org/pdf/2505.17163)
- "Do VLMs Read or Rewrite?" (arXiv 2607.21617) studies transcription faithfulness in VLMs, suggesting that even high-performing models may alter or hallucinate text rather than faithfully transcribing it. — [Do VLMs Read or Rewrite? (arXiv 2607.21617)](https://arxiv.org/pdf/2607.21617)

### Inferences
- High DocVQA/OCRBench scores do not imply robustness to OCR hallucination. Standard benchmarks test clean, well-formatted documents, while real-world degradation (blur, occlusion, low contrast) exposes a large performance gap.
- The architectural improvements in Qwen2-VL and InternVL (dynamic resolution, native aspect ratio) help with text recognition accuracy but do not prevent the model from hallucinating text content when visual evidence is ambiguous.

### Gaps
- Specific architectural details of how Qwen2-VL and InternVL handle text-rich patches at the vision encoder level (vs. objects) were not found.
- No comparison was found of attention patterns on text-rich images between Qwen2-VL, InternVL, and earlier models like LLaVA.

---

## What is the relationship between visual token selection and OCR task performance?

### Takeaway
Visual token selection/pruning disproportionately damages OCR and text-understanding tasks compared to general VQA or object recognition. TextVQA is consistently the most sensitive benchmark across multiple pruning studies. The root causes are positional bias (RoPE), cross-modal misalignment, and the spatially distributed nature of text information.

### Cited Findings
- FEATHER: TextVQA drops 42.0% at 75% pruning with FastV, vs. much smaller drops on other benchmarks. The cause is RoPE's positional decay creating bottom-of-image bias. Baseline TextVQA 54.9; FastV at 68% FLOPS reduction: 31.8; FEATHER at 64% FLOPS reduction: 51.4. — [FEATHER (arXiv 2412.13180)](https://arxiv.org/html/2412.13180v2)
- TokenCarve: TextVQA on LLaVA-1.5-7B: baseline 58.2, 192 tokens: 58.4, 128 tokens: 58.1, 64 tokens: 57.0. At 64 tokens, TextVQA drop (-1.2) is smaller than POPE (-6.2) and VQAV2 (-3.7), suggesting TokenCarve's information-preserving approach helps text patches specifically, though the authors do not claim this. — [TokenCarve (arXiv 2503.10501)](https://arxiv.org/html/2503.10501v1)
- QCPruner (arXiv 2609.19990) uses TextVQA and OCRBench and notes that "fine-grained OCR evidence makes these benchmarks sensitive to aggressive pruning." — [QCPruner (arXiv 2609.19990)](https://arxiv.org/pdf/2609.19990)
- PoRe corrects the positional bias in attention-based pruning. It is plug-and-play and training-free. Gains are largest at aggressive pruning ratios (75% and 50%) on VQAv2, GQA, and SQAIMG. Text-specific OCR numbers not reported. — [PoRe (arXiv 2508.17807)](https://arxiv.org/html/2508.17807v1)
- GridPrune (arXiv 2511.10081) allocates token budgets across spatial zones before local selection, ensuring spatial coverage. — [GridPrune (arXiv 2511.10081)](https://arxiv.org/abs/2511.10081v1)
- ATP-LLaVA pairs attention-based importance with uniform spatial sampling to maintain spatial coverage during pruning. — [ATP-LLaVA (arXiv 2412.00447)](https://arxiv.org/pdf/2412.00447)
- AdaptPrune adds spatial distance and token similarity to attention-based scoring, with adaptive non-maximum suppression. It argues attention alone "overlooks spatial position." — [AdaptPrune (arXiv 2503.08019)](https://arxiv.org/html/2503.08019v1)
- D2Pruner (AAAI 2026) debiases attention scores then selects diverse tokens using a graph combining spatial adjacency and semantic similarity. — [D2Pruner (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/download/38234/42196)

### Inferences
- Text information is fundamentally different from object information in visual tokens: it is spatially distributed (spread across many small patches) rather than concentrated (a few large patches). Pure attention-based selection, designed for the concentrated case, systematically disadvantages text.
- Methods that add spatial diversity (GridPrune, ATP-LLaVA, FEATHER's uniform sampling) partially recover text performance, suggesting spatial coverage is a key mechanism.
- For CVG specifically: the visual grounding step likely acts as an aggressive pruner of text-containing patches. Adapting CVG to maintain spatial coverage (e.g., via uniform spatial sampling in the "irrelevant" branch) could mitigate the TextVQA regression without external OCR.

### Gaps
- No study directly measures the correlation between visual token selection criteria and OCR accuracy across a range of text densities.
- Whether information-theoretic scores (like TokenCarve's SVD-based metric) or visual-only attention (like VisionDrop) better preserve text patches than cross-modal attention remains untested head-to-head.

---

## Are there methods that use spatial attention patterns or local connectivity to identify text regions?

### Takeaway
No method was found that uses spatial attention patterns specifically to identify text regions in an MLLM context. However, several papers use spatial structure in attention for token selection (GridPrune, D2Pruner, PoRe), and VisionDrop's visual-to-visual attention approach implicitly captures local connectivity that could distinguish text from objects.

### Cited Findings
- VisionDrop uses visual-to-visual self-attention in the vision encoder (not cross-modal attention) for token scoring. This approach captures spatial relationships between patches without text-query interference. Training-free. — [VisionDrop (arXiv 2506.22283)](https://arxiv.org/html/2506.22283v2)
- GridPrune divides the image into spatial zones and allocates token budgets per zone before local selection. This ensures text regions (which may be in low-attention zones) still receive representation. — [GridPrune (arXiv 2511.10081)](https://arxiv.org/abs/2511.10081v1)
- D2Pruner builds a graph over visual tokens using spatial adjacency and semantic similarity, then selects diverse tokens from this graph structure. — [D2Pruner (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/download/38234/42196)
- PoRe's position-reweighting corrects the bias where tokens at the image bottom receive inflated attention scores due to RoPE. This is a spatial correction to attention, not spatial identification of text. — [PoRe (arXiv 2508.17807)](https://arxiv.org/html/2508.17807v1)
- HiPrune (arXiv 2508.00553) uses the vision encoder's hierarchical self-attention (not cross-modal attention) to score tokens. This avoids dependence on CLS tokens, which some encoders (SigLIP) lack. — [HiPrune (arXiv 2508.00553)](https://arxiv.org/pdf/2508.00553v1)
- Vision transformer attention studies show that object tokens attend mostly within the same object during early layers (perceptual stage), with cross-object attention appearing later (relational stage). Whether text patches show a distinctive local-connectivity pattern was not tested. — [Beyond the Doors of Perception (arXiv 2406.15955)](https://arxiv.org/pdf/2406.15955)

### Inferences
- Text characters in natural images form spatially coherent but small clusters with high local mutual attention (letters in a word attend to each other). This pattern differs from objects (large contiguous regions) and from background (uniformly low attention). A classifier based on the ratio of local vs. global attention in the vision encoder's self-attention could identify text regions.
- VisionDrop's visual-to-visual attention may already implicitly capture this pattern, but no study has verified it.
- For a CVG extension: computing local attention connectivity (e.g., average attention weight between spatially adjacent patches) in the vision encoder before the visual grounding step could flag text-containing patches and exempt them from the irrelevant-context filter.

### Gaps
- No empirical characterization of text patch attention patterns (local connectivity, attention concentration) exists in the literature.
- Whether self-attention patterns in SigLIP, CLIP ViT, or InternViT differ for text vs. object patches is untested.

---

## What are the latest TextVQA, DocVQA, OCRBench results for hallucination-mitigated models?

### Takeaway
Hallucination-mitigated models show mixed OCR benchmark results. Training-free pruning methods consistently degrade TextVQA, while fine-tuning-based methods ("Seeing is Believing?") preserve OCRBench. Standard DocVQA is near saturation (96.4% for Qwen2.5-VL) so it no longer discriminates effectively. OCRBench v2 was created to address this saturation.

### Cited Findings
- FEATHER (training-free pruning + hallucination mitigation via spatial sampling): TextVQA 51.4 at 64% FLOPS reduction vs. 54.9 baseline (LLaVA-1.5-7B). — [FEATHER (arXiv 2412.13180)](https://arxiv.org/html/2412.13180v2)
- TokenCarve (training-free compression): TextVQA 57.0 at 64 tokens vs. 58.2 baseline (LLaVA-1.5-7B). — [TokenCarve (arXiv 2503.10501)](https://arxiv.org/html/2503.10501v1)
- "Seeing is Believing?" (SFT+GRPO on Qwen2.5-VL-7B): OCRBench Scene/Doc/Info 180/179/183, comparable to base model at 181/181/182. No TextVQA or DocVQA reported. — [Seeing is Believing? (arXiv 2506.20168)](https://arxiv.org/html/2506.20168v2)
- Qwen2.5-VL achieves 96.4% on DocVQA. OCRBench v2 was created because v1 shows "rapid performance saturation." — [OCRBench v2 (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/file/8c2e6bb15be1894b8fb4e0f9bcad1739-Paper-Datasets_and_Benchmarks_Track.pdf)
- Qwen2-VL-7B scores OCRBench 866 per InternVL3 comparison table. — [InternVL3 blog](https://internvl.github.io/blog/2025-04-11-InternVL-3.0/)
- CC-OCR (ICCV 2025): Qwen2-VL-72B Total score 71.14. — [CC-OCR (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_CC-OCR_A_Comprehensive_and_Challenging_OCR_Benchmark_for_Evaluating_Large_ICCV_2025_paper.pdf)

### Inferences
- Training-free methods universally lose some TextVQA performance. The best current result (FEATHER) recovers most of the loss but still falls 3.5 points below baseline.
- Fine-tuning methods can preserve OCRBench while reducing hallucination, but this is incompatible with training-free constraints.
- Standard DocVQA is too saturated to serve as a discriminative benchmark for newer methods. OCRBench v2 or KIE-HVQA should be preferred for evaluation.

### Gaps
- No comprehensive table comparing all hallucination-mitigated models on TextVQA + DocVQA + OCRBench was found in a single paper.
- iTaD's specific TextVQA numbers were not accessible from search results.
- VisionDrop's TextVQA numbers could not be confirmed (57.81 from an aggregator, unverified).
