# Understanding the CVG Paper and Our CVG-E Research

## Part 1: The Original CVG Paper (What They Did)

**Paper:** "Boosting Faithful Multi-Modal LLMs via Complementary Visual Grounding"
**Authors:** Zheren Fu, Zhendong Mao, Lei Zhang, Yongdong Zhang (USTC)
**Published in:** IEEE Transactions on Image Processing, Vol. 34, 2025

### What is the Problem?

Multi-modal Large Language Models (MLLMs) like LLaVA can look at an image and answer questions about it. But they have a big problem called **hallucination** — they sometimes describe things that are NOT actually in the image.

For example, if you show a photo of a dog sitting on a beach, the model might say "A woman is sitting with a dog on the beach. She is about to throw a stick for the dog. Besides, someone is surfing on the beach." — but there is no woman, no stick, and no surfer. The model just made those up.

This is dangerous because if we use these models in real applications (medical imaging, self-driving cars, etc.), wrong answers can cause real harm.

### Why Does Hallucination Happen?

The CVG paper identifies **two root causes**:

**1. Insufficient Visual Context**
When the question asks about something that is visually unclear or only partially visible, the model does not have enough visual information. So instead of saying "I'm not sure," it falls back on its language knowledge and guesses. These guesses are often wrong.

The authors proved this by masking (hiding) important parts of the image. When they masked the parts the model was looking at, hallucination increased a lot — much more than random masking. This shows the model really depends on specific visual regions, and when those regions are ambiguous, it hallucinates.

**2. Progressive Textual Drift**
When the model generates long responses, its attention gradually shifts away from the image and toward its own generated text. After a few sentences, the model is basically ignoring the image and just writing based on language patterns. This causes more hallucination in longer responses.

The authors showed this with an attention plot — visual attention starts strong but drops quickly after 30-40 tokens. And CHAIR benchmark data confirmed: longer responses have higher hallucination rates.

### What is CVG? (Their Solution)

CVG stands for **Complementary Visual Grounding**. It is a "training-free" method — meaning you don't need to retrain the model, fine-tune anything, or use any external tools. It works at inference time (when the model is generating answers).

CVG has **three modules**:

#### Module 1: Visual Token Disentanglement (VTD)

The image goes through a vision encoder (like CLIP ViT) which converts it into a set of "visual tokens" — small patches of the image, each represented as a vector.

VTD scores each visual token based on how important it is for answering the question. It uses two signals:

- **Cross-attention score**: How much does the question text attend to this image patch? (Query relevance)
- **Self-attention score**: How much does the [CLS] token in the vision encoder attend to this patch? (General visual importance)

These are combined: `a = a_cross + a_self`

Then, the top 70% (α = 0.7) become the **chosen set** (relevant patches) and the bottom 30% become the **rejected set** (irrelevant/distracting patches).

Two separate branches are created:

- **Chosen branch**: Keeps only the important patches, suppresses the rest
- **Rejected branch**: Keeps only the unimportant patches, suppresses the rest

#### Module 2: Dynamic Attention Calibration (DAC)

As the model generates text token by token, DAC adjusts attention weights:

- In the **chosen branch**: Visual attention is boosted (so the model keeps looking at the image)
- In the **rejected branch**: Visual attention is reduced (so this branch drifts away from the image on purpose)

The boost increases over time: `δ_t = β(1 - e^(-t/τ))` — this means the correction gets stronger as generation continues, specifically to fight the textual drift problem.

This makes the two branches more different from each other, which helps the next module work better.

#### Module 3: Grounding Flow Fusion (GFF)

This is where the magic happens. The model generates two probability distributions — one from the chosen branch (should be correct) and one from the rejected branch (should be hallucination-prone).

GFF compares these using **KL divergence** (a measure of how different two distributions are):

- **If they disagree a lot** (KL > ε): The model uses contrastive decoding — it amplifies the chosen branch and subtracts the rejected branch's influence. This pushes the output away from hallucinated content.
- **If they agree** (KL < ε): The visual context is clear, so just use the chosen branch as-is.

Formula: `logit_final = logit_chosen + γ × (logit_chosen - logit_rejected)` when KL ≥ ε

### Results in the Original Paper

CVG was tested on LLaVA-1.5 (7B and 13B parameters) and LLaVA-Next (7B and 13B).

**Key results:**

- **POPE benchmark** (object hallucination): CVG beats Vanilla, VCD, OPERA, and PAI across all splits (random, popular, adversarial) on COCO, A-OKVQA, and GQA
- **CHAIR benchmark** (image captioning): CVG produces the fewest hallucinated objects in generated descriptions
- **AMBER benchmark** (multi-dimensional): Best discriminative accuracy across all tested models
- **MME-Hal benchmark**: Highest hallucination mitigation scores
- **General benchmarks** (VQA, TextVQA, MM-Bench, etc.): CVG also improves or maintains general task performance — it doesn't just reduce hallucination, it makes the model more accurate overall

**Efficiency:** CVG adds about 64% more computation (FLOPs) compared to vanilla decoding, which is less than VCD (+100%). This is because each branch only processes half the visual tokens.

### Known Limitations (from their paper)

1. **TextVQA regression**: Text-containing image patches get low attention scores and end up in the rejected set, so CVG hurts performance on text-rich images
2. **Fixed α**: The 70% token selection ratio is the same for all images, regardless of difficulty
3. **Dual forward pass**: Even though each branch processes fewer tokens, running two passes still adds latency

---

## Part 2: What We Did (Our CVG-E Research)

### Our Goal

We wanted to study and extend the CVG framework. Our research paper is titled: **"CVG-E: Efficient Complementary Visual Grounding with Adaptive Token Selection and Single-Pass Approximation"**

### Our Two Proposed Improvements

#### Improvement A: Adaptive Token Selection

Instead of always picking the top 70% of tokens (fixed α), we made it adaptive:

1. **Entropy-adaptive α**: We calculate the entropy (disorder/uniformity) of the attention scores. If attention is spread evenly across many patches (high entropy = ambiguous scene), we keep more tokens. If attention is focused on a few patches (low entropy = clear scene), we prune more aggressively.

   - Formula: `α = α_min + (α_max - α_min) × H(a)/log(N)`
   - Range: α_min=0.5 to α_max=0.85
2. **Text-patch protection**: We detect text-containing patches using a neighbor attention pattern (adjacent characters attend strongly to each other) and boost their scores so they don't get incorrectly rejected.
3. **Z-score normalized hybrid scoring**: Instead of just adding cross-attention and self-attention, we normalize them (z-score) and use tunable weights: `a = 0.5×cross + 0.3×self + 0.2×text`

#### Improvement B: Single-Pass Branch Approximation

The biggest cost in CVG is running two full forward passes through the model. We tried to eliminate the rejected branch's forward pass:

Instead of running the rejected branch through all 22 transformer layers, we:

1. Take the chosen branch's attention weights at one layer
2. Re-weight them — suppress visual token attention by 0.1× and amplify text attention by 10×
3. Recompute only that one layer's output + the LM head
4. Use this as an approximation of what the rejected branch would have produced

This should theoretically be much cheaper (1 layer vs 22 layers).

### Our Experimental Setup

- **Model**: TinyLLaVA-1.5B (a small model — 1.5 billion parameters, vs the 7B-13B models in the original paper)
- **Hardware**: Apple M1 Pro with 16GB RAM (consumer laptop, not a GPU server)
- **Benchmark**: POPE random split, COCO val2014, 180 questions (30 images × 6 questions each)
- **Why small model**: We wanted to test if CVG's approach works on smaller, more deployable models

### What We Found

#### Finding 1: Contrastive Decoding Breaks on Small Models

This was our main finding — and it was unexpected.

| Method  | Parseable       | Accuracy        | Precision       | Recall | Yes-ratio |
| ------- | --------------- | --------------- | --------------- | ------ | --------- |
| Vanilla | 180/180 (100%)  | **86.7%** | **86.7%** | 86.7%  | 50.0%     |
| CVG     | 91/180 (50.6%)  | 62.6%           | 60.2%           | 98.0%  | 91.2%     |
| CVG-E   | 124/180 (68.9%) | 62.9%           | 61.4%           | 89.9%  | 81.5%     |

**The vanilla model (no tricks, just normal decoding) easily beats both CVG and CVG-E.**

Both contrastive methods actually make the model *worse*, not better, on a 1.5B parameter model.

#### Finding 2: Generation Collapse

CVG causes the model to produce garbage output 49% of the time:

- **Question echoing** (53 out of 89 failures): The model just repeats the question back instead of answering
- **Degenerate sequences** (36 out of 89): Empty or nonsensical output

CVG-E is better (31% failure rate vs 49%) because single-layer re-weighting is a weaker perturbation.

#### Finding 3: Yes-Bias (The "Mirage" Effect)

Among the responses that ARE parseable, both methods show extreme "Yes" bias:

- CVG: 91.2% of answers are "Yes" (should be 50%)
- CVG-E: 81.5% of answers are "Yes" (should be 50%)
- Vanilla: 50.0% (perfectly balanced)

This means the contrastive methods aren't reducing hallucination — they're just making the model say "Yes" to everything. High recall (98% for CVG!) looks impressive until you realize it's because the model almost never says "No."

#### Finding 4: The Latency Paradox

| Method  | Time per question | Slowdown |
| ------- | ----------------- | -------- |
| Vanilla | 0.96s             | 1×      |
| CVG     | 8.51s             | 8.9×    |
| CVG-E   | 17.87s            | 18.6×   |

CVG-E is supposed to be faster than CVG (that's the whole point of single-pass). But it's actually **slower**!

Why? Because to get the attention weights from the model, we need to set `output_attentions=True`, which forces the model to use "eager attention" instead of efficient SDPA. This materializes huge attention matrices (1171×1171) at all 22 layers, every generation step — even though we only need 1 layer's attention.

This is fixable with a hook-based implementation, and we have since built it (see Part 4).

#### Finding 5: The Goldilocks Problem

The KL divergence analysis shows the fundamental tension:

- **CVG (dual-pass)**: KL ≈ 0.5 — strong contrastive signal, BUT destroys generation quality
- **CVG-E (single-pass)**: KL < 0.01 — preserves generation quality, BUT no meaningful contrastive signal

The minimum perturbation needed for useful contrastive decoding may be more than a 1.5B model can handle. There might be no "just right" operating point at this scale.

### Small-Model Calibration (What We Had to Fix)

The original CVG hyperparameters cause complete generation collapse on 1.5B models. We had to recalibrate:

| Parameter                 | Original (7B)  | Our Fix (1.5B)    | Why                                        |
| ------------------------- | -------------- | ----------------- | ------------------------------------------ |
| Suppress factor           | 0.0 (zero out) | 0.01 (scale down) | Zeroing creates out-of-distribution inputs |
| γ (contrastive strength) | 1.0            | 0.5               | Full strength overwhelms small model       |
| ε (KL threshold)         | 0.01           | 0.1               | Low threshold = contrastive always fires   |
| β (DAC strength)         | 1.0            | 0.1               | Full calibration over-corrects             |

Even with these fixes, the methods still underperform vanilla. The calibration is necessary but not sufficient.

### Our Contributions (What We Added to Knowledge)

1. **First CVG/CVG-E implementation tested at ≤2B scale** — showing that contrastive decoding methods designed for 7B+ don't work at small scale
2. **Small-model calibration strategies** — the specific parameter changes needed to avoid complete collapse
3. **Failure mode characterization** — documenting question echoing, degenerate outputs, and Yes-bias as distinct failure modes
4. **Latency inversion documentation** — showing that eager attention bottleneck can negate single-pass speedup
5. **The "narrow operating range" hypothesis** — that contrastive decoding may have no viable operating point on small models

---

## Part 3: The Process We Followed

### Step 1: Literature Review

- Read the original CVG paper thoroughly
- Studied related work: VCD, ICD, M3ID, OPERA, PAI, AttnReal, ASCD, DaID
- Identified CVG's three limitations (fixed α, text regression, dual-pass cost)
- Conducted deep research on latest MLLM hallucination mitigation advances (2024-2026)

### Step 2: Implementation

- Set up TinyLLaVA-1.5B on M1 Pro with MPS backend
- Implemented CVG from scratch based on the paper's description (no official code available for our model)
- Built all three modules: VTD, DAC, GFF
- Fixed GQA compatibility (32 query heads, 4 KV heads — needed repeat_interleave)

### Step 3: CVG-E Extensions

- Implemented entropy-adaptive α calculation
- Built text-patch protection scoring
- Implemented z-score normalized hybrid scoring
- Built the single-pass branch approximation (Algorithm 1 in our paper)

### Step 4: Calibration

- Discovered original CVG parameters cause complete generation collapse on 1.5B
- Systematically identified four parameter changes needed
- Tested and validated the recalibrated configuration

### Step 5: Experiments

- Ran vanilla baseline on N=180 POPE questions
- Ran CVG on N=180 POPE questions (took ~25 minutes, 8.5s per question)
- Ran CVG-E on N=180 POPE questions (took ~54 minutes, 17.9s per question)
- All results saved as JSON with per-question details

### Step 6: Analysis

- Built analysis script with Wilson score confidence intervals
- Computed all metrics: accuracy, precision, recall, F1, yes-ratio
- Generated confusion matrices
- Classified failure modes (echo vs degenerate)
- Analyzed KL divergence patterns

### Step 7: Paper Writing

- Wrote the full LaTeX paper (7 pages, IEEE format)
- Created TikZ diagrams: pipeline comparison, bar charts, stacked bars, KL divergence plot
- Included all tables with confidence intervals
- Compiled to clean PDF

### Step 8: Presentation

- Created 14-slide presentation (PPTX + PDF)
- Dark navy theme with accent colors
- Covers: introduction, challenges, problem, related work, methodology, results, failure analysis, latency, 7B scale-up, publication status, next steps

---

## Part 4: Scaling Up to LLaVA-1.5-7B (Current Phase)

The 1.5B results raised one clear question: is the failure caused by CVG itself, or by the small model? The original paper only tested 7B and 13B models. To answer this fairly, we moved our experiments to the same model size the paper used.

### Step 9: Fixing the Latency Problem (Hook-Based Attention)

The 1.5B experiments showed CVG-E was *slower* than CVG because `output_attentions=True` forced every layer to use slow "eager" attention. We rebuilt the attention extraction:

- **`SingleLayerHook`** (in `cvg/hooks.py`) attaches to **one** transformer layer only.
- Only that one layer switches to eager attention, so we can read its attention weights.
- All other layers keep using fast SDPA / Flash Attention.
- The rejected branch is then approximated from that one layer's captured attention and hidden state.

This removes the O(n²) cost at every layer that made CVG-E 18.6× slower than vanilla.

### Step 10: Porting to LLaVA-1.5-7B

LLaVA-1.5-7B has 32 language-model layers and 24 vision-encoder layers (vs 22 in TinyLLaVA). We set the layer choices to match the original paper's setup:

| Setting | Layers used |
| --- | --- |
| Cross-attention (VTD query relevance) | 19, 20, 21, 22 |
| Self-attention (VTD [CLS] importance) | 16, 17, 18 |
| DAC calibration | 20 – 31 |
| CVG-E single-pass approximation | 28 – 31 |

The two new configs are `cvg_7b` and `cvg_e_7b` in `cvg/modules.py`. At 7B we use the **original** CVG hyperparameters, not the small-model calibration, because the original values were designed for this size.

### Step 11: Running on a Cloud GPU (Google Colab)

A 7B model does not fit on the M1 laptop, so we built a Colab notebook (`notebooks/cvg_e_7b_experiments.ipynb`) and an evaluation script (`scripts/run_7b_eval.py`):

- **GPU:** NVIDIA Tesla T4 (15 GB VRAM), free Colab tier
- **Benchmark:** full POPE COCO splits — **3,000 questions per split** (vs 180 before)
- **Metrics:** accuracy, precision, recall, F1, yes-ratio, 95% Wilson confidence intervals, time per question

### Step 12: First 7B Result — Vanilla Baseline

| Method | Split | Parseable | Accuracy | Precision | Recall | F1 | Yes-ratio | Time/q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla (fp16) | COCO random | 3000/3000 | **89.4%** | 91.3% | 87.1% | 89.1% | 47.7% | 1.16s |

This matches the vanilla LLaVA-1.5-7B numbers reported in the literature (around 88–90%), which confirms our pipeline is correct. Every answer was parseable, and the yes-ratio is close to the ideal 50%.

### Step 13: Out-of-Memory Problem and the 4-bit Fix

Our first CVG and CVG-E runs on the T4 failed:

- The 7B model in fp16 uses about **14 GB** of the T4's 15 GB.
- CVG needs a second forward pass, and CVG-E needs one layer's full attention matrix. Neither fits in the remaining ~1 GB.
- Every question hit a CUDA out-of-memory error. The script caught the error, and the result showed 50% accuracy with 0% recall. **These numbers are invalid and were discarded.**

**Fix:** we added 4-bit quantization (NF4, via `bitsandbytes`) with a new `--quantize 4bit` option. This shrinks the model to about **5 GB**, which leaves enough memory for the contrastive branches. The notebook now installs `bitsandbytes` and uses 4-bit for CVG and CVG-E automatically.

**Fairness note:** vanilla ran in fp16 and CVG/CVG-E will run in 4-bit. For the final paper we will also run vanilla in 4-bit, so every method is compared at the same precision.

### What Is Still Pending

| Experiment | Status |
| --- | --- |
| Vanilla 7B, COCO random | Done (89.4%) |
| CVG 7B (4-bit), COCO random | Waiting for Colab GPU quota |
| CVG-E 7B (4-bit), COCO random | Waiting for Colab GPU quota |
| Vanilla 7B (4-bit), for fair comparison | Planned |
| Popular + adversarial splits, all methods | Planned |
| CHAIR, AMBER, TextVQA | Planned |
| VCD baseline | Planned |

### What We Expect to Show at 7B

1. **Accuracy:** CVG-E should match or beat CVG, because adaptive α and text-patch protection keep more useful tokens.
2. **Speed:** CVG-E should now be clearly faster than CVG, because the hook-based single-pass avoids the second full forward pass.
3. **Scale effect:** comparing 1.5B (failure) with 7B (expected success) shows where contrastive decoding starts working.

---

## Key Takeaway

**The original CVG paper shows an excellent method that works great on large (7B-13B) models.** It is published in a top journal (IEEE TIP) and achieves state-of-the-art results.

**Our CVG-E work shows that this approach has boundary conditions.** When you try to use contrastive decoding on small (1.5B) models — which are more practical for real-world deployment — the method fails. The perturbation that creates a useful contrastive signal also destroys the model's ability to generate coherent text.

This is valuable because it tells the research community: contrastive decoding for hallucination mitigation is model-size dependent. Future work needs to find methods that work at small scale too, or find the minimum model size where contrastive decoding starts working.

**Our current phase tests CVG-E at 7B**, the scale CVG was designed for. We fixed the latency bottleneck with a single-layer hook, built a cloud-GPU pipeline, and confirmed a correct vanilla baseline (89.4% on 3,000 POPE questions). The CVG and CVG-E 7B runs, now using 4-bit quantization, will show whether CVG-E can match CVG's accuracy while being faster.
