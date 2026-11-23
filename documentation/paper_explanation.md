# CVG and CVG-E, Explained Simply

## The One-Minute Version

- AI models that look at pictures and answer questions sometimes **make things up**. This is called **hallucination**.
- A 2025 paper called **CVG** found a way to reduce this without retraining the model.
- We built **CVG-E**, an improved version that tries to be **smarter** (better at choosing which parts of the image matter) and **faster** (less computer work per answer).
- On a **small model (1.5B)**, both CVG and CVG-E made things **worse**. That was a surprising and useful finding.
- We are now testing on a **bigger model (7B)**, the same size the CVG paper used. The normal model scored **89.4%**. The CVG and CVG-E results are still running.

---

## Words You Need to Know

| Word | Simple meaning |
| --- | --- |
| **MLLM** | An AI that understands both pictures and text, such as LLaVA. |
| **Hallucination** | The AI says something is in the picture when it is not. |
| **Token** | A small piece of input. An image is cut into 576 small squares (patches), and each square is one "visual token". |
| **Attention** | How much the model "looks at" each token when it answers. |
| **Parameters (1.5B, 7B)** | The size of the model. 7B means 7 billion numbers inside it. Bigger is usually smarter but needs more memory. |
| **Forward pass** | One full run of the model to produce an answer step. |
| **Training-free** | No retraining needed. The trick is applied only when the model answers. |
| **Vanilla** | The normal model with no tricks. Our baseline. |
| **POPE** | A test with yes/no questions like "Is there a dog in the image?" Half the correct answers are "yes" and half are "no". |
| **GPU / VRAM** | The graphics card that runs the model, and the memory on that card. |

---

## Part 1: The Original CVG Paper

**Paper:** "Boosting Faithful Multi-Modal LLMs via Complementary Visual Grounding"
**Authors:** Zheren Fu, Zhendong Mao, Lei Zhang, Yongdong Zhang (USTC)
**Published in:** IEEE Transactions on Image Processing, Vol. 34, 2025

### The Problem

Show the AI a photo of a dog on a beach. It might say: "A woman is playing with a dog, and someone is surfing." There is no woman and no surfer. The AI invented them.

This matters because people want to use these AIs in hospitals, cars and tools for blind users. Made-up answers there can cause real harm.

### Why the AI Makes Things Up

The CVG paper found two reasons.

1. **Not enough visual clues.** When part of the image is unclear, the AI does not say "I am not sure." It guesses from what it learned from text. For example, "beach" often appears with "surfer", so it adds a surfer.
2. **It stops looking at the picture.** In long answers, the AI looks at the image less and less. After about 30–40 words it is mostly writing from memory, not from the picture.

### How CVG Fixes It (Three Steps)

Think of it like a **detective with two witnesses**.

**Step 1 — Split the image (VTD: Visual Token Disentanglement).**
CVG scores every image square by how important it is for the question. It keeps the top 70% as the **"good" set** and puts the bottom 30% in the **"bad" set**. This creates two witnesses:
- The **good witness** sees only the important parts.
- The **bad witness** sees only the unimportant parts, so it is likely to guess wrong.

**Step 2 — Keep the good witness focused (DAC: Dynamic Attention Calibration).**
While the AI writes its answer, CVG pushes the good witness to keep looking at the image. It pushes the bad witness to look away. The push gets stronger as the answer gets longer. This fixes the "stops looking at the picture" problem.

**Step 3 — Compare the two witnesses (GFF: Grounding Flow Fusion).**
- If the two witnesses **disagree**, CVG trusts the good one and moves the answer *away* from what the bad one said. What the bad witness says is probably a made-up guess.
- If they **agree**, the image is clear, so CVG simply uses the good witness.

### How Well CVG Worked

- Tested on LLaVA models with 7B and 13B parameters.
- Beat other methods (VCD, OPERA, PAI) on hallucination tests: POPE, CHAIR, AMBER, MME.
- Also kept or improved general question-answering skills.
- Cost: about **64% more computer work** than the normal model.

### CVG's Weak Spots

1. **Text in images gets thrown away.** Squares that contain writing get low scores and end up in the "bad" set. So CVG does worse on questions about text in pictures (the TextVQA test).
2. **Always keeps exactly 70%.** It does not matter whether the image is simple or busy.
3. **Runs the model twice.** One run for each witness, which makes it slower.

---

## Part 2: Our Work on a Small Model (1.5B)

### Our Idea: CVG-E

We named our version **CVG-E: Efficient Complementary Visual Grounding**. It has two improvements.

**Improvement A — Smarter choice of image squares**

1. **Flexible keep-ratio (adaptive α).** We measure how spread-out the model's attention is.
   - Attention spread over many squares means a busy, unclear image. We keep **more** squares (up to 85%).
   - Attention focused on a few squares means a clear image. We keep **fewer** (down to 50%).
2. **Protect text squares.** We detect squares that probably contain writing. Neighbouring letters look at each other strongly, so we can spot them. We raise their scores so they are not thrown away. This targets the TextVQA problem.
3. **Fairer scoring (z-score).** CVG adds two scores with very different ranges, so one can drown out the other. We rescale both to the same range first, then mix them: 50% question-relevance, 30% general importance, 20% text-likeness.

**Improvement B — Run the model once, not twice**

Instead of a full second run for the bad witness, we **fake** it cheaply:
1. Run the good witness normally.
2. At one layer, take its attention and turn down the image part (×0.1) and turn up the text part (×10).
3. Recompute only that one layer and the final output.

This should cost about **1 layer of work instead of 22 layers**.

### Our Test Setup

- **Model:** TinyLLaVA, 1.5B parameters. Small enough for a laptop.
- **Computer:** MacBook with M1 Pro chip and 16 GB memory.
- **Test:** 180 POPE yes/no questions on 30 COCO photos.
- **Why small?** Phones and cheap devices need small models. Nobody had tested CVG on one.

### What We Found

**Finding 1 — The normal model won.**

| Method | Gave a usable answer | Accuracy | Says "yes" | Time per question |
| --- | --- | --- | --- | --- |
| Vanilla (normal) | 180/180 | **86.7%** | 50% | 0.96 s |
| CVG | 91/180 | 62.6% | 91% | 8.51 s |
| CVG-E | 124/180 | 62.9% | 82% | 17.87 s |

Both tricks made the small model **worse**, not better.

**Finding 2 — The model broke down.**
With CVG, about half the answers were unusable:
- 53 times it just **repeated the question** back.
- 36 times it gave **empty or nonsense** output.

CVG-E broke less often (31% vs 49%) because its change to the model is gentler.

**Finding 3 — It just said "yes" to everything.**
A good model says "yes" about half the time, because half the correct answers are "yes". CVG said "yes" 91% of the time. That makes some numbers look good (it catches almost every real object) but it is cheating, not real improvement.

**Finding 4 — Our "faster" method was slower.**
CVG-E was meant to be faster, but it took 17.87 s per question vs CVG's 8.51 s. The reason: to read attention from one layer, the standard setting (`output_attentions=True`) makes **all 22 layers** use a slow mode. We only needed one. We later fixed this (see Part 3).

**Finding 5 — The "Goldilocks" problem.**
- CVG's change is **too strong**: the witnesses disagree a lot, but the model breaks.
- CVG-E's change is **too weak**: the model stays healthy, but the two witnesses barely disagree, so the trick never activates.
- On a 1.5B model, there may be no "just right" setting.

### Settings We Had to Change for the Small Model

The original settings broke the small model completely, so we softened them:

| Setting | Original (7B) | Ours (1.5B) | Why |
| --- | --- | --- | --- |
| How hard to hide "bad" squares | 0.0 (delete) | 0.01 (dim) | Deleting confuses the model |
| Strength of the comparison (γ) | 1.0 | 0.5 | Full strength overwhelms it |
| When to start comparing (ε) | 0.01 | 0.1 | Otherwise it always fires |
| Strength of the "keep looking" push (β) | 1.0 | 0.1 | Too much push over-corrects |

Even with these changes, the normal model still won.

### What We Learned From the Small Model

1. This is the first test of CVG-style methods on a model of 2B or fewer parameters. **They do not work at that size.**
2. We list the setting changes needed so the small model does not break.
3. We describe exactly *how* it fails: repeating the question, nonsense output, and saying "yes" too often.
4. We show that a "single-pass" method can end up slower if attention is read the wrong way.
5. Small models may simply be too fragile for this kind of trick.

---

## Part 3: Moving to a Bigger Model (7B) — Current Work

### Why

The CVG paper only tested 7B and bigger. So the fair question is: **does CVG-E work at the size CVG was built for?**

### Step 1 — Fixed the speed problem

We built a **single-layer hook** (`SingleLayerHook` in `cvg/hooks.py`).
- Before: all layers switched to slow mode just so we could read one layer.
- Now: **only the one layer we need** uses slow mode. All other layers stay fast.

Think of it like this: before, we stopped every car on the motorway to check one. Now we stop only that car.

### Step 2 — Moved to LLaVA-1.5-7B

The 7B model has 32 layers (the small one had 22). We picked the same layers the CVG paper used:

| Job | Layers |
| --- | --- |
| Score image squares by question relevance | 19–22 |
| Score image squares by general importance | 16–18 |
| "Keep looking at the image" push | 20–31 |
| CVG-E's cheap fake second witness | 28–31 |

At 7B we use the **original** CVG settings, not the softened small-model ones.

### Step 3 — Moved to a cloud computer

A 7B model is too big for the laptop, so we use **Google Colab**, a free online computer with an **NVIDIA T4** graphics card (15 GB memory).
- The test is now much bigger: **3,000 questions** instead of 180.

### Step 4 — First result: the normal 7B model

| Method | Usable answers | Accuracy | Precision | Recall | F1 | Says "yes" | Time per question |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla 7B | 3000/3000 | **89.4%** | 91.3% | 87.1% | 89.1% | 47.7% | 1.16 s |

This matches what other researchers report for this model (about 88–90%). So **our setup is working correctly**.

### Step 5 — Ran out of memory, then fixed it

- The 7B model alone fills about **14 GB of the 15 GB** card.
- CVG needs room for a second run. CVG-E needs room for one big attention table. **Neither fits.**
- Every question crashed. The results showed 50% accuracy and 0 "yes" answers. **Those results are meaningless and we deleted them.**

**Fix: shrink the model (4-bit quantization).**
Every number in the model is stored with less precision, like saving a photo as a smaller JPEG. The model drops to about **5 GB**, which leaves room for CVG and CVG-E.
- New option: `--quantize 4bit`
- The Colab notebook now uses it for CVG and CVG-E automatically.

**Fairness:** the normal model ran at full size, but CVG and CVG-E run shrunk. For the final paper we will also run the normal model shrunk, so all three are compared equally.

### What Is Still To Do

| Experiment | Status |
| --- | --- |
| Normal 7B model, COCO random | Done (89.4%) |
| CVG 7B (shrunk) | Waiting for free Colab time |
| CVG-E 7B (shrunk) | Waiting for free Colab time |
| Normal 7B (shrunk), for fair comparison | Planned |
| Harder question sets (popular, adversarial) | Planned |
| Other tests: CHAIR, AMBER, TextVQA | Planned |
| Compare with another method (VCD) | Planned |

### What We Hope to Show

1. **Accuracy:** CVG-E is as good as CVG or better, because it chooses image squares more cleverly.
2. **Speed:** CVG-E is clearly faster than CVG, because it no longer runs the model twice.
3. **Size matters:** the same method fails at 1.5B but works at 7B. This tells researchers how big a model must be before these tricks help.

---

## How We Did the Work (Step by Step)

1. **Read the research.** Studied the CVG paper and related methods (VCD, ICD, M3ID, OPERA, PAI, AttnReal, ASCD, DaID). Found CVG's three weak spots.
2. **Built CVG from scratch** for TinyLLaVA, since there was no ready-made code for that model.
3. **Built CVG-E's additions:** flexible keep-ratio, text protection, fair scoring and the single-pass trick.
4. **Tuned settings** so the small model did not break.
5. **Ran tests:** 180 questions for each method on the laptop.
6. **Analysed results:** accuracy, error ranges, how often it said "yes", and the types of failure.
7. **Wrote the paper:** 7 pages in IEEE format, with diagrams and tables.
8. **Made the presentation:** 15 slides.
9. **Fixed the speed problem** with the single-layer hook.
10. **Moved to the 7B model** on Google Colab.
11. **Got the 7B normal-model result:** 89.4%.
12. **Hit the memory problem** and fixed it by shrinking the model to 4-bit.

---

## The Big Picture

- **CVG is a good method for big models.** It was published in a top journal and beats other methods at 7B and 13B.
- **On small models it breaks.** The change needed to catch made-up answers also stops the small model from writing sensible answers.
- **We are now checking CVG-E on a big model.** We want to show it is as accurate as CVG but faster. The setup is ready and the baseline (89.4%) is correct. We are waiting for the CVG and CVG-E runs.
