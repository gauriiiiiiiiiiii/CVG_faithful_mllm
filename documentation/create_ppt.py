"""Generate CVG-E research presentation — 14+ slides, classy visual design."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
import os

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── Classy palette ──
BG_DEEP    = RGBColor(0x0F, 0x17, 0x2A)
BG_CARD    = RGBColor(0x17, 0x20, 0x38)
TEAL       = RGBColor(0x00, 0xD2, 0xC6)
CORAL      = RGBColor(0xFF, 0x6B, 0x6B)
GOLD       = RGBColor(0xFF, 0xD9, 0x3D)
MINT       = RGBColor(0x6B, 0xE8, 0xA0)
LAVENDER   = RGBColor(0xA7, 0x8B, 0xFA)
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
SOFT_WHITE = RGBColor(0xE0, 0xE4, 0xED)
DIM        = RGBColor(0x7B, 0x85, 0x9E)
CARD_BORDER= RGBColor(0x2A, 0x34, 0x50)
DARK_CARD  = RGBColor(0x14, 0x1E, 0x33)

TOTAL_SLIDES = 15


def add_bg(slide):
    fill = slide.background.fill; fill.solid(); fill.fore_color.rgb = BG_DEEP

def grad_bar(slide):
    for x, c in [(0, TEAL), (6.667, LAVENDER)]:
        s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), 0, Inches(6.667), Inches(0.06))
        s.fill.solid(); s.fill.fore_color.rgb = c; s.line.fill.background()

def accent(slide, l, t, w, c=TEAL):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(0.04))
    s.fill.solid(); s.fill.fore_color.rgb = c; s.line.fill.background()

def card(slide, l, t, w, h, fc=BG_CARD, bc=CARD_BORDER):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fc; s.line.color.rgb = bc; s.line.width = Pt(1); return s

def txt(sl, l, t, w, h, text, sz=18, c=WHITE, b=False, a=PP_ALIGN.LEFT):
    tb = sl.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; p = tf.paragraphs[0]
    p.text = text; p.font.size = Pt(sz); p.font.color.rgb = c; p.font.bold = b
    p.font.name = "Calibri"; p.alignment = a; return tf

def bul(sl, l, t, w, h, items, sz=14, c=SOFT_WHITE, sp=Pt(5)):
    tb = sl.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item; p.font.size = Pt(sz); p.font.color.rgb = c
        p.font.name = "Calibri"; p.space_after = sp
    return tf

def snum(sl, n):
    pass

def header(sl, title, sub, n):
    add_bg(sl)
    txt(sl, 0.8, 0.35, 10, 0.6, title, sz=30, b=True, c=TEAL)
    if sub: txt(sl, 0.8, 0.9, 10, 0.4, sub, sz=15, c=DIM)
    accent(sl, 0.8, 1.25, 3.5)

def dot(sl, l, t, sz, c):
    s = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(l), Inches(t), Inches(sz), Inches(sz))
    s.fill.solid(); s.fill.fore_color.rgb = c; s.line.fill.background(); return s

def ghost_circle(sl, cx, cy, sz, clr, opacity='7000'):
    s = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx), Inches(cy), Inches(sz), Inches(sz))
    s.fill.solid(); s.fill.fore_color.rgb = clr; s.line.fill.background()
    try:
        spPr = s._element.find(qn('p:spPr'))
        solidFill = spPr.find(qn('a:solidFill'))
        if solidFill is not None:
            srgb = solidFill.find(qn('a:srgbClr'))
            if srgb is not None:
                alpha = srgb.makeelement(qn('a:alpha'), {'val': opacity})
                srgb.append(alpha)
    except Exception:
        pass


# ════════════════════════════════════════
# 1. TITLE
# ════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(sl)
ghost_circle(sl, 11.2, 1.0, 2.0, TEAL)
ghost_circle(sl, 12.0, 2.8, 1.0, LAVENDER)
ghost_circle(sl, 10.5, 3.3, 0.7, GOLD)

txt(sl, 1.0, 1.6, 9.5, 0.9, "CVG-E", sz=56, b=True, c=TEAL)
txt(sl, 1.0, 2.6, 9.5, 0.6, "Efficient Complementary Visual Grounding", sz=28, c=WHITE, b=True)
txt(sl, 1.0, 3.2, 9.5, 0.5, "with Adaptive Token Selection and Single-Pass Approximation", sz=20, c=SOFT_WHITE)
accent(sl, 1.0, 4.0, 4.5)
# author line removed per request
txt(sl, 1.0, 5.2, 9.5, 0.3, "Submitted to: Dr. Gulab Sah", sz=16, c=SOFT_WHITE)
txt(sl, 1.0, 5.6, 9.5, 0.3, "Submitted by: Gauri Agarwal (231210045)   ·   Anirudh Agarwal (231210015)", sz=15, c=SOFT_WHITE)

# ════════════════════════════════════════
# 2. IMPORTANCE & MOTIVATION
# ════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
header(sl, "Introduction", "Importance & Motivation", 2)

card(sl, 0.8, 1.5, 3.7, 3.5)
dot(sl, 1.1, 1.65, 0.45, CORAL)
txt(sl, 1.1, 1.67, 0.45, 0.4, "!", sz=16, b=True, c=WHITE, a=PP_ALIGN.CENTER)
txt(sl, 1.7, 1.7, 2.5, 0.35, "The Problem", sz=17, b=True, c=CORAL)
bul(sl, 1.1, 2.3, 3.2, 2.5, [
    "MLLMs hallucinate objects not present",
    "30-50% of outputs contain fabricated objects",
    "Undermines trust in safety-critical apps",
    "Medical imaging, autonomous driving,",
    "accessibility tools all affected",
], sz=13)

card(sl, 4.8, 1.5, 3.7, 3.5)
dot(sl, 5.1, 1.65, 0.45, MINT)
txt(sl, 5.1, 1.67, 0.45, 0.4, "→", sz=16, b=True, c=WHITE, a=PP_ALIGN.CENTER)
txt(sl, 5.7, 1.7, 2.5, 0.35, "Why Training-Free?", sz=17, b=True, c=MINT)
bul(sl, 5.1, 2.3, 3.2, 2.5, [
    "No additional training data required",
    "No fine-tuning — plug and play",
    "Works with any existing MLLM",
    "Inference-time only: easy deployment",
    "Contrastive decoding: compare signals",
], sz=13)

card(sl, 8.8, 1.5, 3.7, 3.5)
dot(sl, 9.1, 1.65, 0.45, GOLD)
txt(sl, 9.1, 1.67, 0.45, 0.4, "⚡", sz=14, b=True, c=WHITE, a=PP_ALIGN.CENTER)
txt(sl, 9.7, 1.7, 2.5, 0.35, "Why Small Models?", sz=17, b=True, c=GOLD)
bul(sl, 9.1, 2.3, 3.2, 2.5, [
    "Edge deployment demands ≤2B models",
    "Mobile, IoT, resource-constrained",
    "All prior work: 7B+ models only",
    "No one has tested contrastive",
    "decoding at small scale",
], sz=13)

card(sl, 0.8, 5.3, 11.7, 1.4, fc=DARK_CARD)
txt(sl, 1.2, 5.4, 11, 0.35, "Research Goal", sz=17, b=True, c=TEAL)
bul(sl, 1.2, 5.85, 11, 0.7, [
    "Investigate whether contrastive visual grounding can be adapted for efficient, small-model deployment — and characterize the boundary conditions when it cannot.",
], sz=15, c=WHITE)

# ════════════════════════════════════════
# 3. CHALLENGES
# ════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
header(sl, "Key Challenges", "Limitations of Current Contrastive Decoding", 3)

challenges = [
    ("Fixed Token Selection", CORAL, [
        "CVG keeps a fixed α = 0.7 (top 70%) for all images",
        "Simple images: too many tokens kept (noise)",
        "Complex images: too few tokens (information loss)",
        "No adaptation to image difficulty or content",
    ]),
    ("Text Patch Destruction", GOLD, [
        "Visual tokens for text regions scored low",
        "OCR-relevant patches systematically rejected",
        "TextVQA performance degrades significantly",
        "No mechanism to protect text-containing patches",
    ]),
    ("Dual Forward Pass Cost", LAVENDER, [
        "Full forward pass for chosen branch",
        "Full forward pass for rejected branch",
        "2× latency at inference time",
        "Impractical for real-time applications",
    ]),
    ("Small Model Fragility", MINT, [
        "1.5B models: narrow residual capacity",
        "Embedding suppression breaks generation",
        "Hyperparameters tuned for 7B+ only",
        "No calibration guidelines for small scale",
    ]),
]

for i, (title, color, items) in enumerate(challenges):
    x = 0.8 + (i % 4) * 3.1
    y = 1.5
    card(sl, x, y, 2.8, 4.5, bc=color)
    dot(sl, x + 0.15, y + 0.15, 0.4, color)
    txt(sl, x + 0.15, y + 0.17, 0.4, 0.35, str(i+1), sz=14, b=True, c=WHITE, a=PP_ALIGN.CENTER)
    txt(sl, x + 0.65, y + 0.12, 2.05, 0.6, title, sz=15, b=True, c=color)
    accent(sl, x + 0.15, y + 0.85, 1.5, color)
    bul(sl, x + 0.15, y + 1.0, 2.5, 3.3, items, sz=12)

# ════════════════════════════════════════
# 4. PROBLEM DEFINITION
# ════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
header(sl, "Problem Definition", "Object Hallucination in Multi-Modal LLMs", 4)

card(sl, 0.8, 1.5, 5.6, 3.0)
txt(sl, 1.1, 1.6, 5.0, 0.4, "The Hallucination Problem", sz=18, b=True, c=CORAL)
accent(sl, 1.1, 2.0, 2.0, CORAL)
bul(sl, 1.1, 2.15, 5.0, 2.2, [
    "Input: Image I + Question Q",
    "  Example: \"Is there a cat in the image?\"",
    "",
    "Expected: \"No\" (when no cat exists)",
    "Actual: \"Yes, there is a cat\" — HALLUCINATION",
    "",
    "Root cause: language prior dominates visual evidence",
    "Model attends equally to relevant + irrelevant tokens",
], sz=13)

card(sl, 6.8, 1.5, 5.6, 3.0)
txt(sl, 7.1, 1.6, 5.0, 0.4, "CVG's Approach (Fu et al., TIP 2025)", sz=18, b=True, c=MINT)
accent(sl, 7.1, 2.0, 2.5, MINT)
bul(sl, 7.1, 2.15, 5.0, 2.2, [
    "Visual Token Disentanglement (VTD):",
    "  Split tokens → \"chosen\" + \"rejected\" sets",
    "",
    "Dynamic Attention Calibration (DAC):",
    "  Re-weight attention scores dynamically",
    "",
    "Grounding Flow Fusion (GFF):",
    "  Contrast chosen vs rejected → suppress hallucination",
], sz=13)

card(sl, 0.8, 4.8, 11.7, 2.0, fc=DARK_CARD)
txt(sl, 1.2, 4.9, 11, 0.35, "Formal Research Questions", sz=17, b=True, c=GOLD)
for i, (q, c) in enumerate([
    ("RQ1: Does CVG's contrastive decoding transfer to 1.5B-parameter models?", CORAL),
    ("RQ2: Can single-pass attention re-weighting replace the expensive dual forward pass?", LAVENDER),
    ("RQ3: Can entropy-adaptive token selection improve robustness across image types?", MINT),
    ("RQ4: At 7B scale, does CVG-E match CVG's accuracy while running faster?", TEAL),
]):
    y = 5.35 + i * 0.34
    dot(sl, 1.2, y + 0.03, 0.25, c)
    txt(sl, 1.55, y, 10.5, 0.35, q, sz=14, c=SOFT_WHITE)

# ════════════════════════════════════════
# 5. RELATED WORK — CONTRASTIVE METHODS
# ════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
header(sl, "Related Work", "Contrastive Decoding Methods", 5)

methods = [
    ("VCD", "Leng et al., 2024", "Gaussian-noise contrast on image input", CORAL),
    ("ICD", "2024", "Instruction-level contrastive input", CORAL),
    ("M3ID", "2024", "Image-conditioned vs text-only distributions", CORAL),
    ("OPERA", "Huang et al., 2024", "Over-trust attention penalty + rollback", GOLD),
    ("PAI", "Liu et al., 2024", "Contrastive decoding + attention amplification", GOLD),
]

for i, (name, cite, desc, c) in enumerate(methods):
    y = 1.5 + i * 0.95
    card(sl, 0.8, y, 11.7, 0.8, bc=c)
    txt(sl, 1.2, y + 0.1, 1.5, 0.35, name, sz=18, b=True, c=c)
    txt(sl, 2.8, y + 0.1, 2.5, 0.35, cite, sz=12, c=DIM)
    txt(sl, 5.5, y + 0.1, 6.5, 0.35, desc, sz=14, c=SOFT_WHITE)
    txt(sl, 5.5, y + 0.4, 6.5, 0.3, "Requires full corrupted forward pass → ~2× cost", sz=11, c=DIM)

card(sl, 0.8, 6.3, 11.7, 0.7, fc=DARK_CARD, bc=TEAL)
txt(sl, 1.2, 6.4, 11, 0.4,
    "Common limitation: ALL these methods require a second forward pass, roughly doubling inference latency.",
    sz=14, b=True, c=TEAL)

# ════════════════════════════════════════
# 6. RELATED WORK — ATTENTION-BASED & CVG
# ════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
header(sl, "Related Work", "Attention-Based Methods & CVG Baseline", 6)

card(sl, 0.8, 1.5, 5.6, 3.5, bc=MINT)
txt(sl, 1.1, 1.6, 5.0, 0.4, "Attention-Based (Near-Zero Overhead)", sz=17, b=True, c=MINT)
accent(sl, 1.1, 2.0, 2.5, MINT)
bul(sl, 1.1, 2.15, 5.0, 2.7, [
    "AttnReal (2025)",
    "  Directly reallocate attention — no contrastive branches",
    "ASCD (arXiv, 2025)",
    "  Steer individual attention heads by text/object roles",
    "DaID (2025)",
    "  Exploit natural layer-wise divergence",
    "",
    "Key insight: contrastive signals already exist in",
    "internal representations → motivates single-pass",
], sz=13)

card(sl, 6.8, 1.5, 5.6, 3.5, bc=LAVENDER)
txt(sl, 7.1, 1.6, 5.0, 0.4, "CVG — Our Baseline", sz=17, b=True, c=LAVENDER)
accent(sl, 7.1, 2.0, 2.0, LAVENDER)
bul(sl, 7.1, 2.15, 5.0, 2.7, [
    "Fu et al., IEEE TIP 2025",
    "Three-component framework:",
    "  VTD — Visual Token Disentanglement",
    "  DAC — Dynamic Attention Calibration",
    "  GFF — Grounding Flow Fusion",
    "",
    "State-of-the-art results:",
    "  POPE, CHAIR, AMBER benchmarks",
    "  Validated on 7B+ models only",
], sz=13)

card(sl, 0.8, 5.3, 11.7, 1.3, fc=RGBColor(0x2A, 0x15, 0x15), bc=CORAL)
txt(sl, 1.2, 5.4, 1.2, 0.35, "GAP", sz=16, b=True, c=CORAL)
txt(sl, 2.5, 5.4, 9.5, 0.9,
    "No study has tested contrastive decoding at ≤2B scale. No efficient single-pass alternatives validated. "
    "No adaptive token selection beyond fixed ratios. No calibration guidelines for small models.",
    sz=14, c=SOFT_WHITE)

# ════════════════════════════════════════
# 7. METHODOLOGY — IMPROVEMENT A
# ════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
header(sl, "Methodology", "Improvement A: Adaptive Token Selection", 7)

card(sl, 0.8, 1.5, 5.6, 4.5, bc=MINT)
dot(sl, 1.0, 1.6, 0.5, MINT)
txt(sl, 1.0, 1.62, 0.5, 0.4, "1", sz=16, b=True, c=WHITE, a=PP_ALIGN.CENTER)
txt(sl, 1.65, 1.65, 4.5, 0.4, "Entropy-Adaptive α", sz=18, b=True, c=MINT)
accent(sl, 1.1, 2.1, 2.0, MINT)
bul(sl, 1.1, 2.3, 5.0, 2.0, [
    "Selection ratio α adapts per-image based on",
    "attention entropy across visual tokens",
    "",
    "High entropy (complex scene):",
    "  → Keep more tokens (larger α)",
    "  → Preserve diverse visual information",
    "",
    "Low entropy (simple scene):",
    "  → Aggressive filtering (smaller α)",
    "  → Focus on dominant objects",
], sz=13)

card(sl, 6.8, 1.5, 5.6, 2.0, bc=GOLD)
dot(sl, 7.0, 1.6, 0.5, GOLD)
txt(sl, 7.0, 1.62, 0.5, 0.4, "2", sz=16, b=True, c=WHITE, a=PP_ALIGN.CENTER)
txt(sl, 7.65, 1.65, 4.5, 0.4, "Text-Patch Protection", sz=18, b=True, c=GOLD)
accent(sl, 7.1, 2.1, 2.0, GOLD)
bul(sl, 7.1, 2.3, 5.0, 1.0, [
    "Identify OCR-relevant visual patches",
    "Protect them from rejection during VTD",
    "Preserves performance on text-rich images (TextVQA)",
], sz=13)

card(sl, 6.8, 3.7, 5.6, 2.3, bc=LAVENDER)
dot(sl, 7.0, 3.8, 0.5, LAVENDER)
txt(sl, 7.0, 3.82, 0.5, 0.4, "3", sz=16, b=True, c=WHITE, a=PP_ALIGN.CENTER)
txt(sl, 7.65, 3.85, 4.5, 0.4, "Z-Score Hybrid Scoring", sz=18, b=True, c=LAVENDER)
accent(sl, 7.1, 4.3, 2.0, LAVENDER)
bul(sl, 7.1, 4.5, 5.0, 1.3, [
    "Combine attention weights + entropy signals",
    "Z-score normalization across tokens",
    "Fair comparison regardless of score distributions",
    "More robust token classification",
], sz=13)

# ════════════════════════════════════════
# 8. METHODOLOGY — IMPROVEMENT B + CALIBRATION
# ════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
header(sl, "Methodology", "Improvement B: Single-Pass Approximation + Calibration", 8)

card(sl, 0.8, 1.5, 6.5, 3.0, bc=LAVENDER)
txt(sl, 1.1, 1.6, 6.0, 0.4, "Single-Pass Branch Approximation", sz=18, b=True, c=LAVENDER)
accent(sl, 1.1, 2.0, 3.0, LAVENDER)
bul(sl, 1.1, 2.2, 6.0, 2.0, [
    "Instead of running TWO full forward passes:",
    "① Run chosen branch (single forward pass)",
    "② Cache attention maps at a designated layer",
    "③ Re-weight cached attention to approximate rejected branch",
    "④ Compute contrastive signal from re-weighted logits",
    "Theoretical complexity: O(1) layer ops vs O(L) full pass",
], sz=14)

card(sl, 7.7, 1.5, 4.8, 3.0, bc=CORAL)
txt(sl, 8.0, 1.6, 4.3, 0.4, "Small-Model Calibration", sz=18, b=True, c=CORAL)
accent(sl, 8.0, 2.0, 2.0, CORAL)

# Mini table
params = [
    ("Parameter", "Original", "Ours", "Failure Mode"),
    ("suppress", "0.0 (zero)", "0.01 (scale)", "Generation collapse"),
    ("γ", "1.0", "0.5", "Off-distribution output"),
    ("ε", "0.01", "0.1", "Constant contrastive"),
    ("β (DAC)", "1.0", "0.1", "Over-calibration"),
]
for i, (p, orig, ours, fail) in enumerate(params):
    y = 2.15 + i * 0.4
    sz_ = 10 if i > 0 else 11
    b_ = i == 0
    c_ = DIM if i == 0 else SOFT_WHITE
    txt(sl, 7.9, y, 0.95, 0.3, p, sz=sz_, b=b_, c=c_)
    txt(sl, 8.8, y, 1.05, 0.3, orig, sz=sz_, c=c_)
    txt(sl, 9.8, y, 1.1, 0.3, ours, sz=sz_, c=MINT if i > 0 else c_)
    txt(sl, 10.85, y, 1.6, 0.3, fail, sz=sz_, c=CORAL if i > 0 else c_)

card(sl, 0.8, 4.8, 11.7, 1.8, fc=DARK_CARD)
txt(sl, 1.2, 4.9, 11, 0.35, "Evaluation Setup", sz=17, b=True, c=TEAL)
items = [
    ("Model", "TinyLLaVA-1.5B (bczhou/tiny-llava-v1-hf)", TEAL),
    ("Benchmark", "POPE random split, COCO val2014, 180 questions (30 images × 6 q/image, balanced yes/no)", MINT),
    ("Hardware", "Apple M1 Pro, 16GB unified memory, MPS backend, greedy decoding, seed=42", LAVENDER),
    ("Metrics", "Accuracy, Precision, Recall, F1, Yes-ratio, 95% Wilson confidence intervals", GOLD),
]
for i, (label, val, c) in enumerate(items):
    y = 5.35 + i * 0.3
    txt(sl, 1.2, y, 1.5, 0.25, label, sz=12, b=True, c=c)
    txt(sl, 2.8, y, 9.5, 0.25, val, sz=12, c=SOFT_WHITE)

# ════════════════════════════════════════
# 9. RESULTS — POPE TABLE
# ════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
header(sl, "Results", "POPE Evaluation (N=180, TinyLLaVA-1.5B)", 9)

card(sl, 0.8, 1.5, 11.7, 2.6)
headers = ["Method", "Parseable", "Acc %", "Prec %", "Rec %", "F1 %", "Yes %", "Time/q"]
col_x =   [1.0,      2.6,        4.3,     5.6,      6.9,     8.2,    9.5,     10.9]

for i, h in enumerate(headers):
    txt(sl, col_x[i], 1.6, 1.3, 0.3, h, sz=13, b=True, c=TEAL, a=PP_ALIGN.CENTER)
accent(sl, 1.0, 1.95, 11.1, DIM)

rows = [
    (["Vanilla", "180/180", "86.7", "86.7", "86.7", "86.7", "50.0", "0.96s"], MINT, {}),
    (["CVG",     "91/180",  "62.6", "60.2", "98.0", "74.6", "91.2", "8.51s"], GOLD, {2:CORAL, 3:CORAL, 6:CORAL}),
    (["CVG-E",   "124/180", "62.9", "61.4", "89.9", "72.9", "81.5", "17.87s"], LAVENDER, {2:CORAL, 3:CORAL, 6:CORAL}),
]
for ri, (vals, rc, highlights) in enumerate(rows):
    y = 2.1 + ri * 0.55
    for ci, v in enumerate(vals):
        c = rc if ci == 0 else highlights.get(ci, SOFT_WHITE)
        txt(sl, col_x[ci], y, 1.3, 0.35, v, sz=14, c=c, b=(ci==0), a=PP_ALIGN.CENTER)

# Wilson CIs
txt(sl, 1.0, 3.6, 11, 0.3,
    "95% Wilson CIs: Vanilla Acc 81–91%  |  CVG Acc 52–72%  |  CVG Prec 50–70%  |  CVG Rec 90–100%",
    sz=11, c=DIM)

# Interpretation
card(sl, 0.8, 4.3, 11.7, 2.8, fc=DARK_CARD)
txt(sl, 1.2, 4.4, 5.0, 0.35, "Interpretation", sz=17, b=True, c=WHITE)

bul(sl, 1.1, 4.85, 5.3, 2.0, [
    "Vanilla: strong baseline at 86.7%",
    "  Balanced: 50% yes-ratio, symmetric errors",
    "  All 180 responses parseable",
    "",
    "CVG: catastrophic at small scale",
    "  49% responses unparseable",
    "  Parseable subset: extreme Yes-bias (91%)",
    "  Accuracy drops 24pp to 62.6%",
], sz=13)

bul(sl, 6.8, 4.85, 5.3, 2.0, [
    "CVG-E: also fails at small scale",
    "  31% responses unparseable (47 echo, 9 degen)",
    "  82% yes-bias, 63% accuracy = same as CVG",
    "",
    "Key takeaway:",
    "  BOTH contrastive methods actively",
    "  HURT performance at 1.5B scale —",
    "  they don't just fail to help, they degrade",
], sz=13)

# ════════════════════════════════════════
# 10. RESULTS — FAILURE ANALYSIS
# ════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
header(sl, "Failure Analysis", "CVG's Two Failure Modes at Small Scale", 10)

# Mode 1: Unparseable
card(sl, 0.8, 1.5, 5.6, 3.3, bc=CORAL)
txt(sl, 1.1, 1.6, 5.0, 0.4, "Mode 1: Generation Collapse (49%)", sz=17, b=True, c=CORAL)
accent(sl, 1.1, 2.0, 2.5, CORAL)
bul(sl, 1.1, 2.15, 5.0, 2.5, [
    "89 of 180 responses are unparseable:",
    "Question Echoing (53 responses, 60%):",
    "  Model repeats question verbatim",
    "  Insufficient grounding after suppression",
    "  Defaults to copying input tokens",
    "Degenerate Output (36 responses, 40%):",
    "  Empty or truncated responses",
    "  Contrastive adjustment → near-zero probabilities",
], sz=13)

# Mode 2: Yes-bias
card(sl, 6.8, 1.5, 5.6, 3.3, bc=GOLD)
txt(sl, 7.1, 1.6, 5.0, 0.4, "Mode 2: Extreme Yes-Bias (91%)", sz=17, b=True, c=GOLD)
accent(sl, 7.1, 2.0, 2.5, GOLD)
bul(sl, 7.1, 2.15, 5.0, 2.5, [
    "Among 91 parseable CVG responses:",
    "Confusion Matrix:",
    "  TP=50  FP=33  TN=7  FN=1",
    "91.2% yes-ratio (vs 50% ground truth)",
    "98% recall but only 60% precision",
    "33× more false positives than false negatives",
    "Not hallucination reduction — bias shift",
], sz=13)

# Bottom: Goldilocks
card(sl, 0.8, 5.1, 11.7, 1.8, fc=DARK_CARD)
txt(sl, 1.2, 5.2, 11, 0.35, "The Goldilocks Problem — No Viable Operating Point", sz=17, b=True, c=GOLD)
bul(sl, 1.2, 5.65, 3.5, 1.0, [
    "TOO HOT (CVG dual-pass):",
    "D_KL ~ 0.5 → generation collapse",
], sz=13, c=CORAL)
bul(sl, 5.0, 5.65, 3.5, 1.0, [
    "TOO COLD (CVG-E single-pass):",
    "D_KL < 0.01 → no contrastive effect",
], sz=13, c=LAVENDER)
bul(sl, 8.8, 5.65, 3.5, 1.0, [
    "MIDDLE GROUND?",
    "Multi-layer re-weighting — untested",
], sz=13, c=GOLD)

# ════════════════════════════════════════
# 11. RESULTS — LATENCY & KL DIVERGENCE
# ════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
header(sl, "Latency & KL Divergence", "Performance Characteristics", 11)

card(sl, 0.8, 1.5, 5.6, 3.0, bc=TEAL)
txt(sl, 1.1, 1.6, 5.0, 0.4, "Latency Analysis", sz=18, b=True, c=TEAL)
accent(sl, 1.1, 2.0, 2.0, TEAL)

latency = [
    ("Method", "Time/q", "vs Vanilla", "Bottleneck"),
    ("Vanilla", "0.96s", "1.0×", "—"),
    ("CVG", "8.51s", "8.9×", "2nd forward pass"),
    ("CVG-E", "17.87s", "18.6×", "Eager attention"),
]
for i, (m, t, v, b) in enumerate(latency):
    y = 2.2 + i * 0.4
    sz_ = 11 if i > 0 else 12
    b_ = i == 0
    c_ = DIM if i == 0 else SOFT_WHITE
    txt(sl, 1.1, y, 1.3, 0.3, m, sz=sz_, b=b_, c=c_)
    txt(sl, 2.5, y, 1.0, 0.3, t, sz=sz_, c=CORAL if m == "CVG-E" else c_)
    txt(sl, 3.6, y, 1.0, 0.3, v, sz=sz_, c=c_)
    txt(sl, 4.7, y, 2.0, 0.3, b, sz=sz_, c=c_)

bul(sl, 1.1, 3.75, 5.0, 0.5, [
    "Latency inversion: CVG-E SLOWER than CVG!",
    "output_attentions=True → O(n²) at all 22 layers",
], sz=12, c=CORAL)

card(sl, 6.8, 1.5, 5.6, 3.0, bc=LAVENDER)
txt(sl, 7.1, 1.6, 5.0, 0.4, "KL Divergence Analysis", sz=18, b=True, c=LAVENDER)
accent(sl, 7.1, 2.0, 2.5, LAVENDER)
bul(sl, 7.1, 2.2, 5.0, 2.2, [
    "CVG dual-pass: D_KL ~ 0.5",
    "  Contrastive active (above ε = 0.1)",
    "  But: too much divergence → collapse",
    "CVG-E single-pass: D_KL < 0.01",
    "  Far below ε threshold",
    "  Contrastive correction never fires",
    "  Effectively reduces to vanilla decoding",
], sz=13)

card(sl, 0.8, 4.8, 11.7, 1.8, fc=DARK_CARD, bc=MINT)
txt(sl, 1.2, 4.9, 11, 0.35, "Fix Implemented: Single-Layer Hook", sz=17, b=True, c=MINT)
bul(sl, 1.2, 5.3, 11, 1.0, [
    "SingleLayerHook (cvg/hooks.py): eager attention on ONE layer only; all other layers keep SDPA / Flash Attention",
    "Captures that layer's attention weights + input hidden state → rejected branch approximated without a 2nd pass",
    "Removes the O(n²)-at-every-layer cost behind the 18.6× slowdown; being validated at 7B scale (next slide)",
], sz=13)

# ════════════════════════════════════════
# 12. SCALING TO 7B
# ════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
header(sl, "Scaling to 7B", "LLaVA-1.5-7B on Cloud GPU — Same Scale as the Original CVG Paper", 12)

card(sl, 0.8, 1.5, 4.6, 3.0, bc=TEAL)
txt(sl, 1.1, 1.6, 4.2, 0.4, "Setup", sz=18, b=True, c=TEAL)
accent(sl, 1.1, 2.0, 1.5, TEAL)
setup = [
    ("Model", "llava-hf/llava-1.5-7b-hf"),
    ("Layers", "32 LM  ·  24 ViT"),
    ("VTD cross", "layers 19–22"),
    ("VTD self", "layers 16–18"),
    ("DAC", "layers 20–31"),
    ("CVG-E approx", "layers 28–31"),
    ("GPU", "Colab Tesla T4, 15 GB"),
    ("Benchmark", "POPE COCO, 3,000 q / split"),
]
for i, (k, v) in enumerate(setup):
    y = 2.15 + i * 0.28
    txt(sl, 1.1, y, 1.6, 0.25, k, sz=11, b=True, c=DIM)
    txt(sl, 2.7, y, 2.6, 0.25, v, sz=11, c=SOFT_WHITE)

card(sl, 5.7, 1.5, 6.8, 3.0, bc=LAVENDER)
txt(sl, 6.0, 1.6, 6.3, 0.4, "POPE COCO Random — 7B Results", sz=18, b=True, c=LAVENDER)
accent(sl, 6.0, 2.0, 2.5, LAVENDER)
h7 = ["Method", "Acc %", "Prec %", "Rec %", "F1 %", "Yes %", "Time/q"]
cx7 = [6.0, 7.45, 8.25, 9.05, 9.85, 10.65, 11.45]
for i, h in enumerate(h7):
    txt(sl, cx7[i], 2.15, 1.0, 0.3, h, sz=11, b=True, c=TEAL)
rows7 = [
    (["Vanilla", "89.4", "91.3", "87.1", "89.1", "47.7", "1.16s"], MINT, SOFT_WHITE),
    (["CVG", "pending", "", "", "", "", ""], GOLD, DIM),
    (["CVG-E", "pending", "", "", "", "", ""], LAVENDER, DIM),
]
for ri, (vals, rc, vc) in enumerate(rows7):
    y = 2.55 + ri * 0.4
    for ci, v in enumerate(vals):
        txt(sl, cx7[ci], y, 1.0, 0.3, v, sz=13, b=(ci == 0), c=rc if ci == 0 else vc)
bul(sl, 6.0, 3.8, 6.3, 0.6, [
    "Vanilla: 3000/3000 parseable, balanced yes-ratio — matches published LLaVA-1.5-7B (~88–90%)",
    "CVG / CVG-E: re-running with 4-bit quantization (waiting on GPU quota)",
], sz=11, c=SOFT_WHITE, sp=Pt(2))

card(sl, 0.8, 4.8, 5.6, 2.1, bc=CORAL)
txt(sl, 1.1, 4.9, 5.0, 0.35, "Finding: Out-of-Memory on T4", sz=16, b=True, c=CORAL)
bul(sl, 1.1, 5.3, 5.0, 1.5, [
    "7B fp16 weights use ~14 GB of the T4's 15 GB",
    "No room for CVG's 2nd pass or CVG-E's attention map",
    "Every question OOM'd → 50% acc, 0% recall",
    "Those runs are invalid and were discarded",
], sz=12)

card(sl, 6.8, 4.8, 5.7, 2.1, bc=MINT)
txt(sl, 7.1, 4.9, 5.0, 0.35, "Fix: 4-bit NF4 Quantization", sz=16, b=True, c=MINT)
bul(sl, 7.1, 5.3, 5.2, 1.5, [
    "bitsandbytes NF4 + double quant → model ~5 GB",
    "New flag: --quantize 4bit (cvg/model.py, run_7b_eval.py)",
    "Notebook applies it to CVG and CVG-E automatically",
    "Vanilla will also be re-run in 4-bit for a fair comparison",
], sz=12)

# ════════════════════════════════════════
# 13. PUBLICATION
# ════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
header(sl, "Publication", "Paper Status & Target Venues", 13)

card(sl, 0.8, 1.5, 7.5, 2.8, bc=TEAL)
txt(sl, 1.1, 1.6, 7.0, 0.4, "Paper Details", sz=18, b=True, c=TEAL)
accent(sl, 1.1, 2.0, 2.5, TEAL)
bul(sl, 1.1, 2.15, 7.0, 2.0, [
    "Title: CVG-E: Efficient Complementary Visual Grounding",
    "          with Adaptive Token Selection and Single-Pass Approximation",
    "Authors: Dr. Gulab Sah, Gauri Agarwal, Anirudh Agarwal",
    "Length: 7 pages  |  15 references  |  4 figures  |  5 tables",
    "Status: 1.5B study complete; 7B experiments in progress",
], sz=14)

# Progress tracker
card(sl, 8.7, 1.5, 3.8, 2.8, bc=MINT)
txt(sl, 9.0, 1.6, 3.3, 0.35, "Progress Tracker", sz=16, b=True, c=MINT)
accent(sl, 9.0, 1.95, 1.5, MINT)
items = [
    ("1.5B study (N=180)", MINT), ("Hook-based single pass", MINT), ("7B port + Colab pipeline", MINT),
    ("7B vanilla (89.4%)", MINT), ("7B CVG / CVG-E (4-bit)", GOLD), ("CHAIR / AMBER / TextVQA", DIM),
    ("Submission", DIM),
]
for i, (item, c) in enumerate(items):
    y = 2.05 + i * 0.31
    dot(sl, 9.1, y + 0.05, 0.13, c)
    txt(sl, 9.35, y, 3.0, 0.3, item, sz=12, c=SOFT_WHITE)

# Contributions & venues
card(sl, 0.8, 4.6, 5.6, 2.3, bc=GOLD)
txt(sl, 1.1, 4.7, 5.0, 0.35, "Novel Contributions", sz=16, b=True, c=GOLD)
bul(sl, 1.1, 5.1, 5.0, 1.5, [
    "First evaluation of contrastive decoding at ≤2B scale",
    "Single-layer hook: true single-pass CVG-E",
    "1.5B vs 7B scale comparison of CVG / CVG-E",
    "Open-source CVG/CVG-E + calibration guidelines",
], sz=13)

card(sl, 6.8, 4.6, 5.6, 2.3, bc=LAVENDER)
txt(sl, 7.1, 4.7, 5.0, 0.35, "Target Venues", sz=16, b=True, c=LAVENDER)
bul(sl, 7.1, 5.1, 5.0, 1.5, [
    "AAAI / CVPR Workshops on MLLMs",
    "ACL Findings (NLP + Vision track)",
    "IEEE Access / IEEE Signal Processing Letters",
    "arXiv preprint for community feedback",
], sz=13)

# ════════════════════════════════════════
# 14. NEXT 2-MONTH PLAN
# ════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
header(sl, "Next 2-Month Plan", "December 2026 — January 2027", 14)

card(sl, 0.8, 1.5, 5.6, 4.0, bc=MINT)
txt(sl, 1.1, 1.6, 5.0, 0.4, "Month 1 — December 2026", sz=18, b=True, c=MINT)
accent(sl, 1.1, 2.0, 2.5, MINT)

txt(sl, 1.1, 2.15, 5.0, 0.3, "Week 1-2: Finish 7B POPE", sz=14, b=True, c=WHITE)
bul(sl, 1.1, 2.5, 5.0, 1.0, [
    "CVG + CVG-E (4-bit) on COCO random",
    "Vanilla in 4-bit for same-precision comparison",
    "Popular + adversarial splits for all methods",
], sz=12)

txt(sl, 1.1, 3.5, 5.0, 0.3, "Week 3-4: Extend Benchmarks", sz=14, b=True, c=WHITE)
bul(sl, 1.1, 3.85, 5.0, 1.0, [
    "CHAIR and AMBER on 7B",
    "TextVQA (validate text-patch protection)",
    "VCD baseline comparison",
], sz=12)

card(sl, 6.8, 1.5, 5.6, 4.0, bc=LAVENDER)
txt(sl, 7.1, 1.6, 5.0, 0.4, "Month 2 — January 2027", sz=18, b=True, c=LAVENDER)
accent(sl, 7.1, 2.0, 2.5, LAVENDER)

txt(sl, 7.1, 2.15, 5.0, 0.3, "Week 5-6: Ablations & Latency", sz=14, b=True, c=WHITE)
bul(sl, 7.1, 2.5, 5.0, 1.0, [
    "Ablate each CVG-E component at 7B",
    "Latency profiling: CVG vs CVG-E vs vanilla",
    "Multi-layer re-weighting at 1.5B (Goldilocks gap)",
], sz=12)

txt(sl, 7.1, 3.5, 5.0, 0.3, "Week 7-8: Finalize & Submit", sz=14, b=True, c=WHITE)
bul(sl, 7.1, 3.85, 5.0, 1.0, [
    "Complete all tables and figures",
    "Internal review and revisions",
    "Submit to target venue",
    "Release code on GitHub",
], sz=12)

# Timeline
card(sl, 0.8, 5.8, 11.7, 1.2, fc=DARK_CARD)
milestones = [
    ("Dec W2", "7B POPE done", TEAL),
    ("Dec W4", "CHAIR/AMBER/TextVQA", MINT),
    ("Jan W2", "Ablations + VCD", LAVENDER),
    ("Jan W4", "Paper submitted", GOLD),
]
for i, (date, task, c) in enumerate(milestones):
    x = 1.2 + i * 2.9
    d = dot(sl, x, 6.05, 0.22, c)
    txt(sl, x + 0.3, 5.82, 2.2, 0.2, date, sz=12, b=True, c=c)
    txt(sl, x + 0.3, 6.25, 2.2, 0.2, task, sz=11, c=SOFT_WHITE)
    if i < 3:
        line = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE,
            Inches(x + 0.22), Inches(6.14), Inches(2.68), Inches(0.02))
        line.fill.solid(); line.fill.fore_color.rgb = CARD_BORDER; line.line.fill.background()

# ════════════════════════════════════════
# 15. THANK YOU
# ════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(sl)
ghost_circle(sl, 1.5, 4.5, 2.5, TEAL)
ghost_circle(sl, 0.3, 3.5, 1.5, LAVENDER)
ghost_circle(sl, 10.5, 1.2, 1.8, GOLD)
ghost_circle(sl, 11.5, 3.0, 1.0, MINT)

txt(sl, 1.5, 2.5, 10.3, 1.0, "Thank You", sz=52, b=True, c=TEAL, a=PP_ALIGN.CENTER)
accent(sl, 4.5, 3.6, 4.3)
txt(sl, 1.5, 4.2, 10.3, 0.4,
    "https://github.com/gauriiiiiiiiiiii/CVG_faithful_mllm", sz=15, c=DIM, a=PP_ALIGN.CENTER)

# ── Save ──
out = "/Users/gauri/Code/Gauri/Research Paper/documentation"
path = os.path.join(out, "CVG-E_Research_Presentation.pptx")
prs.save(path)
print(f"Saved: {path}")
