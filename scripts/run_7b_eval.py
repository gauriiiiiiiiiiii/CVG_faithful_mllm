"""Run comprehensive POPE evaluation on LLaVA-1.5-7B with GPU.

Evaluates Vanilla, CVG, and CVG-E (with hook-based single-pass) across
all POPE splits. Designed for Colab/cloud GPU with A100/T4/L4.

Usage:
    python scripts/run_7b_eval.py --split random --max-questions 3000
    python scripts/run_7b_eval.py --split all --max-questions 3000
    python scripts/run_7b_eval.py --split random --methods vanilla cvg_7b cvg_e_7b
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cvg.modules import CVGEConfig, ABLATIONS
from cvg.model import CVGModel


POPE_SPLITS = {
    "coco_random": "coco_pope_random.json",
    "coco_popular": "coco_pope_popular.json",
    "coco_adversarial": "coco_pope_adversarial.json",
    "aokvqa_random": "aokvqa_pope_random.json",
    "aokvqa_popular": "aokvqa_pope_popular.json",
    "aokvqa_adversarial": "aokvqa_pope_adversarial.json",
    "gqa_random": "gqa_pope_random.json",
    "gqa_popular": "gqa_pope_popular.json",
    "gqa_adversarial": "gqa_pope_adversarial.json",
}


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return 0, 0
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    spread = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
    return max(0, center - spread), min(1, center + spread)


def parse_yes_no(response: str) -> str:
    response = response.strip().lower()
    if response.startswith("yes"):
        return "yes"
    if response.startswith("no"):
        return "no"
    if "yes" in response and "no" not in response:
        return "yes"
    if "no" in response and "yes" not in response:
        return "no"
    return "unknown"


def load_pope_questions(pope_dir, split_name):
    fname = POPE_SPLITS.get(split_name)
    if fname is None:
        raise ValueError(f"Unknown split: {split_name}. Options: {list(POPE_SPLITS.keys())}")
    fpath = Path(pope_dir) / fname
    if not fpath.exists():
        raise FileNotFoundError(f"POPE file not found: {fpath}")
    questions = []
    with open(fpath) as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                questions.append(q)
    return questions


def run_method(model, questions, image_dir, method_name, generate_fn,
               max_questions=None, save_path=None):
    if max_questions and max_questions < len(questions):
        questions = questions[:max_questions]

    tp = fp = tn = fn = 0
    n_parseable = 0
    n_echo = 0
    n_degen = 0
    total_time = 0.0
    responses = []

    for q in tqdm(questions, desc=f"[{method_name}]"):
        img_name = q.get("image", "")
        label = q.get("label", "").lower()
        text = q.get("text", "")

        img_path = Path(image_dir) / img_name
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        prompt = f"USER: {text}\nASSISTANT:"

        t0 = time.time()
        try:
            gen = generate_fn(img, prompt, max_new_tokens=10)
            response = gen.text
            kls = getattr(gen, "kl_divergences", [])
        except Exception as e:
            response = f"ERROR: {e}"
            kls = []
        elapsed = time.time() - t0
        total_time += elapsed

        pred = parse_yes_no(response)

        entry = {
            "question": text, "image": img_name,
            "label": label, "pred": pred, "response": response,
            "time": elapsed,
        }
        if kls:
            entry["avg_kl"] = sum(kls) / len(kls) if kls else 0.0

        responses.append(entry)

        if pred == "unknown":
            if text.lower() in response.lower():
                n_echo += 1
            else:
                n_degen += 1
            continue

        n_parseable += 1
        if pred == "yes" and label == "yes":
            tp += 1
        elif pred == "yes" and label == "no":
            fp += 1
        elif pred == "no" and label == "no":
            tn += 1
        elif pred == "no" and label == "yes":
            fn += 1

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    yes_ratio = (tp + fp) / total if total > 0 else 0

    acc_lo, acc_hi = wilson_ci(acc, total)
    prec_lo, prec_hi = wilson_ci(prec, tp + fp) if (tp + fp) > 0 else (0, 0)
    rec_lo, rec_hi = wilson_ci(rec, tp + fn) if (tp + fn) > 0 else (0, 0)

    result = {
        "method": method_name,
        "n_attempted": len(responses),
        "n_parseable": n_parseable,
        "n_echo": n_echo,
        "n_degen": n_degen,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": acc * 100,
        "accuracy_ci": [acc_lo * 100, acc_hi * 100],
        "precision": prec * 100,
        "precision_ci": [prec_lo * 100, prec_hi * 100],
        "recall": rec * 100,
        "recall_ci": [rec_lo * 100, rec_hi * 100],
        "f1": f1 * 100,
        "yes_ratio": yes_ratio * 100,
        "avg_time": total_time / max(1, len(responses)),
        "total_time": total_time,
        "responses": responses,
    }

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {save_path}")

    return result


def print_summary(results):
    print(f"\n{'Method':<15} {'Parse':>8} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Yes%':>7} {'Time':>7}")
    print("-" * 76)
    for r in results:
        parse_str = f"{r['n_parseable']}/{r['n_attempted']}"
        print(f"{r['method']:<15} {parse_str:>8} {r['accuracy']:7.1f} {r['precision']:7.1f} "
              f"{r['recall']:7.1f} {r['f1']:7.1f} {r['yes_ratio']:7.1f} {r['avg_time']:7.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="HuggingFace model ID")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing images (COCO val2014)")
    parser.add_argument("--pope-dir", type=str, required=True,
                        help="Directory containing POPE question files")
    parser.add_argument("--split", type=str, default="coco_random",
                        choices=list(POPE_SPLITS.keys()) + ["all"],
                        help="POPE split to evaluate")
    parser.add_argument("--methods", nargs="+",
                        default=["vanilla", "cvg_7b", "cvg_e_7b"],
                        help="Methods to evaluate (from ABLATIONS)")
    parser.add_argument("--max-questions", type=int, default=3000)
    parser.add_argument("--output-dir", type=str, default="experiments/7b")
    parser.add_argument("--use-flash", action="store_true",
                        help="Use SDPA for non-hooked layers (recommended)")
    args = parser.parse_args()

    splits = list(POPE_SPLITS.keys()) if args.split == "all" else [args.split]

    for method_name in args.methods:
        print(f"\n{'='*60}")
        print(f"Loading model for method: {method_name}")
        print(f"{'='*60}")

        if method_name == "vanilla":
            config = ABLATIONS.get(method_name, ABLATIONS["vanilla"])
        else:
            config = ABLATIONS[method_name]

        use_flash = args.use_flash and not (isinstance(config, CVGEConfig) and config.use_single_pass)

        model = CVGModel(
            model_name=args.model,
            config=config,
            dtype=torch.float16,
            use_flash=use_flash,
        )

        if method_name == "vanilla":
            gen_fn = model.generate_vanilla
        elif isinstance(config, CVGEConfig) and config.use_single_pass:
            gen_fn = model.generate_with_cvg_e
        else:
            gen_fn = model.generate_with_cvg

        all_results = []
        for split in splits:
            print(f"\n--- Split: {split} ---")
            questions = load_pope_questions(args.pope_dir, split)

            save_path = Path(args.output_dir) / f"{method_name}_{split}.json"
            if save_path.exists():
                print(f"  Already exists, skipping: {save_path}")
                with open(save_path) as f:
                    all_results.append(json.load(f))
                continue

            result = run_method(
                model, questions, args.image_dir, f"{method_name}/{split}",
                gen_fn, max_questions=args.max_questions,
                save_path=str(save_path),
            )
            all_results.append(result)

        print_summary(all_results)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()
