"""Run POPE evaluation on TinyLLaVA with Vanilla, CVG, and CVG-E methods.

Usage:
    python scripts/run_pope_eval.py --coco-dir /path/to/coco/val2014 [--max-questions 100]
    python scripts/run_pope_eval.py --synthetic --max-questions 50

If --synthetic, creates simple test images instead of loading COCO.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cvg.modules import CVGConfig, CVGEConfig, ABLATIONS
from cvg.model import CVGModel


@dataclass
class POPEResult:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    yes_ratio: float = 0.0
    total: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    avg_time: float = 0.0
    method: str = ""
    responses: list = field(default_factory=list)


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


def create_synthetic_pope(n_questions=50):
    """Create synthetic POPE-style questions with generated images."""
    objects = ["dog", "cat", "car", "person", "tree", "house", "bird",
               "chair", "table", "phone", "book", "cup", "ball", "flower"]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (128, 0, 128), (255, 165, 0)]

    questions = []
    images = {}
    import random
    random.seed(42)

    for i in range(n_questions):
        obj = objects[i % len(objects)]
        color = colors[i % len(colors)]
        img_name = f"synthetic_{i:04d}.jpg"

        # Simple colored rectangle on white background
        img = Image.new("RGB", (336, 336), "white")
        for x in range(84, 252):
            for y in range(84, 252):
                img.putpixel((x, y), color)
        images[img_name] = img

        # Half yes, half no
        if i % 2 == 0:
            questions.append({
                "image": img_name,
                "text": f"Is there a colored shape in the image?",
                "label": "yes",
            })
        else:
            absent = random.choice([o for o in objects if o != "colored shape"])
            questions.append({
                "image": img_name,
                "text": f"Is there a {absent} in the image?",
                "label": "no",
            })

    return questions, images


def run_pope_method(model, questions, images_or_dir, method_name, generate_fn,
                    max_questions=None, **gen_kwargs):
    """Run POPE evaluation with a specific generation method."""
    if max_questions:
        questions = questions[:max_questions]

    result = POPEResult(method=method_name)
    total_time = 0.0
    n_valid = 0

    for q in tqdm(questions, desc=f"POPE [{method_name}]"):
        img_name = q["image"]
        label = q["label"].lower()

        # Load image
        if isinstance(images_or_dir, dict):
            img = images_or_dir.get(img_name)
        else:
            img_path = Path(images_or_dir) / img_name
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")

        if img is None:
            continue

        prompt = f"USER: {q['text']}\nASSISTANT:"

        t0 = time.time()
        try:
            gen_result = generate_fn(img, prompt, max_new_tokens=10, **gen_kwargs)
            response = gen_result.text
        except Exception as e:
            response = f"ERROR: {e}"
        elapsed = time.time() - t0
        total_time += elapsed

        pred = parse_yes_no(response)
        result.responses.append({
            "question": q["text"], "image": img_name,
            "label": label, "pred": pred, "response": response,
            "time": elapsed,
        })

        if pred == "unknown":
            continue

        n_valid += 1
        if pred == "yes" and label == "yes":
            result.tp += 1
        elif pred == "yes" and label == "no":
            result.fp += 1
        elif pred == "no" and label == "no":
            result.tn += 1
        elif pred == "no" and label == "yes":
            result.fn += 1

    result.total = n_valid
    if n_valid > 0:
        result.accuracy = (result.tp + result.tn) / n_valid * 100
        result.precision = result.tp / (result.tp + result.fp) * 100 if (result.tp + result.fp) > 0 else 0
        result.recall = result.tp / (result.tp + result.fn) * 100 if (result.tp + result.fn) > 0 else 0
        if result.precision + result.recall > 0:
            result.f1 = 2 * result.precision * result.recall / (result.precision + result.recall)
        result.yes_ratio = (result.tp + result.fp) / n_valid * 100
        result.avg_time = total_time / n_valid

    return result


def print_result(r):
    print(f"\n{'='*50}")
    print(f"Method: {r.method}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {r.accuracy:.1f}%")
    print(f"  Precision: {r.precision:.1f}%")
    print(f"  Recall:    {r.recall:.1f}%")
    print(f"  F1:        {r.f1:.1f}%")
    print(f"  Yes-ratio: {r.yes_ratio:.1f}%")
    print(f"  TP/FP/TN/FN: {r.tp}/{r.fp}/{r.tn}/{r.fn}")
    print(f"  Valid: {r.total}, Avg time: {r.avg_time:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-dir", type=str, help="Path to COCO val2014 images")
    parser.add_argument("--pope-file", type=str, help="Path to POPE questions JSONL")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--max-questions", type=int, default=50)
    parser.add_argument("--methods", nargs="+",
                        default=["vanilla", "cvg_small", "cvg_e_small"],
                        help="Methods to evaluate")
    parser.add_argument("--output", type=str, default="experiments/pope_results.json")
    args = parser.parse_args()

    if args.synthetic:
        questions, images = create_synthetic_pope(args.max_questions)
        images_or_dir = images
    elif args.pope_file and args.coco_dir:
        with open(args.pope_file) as f:
            questions = [json.loads(line) for line in f if line.strip()]
        images_or_dir = args.coco_dir
    else:
        print("Need either --synthetic or --pope-file + --coco-dir")
        sys.exit(1)

    results = {}

    for method in args.methods:
        config = ABLATIONS[method]
        model = CVGModel(model_name="bczhou/tiny-llava-v1-hf", config=config)

        if method == "vanilla":
            gen_fn = model.generate_vanilla
        elif "cvg_e" in method or (isinstance(config, CVGEConfig) and config.use_single_pass):
            gen_fn = model.generate_with_cvg_e
        else:
            gen_fn = model.generate_with_cvg

        result = run_pope_method(
            model, questions, images_or_dir, method, gen_fn,
            max_questions=args.max_questions,
        )
        print_result(result)
        results[method] = {
            "accuracy": result.accuracy, "precision": result.precision,
            "recall": result.recall, "f1": result.f1,
            "yes_ratio": result.yes_ratio,
            "tp": result.tp, "fp": result.fp, "tn": result.tn, "fn": result.fn,
            "total": result.total, "avg_time": result.avg_time,
            "responses": result.responses,
        }

        del model
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'Method':<20} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Yes%':>6} {'Time':>6}")
    print("-" * 68)
    for m, r in results.items():
        print(f"{m:<20} {r['accuracy']:6.1f} {r['precision']:6.1f} "
              f"{r['recall']:6.1f} {r['f1']:6.1f} {r['yes_ratio']:6.1f} {r['avg_time']:6.2f}")


if __name__ == "__main__":
    main()
