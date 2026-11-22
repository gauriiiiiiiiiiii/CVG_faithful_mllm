"""Analyze POPE results and compute confidence intervals."""

import json
import math
import sys
from pathlib import Path


def wilson_ci(p, n, z=1.96):
    """Wilson score interval for proportion p with n observations."""
    if n == 0:
        return 0, 0
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    spread = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
    return max(0, center - spread), min(1, center + spread)


def analyze_method(filepath):
    """Analyze results from a single method run."""
    with open(filepath) as f:
        data = json.load(f)

    # Handle both formats: {method: {responses: [...]}} and direct {responses: [...]}
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, dict) and "responses" in val:
                method = key
                responses = val["responses"]
                stats = val
                break
        else:
            method = "unknown"
            responses = data.get("responses", [])
            stats = data

    # Count parseability
    n_attempted = len(responses)
    preds = [r.get("pred", "unknown") for r in responses]
    n_parseable = sum(1 for p in preds if p in ("yes", "no"))
    n_unknown = sum(1 for p in preds if p == "unknown")

    # Classify failure modes
    n_echo = 0
    n_degen = 0
    for r in responses:
        if r.get("pred") == "unknown":
            resp = r.get("response", "").strip()
            q = r.get("question", "").strip()
            if q and q.lower() in resp.lower():
                n_echo += 1
            elif len(resp) < 5:
                n_degen += 1
            else:
                n_degen += 1

    tp = stats.get("tp", 0)
    fp = stats.get("fp", 0)
    tn = stats.get("tn", 0)
    fn = stats.get("fn", 0)
    total = tp + fp + tn + fn

    # Metrics
    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    yes_ratio = (tp + fp) / total if total > 0 else 0

    # Confidence intervals (Wilson)
    acc_lo, acc_hi = wilson_ci(acc, total)
    prec_lo, prec_hi = wilson_ci(prec, tp + fp) if (tp + fp) > 0 else (0, 0)
    rec_lo, rec_hi = wilson_ci(rec, tp + fn) if (tp + fn) > 0 else (0, 0)

    # Average time
    times = [r.get("time", 0) for r in responses if r.get("pred") != "unknown" or True]
    actual_times = [r.get("time", 0) for r in responses if r.get("time", 0) > 0.5]  # Only actual inference
    avg_time = sum(actual_times) / len(actual_times) if actual_times else 0

    return {
        "method": method,
        "n_attempted": n_attempted,
        "n_parseable": n_parseable,
        "n_unknown": n_unknown,
        "n_echo": n_echo,
        "n_degen": n_degen,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": total,
        "acc": acc * 100,
        "acc_ci": (acc_lo * 100, acc_hi * 100),
        "prec": prec * 100,
        "prec_ci": (prec_lo * 100, prec_hi * 100),
        "rec": rec * 100,
        "rec_ci": (rec_lo * 100, rec_hi * 100),
        "f1": f1 * 100,
        "yes_ratio": yes_ratio * 100,
        "avg_time": avg_time,
    }


def main():
    base = Path("/Users/gauri/Code/Gauri/Research Paper/experiments")

    files = {
        "vanilla": base / "pope_180_vanilla.json",
        "cvg": base / "pope_180_cvg.json",
        "cvge": base / "pope_180_cvge.json",
    }

    print("=" * 80)
    print("POPE Results Analysis (N=180, COCO val2014 random split)")
    print("=" * 80)

    for name, path in files.items():
        if not path.exists():
            print(f"\n{name}: NOT YET AVAILABLE ({path.name})")
            continue

        r = analyze_method(path)

        print(f"\n--- {r['method']} ---")
        print(f"  Parseable: {r['n_parseable']}/{r['n_attempted']}")
        if r['n_unknown'] > 0:
            print(f"  Unparseable: {r['n_unknown']} (echo: {r['n_echo']}, degen: {r['n_degen']})")
        print(f"  TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']} (N={r['total']})")
        print(f"  Accuracy:  {r['acc']:.1f}% (95% CI: {r['acc_ci'][0]:.1f}--{r['acc_ci'][1]:.1f})")
        print(f"  Precision: {r['prec']:.1f}% (95% CI: {r['prec_ci'][0]:.1f}--{r['prec_ci'][1]:.1f})")
        print(f"  Recall:    {r['rec']:.1f}% (95% CI: {r['rec_ci'][0]:.1f}--{r['rec_ci'][1]:.1f})")
        print(f"  F1:        {r['f1']:.1f}%")
        print(f"  Yes-ratio: {r['yes_ratio']:.1f}%")
        print(f"  Avg time:  {r['avg_time']:.2f}s")

    print("\n" + "=" * 80)
    print("LaTeX table row format:")
    print("=" * 80)
    for name, path in files.items():
        if not path.exists():
            continue
        r = analyze_method(path)
        n_str = f"{r['n_parseable']}/{r['n_attempted']}" if r['n_unknown'] > 0 else str(r['total'])
        if r['total'] > 0:
            print(f"{r['method']} & {n_str} & {r['acc']:.1f} & {r['prec']:.1f} & {r['rec']:.1f} & {r['f1']:.1f} & {r['yes_ratio']:.1f} \\\\")


if __name__ == "__main__":
    main()
