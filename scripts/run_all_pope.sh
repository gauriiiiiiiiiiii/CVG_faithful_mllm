#!/bin/bash
# Run all POPE evaluations sequentially to avoid memory thrashing
set -e
cd "/Users/gauri/Code/Gauri/Research Paper"
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

POPE="cvg/data/pope_filtered.jsonl"
COCO="cvg/data/coco/val2014"
N=60

echo "=== Vanilla (60 questions) ==="
python -I scripts/run_pope_eval.py --pope-file $POPE --coco-dir $COCO --max-questions $N --methods vanilla --output experiments/pope_coco_vanilla_60.json

echo ""
echo "=== CVG dual-pass small (60 questions) ==="
python -I scripts/run_pope_eval.py --pope-file $POPE --coco-dir $COCO --max-questions $N --methods cvg_small --output experiments/pope_coco_cvg_60.json

echo ""
echo "=== CVG-E single-pass small (60 questions) ==="
python -I scripts/run_pope_eval.py --pope-file $POPE --coco-dir $COCO --max-questions $N --methods cvg_e_small --output experiments/pope_coco_cvge_60.json

echo ""
echo "=== ALL DONE ==="
