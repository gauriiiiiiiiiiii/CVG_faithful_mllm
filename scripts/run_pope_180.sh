#!/bin/bash
# Run POPE evaluation with all 180 answerable questions (30 images)
# Methods run separately to manage memory on 16GB M1 Pro

set -e
cd "$(dirname "$0")/.."

POPE_FILE="cvg/data/pope/coco_pope_random.json"
COCO_DIR="cvg/data/coco/val2014"

echo "=== Running Vanilla (180q, ~3 min) ==="
python scripts/run_pope_eval.py \
    --pope-file "$POPE_FILE" \
    --coco-dir "$COCO_DIR" \
    --max-questions 3000 \
    --methods vanilla \
    --output experiments/pope_180_vanilla.json

echo "=== Running CVG small (180q, ~15 min) ==="
python scripts/run_pope_eval.py \
    --pope-file "$POPE_FILE" \
    --coco-dir "$COCO_DIR" \
    --max-questions 3000 \
    --methods cvg_small \
    --output experiments/pope_180_cvg.json

echo "=== Running CVG-E small (180q, ~48 min) ==="
python scripts/run_pope_eval.py \
    --pope-file "$POPE_FILE" \
    --coco-dir "$COCO_DIR" \
    --max-questions 3000 \
    --methods cvg_e_small \
    --output experiments/pope_180_cvge.json

echo "=== All done ==="
