"""Download COCO val images needed for POPE from HuggingFace datasets."""

import json
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
POPE_DIR = BASE_DIR / "cvg" / "data" / "POPE_repo" / "output" / "coco"
COCO_DIR = BASE_DIR / "cvg" / "data" / "coco" / "val2014"


def get_pope_image_ids():
    """Get image IDs needed for POPE evaluation."""
    ids = set()
    for split in ["coco_pope_random.json", "coco_pope_popular.json", "coco_pope_adversarial.json"]:
        path = POPE_DIR / split
        if path.exists():
            with open(path) as f:
                for line in f:
                    if line.strip():
                        q = json.loads(line)
                        # Extract numeric ID from filename
                        name = q["image"]  # COCO_val2014_000000XXXXXX.jpg
                        img_id = int(name.split("_")[-1].split(".")[0])
                        ids.add(img_id)
    return ids


def main():
    max_imgs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    COCO_DIR.mkdir(parents=True, exist_ok=True)

    needed = get_pope_image_ids()
    print(f"POPE needs {len(needed)} unique images")

    # Load HF COCO dataset (streaming to avoid downloading everything)
    ds = load_dataset("detection-datasets/coco", split="val", streaming=True)

    saved = 0
    for sample in tqdm(ds, desc="Scanning COCO val"):
        img_id = sample["image_id"]
        if img_id in needed:
            filename = f"COCO_val2014_{img_id:012d}.jpg"
            dest = COCO_DIR / filename
            if not dest.exists():
                sample["image"].save(str(dest))
                saved += 1
                if saved % 5 == 0:
                    print(f"  Saved {saved} images...")
            if saved >= max_imgs:
                break

    total = len(list(COCO_DIR.glob("*.jpg")))
    print(f"\nDone: {saved} new, {total} total images in {COCO_DIR}")


if __name__ == "__main__":
    main()
