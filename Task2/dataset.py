"""
Dataset preparation script for Task 2.
Merges multiple COCO-format pool ball datasets into train/val/test splits.

Split strategy:
- Test : 20% of original dataset only
- Train: 80% of all datasets combined (after removing test from original)
- Val  : 20% of all datasets combined (after removing test from original)

Output: dataset.json with structure:
{
    "train": [{"image_path": "...", "count": N}, ...],
    "val":   [{"image_path": "...", "count": N}, ...],
    "test":  [{"image_path": "...", "count": N}, ...]
}
"""

import os
import json
import random
from collections import defaultdict

# ─── Config ───────────────────────────────────────────────────────────────────

SEED       = 42
VAL_RATIO  = 0.20
TEST_RATIO = 0.20  # only from original dataset

# Each entry: (annotations_json_path, images_dir, set_of_ball_category_names_or_None_for_all)
# None means "count all categories as balls"

BILLIARD_CATS = {
    "Cue_Ball", "Eight", "Five", "Four", "Nine",
    "One", "Seven", "Six", "Three", "Two", "Object_Ball"
}

DATASETS = [
    # ── Original dataset (test split taken from here only) ──────────────────
    {
        "name":            "8-Ball Pool (original)",
        "annotations":     "data/8-Ball Pool/train/_annotations.coco.json",
        "images_dir":      "data/8-Ball Pool/train",
        "is_original":     True,
        "ball_categories": {"Black", "Cue", "Solid", "Striped"},
    },

    # ── Pool Billiard (all 3 splits go into train/val pool) ─────────────────
    {
        "name":            "Pool Billiard - train",
        "annotations":     "data/Pool Billiardv1i/train/_annotations.coco.json",
        "images_dir":      "data/Pool Billiardv1i/train",
        "is_original":     False,
        "ball_categories": BILLIARD_CATS,
    },
    {
        "name":            "Pool Billiard - valid",
        "annotations":     "data/Pool Billiardv1i/valid/_annotations.coco.json",
        "images_dir":      "data/Pool Billiardv1i/valid",
        "is_original":     False,
        "ball_categories": BILLIARD_CATS,
    },
    {
        "name":            "Pool Billiard - test",
        "annotations":     "data/Pool Billiardv1i/test/_annotations.coco.json",
        "images_dir":      "data/Pool Billiardv1i/test",
        "is_original":     False,
        "ball_categories": BILLIARD_CATS,
    },

    # ── Pool Balls Detection (all 3 splits) ──────────────────────────────────
    {
        "name":            "Pool Balls Detection - train",
        "annotations":     "data/Pool Balls Detectionv13-v13/train/_annotations.coco.json",
        "images_dir":      "data/Pool Balls Detectionv13-v13/train",
        "is_original":     False,
        "ball_categories": None,
    },
    {
        "name":            "Pool Balls Detection - valid",
        "annotations":     "data/Pool Balls Detectionv13-v13/valid/_annotations.coco.json",
        "images_dir":      "data/Pool Balls Detectionv13-v13/valid",
        "is_original":     False,
        "ball_categories": None,
    },
    {
        "name":            "Pool Balls Detection - test",
        "annotations":     "data/Pool Balls Detectionv13-v13/test/_annotations.coco.json",
        "images_dir":      "data/Pool Balls Detectionv13-v13/test",
        "is_original":     False,
        "ball_categories": None,
    },

    # ── Pool Ball Detection (all 3 splits) ───────────────────────────────────
    {
        "name":            "Pool Ball Detection - train",
        "annotations":     "data/Pool Ball Detectionv5i/train/_annotations.coco.json",
        "images_dir":      "data/Pool Ball Detectionv5i/train",
        "is_original":     False,
        "ball_categories": None,
    },
    {
        "name":            "Pool Ball Detection - valid",
        "annotations":     "data/Pool Ball Detectionv5i/valid/_annotations.coco.json",
        "images_dir":      "data/Pool Ball Detectionv5i/valid",
        "is_original":     False,
        "ball_categories": None,
    },
    {
        "name":            "Pool Ball Detection - test",
        "annotations":     "data/Pool Ball Detectionv5i/test/_annotations.coco.json",
        "images_dir":      "data/Pool Ball Detectionv5i/test",
        "is_original":     False,
        "ball_categories": None,
    },
]

OUTPUT_JSON = "data/dataset.json"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_samples(annotations_path, images_dir, ball_categories):
    """
    Parse a COCO JSON and return list of {"image_path": ..., "count": N}.
    ball_categories: set of category names to count, or None to count all.
    """
    with open(annotations_path, "r") as f:
        coco = json.load(f)

    # build category id filter
    if ball_categories is not None:
        ball_cat_ids = {
            c["id"] for c in coco["categories"]
            if c["name"] in ball_categories
        }
        matched = {c["name"] for c in coco["categories"] if c["id"] in ball_cat_ids}
        all_names = {c["name"] for c in coco["categories"]}
        print(f"  Categories in file : {sorted(all_names)}")
        print(f"  Matched as balls   : {sorted(matched)}")
    else:
        ball_cat_ids = None
        all_names = {c["name"] for c in coco["categories"]}
        print(f"  Categories in file : {sorted(all_names)}")
        print(f"  Counting all categories as balls")

    # count annotations per image
    count_per_image = defaultdict(int)
    for ann in coco["annotations"]:
        if ball_cat_ids is None or ann["category_id"] in ball_cat_ids:
            count_per_image[ann["image_id"]] += 1

    samples = []
    missing = 0
    for img_info in coco["images"]:
        img_path = os.path.join(images_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            missing += 1
            continue
        count = count_per_image.get(img_info["id"], 0)
        samples.append({"image_path": img_path, "count": count})

    if missing:
        print(f"  ⚠ {missing} images not found on disk (skipped)")

    return samples


def split_off_test(samples, test_ratio, seed):
    """Split samples into (remaining, test)."""
    random.seed(seed)
    shuffled = samples[:]
    random.shuffle(shuffled)
    n_test = int(len(shuffled) * test_ratio)
    return shuffled[n_test:], shuffled[:n_test]


def split_train_val(samples, val_ratio, seed):
    """Split samples into (train, val)."""
    random.seed(seed)
    shuffled = samples[:]
    random.shuffle(shuffled)
    n_val = int(len(shuffled) * val_ratio)
    return shuffled[n_val:], shuffled[:n_val]

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)

    all_non_test = []   # pool for train+val (from all datasets)
    test_samples = []   # only from original dataset

    for ds in DATASETS:
        print(f"\n── {ds['name']} ──────────────────────────────────")
        samples = load_samples(ds["annotations"], ds["images_dir"], ds["ball_categories"])
        print(f"  Loaded {len(samples)} samples")

        if ds["is_original"]:
            remaining, test = split_off_test(samples, TEST_RATIO, SEED)
            test_samples.extend(test)
            all_non_test.extend(remaining)
            print(f"  → {len(test)} held out for test, {len(remaining)} available for train/val")
        else:
            all_non_test.extend(samples)
            print(f"  → {len(samples)} added to train/val pool")

    # split combined pool into 80% train / 20% val
    train_samples, val_samples = split_train_val(all_non_test, VAL_RATIO, SEED)

    print(f"\n── Final Split ───────────────────────────────────────")
    print(f"  Train : {len(train_samples)}")
    print(f"  Val   : {len(val_samples)}")
    print(f"  Test  : {len(test_samples)}  (original dataset only)")
    print(f"  Total : {len(train_samples) + len(val_samples) + len(test_samples)}")

    # count distribution summary
    from collections import Counter
    for split_name, split in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        dist = Counter(s["count"] for s in split)
        print(f"\n  {split_name} count distribution: {dict(sorted(dist.items()))}")

    dataset = {
        "train": train_samples,
        "val":   val_samples,
        "test":  test_samples,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✓ Saved dataset splits to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()