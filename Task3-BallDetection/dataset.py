"""
Dataset preparation for Task 3 - Pool Ball Detection.

All datasets are mapped to a shared 4-class taxonomy taken from the original
8-Ball Pool annotations: Cue, Solid, Striped, Black (the "Dot" class and the
"balls" supercategory are excluded).

Split strategy:
  - Test : 20% of original dataset only (held out forever)
  - Train: 80% of each experiment's data pool
  - Val  : 20% of each experiment's data pool

Experiments (progressive):
  exp1 - 8-Ball Pool (original only)
  exp2 - + Pool Billiard (all splits)
  exp3 - + Pool Balls Detection (all splits)
  exp4 - + Pool Ball Detection (all splits)

Output: one YAML + YOLO-format label files per experiment (for YOLO),
        and raw sample lists returned for DETR.

YOLO label format (per image .txt):
    <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>

COCO bbox format: [x_topleft, y_topleft, width, height] (absolute pixels)
"""

import os
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────

SEED       = 42
VAL_RATIO  = 0.20
TEST_RATIO = 0.20  # from original dataset only

DATA_ROOT  = "data"       # relative to Task3-BallDetection/
YOLO_ROOT  = "yolo_data"  # where YOLO datasets are written

# Unified class taxonomy, derived from the original 8-Ball Pool dataset
# annotations, excluding the "Dot" (table spots) class and the "balls"
# supercategory. These are the classes the detectors must output.
CLASS_NAMES = ["Cue", "Solid", "Striped", "Black"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# ─── Per-dataset category → unified class mapping ─────────────────────────
# Any category name not present in a map is dropped (its boxes are skipped).

# Original 8-Ball Pool: keep the four ball classes, drop "Dot" and "balls".
ORIGINAL_MAP = {
    "Cue":     "Cue",
    "Solid":   "Solid",
    "Striped": "Striped",
    "Black":   "Black",
}

# Pool Billiard: numbered balls → standard 8-ball convention
# (1-7 solids, 8 black, 9-15 stripes). Generic/ambiguous categories
# ("Object_Ball", "Break", "ObjectBall-TargetBall") are dropped.
BILLIARD_MAP = {
    "Cue_Ball": "Cue",
    "One":   "Solid", "Two":  "Solid", "Three": "Solid", "Four": "Solid",
    "Five":  "Solid", "Six":  "Solid", "Seven": "Solid",
    "Eight": "Black",
    "Nine":  "Striped",
}

# Pool Balls Detection v13: ball numbers 0-15 → 8-ball convention
# (0 = cue, 1-7 solids, 8 black, 9-15 stripes).
V13_MAP = {
    "0": "Cue",
    "1": "Solid", "2": "Solid", "3": "Solid", "4": "Solid",
    "5": "Solid", "6": "Solid", "7": "Solid",
    "8": "Black",
    "9":  "Striped", "10": "Striped", "11": "Striped", "12": "Striped",
    "13": "Striped", "14": "Striped", "15": "Striped",
}

# Pool Ball Detection v5: a different (blackball-style) variant whose
# "blue"/"yellow" balls have no solid/striped equivalent, so they are
# dropped; only the unambiguous cue and black balls are mapped.
V5_MAP = {
    "cue_ball": "Cue",
    "black":    "Black",
}

# DATASETS: all raw annotation sources
DATASETS = [
    # ── Original (test split taken from here only) ───────────────────────────
    {
        "name":        "original",
        "annotations": f"{DATA_ROOT}/8-Ball Pool.v3i.coco/train/_annotations.coco.json",
        "images_dir":  f"{DATA_ROOT}/8-Ball Pool.v3i.coco/train",
        "is_original": True,
        "cat_map":     ORIGINAL_MAP,
    },
    # ── Pool Billiard ─────────────────────────────────────────────────────────
    {
        "name":        "billiard_train",
        "annotations": f"{DATA_ROOT}/Pool Billiard.v1i.coco/train/_annotations.coco.json",
        "images_dir":  f"{DATA_ROOT}/Pool Billiard.v1i.coco/train",
        "is_original": False,
        "cat_map":     BILLIARD_MAP,
    },
    {
        "name":        "billiard_valid",
        "annotations": f"{DATA_ROOT}/Pool Billiard.v1i.coco/valid/_annotations.coco.json",
        "images_dir":  f"{DATA_ROOT}/Pool Billiard.v1i.coco/valid",
        "is_original": False,
        "cat_map":     BILLIARD_MAP,
    },
    {
        "name":        "billiard_test",
        "annotations": f"{DATA_ROOT}/Pool Billiard.v1i.coco/test/_annotations.coco.json",
        "images_dir":  f"{DATA_ROOT}/Pool Billiard.v1i.coco/test",
        "is_original": False,
        "cat_map":     BILLIARD_MAP,
    },
    # ── Pool Balls Detection v13 ──────────────────────────────────────────────
    {
        "name":        "balls_det_v13_train",
        "annotations": f"{DATA_ROOT}/Pool Balls Detection.v13-v13.coco/train/_annotations.coco.json",
        "images_dir":  f"{DATA_ROOT}/Pool Balls Detection.v13-v13.coco/train",
        "is_original": False,
        "cat_map":     V13_MAP,
    },
    {
        "name":        "balls_det_v13_valid",
        "annotations": f"{DATA_ROOT}/Pool Balls Detection.v13-v13.coco/valid/_annotations.coco.json",
        "images_dir":  f"{DATA_ROOT}/Pool Balls Detection.v13-v13.coco/valid",
        "is_original": False,
        "cat_map":     V13_MAP,
    },
    {
        "name":        "balls_det_v13_test",
        "annotations": f"{DATA_ROOT}/Pool Balls Detection.v13-v13.coco/test/_annotations.coco.json",
        "images_dir":  f"{DATA_ROOT}/Pool Balls Detection.v13-v13.coco/test",
        "is_original": False,
        "cat_map":     V13_MAP,
    },
    # ── Pool Ball Detection v5 ────────────────────────────────────────────────
    {
        "name":        "ball_det_v5_train",
        "annotations": f"{DATA_ROOT}/Pool Ball Detection.v5i.coco/train/_annotations.coco.json",
        "images_dir":  f"{DATA_ROOT}/Pool Ball Detection.v5i.coco/train",
        "is_original": False,
        "cat_map":     V5_MAP,
    },
    {
        "name":        "ball_det_v5_valid",
        "annotations": f"{DATA_ROOT}/Pool Ball Detection.v5i.coco/valid/_annotations.coco.json",
        "images_dir":  f"{DATA_ROOT}/Pool Ball Detection.v5i.coco/valid",
        "is_original": False,
        "cat_map":     V5_MAP,
    },
    {
        "name":        "ball_det_v5_test",
        "annotations": f"{DATA_ROOT}/Pool Ball Detection.v5i.coco/test/_annotations.coco.json",
        "images_dir":  f"{DATA_ROOT}/Pool Ball Detection.v5i.coco/test",
        "is_original": False,
        "cat_map":     V5_MAP,
    },
]

# ─── Experiments: progressively add datasets ──────────────────────────────────

EXPERIMENTS = [
    {
        "name":     "exp1_original_only",
        "datasets": ["original"],
    },
    {
        "name":     "exp2_plus_billiard",
        "datasets": ["original",
                     "billiard_train", "billiard_valid", "billiard_test"],
    },
    {
        "name":     "exp3_plus_balls_det",
        "datasets": ["original",
                     "billiard_train", "billiard_valid", "billiard_test",
                     "balls_det_v13_train", "balls_det_v13_valid", "balls_det_v13_test"],
    },
    {
        "name":     "exp4_all_data",
        "datasets": ["original",
                     "billiard_train", "billiard_valid", "billiard_test",
                     "balls_det_v13_train", "balls_det_v13_valid", "balls_det_v13_test",
                     "ball_det_v5_train", "ball_det_v5_valid", "ball_det_v5_test"],
    },
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_detection_samples(annotations_path, images_dir, cat_map):
    """
    Parse a COCO JSON and return a list of detection samples.

    Each sample:
        {
            "image_path": str,
            "width":      int,
            "height":     int,
            "boxes":      [[x, y, w, h], ...],  # absolute pixels, COCO format
            "labels":     [class_id, ...],      # unified class id per box
        }

    cat_map: dict mapping COCO category name -> unified class name.
             Categories absent from the map are dropped.
    """
    with open(annotations_path) as f:
        coco = json.load(f)

    # COCO category id -> unified class id (only for mapped categories)
    catid_to_classid = {
        c["id"]: CLASS_TO_ID[cat_map[c["name"]]]
        for c in coco["categories"] if c["name"] in cat_map
    }
    all_names = sorted(c["name"] for c in coco["categories"])
    kept = {c["name"]: cat_map[c["name"]]
            for c in coco["categories"] if c["name"] in cat_map}
    print(f"    Categories  : {all_names}")
    print(f"    Class map   : {kept}")

    boxes_per_image  = defaultdict(list)
    labels_per_image = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        cid = catid_to_classid.get(ann["category_id"])
        if cid is None:
            continue
        boxes_per_image[ann["image_id"]].append(ann["bbox"])
        labels_per_image[ann["image_id"]].append(cid)

    samples = []
    missing = 0
    for img_info in coco["images"]:
        img_path = os.path.join(images_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            missing += 1
            continue
        samples.append({
            "image_path": img_path,
            "width":      img_info["width"],
            "height":     img_info["height"],
            "boxes":      boxes_per_image.get(img_info["id"], []),
            "labels":     labels_per_image.get(img_info["id"], []),
        })

    if missing:
        print(f"    ⚠ {missing} images missing on disk (skipped)")

    return samples


def split_off_test(samples, test_ratio, seed):
    """Return (remaining, test) lists."""
    random.seed(seed)
    s = samples[:]
    random.shuffle(s)
    n_test = int(len(s) * test_ratio)
    return s[n_test:], s[:n_test]


def split_train_val(samples, val_ratio, seed):
    """Return (train, val) lists."""
    random.seed(seed)
    s = samples[:]
    random.shuffle(s)
    n_val = int(len(s) * val_ratio)
    return s[n_val:], s[:n_val]


def build_splits(exp_dataset_names):
    """
    Given a list of dataset names for an experiment, return
    (train_samples, val_samples, test_samples).

    test_samples are fixed across all experiments (from original dataset only).
    """
    ds_by_name = {ds["name"]: ds for ds in DATASETS}

    all_non_test = []
    test_samples = []

    for ds_name in exp_dataset_names:
        ds = ds_by_name[ds_name]
        print(f"  ── {ds['name']} ──")
        samples = load_detection_samples(
            ds["annotations"], ds["images_dir"], ds["cat_map"]
        )
        print(f"    Loaded {len(samples)} images")

        if ds["is_original"]:
            remaining, test = split_off_test(samples, TEST_RATIO, SEED)
            test_samples.extend(test)
            all_non_test.extend(remaining)
            print(f"    → {len(test)} held for test, {len(remaining)} in pool")
        else:
            all_non_test.extend(samples)
            print(f"    → {len(samples)} added to pool")

    train_samples, val_samples = split_train_val(all_non_test, VAL_RATIO, SEED)
    return train_samples, val_samples, test_samples


# ─── YOLO format export ───────────────────────────────────────────────────────

def coco_bbox_to_yolo(bbox, img_w, img_h):
    """
    Convert COCO bbox [x, y, w, h] (absolute) to YOLO
    [cx_norm, cy_norm, w_norm, h_norm] (normalized).
    """
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    wn = w / img_w
    hn = h / img_h
    # clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    wn = max(0.0, min(1.0, wn))
    hn = max(0.0, min(1.0, hn))
    return cx, cy, wn, hn


def write_yolo_split(samples, out_img_dir, out_lbl_dir):
    """
    Symlink images and write YOLO label txt files.
    Returns list of absolute image paths (for YOLO yaml).
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for sample in samples:
        src = os.path.abspath(sample["image_path"])
        fname = os.path.basename(src)
        dst_img = os.path.join(out_img_dir, fname)
        dst_lbl = os.path.join(out_lbl_dir, os.path.splitext(fname)[0] + ".txt")

        # copy image (use hard link if possible, else copy)
        if not os.path.exists(dst_img):
            try:
                os.link(src, dst_img)
            except OSError:
                shutil.copy2(src, dst_img)

        # write label file
        with open(dst_lbl, "w") as f:
            for bbox, cls in zip(sample["boxes"], sample["labels"]):
                cx, cy, wn, hn = coco_bbox_to_yolo(
                    bbox, sample["width"], sample["height"]
                )
                f.write(f"{cls} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}\n")


def build_yolo_dataset(exp_name, train_s, val_s, test_s):
    """
    Create YOLO directory structure for one experiment and return
    path to the generated YAML config file.
    """
    base = os.path.abspath(os.path.join(YOLO_ROOT, exp_name))

    print(f"  Writing YOLO dataset to: {base}")
    for split_name, samples in [("train", train_s), ("val", val_s), ("test", test_s)]:
        write_yolo_split(
            samples,
            os.path.join(base, "images", split_name),
            os.path.join(base, "labels", split_name),
        )

    yaml_path = os.path.join(base, "dataset.yaml")
    yaml_content = (
        f"path: {base}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    return yaml_path


# ─── Main (standalone use) ────────────────────────────────────────────────────

def prepare_all_experiments():
    """
    Build and cache splits + YOLO datasets for all experiments.
    Returns dict: exp_name -> {"train": [...], "val": [...], "test": [...], "yaml": path}
    """
    random.seed(SEED)
    result = {}

    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  Experiment: {exp['name']}")
        print(f"{'='*60}")

        train_s, val_s, test_s = build_splits(exp["datasets"])
        print(f"\n  Split summary:")
        print(f"    Train : {len(train_s)}")
        print(f"    Val   : {len(val_s)}")
        print(f"    Test  : {len(test_s)}")

        yaml_path = build_yolo_dataset(exp["name"], train_s, val_s, test_s)
        print(f"  YOLO yaml : {yaml_path}")

        result[exp["name"]] = {
            "train": train_s,
            "val":   val_s,
            "test":  test_s,
            "yaml":  yaml_path,
        }

    return result


if __name__ == "__main__":
    prepare_all_experiments()
    print("\n✓ All YOLO datasets prepared.")
