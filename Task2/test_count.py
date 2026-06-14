import json
from collections import Counter, defaultdict

BALL_CATEGORY_NAMES = {"Black", "Cue", "Solid", "Striped"}

with open("data/train/_annotations.coco.json") as f:
    coco = json.load(f)

ball_cat_ids = {c["id"] for c in coco["categories"] if c["name"] in BALL_CATEGORY_NAMES}

counts = defaultdict(int)
for ann in coco["annotations"]:
    if ann["category_id"] in ball_cat_ids:
        counts[ann["image_id"]] += 1

dist = Counter(counts.values())
print("Ball count distribution:", sorted(dist.items()))
print("Total images:", len(coco["images"]))
print("Images with 0 balls:", len(coco["images"]) - len(counts))