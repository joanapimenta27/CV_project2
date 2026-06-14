import json
from collections import Counter

with open("data/train/_annotations.coco.json") as f:
    coco = json.load(f)

from collections import defaultdict
counts = defaultdict(int)
for ann in coco["annotations"]:
    counts[ann["image_id"]] += 1

dist = Counter(counts.values())
print("Ball count distribution:", sorted(dist.items()))
print("Total images:", len(coco["images"]))
print("Images with 0 balls:", len(coco["images"]) - len(counts))