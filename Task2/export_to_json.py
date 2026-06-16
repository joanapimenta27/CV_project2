import os
import json

# Use the same config as your test script
ANNOTATIONS = "data/8-Ball Pool.v3i.coco/train/_annotations.coco.json"
IMAGES_DIR  = "data/8-Ball Pool.v3i.coco/train"
OUTPUT_FILE = "input.json"

def export_paths_to_json():
    # Load COCO annotations
    if not os.path.exists(ANNOTATIONS):
        print(f"Error: Annotations file not found at {ANNOTATIONS}")
        return

    with open(ANNOTATIONS, "r") as f:
        coco = json.load(f)

    # Extract all valid image paths
    image_paths = []
    for img_info in coco["images"]:
        img_path = os.path.join(IMAGES_DIR, img_info["file_name"])
        if os.path.exists(img_path):
            image_paths.append(img_path)

    # Wrap the list in a dictionary with the "image_path" key
    output_data = {
        "image_path": image_paths
    }

    # Save to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Successfully exported {len(image_paths)} image paths to {OUTPUT_FILE}")

if __name__ == "__main__":
    export_paths_to_json()