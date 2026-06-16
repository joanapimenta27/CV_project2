"""
Task 2 - Inference Script
Reads a JSON file containing a dictionary of image paths, outputs predicted ball counts per image.

Input JSON format:
    {
        "image_path": [
            "path/to/image1.jpg",
            "path/to/image2.jpg",
            ...
        ]
    }

Output JSON format:
    [
        {"image": "path/to/image1.jpg", "ball_count": 10},
        {"image": "path/to/image2.jpg", "ball_count": 7},
        ...
    ]

Usage:
    python inference.py --input input.json --output output.json
    python inference.py --input input.json  # outputs to predictions.json
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

# ─── Config ───────────────────────────────────────────────────────────────────

BACKBONE   = "vgg16"
MODEL_PATH = "models/vgg16.pth"
IMG_SIZE   = 224
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model ────────────────────────────────────────────────────────────────────

def build_model():
    model = models.vgg16(weights=None)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 1))
    return model


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── Inference ────────────────────────────────────────────────────────────────

def predict(model, image_paths):
    model.eval()
    results = []

    with torch.no_grad():
        # process in batches
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[i:i + BATCH_SIZE]
            imgs = []
            valid_paths = []

            for path in batch_paths:
                if not os.path.exists(path):
                    print(f"  Image not found, skipping: {path}")
                    continue
                try:
                    img = Image.open(path).convert("RGB")
                    imgs.append(transform(img))
                    valid_paths.append(path)
                except Exception as e:
                    print(f"  Error loading {path}: {e}")
                    continue

            if not imgs:
                continue

            batch_tensor = torch.stack(imgs).to(DEVICE)
            preds = model(batch_tensor).squeeze(1)
            counts = torch.round(preds).clamp(min=0).int().cpu().numpy()

            for path, count in zip(valid_paths, counts):
                results.append({
                    "image": path,
                    "ball_count": int(count)
                })

    return results

# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Task 2 - Ball Count Inference")
    parser.add_argument("--input",  default="input.json",              help="Path to input JSON file")
    parser.add_argument("--output", default="output.json", help="Path to output JSON file")
    parser.add_argument("--model",  default=MODEL_PATH,         help="Path to model .pth file")
    args = parser.parse_args()

    # load input
    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)

    with open(args.input, "r") as f:
        data = json.load(f)

    # Check if it's a dictionary and has the 'image_path' key
    if isinstance(data, dict) and "image_path" in data:
        image_paths = data["image_path"]
    else:
        print("Error: input JSON must be a dictionary with an 'image_path' key")
        sys.exit(1)

    # Ensure the extracted value is actually a list
    if not isinstance(image_paths, list):
        print("Error: 'image_path' must contain a list of strings")
        sys.exit(1)

    print(f"Device    : {DEVICE}")
    print(f"Model     : {args.model}")
    print(f"Images    : {len(image_paths)}")

    # load model
    if not os.path.exists(args.model):
        print(f"Error: model not found: {args.model}")
        sys.exit(1)

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    print(f"Loaded model from {args.model}")

    # run inference
    results = predict(model, image_paths)

    # save output
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPredictions saved to {args.output}")
    print(f"  Processed {len(results)}/{len(image_paths)} images")


if __name__ == "__main__":
    main()