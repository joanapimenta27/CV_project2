"""
GradCAM Visualization for Task 2 — Pool Ball Counter
Generates GradCAM heatmaps for VGG16 and Inception_v3.

Samples 3 images from each source:
  - "original"  → data/dataset.json ["test"]  (8-Ball Pool only)
  - "other"     → data/dataset.json ["train"] (non-original datasets)

Output: report/figures/gradcam/{backbone}_{source}_{img_idx}.png
"""

import os
import json
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ─── Config ───────────────────────────────────────────────────────────────────

DATASET_JSON = "data/dataset.json"
MODELS_DIR   = "models"
OUTPUT_DIR   = "report/figures/gradcam"
SEED         = 42
NUM_SAMPLES  = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BACKBONES_TO_VIS = ["vgg16", "inception_v3"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Model ────────────────────────────────────────────────────────────────────

def build_model(backbone):
    if backbone == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 1))
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 1))
    elif backbone == "vgg16":
        model = models.vgg16(weights=None)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 1))
    elif backbone == "inception_v3":
        model = models.inception_v3(weights=None)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 1)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 1))
    elif backbone == "mobilenet_v3":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 1))
    return model


def get_img_size(backbone):
    return 299 if backbone == "inception_v3" else 224


def get_target_layer(model, backbone):
    if backbone == "vgg16":
        return [model.features[-1]]
    elif backbone == "inception_v3":
        return [model.Mixed_7c]
    elif backbone == "resnet18":
        return [model.layer4[-1]]
    elif backbone == "efficientnet_b0":
        return [model.features[-1]]
    elif backbone == "mobilenet_v3":
        return [model.features[-1]]
    return None


# ─── Custom target for regression ─────────────────────────────────────────────

class RegressionTarget:
    def __call__(self, model_output):
        return model_output.squeeze()


# ─── Dataset loading ──────────────────────────────────────────────────────────

def load_test_samples():
    with open(DATASET_JSON, "r") as f:
        return json.load(f)["test"]


def select_diverse_samples(samples, n):
    """Round-robin across ball counts for variety, tops up randomly if needed."""
    by_count = {}
    for s in samples:
        by_count.setdefault(s["count"], []).append(s)

    sorted_counts = sorted(by_count.keys())
    selected = []
    while len(selected) < n:
        progress = False
        for c in sorted_counts:
            if len(selected) >= n:
                break
            if by_count[c]:
                selected.append(by_count[c].pop(0))
                progress = True
        if not progress:
            break

    remaining = [s for bucket in by_count.values() for s in bucket]
    random.shuffle(remaining)
    for s in remaining:
        if len(selected) >= n:
            break
        selected.append(s)

    return selected[:n]


# ─── GradCAM generation ───────────────────────────────────────────────────────

def run_gradcam_for_source(cam, model, eval_transform, img_size, samples, backbone):
    """Run GradCAM for one set of samples and save figures."""
    for i, sample in enumerate(samples):
        img_path  = sample["image_path"]
        gt_count  = sample["count"]

        if not os.path.exists(img_path):
            print(f"  Image not found: {img_path}")
            continue

        orig_img     = Image.open(img_path).convert("RGB")
        orig_resized = orig_img.resize((img_size, img_size))
        rgb_img      = np.array(orig_resized).astype(np.float32) / 255.0

        input_tensor = eval_transform(orig_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(input_tensor).squeeze().item()
        pred_count = max(0, round(pred))

        targets       = [RegressionTarget()]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(orig_resized)
        axes[0].set_title("Original", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(cam_image)
        axes[1].set_title("GradCAM Overlay", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        fig.suptitle(
            f"GT: {gt_count} balls  |  Pred: {pred_count} (raw: {pred:.2f})",
            fontsize=11, fontweight="bold", y=0.02
        )
        plt.tight_layout(rect=[0, 0.06, 1, 1])

        out_png = os.path.join(OUTPUT_DIR, f"{backbone}_{i:02d}.png")
        fig.savefig(out_png, bbox_inches="tight", dpi=150)
        plt.close(fig)

        status = "✓" if pred_count == gt_count else "✗"
        print(f"  [{status}] img {i:02d}: GT={gt_count}, Pred={pred_count} "
              f"(raw={pred:.2f}) → {os.path.basename(out_png)}")


def generate_gradcam(backbone):
    model_path = os.path.join(MODELS_DIR, f"{backbone}.pth")
    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}, skipping")
        return

    img_size       = get_img_size(backbone)
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = build_model(backbone).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    target_layers = get_target_layer(model, backbone)
    cam           = GradCAM(model=model, target_layers=target_layers)

    test_pool = load_test_samples()
    selected  = select_diverse_samples(list(test_pool), NUM_SAMPLES)

    print(f"\n── GradCAM: {backbone} ──────────────────────────────")
    print(f"  Test pool : {len(test_pool)} images → selected {len(selected)}")

    run_gradcam_for_source(cam, model, eval_transform, img_size,
                           selected, backbone)

    cam.__del__()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    for backbone in BACKBONES_TO_VIS:
        generate_gradcam(backbone)
    print(f"\n✓ GradCAM figures saved to {OUTPUT_DIR}/")