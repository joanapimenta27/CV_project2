"""
GradCAM Visualization for Task 2 — Pool Ball Counter
Generates GradCAM heatmaps for VGG16 and Inception_v3 on test set images.
Output: report/figures/gradcam/{backbone}_{img_idx}.pdf
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
NUM_SAMPLES  = 10  # number of test images to visualize

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
    """Return the target layer for GradCAM."""
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


# ─── Custom target for regression (GradCAM expects a target) ─────────────────

class RegressionTarget:
    """GradCAM target for regression models — use the single output."""
    def __call__(self, model_output):
        return model_output.squeeze()


# ─── Main ─────────────────────────────────────────────────────────────────────

def load_test_samples():
    with open(DATASET_JSON, "r") as f:
        dataset = json.load(f)
    return dataset["test"]


def select_diverse_samples(samples, n=NUM_SAMPLES):
    """Select a diverse set of samples covering different ball counts."""
    # group by count
    by_count = {}
    for s in samples:
        c = s["count"]
        if c not in by_count:
            by_count[c] = []
        by_count[c].append(s)

    # sort by count
    sorted_counts = sorted(by_count.keys())
    selected = []

    # round-robin through counts to get diversity
    idx = 0
    while len(selected) < n and idx < 100:
        for c in sorted_counts:
            if len(selected) >= n:
                break
            if by_count[c]:
                selected.append(by_count[c].pop(0))
        idx += 1

    return selected


def generate_gradcam(backbone):
    """Generate GradCAM visualizations for a given backbone."""
    model_path = os.path.join(MODELS_DIR, f"{backbone}.pth")
    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}, skipping")
        return

    img_size = get_img_size(backbone)

    # transforms
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # load model
    model = build_model(backbone).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    target_layers = get_target_layer(model, backbone)

    # load test samples
    test_samples = load_test_samples()
    selected = select_diverse_samples(test_samples, NUM_SAMPLES)

    print(f"\n── GradCAM: {backbone} ──────────────────────────────")
    print(f"  Selected {len(selected)} test images")

    cam = GradCAM(model=model, target_layers=target_layers)

    for i, sample in enumerate(selected):
        img_path = sample["image_path"]
        gt_count = sample["count"]

        if not os.path.exists(img_path):
            print(f"  Image not found: {img_path}")
            continue

        # load original image for visualization
        orig_img = Image.open(img_path).convert("RGB")
        orig_resized = orig_img.resize((img_size, img_size))
        rgb_img = np.array(orig_resized).astype(np.float32) / 255.0

        # prepare tensor
        input_tensor = eval_transform(orig_img).unsqueeze(0).to(DEVICE)

        # get prediction
        with torch.no_grad():
            pred = model(input_tensor).squeeze().item()
        pred_count = max(0, round(pred))

        # generate GradCAM
        targets = [RegressionTarget()]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # create overlay
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # plot: original | GradCAM overlay
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(orig_resized)
        axes[0].set_title("Original", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(cam_image)
        axes[1].set_title("GradCAM Overlay", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        fig.suptitle(
            f"GT: {gt_count} balls  |  Pred: {pred_count} balls  (raw: {pred:.2f})",
            fontsize=12, fontweight="bold", y=0.02
        )
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        out_path = os.path.join(OUTPUT_DIR, f"{backbone}_{i:02d}.pdf")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

        # also save as PNG for easy viewing
        out_png = os.path.join(OUTPUT_DIR, f"{backbone}_{i:02d}.png")
        fig2, axes2 = plt.subplots(1, 2, figsize=(8, 4))
        axes2[0].imshow(orig_resized)
        axes2[0].set_title("Original", fontsize=11, fontweight="bold")
        axes2[0].axis("off")
        axes2[1].imshow(cam_image)
        axes2[1].set_title("GradCAM Overlay", fontsize=11, fontweight="bold")
        axes2[1].axis("off")
        fig2.suptitle(
            f"GT: {gt_count} balls  |  Pred: {pred_count} balls  (raw: {pred:.2f})",
            fontsize=12, fontweight="bold", y=0.02
        )
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        fig2.savefig(out_png, bbox_inches="tight", dpi=150)
        plt.close(fig2)

        status = "✓" if pred_count == gt_count else "✗"
        print(f"  [{status}] img {i:02d}: GT={gt_count}, Pred={pred_count} (raw={pred:.2f}) → {out_path}")

    cam.__del__()  # cleanup hooks


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    for backbone in BACKBONES_TO_VIS:
        generate_gradcam(backbone)
    print(f"\n✓ GradCAM figures saved to {OUTPUT_DIR}/")
