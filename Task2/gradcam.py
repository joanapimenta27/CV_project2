"""
GradCAM Visualization for Task 2 — Pool Ball Counter
Generates GradCAM heatmaps for VGG16 and Inception_v3 on original test images.

Selects 2 correct + 1 wrong prediction per model, deterministically (SEED=42).
Output: report/figures/gradcam/{backbone}_{img_idx}.png
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
N_CORRECT    = 2
N_WRONG      = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BACKBONES_TO_VIS = ["inception_v3"]

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
        model = models.inception_v3(weights=None, transform_input=True)
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


# ─── Sample selection ─────────────────────────────────────────────────────────

def load_test_samples():
    with open(DATASET_JSON, "r") as f:
        return json.load(f)["test"]


def run_inference_on_pool(samples, model, eval_transform, batch_size=16):
    """
    Batched inference — matches train.py eval behaviour.
    Single-image inference shifts BatchNorm stats for models like Inception v3,
    causing correct/wrong counts to diverge from the reported accuracy.
    """
    valid_samples = [s for s in samples if os.path.exists(s["image_path"])]

    class _InferDataset(torch.utils.data.Dataset):
        def __init__(self, samples, transform):
            self.samples   = samples
            self.transform = transform
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            s   = self.samples[idx]
            img = Image.open(s["image_path"]).convert("RGB")
            return self.transform(img), idx

    from torch.utils.data import DataLoader
    loader  = DataLoader(_InferDataset(valid_samples, eval_transform),
                         batch_size=batch_size, shuffle=False, num_workers=0)
    correct, wrong = [], []

    with torch.no_grad():
        for imgs, idxs in loader:
            preds = model(imgs.to(DEVICE)).squeeze(1).cpu().numpy()
            for pred, idx in zip(preds, idxs.numpy()):
                sample     = valid_samples[idx]
                pred_count = max(0, round(float(pred)))
                entry      = {**sample, "pred": float(pred), "pred_count": pred_count}
                (correct if pred_count == sample["count"] else wrong).append(entry)

    return correct, wrong


def pick_diverse(pool, n):
    """Round-robin across ball counts to maximise variety. Deterministic via SEED."""
    by_count = {}
    for s in pool:
        by_count.setdefault(s["count"], []).append(s)
    # shuffle within each bucket using the global seed for determinism
    for v in by_count.values():
        random.shuffle(v)
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
    return selected[:n]


def select_samples(correct, wrong):
    """Pick N_CORRECT correct + N_WRONG wrong; falls back gracefully if short."""
    chosen_correct = pick_diverse(correct, N_CORRECT)
    chosen_wrong   = pick_diverse(wrong,   N_WRONG)

    if len(chosen_correct) < N_CORRECT:
        shortfall = N_CORRECT - len(chosen_correct)
        print(f"  ⚠ Only {len(chosen_correct)} correct — borrowing {shortfall} from wrong")
        chosen_correct += pick_diverse(wrong, shortfall)

    if len(chosen_wrong) < N_WRONG:
        shortfall = N_WRONG - len(chosen_wrong)
        print(f"  ⚠ Only {len(chosen_wrong)} wrong — borrowing {shortfall} from correct")
        chosen_wrong += pick_diverse(correct, shortfall)

    # order: correct first, wrong last
    return chosen_correct[:N_CORRECT] + chosen_wrong[:N_WRONG]


# ─── GradCAM generation ───────────────────────────────────────────────────────

def render_gradcam(cam, model, eval_transform, img_size, samples, backbone):
    for i, sample in enumerate(samples):
        img_path   = sample["image_path"]
        gt_count   = sample["count"]
        pred_count = sample["pred_count"]
        pred_raw   = sample["pred"]

        orig_img     = Image.open(img_path).convert("RGB")
        orig_resized = orig_img.resize((img_size, img_size))
        rgb_img      = np.array(orig_resized).astype(np.float32) / 255.0

        input_tensor  = eval_transform(orig_img).unsqueeze(0).to(DEVICE)
        grayscale_cam = cam(input_tensor=input_tensor, targets=[RegressionTarget()])[0]
        cam_image     = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(orig_resized)
        axes[0].set_title("Original", fontsize=11, fontweight="bold")
        axes[0].axis("off")
        axes[1].imshow(cam_image)
        axes[1].set_title("GradCAM Overlay", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        fig.suptitle(
            f"GT: {gt_count} balls  |  Pred: {pred_count} (raw: {pred_raw:.2f})",
            fontsize=11, fontweight="bold", y=0.02
        )
        plt.tight_layout(rect=[0, 0.06, 1, 1])

        out_png = os.path.join(OUTPUT_DIR, f"{backbone}_{i:02d}.png")
        fig.savefig(out_png, bbox_inches="tight", dpi=150)
        plt.close(fig)

        status = "✓" if pred_count == gt_count else "✗"
        print(f"  [{status}] img {i:02d}: GT={gt_count}, Pred={pred_count} "
              f"(raw={pred_raw:.2f}) → {os.path.basename(out_png)}")


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

    print(f"\n── GradCAM: {backbone} ──────────────────────────────")

    test_pool       = load_test_samples()
    correct, wrong  = run_inference_on_pool(test_pool, model, eval_transform)
    print(f"  Test pool: {len(test_pool)} images → {len(correct)} correct, {len(wrong)} wrong")

    selected = select_samples(correct, wrong)
    print(f"  Selected : {N_CORRECT} correct + {N_WRONG} wrong")

    cam = GradCAM(model=model, target_layers=get_target_layer(model, backbone))
    render_gradcam(cam, model, eval_transform, img_size, selected, backbone)
    cam.__del__()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    for backbone in BACKBONES_TO_VIS:
        generate_gradcam(backbone)
    print(f"\n✓ GradCAM figures saved to {OUTPUT_DIR}/")