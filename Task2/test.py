"""
Test all trained models on the entire original dataset (all 247 images).
"""

import os
import json
import random
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ─── Config ───────────────────────────────────────────────────────────────────

ANNOTATIONS  = "data/8-Ball Pool.v3i.coco/train/_annotations.coco.json"
IMAGES_DIR   = "data/8-Ball Pool.v3i.coco/train"
MODELS_DIR   = "models"
RESULTS_FILE = "results_full_dataset.txt"

BALL_CATEGORY_NAMES = {"Black", "Cue", "Solid", "Striped"}

BATCH_SIZE   = 16
SEED         = 42

BACKBONES = ["resnet18", "efficientnet_b0", "vgg16", "inception_v3", "mobilenet_v3"]

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = DEVICE.type == "cuda"

# ─── Reproducibility ──────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ─── Dataset ──────────────────────────────────────────────────────────────────

class BallCountDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(s["count"], dtype=torch.float32)


def load_original_dataset():
    with open(ANNOTATIONS, "r") as f:
        coco = json.load(f)

    ball_cat_ids = {
        c["id"] for c in coco["categories"]
        if c["name"] in BALL_CATEGORY_NAMES
    }

    count_per_image = defaultdict(int)
    for ann in coco["annotations"]:
        if ann["category_id"] in ball_cat_ids:
            count_per_image[ann["image_id"]] += 1

    samples = []
    for img_info in coco["images"]:
        img_path = os.path.join(IMAGES_DIR, img_info["file_name"])
        if not os.path.exists(img_path):
            continue
        count = count_per_image.get(img_info["id"], 0)
        samples.append({"image_path": img_path, "count": count})

    return samples

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

# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).squeeze(1)
        loss  = criterion(preds, labels)
        total_loss += loss.item() * len(imgs)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss   = total_loss / len(loader.dataset)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    mae  = np.mean(np.abs(all_preds - all_labels))
    mse  = np.mean((all_preds - all_labels) ** 2)
    rmse = np.sqrt(mse)
    acc  = np.mean(np.round(all_preds).astype(int) == all_labels.astype(int))
    return avg_loss, mae, mse, rmse, acc

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")

    samples = load_original_dataset()
    print(f"Loaded {len(samples)} images from original dataset")

    all_results = []

    for backbone in BACKBONES:
        model_path = os.path.join(MODELS_DIR, f"{backbone}.pth")
        if not os.path.exists(model_path):
            print(f"\n── {backbone}: model not found, skipping")
            continue

        print(f"\n── {backbone} ──────────────────────────────────────")

        img_size = get_img_size(backbone)
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        ds = BallCountDataset(samples, transform)
        g  = torch.Generator(); g.manual_seed(SEED)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, worker_init_fn=seed_worker,
                            generator=g, pin_memory=pin_memory)

        model = build_model(backbone).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        criterion = nn.HuberLoss()

        loss, mae, mse, rmse, acc = evaluate(model, loader, criterion)

        print(f"  Loss : {loss:.4f}")
        print(f"  MAE  : {mae:.3f}")
        print(f"  MSE  : {mse:.3f}")
        print(f"  RMSE : {rmse:.3f}")
        print(f"  Acc  : {acc*100:.1f}%")

        all_results.append({"backbone": backbone, "loss": loss,
                            "mae": mae, "mse": mse, "rmse": rmse, "acc": acc})

    # save results
    with open(RESULTS_FILE, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  Full Original Dataset Evaluation\n")
        f.write(f"  Images: {len(samples)}\n")
        f.write("=" * 60 + "\n\n")
        for r in all_results:
            f.write(f"Backbone : {r['backbone']}\n")
            f.write(f"  Loss   : {r['loss']:.4f}\n")
            f.write(f"  MAE    : {r['mae']:.3f}\n")
            f.write(f"  MSE    : {r['mse']:.3f}\n")
            f.write(f"  RMSE   : {r['rmse']:.3f}\n")
            f.write(f"  Acc    : {r['acc']*100:.1f}%\n\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Backbone':<20} {'MAE':>8} {'MSE':>8} {'RMSE':>8} {'Acc':>8}\n")
        f.write("-" * 60 + "\n")
        for r in all_results:
            f.write(f"{r['backbone']:<20} {r['mae']:>8.3f} {r['mse']:>8.3f} {r['rmse']:>8.3f} {r['acc']*100:>7.1f}%\n")
        f.write("-" * 60 + "\n")

    print(f"\n✓ Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()