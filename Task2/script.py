"""
Task 2 - Pool Ball Counter (CNN Regression)
Backbone: ResNet18 pretrained
Loss: Huber
Splits loaded from dataset.json (produced by prepare_dataset.py)
"""

import os
import json
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ─── Config ───────────────────────────────────────────────────────────────────

DATASET_JSON = "data/dataset.json"
MODEL_PATH   = "model.pth"

IMG_SIZE     = 224
BATCH_SIZE   = 16
NUM_EPOCHS   = 80
LR           = 1e-4
WEIGHT_DECAY = 1e-4
SEED         = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Reproducibility ──────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ─── Dataset ──────────────────────────────────────────────────────────────────

class BallCountDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: list of {"image_path": ..., "count": N} dicts
        """
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


def load_splits():
    with open(DATASET_JSON, "r") as f:
        dataset = json.load(f)
    return dataset["train"], dataset["val"], dataset["test"]

# ─── Model ────────────────────────────────────────────────────────────────────

def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 1)
    )
    return model

# ─── Transforms ───────────────────────────────────────────────────────────────

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=360),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── Training ─────────────────────────────────────────────────────────────────

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        preds = model(imgs).squeeze(1)
        loss  = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss   = total_loss / len(loader.dataset)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    mae  = np.mean(np.abs(all_preds - all_labels))
    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
    acc  = np.mean(np.round(all_preds).astype(int) == all_labels.astype(int))
    return avg_loss, mae, rmse, acc


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
    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
    acc  = np.mean(np.round(all_preds).astype(int) == all_labels.astype(int))
    return avg_loss, mae, rmse, acc


def run_training():
    print(f"Device: {DEVICE}")

    train_s, val_s, test_s = load_splits()
    print(f"Split → train: {len(train_s)}  val: {len(val_s)}  test: {len(test_s)}")

    train_ds = BallCountDataset(train_s, train_transform)
    val_ds   = BallCountDataset(val_s,   eval_transform)
    test_ds  = BallCountDataset(test_s,  eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model     = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.HuberLoss()

    best_val_loss = float("inf")
    best_epoch    = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_mae, train_rmse, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_mae, val_rmse, val_acc         = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"train_loss: {train_loss:.4f} | train_mae: {train_mae:.3f} | train_rmse: {train_rmse:.3f} | train_acc: {train_acc*100:.1f}% | "
            f"val_loss: {val_loss:.4f} | val_mae: {val_mae:.3f} | val_rmse: {val_rmse:.3f} | val_acc: {val_acc*100:.1f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ saved model.pth (epoch {epoch})")

    print(f"\nBest model: epoch {best_epoch} | val_loss: {best_val_loss:.4f}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    test_loss, test_mae, test_rmse, test_acc = evaluate(model, test_loader, criterion)
    print(f"\n── Test Results ──────────────────────────")
    print(f"  Loss : {test_loss:.4f}")
    print(f"  MAE  : {test_mae:.3f}")
    print(f"  RMSE : {test_rmse:.3f}")
    print(f"  Acc  : {test_acc*100:.1f}%  (rounded prediction == true count)")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        print(f"Found {MODEL_PATH} — loading and running eval on test split.")
        _, _, test_s = load_splits()
        test_ds     = BallCountDataset(test_s, eval_transform)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        model       = build_model().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        criterion   = nn.HuberLoss()
        loss, mae, rmse, acc = evaluate(model, test_loader, criterion)
        print(f"Test → loss: {loss:.4f} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | Acc: {acc*100:.1f}%")
    else:
        print("No model.pth found — training from scratch.")
        run_training()