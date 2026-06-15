"""
Task 2 - Pool Ball Counter (CNN Regression)
Backbones: ResNet18, EfficientNet-B0, VGG16, Inception_v3
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
MODELS_DIR   = "models"
RESULTS_FILE = "results.txt"

IMG_SIZE     = 224
BATCH_SIZE   = 16
NUM_EPOCHS   = 100
LR           = 1e-4
WEIGHT_DECAY = 1e-4
SEED         = 42

BACKBONES = ["resnet18", "efficientnet_b0", "vgg16", "inception_v3"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pin_memory = DEVICE.type == "cuda"

os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Reproducibility ──────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


def load_splits():
    with open(DATASET_JSON, "r") as f:
        dataset = json.load(f)
    return dataset["train"], dataset["val"], dataset["test"]

# ─── Model ────────────────────────────────────────────────────────────────────

def build_model(backbone):
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 1))

    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 1))

    elif backbone == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 1))


    elif backbone == "inception_v3":
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 1)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 1))

    return model


def get_img_size(backbone):
    return 299 if backbone == "inception_v3" else 224

# ─── Training / Evaluation ────────────────────────────────────────────────────

def train(model, loader, optimizer, criterion, backbone="resnet18"):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        if backbone == "inception_v3":
            outputs = model(imgs)
            preds   = outputs.logits.squeeze(1)
            loss    = criterion(preds, labels) + 0.4 * criterion(outputs.aux_logits.squeeze(1), labels)
        else:
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
    mse  = np.mean((all_preds - all_labels) ** 2)
    rmse = np.sqrt(mse)
    acc  = np.mean(np.round(all_preds).astype(int) == all_labels.astype(int))
    return avg_loss, mae, mse, rmse, acc


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


def run_training(backbone):
    model_path = os.path.join(MODELS_DIR, f"{backbone}.pth")
    print(f"\n{'='*60}")
    print(f"  Backbone: {backbone}")
    print(f"{'='*60}")

    img_size = get_img_size(backbone)
    t_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=360),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    e_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if os.path.exists(model_path):
        print(f"Found {model_path} — skipping training, running test eval.")
        _, _, test_s = load_splits()
        test_ds     = BallCountDataset(test_s, e_transform)
        g = torch.Generator(); g.manual_seed(SEED)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=2, worker_init_fn=seed_worker, generator=g)
        model = build_model(backbone).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        criterion = nn.HuberLoss()
        loss, mae, mse, rmse, acc = evaluate(model, test_loader, criterion)
        return {"backbone": backbone, "loss": loss, "mae": mae, "mse": mse, "rmse": rmse, "acc": acc}

    train_s, val_s, test_s = load_splits()
    print(f"Split → train: {len(train_s)}  val: {len(val_s)}  test: {len(test_s)}")

    train_ds = BallCountDataset(train_s, t_transform)
    val_ds   = BallCountDataset(val_s,   e_transform)
    test_ds  = BallCountDataset(test_s,  e_transform)

    g = torch.Generator(); g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, worker_init_fn=seed_worker, generator=g,
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, worker_init_fn=seed_worker, generator=g,
                            pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, worker_init_fn=seed_worker, generator=g,
                             pin_memory=pin_memory)

    model     = build_model(backbone).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.HuberLoss()

    best_val_loss = float("inf")
    best_epoch    = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_mae, train_mse, train_rmse, train_acc = train(model, train_loader, optimizer, criterion, backbone)
        val_loss, val_mae, val_mse, val_rmse, val_acc           = evaluate(model, val_loader, criterion)

        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        lr_after = optimizer.param_groups[0]["lr"]
        if lr_after < lr_before:
            print(f"  ↓ LR reduced to {lr_after:.2e}")

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"train_loss: {train_loss:.4f} | train_mae: {train_mae:.3f} | train_mse: {train_mse:.3f} | train_rmse: {train_rmse:.3f} | train_acc: {train_acc*100:.1f}% | "
            f"val_loss: {val_loss:.4f} | val_mae: {val_mae:.3f} | val_mse: {val_mse:.3f} | val_rmse: {val_rmse:.3f} | val_acc: {val_acc*100:.1f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ saved {model_path} (epoch {epoch})")

    print(f"\nBest model: epoch {best_epoch} | val_loss: {best_val_loss:.4f}")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    test_loss, test_mae, test_mse, test_rmse, test_acc = evaluate(model, test_loader, criterion)

    print(f"\n── Test Results [{backbone}] ──────────────────────────")
    print(f"  Loss : {test_loss:.4f}")
    print(f"  MAE  : {test_mae:.3f}")
    print(f"  MSE  : {test_mse:.3f}")
    print(f"  RMSE : {test_rmse:.3f}")
    print(f"  Acc  : {test_acc*100:.1f}%")

    return {"backbone": backbone, "loss": test_loss, "mae": test_mae,
            "mse": test_mse, "rmse": test_rmse, "acc": test_acc}


def save_results(all_results):
    with open(RESULTS_FILE, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  Task 2 - Test Results Comparison\n")
        f.write("=" * 60 + "\n\n")
        for r in all_results:
            f.write(f"Backbone : {r['backbone']}\n")
            f.write(f"  Loss   : {r['loss']:.4f}\n")
            f.write(f"  MAE    : {r['mae']:.3f}\n")
            f.write(f"  MSE    : {r['mse']:.3f}\n")
            f.write(f"  RMSE   : {r['rmse']:.3f}\n")
            f.write(f"  Acc    : {r['acc']*100:.1f}%\n\n")
        # summary table
        f.write("-" * 60 + "\n")
        f.write(f"{'Backbone':<20} {'MAE':>8} {'MSE':>8} {'RMSE':>8} {'Acc':>8}\n")
        f.write("-" * 60 + "\n")
        for r in all_results:
            f.write(f"{r['backbone']:<20} {r['mae']:>8.3f} {r['mse']:>8.3f} {r['rmse']:>8.3f} {r['acc']*100:>7.1f}%\n")
        f.write("-" * 60 + "\n")
    print(f"\n✓ Results saved to {RESULTS_FILE}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    all_results = []
    for backbone in BACKBONES:
        result = run_training(backbone)
        all_results.append(result)
        save_results(all_results)