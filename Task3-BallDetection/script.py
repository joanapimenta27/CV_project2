"""
Task 3 - Pool Ball Detection
Finetune YOLOv8 and DETR on pool ball datasets with progressive data addition.

Experiments (same test set across all):
  exp1 - 8-Ball Pool (original only)
  exp2 - + Pool Billiard
  exp3 - + Pool Balls Detection
  exp4 - + Pool Ball Detection (all data)

Models:
  YOLO  : YOLOv8n finetuned via ultralytics
  DETR  : facebook/detr-resnet-50 finetuned via transformers

Metrics reported: mAP@50, mAP@50:95, Precision, Recall
Accuracy is reported as detection accuracy: TP / (TP + FP + FN), derived from precision and recall.
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── Config ───────────────────────────────────────────────────────────────────

TASK_DIR      = Path(__file__).resolve().parent
RUNS_DIR      = TASK_DIR / "runs"

SEED          = 42
YOLO_EPOCHS   = 50
DETR_EPOCHS   = 100
BATCH_SIZE    = 8
LR_DETR       = 1e-5
WEIGHT_DECAY  = 1e-4
IMG_SIZE_YOLO = 640
IMG_SIZE_DETR = 800
MODELS_DIR    = str(TASK_DIR / "models")
RESULTS_FILE  = str(TASK_DIR / "results.txt")
NUM_CLASSES   = 4   # Cue, Solid, Striped, Black (dataset classes, "Dot" excluded)

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
torch.backends.cudnn.benchmark     = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ─── DETR Dataset ─────────────────────────────────────────────────────────────

class DetectionDataset(Dataset):
    """
    Dataset for DETR training/evaluation.
    Each item returns (pil_image, target) where:
        pil_image : PIL.Image (RGB)  — processor is applied in the collate fn
        target    : dict with keys
            "boxes"       : FloatTensor [N, 4]  (cx, cy, w, h) normalised [0,1]
            "class_labels": LongTensor  [N]     unified class id per box
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        img   = Image.open(s["image_path"]).convert("RGB")
        w_img, h_img = img.size

        boxes_coco = s["boxes"]   # [[x, y, w, h], ...] absolute

        if len(boxes_coco) > 0:
            boxes_arr = np.array(boxes_coco, dtype=np.float32)
            cx = (boxes_arr[:, 0] + boxes_arr[:, 2] / 2) / w_img
            cy = (boxes_arr[:, 1] + boxes_arr[:, 3] / 2) / h_img
            bw = boxes_arr[:, 2] / w_img
            bh = boxes_arr[:, 3] / h_img
            boxes_norm = np.stack([cx, cy, bw, bh], axis=1)
            boxes_norm = np.clip(boxes_norm, 0.0, 1.0)
        else:
            boxes_norm = np.zeros((0, 4), dtype=np.float32)

        labels = torch.tensor(s["labels"], dtype=torch.long)
        boxes  = torch.tensor(boxes_norm, dtype=torch.float32)

        return img, {"class_labels": labels, "boxes": boxes}


def make_detr_collate_fn(processor):
    """
    Returns a collate function that lets the DETR processor pad the batch to
    the largest height/width and build the matching pixel_mask, so the model
    can ignore the padded borders during attention and positional encoding.
    """
    def collate_fn(batch):
        images  = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        # processor pads to the batch max and produces the pixel_mask
        encoding = processor(images=images, return_tensors="pt")

        return encoding["pixel_values"], encoding["pixel_mask"], targets
    return collate_fn


# ─── DETR metrics ─────────────────────────────────────────────────────────────

def box_iou(boxes_a, boxes_b):
    """
    Compute pairwise IoU between two sets of boxes.
    Boxes in [x1, y1, x2, y2] format.
    boxes_a: [N, 4], boxes_b: [M, 4]
    Returns: [N, M] iou matrix
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter   = inter_w * inter_h

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


def cxcywh_to_xyxy(boxes):
    """[cx, cy, w, h] → [x1, y1, x2, y2]"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


@torch.no_grad()
def compute_detr_map(model, loader, iou_thresholds=None):
    """
    Multi-class mAP computation for DETR (averaged over NUM_CLASSES).
    Returns dict: {mAP50, mAP50_95, precision, recall, accuracy}
    """
    if iou_thresholds is None:
        iou_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    model.eval()
    all_det_scores = []  # list of 1D tensors per image
    all_det_boxes  = []  # list of [Q,4] tensors per image (xyxy, normalised)
    all_det_labels = []  # list of 1D tensors per image (predicted class id)
    all_gt_boxes   = []  # list of [M,4] tensors per image (xyxy, normalised)
    all_gt_labels  = []  # list of 1D tensors per image (gt class id)

    for pixel_values, pixel_mask, targets in loader:
        pixel_values = pixel_values.to(DEVICE)
        pixel_mask   = pixel_mask.to(DEVICE)
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        # DETR logits: [B, num_queries, num_classes+1]
        # DETR boxes:  [B, num_queries, 4] (cx, cy, w, h, normalised)
        logits = outputs.logits    # [B, Q, C+1]
        boxes  = outputs.pred_boxes  # [B, Q, 4]

        probs = logits.softmax(-1)[:, :, :-1]   # exclude no-object class
        scores, labels = probs.max(-1)           # [B, Q]

        for i in range(len(targets)):
            all_det_scores.append(scores[i].cpu())
            all_det_boxes.append(cxcywh_to_xyxy(boxes[i]).cpu())
            all_det_labels.append(labels[i].cpu())

            gt_raw = targets[i]["boxes"]   # [M, 4] cx,cy,w,h normalised
            if len(gt_raw) > 0:
                gt = cxcywh_to_xyxy(gt_raw)
            else:
                gt = torch.zeros((0, 4))
            all_gt_boxes.append(gt)
            all_gt_labels.append(targets[i]["class_labels"])

    # mAP = mean over classes of AP, averaged over IoU thresholds
    aps_per_threshold = []
    for iou_thr in iou_thresholds:
        class_aps = []
        for cls in range(NUM_CLASSES):
            ds, db, gb = _filter_by_class(
                all_det_scores, all_det_boxes, all_det_labels,
                all_gt_boxes, all_gt_labels, cls,
            )
            class_aps.append(_compute_ap_at_iou(ds, db, gb, iou_thr))
        aps_per_threshold.append(float(np.mean(class_aps)))

    ap50     = aps_per_threshold[0]
    ap50_95  = float(np.mean(aps_per_threshold))

    # precision / recall at IoU=0.50, averaged over classes
    precs, recs = [], []
    for cls in range(NUM_CLASSES):
        ds, db, gb = _filter_by_class(
            all_det_scores, all_det_boxes, all_det_labels,
            all_gt_boxes, all_gt_labels, cls,
        )
        p, r = _pr_at_iou(ds, db, gb, 0.50)
        precs.append(p)
        recs.append(r)
    precision = float(np.mean(precs))
    recall    = float(np.mean(recs))
    accuracy  = _detection_accuracy(precision, recall)

    return {
        "mAP50": ap50,
        "mAP50_95": ap50_95,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def _filter_by_class(all_scores, all_boxes, all_labels,
                     all_gt_boxes, all_gt_labels, cls):
    """
    Restrict per-image detections and ground truth to a single class `cls`.
    Returns (det_scores, det_boxes, gt_boxes) aligned per image, so the
    single-class AP/PR helpers can be reused directly.
    """
    det_scores, det_boxes, gt_boxes = [], [], []
    for scores, boxes, labels in zip(all_scores, all_boxes, all_labels):
        mask = labels == cls
        det_scores.append(scores[mask])
        det_boxes.append(boxes[mask])
    for gboxes, glabels in zip(all_gt_boxes, all_gt_labels):
        if len(gboxes) > 0:
            gmask = glabels == cls
            gt_boxes.append(gboxes[gmask])
        else:
            gt_boxes.append(torch.zeros((0, 4)))
    return det_scores, det_boxes, gt_boxes


def _compute_ap_at_iou(all_scores, all_boxes, all_gt, iou_thr):
    """Average Precision at a given IoU threshold (single class)."""
    n_gt_total = sum(len(g) for g in all_gt)
    if n_gt_total == 0:
        return 0.0

    # flatten all detections, sort by score descending
    det_list = []
    for img_idx, (scores, boxes) in enumerate(zip(all_scores, all_boxes)):
        for j in range(len(scores)):
            det_list.append((scores[j].item(), img_idx, j))
    det_list.sort(key=lambda x: -x[0])

    tp_arr = np.zeros(len(det_list))
    fp_arr = np.zeros(len(det_list))
    matched = [set() for _ in range(len(all_gt))]

    for rank, (score, img_idx, det_j) in enumerate(det_list):
        gt_boxes  = all_gt[img_idx]
        det_box   = all_boxes[img_idx][det_j].unsqueeze(0)

        if len(gt_boxes) == 0:
            fp_arr[rank] = 1
            continue

        ious = box_iou(det_box, gt_boxes)[0]   # [M]
        best_iou, best_gt = ious.max(0)
        best_iou = best_iou.item()
        best_gt  = best_gt.item()

        if best_iou >= iou_thr and best_gt not in matched[img_idx]:
            tp_arr[rank] = 1
            matched[img_idx].add(best_gt)
        else:
            fp_arr[rank] = 1

    tp_cum = np.cumsum(tp_arr)
    fp_cum = np.cumsum(fp_arr)
    recall_arr    = tp_cum / n_gt_total
    precision_arr = tp_cum / (tp_cum + fp_cum + 1e-9)

    # 11-point interpolation
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at = precision_arr[recall_arr >= thr]
        ap += (prec_at.max() if len(prec_at) > 0 else 0.0)
    return ap / 11.0


def _pr_at_iou(all_scores, all_boxes, all_gt, iou_thr):
    """Best-F1 precision/recall at a given IoU threshold."""
    n_gt_total = sum(len(g) for g in all_gt)
    if n_gt_total == 0:
        return 0.0, 0.0

    det_list = []
    for img_idx, (scores, boxes) in enumerate(zip(all_scores, all_boxes)):
        for j in range(len(scores)):
            det_list.append((scores[j].item(), img_idx, j))
    det_list.sort(key=lambda x: -x[0])

    # no detections for this class → zero precision/recall
    if len(det_list) == 0:
        return 0.0, 0.0

    tp_arr = np.zeros(len(det_list))
    fp_arr = np.zeros(len(det_list))
    matched = [set() for _ in range(len(all_gt))]

    for rank, (score, img_idx, det_j) in enumerate(det_list):
        gt_boxes = all_gt[img_idx]
        det_box  = all_boxes[img_idx][det_j].unsqueeze(0)
        if len(gt_boxes) == 0:
            fp_arr[rank] = 1
            continue
        ious = box_iou(det_box, gt_boxes)[0]
        best_iou, best_gt = ious.max(0)
        if best_iou.item() >= iou_thr and best_gt.item() not in matched[img_idx]:
            tp_arr[rank] = 1
            matched[img_idx].add(best_gt.item())
        else:
            fp_arr[rank] = 1

    tp_cum  = np.cumsum(tp_arr)
    fp_cum  = np.cumsum(fp_arr)
    rec_arr = tp_cum / n_gt_total
    pre_arr = tp_cum / (tp_cum + fp_cum + 1e-9)

    f1_arr = 2 * pre_arr * rec_arr / (pre_arr + rec_arr + 1e-9)
    best   = int(np.argmax(f1_arr))
    return float(pre_arr[best]), float(rec_arr[best])


# ─── DETR training ────────────────────────────────────────────────────────────

def train_detr_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for pixel_values, pixel_mask, targets in loader:
        pixel_values = pixel_values.to(DEVICE)
        pixel_mask   = pixel_mask.to(DEVICE)

        # DETR expects labels as list of dicts with keys
        # "class_labels" (LongTensor) and "boxes" (FloatTensor cx,cy,w,h)
        hf_targets = [
            {
                "class_labels": t["class_labels"].to(DEVICE),
                "boxes":        t["boxes"].to(DEVICE),
            }
            for t in targets
        ]

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=hf_targets)
        loss    = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        total_loss += loss.item() * pixel_values.size(0)

    return total_loss / len(loader.dataset)


def run_detr_experiment(exp_name, train_s, val_s, test_s):
    """Finetune DETR and evaluate on test set. Returns metrics dict."""
    from transformers import AutoImageProcessor, DetrForObjectDetection

    print(f"\n{'='*60}")
    print(f"  DETR  |  {exp_name}")
    print(f"  Train: {len(train_s)}  Val: {len(val_s)}  Test: {len(test_s)}")
    print(f"{'='*60}")

    model_save = os.path.join(MODELS_DIR, f"detr_{exp_name}.pth")

    processor = AutoImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        size={"shortest_edge": IMG_SIZE_DETR, "longest_edge": IMG_SIZE_DETR},
    )

    g = torch.Generator()
    g.manual_seed(SEED)

    collate_fn = make_detr_collate_fn(processor)

    train_ds = DetectionDataset(train_s)
    val_ds   = DetectionDataset(val_s)
    test_ds  = DetectionDataset(test_s)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, collate_fn=collate_fn,
        worker_init_fn=seed_worker, generator=g, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=pin_memory,
    )

    if os.path.exists(model_save):
        print(f"  Found {model_save} — loading for test evaluation.")
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
        ).to(DEVICE)

        print(model.config.num_labels)              # expect 4
        print(model.class_labels_classifier)         # Linear(in=256, out=5)
        print(len(model.config.id2label)) 
        print(model.config.id2label)
        model.load_state_dict(torch.load(model_save, map_location=DEVICE))
        metrics = compute_detr_map(model, test_loader)
        _print_detr_metrics(metrics, exp_name)
        return metrics

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    # freeze backbone, only finetune transformer + heads
    for param in model.model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_DETR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=DETR_EPOCHS
    )

    best_val_map = -1.0
    best_epoch   = 0

    for epoch in range(1, DETR_EPOCHS + 1):
        train_loss = train_detr_epoch(model, train_loader, optimizer)
        val_met    = compute_detr_map(model, val_loader)
        scheduler.step()

        print(
            f"  Epoch {epoch:03d}/{DETR_EPOCHS} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_mAP50: {val_met['mAP50']:.4f} | "
            f"val_mAP50:95: {val_met['mAP50_95']:.4f}"
        )

        if val_met["mAP50"] > best_val_map:
            best_val_map = val_met["mAP50"]
            best_epoch   = epoch
            torch.save(model.state_dict(), model_save)
            print(f"  ✓ saved {model_save} (epoch {epoch})")

    print(f"\n  Best epoch: {best_epoch}  val_mAP50: {best_val_map:.4f}")

    model.load_state_dict(torch.load(model_save, map_location=DEVICE))
    metrics = compute_detr_map(model, test_loader)
    _print_detr_metrics(metrics, exp_name)
    return metrics


def _print_detr_metrics(m, name):
    print(f"\n── DETR Test Results [{name}] ────────────────────────")
    print(f"  mAP@50    : {m['mAP50']:.4f}")
    print(f"  mAP@50:95 : {m['mAP50_95']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  Accuracy  : {m['accuracy']:.4f}")


# ─── YOLO training ────────────────────────────────────────────────────────────

def run_yolo_experiment(exp_name, yaml_path):
    """Finetune YOLOv8n and evaluate on test set. Returns metrics dict."""
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"  YOLO  |  {exp_name}")
    print(f"  YAML  : {yaml_path}")
    print(f"{'='*60}")

    run_dir  = os.path.join(str(RUNS_DIR), "yolo", exp_name)
    weights  = os.path.join(run_dir, "weights", "best.pt")

    if os.path.exists(weights):
        print(f"  Found {weights} — running test eval only.")
        model = YOLO(weights)
    else:
        model = YOLO("yolov8n.pt")   # pretrained nano as base
        model.train(
            data        = yaml_path,
            epochs      = YOLO_EPOCHS,
            imgsz       = IMG_SIZE_YOLO,
            batch       = BATCH_SIZE,
            device      = 0 if torch.cuda.is_available() else "cpu",
            project     = str(RUNS_DIR / "yolo"),
            name        = exp_name,
            exist_ok    = True,
            seed        = SEED,
            patience    = 15,
            optimizer   = "AdamW",
            lr0         = 1e-4,
            lrf         = 0.01,
            weight_decay= WEIGHT_DECAY,
            augment     = True,
            verbose     = True,
        )
        # After training the model object already holds best weights.
        # Reload from file only if ultralytics wrote best.pt (it normally does).
        if os.path.exists(weights):
            model = YOLO(weights)
        # else: use the model as-is (best weights are loaded internally)

    results = model.val(
        data   = yaml_path,
        split  = "test",
        imgsz  = IMG_SIZE_YOLO,
        device = 0 if torch.cuda.is_available() else "cpu",
        verbose= False,
    )

    metrics = {
        "mAP50":     float(results.box.map50),
        "mAP50_95":  float(results.box.map),
        "precision": float(results.box.mp),
        "recall":    float(results.box.mr),
        "accuracy":  _detection_accuracy(float(results.box.mp), float(results.box.mr)),
    }

    print(f"\n── YOLO Test Results [{exp_name}] ─────────────────────")
    print(f"  mAP@50    : {metrics['mAP50']:.4f}")
    print(f"  mAP@50:95 : {metrics['mAP50_95']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")

    return metrics


# ─── Results saving ───────────────────────────────────────────────────────────

def save_results(all_results):
    """
    all_results: list of {exp_name, model, mAP50, mAP50_95, precision, recall, accuracy}
    """
    col = 30
    with open(RESULTS_FILE, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  Task 3 - Pool Ball Detection — Test Results\n")
        f.write("=" * 70 + "\n\n")

        for r in all_results:
            f.write(f"Model      : {r['model']}\n")
            f.write(f"Experiment : {r['exp_name']}\n")
            f.write(f"  mAP@50    : {r['mAP50']:.4f}\n")
            f.write(f"  mAP@50:95 : {r['mAP50_95']:.4f}\n")
            f.write(f"  Precision : {r['precision']:.4f}\n")
            f.write(f"  Recall    : {r['recall']:.4f}\n\n")
            f.write(f"  Accuracy  : {r['accuracy']:.4f}\n\n")

        # summary table
        f.write("-" * 70 + "\n")
        header = f"{'Model':<8} {'Experiment':<35} {'mAP50':>7} {'mAP50:95':>9} {'Prec':>7} {'Rec':>7} {'Acc':>7}"
        f.write(header + "\n")
        f.write("-" * 70 + "\n")
        for r in all_results:
            f.write(
                f"{'YOLO' if r['model']=='yolo' else 'DETR':<8} "
                f"{r['exp_name']:<35} "
                f"{r['mAP50']:>7.4f} "
                f"{r['mAP50_95']:>9.4f} "
                f"{r['precision']:>7.4f} "
                f"{r['recall']:>7.4f} "
                f"{r['accuracy']:>7.4f}\n"
            )
        f.write("-" * 70 + "\n")

    print(f"\n✓ Results saved to {RESULTS_FILE}")


def _detection_accuracy(precision, recall):
    """Detection accuracy proxy: TP / (TP + FP + FN), derived from precision and recall."""
    denom = (1.0 / max(precision, 1e-9)) + (1.0 / max(recall, 1e-9)) - 1.0
    if denom <= 0:
        return 0.0
    return float(1.0 / denom)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # ── Step 1: prepare all datasets ─────────────────────────────────────────
    from dataset import prepare_all_experiments, EXPERIMENTS
    print("\n── Preparing datasets ───────────────────────────────────────")
    exp_data = prepare_all_experiments()

    all_results = []

    # ── Step 2: YOLO experiments ─────────────────────────────────────────────
    # print("\n\n" + "=" * 70)
    # print("  YOLO Experiments")
    # print("=" * 70)
# 
    # for exp in EXPERIMENTS:
        # name = exp["name"]
        # data = exp_data[name]
        # metrics = run_yolo_experiment(name, data["yaml"])
        # all_results.append({
            # "model":     "yolo",
            # "exp_name":  name,
            # **metrics,
        # })
        # save_results(all_results)

    # ── Step 3: DETR experiments ─────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  DETR Experiments")
    print("=" * 70)

    for exp in EXPERIMENTS:
        name = exp["name"]
        data = exp_data[name]
        metrics = run_detr_experiment(
            name,
            data["train"],
            data["val"],
            data["test"],
        )
        all_results.append({
            "model":     "detr",
            "exp_name":  name,
            **metrics,
        })
        save_results(all_results)

    print("\n✓ All experiments complete.")
