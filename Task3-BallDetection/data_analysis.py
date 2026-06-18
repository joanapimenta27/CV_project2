"""
Dataset domain analysis for Task 3 (Pool Ball Detection).

This script compares dataset domains across:
- image sizes (width, height, aspect ratio)
- color statistics (RGB means, saturation)
- lighting statistics (brightness and contrast proxies)
- bbox/object size statistics (from COCO annotations)

Outputs are written to ./data_analysis by default:
- dataset_summary.csv
- dataset_summary.json
- per_image_metrics.csv
- per_bbox_metrics.csv
- analysis_report.md

Usage:
    python data_analysis.py
    python data_analysis.py --output-dir ./data_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from dataset import DATASETS


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _safe_min(values: List[float]) -> float:
    return float(min(values)) if values else 0.0


def _safe_max(values: List[float]) -> float:
    return float(max(values)) if values else 0.0


def _calc_image_stats(image_path: Path) -> Dict[str, float]:
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)

    # Channel means in [0, 255].
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    r_mean = float(r.mean())
    g_mean = float(g.mean())
    b_mean = float(b.mean())

    # Relative luminance in [0, 255].
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    brightness_mean = float(lum.mean() / 255.0)
    contrast_std = float(lum.std(ddof=0) / 255.0)

    # Mean saturation approximation from RGB (HSV S = delta / max).
    c_max = np.maximum(np.maximum(r, g), b)
    c_min = np.minimum(np.minimum(r, g), b)
    delta = c_max - c_min
    sat = np.zeros_like(c_max, dtype=np.float32)
    non_zero = c_max > 0
    sat[non_zero] = delta[non_zero] / c_max[non_zero]
    saturation_mean = float(sat.mean())

    width, height = img.size
    aspect_ratio = float(width / height) if height > 0 else 0.0

    return {
        "width": float(width),
        "height": float(height),
        "aspect_ratio": aspect_ratio,
        "r_mean": r_mean,
        "g_mean": g_mean,
        "b_mean": b_mean,
        "brightness_mean": brightness_mean,
        "contrast_std": contrast_std,
        "saturation_mean": saturation_mean,
    }


def analyze_dataset(ds: Dict[str, object], base_dir: Path) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    ann_path = base_dir / str(ds["annotations"])
    img_dir = base_dir / str(ds["images_dir"])

    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    if ds.get("ball_cats") is None:
        keep_ids = set(cat_id_to_name.keys())
    else:
        keep_names = set(ds["ball_cats"])
        keep_ids = {cid for cid, name in cat_id_to_name.items() if name in keep_names}

    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        if ann["category_id"] in keep_ids:
            anns_by_img[ann["image_id"]].append(ann)

    widths: List[float] = []
    heights: List[float] = []
    aspects: List[float] = []

    r_means: List[float] = []
    g_means: List[float] = []
    b_means: List[float] = []
    brightness: List[float] = []
    contrast: List[float] = []
    saturation: List[float] = []

    bbox_area_px: List[float] = []
    bbox_area_rel: List[float] = []
    bbox_w_rel: List[float] = []
    bbox_h_rel: List[float] = []

    per_image_rows: List[Dict[str, object]] = []
    per_bbox_rows: List[Dict[str, object]] = []

    missing_images = 0

    for img_meta in coco["images"]:
        image_id = img_meta["id"]
        file_name = img_meta["file_name"]
        image_path = img_dir / file_name

        if not image_path.exists():
            missing_images += 1
            continue

        stats = _calc_image_stats(image_path)

        widths.append(stats["width"])
        heights.append(stats["height"])
        aspects.append(stats["aspect_ratio"])
        r_means.append(stats["r_mean"])
        g_means.append(stats["g_mean"])
        b_means.append(stats["b_mean"])
        brightness.append(stats["brightness_mean"])
        contrast.append(stats["contrast_std"])
        saturation.append(stats["saturation_mean"])

        per_image_rows.append(
            {
                "dataset": ds["name"],
                "image_id": image_id,
                "file_name": file_name,
                "width": int(stats["width"]),
                "height": int(stats["height"]),
                "aspect_ratio": stats["aspect_ratio"],
                "r_mean": stats["r_mean"],
                "g_mean": stats["g_mean"],
                "b_mean": stats["b_mean"],
                "brightness_mean": stats["brightness_mean"],
                "contrast_std": stats["contrast_std"],
                "saturation_mean": stats["saturation_mean"],
                "n_boxes": len(anns_by_img.get(image_id, [])),
            }
        )

        img_w = float(img_meta["width"])
        img_h = float(img_meta["height"])
        img_area = max(1.0, img_w * img_h)

        for ann in anns_by_img.get(image_id, []):
            x, y, w, h = ann["bbox"]
            area = float(max(0.0, w) * max(0.0, h))
            area_rel = area / img_area
            w_rel = float(max(0.0, w) / img_w) if img_w > 0 else 0.0
            h_rel = float(max(0.0, h) / img_h) if img_h > 0 else 0.0

            bbox_area_px.append(area)
            bbox_area_rel.append(area_rel)
            bbox_w_rel.append(w_rel)
            bbox_h_rel.append(h_rel)

            per_bbox_rows.append(
                {
                    "dataset": ds["name"],
                    "image_id": image_id,
                    "category": cat_id_to_name.get(ann["category_id"], "unknown"),
                    "bbox_x": float(x),
                    "bbox_y": float(y),
                    "bbox_w": float(w),
                    "bbox_h": float(h),
                    "bbox_area_px": area,
                    "bbox_area_rel": area_rel,
                    "bbox_w_rel": w_rel,
                    "bbox_h_rel": h_rel,
                }
            )

    width_mean, width_std = _mean_std(widths)
    height_mean, height_std = _mean_std(heights)
    aspect_mean, aspect_std = _mean_std(aspects)
    r_mean, r_std = _mean_std(r_means)
    g_mean, g_std = _mean_std(g_means)
    b_mean, b_std = _mean_std(b_means)
    bright_mean, bright_std = _mean_std(brightness)
    contrast_mean, contrast_std = _mean_std(contrast)
    sat_mean, sat_std = _mean_std(saturation)
    bbox_px_mean, bbox_px_std = _mean_std(bbox_area_px)
    bbox_rel_mean, bbox_rel_std = _mean_std(bbox_area_rel)

    summary = {
        "dataset": ds["name"],
        "annotation_file": str(ann_path),
        "images_dir": str(img_dir),
        "n_images_total_annotation": len(coco["images"]),
        "n_images_analyzed": len(per_image_rows),
        "n_missing_images": missing_images,
        "n_boxes": len(per_bbox_rows),
        "width_mean": width_mean,
        "width_std": width_std,
        "width_min": _safe_min(widths),
        "width_max": _safe_max(widths),
        "height_mean": height_mean,
        "height_std": height_std,
        "height_min": _safe_min(heights),
        "height_max": _safe_max(heights),
        "aspect_ratio_mean": aspect_mean,
        "aspect_ratio_std": aspect_std,
        "r_mean": r_mean,
        "r_std": r_std,
        "g_mean": g_mean,
        "g_std": g_std,
        "b_mean": b_mean,
        "b_std": b_std,
        "brightness_mean": bright_mean,
        "brightness_std": bright_std,
        "contrast_mean": contrast_mean,
        "contrast_std": contrast_std,
        "saturation_mean": sat_mean,
        "saturation_std": sat_std,
        "bbox_area_px_mean": bbox_px_mean,
        "bbox_area_px_std": bbox_px_std,
        "bbox_area_rel_mean": bbox_rel_mean,
        "bbox_area_rel_std": bbox_rel_std,
        "bbox_area_rel_min": _safe_min(bbox_area_rel),
        "bbox_area_rel_max": _safe_max(bbox_area_rel),
    }

    return summary, per_image_rows, per_bbox_rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        with path.open("w", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summaries: List[Dict[str, object]]) -> None:
    if not summaries:
        path.write_text("No datasets analyzed.\n", encoding="utf-8")
        return

    by_brightness = sorted(summaries, key=lambda x: x["brightness_mean"])
    by_contrast = sorted(summaries, key=lambda x: x["contrast_mean"])
    by_bbox = sorted(summaries, key=lambda x: x["bbox_area_rel_mean"])

    lines = []
    lines.append("# Dataset Domain Analysis\n")
    lines.append("## Quick Findings\n")

    lines.append(
        f"- Darkest dataset: {by_brightness[0]['dataset']} "
        f"(brightness_mean={by_brightness[0]['brightness_mean']:.4f})"
    )
    lines.append(
        f"- Brightest dataset: {by_brightness[-1]['dataset']} "
        f"(brightness_mean={by_brightness[-1]['brightness_mean']:.4f})"
    )
    lines.append(
        f"- Lowest contrast: {by_contrast[0]['dataset']} "
        f"(contrast_mean={by_contrast[0]['contrast_mean']:.4f})"
    )
    lines.append(
        f"- Highest contrast: {by_contrast[-1]['dataset']} "
        f"(contrast_mean={by_contrast[-1]['contrast_mean']:.4f})"
    )
    lines.append(
        f"- Smallest avg relative bbox: {by_bbox[0]['dataset']} "
        f"(bbox_area_rel_mean={by_bbox[0]['bbox_area_rel_mean']:.6f})"
    )
    lines.append(
        f"- Largest avg relative bbox: {by_bbox[-1]['dataset']} "
        f"(bbox_area_rel_mean={by_bbox[-1]['bbox_area_rel_mean']:.6f})"
    )

    lines.append("\n## Per-Dataset Summary\n")
    for s in summaries:
        lines.append(f"### {s['dataset']}")
        lines.append(f"- Images analyzed: {s['n_images_analyzed']} (missing: {s['n_missing_images']})")
        lines.append(
            f"- Resolution (W x H): "
            f"{s['width_mean']:.1f} +/- {s['width_std']:.1f} x "
            f"{s['height_mean']:.1f} +/- {s['height_std']:.1f}"
        )
        lines.append(
            f"- Brightness mean +/- std: {s['brightness_mean']:.4f} +/- {s['brightness_std']:.4f}"
        )
        lines.append(
            f"- Contrast mean +/- std: {s['contrast_mean']:.4f} +/- {s['contrast_std']:.4f}"
        )
        lines.append(
            f"- RGB means: R={s['r_mean']:.2f}, G={s['g_mean']:.2f}, B={s['b_mean']:.2f}"
        )
        lines.append(
            f"- Saturation mean +/- std: {s['saturation_mean']:.4f} +/- {s['saturation_std']:.4f}"
        )
        lines.append(
            f"- Avg relative bbox area: {s['bbox_area_rel_mean']:.6f} +/- {s['bbox_area_rel_std']:.6f}"
        )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_plots(
    summaries: List[Dict[str, object]],
    per_image_rows: List[Dict[str, object]],
    per_bbox_rows: List[Dict[str, object]],
    out_dir: Path,
) -> List[Path]:
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    dataset_names = [str(s["dataset"]) for s in summaries]
    x = np.arange(len(dataset_names), dtype=np.float32)

    # 1) Brightness and contrast bars.
    bright = np.asarray([float(s["brightness_mean"]) for s in summaries], dtype=np.float32)
    contrast = np.asarray([float(s["contrast_mean"]) for s in summaries], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.38
    ax.bar(x - width / 2, bright, width, label="brightness_mean")
    ax.bar(x + width / 2, contrast, width, label="contrast_mean")
    ax.set_title("Lighting by Dataset")
    ax.set_ylabel("Normalized value")
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    p = plots_dir / "lighting_comparison.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    generated.append(p)

    # 2) RGB channel means bars.
    r = np.asarray([float(s["r_mean"]) for s in summaries], dtype=np.float32)
    g = np.asarray([float(s["g_mean"]) for s in summaries], dtype=np.float32)
    b = np.asarray([float(s["b_mean"]) for s in summaries], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.25
    ax.bar(x - width, r, width, label="R mean")
    ax.bar(x, g, width, label="G mean")
    ax.bar(x + width, b, width, label="B mean")
    ax.set_title("Color Means by Dataset")
    ax.set_ylabel("Channel mean [0, 255]")
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    p = plots_dir / "rgb_means_comparison.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    generated.append(p)

    # 3) Relative bbox area boxplot.
    bbox_by_dataset: Dict[str, List[float]] = defaultdict(list)
    for row in per_bbox_rows:
        bbox_by_dataset[str(row["dataset"])].append(float(row["bbox_area_rel"]))

    box_data = [bbox_by_dataset[name] for name in dataset_names]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(box_data, tick_labels=dataset_names, showfliers=False)
    ax.set_title("Relative BBox Area Distribution by Dataset")
    ax.set_ylabel("bbox_area_rel")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    p = plots_dir / "bbox_area_rel_boxplot.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    generated.append(p)

    # 4) Resolution profile (mean +/- std) per dataset.
    w_mean = np.asarray([float(s["width_mean"]) for s in summaries], dtype=np.float32)
    w_std = np.asarray([float(s["width_std"]) for s in summaries], dtype=np.float32)
    h_mean = np.asarray([float(s["height_mean"]) for s in summaries], dtype=np.float32)
    h_std = np.asarray([float(s["height_std"]) for s in summaries], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.errorbar(x, w_mean, yerr=w_std, fmt="o-", capsize=4, label="width mean +/- std")
    ax.errorbar(x, h_mean, yerr=h_std, fmt="s-", capsize=4, label="height mean +/- std")
    ax.set_title("Resolution Profile by Dataset")
    ax.set_ylabel("Pixels")
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    p = plots_dir / "resolution_profile.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    generated.append(p)

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dataset domain differences.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data_analysis",
        help="Where analysis files are written.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = (script_dir / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, object]] = []
    all_image_rows: List[Dict[str, object]] = []
    all_bbox_rows: List[Dict[str, object]] = []

    print("Starting dataset analysis...")
    for ds in DATASETS:
        print(f"Analyzing: {ds['name']}")
        summary, image_rows, bbox_rows = analyze_dataset(ds, script_dir)
        summaries.append(summary)
        all_image_rows.extend(image_rows)
        all_bbox_rows.extend(bbox_rows)

    summary_csv = out_dir / "dataset_summary.csv"
    summary_json = out_dir / "dataset_summary.json"
    per_image_csv = out_dir / "per_image_metrics.csv"
    per_bbox_csv = out_dir / "per_bbox_metrics.csv"
    report_md = out_dir / "analysis_report.md"

    write_csv(summary_csv, summaries)
    summary_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    write_csv(per_image_csv, all_image_rows)
    write_csv(per_bbox_csv, all_bbox_rows)
    write_report(report_md, summaries)

    plot_paths: List[Path] = []
    try:
        plot_paths = generate_plots(summaries, all_image_rows, all_bbox_rows, out_dir)
    except Exception as exc:
        print(f"Plot generation skipped due to error: {exc}")

    print("Analysis complete.")
    print(f"- {summary_csv}")
    print(f"- {summary_json}")
    print(f"- {per_image_csv}")
    print(f"- {per_bbox_csv}")
    print(f"- {report_md}")
    if plot_paths:
        for p in plot_paths:
            print(f"- {p}")


if __name__ == "__main__":
    main()
