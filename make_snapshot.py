"""Create side-by-side snapshot visual comparisons for the PSU-reservoir lane dataset.

Generates side-by-side figures:
  [Original Frame] | [Ground-Truth Overlay] | [Predicted Overlay]
at native 1280x720 resolution.
Picks representative examples including high-performing cases and edge/failure cases.
Saves individual snapshots and a composite comparison figure to assets/.
"""
import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from dataset import (
    default_data_root,
    label_path_for_image,
    list_image_files,
    load_yolo_seg_polygons,
    polygons_to_mask,
    resolve_dataset_dirs,
)


def blend_overlay(
    image_rgb: np.ndarray,
    binary_mask: np.ndarray,
    color_rgb: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.45
) -> np.ndarray:
    """Create a semi-transparent colored overlay on RGB image."""
    result = image_rgb.copy()
    lane_idx = binary_mask > 0
    if np.any(lane_idx):
        color_layer = np.zeros_like(image_rgb)
        color_layer[lane_idx] = color_rgb
        blended = cv2.addWeighted(image_rgb, 1.0 - alpha, color_layer, alpha, 0)
        result[lane_idx] = blended[lane_idx]
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate side-by-side snapshot comparison figures.")
    parser.add_argument("--result-dir", default="result", help="Directory containing predicted masks (.png)")
    parser.add_argument("--data-root", default=default_data_root(), help="Dataset root")
    parser.add_argument("--eval-csv", default="test_eval_per_image.csv", help="Per-image evaluation CSV")
    parser.add_argument("--output-dir", default="assets", help="Output directory for snapshots")
    parser.add_argument("--lane-class-id", type=int, default=0, help="Lane polygon class ID")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of representative samples")
    args = parser.parse_args()

    images_dir, labels_dir = resolve_dataset_dirs(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_csv_path = Path(args.eval_csv)
    records = []
    if eval_csv_path.exists():
        with open(eval_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "image": row["image"],
                    "iou": float(row["iou"]),
                    "detected": row["detected"]
                })
        records.sort(key=lambda r: r["iou"])
    else:
        pred_files = sorted(Path(args.result_dir).glob("*.png"))
        for p in pred_files:
            records.append({"image": f"{p.stem}.jpg", "iou": 0.0, "detected": "N/A"})

    if not records:
        print("No evaluation records or prediction files found.")
        return

    n_rec = len(records)
    selected_indices = [
        0,              # Minimum IoU (challenge / edge case)
        n_rec // 4,     # Lower quartile
        n_rec // 2,     # Median
        n_rec - 1       # Top performer
    ]
    selected_indices = sorted(list(set(selected_indices)))
    selected_records = [records[i] for i in selected_indices[:args.num_samples]]

    print(f"Creating snapshots for {len(selected_records)} representative test frames...")

    all_available = {p.name: p for p in list_image_files(images_dir)}
    composite_rows = []

    for rank_idx, rec in enumerate(selected_records):
        img_name = rec["image"]
        img_path = all_available.get(img_name, images_dir / img_name)
        stem = Path(img_name).stem
        pred_path = Path(args.result_dir) / f"{stem}.png"

        if not img_path.exists() or not pred_path.exists():
            continue

        image_bgr = cv2.imread(str(img_path))
        orig_h, orig_w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        lbl_path = label_path_for_image(img_path, labels_dir)
        polys = load_yolo_seg_polygons(lbl_path, orig_w, orig_h, lane_class_id=args.lane_class_id)
        gt_mask = polygons_to_mask(polys, orig_w, orig_h)

        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if pred_mask.shape != (orig_h, orig_w):
            pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        gt_overlay = blend_overlay(image_rgb, gt_mask, color_rgb=(0, 220, 220), alpha=0.45)
        pred_overlay = blend_overlay(image_rgb, pred_mask, color_rgb=(30, 220, 50), alpha=0.45)

        composite_rows.append((stem, rec["iou"], rec["detected"], image_rgb, gt_overlay, pred_overlay))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=200)
        axes[0].imshow(image_rgb)
        axes[0].set_title(f"Original Frame: {stem}", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(gt_overlay)
        axes[1].set_title("Ground Truth Polygon Overlay (Cyan)", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(pred_overlay)
        axes[2].set_title(f"Predicted Custom UNet Overlay (IoU: {rec['iou']:.4f} | Detected: {rec['detected']})",
                          fontsize=12, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()
        single_path = out_dir / f"snapshot_{rank_idx+1}_{stem}.png"
        plt.savefig(str(single_path), bbox_inches="tight")
        plt.close()
        print(f"  [+] Saved {single_path}")

    if composite_rows:
        num_rows = len(composite_rows)
        fig, axes = plt.subplots(num_rows, 3, figsize=(18, 4.5 * num_rows), dpi=200)
        if num_rows == 1:
            axes = np.expand_dims(axes, 0)

        for i, (stem, iou_val, det_flag, orig, gt_ov, pred_ov) in enumerate(composite_rows):
            axes[i, 0].imshow(orig)
            axes[i, 0].set_title(f"Frame {stem} (Raw 1280x720)", fontsize=11, fontweight="bold")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(gt_ov)
            axes[i, 1].set_title("Ground Truth Lane Mask Overlay", fontsize=11, fontweight="bold")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pred_ov)
            axes[i, 2].set_title(f"Predicted Custom UNet (IoU = {iou_val:.4f} | Detected: {det_flag})",
                                 fontsize=11, fontweight="bold")
            axes[i, 2].axis("off")

        plt.tight_layout()
        composite_path = out_dir / "snapshot_comparison.png"
        plt.savefig(str(composite_path), bbox_inches="tight")
        plt.close()
        print(f"\n[+] Master comparison snapshot saved to: {composite_path}")


if __name__ == "__main__":
    main()
