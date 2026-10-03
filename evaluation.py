"""Evaluate predicted lane masks against native 1280x720 ground truth via pixel-wise IoU.

Metrics computed:
- Total test images evaluated
- Detection rate (% of images with IoU >= iou_threshold, default 0.6)
- Average IoU of detected (positive) images
- Average IoU of all images (reference)
Outputs saved:
- metrics.json
- test_eval_per_image.csv
"""
import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from dataset import (
    default_data_root,
    label_path_for_image,
    list_image_files,
    load_yolo_seg_polygons,
    polygons_to_mask,
    resolve_dataset_dirs,
)


def compute_pixel_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    """Compute per-image pixel-wise Intersection over Union (IoU)."""
    pred_bin = pred_mask > 0
    gt_bin = gt_mask > 0

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union + eps)


def main():
    parser = argparse.ArgumentParser(description="Evaluate lane segmentation performance.")
    parser.add_argument("--result-dir", default="result", help="Directory containing predicted masks (.png)")
    parser.add_argument("--data-root", default=default_data_root(), help="Path to dataset root")
    parser.add_argument("--splits-dir", default="splits", help="Directory containing test.txt")
    parser.add_argument("--split", default="test", help="Which split to evaluate")
    parser.add_argument("--lane-class-id", type=int, default=0, help="YOLO polygon class ID for lane")
    parser.add_argument("--iou-threshold", type=float, default=0.6, help="IoU threshold for positive detection (Yes/No)")
    parser.add_argument("--output-json", default="metrics.json", help="Path to output metrics JSON file")
    parser.add_argument("--output-csv", default="test_eval_per_image.csv", help="Path to output per-image CSV")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    images_dir, labels_dir = resolve_dataset_dirs(args.data_root)

    split_file = Path(args.splits_dir) / f"{args.split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, "r") as f:
        target_images = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(target_images)} image names from {split_file}")
    print(f"Comparing predicted masks in {result_dir} against GT in {labels_dir} (IoU threshold: {args.iou_threshold})")

    all_available = {p.name: p for p in list_image_files(images_dir)}
    per_image_results = []
    missing_preds = 0

    for img_name in target_images:
        img_path = all_available.get(img_name, images_dir / img_name)
        stem = Path(img_name).stem
        pred_path = result_dir / f"{stem}.png"

        if not pred_path.exists():
            missing_preds += 1
            continue

        if img_path.exists():
            orig_img = cv2.imread(str(img_path))
            h, w = orig_img.shape[:2]
        else:
            w, h = 1280, 720

        lbl_path = label_path_for_image(img_path, labels_dir)
        polys = load_yolo_seg_polygons(lbl_path, w, h, lane_class_id=args.lane_class_id)
        gt_mask = polygons_to_mask(polys, w, h)

        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if pred_mask is None:
            missing_preds += 1
            continue

        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        iou = compute_pixel_iou(pred_mask, gt_mask)
        is_detected = iou >= args.iou_threshold
        status = "Yes" if is_detected else "No"

        per_image_results.append({
            "image": img_name,
            "iou": round(iou, 5),
            "detected": status,
            "is_positive": is_detected
        })

    if not per_image_results:
        print("Error: No valid predictions evaluated!")
        return

    n_eval = len(per_image_results)
    positives = [r for r in per_image_results if r["is_positive"]]
    n_positives = len(positives)
    detection_rate = (n_positives / n_eval) * 100.0

    avg_iou_positives = float(np.mean([r["iou"] for r in positives])) if positives else 0.0
    avg_iou_all = float(np.mean([r["iou"] for r in per_image_results]))

    summary = {
        "num_test_images": n_eval,
        "missing_predictions": missing_preds,
        "iou_threshold": args.iou_threshold,
        "num_detected_positive": n_positives,
        "detection_rate_pct": round(detection_rate, 2),
        "avg_iou_detected_images": round(avg_iou_positives, 4),
        "avg_iou_all_images": round(avg_iou_all, 4)
    }

    # Print Report
    print("\n" + "=" * 65)
    print("           LANE SEGMENTATION EVALUATION REPORT")
    print("=" * 65)
    print(f" Evaluated Test Images         : {n_eval}")
    if missing_preds > 0:
        print(f" Missing Predictions          : {missing_preds}")
    print(f" Positive IoU Threshold        : {args.iou_threshold}")
    print(f" Detected (Positive Images)    : {n_positives} / {n_eval}")
    print(f" Detection Rate                : {detection_rate:.2f}%")
    print(f" Average IoU (Detected Only)   : {avg_iou_positives:.4f}")
    print(f" Average IoU (All Test Images) : {avg_iou_all:.4f}")
    print("=" * 65)

    # Save metrics.json
    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump({"summary": summary, "per_image": per_image_results}, f, indent=2)
    print(f"Detailed metrics saved to: {output_json_path}")

    # Save per-image CSV
    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "iou", "detected"])
        for r in per_image_results:
            writer.writerow([r["image"], f"{r['iou']:.5f}", r["detected"]])
    print(f"Per-image evaluation CSV saved to: {output_csv_path}")


if __name__ == "__main__":
    main()
