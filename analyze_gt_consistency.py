"""Analyze Ground Truth (GT) polygon annotation consistency across the dataset.

Measures the GT lane area ratio for each frame and outputs summary statistics,
histogram bins, and outlier counts (e.g. single-lane annotations vs. both-lane annotations).
"""
import argparse
import json
from pathlib import Path
import numpy as np

from dataset import (
    default_data_root,
    resolve_dataset_dirs,
    list_image_files,
    label_path_for_image,
    load_yolo_seg_polygons,
    polygons_to_mask
)


def analyze_gt_consistency(data_root: str, lane_class_id: int = 0) -> dict:
    images_dir, labels_dir = resolve_dataset_dirs(data_root)
    image_paths = list_image_files(images_dir)

    total_images = len(image_paths)
    if total_images == 0:
        raise ValueError(f"No images found in {data_root}")

    area_ratios = []
    frames_under_35 = []
    frames_under_40 = []
    frame_0364_stats = None

    for p in image_paths:
        lbl = label_path_for_image(p, labels_dir)
        polygons = load_yolo_seg_polygons(lbl, 1280, 720, lane_class_id=lane_class_id)
        mask = polygons_to_mask(polygons, 1280, 720)
        pixel_count = int(np.sum(mask))
        ratio = float(pixel_count / (1280 * 720))
        area_ratios.append(ratio)

        if "frame_0364_00014217" in p.name:
            frame_0364_stats = {
                "filename": p.name,
                "gt_pixels": pixel_count,
                "gt_area_ratio": round(ratio, 4),
                "num_polygons": len(polygons)
            }

        if ratio < 0.35:
            frames_under_35.append(p.name)
        if ratio < 0.40:
            frames_under_40.append(p.name)

    ratios = np.array(area_ratios)

    bin_edges = [0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
    hist, _ = np.histogram(ratios, bins=bin_edges)
    histogram = {
        f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})": int(hist[i])
        for i in range(len(hist))
    }

    stats = {
        "total_frames": total_images,
        "mean_gt_area_ratio": round(float(np.mean(ratios)), 4),
        "median_gt_area_ratio": round(float(np.median(ratios)), 4),
        "std_gt_area_ratio": round(float(np.std(ratios)), 4),
        "min_gt_area_ratio": round(float(np.min(ratios)), 4),
        "max_gt_area_ratio": round(float(np.max(ratios)), 4),
        "frames_with_gt_ratio_under_0_35": len(frames_under_35),
        "frames_with_gt_ratio_under_0_35_pct": round(len(frames_under_35) / total_images * 100, 2),
        "frames_with_gt_ratio_under_0_40": len(frames_under_40),
        "frames_with_gt_ratio_under_0_40_pct": round(len(frames_under_40) / total_images * 100, 2),
        "frames_with_gt_ratio_gte_0_40": total_images - len(frames_under_40),
        "frames_with_gt_ratio_gte_0_40_pct": round((total_images - len(frames_under_40)) / total_images * 100, 2),
        "histogram_distribution": histogram,
        "frame_0364_details": frame_0364_stats
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze GT lane polygon area consistency.")
    parser.add_argument("--data-root", default=default_data_root(), help="Path to dataset root")
    parser.add_argument("--lane-class-id", type=int, default=0, help="YOLO polygon class ID for lane")
    parser.add_argument("--output-json", default="results/gt_area_stats.json", help="Output JSON path")
    args = parser.parse_args()

    stats = analyze_gt_consistency(args.data_root, lane_class_id=args.lane_class_id)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved GT consistency statistics to {out_path}")
    print(f"Total frames: {stats['total_frames']}")
    print(f"Mean GT area ratio: {stats['mean_gt_area_ratio'] * 100:.2f}%")
    print(f"Frames >= 40% (both lanes annotated): {stats['frames_with_gt_ratio_gte_0_40']} ({stats['frames_with_gt_ratio_gte_0_40_pct']}%)")
    print(f"Frames < 40% (partial/single lane annotated): {stats['frames_with_gt_ratio_under_0_40']} ({stats['frames_with_gt_ratio_under_0_40_pct']}%)")
    if stats["frame_0364_details"]:
        print(f"frame_0364 GT area ratio: {stats['frame_0364_details']['gt_area_ratio'] * 100:.2f}%")


if __name__ == "__main__":
    main()
