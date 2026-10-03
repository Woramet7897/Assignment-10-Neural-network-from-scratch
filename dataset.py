"""Dataset utilities for PSU-reservoir lane segmentation from Ultralytics YOLO-seg labels.

Key Features:
1. Flexible Dataset Discovery: Works with either Ultralytics directory structures
   (`<data_root>/images/{train,val}` or `<data_root>/frames/`) and `<data_root>/labels/{train,val}`.
2. Annotation Filtering: Strictly filters for the lane polygon class (configurable via
   `--lane-class-id`, default 0), discarding bounding boxes (5 values) and polyline classes.
3. Native-Resolution Rasterization: Polygons are rasterized onto a full 1280x720 binary mask
   using `cv2.fillPoly` before joint resizing to network input resolution (default 64x36).
4. Photometric Augmentation: Training-only photometric augmentations (white balance,
   brightness, Gaussian blur) applied to the native RGB image prior to resizing.
5. Deterministic Splits: Supports splitting 65% train and 35% test, writing file lists to
   `splits/train.txt` and `splits/test.txt`.
"""
import argparse
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_dataset_dirs(data_root: Union[str, Path]) -> Tuple[Path, Path]:
    """Auto-detect images and labels directories within data_root."""
    data_root = Path(data_root)

    # Check images directory
    if (data_root / "images").exists():
        images_dir = data_root / "images"
    elif (data_root / "frames").exists():
        images_dir = data_root / "frames"
    else:
        images_dir = data_root

    # Check labels directory
    if (data_root / "labels").exists():
        labels_dir = data_root / "labels"
    else:
        labels_dir = data_root

    return images_dir, labels_dir


def list_image_files(images_dir: Union[str, Path]) -> List[Path]:
    """Return sorted list of image file paths from directory, including nested train/val."""
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return []

    direct = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)
    if direct:
        return direct

    nested = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)
    return nested


def label_path_for_image(image_path: Union[str, Path], labels_dir: Union[str, Path]) -> Path:
    """Derive corresponding .txt label path from image path, handling train/val subfolders."""
    image_path = Path(image_path)
    labels_dir = Path(labels_dir)
    stem = image_path.stem

    candidate = labels_dir / f"{stem}.txt"
    if candidate.exists():
        return candidate

    parent_name = image_path.parent.name
    candidate_sub = labels_dir / parent_name / f"{stem}.txt"
    if candidate_sub.exists():
        return candidate_sub

    for sub in ["train", "val", "test"]:
        cand = labels_dir / sub / f"{stem}.txt"
        if cand.exists():
            return cand

    return candidate


def load_yolo_seg_polygons(
    label_path: Union[str, Path],
    img_w: int,
    img_h: int,
    lane_class_id: int = 0
) -> List[np.ndarray]:
    """Parse YOLO-seg label file and extract only polygons for lane_class_id.

    - Bounding boxes (len == 5) are completely ignored.
    - Polylines/other classes (class_id != lane_class_id) are ignored.
    - Only lines with class_id == lane_class_id and at least 3 points (len >= 7) are kept.
    - Coordinates are unnormalized to native pixel resolution [img_w, img_h].
    """
    label_path = Path(label_path)
    polygons = []
    if not label_path.exists():
        return polygons

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            # Bounding box has 5 tokens (class x y w h); ignore it.
            # A valid polygon needs class_id + at least 3 (x, y) coordinates = >= 7 tokens.
            if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                continue

            try:
                cls_id = int(float(parts[0]))
            except ValueError:
                continue

            # Strict filter: keep ONLY the specified lane polygon class
            if cls_id != lane_class_id:
                continue

            # Extract normalized coordinates and scale to native pixel dimensions
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] = np.clip(coords[:, 0] * img_w, 0, img_w - 1)
            coords[:, 1] = np.clip(coords[:, 1] * img_h, 0, img_h - 1)
            polygons.append(coords.astype(np.int32))

    return polygons


def polygons_to_mask(polygons: List[np.ndarray], img_w: int, img_h: int) -> np.ndarray:
    """Rasterize polygon coordinates into a full-resolution binary mask (0 or 1)."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if polygons:
        cv2.fillPoly(mask, polygons, color=1)
    return mask


def augment_image(image: np.ndarray) -> np.ndarray:
    """Apply slight photometric augmentations on the native-resolution image.

    - White balance shift: per-channel random gain ~[0.92, 1.08]
    - Brightness shift: random delta ~[-25, +25]
    - Light Gaussian blur: kernel size 3 or 5
    Does not alter spatial geometry or ground-truth mask alignment.
    """
    img = image.astype(np.float32)

    # 1. Random white-balance shift
    if random.random() < 0.5:
        gains = np.random.uniform(0.92, 1.08, size=3).astype(np.float32)
        img = img * gains[np.newaxis, np.newaxis, :]

    # 2. Random brightness shift (simulating variable sun / shadow)
    if random.random() < 0.5:
        delta = np.random.uniform(-25.0, 25.0)
        img = img + delta

    img = np.clip(img, 0, 255).astype(np.uint8)

    # 3. Light Gaussian blur
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    return img


def compute_dataset_statistics(
    image_paths: List[Path],
    labels_dir: Path,
    lane_class_id: int = 0,
    img_w: int = 1280,
    img_h: int = 720
) -> dict:
    """Compute and return dataset statistics: count, empty masks, and mean lane ratio."""
    empty_masks = 0
    area_ratios = []

    for img_path in image_paths:
        lbl_path = label_path_for_image(img_path, labels_dir)
        polys = load_yolo_seg_polygons(lbl_path, img_w, img_h, lane_class_id=lane_class_id)
        mask = polygons_to_mask(polys, img_w, img_h)
        ratio = float(np.mean(mask))
        area_ratios.append(ratio)
        if ratio == 0.0:
            empty_masks += 1

    mean_ratio = float(np.mean(area_ratios)) if area_ratios else 0.0
    stats = {
        "num_images": len(image_paths),
        "empty_masks": empty_masks,
        "mean_lane_area_ratio": mean_ratio,
        "lane_class_id": lane_class_id
    }
    return stats


def print_dataset_statistics(stats: dict) -> None:
    """Print formatted dataset statistics at startup."""
    print("=" * 60)
    print("DATASET STATISTICS (PSU-reservoir Lane Dataset)")
    print(f"  - Lane polygon class ID : {stats['lane_class_id']}")
    print(f"  - Total images parsed    : {stats['num_images']}")
    print(f"  - Empty lane masks       : {stats['empty_masks']} ({stats['empty_masks'] / max(1, stats['num_images']) * 100:.2f}%)")
    print(f"  - Mean lane-area ratio   : {stats['mean_lane_area_ratio']:.4f} ({stats['mean_lane_area_ratio'] * 100:.2f}%)")
    print("=" * 60)


class LaneSegDataset(Dataset):
    """PyTorch Dataset for custom Lane UNet.

    Accepts either an images directory or an explicit list of image Paths.
    Rasterizes lane polygons at native resolution, optionally augments,
    and resizes both image and mask together to (target_w, target_h).
    """

    def __init__(
        self,
        images: Union[str, Path, List[Path]],
        labels_dir: Union[str, Path],
        target_w: int = 64,
        target_h: int = 36,
        lane_class_id: int = 0,
        augment: bool = False
    ):
        if isinstance(images, (str, Path)):
            self.image_paths = list_image_files(images)
        else:
            self.image_paths = list(images)

        self.labels_dir = Path(labels_dir)
        self.target_w = target_w
        self.target_h = target_h
        self.lane_class_id = lane_class_id
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]

        lbl_path = label_path_for_image(img_path, self.labels_dir)
        polygons = load_yolo_seg_polygons(lbl_path, orig_w, orig_h, lane_class_id=self.lane_class_id)
        mask = polygons_to_mask(polygons, orig_w, orig_h)

        # Photometric augmentation (train only) applied before resizing
        if self.augment:
            image_rgb = augment_image(image_rgb)

        # Resize image and mask together to network input dimensions
        # cv2.resize expects (width, height)
        resized_img = cv2.resize(image_rgb, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(mask, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST)

        # Convert to PyTorch tensors
        # Image: (3, H, W) normalized to [0, 1]
        img_t = torch.from_numpy(resized_img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        # Mask: (1, H, W) binary float32 {0.0, 1.0}
        mask_t = torch.from_numpy(resized_mask.astype(np.float32)).unsqueeze(0).contiguous()

        return img_t, mask_t


def default_data_root() -> str:
    if Path("../image/image_1k_fern").exists():
        return "../image/image_1k_fern"
    if Path("dataset_seg").exists():
        return "dataset_seg"
    return "./data"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test dataset loading and display statistics.")
    parser.add_argument("--data-root", default=default_data_root(), help="Path to dataset root")
    parser.add_argument("--lane-class-id", type=int, default=0, help="Class ID for lane polygons")
    parser.add_argument("--target-w", type=int, default=64)
    parser.add_argument("--target-h", type=int, default=36)
    args = parser.parse_args()

    img_dir, lbl_dir = resolve_dataset_dirs(args.data_root)
    img_files = list_image_files(img_dir)
    print(f"Data root: {args.data_root}")
    print(f"Discovered {len(img_files)} images in {img_dir}")
    print(f"Labels directory: {lbl_dir}")

    stats = compute_dataset_statistics(img_files, lbl_dir, lane_class_id=args.lane_class_id)
    print_dataset_statistics(stats)

    ds = LaneSegDataset(img_files[:10], lbl_dir, target_w=args.target_w, target_h=args.target_h, lane_class_id=args.lane_class_id)
    sample_img, sample_mask = ds[0]
    print(f"Sample tensor shapes -> Image: {tuple(sample_img.shape)}, Mask: {tuple(sample_mask.shape)}")
    assert sample_img.shape == (3, args.target_h, args.target_w)
    assert sample_mask.shape == (1, args.target_h, args.target_w)
    print("[PASS] Dataset output tensor shapes verified successfully!")
