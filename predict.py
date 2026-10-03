"""Run inference on the held-out test split using trained custom UNet.

Specs:
- Loads model checkpoint (.pt) and config (img_w, img_h, base_ch).
- Evaluates on the 35% TEST split (from splits/test.txt).
- Resizes input to 64x36 -> inference -> sigmoid -> upscales to native 1280x720.
- Saves binary masks (0/255 PNG) to result/<run-name>/ (default result/).
- Optional --save-overlay saves blended visual overlay on the original frame.
"""
import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from dataset import default_data_root, list_image_files, resolve_dataset_dirs
from model import UNet


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[UNet, dict]:
    """Load model architecture and weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {"img_w": 64, "img_h": 36, "base_ch": 16})
    base_ch = config.get("base_ch", 16)

    model = UNet(in_channels=3, num_classes=1, base_ch=base_ch)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def predict_single_image(
    model: UNet,
    image_bgr: np.ndarray,
    img_w: int,
    img_h: int,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess image to (img_w, img_h), run model, and resize mask back to native size."""
    orig_h, orig_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(image_rgb, (img_w, img_h), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    logits = model(tensor)
    prob_small = torch.sigmoid(logits)[0, 0].cpu().numpy()

    prob_full = cv2.resize(prob_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    binary_mask = (prob_full > threshold).astype(np.uint8) * 255

    return binary_mask, prob_full


def create_overlay(
    image_bgr: np.ndarray,
    binary_mask: np.ndarray,
    color_bgr: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4
) -> np.ndarray:
    """Create blended overlay with semi-transparent color over lane pixels."""
    overlay = image_bgr.copy()
    lane_idx = binary_mask > 0
    if np.any(lane_idx):
        colored_mask = np.zeros_like(image_bgr)
        colored_mask[lane_idx] = color_bgr
        blended = cv2.addWeighted(image_bgr, 1.0 - alpha, colored_mask, alpha, 0)
        overlay[lane_idx] = blended[lane_idx]
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Run lane segmentation inference on test split.")
    parser.add_argument("--checkpoint", default="checkpoints/unet_lane/best.pt", help="Path to model checkpoint")
    parser.add_argument("--data-root", default=default_data_root(), help="Path to dataset root")
    parser.add_argument("--splits-dir", default="splits", help="Directory containing test.txt")
    parser.add_argument("--split", default="test", help="Which split to evaluate: test or train")
    parser.add_argument("--run-name", default="", help="Subdirectory under result/ (default: direct in result/)")
    parser.add_argument("--output-dir", default="result", help="Base output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification probability threshold")
    parser.add_argument("--save-overlay", action="store_true", help="Also save blended visual overlays")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, config = load_model_from_checkpoint(ckpt_path, device)
    img_w = config.get("img_w", 64)
    img_h = config.get("img_h", 36)
    print(f"Loaded checkpoint from {ckpt_path} (Model input: {img_w}x{img_h}, base_ch: {config.get('base_ch', 16)})")

    out_dir = Path(args.output_dir)
    if args.run_name:
        out_dir = out_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay_dir = out_dir / "overlays" if args.save_overlay else None
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    images_dir, _ = resolve_dataset_dirs(args.data_root)
    split_file = Path(args.splits_dir) / f"{args.split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}. Run train.py first to generate splits.")

    with open(split_file, "r") as f:
        target_names = [line.strip() for line in f if line.strip()]

    target_paths = []
    all_available = {p.name: p for p in list_image_files(images_dir)}
    for name in target_names:
        if name in all_available:
            target_paths.append(all_available[name])
        elif (images_dir / name).exists():
            target_paths.append(images_dir / name)

    print(f"Running inference on {len(target_paths)} test frames from {split_file}...")

    for img_path in tqdm(target_paths, desc="Predicting"):
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue

        pred_mask, _ = predict_single_image(model, image_bgr, img_w, img_h, device, args.threshold)

        mask_out_path = out_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(mask_out_path), pred_mask)

        if args.save_overlay:
            overlay = create_overlay(image_bgr, pred_mask, color_bgr=(0, 220, 0), alpha=0.45)
            cv2.imwrite(str(overlay_dir / f"{img_path.stem}_overlay.jpg"), overlay)

    print(f"\nInference finished! Saved {len(target_paths)} masks to {out_dir}")
    if args.save_overlay:
        print(f"Overlays saved to {overlay_dir}")


if __name__ == "__main__":
    main()
