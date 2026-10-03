"""Train custom UNet from scratch on PSU-reservoir lane dataset.

Specs followed:
- 65% Train : 35% Test deterministic split saved to splits/train.txt & splits/test.txt.
- 10% validation carved out of train split for checkpoint selection. Test set is NEVER touched.
- 64x36 RGB input (16:9), binary mask output.
- Training-only photometric augmentation on native 1280x720 frames before resize.
- BCE + Dice Loss, Adam optimizer, TensorBoard logging, CSV history, loss curve plot.
- Checkpoints saved to checkpoints/<run-name>/{last,best}.pt with model_state_dict and config.
"""
import argparse
import csv
import os
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import (
    LaneSegDataset,
    compute_dataset_statistics,
    list_image_files,
    print_dataset_statistics,
    resolve_dataset_dirs,
)
from model import UNet


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice Loss for binary segmentation."""
    probs = torch.sigmoid(logits)
    probs_flat = probs.contiguous().view(probs.size(0), -1)
    targets_flat = targets.contiguous().view(targets.size(0), -1)

    intersection = (probs_flat * targets_flat).sum(dim=1)
    cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combined BCE with Logits and Soft Dice loss."""

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        dice = dice_loss(logits, targets)
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice


@torch.no_grad()
def compute_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """Compute mean Intersection-over-Union (IoU) across batch."""
    preds = (torch.sigmoid(logits) > threshold).float()
    preds_flat = preds.contiguous().view(preds.size(0), -1)
    targets_flat = targets.contiguous().view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return float(iou.mean().item())


def prepare_splits(
    all_images: List[Path],
    splits_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.65,
    val_ratio: float = 0.10
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Create or load deterministic 65:35 split and carve 10% validation from train."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_split_file = splits_dir / "train.txt"
    test_split_file = splits_dir / "test.txt"

    # If splits already exist, load from files
    if train_split_file.exists() and test_split_file.exists():
        print(f"Loading existing splits from {splits_dir}")
        with open(train_split_file, "r") as f:
            train_names = {line.strip() for line in f if line.strip()}
        with open(test_split_file, "r") as f:
            test_names = {line.strip() for line in f if line.strip()}

        train_full = [p for p in all_images if p.name in train_names]
        test_imgs = [p for p in all_images if p.name in test_names]
    else:
        print(f"Generating deterministic 65:35 split with seed={seed}...")
        rng = random.Random(seed)
        shuffled = sorted(all_images)
        rng.shuffle(shuffled)

        n_train = int(round(len(shuffled) * train_ratio))
        train_full = shuffled[:n_train]
        test_imgs = shuffled[n_train:]

        with open(train_split_file, "w") as f:
            for p in sorted(train_full):
                f.write(f"{p.name}\n")
        with open(test_split_file, "w") as f:
            for p in sorted(test_imgs):
                f.write(f"{p.name}\n")
        print(f"Saved {len(train_full)} to {train_split_file} and {len(test_imgs)} to {test_split_file}")

    # Carve validation set out of train portion only (never touch test portion!)
    rng_val = random.Random(seed)
    train_pool = list(train_full)
    rng_val.shuffle(train_pool)

    n_val = max(1, int(round(len(train_pool) * val_ratio)))
    val_imgs = train_pool[:n_val]
    train_imgs = train_pool[n_val:]

    return train_imgs, val_imgs, test_imgs


def plot_and_save_loss_curve(history_csv: Path, out_path: Path) -> None:
    """Plot training and validation Loss and IoU curves, highlighting best epoch."""
    epochs, t_losses, v_losses, t_ious, v_ious = [], [], [], [], []
    with open(history_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            t_losses.append(float(row["train_loss"]))
            v_losses.append(float(row["val_loss"]))
            t_ious.append(float(row["train_iou"]))
            v_ious.append(float(row["val_iou"]))

    if not epochs:
        return

    best_idx = int(torch.tensor(v_losses).argmin().item())
    best_epoch = epochs[best_idx]
    best_val_loss = v_losses[best_idx]
    best_val_iou = v_ious[best_idx]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Subplot 1: Loss
    ax1.plot(epochs, t_losses, "b-o", markersize=3, label="Train Loss (BCE+Dice)")
    ax1.plot(epochs, v_losses, "r-s", markersize=3, label="Val Loss (BCE+Dice)")
    ax1.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7)
    ax1.scatter([best_epoch], [best_val_loss], color="gold", s=100, zorder=5, edgecolors="black",
                label=f"Best Ep {best_epoch} (Loss {best_val_loss:.4f})")
    ax1.set_title("Loss Convergence (Train vs. Val)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Subplot 2: IoU
    ax2.plot(epochs, t_ious, "b-o", markersize=3, label="Train IoU")
    ax2.plot(epochs, v_ious, "g-^", markersize=3, label="Val IoU")
    ax2.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7)
    ax2.scatter([best_epoch], [best_val_iou], color="gold", s=100, zorder=5, edgecolors="black",
                label=f"Best Ep {best_epoch} (IoU {best_val_iou:.4f})")
    ax2.set_title("Mean IoU Metric (Train vs. Val)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("IoU", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()
    print(f"Loss curve plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train custom UNet for lane segmentation.")
    parser.add_argument("--data-root", default="../image/image_1k_fern", help="Path to dataset root")
    parser.add_argument("--lane-class-id", type=int, default=0, help="YOLO-seg class ID for lane polygon")
    parser.add_argument("--img-w", type=int, default=64, help="Input width (default: 64)")
    parser.add_argument("--img-h", type=int, default=36, help="Input height (default: 36)")
    parser.add_argument("--base-ch", type=int, default=16, help="Base channel width for UNet")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--splits-dir", default="splits", help="Directory to save/load train.txt and test.txt")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for saved model checkpoints")
    parser.add_argument("--log-dir", default="runs", help="Directory for TensorBoard logs")
    parser.add_argument("--run-name", default="unet_lane", help="Run identifier")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    args = parser.parse_args()

    # Set deterministic seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # Device selection (CUDA / MPS / CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Compute device selected: {device}")

    # Discover and print dataset stats
    images_dir, labels_dir = resolve_dataset_dirs(args.data_root)
    all_images = list_image_files(images_dir)
    if not all_images:
        raise FileNotFoundError(f"No images found in {images_dir}")

    stats = compute_dataset_statistics(all_images, labels_dir, lane_class_id=args.lane_class_id)
    print_dataset_statistics(stats)

    # 65% train / 35% test split (with 10% carved from train for validation)
    splits_dir = Path(args.splits_dir)
    train_imgs, val_imgs, test_imgs = prepare_splits(all_images, splits_dir, seed=args.seed)
    print(f"Split Summary -> Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test (Held-out): {len(test_imgs)}")

    # Datasets and Loaders
    train_ds = LaneSegDataset(
        train_imgs, labels_dir, target_w=args.img_w, target_h=args.img_h,
        lane_class_id=args.lane_class_id, augment=True
    )
    val_ds = LaneSegDataset(
        val_imgs, labels_dir, target_w=args.img_w, target_h=args.img_h,
        lane_class_id=args.lane_class_id, augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    # Initialize model, loss, optimizer
    model = UNet(in_channels=3, num_classes=1, base_ch=args.base_ch).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Checkpoint & logging directories
    ckpt_dir = Path(args.checkpoint_dir) / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(Path(args.log_dir) / args.run_name))

    history_csv = Path(args.log_dir) / args.run_name / "history.csv"
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(history_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_iou", "val_iou", "lr"])

    best_val_loss = float("inf")
    print("\nStarting training loop...")

    for epoch in range(1, args.epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss, train_iou, train_batches = 0.0, 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]", leave=False)
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            iou = compute_iou(logits, masks)
            train_loss += loss.item()
            train_iou += iou
            train_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{iou:.4f}"})

        train_loss /= max(1, train_batches)
        train_iou /= max(1, train_batches)

        # 2. Validation Phase
        model.eval()
        val_loss, val_iou, val_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                iou = compute_iou(logits, masks)

                val_loss += loss.item()
                val_iou += iou
                val_batches += 1

        val_loss /= max(1, val_batches)
        val_iou /= max(1, val_batches)
        curr_lr = optimizer.param_groups[0]["lr"]

        # 3. Logging
        tb_writer.add_scalar("Loss/train", train_loss, epoch)
        tb_writer.add_scalar("Loss/val", val_loss, epoch)
        tb_writer.add_scalar("IoU/train", train_iou, epoch)
        tb_writer.add_scalar("IoU/val", val_iou, epoch)
        tb_writer.add_scalar("Learning_rate", curr_lr, epoch)

        with open(history_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{train_iou:.6f}", f"{val_iou:.6f}", f"{curr_lr:.6e}"])

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss:.4f}, IoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f} | "
              f"lr: {curr_lr:.2e}")

        # 4. Checkpointing
        ckpt_config = {
            "img_w": args.img_w,
            "img_h": args.img_h,
            "base_ch": args.base_ch,
            "lane_class_id": args.lane_class_id
        }
        last_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": ckpt_config,
            "val_loss": val_loss,
            "val_iou": val_iou
        }
        torch.save(last_ckpt, ckpt_dir / "last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": ckpt_config,
                "val_loss": val_loss,
                "val_iou": val_iou
            }
            torch.save(best_ckpt, ckpt_dir / "best.pt")
            print(f"  --> Saved new best checkpoint at epoch {epoch} (Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f})")

    tb_writer.close()
    print(f"\nTraining completed! Checkpoints saved in {ckpt_dir}")

    # Generate loss curve visualization
    loss_curve_path = Path("assets") / "loss_curve.png"
    plot_and_save_loss_curve(history_csv, loss_curve_path)


if __name__ == "__main__":
    main()
