"""
Three-Phase Training Script for the Two-Stage NVS Model.

Phase 1: Train GeometricFlowNet (Model 1) alone — coarse prediction
Phase 2: Train RefineUNet (Model 2) with frozen Model 1
Phase 3: Joint end-to-end fine-tuning

Usage:
    # Phase 1 only (start fresh):
    python train_nvs.py --data-dir data/train --phase 1

    # Phase 2 (requires Phase 1 checkpoint):
    python train_nvs.py --data-dir data/train --phase 2 \
        --coarse-ckpt checkpoints/coarse_best.pth

    # Phase 3 (requires Phase 2 checkpoint):
    python train_nvs.py --data-dir data/train --phase 3 \
        --coarse-ckpt checkpoints/coarse_best.pth \
        --refine-ckpt checkpoints/refine_best.pth

    # Run all three phases sequentially:
    python train_nvs.py --data-dir data/train --phase all

Checkpoints are saved to ./checkpoints/ by default.
TensorBoard logs are saved to ./runs/.
"""

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torch.utils.tensorboard import SummaryWriter

from model import IFNet, IFNet_m
from model import RifeModel
from dataset import IFNetDataset, collate_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute PSNR (dB) between [0,1] images."""
    mse = ((pred - gt) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def save_checkpoint(state: dict, path: Path, is_best: bool = False, best_path: Path = None):
    torch.save(state, str(path))
    if is_best and best_path is not None:
        shutil.copy(str(path), str(best_path))


def load_checkpoint(path: Path, model: nn.Module, optimizer=None, scheduler=None):
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_psnr", 0.0)


class Logger:
    """Simple CSV logger + optional TensorBoard."""

    def __init__(self, log_dir: Path, tag: str):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = log_dir / f"{tag}.csv"
        self._tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb = SummaryWriter(log_dir=str(log_dir / tag))
        except ImportError:
            pass
        self._header_written = False

    def log(self, step: int, metrics: dict):
        # CSV
        if not self._header_written:
            self.csv_path.write_text("step," + ",".join(metrics.keys()) + "\n")
            self._header_written = True
        with open(self.csv_path, "a") as f:
            f.write(f"{step}," + ",".join(f"{v:.6f}" for v in metrics.values()) + "\n")
        # TensorBoard
        if self._tb is not None:
            for k, v in metrics.items():
                self._tb.add_scalar(k, v, step)

    def close(self):
        if self._tb is not None:
            self._tb.close()


device = torch.device("cuda")

log_path = 'train_log'

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)


def evaluate(model, val_loader, nr_eval, writer_val):
    loss_l1_list = []
    loss_distill_list = []
    loss_tea_list = []
    loss_depth_list = []
    psnr_list = []
    psnr_list_teacher = []
    time_stamp = time.time()
    for i, batch in enumerate(val_loader):
        input_tensor = batch["input_tensor"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        with torch.no_grad():
            pred, info = model.update(input_tensor, gt, training=False)
            merged_img = info['merged_tea']
        loss_l1_list.append(info['loss_l1'].cpu().numpy())
        loss_tea_list.append(info['loss_tea'].cpu().numpy())
        loss_distill_list.append(info['loss_distill'].cpu().numpy())
        loss_depth_list.append(info['loss_depth'].cpu().numpy())
        for j in range(gt.shape[0]):
            psnr_val = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr_val)
            psnr_val = -10 * math.log10(torch.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).cpu().data)
            psnr_list_teacher.append(psnr_val)
        gt_np = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        pred_np = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        merged_np = (merged_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        flow0 = info['flow'].permute(0, 2, 3, 1).cpu().numpy()
        flow1 = info['flow_tea'].permute(0, 2, 3, 1).cpu().numpy()
        if i == 0 and writer_val is not None:
            for j in range(min(10, gt_np.shape[0])):
                imgs_vis = np.concatenate((merged_np[j], pred_np[j], gt_np[j]), 1)[:, :, ::-1]
                writer_val.add_image(str(j) + '/img', imgs_vis.copy(), nr_eval, dataformats='HWC')
                writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]), nr_eval, dataformats='HWC')

    eval_time_interval = time.time() - time_stamp

    if writer_val is not None:
        writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
        writer_val.add_scalar('psnr_teacher', np.array(psnr_list_teacher).mean(), nr_eval)
        writer_val.add_scalar('loss_depth', np.array(loss_depth_list).mean(), nr_eval)
    return float(np.array(psnr_list).mean()) if psnr_list else 0.0


# ---------------------------------------------------------------------------
# Phase 1: Train GeometricFlowNet
# ---------------------------------------------------------------------------

def train_phase1(args, device: torch.device):
    """Train Model 1 (GeometricFlowNet) alone."""

    writer = SummaryWriter('train')
    writer_val = SummaryWriter('validate')

    print("\n" + "=" * 70)
    print("Training IFnet")
    print("=" * 70)

    # Dataset
    full_dataset = IFNetDataset(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        crop_size=(args.crop_h, args.crop_w),
        augment=True,
        is_train=True,
        max_depth=args.max_depth,
    )
    n_val = max(1, len(full_dataset) // 5)
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Model — RifeModel manages its own device and internal optimizer
    step_per_epoch = len(train_loader)
    total_steps = args.epochs * step_per_epoch
    model = RifeModel(args, total_steps=total_steps, lr=args.lr)
    print(f"  Flownet params: {sum(p.numel() for p in model.flownet.parameters()):,}")

    ckpt_dir = args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_psnr = 0.0

    if args.model_ckpt and args.model_ckpt.exists():
        print(f"  Resuming from {args.model_ckpt}")
        ckpt = torch.load(str(args.model_ckpt), map_location="cpu")
        model.flownet.load_state_dict(ckpt["flownet_state"])
        start_epoch = ckpt.get("epoch", 0)
        best_psnr = ckpt.get("best_psnr", 0.0)

    patience = args.patience
    no_improve = 0
    nr_eval = 0

    for epoch in range(start_epoch, args.epochs):
        # --- Train ---
        model.train()
        train_losses = []
        t_ep = time.time()
        time_stamp = time.time()
        for step, batch in enumerate(train_loader):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            input_tensor = batch["input_tensor"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            pred, info = model.update(input_tensor, gt, training=True)
            train_losses.append(info['loss_l1'].item())

            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 10 == 1:
                lr_now = model.optimG.param_groups[0]['lr']
                writer.add_scalar('learning_rate', lr_now, epoch * step_per_epoch + step)
                writer.add_scalar('loss/l1', info['loss_l1'], step)
                writer.add_scalar('loss/tea', info['loss_tea'], step)
                writer.add_scalar('loss/distill', info['loss_distill'], step)
                writer.add_scalar('loss/depth', info['loss_depth'], step)
            if step % 1000 == 1:
                gt_np = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                mask = (torch.cat((info['mask'], info['mask_tea']), 3).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred_np = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                merged_img = (info['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                if merged_img.shape[3] == 1:
                    merged_img = np.repeat(merged_img, 3, axis=3)
                flow0 = info['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                flow1 = info['flow_tea'].permute(0, 2, 3, 1).detach().cpu().numpy()
                for j in range(min(5, gt_np.shape[0])):
                    imgs_vis = np.concatenate((merged_img[j], pred_np[j], gt_np[j]), 1)[:, :, ::-1]
                    writer.add_image(str(j) + '/img', imgs_vis, step, dataformats='HWC')
                    writer.add_image(str(j) + '/flow', np.concatenate((flow2rgb(flow0[j]), flow2rgb(flow1[j])), 1), step, dataformats='HWC')
                    writer.add_image(str(j) + '/mask', mask[j], step, dataformats='HWC')
                writer.flush()
            print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4e}'.format(
                epoch, step, len(train_loader), data_time_interval, train_time_interval, info['loss_l1']))

        # --- Validate ---
        mean_psnr = evaluate(model, val_loader, nr_eval, writer_val)
        nr_eval += 1
        mean_train_loss = float(np.mean(train_losses))
        ep_time = time.time() - t_ep

        print(
            f"\n  Epoch {epoch+1:03d}/{args.epochs}"
            f"  train_loss={mean_train_loss:.4f}"
            f"  val_PSNR={mean_psnr:.2f} dB"
            f"  [{ep_time:.1f}s]"
        )

        writer.add_scalar('train/loss', mean_train_loss, epoch)
        writer.add_scalar('val/psnr', mean_psnr, epoch)

        is_best = mean_psnr > best_psnr
        if is_best:
            best_psnr = mean_psnr
            no_improve = 0
        else:
            no_improve += 1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "flownet_state": model.flownet.state_dict(),
                "best_psnr": best_psnr,
            },
            ckpt_dir / "coarse_last.pth",
            is_best=is_best,
            best_path=ckpt_dir / "coarse_best.pth",
        )

        if no_improve >= patience:
            print(f"  Early stopping after {patience} epochs without improvement.")
            break

    writer.close()
    writer_val.close()
    print(f"\nPhase 1 complete. Best val PSNR: {best_psnr:.2f} dB")
    return model, ckpt_dir / "coarse_best.pth"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train two-stage NVS model")

    # Data
    p.add_argument("--data-dir", type=Path, required=True, help="Training data directory")
    p.add_argument("--cache-dir", type=Path, default=None, help="Pre-computed cache directory")
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"), help="Checkpoint directory")
    p.add_argument("--log-dir", type=Path, default=Path("runs"), help="TensorBoard log directory")

    p.add_argument("--model-ckpt", type=Path, default=None, help="Path to coarse model checkpoint")
    p.add_argument("--crop-h", type=int, default=512, help="Crop height for training")
    p.add_argument("--crop-w", type=int, default=512, help="Crop width for training")

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=300, help="Phase 1 epochs")
    p.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    p.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    p.add_argument("--warm-up", type=int, default=500, help="warm-up steps")
    p.add_argument("--loss-depth-alpha", type=float, default=0.1, help="weight for depth loss")
    p.add_argument("--max-depth", type=float, default=80.0, help="Depth normalisation range (m)")
    p.add_argument("--patience", type=int, default=40, help="Early stopping patience (epochs)")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--use-perceptual", action="store_true", default=True,
        help="Use VGG perceptual loss in Phase 2/3 (requires torchvision)",
    )
    p.add_argument(
        "--no-perceptual", dest="use_perceptual", action="store_false",
        help="Disable VGG perceptual loss",
    )

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_phase1(args, device)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
