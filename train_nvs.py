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
from torch.utils.data import DataLoader, random_split

from nvs_model import GeometricFlowNet, RefineUNet, CoarseLoss, RefineLoss, NVSDataset
from nvs_model.dataset import collate_fn


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


# ---------------------------------------------------------------------------
# Phase 1: Train GeometricFlowNet
# ---------------------------------------------------------------------------

def train_phase1(args, device: torch.device):
    """Train Model 1 (GeometricFlowNet) alone."""
    print("\n" + "=" * 70)
    print("PHASE 1: Training GeometricFlowNet (coarse predictor)")
    print("=" * 70)

    # Dataset
    full_dataset = NVSDataset(
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

    # Model
    model = GeometricFlowNet(base_ch=args.coarse_base_ch).to(device)
    print(f"  Model 1 params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=args.lr1, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs1, eta_min=args.lr1 * 0.01)
    criterion = CoarseLoss()

    ckpt_dir = args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.log_dir, "phase1")

    start_epoch = 0
    best_psnr = 0.0
    if args.coarse_ckpt and args.coarse_ckpt.exists():
        print(f"  Resuming from {args.coarse_ckpt}")
        start_epoch, best_psnr = load_checkpoint(args.coarse_ckpt, model, optimizer, scheduler)

    patience = args.patience
    no_improve = 0

    for epoch in range(start_epoch, args.epochs1):
        # --- Train ---
        model.train()
        train_losses = []
        t_ep = time.time()
        for step, batch in enumerate(train_loader):
            x = batch["input_tensor"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            tgt_depth = batch["target_depth"].to(device, non_blocking=True)

            outputs = model(x)
            loss, loss_dict = criterion(outputs, gt, tgt_depth)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

            if step % 10 == 0:
                print(
                    f"  E{epoch+1:03d} S{step:04d}/{len(train_loader):04d}"
                    f"  loss={loss.item():.4f}"
                    f"  l1={loss_dict['l1']:.4f}"
                    f"  ssim={loss_dict['ssim']:.4f}"
                    f"  lr={scheduler.get_last_lr()[0]:.2e}",
                    end="\r",
                )

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_psnrs = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_tensor"].to(device)
                gt = batch["gt"].to(device)
                outputs = model(x)
                pred = outputs["coarse"]
                # Resize pred to gt if needed
                if pred.shape != gt.shape:
                    pred = torch.nn.functional.interpolate(
                        pred, size=gt.shape[2:], mode="bilinear", align_corners=False
                    )
                val_psnrs.append(psnr(pred, gt))

        mean_psnr = float(np.mean(val_psnrs))
        mean_train_loss = float(np.mean(train_losses))
        ep_time = time.time() - t_ep

        print(
            f"\n  Epoch {epoch+1:03d}/{args.epochs1}"
            f"  train_loss={mean_train_loss:.4f}"
            f"  val_PSNR={mean_psnr:.2f} dB"
            f"  [{ep_time:.1f}s]"
        )

        logger.log(epoch, {"train_loss": mean_train_loss, "val_psnr": mean_psnr})

        is_best = mean_psnr > best_psnr
        if is_best:
            best_psnr = mean_psnr
            no_improve = 0
        else:
            no_improve += 1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_psnr": best_psnr,
            },
            ckpt_dir / "coarse_last.pth",
            is_best=is_best,
            best_path=ckpt_dir / "coarse_best.pth",
        )

        if no_improve >= patience:
            print(f"  Early stopping after {patience} epochs without improvement.")
            break

    logger.close()
    print(f"\nPhase 1 complete. Best val PSNR: {best_psnr:.2f} dB")
    return model, ckpt_dir / "coarse_best.pth"


# ---------------------------------------------------------------------------
# Phase 2: Train RefineUNet with frozen Model 1
# ---------------------------------------------------------------------------

def train_phase2(args, device: torch.device, coarse_model: GeometricFlowNet = None):
    """Train Model 2 (RefineUNet) with frozen Model 1."""
    print("\n" + "=" * 70)
    print("PHASE 2: Training RefineUNet (refinement, Model 1 frozen)")
    print("=" * 70)

    # Load coarse model
    if coarse_model is None:
        coarse_model = GeometricFlowNet(base_ch=args.coarse_base_ch).to(device)
        ckpt_path = args.coarse_ckpt or (args.ckpt_dir / "coarse_best.pth")
        assert ckpt_path.exists(), f"Coarse checkpoint not found: {ckpt_path}"
        load_checkpoint(ckpt_path, coarse_model)
        print(f"  Loaded coarse model from {ckpt_path}")

    coarse_model.eval()
    for p in coarse_model.parameters():
        p.requires_grad_(False)

    # Dataset (no crop for refine — work at full resolution with smaller batches)
    dataset = NVSDataset(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        crop_size=(args.crop_h, args.crop_w),
        augment=True,
        is_train=True,
        max_depth=args.max_depth,
    )
    n_val = max(1, len(dataset) // 5)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=max(1, args.batch_size // 2),
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

    # Model 2
    refine_model = RefineUNet(base_ch=args.refine_base_ch).to(device)
    print(f"  Model 2 params: {sum(p.numel() for p in refine_model.parameters()):,}")

    optimizer = AdamW(refine_model.parameters(), lr=args.lr2, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs2, eta_min=args.lr2 * 0.01)
    criterion = RefineLoss(use_perceptual=args.use_perceptual)

    ckpt_dir = args.ckpt_dir
    logger = Logger(args.log_dir, "phase2")

    start_epoch = 0
    best_psnr = 0.0
    if args.refine_ckpt and args.refine_ckpt.exists():
        print(f"  Resuming from {args.refine_ckpt}")
        start_epoch, best_psnr = load_checkpoint(args.refine_ckpt, refine_model, optimizer, scheduler)

    patience = args.patience
    no_improve = 0

    for epoch in range(start_epoch, args.epochs2):
        refine_model.train()
        train_losses = []
        t_ep = time.time()

        for step, batch in enumerate(train_loader):
            x = batch["input_tensor"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            tgt_depth = batch["target_depth"].to(device, non_blocking=True)
            warp_conf = batch["warp_confidence"].to(device, non_blocking=True)
            warped_blend = batch["warped_blend"].to(device, non_blocking=True)

            # Get coarse prediction (no grad)
            with torch.no_grad():
                coarse_out = coarse_model(x)
                coarse = coarse_out["coarse"]
                blend = coarse_out["blend"]

            # Build refine input (14ch):
            # I_coarse(3), residual_diff(3), tgt_depth(1), confidence(1), img_t0(3), img_t1(3)
            img_t0 = x[:, 0:3]
            img_t1 = x[:, 3:6]
            residual_diff = (coarse - warped_blend).clamp(-1, 1)
            refine_input = torch.cat([
                coarse,          # 3
                residual_diff,   # 3
                tgt_depth,       # 1
                blend,           # 1
                img_t0,          # 3
                img_t1,          # 3
            ], dim=1)  # (B, 14, H, W)

            # Hole mask: low confidence areas = holes
            hole_mask = blend.clone()  # (B, 1, H, W) from coarse model

            refine_out = refine_model(refine_input, hole_mask)
            loss, loss_dict, final = criterion(
                refine_out, coarse, gt, tgt_depth, warp_confidence=warp_conf
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(refine_model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

            if step % 10 == 0:
                print(
                    f"  E{epoch+1:03d} S{step:04d}/{len(train_loader):04d}"
                    f"  loss={loss.item():.4f}"
                    f"  l1={loss_dict['l1']:.4f}"
                    f"  grad={loss_dict['gradient']:.4f}",
                    end="\r",
                )

        scheduler.step()

        # Validate
        refine_model.eval()
        val_psnrs = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_tensor"].to(device)
                gt = batch["gt"].to(device)
                tgt_depth = batch["target_depth"].to(device)
                warped_blend = batch["warped_blend"].to(device)

                coarse_out = coarse_model(x)
                coarse = coarse_out["coarse"]
                blend = coarse_out["blend"]

                img_t0 = x[:, 0:3]
                img_t1 = x[:, 3:6]
                residual_diff = (coarse - warped_blend).clamp(-1, 1)
                refine_input = torch.cat([coarse, residual_diff, tgt_depth, blend, img_t0, img_t1], dim=1)
                refine_out = refine_model(refine_input, blend)

                final = (coarse + refine_out["residual"]).clamp(0, 1)
                if final.shape != gt.shape:
                    final = torch.nn.functional.interpolate(final, size=gt.shape[2:], mode="bilinear", align_corners=False)
                val_psnrs.append(psnr(final, gt))

        mean_psnr = float(np.mean(val_psnrs))
        mean_train_loss = float(np.mean(train_losses))
        ep_time = time.time() - t_ep

        print(
            f"\n  Epoch {epoch+1:03d}/{args.epochs2}"
            f"  train_loss={mean_train_loss:.4f}"
            f"  val_PSNR={mean_psnr:.2f} dB"
            f"  [{ep_time:.1f}s]"
        )
        logger.log(epoch, {"train_loss": mean_train_loss, "val_psnr": mean_psnr})

        is_best = mean_psnr > best_psnr
        if is_best:
            best_psnr = mean_psnr
            no_improve = 0
        else:
            no_improve += 1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_state": refine_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_psnr": best_psnr,
            },
            ckpt_dir / "refine_last.pth",
            is_best=is_best,
            best_path=ckpt_dir / "refine_best.pth",
        )

        if no_improve >= patience:
            print(f"  Early stopping after {patience} epochs without improvement.")
            break

    logger.close()
    print(f"\nPhase 2 complete. Best val PSNR: {best_psnr:.2f} dB")
    return refine_model, ckpt_dir / "refine_best.pth"


# ---------------------------------------------------------------------------
# Phase 3: Joint end-to-end fine-tuning
# ---------------------------------------------------------------------------

def train_phase3(args, device: torch.device):
    """Fine-tune both models end-to-end."""
    print("\n" + "=" * 70)
    print("PHASE 3: Joint end-to-end fine-tuning")
    print("=" * 70)

    # Load both models
    coarse_model = GeometricFlowNet(base_ch=args.coarse_base_ch).to(device)
    coarse_ckpt = args.coarse_ckpt or (args.ckpt_dir / "coarse_best.pth")
    assert coarse_ckpt.exists(), f"Coarse checkpoint not found: {coarse_ckpt}"
    load_checkpoint(coarse_ckpt, coarse_model)
    print(f"  Loaded coarse model from {coarse_ckpt}")

    refine_model = RefineUNet(base_ch=args.refine_base_ch).to(device)
    refine_ckpt = args.refine_ckpt or (args.ckpt_dir / "refine_best.pth")
    assert refine_ckpt.exists(), f"Refine checkpoint not found: {refine_ckpt}"
    load_checkpoint(refine_ckpt, refine_model)
    print(f"  Loaded refine model from {refine_ckpt}")

    # Both models trainable, but use lower LR
    params = list(coarse_model.parameters()) + list(refine_model.parameters())
    optimizer = AdamW(params, lr=args.lr3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs3, eta_min=args.lr3 * 0.01)

    crit_coarse = CoarseLoss()
    crit_refine = RefineLoss(use_perceptual=args.use_perceptual)

    dataset = NVSDataset(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        crop_size=(args.crop_h, args.crop_w),
        augment=True,
        is_train=True,
        max_depth=args.max_depth,
    )
    n_val = max(1, len(dataset) // 5)
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=max(1, args.batch_size // 2),
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

    ckpt_dir = args.ckpt_dir
    logger = Logger(args.log_dir, "phase3")

    best_psnr = 0.0
    patience = args.patience
    no_improve = 0

    for epoch in range(args.epochs3):
        coarse_model.train()
        refine_model.train()
        train_losses = []
        t_ep = time.time()

        for step, batch in enumerate(train_loader):
            x = batch["input_tensor"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            tgt_depth = batch["target_depth"].to(device, non_blocking=True)
            warp_conf = batch["warp_confidence"].to(device, non_blocking=True)
            warped_blend = batch["warped_blend"].to(device, non_blocking=True)

            # Coarse forward
            coarse_out = coarse_model(x)
            coarse = coarse_out["coarse"]
            blend = coarse_out["blend"]

            loss_c, _ = crit_coarse(coarse_out, gt, tgt_depth)

            # Refine forward
            img_t0 = x[:, 0:3]
            img_t1 = x[:, 3:6]
            residual_diff = (coarse - warped_blend).clamp(-1, 1)
            refine_input = torch.cat([coarse, residual_diff, tgt_depth, blend, img_t0, img_t1], dim=1)
            refine_out = refine_model(refine_input, blend)

            loss_r, _, final = crit_refine(refine_out, coarse, gt, tgt_depth, warp_confidence=warp_conf)

            # Combined: heavier weight on final (refine) output
            loss = loss_c + 2.0 * loss_r

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

            if step % 10 == 0:
                print(
                    f"  E{epoch+1:03d} S{step:04d}/{len(train_loader):04d}"
                    f"  loss={loss.item():.4f}"
                    f"  l_c={loss_c.item():.4f}"
                    f"  l_r={loss_r.item():.4f}",
                    end="\r",
                )

        scheduler.step()

        # Validate
        coarse_model.eval()
        refine_model.eval()
        val_psnrs_coarse = []
        val_psnrs_final = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_tensor"].to(device)
                gt = batch["gt"].to(device)
                tgt_depth = batch["target_depth"].to(device)
                warped_blend = batch["warped_blend"].to(device)

                coarse_out = coarse_model(x)
                coarse = coarse_out["coarse"]
                blend = coarse_out["blend"]

                img_t0 = x[:, 0:3]
                img_t1 = x[:, 3:6]
                residual_diff = (coarse - warped_blend).clamp(-1, 1)
                refine_input = torch.cat([coarse, residual_diff, tgt_depth, blend, img_t0, img_t1], dim=1)
                refine_out = refine_model(refine_input, blend)
                final = (coarse + refine_out["residual"]).clamp(0, 1)

                if coarse.shape != gt.shape:
                    coarse = torch.nn.functional.interpolate(coarse, size=gt.shape[2:], mode="bilinear", align_corners=False)
                    final = torch.nn.functional.interpolate(final, size=gt.shape[2:], mode="bilinear", align_corners=False)

                val_psnrs_coarse.append(psnr(coarse, gt))
                val_psnrs_final.append(psnr(final, gt))

        mean_psnr_c = float(np.mean(val_psnrs_coarse))
        mean_psnr_f = float(np.mean(val_psnrs_final))
        mean_train_loss = float(np.mean(train_losses))
        ep_time = time.time() - t_ep

        print(
            f"\n  Epoch {epoch+1:03d}/{args.epochs3}"
            f"  train_loss={mean_train_loss:.4f}"
            f"  val_PSNR_coarse={mean_psnr_c:.2f}"
            f"  val_PSNR_final={mean_psnr_f:.2f} dB"
            f"  [{ep_time:.1f}s]"
        )
        logger.log(epoch, {
            "train_loss": mean_train_loss,
            "val_psnr_coarse": mean_psnr_c,
            "val_psnr_final": mean_psnr_f,
        })

        is_best = mean_psnr_f > best_psnr
        if is_best:
            best_psnr = mean_psnr_f
            no_improve = 0
        else:
            no_improve += 1

        save_checkpoint(
            {"epoch": epoch + 1, "model_state": coarse_model.state_dict(), "best_psnr": best_psnr},
            ckpt_dir / "coarse_e2e_last.pth",
            is_best=is_best,
            best_path=ckpt_dir / "coarse_e2e_best.pth",
        )
        save_checkpoint(
            {"epoch": epoch + 1, "model_state": refine_model.state_dict(), "best_psnr": best_psnr},
            ckpt_dir / "refine_e2e_last.pth",
            is_best=is_best,
            best_path=ckpt_dir / "refine_e2e_best.pth",
        )

        if no_improve >= patience:
            print(f"  Early stopping after {patience} epochs without improvement.")
            break

    logger.close()
    print(f"\nPhase 3 complete. Best val PSNR: {best_psnr:.2f} dB")


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

    # Phase control
    p.add_argument(
        "--phase", choices=["1", "2", "3", "all"], default="all",
        help="Training phase (1=coarse, 2=refine, 3=e2e, all=sequential)",
    )
    p.add_argument("--coarse-ckpt", type=Path, default=None, help="Path to coarse model checkpoint")
    p.add_argument("--refine-ckpt", type=Path, default=None, help="Path to refine model checkpoint")

    # Model sizes
    p.add_argument("--coarse-base-ch", type=int, default=32, help="GeometricFlowNet base channels")
    p.add_argument("--refine-base-ch", type=int, default=32, help="RefineUNet base channels")

    # Training hyperparameters
    p.add_argument("--epochs1", type=int, default=300, help="Phase 1 epochs")
    p.add_argument("--epochs2", type=int, default=200, help="Phase 2 epochs")
    p.add_argument("--epochs3", type=int, default=100, help="Phase 3 epochs")
    p.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    p.add_argument("--lr1", type=float, default=2e-4, help="Phase 1 learning rate")
    p.add_argument("--lr2", type=float, default=1e-4, help="Phase 2 learning rate")
    p.add_argument("--lr3", type=float, default=5e-5, help="Phase 3 learning rate")
    p.add_argument("--crop-h", type=int, default=512, help="Random crop height")
    p.add_argument("--crop-w", type=int, default=512, help="Random crop width")
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

    if args.phase in ("1", "all"):
        coarse_model, coarse_ckpt_path = train_phase1(args, device)
        args.coarse_ckpt = coarse_ckpt_path
    else:
        coarse_model = None

    if args.phase in ("2", "all"):
        refine_model, refine_ckpt_path = train_phase2(args, device, coarse_model)
        args.refine_ckpt = refine_ckpt_path

    if args.phase in ("3", "all"):
        train_phase3(args, device)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
