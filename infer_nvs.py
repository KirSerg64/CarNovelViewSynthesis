"""
Inference / Submission Script for the Two-Stage NVS Model.

Generates pred.jpg for each test sample in submission/<sample_id>/pred.jpg.

Usage:
    # Generate test submission (requires trained checkpoints):
    python infer_nvs.py \
        --data-dir data/test \
        --output-dir submission \
        --coarse-ckpt checkpoints/coarse_e2e_best.pth \
        --refine-ckpt checkpoints/refine_e2e_best.pth

    # Evaluate on training set (produces submission + runs evaluate.py):
    python infer_nvs.py \
        --data-dir data/train \
        --output-dir output_train \
        --coarse-ckpt checkpoints/coarse_e2e_best.pth \
        --refine-ckpt checkpoints/refine_e2e_best.pth \
        --evaluate

    # Coarse-only mode (skip refinement):
    python infer_nvs.py \
        --data-dir data/test \
        --output-dir submission \
        --coarse-ckpt checkpoints/coarse_best.pth \
        --coarse-only

Fallback behaviour:
    If a sample fails (OOM, missing data, etc.) the script falls back to the
    simple geometric warp baseline so the submission is always complete.
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nvs_model import GeometricFlowNet, RefineUNet
from lidar_warp_nvs import (
    get_intrinsic_matrix,
    load_lidar,
    project_lidar_to_camera,
    densify_depth_map,
    inverse_warp_same_camera,
)

CAMERAS = ["front", "left_fwd", "left_bwd", "right_fwd", "right_bwd", "rear"]
MAX_DEPTH = 80.0


# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------

def load_sample(sample_dir: Path, meta: dict, max_depth: float = MAX_DEPTH, device: torch.device = None):
    """
    Load a single sample and build the 21-channel input tensor.

    Returns:
        x:            (1, 21, H, W) tensor ready for GeometricFlowNet
        alpha:        scalar float
        warped_blend: (1, 3, H, W) geometric-warp baseline (fallback)
        target_depth: (1, 1, H, W) normalised LiDAR depth at target view
        blend_confidence: (1, 1, H, W) warp validity mask
        img_size:     (H, W) original image size
    """
    target_cam = meta["target_camera"]
    intr = meta["intrinsics"][target_cam]
    H, W = intr["height"], intr["width"]
    K = get_intrinsic_matrix(intr)

    c2w_t0 = np.array(meta["poses_c2w"]["t0"][target_cam], dtype=np.float64)
    c2w_t1 = np.array(meta["poses_c2w"]["t1"][target_cam], dtype=np.float64)
    c2w_tgt = np.array(meta["poses_c2w"]["target"][target_cam], dtype=np.float64)

    ts = meta["timestamps_ns"]
    alpha = float(np.clip(
        (ts["target"] - ts["t0"]) / (ts["t1"] - ts["t0"]), 0.0, 1.0
    ))

    # LiDAR
    xyz_world = load_lidar(sample_dir)

    # Depth maps
    depth_t0 = densify_depth_map(project_lidar_to_camera(xyz_world, c2w_t0, K, W, H)[0])
    depth_t1 = densify_depth_map(project_lidar_to_camera(xyz_world, c2w_t1, K, W, H)[0])
    depth_tgt = densify_depth_map(project_lidar_to_camera(xyz_world, c2w_tgt, K, W, H)[0])

    def norm_d(d):
        return (d / max_depth).clip(0, 1).astype(np.float32)[np.newaxis]  # (1,H,W)

    # Images
    def load_rgb(path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0  # (H,W,3)

    img_t0 = load_rgb(sample_dir / "input" / "t0" / f"{target_cam}.jpg")
    img_t1 = load_rgb(sample_dir / "input" / "t1" / f"{target_cam}.jpg")

    # Geometric warps
    def chw(img):
        return np.transpose(img, (2, 0, 1))  # (3,H,W)

    img_t0_u8 = (img_t0 * 255).clip(0, 255).astype(np.uint8)
    img_t1_u8 = (img_t1 * 255).clip(0, 255).astype(np.uint8)

    warped_t0_u8, mask_t0 = inverse_warp_same_camera(
        img_t0_u8, depth_t0, c2w_t0, K, c2w_tgt, K, W, H
    )
    warped_t1_u8, mask_t1 = inverse_warp_same_camera(
        img_t1_u8, depth_t1, c2w_t1, K, c2w_tgt, K, W, H
    )

    warped_t0 = warped_t0_u8.astype(np.float32) / 255.0
    warped_t1 = warped_t1_u8.astype(np.float32) / 255.0
    mask_t0_f = mask_t0.astype(np.float32)[np.newaxis]  # (1,H,W)
    mask_t1_f = mask_t1.astype(np.float32)[np.newaxis]

    # 18-channel input
    input_tensor = np.concatenate([
        chw(img_t0),       # 3
        chw(img_t1),       # 3
        norm_d(depth_t0),  # 1
        norm_d(depth_t1),  # 1
        chw(warped_t0),    # 3
        chw(warped_t1),    # 3
        mask_t0_f,         # 1
        mask_t1_f,         # 1
        norm_d(depth_tgt), # 1
        np.full_like(mask_t0_f, alpha),  # 1
    ], axis=0)  # (21,H,W)

    warped_blend = (
        (1.0 - alpha) * chw(warped_t0) + alpha * chw(warped_t1)
    )  # (3,H,W)
    blend_conf = (mask_t0_f + mask_t1_f).clip(0, 1)  # (1,H,W)

    def to_tensor(arr):
        t = torch.from_numpy(arr.copy()).unsqueeze(0)  # (1, C, H, W)
        if device is not None:
            t = t.to(device)
        return t

    return {
        "x": to_tensor(input_tensor),
        "alpha": alpha,
        "warped_blend": to_tensor(warped_blend),
        "target_depth": to_tensor(norm_d(depth_tgt)),
        "blend_confidence": to_tensor(blend_conf),
        "img_size": (H, W),
        "img_t0": to_tensor(chw(img_t0)),
        "img_t1": to_tensor(chw(img_t1)),
    }


def geometric_baseline(sample_dir: Path, meta: dict) -> np.ndarray:
    """
    Fallback: simple geometric warp blend.
    Returns (H, W, 3) uint8 image.
    """
    target_cam = meta["target_camera"]
    intr = meta["intrinsics"][target_cam]
    H, W = intr["height"], intr["width"]
    K = get_intrinsic_matrix(intr)

    c2w_t0 = np.array(meta["poses_c2w"]["t0"][target_cam], dtype=np.float64)
    c2w_t1 = np.array(meta["poses_c2w"]["t1"][target_cam], dtype=np.float64)
    c2w_tgt = np.array(meta["poses_c2w"]["target"][target_cam], dtype=np.float64)

    ts = meta["timestamps_ns"]
    alpha = float(np.clip((ts["target"] - ts["t0"]) / (ts["t1"] - ts["t0"]), 0, 1))

    xyz = load_lidar(sample_dir)
    depth_t0 = densify_depth_map(project_lidar_to_camera(xyz, c2w_t0, K, W, H)[0])
    depth_t1 = densify_depth_map(project_lidar_to_camera(xyz, c2w_t1, K, W, H)[0])

    def load_u8(path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_t0 = load_u8(sample_dir / "input" / "t0" / f"{target_cam}.jpg")
    img_t1 = load_u8(sample_dir / "input" / "t1" / f"{target_cam}.jpg")

    w0, m0 = inverse_warp_same_camera(img_t0, depth_t0, c2w_t0, K, c2w_tgt, K, W, H)
    w1, m1 = inverse_warp_same_camera(img_t1, depth_t1, c2w_t1, K, c2w_tgt, K, W, H)

    blend = (
        (1.0 - alpha) * w0.astype(np.float32) + alpha * w1.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    # Fill holes with direct temporal blend
    hole = ~(m0 | m1)
    if hole.any():
        avg = ((img_t0.astype(np.float32) + img_t1.astype(np.float32)) * 0.5).astype(np.uint8)
        blend[hole] = avg[hole]

    return blend


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_sample(
    sample_dir: Path,
    meta: dict,
    coarse_model: GeometricFlowNet,
    refine_model: RefineUNet = None,
    device: torch.device = None,
    coarse_only: bool = False,
) -> np.ndarray:
    """Run the full two-stage pipeline on a single sample."""
    sample = load_sample(sample_dir, meta, device=device)
    x = sample["x"]                           # (1,21,H,W)
    tgt_depth = sample["target_depth"]        # (1,1,H,W)
    warped_blend = sample["warped_blend"]     # (1,3,H,W)

    # --- Stage 1: Coarse ---
    coarse_out = coarse_model(x)
    coarse = coarse_out["coarse"]             # (1,3,H,W)
    blend = coarse_out["blend"]               # (1,1,H,W) — used as hole mask

    if coarse_only or refine_model is None:
        pred = coarse
    else:
        # --- Stage 2: Refinement ---
        img_t0 = x[:, 0:3]
        img_t1 = x[:, 3:6]
        residual_diff = (coarse - warped_blend).clamp(-1, 1)
        refine_input = torch.cat(
            [coarse, residual_diff, tgt_depth, blend, img_t0, img_t1], dim=1
        )  # (1,14,H,W)

        refine_out = refine_model(refine_input, blend)
        pred = (coarse + refine_out["residual"]).clamp(0, 1)

    # Convert to uint8 RGB numpy
    pred_np = pred[0].permute(1, 2, 0).cpu().numpy()  # (H,W,3)
    pred_np = (pred_np * 255).clip(0, 255).astype(np.uint8)
    return pred_np


def main():
    parser = argparse.ArgumentParser(description="NVS inference / submission generation")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coarse-ckpt", type=Path, default=None)
    parser.add_argument("--refine-ckpt", type=Path, default=None)
    parser.add_argument("--coarse-base-ch", type=int, default=32)
    parser.add_argument("--refine-base-ch", type=int, default=32)
    parser.add_argument("--coarse-only", action="store_true")
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run evaluate.py after generating predictions (training set only)",
    )
    parser.add_argument("--samples", nargs="*", default=None, help="Specific sample IDs")
    parser.add_argument(
        "--fallback-on-error", action="store_true", default=True,
        help="Fall back to geometric baseline if model inference fails",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    coarse_model = None
    refine_model = None

    if args.coarse_ckpt and args.coarse_ckpt.exists():
        coarse_model = GeometricFlowNet(base_ch=args.coarse_base_ch).to(device).eval()
        ckpt = torch.load(str(args.coarse_ckpt), map_location=device)
        coarse_model.load_state_dict(ckpt["model_state"])
        print(f"Loaded coarse model from {args.coarse_ckpt}")
    else:
        print("WARNING: No coarse checkpoint — falling back to geometric baseline for all samples")

    if not args.coarse_only and args.refine_ckpt and args.refine_ckpt.exists():
        refine_model = RefineUNet(base_ch=args.refine_base_ch).to(device).eval()
        ckpt = torch.load(str(args.refine_ckpt), map_location=device)
        refine_model.load_state_dict(ckpt["model_state"])
        print(f"Loaded refine model from {args.refine_ckpt}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_dirs = sorted(p for p in args.data_dir.iterdir() if p.is_dir())
    if args.samples:
        sample_dirs = [s for s in sample_dirs if s.name in args.samples]

    print(f"\nProcessing {len(sample_dirs)} samples...")
    results = []

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        out_dir = args.output_dir / sample_id
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_path = out_dir / "pred.jpg"

        meta = json.loads((sample_dir / "meta.json").read_text())
        t0 = time.time()

        try:
            if coarse_model is not None:
                pred_rgb = infer_sample(
                    sample_dir, meta, coarse_model, refine_model,
                    device=device, coarse_only=args.coarse_only,
                )
            else:
                raise RuntimeError("No model loaded")
        except Exception as e:
            if args.fallback_on_error:
                print(f"  [WARN] {sample_id}: model failed ({e}) — using geometric fallback")
                pred_rgb = geometric_baseline(sample_dir, meta)
            else:
                raise

        # Save as JPEG
        pred_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(pred_path), pred_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        elapsed = time.time() - t0
        results.append(sample_id)
        print(f"  {sample_id}: saved in {elapsed:.1f}s")

    print(f"\nGenerated {len(results)} predictions → {args.output_dir}")

    # Optionally run evaluation
    if args.evaluate:
        print("\nRunning evaluation...")
        subprocess.run(
            [
                sys.executable, "evaluate.py",
                "--pred-dir", str(args.output_dir),
                "--gt-dir", str(args.data_dir),
            ],
            check=False,
        )


if __name__ == "__main__":
    main()
