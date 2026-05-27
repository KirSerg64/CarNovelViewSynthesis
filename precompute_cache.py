"""
Offline Pre-computation of LiDAR Depth Maps and Geometric Warps.

Run this ONCE before training to cache:
  - depth_t0.npz, depth_t1.npz      : LiDAR projected to t0/t1 source camera (normalised)
  - depth_tgt.npz                    : LiDAR projected to target camera
  - warped_t0.npz, warped_t1.npz    : images warped to target pose
  - mask_t0.npz, mask_t1.npz        : validity masks

Caching accelerates training by ~5-10x vs computing on-the-fly per epoch.

Usage:
    python precompute_cache.py --data-dir data/train --cache-dir cache/train
    python precompute_cache.py --data-dir data/test  --cache-dir cache/test
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Reuse existing utilities from the baseline
from lidar_warp_nvs import (
    get_intrinsic_matrix,
    load_lidar,
    project_lidar_to_camera,
    densify_depth_map,
    inverse_warp_same_camera,
)

MAX_DEPTH = 80.0  # metres, for normalisation


def normalise_depth(depth: np.ndarray) -> np.ndarray:
    """Normalise depth to [0, 1] with 0 = invalid."""
    out = depth / MAX_DEPTH
    return out.clip(0.0, 1.0).astype(np.float32)


def process_sample(sample_dir: Path, cache_dir: Path, force: bool = False):
    """Compute and save depth maps + warps for one sample."""
    sample_id = sample_dir.name
    out_dir = cache_dir / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already cached
    if not force and (out_dir / "depth_tgt.npz").exists():
        print(f"  [SKIP] {sample_id} (already cached)")
        return

    print(f"  Processing {sample_id}...")
    t0 = time.time()

    meta = json.loads((sample_dir / "meta.json").read_text())
    target_cam = meta["target_camera"]
    intr = meta["intrinsics"][target_cam]
    H = intr["height"]
    W = intr["width"]
    K = get_intrinsic_matrix(intr)

    c2w_t0 = np.array(meta["poses_c2w"]["t0"][target_cam], dtype=np.float64)
    c2w_t1 = np.array(meta["poses_c2w"]["t1"][target_cam], dtype=np.float64)
    c2w_tgt = np.array(meta["poses_c2w"]["target"][target_cam], dtype=np.float64)

    # Load LiDAR
    xyz_world = load_lidar(sample_dir)
    print(f"    LiDAR loaded: {len(xyz_world):,} points")

    # --- Depth maps ---
    depth_t0_raw = project_lidar_to_camera(xyz_world, c2w_t0, K, W, H)[0]
    depth_t0_raw = densify_depth_map(depth_t0_raw)
    np.savez_compressed(str(out_dir / "depth_t0.npz"), data=normalise_depth(depth_t0_raw))

    depth_t1_raw = project_lidar_to_camera(xyz_world, c2w_t1, K, W, H)[0]
    depth_t1_raw = densify_depth_map(depth_t1_raw)
    np.savez_compressed(str(out_dir / "depth_t1.npz"), data=normalise_depth(depth_t1_raw))

    depth_tgt_raw = project_lidar_to_camera(xyz_world, c2w_tgt, K, W, H)[0]
    depth_tgt_raw = densify_depth_map(depth_tgt_raw)
    np.savez_compressed(str(out_dir / "depth_tgt.npz"), data=normalise_depth(depth_tgt_raw))

    # --- Geometric warps (inverse warp: same camera, different timestamp) ---
    img_t0 = cv2.imread(
        str(sample_dir / "input" / "t0" / f"{target_cam}.jpg"), cv2.IMREAD_COLOR
    )
    img_t0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2RGB)

    img_t1 = cv2.imread(
        str(sample_dir / "input" / "t1" / f"{target_cam}.jpg"), cv2.IMREAD_COLOR
    )
    img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)

    warped_t0, mask_t0 = inverse_warp_same_camera(
        img_t0, depth_t0_raw, c2w_t0, K, c2w_tgt, K, W, H
    )
    np.savez_compressed(
        str(out_dir / "warped_t0.npz"),
        data=(warped_t0.astype(np.float32) / 255.0),
    )
    np.savez_compressed(
        str(out_dir / "mask_t0.npz"),
        data=mask_t0.astype(np.float32),
    )

    warped_t1, mask_t1 = inverse_warp_same_camera(
        img_t1, depth_t1_raw, c2w_t1, K, c2w_tgt, K, W, H
    )
    np.savez_compressed(
        str(out_dir / "warped_t1.npz"),
        data=(warped_t1.astype(np.float32) / 255.0),
    )
    np.savez_compressed(
        str(out_dir / "mask_t1.npz"),
        data=mask_t1.astype(np.float32),
    )

    # --- Meta (store alpha and image size for dataset loader) ---
    ts = meta["timestamps_ns"]
    alpha = float((ts["target"] - ts["t0"]) / (ts["t1"] - ts["t0"]))
    cache_meta = {
        "alpha": float(np.clip(alpha, 0.0, 1.0)),
        "target_camera": target_cam,
        "height": H,
        "width": W,
    }
    import json as _json
    (out_dir / "cache_meta.json").write_text(_json.dumps(cache_meta))

    print(f"    Done in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute NVS depth & warp cache")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Cache output directory")
    parser.add_argument(
        "--force", action="store_true", help="Recompute even if cache exists"
    )
    parser.add_argument(
        "--samples", nargs="*", default=None,
        help="Specific sample IDs to process (default: all)",
    )
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    sample_dirs = sorted(p for p in args.data_dir.iterdir() if p.is_dir())

    if args.samples:
        sample_dirs = [s for s in sample_dirs if s.name in args.samples]

    print(f"Processing {len(sample_dirs)} samples → {args.cache_dir}")
    for sample_dir in sample_dirs:
        process_sample(sample_dir, args.cache_dir, force=args.force)

    print("\nPre-computation complete.")


if __name__ == "__main__":
    main()
