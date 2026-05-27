"""
PyTorch Dataset for Novel View Synthesis.

Loads multi-camera images, LiDAR depth maps, and geometric warps
for training/inference. Optionally reads from a pre-computed cache
(see precompute_cache.py) to avoid redundant LiDAR projection on every epoch.

Each sample yields a dict:
    'input_tensor'    : (21,) channels of float32 ready for GeometricFlowNet
    'gt'              : (3, H, W) ground-truth image [0, 1]
    'target_depth'    : (1, H, W) LiDAR depth at target camera
    'warp_confidence' : (1, H, W) initial blend confidence (from geometric warp)
    'warped_blend'    : (3, H, W) simple 50/50 blend of geometric warps
    'alpha'           : scalar float temporal interpolation factor
    'sample_id'       : str
"""

import json
import random
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Reuse the numpy warping functions from the baseline
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from lidar_warp_nvs import (
    get_intrinsic_matrix,
    load_lidar,
    project_lidar_to_camera,
    densify_depth_map,
    inverse_warp_same_camera,
)

CAMERAS = ["front", "left_fwd", "left_bwd", "right_fwd", "right_bwd", "rear"]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _load_image(path: Path, as_float: bool = True) -> np.ndarray:
    """Load JPEG → (H, W, 3) RGB float32 in [0,1] or uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if as_float:
        return img.astype(np.float32) / 255.0
    return img


def _normalise_depth(depth: np.ndarray, max_depth: float = 80.0) -> np.ndarray:
    """Normalise depth to [0, 1]; 0 stays 0 (invalid)."""
    out = depth / max_depth
    return out.clip(0.0, 1.0).astype(np.float32)


def _hwc_to_chw(img: np.ndarray) -> np.ndarray:
    """(H, W, C) → (C, H, W)."""
    return np.transpose(img, (2, 0, 1))


def _random_crop(arrays: list, size: Tuple[int, int]) -> list:
    """Apply the same random crop to all (C, H, W) arrays."""
    h, w = arrays[0].shape[1], arrays[0].shape[2]
    th, tw = size
    if h == th and w == tw:
        return arrays
    top = random.randint(0, h - th)
    left = random.randint(0, w - tw)
    return [a[:, top:top + th, left:left + tw] for a in arrays]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NVSDataset(Dataset):
    """
    Dataset for the two-stage NVS pipeline.

    Args:
        data_dir:   Path to data/train or data/test
        cache_dir:  Optional path to pre-computed cache (from precompute_cache.py)
        crop_size:  (H, W) random crop for training; None = no crop
        augment:    Apply horizontal flip + colour jitter
        is_train:   If False (test mode), no GT is loaded/returned
        max_depth:  Depth normalisation range in metres
    """

    def __init__(
        self,
        data_dir: Path,
        cache_dir: Optional[Path] = None,
        crop_size: Optional[Tuple[int, int]] = (512, 512),
        augment: bool = True,
        is_train: bool = True,
        max_depth: float = 80.0,
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.crop_size = crop_size
        self.augment = augment
        self.is_train = is_train
        self.max_depth = max_depth

        self.samples = sorted(
            p for p in self.data_dir.iterdir() if p.is_dir()
        )

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------

    def _load_from_cache(self, sample_id: str, key: str) -> Optional[np.ndarray]:
        """Load a pre-computed numpy array from cache."""
        if self.cache_dir is None:
            return None
        path = self.cache_dir / sample_id / f"{key}.npz"
        if path.exists():
            return np.load(str(path))["data"]
        return None

    # ------------------------------------------------------------------

    def _compute_depth_and_warp(
        self,
        meta: dict,
        xyz_world: np.ndarray,
        sample_dir: Path,
        target_cam: str,
    ) -> dict:
        """
        Compute (or load from cache) LiDAR depth maps and geometric warps
        for the target camera.

        Returns dict with keys:
            depth_t0, depth_t1, target_depth: (H, W) float32 normalised
            warped_t0, warped_t1: (H, W, 3) float32 [0,1]
            mask_t0, mask_t1:    (H, W) float32 binary validity
        """
        intr = meta["intrinsics"][target_cam]
        H = intr["height"]
        W = intr["width"]
        K = get_intrinsic_matrix(intr)

        c2w_t0 = np.array(meta["poses_c2w"]["t0"][target_cam], dtype=np.float64)
        c2w_t1 = np.array(meta["poses_c2w"]["t1"][target_cam], dtype=np.float64)
        c2w_tgt = np.array(meta["poses_c2w"]["target"][target_cam], dtype=np.float64)

        # --- Depth at t0 ---
        depth_t0 = project_lidar_to_camera(xyz_world, c2w_t0, K, W, H)[0]
        depth_t0 = densify_depth_map(depth_t0)
        depth_t0_norm = _normalise_depth(depth_t0, self.max_depth)

        # --- Depth at t1 ---
        depth_t1 = project_lidar_to_camera(xyz_world, c2w_t1, K, W, H)[0]
        depth_t1 = densify_depth_map(depth_t1)
        depth_t1_norm = _normalise_depth(depth_t1, self.max_depth)

        # --- Depth at target view ---
        depth_tgt = project_lidar_to_camera(xyz_world, c2w_tgt, K, W, H)[0]
        depth_tgt = densify_depth_map(depth_tgt)
        depth_tgt_norm = _normalise_depth(depth_tgt, self.max_depth)

        # --- Geometric warp from t0 and t1 to target ---
        img_t0 = cv2.imread(
            str(sample_dir / "input" / "t0" / f"{target_cam}.jpg"), cv2.IMREAD_COLOR
        )
        img_t0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2RGB)
        img_t1 = cv2.imread(
            str(sample_dir / "input" / "t1" / f"{target_cam}.jpg"), cv2.IMREAD_COLOR
        )
        img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)

        warped_t0, mask_t0 = inverse_warp_same_camera(
            img_t0, depth_t0, c2w_t0, K, c2w_tgt, K, W, H
        )
        warped_t1, mask_t1 = inverse_warp_same_camera(
            img_t1, depth_t1, c2w_t1, K, c2w_tgt, K, W, H
        )

        warped_t0 = warped_t0.astype(np.float32) / 255.0
        warped_t1 = warped_t1.astype(np.float32) / 255.0
        mask_t0 = mask_t0.astype(np.float32)
        mask_t1 = mask_t1.astype(np.float32)

        return {
            "depth_t0": depth_t0_norm,
            "depth_t1": depth_t1_norm,
            "target_depth": depth_tgt_norm,
            "warped_t0": warped_t0,
            "warped_t1": warped_t1,
            "mask_t0": mask_t0,
            "mask_t1": mask_t1,
            "img_t0": img_t0.astype(np.float32) / 255.0,
            "img_t1": img_t1.astype(np.float32) / 255.0,
        }

    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        sample_dir = self.samples[idx]
        sample_id = sample_dir.name
        meta = json.loads((sample_dir / "meta.json").read_text())
        target_cam = meta["target_camera"]

        # Temporal alpha
        ts = meta["timestamps_ns"]
        alpha = float((ts["target"] - ts["t0"]) / (ts["t1"] - ts["t0"]))
        alpha = float(np.clip(alpha, 0.0, 1.0))

        # Load LiDAR
        xyz_world = load_lidar(sample_dir)

        # Compute / load depth maps and warps
        geo = self._compute_depth_and_warp(meta, xyz_world, sample_dir, target_cam)

        img_t0 = geo["img_t0"]          # (H, W, 3)
        img_t1 = geo["img_t1"]          # (H, W, 3)
        warped_t0 = geo["warped_t0"]    # (H, W, 3)
        warped_t1 = geo["warped_t1"]
        mask_t0 = geo["mask_t0"]        # (H, W)
        mask_t1 = geo["mask_t1"]
        depth_t0 = geo["depth_t0"]      # (H, W)
        depth_t1 = geo["depth_t1"]
        tgt_depth = geo["target_depth"] # (H, W)

        # Convert images/masks to (C, H, W)
        def chw(x):
            if x.ndim == 2:
                return x[np.newaxis]
            return _hwc_to_chw(x)

        arr_img_t0 = chw(img_t0)       # (3, H, W)
        arr_img_t1 = chw(img_t1)
        arr_wt0 = chw(warped_t0)
        arr_wt1 = chw(warped_t1)
        arr_mt0 = chw(mask_t0)         # (1, H, W)
        arr_mt1 = chw(mask_t1)
        arr_dt0 = chw(depth_t0)        # (1, H, W)
        arr_dt1 = chw(depth_t1)
        arr_tdepth = chw(tgt_depth)    # (1, H, W)
        arr_alpha = np.full_like(arr_tdepth, alpha)  # (1, H, W)

        # GT
        if self.is_train:
            intr = meta["intrinsics"][target_cam]
            gt_path = sample_dir / "target" / f"{target_cam}.jpg"
            gt = _load_image(gt_path)  # (H, W, 3)
            arr_gt = chw(gt)           # (3, H, W)
        else:
            arr_gt = None

        # Temporal reversal augmentation (50% chance)
        if self.augment and random.random() < 0.5:
            arr_img_t0, arr_img_t1 = arr_img_t1, arr_img_t0
            arr_wt0, arr_wt1 = arr_wt1, arr_wt0
            arr_mt0, arr_mt1 = arr_mt1, arr_mt0
            arr_dt0, arr_dt1 = arr_dt1, arr_dt0
            alpha = 1.0 - alpha
            arr_alpha = np.full_like(arr_tdepth, alpha)

        # Horizontal flip augmentation (50% chance)
        if self.augment and random.random() < 0.5:
            def hflip(a):
                return a[..., ::-1].copy()
            arr_img_t0 = hflip(arr_img_t0)
            arr_img_t1 = hflip(arr_img_t1)
            arr_wt0 = hflip(arr_wt0)
            arr_wt1 = hflip(arr_wt1)
            arr_mt0 = hflip(arr_mt0)
            arr_mt1 = hflip(arr_mt1)
            arr_dt0 = hflip(arr_dt0)
            arr_dt1 = hflip(arr_dt1)
            arr_tdepth = hflip(arr_tdepth)
            arr_alpha = hflip(arr_alpha)
            if arr_gt is not None:
                arr_gt = hflip(arr_gt)

        # Colour jitter augmentation (independent per channel)
        if self.augment and random.random() < 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            for arr in [arr_img_t0, arr_img_t1, arr_wt0, arr_wt1]:
                arr *= brightness
                mean = arr.mean(axis=(1, 2), keepdims=True)
                arr[:] = (arr - mean) * contrast + mean
            np.clip(arr_img_t0, 0, 1, out=arr_img_t0)
            np.clip(arr_img_t1, 0, 1, out=arr_img_t1)
            np.clip(arr_wt0, 0, 1, out=arr_wt0)
            np.clip(arr_wt1, 0, 1, out=arr_wt1)

        # Concatenate into the 21-channel input tensor
        # Order: img_t0(3), img_t1(3), depth_t0(1), depth_t1(1),
        #        warped_t0(3), warped_t1(3), mask_t0(1), mask_t1(1),
        #        target_depth(1), alpha(1)
        input_tensor = np.concatenate([
            arr_img_t0,     # 3
            arr_img_t1,     # 3
            arr_dt0,        # 1
            arr_dt1,        # 1
            arr_wt0,        # 3
            arr_wt1,        # 3
            arr_mt0,        # 1
            arr_mt1,        # 1
            arr_tdepth,     # 1
            arr_alpha,      # 1
        ], axis=0)  # (21, H, W)

        # Random crop
        if self.crop_size is not None:
            all_arrs = [input_tensor, arr_tdepth]
            if arr_gt is not None:
                all_arrs.append(arr_gt)
            all_arrs = _random_crop(all_arrs, self.crop_size)
            input_tensor = all_arrs[0]
            arr_tdepth = all_arrs[1]
            if arr_gt is not None:
                arr_gt = all_arrs[2]

        # Warped blend (simple geometric fallback): alpha blend of warps
        warp_confidence = (arr_mt0 + arr_mt1).clip(0, 1)  # union of valid regions
        warped_blend = (1 - alpha) * arr_wt0 + alpha * arr_wt1

        # Convert to tensors
        sample = {
            "input_tensor": torch.from_numpy(input_tensor.copy()),
            "target_depth": torch.from_numpy(arr_tdepth.copy()),
            "warp_confidence": torch.from_numpy(warp_confidence.copy()),
            "warped_blend": torch.from_numpy(warped_blend.copy()),
            "alpha": torch.tensor(alpha, dtype=torch.float32),
            "sample_id": sample_id,
        }

        if arr_gt is not None:
            sample["gt"] = torch.from_numpy(arr_gt.copy())

        return sample


# ---------------------------------------------------------------------------
# Collate function (handles variable-size images if crop_size=None)
# ---------------------------------------------------------------------------

def collate_fn(batch: list) -> dict:
    """Default collate; strings are kept as a list."""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            result[k] = torch.stack(vals, dim=0)
        else:
            result[k] = vals  # strings etc.
    return result
