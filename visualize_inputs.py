"""
Visualise the first-stage network inputs for one or more dataset samples.

For each sample the script extracts and displays six fields that are packed
inside `input_tensor`:

    depth_t0    — normalised LiDAR depth at time t0   (channel 6)
    depth_t1    — normalised LiDAR depth at time t1   (channel 7)
    warped_t0   — geometric warp from t0 to target    (channels 8-10, RGB)
    warped_t1   — geometric warp from t1 to target    (channels 11-13, RGB)
    warp_mask_t0 — validity mask for warped_t0        (channel 14)
    warp_mask_t1 — validity mask for warped_t1        (channel 15)

Channel layout (18 channels total):
    0-2   img_t0
    3-5   img_t1
    6     depth_t0
    7     depth_t1
    8-10  warped_t0
    11-13 warped_t1
    14    warp_mask_t0
    15    warp_mask_t1
    16    target_depth
    17    alpha

Usage
-----
    # Visualise the first 3 samples from the training set (saves PNG files):
    python visualize_inputs.py --data-dir data/train --n-samples 3

    # Show interactively instead of saving:
    python visualize_inputs.py --data-dir data/train --n-samples 1 --show

    # Use a specific output directory:
    python visualize_inputs.py --data-dir data/train --out-dir my_viz

    # Disable crop so full-resolution images are shown:
    python visualize_inputs.py --data-dir data/train --no-crop
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chw_to_hwc(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) tensor → (H, W, C) float32 numpy array clipped to [0, 1]."""
    return t.permute(1, 2, 0).float().numpy().clip(0.0, 1.0)


def _chw1_to_hw(t: torch.Tensor) -> np.ndarray:
    """(1, H, W) tensor → (H, W) float32 numpy array."""
    return t[0].float().numpy()


def visualize_sample(sample: dict, sample_idx: int, out_dir: Path, show: bool) -> None:
    """
    Build a 4×2 grid figure for one dataset sample and either display it or
    save it to `out_dir`.

    Grid layout:
        [depth_t0]      [depth_t1]
        [warped_t0]     [warped_t1]
        [warp_mask_t0]  [warp_mask_t1]
        [target_depth]  [<empty>]
    """
    it = sample["input_tensor"]  # (18, H, W)
    sample_id = sample["sample_id"] if isinstance(sample["sample_id"], str) \
        else sample["sample_id"][0]

    depth_t0     = _chw1_to_hw(it[6:7])    # (H, W)
    depth_t1     = _chw1_to_hw(it[7:8])
    warped_t0    = _chw_to_hwc(it[8:11])   # (H, W, 3)
    warped_t1    = _chw_to_hwc(it[11:14])
    warp_mask_t0 = _chw1_to_hw(it[14:15])  # (H, W)
    warp_mask_t1 = _chw1_to_hw(it[15:16])
    target_depth = _chw1_to_hw(it[16:17])  # (H, W)

    fig, axes = plt.subplots(4, 2, figsize=(12, 18))
    fig.suptitle(f"Sample {sample_idx}: {sample_id}", fontsize=11, y=1.01)

    # Row 0: depth maps (plasma colormap, range [0, 1])
    im0 = axes[0, 0].imshow(depth_t0, cmap="plasma", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("depth_t0 (normalised)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(depth_t1, cmap="plasma", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("depth_t1 (normalised)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Row 1: geometric warps (RGB images)
    axes[1, 0].imshow(warped_t0)
    axes[1, 0].set_title("warped_t0 → target")

    axes[1, 1].imshow(warped_t1)
    axes[1, 1].set_title("warped_t1 → target")

    # Row 2: warp validity masks (binary, 0/1)
    im2 = axes[2, 0].imshow(warp_mask_t0, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2, 0].set_title("warp_mask_t0")
    fig.colorbar(im2, ax=axes[2, 0], fraction=0.046, pad=0.04)

    im3 = axes[2, 1].imshow(warp_mask_t1, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2, 1].set_title("warp_mask_t1")
    fig.colorbar(im3, ax=axes[2, 1], fraction=0.046, pad=0.04)

    # Row 3: target depth (plasma colormap, range [0, 1])
    im4 = axes[3, 0].imshow(target_depth, cmap="plasma", vmin=0.0, vmax=1.0)
    axes[3, 0].set_title("target_depth (normalised)")
    fig.colorbar(im4, ax=axes[3, 0], fraction=0.046, pad=0.04)

    axes[3, 1].set_visible(False)  # unused cell

    for ax in axes.flat:
        ax.axis("off")

    fig.tight_layout()

    if show:
        plt.show()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_id = sample_id.replace("/", "_").replace(" ", "_")
        out_path = out_dir / f"sample_{sample_idx:04d}_{safe_id}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"  Saved → {out_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualise NVS network inputs (depth maps, warps, masks)."
    )
    parser.add_argument(
        "--data-dir", default="data/train",
        help="Path to the dataset split directory (default: data/train)."
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Optional path to pre-computed cache directory."
    )
    parser.add_argument(
        "--n-samples", type=int, default=3,
        help="Number of samples to visualise (default: 3)."
    )
    parser.add_argument(
        "--no-crop", action="store_true",
        help="Disable random crop so full-resolution images are shown."
    )
    parser.add_argument(
        "--crop-size", type=int, nargs=2, default=[512, 512],
        metavar=("H", "W"),
        help="Crop size in pixels (default: 512 512). Ignored when --no-crop is set."
    )
    parser.add_argument(
        "--out-dir", default="viz_inputs",
        help="Output directory for PNG files (default: viz_inputs)."
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display figures interactively instead of saving to disk."
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Disable data augmentation (flip / jitter) for cleaner visuals."
    )
    args = parser.parse_args()

    # Import here so the script fails clearly if dependencies are missing
    from nvs_model.dataset import NVSDataset

    crop_size = None if args.no_crop else tuple(args.crop_size)

    dataset = NVSDataset(
        data_dir=Path(args.data_dir),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        crop_size=crop_size,
        augment=not args.no_augment,
        is_train=True,
    )

    n = min(args.n_samples, len(dataset))
    if n == 0:
        print("No samples found in the dataset.")
        return

    print(f"Visualising {n} sample(s) from '{args.data_dir}' …")

    if args.show:
        matplotlib.use("TkAgg")  # change to Qt5Agg / Agg as needed

    out_dir = Path(args.out_dir)

    for i in range(n):
        print(f"[{i + 1}/{n}] Loading sample index {i} …")
        sample = dataset[i]
        visualize_sample(sample, sample_idx=i, out_dir=out_dir, show=args.show)

    if not args.show:
        print(f"\nDone. {n} figure(s) saved to '{out_dir}/'.")


if __name__ == "__main__":
    main()
