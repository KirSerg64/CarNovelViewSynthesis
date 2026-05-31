"""
Evaluate predictions against ground truth for Novel View Synthesis.

Computes PSNR and normalized score per sample and overall.

Usage:
    python evaluate.py --pred-dir submission --gt-dir data/train
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute PSNR between predicted and ground truth images."""
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def compute_normalized_score(psnr: float) -> float:
    """Normalize PSNR to [0, 100] range as per competition rules."""
    clamped = np.clip(psnr, 10.0, 30.0)
    return (clamped - 10.0) / 20.0 * 100.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate NVS predictions")
    parser.add_argument(
        "--pred-dir", type=Path, required=True,
        help="Directory with predictions (each sample_id/pred.jpg)"
    )
    parser.add_argument(
        "--gt-dir", type=Path, required=True,
        help="Dataset directory with ground truth (each sample_id/target/<cam>.jpg)"
    )
    args = parser.parse_args()

    results = []
    sample_dirs = sorted(p for p in args.gt_dir.iterdir() if p.is_dir())

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        pred_path = args.pred_dir / sample_id / "pred.jpg"

        if not pred_path.exists():
            print(f"  SKIP {sample_id}: no prediction found")
            continue

        # Get target camera name from meta
        meta = json.loads((sample_dir / "meta.json").read_text())
        target_camera = meta["target_camera"]
        gt_path = sample_dir / "target" / f"{target_camera}.jpg"

        if not gt_path.exists():
            print(f"  SKIP {sample_id}: no ground truth")
            continue

        pred = np.array(Image.open(pred_path))
        gt = np.array(Image.open(gt_path))

        # Handle size mismatch
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        psnr = compute_psnr(pred, gt)
        score = compute_normalized_score(psnr)
        results.append({"sample_id": sample_id, "psnr": psnr, "score": score})
        print(f"  {sample_id}: PSNR={psnr:.2f} dB, Score={score:.1f}")

    if results:
        psnrs = [r["psnr"] for r in results]
        scores = [r["score"] for r in results]
        print(f"\n{'='*60}")
        print(f"Summary ({len(results)} samples):")
        print(f"  Mean PSNR:   {np.mean(psnrs):.2f} dB")
        print(f"  Median PSNR: {np.median(psnrs):.2f} dB")
        print(f"  Min PSNR:    {np.min(psnrs):.2f} dB")
        print(f"  Max PSNR:    {np.max(psnrs):.2f} dB")
        print(f"  Mean Score:  {np.mean(scores):.1f} / 100")
        print(f"{'='*60}")
    else:
        print("No samples evaluated.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
