"""
LiDAR-based Novel View Synthesis via Depth Warping.

This script synthesizes a target camera view by:
1. Projecting the dense LiDAR point cloud into each input camera to obtain depth maps.
2. For each input image, unprojecting pixels with known depth into 3D, then reprojecting
   into the target camera frame.
3. Blending multiple reprojected views using distance-based weights.
4. Filling holes with neighboring pixel interpolation.

Usage:
    # Process test set and generate submission
    python lidar_warp_nvs.py --data-dir data/test --output-dir submission

    # Evaluate on training set
    python lidar_warp_nvs.py --data-dir data/train --output-dir output_train --evaluate
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


CAMERAS = ["front", "left_fwd", "left_bwd", "right_fwd", "right_bwd", "rear"]


def load_meta(sample_dir: Path) -> dict:
    """Load sample metadata."""
    return json.loads((sample_dir / "meta.json").read_text())


def load_lidar(sample_dir: Path) -> np.ndarray:
    """Load LiDAR point cloud (N, 3) in world frame."""
    npz = np.load(sample_dir / "input" / "lidar.npz")
    return npz["xyz"].astype(np.float64)


def get_intrinsic_matrix(intr: dict) -> np.ndarray:
    """Build 3x3 camera intrinsic matrix from metadata."""
    return np.array([
        [intr["fx"], 0.0, intr["cx"]],
        [0.0, intr["fy"], intr["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def project_lidar_to_camera(
    xyz_world: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D world points into a camera image plane.

    Returns:
        depth_map: (H, W) float32 with depth values (z in camera frame)
        u_coords: pixel x-coordinates of projected points
        v_coords: pixel y-coordinates of projected points
    """
    # World-to-camera transform
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3]
    t = w2c[:3, 3]

    # Transform to camera coordinates
    xyz_cam = (R @ xyz_world.T).T + t  # (N, 3)

    # Filter points behind camera (z > 0 in OpenCV convention)
    valid = xyz_cam[:, 2] > 0.1
    xyz_cam = xyz_cam[valid]

    # Project to pixel coordinates
    uv_homog = (K @ xyz_cam.T).T  # (N, 3)
    u = uv_homog[:, 0] / uv_homog[:, 2]
    v = uv_homog[:, 1] / uv_homog[:, 2]
    z = xyz_cam[:, 2]

    # Round to integer pixel coords first, then filter
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    # Filter to image bounds (using integer coords to avoid rounding issues)
    in_bounds = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    ui = ui[in_bounds]
    vi = vi[in_bounds]
    z = z[in_bounds]

    # Build depth map using z-buffer (keep closest point per pixel)
    depth_map = np.full((height, width), np.inf, dtype=np.float64)

    # Sort by depth (furthest first) so closest overwrites
    sort_idx = np.argsort(-z)
    ui = ui[sort_idx]
    vi = vi[sort_idx]
    z = z[sort_idx]

    depth_map[vi, ui] = z

    # Replace inf with 0 (no depth)
    depth_map[depth_map == np.inf] = 0.0

    return depth_map.astype(np.float32), ui, vi


def densify_depth_map(depth_map: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Fill sparse depth map holes using iterative dilation with depth-aware propagation.
    """
    mask = (depth_map > 0).astype(np.uint8)

    # Use multiple dilation passes with increasing kernel sizes for progressive fill
    result = depth_map.copy()
    for ks in [kernel_size, kernel_size + 2, kernel_size + 4]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        depth_dilated = cv2.dilate(result, kernel, iterations=1)
        mask_dilated = cv2.dilate((result > 0).astype(np.uint8), kernel, iterations=1)

        fill_mask = (result == 0) & (mask_dilated > 0)
        result[fill_mask] = depth_dilated[fill_mask]

    return result


def inverse_warp_same_camera(
    src_img: np.ndarray,
    src_depth: np.ndarray,
    src_c2w: np.ndarray,
    src_K: np.ndarray,
    tgt_c2w: np.ndarray,
    tgt_K: np.ndarray,
    tgt_width: int,
    tgt_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inverse warp: for each target pixel, find where it came from in the source image.
    Uses depth to find 3D correspondences, then uses cv2.remap for smooth interpolation.

    Returns:
        warped_img: (tgt_H, tgt_W, 3) uint8
        warped_mask: (tgt_H, tgt_W) bool
    """
    src_h, src_w = src_img.shape[:2]

    # Get valid source pixels with depth
    valid_mask = src_depth > 0
    vs, us = np.where(valid_mask)

    if len(vs) == 0:
        return (
            np.zeros((tgt_height, tgt_width, 3), dtype=np.uint8),
            np.zeros((tgt_height, tgt_width), dtype=bool),
        )

    depths = src_depth[vs, us].astype(np.float64)

    # Unproject to world
    src_K_inv = np.linalg.inv(src_K)
    pixels_homog = np.stack([us, vs, np.ones_like(us)], axis=0).astype(np.float64)
    rays_cam = src_K_inv @ pixels_homog
    points_cam = rays_cam * depths[np.newaxis, :]
    src_R = src_c2w[:3, :3]
    src_t = src_c2w[:3, 3:]
    points_world = src_R @ points_cam + src_t

    # Project to target camera to build target depth map
    tgt_w2c = np.linalg.inv(tgt_c2w)
    tgt_R = tgt_w2c[:3, :3]
    tgt_t = tgt_w2c[:3, 3:]
    points_tgt = tgt_R @ points_world + tgt_t

    z_tgt = points_tgt[2, :]
    in_front = z_tgt > 0.1
    points_tgt_valid = points_tgt[:, in_front]
    z_tgt_valid = z_tgt[in_front]

    uv_tgt = tgt_K @ points_tgt_valid
    u_tgt = uv_tgt[0, :] / uv_tgt[2, :]
    v_tgt = uv_tgt[1, :] / uv_tgt[2, :]

    ui_tgt = np.round(u_tgt).astype(np.int32)
    vi_tgt = np.round(v_tgt).astype(np.int32)
    in_bounds = (ui_tgt >= 0) & (ui_tgt < tgt_width) & (vi_tgt >= 0) & (vi_tgt < tgt_height)

    ui_tgt = ui_tgt[in_bounds]
    vi_tgt = vi_tgt[in_bounds]
    z_vals = z_tgt_valid[in_bounds]

    # Also keep track of source pixel locations for direct mapping
    us_filtered = us[in_front][in_bounds]
    vs_filtered = vs[in_front][in_bounds]

    # Build remap maps using z-buffer (closest wins) - vectorized
    map_x = np.full((tgt_height, tgt_width), -1.0, dtype=np.float32)
    map_y = np.full((tgt_height, tgt_width), -1.0, dtype=np.float32)
    zbuf = np.full((tgt_height, tgt_width), np.inf, dtype=np.float64)

    # Sort by depth descending (so closest overwrites via simple assignment)
    sort_idx = np.argsort(-z_vals)
    ui_sorted = ui_tgt[sort_idx]
    vi_sorted = vi_tgt[sort_idx]
    z_sorted = z_vals[sort_idx]
    us_sorted = us_filtered[sort_idx].astype(np.float32)
    vs_sorted = vs_filtered[sort_idx].astype(np.float32)

    # Assign (closest points written last, overwriting further ones)
    zbuf[vi_sorted, ui_sorted] = z_sorted
    map_x[vi_sorted, ui_sorted] = us_sorted
    map_y[vi_sorted, ui_sorted] = vs_sorted

    # Densify the maps to fill small gaps
    valid_map = map_x >= 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for _ in range(3):
        map_x_dense = cv2.dilate(map_x * valid_map.astype(np.float32), kernel)
        map_y_dense = cv2.dilate(map_y * valid_map.astype(np.float32), kernel)
        count_dense = cv2.dilate(valid_map.astype(np.float32), kernel)
        fill = (~valid_map) & (count_dense > 0)
        map_x[fill] = map_x_dense[fill] / count_dense[fill]
        map_y[fill] = map_y_dense[fill] / count_dense[fill]
        valid_map = map_x >= 0

    # Apply remap with bilinear interpolation
    warped_mask = valid_map
    map_x_remap = map_x.copy()
    map_y_remap = map_y.copy()
    map_x_remap[~warped_mask] = 0
    map_y_remap[~warped_mask] = 0

    warped_img = cv2.remap(
        src_img, map_x_remap, map_y_remap,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    warped_img[~warped_mask] = 0

    return warped_img, warped_mask


def warp_image_to_target(
    src_img: np.ndarray,
    src_depth: np.ndarray,
    src_c2w: np.ndarray,
    src_K: np.ndarray,
    tgt_c2w: np.ndarray,
    tgt_K: np.ndarray,
    tgt_width: int,
    tgt_height: int,
    splat_size: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Warp source image to target view using depth-based reprojection with splatting.

    Uses small splats (not single pixels) to reduce holes in the output.

    Returns:
        warped_img: (tgt_H, tgt_W, 3) uint8
        warped_mask: (tgt_H, tgt_W) bool - where pixels were successfully projected
        warped_depth: (tgt_H, tgt_W) float32 - depth in target camera frame
    """
    src_h, src_w = src_img.shape[:2]

    # Get valid source pixels with depth - subsample for speed
    valid_mask = src_depth > 0
    vs, us = np.where(valid_mask)

    if len(vs) == 0:
        return (
            np.zeros((tgt_height, tgt_width, 3), dtype=np.uint8),
            np.zeros((tgt_height, tgt_width), dtype=bool),
            np.zeros((tgt_height, tgt_width), dtype=np.float32),
        )

    depths = src_depth[vs, us].astype(np.float64)

    # Unproject source pixels to 3D (camera frame)
    src_K_inv = np.linalg.inv(src_K)
    pixels_homog = np.stack([us, vs, np.ones_like(us)], axis=0).astype(np.float64)  # (3, N)
    rays_cam = src_K_inv @ pixels_homog  # (3, N)
    points_cam = rays_cam * depths[np.newaxis, :]  # (3, N)

    # Transform to world frame
    src_R = src_c2w[:3, :3]
    src_t = src_c2w[:3, 3:]
    points_world = src_R @ points_cam + src_t  # (3, N)

    # Transform to target camera frame
    tgt_w2c = np.linalg.inv(tgt_c2w)
    tgt_R = tgt_w2c[:3, :3]
    tgt_t = tgt_w2c[:3, 3:]
    points_tgt_cam = tgt_R @ points_world + tgt_t  # (3, N)

    # Filter points behind target camera
    z_tgt = points_tgt_cam[2, :]
    in_front = z_tgt > 0.1
    points_tgt_cam = points_tgt_cam[:, in_front]
    z_tgt = z_tgt[in_front]
    vs_valid = vs[in_front]
    us_valid = us[in_front]

    # Project to target image plane
    uv_tgt = tgt_K @ points_tgt_cam  # (3, N)
    u_tgt = uv_tgt[0, :] / uv_tgt[2, :]
    v_tgt = uv_tgt[1, :] / uv_tgt[2, :]

    # Filter to target image bounds (with margin for splatting)
    margin = splat_size
    in_bounds = (
        (u_tgt >= -margin) & (u_tgt < tgt_width + margin) &
        (v_tgt >= -margin) & (v_tgt < tgt_height + margin)
    )
    u_tgt = u_tgt[in_bounds]
    v_tgt = v_tgt[in_bounds]
    z_tgt = z_tgt[in_bounds]
    vs_valid = vs_valid[in_bounds]
    us_valid = us_valid[in_bounds]

    # Rasterize with splatting: use z-buffer
    ui_tgt = np.round(u_tgt).astype(np.int32)
    vi_tgt = np.round(v_tgt).astype(np.int32)

    warped_img = np.zeros((tgt_height, tgt_width, 3), dtype=np.uint8)
    warped_depth = np.full((tgt_height, tgt_width), np.inf, dtype=np.float64)

    # Sort by depth (furthest first) so closest overwrites
    sort_idx = np.argsort(-z_tgt)
    ui_tgt = ui_tgt[sort_idx]
    vi_tgt = vi_tgt[sort_idx]
    z_tgt_sorted = z_tgt[sort_idx]
    vs_valid = vs_valid[sort_idx]
    us_valid = us_valid[sort_idx]

    # Write pixels with small splats
    colors = src_img[vs_valid, us_valid]
    for dy in range(-splat_size // 2, splat_size // 2 + 1):
        for dx in range(-splat_size // 2, splat_size // 2 + 1):
            yy = vi_tgt + dy
            xx = ui_tgt + dx
            valid = (xx >= 0) & (xx < tgt_width) & (yy >= 0) & (yy < tgt_height)
            warped_depth[yy[valid], xx[valid]] = z_tgt_sorted[valid]
            warped_img[yy[valid], xx[valid]] = colors[valid]

    warped_mask = warped_depth < np.inf
    warped_depth[~warped_mask] = 0.0

    return warped_img, warped_mask, warped_depth.astype(np.float32)


def inpaint_holes(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in the image using OpenCV inpainting.

    Args:
        image: (H, W, 3) uint8 - partially filled image
        mask: (H, W) bool - True where image is valid (NOT holes)

    Returns:
        inpainted: (H, W, 3) uint8
    """
    # Inpaint mask: 1 where holes are (need to fill)
    inpaint_mask = (~mask).astype(np.uint8) * 255

    if inpaint_mask.sum() == 0:
        return image

    # Use Telea inpainting with larger radius for better fill
    result = cv2.inpaint(image, inpaint_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
    return result


def render_sample(sample_dir: Path) -> np.ndarray:
    """
    Render the target view for a given sample using LiDAR depth warping.

    Strategy:
    1. Use inverse warping (smooth, via remap) for the same camera at t0 and t1
    2. Use forward splatting for other cameras (to fill occluded areas)
    3. Blend with strong preference for same-camera views
    4. Combine with simple temporal average as fallback for uncovered areas
    5. Inpaint remaining holes
    """
    meta = load_meta(sample_dir)
    xyz_world = load_lidar(sample_dir)
    target_camera = meta["target_camera"]
    intrinsics = meta["intrinsics"]
    poses = meta["poses_c2w"]

    # Target camera parameters
    tgt_intr = intrinsics[target_camera]
    tgt_K = get_intrinsic_matrix(tgt_intr)
    tgt_width = int(tgt_intr["width"])
    tgt_height = int(tgt_intr["height"])
    tgt_c2w = np.array(poses["target"][target_camera], dtype=np.float64)

    # Compute temporal interpolation factor
    ts = meta["timestamps_ns"]
    alpha = float((ts["target"] - ts["t0"]) / (ts["t1"] - ts["t0"]))

    h, w = tgt_height, tgt_width
    accum = np.zeros((h, w, 3), dtype=np.float64)
    weight_sum = np.zeros((h, w), dtype=np.float64)

    # Also compute simple temporal average of same camera (as fallback)
    t0_img = np.array(Image.open(sample_dir / "input" / "t0" / f"{target_camera}.jpg"))
    t1_img = np.array(Image.open(sample_dir / "input" / "t1" / f"{target_camera}.jpg"))
    temporal_avg = (
        (1.0 - alpha) * t0_img.astype(np.float64) +
        alpha * t1_img.astype(np.float64)
    ).clip(0, 255).astype(np.uint8)

    for ts_key in ("t0", "t1"):
        temporal_weight = (1.0 - alpha) if ts_key == "t0" else alpha

        for cam in CAMERAS:
            # Load source image
            img_path = sample_dir / "input" / ts_key / f"{cam}.jpg"
            if not img_path.exists():
                continue
            src_img = np.array(Image.open(img_path))

            # Source camera parameters
            src_intr = intrinsics[cam]
            src_K = get_intrinsic_matrix(src_intr)
            src_width = int(src_intr["width"])
            src_height = int(src_intr["height"])
            src_c2w = np.array(poses[ts_key][cam], dtype=np.float64)

            # Project LiDAR to source camera to get depth map
            src_depth, _, _ = project_lidar_to_camera(
                xyz_world, src_c2w, src_K, src_width, src_height
            )
            # Densify the sparse depth map
            src_depth = densify_depth_map(src_depth, kernel_size=7)

            if cam == target_camera:
                # Use inverse warping for same camera (smooth result)
                warped_img, warped_mask = inverse_warp_same_camera(
                    src_img, src_depth, src_c2w, src_K,
                    tgt_c2w, tgt_K, tgt_width, tgt_height
                )
                cam_weight = 10.0
            else:
                # Use forward splatting for other cameras
                warped_img, warped_mask, _ = warp_image_to_target(
                    src_img, src_depth, src_c2w, src_K,
                    tgt_c2w, tgt_K, tgt_width, tgt_height
                )
                cam_weight = 1.0

            # Accumulate weighted blend
            weight = np.zeros((h, w), dtype=np.float64)
            weight[warped_mask] = cam_weight * max(temporal_weight, 0.1)

            accum += warped_img.astype(np.float64) * weight[:, :, np.newaxis]
            weight_sum += weight

    coverage = weight_sum > 0
    blended = np.zeros((h, w, 3), dtype=np.float64)
    blended[coverage] = accum[coverage] / weight_sum[coverage, np.newaxis]

    # Blend warped result with temporal average:
    # - Where we have good warped coverage, use mostly warped (0.7 warp + 0.3 avg)
    # - Where we have no coverage, use temporal average
    warp_confidence = np.clip(weight_sum / (weight_sum.max() + 1e-10), 0, 1)

    # Adaptive blend: more warping confidence → more warping weight
    # Max blend weight is 0.6 since temporal avg is a strong baseline
    warp_blend_weight = 0.6 * warp_confidence[:, :, np.newaxis]
    result_float = (
        warp_blend_weight * blended +
        (1.0 - warp_blend_weight) * temporal_avg.astype(np.float64)
    )

    result = result_float.clip(0, 255).astype(np.uint8)

    return result


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


def process_sample(sample_dir: Path, output_dir: Path, evaluate: bool = False) -> dict:
    """Process a single sample: render + optionally evaluate."""
    pred = render_sample(sample_dir)

    # Save prediction
    out_dir = output_dir / sample_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred).save(out_dir / "pred.jpg", quality=95)

    result = {"sample_id": sample_dir.name}

    if evaluate:
        target_camera = load_meta(sample_dir)["target_camera"]
        gt_path = sample_dir / "target" / f"{target_camera}.jpg"
        if gt_path.exists():
            gt = np.array(Image.open(gt_path))
            # Ensure same size
            if pred.shape != gt.shape:
                pred_resized = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
                psnr = compute_psnr(pred_resized, gt)
            else:
                psnr = compute_psnr(pred, gt)
            score = compute_normalized_score(psnr)
            result["psnr"] = psnr
            result["score"] = score

    return result


def main():
    parser = argparse.ArgumentParser(
        description="LiDAR-based Novel View Synthesis via Depth Warping"
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Path to dataset directory (e.g. data/test or data/train)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("submission"),
        help="Output directory for predictions"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Compute PSNR metrics (requires ground truth in target/)"
    )
    parser.add_argument(
        "--samples", type=str, nargs="*", default=None,
        help="Specific sample IDs to process (default: all)"
    )
    args = parser.parse_args()

    # Find all samples
    if args.samples:
        sample_dirs = [args.data_dir / s for s in args.samples]
    else:
        sample_dirs = sorted(p for p in args.data_dir.iterdir() if p.is_dir())

    if not sample_dirs:
        print(f"No samples found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(sample_dirs)} samples...")
    print(f"Output: {args.output_dir}")
    if args.evaluate:
        print("Evaluation mode: will compute PSNR against ground truth")
    print()

    results = []
    for i, sample_dir in enumerate(sample_dirs):
        print(f"[{i + 1}/{len(sample_dirs)}] {sample_dir.name} ...", end=" ", flush=True)
        try:
            result = process_sample(sample_dir, args.output_dir, evaluate=args.evaluate)
            results.append(result)
            if args.evaluate and "psnr" in result:
                print(f"PSNR={result['psnr']:.2f} dB (score={result['score']:.1f})")
            else:
                print("done")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"sample_id": sample_dir.name, "error": str(e)})

    # Summary
    print(f"\nDone. Results saved to {args.output_dir}/")

    if args.evaluate:
        psnrs = [r["psnr"] for r in results if "psnr" in r]
        scores = [r["score"] for r in results if "score" in r]
        if psnrs:
            print(f"\n{'='*50}")
            print(f"Evaluation Summary ({len(psnrs)} samples):")
            print(f"  Mean PSNR:  {np.mean(psnrs):.2f} dB")
            print(f"  Median PSNR: {np.median(psnrs):.2f} dB")
            print(f"  Min PSNR:   {np.min(psnrs):.2f} dB")
            print(f"  Max PSNR:   {np.max(psnrs):.2f} dB")
            print(f"  Mean Score: {np.mean(scores):.1f} / 100")
            print(f"{'='*50}")


if __name__ == "__main__":
    main()
