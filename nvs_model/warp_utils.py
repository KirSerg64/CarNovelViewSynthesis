"""
Differentiable image warping utilities.

All operations are PyTorch-differentiable, enabling gradient flow through warps
into the upstream flow/depth estimation networks.
"""

import torch
import torch.nn.functional as F


def warp_flow(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Backward-warp an image using a dense optical flow field.

    For each target pixel (x, y), samples from source pixel at (x + flow_x, y + flow_y).

    Args:
        img:  (B, C, H, W)  — image to warp, float32
        flow: (B, 2, H, W)  — displacement in pixels [dx, dy]

    Returns:
        warped: (B, C, H, W) — warped image
    """
    B, _C, H, W = img.shape
    device = img.device

    # Normalised base grid in [-1, 1]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    # Shift to [-1, 1] range
    grid_x = 2.0 * grid_x / max(W - 1, 1) - 1.0
    grid_y = 2.0 * grid_y / max(H - 1, 1) - 1.0
    base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

    # Normalise pixel displacements to [-1, 1] scale
    flow_norm = flow.permute(0, 2, 3, 1)  # (B, H, W, 2)
    flow_norm_x = flow_norm[..., 0] / max((W - 1) / 2.0, 1e-6)
    flow_norm_y = flow_norm[..., 1] / max((H - 1) / 2.0, 1e-6)
    flow_norm = torch.stack([flow_norm_x, flow_norm_y], dim=-1)

    sampling_grid = base_grid + flow_norm  # (B, H, W, 2)
    return F.grid_sample(
        img, sampling_grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def warp_flow_mask(flow: torch.Tensor) -> torch.Tensor:
    """
    Compute a validity mask for a warp: pixels whose sampling location
    falls outside [-1, 1] after normalisation are marked invalid.

    Args:
        flow: (B, 2, H, W)

    Returns:
        mask: (B, 1, H, W) float32 in {0, 1}
    """
    B, _two, H, W = flow.shape
    device = flow.device

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    grid_x = 2.0 * grid_x / max(W - 1, 1) - 1.0
    grid_y = 2.0 * grid_y / max(H - 1, 1) - 1.0
    base = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    flow_norm = flow.permute(0, 2, 3, 1)
    flow_norm_x = flow_norm[..., 0] / max((W - 1) / 2.0, 1e-6)
    flow_norm_y = flow_norm[..., 1] / max((H - 1) / 2.0, 1e-6)
    coords = base + torch.stack([flow_norm_x, flow_norm_y], dim=-1)

    valid = (
        (coords[..., 0] >= -1.0)
        & (coords[..., 0] <= 1.0)
        & (coords[..., 1] >= -1.0)
        & (coords[..., 1] <= 1.0)
    )  # (B, H, W)
    return valid.unsqueeze(1).float()


def depth_to_flow(
    depth_src: torch.Tensor,
    K_src: torch.Tensor,
    K_tgt: torch.Tensor,
    c2w_src: torch.Tensor,
    c2w_tgt: torch.Tensor,
) -> torch.Tensor:
    """
    Compute a pixel-level flow from source to target camera using depth.

    Useful as a geometric initialisation for the learned flow.

    Args:
        depth_src: (B, 1, H, W) depth in source camera (metres, 0 = invalid)
        K_src:     (B, 3, 3) source intrinsics
        K_tgt:     (B, 3, 3) target intrinsics
        c2w_src:   (B, 4, 4) source camera-to-world
        c2w_tgt:   (B, 4, 4) target camera-to-world

    Returns:
        flow: (B, 2, H, W) displacement from source pixel to target pixel
    """
    B, _, H, W = depth_src.shape
    device = depth_src.device

    # Pixel grid (u, v) in source
    v_grid, u_grid = torch.meshgrid(
        torch.arange(H, dtype=torch.float64, device=device),
        torch.arange(W, dtype=torch.float64, device=device),
        indexing="ij",
    )
    ones = torch.ones_like(u_grid)
    uvh = torch.stack([u_grid, v_grid, ones], dim=0)  # (3, H, W)
    uvh = uvh.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 3, H, W)
    uvh_flat = uvh.reshape(B, 3, H * W)  # (B, 3, N)

    # Unproject to source camera frame
    K_src_inv = torch.inverse(K_src.double())  # (B, 3, 3)
    rays = K_src_inv @ uvh_flat  # (B, 3, N)
    depth_flat = depth_src.double().reshape(B, 1, H * W)  # (B, 1, N)
    points_cam = rays * depth_flat  # (B, 3, N)

    # Transform to world frame
    R_src = c2w_src[:, :3, :3].double()  # (B, 3, 3)
    t_src = c2w_src[:, :3, 3:].double()  # (B, 3, 1)
    points_world = R_src @ points_cam + t_src  # (B, 3, N)

    # Transform to target camera frame
    w2c_tgt = torch.inverse(c2w_tgt.double())  # (B, 4, 4)
    R_tgt = w2c_tgt[:, :3, :3]
    t_tgt = w2c_tgt[:, :3, 3:]
    points_tgt = R_tgt @ points_world + t_tgt  # (B, 3, N)

    # Project to target image plane
    K_tgt_d = K_tgt.double()
    uv_tgt = K_tgt_d @ points_tgt  # (B, 3, N)
    z_tgt = uv_tgt[:, 2:3, :].clamp(min=1e-6)
    u_tgt = uv_tgt[:, 0, :] / z_tgt[:, 0, :]  # (B, N)
    v_tgt = uv_tgt[:, 1, :] / z_tgt[:, 0, :]  # (B, N)

    # Flow = target_pixel - source_pixel
    u_src = uvh_flat[:, 0, :]
    v_src = uvh_flat[:, 1, :]
    flow_x = u_tgt - u_src  # (B, N)
    flow_y = v_tgt - v_src

    # Mask invalid depths and out-of-frame projections
    valid = (depth_flat[:, 0, :] > 0) & (z_tgt[:, 0, :] > 0.1)
    flow_x = flow_x * valid.double()
    flow_y = flow_y * valid.double()

    flow = torch.stack([flow_x, flow_y], dim=1).reshape(B, 2, H, W)
    return flow.float()
