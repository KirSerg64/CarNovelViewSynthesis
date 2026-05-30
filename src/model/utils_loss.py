import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ---------------------------------------------------------------------------
# Flow smoothness
# ---------------------------------------------------------------------------

def flow_smoothness_loss(
    flow: torch.Tensor, img: torch.Tensor, edge_aware: bool = True
) -> torch.Tensor:
    """
    Edge-aware first-order flow smoothness loss.
    Penalises large spatial flow gradients unless they occur at image edges.

    flow: (B, 2, H, W)
    img:  (B, 3, H, W) reference image to derive edge weights from
    """
    dx_flow = flow[:, :, :, 1:] - flow[:, :, :, :-1]  # (B,2,H,W-1)
    dy_flow = flow[:, :, 1:, :] - flow[:, :, :-1, :]  # (B,2,H-1,W)

    if edge_aware:
        dx_img = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
        dy_img = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
        w_x = torch.exp(-10.0 * dx_img)  # (B,1,H,W-1)
        w_y = torch.exp(-10.0 * dy_img)  # (B,1,H-1,W)
        loss_x = (dx_flow.abs() * w_x).mean()
        loss_y = (dy_flow.abs() * w_y).mean()
    else:
        loss_x = dx_flow.abs().mean()
        loss_y = dy_flow.abs().mean()

    return (loss_x + loss_y) * 0.5


def _ms_ssim(
    pred: torch.Tensor,
    gt: torch.Tensor,
    weights: tuple = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
) -> torch.Tensor:
    """
    Multi-Scale SSIM (Wang et al., 2003).
    Both inputs: (B, C, H, W) in [0, 1].
    """
    msssim = torch.ones(1, device=pred.device, dtype=pred.dtype)
    for i, w in enumerate(weights):
        if i > 0:
            pred = F.avg_pool2d(pred, 2)
            gt = F.avg_pool2d(gt, 2)
        s = _ssim(pred, gt)
        if i < len(weights) - 1:
            msssim = msssim * (s ** w)
        else:
            msssim = msssim * (s ** w)
    return msssim


def _sobel(img: torch.Tensor) -> torch.Tensor:
    """
    Compute spatial gradients using Sobel filters.
    img: (B, C, H, W) in [0,1]
    Returns: magnitude (B, C, H, W)
    """
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=img.device
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    C = img.shape[1]
    kx = kx.expand(C, 1, 3, 3)
    ky = ky.expand(C, 1, 3, 3)
    gx = F.conv2d(img, kx, padding=1, groups=C)
    gy = F.conv2d(img, ky, padding=1, groups=C)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def _census_transform(img: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Compute ternary census transform of an image.
    img: (B, C, H, W) in [0, 1]
    Returns: (B, C*k*k, H, W) ternary values in {-1, 0, 1}
    """
    B, C, H, W = img.shape
    pad = kernel_size // 2
    unfolded = F.unfold(img, kernel_size, padding=pad)   # (B, C*k*k, H*W)
    center_idx = (kernel_size ** 2) // 2
    patch_size = kernel_size ** 2

    center = unfolded[:, center_idx::patch_size, :]      # (B, C, H*W)  — wrong stride
    # Re-index correctly: for each channel c, the patch pixels are at
    # [c*patch_size : (c+1)*patch_size]
    center_vals = []
    patches = []
    for c in range(C):
        patch_c = unfolded[:, c * patch_size:(c + 1) * patch_size, :]  # (B, k*k, H*W)
        center_c = patch_c[:, center_idx:center_idx + 1, :]             # (B, 1, H*W)
        center_vals.append(center_c.expand_as(patch_c))
        patches.append(patch_c)

    patches = torch.cat(patches, dim=1)       # (B, C*k*k, H*W)
    centers = torch.cat(center_vals, dim=1)   # (B, C*k*k, H*W)

    threshold = 0.02  # tolerance before ternary comparison
    diff = patches - centers
    census = torch.zeros_like(diff)
    census[diff > threshold] = 1.0
    census[diff < -threshold] = -1.0

    return census.reshape(B, C * patch_size, H, W)


def _census_loss(pred: torch.Tensor, gt: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Hamming distance between census transforms (normalised)."""
    c_pred = _census_transform(pred, kernel_size)
    c_gt = _census_transform(gt, kernel_size)
    # Soft Hamming: count of positions where signs differ
    diff = (c_pred - c_gt).abs()  # in [0, 2]
    return diff.mean()

def _gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """1D Gaussian kernel, normalised."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()

def _gaussian_kernel_2d(size: int, sigma: float, channels: int, device: torch.device):
    """Returns (channels, 1, size, size) separable Gaussian conv kernel."""
    g1d = _gaussian_kernel(size, sigma, device)
    g2d = g1d.unsqueeze(0) * g1d.unsqueeze(1)  # (size, size)
    g2d = g2d.unsqueeze(0).unsqueeze(0)        # (1, 1, size, size)
    return g2d.expand(channels, 1, size, size)  # (C, 1, size, size)

def _ssim(
    pred: torch.Tensor,
    gt: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    Compute SSIM map between pred and gt, returns the mean SSIM scalar.
    Both inputs: (B, C, H, W) in [0, data_range].
    """
    C = pred.shape[1]
    kernel = _gaussian_kernel_2d(window_size, sigma, C, pred.device)
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(gt, kernel, padding=pad, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, kernel, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * gt, kernel, padding=pad, groups=C) - mu12

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    num = (2 * mu12 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / den.clamp(min=1e-8)
    return ssim_map.mean()


