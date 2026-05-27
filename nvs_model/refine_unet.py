"""
Model 2: RefineUNet — Fine-Detail Correction and Hole Inpainting.

Architecture: U-Net encoder-decoder with:
  - Partial Convolutions in the first encoder layer (respects warp validity masks)
  - CBAM (Channel + Spatial) attention at each decoder block
  - ResNet-style skip connections
  - Output: additive RGB residual Δ_fine

Input channels (14 total):
  I_coarse        : 3   (output from GeometricFlowNet)
  residual_diff   : 3   (I_coarse − warped_geometric_blend — highlights holes)
  target_depth    : 1   (normalised, 0 = invalid)
  warp_confidence : 1   (blend map from Model 1, range [0,1])
  img_t0          : 3   (original reference frame)
  img_t1          : 3   (original reference frame)

Final output: I_final = clip(I_coarse + Δ_fine, 0, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Partial Convolution
# ---------------------------------------------------------------------------

class PartialConv2d(nn.Module):
    """
    Partial Convolution layer from "Image Inpainting for Irregular Holes
    Using Partial Convolutions" (Liu et al., 2018).

    Performs a masked convolution where only valid (unmasked) pixels
    contribute, and the output is re-normalised by the number of valid inputs.

    Args:
        in_channels, out_channels, kernel_size, stride, padding — same as Conv2d
        multi_channel: if True, per-channel masks; else single-channel mask broadcast
        return_mask: if True, forward() returns (out, updated_mask)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        multi_channel: bool = False,
        return_mask: bool = True,
    ):
        super().__init__()
        self.return_mask = return_mask
        self.multi_channel = multi_channel
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)

        mask_in = in_channels if multi_channel else 1
        self.register_buffer(
            "mask_weight",
            torch.ones(out_channels, mask_in, kernel_size, kernel_size),
        )

        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x:    (B, C, H, W) input (invalid/hole pixels can be any value)
            mask: (B, 1, H, W) or (B, C, H, W) float32 — 1=valid, 0=hole

        Returns:
            out:         (B, out_ch, H', W')
            updated_mask (only if return_mask=True): (B, 1, H', W')
        """
        if mask is None:
            mask = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)

        if self.multi_channel:
            mask_ch = mask.expand_as(x)
        else:
            mask_ch = mask.expand(x.shape[0], 1, x.shape[2], x.shape[3])

        # Masked input
        x_masked = x * (mask_ch if self.multi_channel else mask_ch)

        # Convolution on masked input
        out = self.conv(x_masked)

        # Compute sum of valid kernel entries for re-normalisation
        with torch.no_grad():
            mask_in = mask_ch[:, :1, :, :] if not self.multi_channel else mask_ch
            mask_sum = F.conv2d(
                mask_in,
                self.mask_weight[:1, :1, :, :] if not self.multi_channel else self.mask_weight,
                stride=self.stride,
                padding=self.padding,
            )
            # Scale = total_kernel_area / valid_kernel_area (clamp to avoid /0)
            kernel_area = self.kernel_size ** 2
            scale = kernel_area / (mask_sum.clamp(min=1e-6))
            new_mask = (mask_sum > 0).float()

        out = out * scale

        if self.return_mask:
            return out, new_mask
        return out


# ---------------------------------------------------------------------------
# CBAM Attention
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        mid = max(ch // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        avg = x.mean(dim=[2, 3])               # (B, C)
        mx = x.amax(dim=[2, 3])               # (B, C)
        attn = self.sigmoid(self.fc(avg) + self.fc(mx))  # (B, C)
        return x * attn.view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)      # (B, 1, H, W)
        mx, _ = x.max(dim=1, keepdim=True)     # (B, 1, H, W)
        attn = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., 2018)."""

    def __init__(self, ch: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(ch, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        groups = min(8, out_ch)
        while out_ch % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Strided conv + conv (with group norm + relu) + optional CBAM."""

    def __init__(self, in_ch: int, out_ch: int, use_cbam: bool = False):
        super().__init__()
        self.down = ConvBNReLU(in_ch, out_ch, stride=2)
        self.conv = ConvBNReLU(out_ch, out_ch)
        self.attn = CBAM(out_ch) if use_cbam else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.conv(self.down(x)))


class DecoderBlock(nn.Module):
    """Bilinear upsample + skip concat + conv + CBAM."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)
        self.attn = CBAM(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(self.conv(x))
        return self.attn(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class RefineUNet(nn.Module):
    """
    Refinement network that corrects the coarse NVS prediction.

    Takes a 14-channel input, outputs a 3-channel RGB residual Δ_fine.
    The caller adds Δ_fine to I_coarse and clips to [0, 1].
    """

    IN_CH = 14

    def __init__(self, base_ch: int = 32):
        super().__init__()
        c = base_ch

        # First layer: Partial Convolution (respects hole mask in warp_confidence)
        self.pconv1 = PartialConv2d(
            self.IN_CH, c, kernel_size=7, stride=1, padding=3, return_mask=True
        )
        self.pnorm1 = nn.GroupNorm(min(8, c), c)
        self.prelu1 = nn.ReLU(inplace=True)

        # Second partial conv before first encoder stride
        self.pconv2 = PartialConv2d(c, c, kernel_size=3, stride=1, padding=1, return_mask=True)
        self.pnorm2 = nn.GroupNorm(min(8, c), c)
        self.prelu2 = nn.ReLU(inplace=True)

        # Encoder (standard conv from here, mask no longer needed)
        self.enc2 = EncoderBlock(c, c * 2)           # 1/2, 64ch
        self.enc3 = EncoderBlock(c * 2, c * 4)       # 1/4, 128ch
        self.enc4 = EncoderBlock(c * 4, c * 8)       # 1/8, 256ch

        # Bottleneck with CBAM
        self.bottleneck = nn.Sequential(
            ConvBNReLU(c * 8, c * 8, stride=2),      # 1/16
            ConvBNReLU(c * 8, c * 8),
            CBAM(c * 8),
        )

        # Decoder
        self.dec4 = DecoderBlock(c * 8, c * 8, c * 8)   # → 1/8
        self.dec3 = DecoderBlock(c * 8, c * 4, c * 4)   # → 1/4
        self.dec2 = DecoderBlock(c * 4, c * 2, c * 2)   # → 1/2
        self.dec1 = DecoderBlock(c * 2, c, c)            # → full

        # Output head: RGB residual in [-1, 1]
        self.out_head = nn.Sequential(
            nn.Conv2d(c, c // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, 3, 1),
            nn.Tanh(),
        )

        # Optional per-pixel confidence head (heteroscedastic uncertainty)
        self.conf_head = nn.Sequential(
            nn.Conv2d(c, c // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, 1, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, hole_mask: torch.Tensor = None):
        """
        Args:
            x:         (B, 14, H, W) concatenated input
            hole_mask: (B, 1, H, W) float32 — 1=valid, 0=hole.
                       If None, all pixels are treated as valid.

        Returns:
            dict with:
                'residual'   : (B, 3, H, W) Δ_fine in [-1, 1]
                'confidence' : (B, 1, H, W) per-pixel confidence in [0, 1]
        """
        if hole_mask is None:
            hole_mask = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)

        # ---- First partial conv layers ----
        p1_out, mask1 = self.pconv1(x, hole_mask)
        e1 = self.prelu1(self.pnorm1(p1_out))

        p2_out, _mask2 = self.pconv2(e1, mask1)
        e1 = self.prelu2(self.pnorm2(p2_out))  # (B, c, H, W)

        # ---- Encoder ----
        e2 = self.enc2(e1)   # (B, 2c, H/2)
        e3 = self.enc3(e2)   # (B, 4c, H/4)
        e4 = self.enc4(e3)   # (B, 8c, H/8)
        eb = self.bottleneck(e4)  # (B, 8c, H/16)

        # ---- Decoder ----
        d4 = self.dec4(eb, e4)   # (B, 8c, H/8)
        d3 = self.dec3(d4, e3)   # (B, 4c, H/4)
        d2 = self.dec2(d3, e2)   # (B, 2c, H/2)
        d1 = self.dec1(d2, e1)   # (B, c, H)

        # ---- Output ----
        residual = self.out_head(d1)      # (B, 3, H, W), tanh → [-1,1]
        confidence = self.conf_head(d1)   # (B, 1, H, W), sigmoid → [0,1]

        return {
            "residual": residual,
            "confidence": confidence,
        }
