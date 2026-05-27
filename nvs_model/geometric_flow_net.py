"""
Model 1: GeometricFlowNet — Depth-Conditioned Coarse View Predictor.

IFNet-inspired multi-scale encoder-decoder that takes 18 input channels:
  img_t0(3) | img_t1(3) | depth_t0(1) | depth_t1(1) |
  warped_t0(3) | warped_t1(3) | mask_t0(1) | mask_t1(1) |
  target_depth(1) | alpha(1)
  Total: 3+3+1+1+3+3+1+1+1+1 = 18

Outputs at multiple pyramid scales for intermediate supervision:
  - flow_t0 (2ch): displacement from target → t0 image
  - flow_t1 (2ch): displacement from target → t1 image
  - blend  (1ch): sigmoid weight for blending (warped_t0 vs warped_t1)
  - residual (3ch): additive RGB correction

Final coarse image:
  I_coarse = blend * warp(img_t0, flow_t0) + (1-blend) * warp(img_t1, flow_t1) + residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .warp_utils import warp_flow


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    """Conv2d + GroupNorm + PReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, groups: int = 8):
        super().__init__()
        pad = kernel // 2
        num_groups = min(groups, out_ch)
        # Make sure out_ch is divisible by num_groups
        while out_ch % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.PReLU(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    """Residual block with two ConvBNReLU layers and identity shortcut."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = ConvBNReLU(ch, ch)
        self.conv2 = ConvBNReLU(ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class DownBlock(nn.Module):
    """Strided conv + residual block (halves spatial resolution)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = ConvBNReLU(in_ch, out_ch, stride=2)
        self.res = ResBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


class UpBlock(nn.Module):
    """Bilinear upsample + conv + residual block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.res = ResBlock(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.res(self.conv(x))


class FlowHead(nn.Module):
    """
    Predicts (flow_t0, flow_t1, blend, residual) from decoder features.

    Output layout (8 channels total):
      [0:2]  flow_t0  — displacement target→t0 (pixels)
      [2:4]  flow_t1  — displacement target→t1 (pixels)
      [4:5]  blend    — raw logit for sigmoid blending weight
      [5:8]  residual — raw additive RGB correction (range approx ±1)
    """

    def __init__(self, in_ch: int):
        super().__init__()
        mid = max(in_ch // 2, 32)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1),
            nn.PReLU(mid),
            nn.Conv2d(mid, 8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class GeometricFlowNet(nn.Module):
    """
    Depth-conditioned coarse view predictor.

    Architecture: 4-level U-Net encoder-decoder with a flow/blend/residual
    head at each decoder level (for intermediate supervision) and a final
    full-resolution prediction.

    Input channels (18 total):
        img_t0     : 3
        img_t1     : 3
        depth_t0   : 1  (normalised, 0 = invalid)
        depth_t1   : 1
        warped_t0  : 3  (img_t0 projected to target view via LiDAR)
        warped_t1  : 3
        mask_t0    : 1  (1 = valid warp, 0 = hole)
        mask_t1    : 1
        tgt_depth  : 1  (LiDAR projected to target camera)
        alpha      : 1  (temporal interpolation scalar, broadcast to HxW)
    """

    IN_CH = 18  # 3+3+1+1+3+3+1+1+1+1 = 18 (see module docstring)

    def __init__(self, base_ch: int = 32):
        super().__init__()
        c = base_ch  # 32

        # Encoder
        self.enc1 = ConvBNReLU(self.IN_CH, c)       # H×W
        self.enc2 = DownBlock(c, c * 2)              # H/2
        self.enc3 = DownBlock(c * 2, c * 4)          # H/4
        self.enc4 = DownBlock(c * 4, c * 8)          # H/8
        self.bottleneck = nn.Sequential(
            DownBlock(c * 8, c * 8),                 # H/16
            ResBlock(c * 8),
        )

        # Decoder (U-Net style) — upsamples back with skip connections
        self.up4 = UpBlock(c * 8, c * 8, c * 8)     # H/8
        self.up3 = UpBlock(c * 8, c * 4, c * 4)     # H/4
        self.up2 = UpBlock(c * 4, c * 2, c * 2)     # H/2
        self.up1 = UpBlock(c * 2, c, c)              # H

        # Flow/blend/residual heads at each decoder level (coarse→fine)
        self.head4 = FlowHead(c * 8)  # H/8  — coarsest
        self.head3 = FlowHead(c * 4)  # H/4
        self.head2 = FlowHead(c * 2)  # H/2
        self.head1 = FlowHead(c)      # H    — finest

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    @staticmethod
    def _decode_head(raw: torch.Tensor, scale: float = 1.0):
        """
        Split raw head output into (flow_t0, flow_t1, blend, residual).

        Args:
            raw:   (B, 8, H, W)
            scale: multiply flow magnitudes (larger at coarser scales)
        """
        flow_t0 = raw[:, 0:2] * scale          # pixels
        flow_t1 = raw[:, 2:4] * scale
        blend = torch.sigmoid(raw[:, 4:5])      # [0, 1]
        residual = torch.tanh(raw[:, 5:8])      # [-1, 1] → colour range ±1
        return flow_t0, flow_t1, blend, residual

    # ------------------------------------------------------------------

    def _compose(
        self,
        img_t0: torch.Tensor,
        img_t1: torch.Tensor,
        flow_t0: torch.Tensor,
        flow_t1: torch.Tensor,
        blend: torch.Tensor,
        residual: torch.Tensor,
        normalised: bool = True,
    ) -> torch.Tensor:
        """
        Warp t0 and t1 images with their respective flows, blend, add residual.

        If normalised=True, img_t0/t1 are assumed in [0, 1].
        Returns image in [0, 1].
        """
        w0 = warp_flow(img_t0, flow_t0)
        w1 = warp_flow(img_t1, flow_t1)
        out = blend * w0 + (1.0 - blend) * w1
        out = out + residual
        return out.clamp(0.0, 1.0)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 21, H, W) concatenated input tensor (values in [0,1] for
               image channels; raw float for depth/alpha channels)

        Returns:
            dict with:
                'coarse'   : (B, 3, H, W) final coarse prediction
                'flow_t0'  : (B, 2, H, W) final flow (target→t0)
                'flow_t1'  : (B, 2, H, W) final flow (target→t1)
                'blend'    : (B, 1, H, W) blending weight (img_t0 contribution)
                'residual' : (B, 3, H, W) additive residual (tanh-normalised)
                'preds_ms' : list of (B,3,H',W') intermediate predictions (coarse→fine)
        """
        # Split out img channels (first 6ch) for warping at full resolution
        img_t0 = x[:, 0:3]
        img_t1 = x[:, 3:6]

        # ---- Encoder ----
        e1 = self.enc1(x)         # (B, c, H, W)
        e2 = self.enc2(e1)        # (B, 2c, H/2, W/2)
        e3 = self.enc3(e2)        # (B, 4c, H/4, W/4)
        e4 = self.enc4(e3)        # (B, 8c, H/8, W/8)
        eb = self.bottleneck(e4)  # (B, 8c, H/16, W/16)

        # ---- Decoder with skip connections ----
        d4 = self.up4(eb, e4)     # (B, 8c, H/8, W/8)
        d3 = self.up3(d4, e3)     # (B, 4c, H/4, W/4)
        d2 = self.up2(d3, e2)     # (B, 2c, H/2, W/2)
        d1 = self.up1(d2, e1)     # (B, c, H, W)

        # ---- Multi-scale flow predictions ----
        # Scale flows by spatial size (coarser levels represent larger motions)
        H, W = x.shape[2], x.shape[3]
        scales = [H / 8, H / 4, H / 2, 1.0]  # rough pixel-scale factors

        # Coarsest: H/8
        r4 = self.head4(d4)
        ft0_4, ft1_4, blend_4, res_4 = self._decode_head(r4, scale=scales[0])

        # H/4
        r3 = self.head3(d3)
        ft0_3, ft1_3, blend_3, res_3 = self._decode_head(r3, scale=scales[1])

        # H/2
        r2 = self.head2(d2)
        ft0_2, ft1_2, blend_2, res_2 = self._decode_head(r2, scale=scales[2])

        # Full resolution
        r1 = self.head1(d1)
        ft0_1, ft1_1, blend_1, res_1 = self._decode_head(r1, scale=scales[3])

        # ---- Compose intermediate predictions for auxiliary losses ----
        def _compose_at_scale(ft0, ft1, blend, res):
            """Resize imgs to the output scale and compose."""
            _H, _W = ft0.shape[2], ft0.shape[3]
            i0 = F.interpolate(img_t0, size=(_H, _W), mode="bilinear", align_corners=False)
            i1 = F.interpolate(img_t1, size=(_H, _W), mode="bilinear", align_corners=False)
            return self._compose(i0, i1, ft0, ft1, blend, res)

        pred4 = _compose_at_scale(ft0_4, ft1_4, blend_4, res_4)
        pred3 = _compose_at_scale(ft0_3, ft1_3, blend_3, res_3)
        pred2 = _compose_at_scale(ft0_2, ft1_2, blend_2, res_2)
        pred1 = self._compose(img_t0, img_t1, ft0_1, ft1_1, blend_1, res_1)

        return {
            "coarse": pred1,
            "flow_t0": ft0_1,
            "flow_t1": ft1_1,
            "blend": blend_1,
            "residual": res_1,
            # Coarse-to-fine list (index 0 = coarsest)
            "preds_ms": [pred4, pred3, pred2, pred1],
        }
