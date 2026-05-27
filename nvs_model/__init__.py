"""
Two-Stage Novel View Synthesis Model Package.

Stage 1: GeometricFlowNet  — coarse prediction via depth-conditioned optical flow
Stage 2: RefineUNet        — fine-detail correction and hole inpainting
"""

from .geometric_flow_net import GeometricFlowNet
from .refine_unet import RefineUNet
from .losses import CoarseLoss, RefineLoss
from .dataset import NVSDataset

__all__ = ["GeometricFlowNet", "RefineUNet", "CoarseLoss", "RefineLoss", "NVSDataset"]
