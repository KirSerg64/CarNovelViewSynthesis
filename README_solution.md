# LiDAR-Based Novel View Synthesis

A novel view synthesis pipeline that uses LiDAR depth information and multi-camera images to render target views via geometric reprojection.

## Approach

This implementation uses **LiDAR-guided depth warping** to synthesize novel views:

1. **Depth Projection**: Projects the dense LiDAR point cloud into each input camera to obtain per-pixel depth maps
2. **Inverse Warping (same camera)**: For the target camera's own images at t0 and t1, uses inverse warping with `cv2.remap` for smooth bilinear interpolation
3. **Forward Splatting (other cameras)**: For neighboring cameras, uses forward splatting to fill areas occluded from the target camera's perspective  
4. **Multi-view Blending**: Combines all warped views with priority weighting (same camera gets 10× weight)
5. **Temporal Fusion**: Blends the geometric warp result with a simple temporal average (fallback for sparse/noisy regions)

### Key Design Decisions
- **Inverse warping** for same-camera views produces much smoother results than forward splatting
- **Adaptive blending** with temporal average ensures robustness when depth coverage is sparse
- **Depth densification** via morphological dilation fills gaps in the sparse LiDAR projection

## Results

On the training set (3 samples):
- **Mean PSNR: ~22.6 dB** (Score: ~63/100)

## Usage

### Requirements
```bash
pip install -r requirements.txt
```

### Generate Test Submission
```bash
python lidar_warp_nvs.py --data-dir data/test --output-dir submission
```

### Evaluate on Training Data
```bash
python lidar_warp_nvs.py --data-dir data/train --output-dir output_train --evaluate
```

### Standalone Evaluation
```bash
python evaluate.py --pred-dir output_train --gt-dir data/train
```

### Process Specific Samples
```bash
python lidar_warp_nvs.py --data-dir data/train --output-dir output --evaluate \
    --samples sample_id_1 sample_id_2
```

## Output Format

```
submission/
└── <sample_id>/
    └── pred.jpg
```

## Technical Details

- **No GPU required** - runs entirely on CPU with NumPy/OpenCV
- **No training required** - purely geometric approach
- Processing time: ~2-3 minutes per sample (depending on LiDAR density)
- Memory: ~4-8 GB RAM (for large point clouds)

## Potential Improvements

1. **Optical flow refinement** - use RAFT/DIS flow between t0/t1 to refine dynamic object positions
2. **Neural inpainting** - replace simple inpainting with a learned model for occluded regions
3. **Adaptive blend weights** - learn per-pixel blend weights instead of fixed ratio
4. **GPU acceleration** - use PyTorch for parallel point projection and warping
