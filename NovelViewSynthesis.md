# Task: Novel View Synthesis

This dataset contains data from autonomous vehicles. The vehicle is equipped with six cameras (front, rear, two front side cameras, and two rear side cameras) and a lidar.

For each sample, the following is provided:

| Data | Description |
|------|-------------|
| **12 images** | 6 cameras × 2 time moments (t0 and t1), separated by 1 or 2 seconds |
| **Dense point cloud** | Aggregation of all LiDAR sweeps between t0 and t1 (~10–20 sweeps), coordinates in world frame |
| **Camera poses** | 4×4 camera-to-world matrices for all 12 input views and the target view |
| **Camera intrinsics** | fx, fy, cx, cy, width, height, distortion coefficients |

**Goal**: using 12 input images, the point cloud, and poses, predict the image from a specified camera at an intermediate time (≈ midpoint between t0 and t1).

Metric: PSNR (Peak Signal-to-Noise Ratio) between the predicted and ground truth image.

## **Evaluation**

The test set is divided into two parts:

| Test set | Description |
|----------|----------|
| Public (samples) | Results are visible on the leaderboard during the competition. The final leaderboard score is calculated based on this part. |
| Private (tests) | Results are hidden until the end of the competition. Used for final ranking. |

**Metric Formula**

1.**PSNR** for each sample is calculated as follows:

$PSNR = 20 \times \log_{10} \left( \frac{255}{\sqrt{MSE}} \right)$

where MSE is the mean squared error over all pixel values (R, G, B) of the predicted and ground truth images.

2.**Normalization** of PSNR to the range [0, 100]:

$score = \frac{clamp(PSNR, 10, 30) - 10}{20} \times 100$

    PSNR ≤ 10 dB → score = 0
    PSNR ≥ 30 dB → score = 100

3.**Final Score** is the arithmetic mean of the normalized scores over all samples of the corresponding test set.

## **Dataset Structure:**

```
dataset/
├── train/
│   └── <sample_id>/
│       ├── meta.json                    # Metadata, poses, camera parameters
│       ├── input/
│       │   ├── t0/
│       │   │   ├── front.jpg
│       │   │   ├── left_fwd.jpg
│       │   │   ├── left_bwd.jpg
│       │   │   ├── right_fwd.jpg
│       │   │   ├── right_bwd.jpg
│       │   │   └── rear.jpg
│       │   ├── t1/
│       │   │   ├── front.jpg
│       │   │   └── ...                  # the same 6 cameras
│       │   └── lidar.npz                # dense point cloud
│       └── target/
│           └── <camera_name>.jpg        # GT image for the target pose
└── test/
    └── <sample_id>/
        ├── meta.json                    # Metadata, poses, camera parameters
        ├── input/
        │   ├── t0/
        │   │   ├── front.jpg
        │   │   ├── left_fwd.jpg
        │   │   ├── left_bwd.jpg
        │   │   ├── right_fwd.jpg
        │   │   ├── right_bwd.jpg
        │   │   └── rear.jpg
        │   ├── t1/
        │   │   ├── front.jpg
        │   │   └── ...                  # the same 6 cameras
        │   └── lidar.npz                # dense point cloud
        └── target/                      # EMPTY (GT not available)
```

**lidar.npz**

Dense point cloud assembled from LiDAR sweeps in an extended window around `[t0, t1]`.
The aggregation time range is 3× the interval length `delta`: one `delta` before `t0`, the interval `[t0, t1]` itself, and one `delta` after `t1` (with clamping at scene boundaries).

- `xyz` — point coordinates `(N, 3)`, float32, in world frame
- `intensity` — reflection intensity `(N,)`, float32

Typical size: 300k–1.5M points (depends on interval duration and environment).      

**meta.json**

```json
{
  "sample_id": "2025-02-10_...__000",
  "scene": "2025-02-10_...",
  "delta_s": 1.0,
  "target_camera": "left_bwd",
  "lidar_info": {
    "n_points": 234567,
    "n_sweeps": 12,
    "t0_ns": 1739195519498333000,
    "t1_ns": 1739195520498331000
  },
  "intrinsics": {
    "front": {"fx": 1028.5, "fy": 1028.5, "cx": 960.0, "cy": 600.0,
              "width": 1920, "height": 1200,
              "distortion_model": "...", "distortion_coeffs": [...]},
    "left_fwd": {...}, "left_bwd": {...},
    "right_fwd": {...}, "right_bwd": {...}, "rear": {...}
  },
  "poses_c2w": {
    "t0": {"front": [[4x4]], "left_fwd": [...], ...},
    "t1": {"front": [[4x4]], ...},
    "target": {"<target_camera>": [[4x4]]}
  }
}
```
Key fields:
- `target_camera` — name of the camera for which the image must be predicted
- `delta_s` — time interval between t0 and t1 (1.0 or 2.0 seconds)
- `poses_c2w` — 4×4 camera-to-world matrices for each camera at each time
- `intrinsics` — camera parameters (the same for t0, t1, and target for a given camera)

---

## Coordinate systems

### Cameras (poses_c2w)
4×4 **camera-to-world** matrices. Camera axes follow **OpenCV**:
- x → right
- y → down
- z → forward (into the scene)

### World frame
Based on the vehicle position at the initial scene moment:
- x → forward (along driving direction)
- y → left
- z → up

### LiDAR
`xyz` coordinates in `lidar.npz` are in the **same world frame** as camera poses.
The cloud is aggregated from the extended window `[t0 − delta, t1 + delta]` (3× delta), with clamping at scene boundaries. This provides denser scene coverage.

**Cameras**

| Name | Location |
|------|----------|
| front | Front (center of windshield) |
| left_fwd | Left front side |
| left_bwd | Left rear side |
| right_fwd | Right front side |
| right_bwd | Right rear side |
| rear | Rear |

---

## Visualization

To inspect data (camera positions, directions, point cloud):

```bash
pip install plotly
python3 scripts/visualize_sample.py --sample-dir dataset/samples/<sample_id>
```

This opens an interactive 3D visualization in a browser.

---

---

## Submission format

For each sample in the test split — one predicted image:

```
submission/
└── <sample_id>/
    └── pred.jpg
```

The `pred.jpg` image must be JPEG, RGB, and have the same resolution as GT (usually 1920×1200 for the front camera, may differ for side cameras).

---