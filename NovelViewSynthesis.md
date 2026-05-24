# Task: Novel View Synthesis

This dataset contains data from autonomous vehicles. The vehicle is equipped with six cameras (front, rear, two front side cameras, and two rear side cameras) and a lidar.

For each sample provided:

| Data | Description |
|--------|----------|
| 12 images | 6 cameras × 2 time points (t0 and t1), separated by an interval of 1 or 2 seconds |
| Dense point cloud | Aggregation of all lidar scans near t0 and t1 (~10–20 scans), coordinates in the world system |
| Camera poses | 4×4 camera-to-world matrices for all 12 input views and the target view |
| Camera parameters | fx, fy, cx, cy, width, height, distortion coefficients |

Goal: Given 12 input images, a point cloud, and camera poses, predict the image from a specified camera at an intermediate time point (≈ midpoint between t0 and t1).

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

Dense point cloud, collected from lidar scans in an extended window around [t0, t1]. The aggregation time range is 3× the length of the delta interval: one delta before t0, the interval [t0, t1] itself, and one delta after t1 (may be truncated earlier at scene boundaries).
    xyz — point coordinates (N, 3), float32, in the world coordinate system
    intensity — reflection intensity (N,), float32
Typical size: 300k–1.5M points (depends on the duration of the interval and the environment).        

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
Main fields:

    target_camera — the name of the camera for which the image needs to be predicted
    delta_s — the time interval between t0 and t1 (1.0 or 2.0 seconds)
    poses_c2w — camera-to-world 4×4 matrices for each camera at each time
    intrinsics — camera parameters (same for t0, t1, and target of the same camera)


## **Coordinate System**

**Cameras (poses_c2w)**

4×4 camera-to-world matrices. Camera axes — OpenCV:

    x → right
    y → down
    z → forward (into the scene)

World frame
Based on the vehicle's position at the start of the scene:

    x → forward (in the direction of travel)
    y → left
    z → up

**LiDAR**

Coordinates xyz in lidar.npz are in the same world coordinate system as the camera poses. The point cloud is aggregated from an extended window [t0 − delta, t1 + delta] (3× delta), and may be truncated earlier at scene boundaries. This provides denser coverage of the scene.

**Cameras**

| Name | Location |
|------|----------|
| front | Front (center of windshield) |
| left_fwd | Left front side |
| left_bwd | Left rear side |
| right_fwd | Right front side |
| right_bwd | Right rear side |
| rear | Rear |

