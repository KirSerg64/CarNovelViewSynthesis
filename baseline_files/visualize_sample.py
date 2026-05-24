"""Interactive 3D visualization of a dataset sample using Plotly.

Shows:
  - Dense lidar point cloud (subsampled for performance)
  - Camera positions and viewing directions for t0, t1, and target
  - Camera frustums (wireframe)
  - Color-coded by timestep: blue=t0, red=t1, green=target

Usage:
    python scripts/visualize_sample.py --sample-dir dataset/samples/<sample_id>
    python scripts/visualize_sample.py --sample-dir dataset/samples/<sample_id> --max-points 50000
    python scripts/visualize_sample.py --sample-dir dataset/samples/<sample_id> --no-browser --out viz.html
"""

import argparse
import json
import pathlib
import sys

import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    print("plotly is required: pip install plotly", file=sys.stderr)
    sys.exit(1)


CAMERAS = ["front", "left_fwd", "left_bwd", "right_fwd", "right_bwd", "rear"]

# Colors per timestep
TS_COLORS = {"t0": "royalblue", "t1": "crimson", "target": "limegreen"}

# Colors per camera (for distinguishing within a timestep)
CAM_COLORS = {
    "front":     "#1f77b4",
    "left_fwd":  "#ff7f0e",
    "left_bwd":  "#2ca02c",
    "right_fwd": "#d62728",
    "right_bwd": "#9467bd",
    "rear":      "#8c564b",
}


def _intrinsics_to_K(intr: dict) -> np.ndarray:
    return np.array([
        [intr["fx"], 0.0,        intr["cx"]],
        [0.0,        intr["fy"], intr["cy"]],
        [0.0,        0.0,        1.0],
    ], dtype=np.float32)


def load_meta(sample_dir: pathlib.Path) -> dict:
    return json.load(open(sample_dir / "meta.json"))


def load_lidar(sample_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    npz = np.load(sample_dir / "input" / "lidar.npz")
    return npz["xyz"].astype(np.float32), npz["intensity"].astype(np.float32)


def make_frustum_lines(c2w: np.ndarray, K: np.ndarray, W: int, H: int,
                       scale: float = 1.0) -> np.ndarray:
    """Generate frustum wireframe corners in world space.

    Returns (5, 3) array: 4 corners of the image plane + camera center,
    suitable for drawing lines.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Image corners in normalized camera coords (OpenCV: z-forward)
    corners_cam = np.array([
        [0 - cx, 0 - cy, fx],   # top-left
        [W - cx, 0 - cy, fx],   # top-right
        [W - cx, H - cy, fx],   # bottom-right
        [0 - cx, H - cy, fx],   # bottom-left
    ], dtype=np.float32)

    # Normalize to unit depth and scale
    corners_cam = corners_cam / np.linalg.norm(corners_cam, axis=1, keepdims=True) * scale

    # Transform to world
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    corners_world = (R @ corners_cam.T).T + t

    return corners_world, t


def build_camera_traces(meta: dict, frustum_scale: float = 2.0) -> list:
    """Build plotly traces for all cameras at t0, t1, and target."""
    traces = []
    intrinsics = meta["intrinsics"]

    for ts_key in ("t0", "t1"):
        color = TS_COLORS[ts_key]
        for cam in CAMERAS:
            c2w = np.array(meta["poses_c2w"][ts_key][cam], dtype=np.float32)
            K = _intrinsics_to_K(intrinsics[cam])
            W = int(intrinsics[cam]["width"])
            H = int(intrinsics[cam]["height"])

            corners, center = make_frustum_lines(c2w, K, W, H, scale=frustum_scale)

            # Camera position marker
            traces.append(go.Scatter3d(
                x=[center[0]], y=[center[1]], z=[center[2]],
                mode="markers+text",
                marker=dict(size=4, color=color),
                text=[f"{ts_key}/{cam}"],
                textposition="top center",
                textfont=dict(size=8),
                name=f"{ts_key}/{cam}",
                legendgroup=ts_key,
                showlegend=(cam == "front"),
            ))

            # Frustum wireframe: center -> 4 corners -> back to first
            xs, ys, zs = [], [], []
            for i in range(4):
                # Line from center to corner
                xs.extend([center[0], corners[i, 0], None])
                ys.extend([center[1], corners[i, 1], None])
                zs.extend([center[2], corners[i, 2], None])
            # Rectangle connecting corners
            for i in range(4):
                j = (i + 1) % 4
                xs.extend([corners[i, 0], corners[j, 0], None])
                ys.extend([corners[i, 1], corners[j, 1], None])
                zs.extend([corners[i, 2], corners[j, 2], None])

            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=color, width=2),
                name=f"{ts_key}/{cam} frustum",
                legendgroup=ts_key,
                showlegend=False,
            ))

            # Forward direction arrow (z-axis of camera in world)
            fwd = c2w[:3, 2]  # OpenCV z = forward
            arrow_end = center + fwd * frustum_scale * 0.5
            traces.append(go.Scatter3d(
                x=[center[0], arrow_end[0]],
                y=[center[1], arrow_end[1]],
                z=[center[2], arrow_end[2]],
                mode="lines",
                line=dict(color=color, width=3),
                legendgroup=ts_key,
                showlegend=False,
            ))

    # Target camera
    target_cam = meta["target_camera"]
    color = TS_COLORS["target"]
    c2w = np.array(meta["poses_c2w"]["target"][target_cam], dtype=np.float32)
    K = _intrinsics_to_K(intrinsics[target_cam])
    W = int(intrinsics[target_cam]["width"])
    H = int(intrinsics[target_cam]["height"])

    corners, center = make_frustum_lines(c2w, K, W, H, scale=frustum_scale)

    traces.append(go.Scatter3d(
        x=[center[0]], y=[center[1]], z=[center[2]],
        mode="markers+text",
        marker=dict(size=6, color=color, symbol="diamond"),
        text=[f"TARGET ({target_cam})"],
        textposition="top center",
        textfont=dict(size=10, color=color),
        name=f"target/{target_cam}",
        legendgroup="target",
    ))

    xs, ys, zs = [], [], []
    for i in range(4):
        xs.extend([center[0], corners[i, 0], None])
        ys.extend([center[1], corners[i, 1], None])
        zs.extend([center[2], corners[i, 2], None])
    for i in range(4):
        j = (i + 1) % 4
        xs.extend([corners[i, 0], corners[j, 0], None])
        ys.extend([corners[i, 1], corners[j, 1], None])
        zs.extend([corners[i, 2], corners[j, 2], None])

    traces.append(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=color, width=3),
        name=f"target frustum",
        legendgroup="target",
        showlegend=False,
    ))

    fwd = c2w[:3, 2]
    arrow_end = center + fwd * frustum_scale * 0.5
    traces.append(go.Scatter3d(
        x=[center[0], arrow_end[0]],
        y=[center[1], arrow_end[1]],
        z=[center[2], arrow_end[2]],
        mode="lines",
        line=dict(color=color, width=4),
        legendgroup="target",
        showlegend=False,
    ))

    return traces


def build_lidar_trace(xyz: np.ndarray, intensity: np.ndarray,
                      max_points: int = 50000) -> go.Scatter3d:
    """Build a point cloud trace, subsampled if needed."""
    n = xyz.shape[0]
    if n > max_points:
        idx = np.random.default_rng(42).choice(n, max_points, replace=False)
        xyz = xyz[idx]
        intensity = intensity[idx]

    # Normalize intensity for coloring
    if intensity.max() > 0:
        colors = intensity / intensity.max()
    else:
        colors = np.zeros(len(intensity))

    return go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="markers",
        marker=dict(
            size=1,
            color=colors,
            colorscale="Viridis",
            opacity=0.6,
            colorbar=dict(title="Intensity", len=0.5),
        ),
        name=f"lidar ({n:,} pts, showing {len(xyz):,})",
    )


def build_axes_traces(origin: np.ndarray, length: float = 5.0) -> list:
    """World frame axes at given origin: X=red (forward), Y=green (left), Z=blue (up)."""
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    axes = [
        ("X (fwd)", [ox + length, oy, oz], "red"),
        ("Y (left)", [ox, oy + length, oz], "green"),
        ("Z (up)", [ox, oy, oz + length], "blue"),
    ]
    traces = []
    for name, end, color in axes:
        traces.append(go.Scatter3d(
            x=[ox, end[0]], y=[oy, end[1]], z=[oz, end[2]],
            mode="lines+text",
            line=dict(color=color, width=4),
            text=["", name],
            textposition="top center",
            name=f"axis {name}",
            showlegend=False,
        ))
    return traces


def visualize_sample(sample_dir: pathlib.Path, max_points: int = 50000,
                     frustum_scale: float = 2.0) -> go.Figure:
    """Build a complete 3D visualization figure for a sample.

    Includes toggle buttons: All / Cameras only / Lidar only.
    """
    meta = load_meta(sample_dir)
    xyz, intensity = load_lidar(sample_dir)

    # Use median of lidar cloud as origin for axes
    cloud_center = np.median(xyz, axis=0)

    # Build traces and track categories
    lidar_traces = [build_lidar_trace(xyz, intensity, max_points=max_points)]
    camera_traces = build_camera_traces(meta, frustum_scale=frustum_scale)
    axes_traces = build_axes_traces(cloud_center)

    # Order: lidar, cameras, axes
    all_traces = lidar_traces + camera_traces + axes_traces
    n_lidar = len(lidar_traces)
    n_camera = len(camera_traces)
    n_axes = len(axes_traces)
    n_total = n_lidar + n_camera + n_axes

    fig = go.Figure(data=all_traces)

    # Visibility masks for toggle buttons
    vis_all = [True] * n_total
    vis_cameras = [False] * n_lidar + [True] * n_camera + [True] * n_axes
    vis_lidar = [True] * n_lidar + [False] * n_camera + [True] * n_axes

    title = (f"Sample: {meta['sample_id']}<br>"
             f"Δt={meta['delta_s']}s | target={meta['target_camera']} | "
             f"lidar={xyz.shape[0]:,} pts")
    if "lidar_info" in meta:
        title += f" ({meta['lidar_info']['n_sweeps']} sweeps)"

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (forward)",
            yaxis_title="Y (left)",
            zaxis_title="Z (up)",
            aspectmode="data",
        ),
        legend=dict(x=0.01, y=0.99),
        width=1400,
        height=900,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0, y=1.12,
                xanchor="left",
                buttons=[
                    dict(label="All",
                         method="update",
                         args=[{"visible": vis_all}]),
                    dict(label="Cameras only",
                         method="update",
                         args=[{"visible": vis_cameras}]),
                    dict(label="Lidar only",
                         method="update",
                         args=[{"visible": vis_lidar}]),
                ],
            ),
        ],
    )

    return fig


def main():
    p = argparse.ArgumentParser(description="Visualize a dataset sample in 3D")
    p.add_argument("--sample-dir", type=pathlib.Path, required=True,
                   help="Path to sample directory")
    p.add_argument("--max-points", type=int, default=50000,
                   help="Max lidar points to display (default: 50000)")
    p.add_argument("--frustum-scale", type=float, default=2.0,
                   help="Size of camera frustums in meters")
    p.add_argument("--out", type=str, default=None,
                   help="Save HTML to file instead of opening browser")
    p.add_argument("--no-browser", action="store_true",
                   help="Don't open browser (use with --out)")
    args = p.parse_args()

    fig = visualize_sample(args.sample_dir, max_points=args.max_points,
                           frustum_scale=args.frustum_scale)

    if args.out:
        fig.write_html(args.out)
        print(f"Saved to {args.out}")
    if not args.no_browser and not args.out:
        fig.show()
    elif not args.no_browser and args.out:
        fig.show()


if __name__ == "__main__":
    main()
