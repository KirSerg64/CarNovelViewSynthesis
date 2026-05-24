# чтоб бейзлайн завелся нужно слонировать вот это https://github.com/hzwer/ECCV2022-RIFE?ysclid=mpfgpbp25p684963391

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "ECCV2022-RIFE"))

import torch
import torch.nn.functional as F
from train_log.RIFE_HDv3 import Model

TEST_DIR = Path("dataset/test")
OUTPUT_DIR = Path("submission")

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_rife():
    m = Model()
    m.load_model("train_log", -1)
    m.eval()
    return m


def infer_rife(model, img0_np: np.ndarray, img1_np: np.ndarray) -> np.ndarray:
    h, w = img0_np.shape[:2]
    ph = ((h - 1) // 64 + 1) * 64
    pw = ((w - 1) // 64 + 1) * 64

    def to_t(x):
        return torch.from_numpy(x).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

    t0 = F.pad(to_t(img0_np), (0, pw - w, 0, ph - h))
    t1 = F.pad(to_t(img1_np), (0, pw - w, 0, ph - h))
    with torch.no_grad():
        out = model.inference(t0, t1)
    return (out[0, :, :h, :w].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def warp(img: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(
        img,
        (xs + dx).astype(np.float32),
        (ys + dy).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def render_sample(model, dis, sample_dir: Path) -> np.ndarray:
    meta = json.loads((sample_dir / "meta.json").read_text())
    camera = meta["target_camera"]
    ts = meta["timestamps_ns"]
    alpha = float((ts["target"] - ts["t0"]) / (ts["t1"] - ts["t0"]))

    img0 = np.array(Image.open(sample_dir / "input" / "t0" / f"{camera}.jpg"))
    img1 = np.array(Image.open(sample_dir / "input" / "t1" / f"{camera}.jpg"))

    mean = (img0.astype(np.float32) + img1.astype(np.float32)) * 0.5

    gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    flow = dis.calc(gray0, gray1, None)
    dx, dy = flow[..., 0], flow[..., 1]
    w0 = warp(img0, -alpha * dx, -alpha * dy)
    w1 = warp(img1, (1 - alpha) * dx, (1 - alpha) * dy)
    dis_warped = (1 - alpha) * w0.astype(np.float32) + alpha * w1.astype(np.float32)
    dis_blend = (0.55 * dis_warped + 0.45 * mean).clip(0, 255)

    rife = infer_rife(model, img0, img1).astype(np.float32)

    result = (0.60 * dis_blend + 0.40 * rife).clip(0, 255)
    return result.astype(np.uint8)


def process_sample(model, dis, sample_dir: Path) -> None:
    pred = render_sample(model, dis, sample_dir)
    out_dir = OUTPUT_DIR / sample_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred).save(out_dir / "pred.jpg")


def main() -> None:
    model = load_rife()
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    samples = sorted(p for p in TEST_DIR.iterdir() if p.is_dir())
    for i, sample_dir in enumerate(samples):
        process_sample(model, dis, sample_dir)
        print(f"[{i + 1}/{len(samples)}] {sample_dir.name}")
    print(f"Done. Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
