"""W6：校验驱动 JSON，并画出 mouthSmileLeft / eyeSquintLeft 曲线。

用法：
    venv\\Scripts\\python.exe src\\translator\\verify_json.py
    venv\\Scripts\\python.exe src\\translator\\verify_json.py --json <path> --out <png>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = PROJECT_ROOT / "outputs/translator/real_clean_04_driving.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验 Blendshape 驱动 JSON")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs/translator/verify_curve.png",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.json.exists():
        raise FileNotFoundError(f"JSON 不存在: {args.json}")

    with args.json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    meta = data["meta"]
    frames = data["frames"]
    names = list(frames[0]["blendshapes"].keys())
    times = np.array([f["time"] for f in frames], dtype=float)
    mouth = np.array([f["blendshapes"]["mouthSmileLeft"] for f in frames], dtype=float)
    eye = np.array([f["blendshapes"]["eyeSquintLeft"] for f in frames], dtype=float)

    # 结构校验
    assert len(frames) == meta["total_frames"], "total_frames 不一致"
    assert len(names) == 52, f"通道数 {len(names)} != 52"
    assert all(len(f["blendshapes"]) == 52 for f in frames), "存在通道缺失的帧"
    assert np.isfinite(mouth).all() and np.isfinite(eye).all(), "存在 NaN"
    assert np.all(np.diff(times) >= 0), "时间戳非单调"

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(times, mouth, color="#d62728", lw=1.6, label="mouthSmileLeft")
    ax.plot(times, eye, color="#1565c0", lw=1.6, label="eyeSquintLeft")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("blendshape value")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        f"Driving signal check - {meta['source']} "
        f"({meta['total_frames']} frames, {meta['fps']:g} fps)"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)

    print(f"frames: {meta['total_frames']}, channels: {len(names)}")
    print(f"duration: {times[-1]:.2f}s, fps: {meta['fps']:g}")
    print(f"mouthSmileLeft: min={mouth.min():.4f} max={mouth.max():.4f}")
    print(f"eyeSquintLeft : min={eye.min():.4f} max={eye.max():.4f}")
    print(f"face_found: {meta['face_found_ratio']:.1%}")
    print(f"curve: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
