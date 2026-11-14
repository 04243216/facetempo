"""W6：把一段正脸视频导出为通用 Blendshape 驱动 JSON。

复用 `src/extract_landmarks.py` 的提取逻辑（52 个 Blendshape + 3 帧中值平滑），
输出结构：

{
  "meta": {"fps": ..., "total_frames": ..., "source": ..., "model": ...},
  "frames": [
    {"frame": 0, "time": 0.0, "blendshapes": {"eyeBlinkLeft": 0.05, ...}}
  ]
}

用法：
    venv\\Scripts\\python.exe src\\translator\\blendshape_export.py
    venv\\Scripts\\python.exe src\\translator\\blendshape_export.py --video <path> --out <path>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import extract_landmarks as core  # noqa: E402

EXPECTED_CHANNELS = 52


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出标准 Blendshape 驱动 JSON")
    parser.add_argument(
        "--video",
        type=Path,
        default=PROJECT_ROOT / "data/raw/samples/w4_clean/real_clean_04.mp4",
        help="输入正脸视频",
    )
    parser.add_argument("--out", type=Path, default=None, help="输出 JSON 路径")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/face_landmarker.task",
    )
    parser.add_argument(
        "--temp_dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/translator/_tmp",
        help="复用提取链路时的临时产物目录",
    )
    return parser.parse_args(argv)


def read_fps(video: Path) -> float:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps > 1 else 25.0


def build_payload(df: pd.DataFrame, video: Path, fps: float) -> dict:
    smooth_cols = [c for c in df.columns if c.endswith("_smooth")]
    if len(smooth_cols) != EXPECTED_CHANNELS:
        raise RuntimeError(
            f"期望 {EXPECTED_CHANNELS} 个平滑 Blendshape 列，实际 {len(smooth_cols)} 个"
        )

    names = [c[: -len("_smooth")] for c in smooth_cols]
    # 逐列线插值，处理掉脸帧；整列为空时补 0
    values = (
        df[smooth_cols]
        .apply(pd.to_numeric, errors="coerce")
        .interpolate(limit_direction="both")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    frames = df["frame"].to_numpy(dtype=int)
    times = frames / fps

    payload = {
        "meta": {
            "fps": round(fps, 3),
            "total_frames": int(len(df)),
            "source": video.name,
            "model": "MediaPipe FaceLandmarker",
            "schema_version": "1.0",
            "smoothing": "3-frame centered median",
            "face_found_ratio": round(float(df["face_found"].mean()), 4),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "frames": [
            {
                "frame": int(frames[i]),
                "time": round(float(times[i]), 4),
                "blendshapes": {
                    name: round(float(values[i, j]), 5)
                    for j, name in enumerate(names)
                },
            }
            for i in range(len(df))
        ],
    }
    return payload


def validate_payload(payload: dict) -> None:
    meta = payload["meta"]
    frames = payload["frames"]
    if len(frames) != meta["total_frames"]:
        raise ValueError("total_frames 与实际帧数不一致")
    if not frames:
        raise ValueError("frames 为空")

    expected = set(frames[0]["blendshapes"].keys())
    if len(expected) != EXPECTED_CHANNELS:
        raise ValueError(f"通道数不是 {EXPECTED_CHANNELS}: {len(expected)}")

    last_time = -1.0
    for item in frames:
        if set(item["blendshapes"].keys()) != expected:
            raise ValueError(f"帧 {item['frame']} 的通道集合不一致")
        for name, value in item["blendshapes"].items():
            if not np.isfinite(value):
                raise ValueError(f"帧 {item['frame']} 的 {name} 不是有限数值")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"帧 {item['frame']} 的 {name} 超出 [0,1]: {value}")
        if item["time"] < last_time:
            raise ValueError(f"帧 {item['frame']} 的时间戳非单调")
        last_time = item["time"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    video = args.video.resolve()
    if not video.exists():
        raise FileNotFoundError(f"输入视频不存在: {video}")
    if not args.model.exists():
        raise FileNotFoundError(f"模型不存在: {args.model}")

    out_path = args.out or (
        PROJECT_ROOT / "outputs/translator" / f"{video.stem}_driving.json"
    )
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"input: {video.name}")
    print(f"fps: {read_fps(video):g}")
    print("extracting 52 blendshapes (3-frame median smoothing) ...")

    # 复用现有提取链路；产物写入临时目录，读取 CSV 后删除视频等大文件
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary = core.process_video(
            video, args.temp_dir, args.model, blendshapes=True, batch=False
        )
        df = pd.read_csv(summary["features"])
    finally:
        shutil.rmtree(args.temp_dir, ignore_errors=True)

    payload = build_payload(df, video, read_fps(video))
    validate_payload(payload)

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)

    size_kb = out_path.stat().st_size / 1024
    meta = payload["meta"]
    print(
        f"done: {meta['total_frames']} frames, {EXPECTED_CHANNELS} channels, "
        f"{meta['total_frames'] / meta['fps']:.2f}s, "
        f"face_found={meta['face_found_ratio']:.1%}"
    )
    print(f"output: {out_path}")
    print(f"size: {size_kb:.1f} KB")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # KI-003：mediapipe 析构会阻塞约 42 秒，产物已落盘，直接退出
    os._exit(code)
