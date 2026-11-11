"""
FaceTempo W1 · extract_landmarks.py  (MediaPipe Tasks API, 1.x)

用法:
    python src/extract_landmarks.py --input data/raw/samples/adult_frontal.mp4
    python src/extract_landmarks.py --input video.mp4 --output_dir outputs

输入:
    <input>                       源视频
    models/face_landmarker.task   模型（可用 --model 指定）

输出（默认 output_dir = outputs/）:
    outputs/overlay/overlay.mp4
    outputs/features/features.csv
    outputs/curves/curve.png

加固点:
    - face_found 列：每帧都有一行，未检出为 0，特征列为 NaN
    - 分母下限：眼角/嘴角横向距离 < 1e-3（约 1px）时该帧特征判为 NaN
    - mar 上限截断：clip 到 [0, 2.0]，防止异常值拉平曲线
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

MIN_DEN = 1e-3      # 归一化坐标下的最小分母，低于此值视为关键点退化
MAR_MAX = 2.0       # mar 截断上限

# 关键点索引（与旧 FaceMesh 的公开布局一致）
IDX_LEFT_EYE_OUT, IDX_LEFT_EYE_IN = 33, 133
IDX_LEFT_EYE_UP, IDX_LEFT_EYE_LOW = 159, 145
IDX_RIGHT_EYE_IN, IDX_RIGHT_EYE_OUT = 362, 263
IDX_RIGHT_EYE_UP, IDX_RIGHT_EYE_LOW = 386, 374
IDX_MOUTH_UP, IDX_MOUTH_LOW = 13, 14
IDX_MOUTH_LEFT, IDX_MOUTH_RIGHT = 78, 308
IDX_BROW, IDX_BROW_REF = 107, 159


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FaceTempo W1: 关键点叠加视频 + 逐帧几何特征 + 曲线图"
    )
    parser.add_argument("--input", type=Path, required=True, help="输入视频路径")
    parser.add_argument(
        "--output_dir", type=Path, default=Path("outputs"), help="输出根目录（默认 outputs/）"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/face_landmarker.task"),
        help="face_landmarker.task 模型路径",
    )
    return parser.parse_args(argv)


def safe_ratio(num: float, den: float) -> float:
    """分母过小说明关键点退化（侧脸/遮挡），返回 NaN，避免数值爆炸。"""
    if abs(den) < MIN_DEN:
        return float("nan")
    return abs(num) / abs(den)


def compute_features(lm) -> dict[str, float]:
    ear_l = safe_ratio(
        lm[IDX_LEFT_EYE_UP].y - lm[IDX_LEFT_EYE_LOW].y,
        lm[IDX_LEFT_EYE_OUT].x - lm[IDX_LEFT_EYE_IN].x,
    )
    ear_r = safe_ratio(
        lm[IDX_RIGHT_EYE_UP].y - lm[IDX_RIGHT_EYE_LOW].y,
        lm[IDX_RIGHT_EYE_IN].x - lm[IDX_RIGHT_EYE_OUT].x,
    )
    mar = safe_ratio(
        lm[IDX_MOUTH_UP].y - lm[IDX_MOUTH_LOW].y,
        lm[IDX_MOUTH_LEFT].x - lm[IDX_MOUTH_RIGHT].x,
    )
    if not np.isnan(mar):
        mar = float(np.clip(mar, 0.0, MAR_MAX))
    brow = abs(lm[IDX_BROW].y - lm[IDX_BROW_REF].y)
    return {"ear_l": ear_l, "ear_r": ear_r, "mar": mar, "brow": brow}


def draw_landmarks(frame, lm) -> None:
    """Tasks API 不提供 FACEMESH_TESSELATION 常量，先画关键索引圆点。"""
    h_img, w_img = frame.shape[:2]
    for idx in [
        IDX_LEFT_EYE_OUT, IDX_LEFT_EYE_IN, IDX_LEFT_EYE_UP, IDX_LEFT_EYE_LOW,
        IDX_RIGHT_EYE_IN, IDX_RIGHT_EYE_OUT, IDX_RIGHT_EYE_UP, IDX_RIGHT_EYE_LOW,
        61, 291, IDX_MOUTH_UP, IDX_MOUTH_LOW, IDX_MOUTH_LEFT, IDX_MOUTH_RIGHT,
        IDX_BROW, 336,
    ]:
        if idx < len(lm):
            x = int(lm[idx].x * w_img)
            y = int(lm[idx].y * h_img)
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)


def plot_curves(df: pd.DataFrame, out_png: Path, source_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_mar, ax_ear, ax_brow) = plt.subplots(
        3, 1, figsize=(12, 7), sharex=True
    )

    for ax in (ax_mar, ax_ear, ax_brow):
        missing = df["face_found"] == 0
        if missing.any():
            ax.fill_between(
                df["frame"],
                0,
                1,
                where=missing,
                transform=ax.get_xaxis_transform(),
                color="0.85",
                linewidth=0,
                label="face not found",
            )

    ax_mar.plot(df["frame"], df["mar"], color="#d62728", lw=1.4)
    ax_mar.set_ylabel("mar")
    ax_mar.set_title(f"FaceTempo W1 features - {source_name}")

    ax_ear.plot(df["frame"], df["ear_l"], lw=1.1, label="ear_l")
    ax_ear.plot(df["frame"], df["ear_r"], lw=1.1, label="ear_r")
    ax_ear.set_ylabel("ear")
    ax_ear.legend(loc="upper right", fontsize=8, ncol=2)

    ax_brow.plot(df["frame"], df["brow"], color="#2ca02c", lw=1.1)
    ax_brow.set_ylabel("brow")
    ax_brow.set_xlabel("frame")

    if df["mar"].notna().any():
        peak = int(df["mar"].idxmax())
        ax_mar.annotate(
            f"max {df.loc[peak, 'mar']:.3f} @ frame {int(df.loc[peak, 'frame'])}",
            xy=(df.loc[peak, "frame"], df.loc[peak, "mar"]),
            xytext=(10, 8),
            textcoords="offset points",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.input.exists():
        raise FileNotFoundError(f"输入视频不存在: {args.input}")
    if not args.model.exists():
        raise FileNotFoundError(f"模型不存在: {args.model}")

    out_overlay = args.output_dir / "overlay" / "overlay.mp4"
    out_csv = args.output_dir / "features" / "features.csv"
    out_curve = args.output_dir / "curves" / "curve.png"
    for p in (out_overlay.parent, out_csv.parent, out_curve.parent):
        p.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.input}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0
    if width <= 0 or height <= 0:
        raise RuntimeError(f"无法读取视频尺寸: {width}x{height}")

    writer = cv2.VideoWriter(
        str(out_overlay), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频: {out_overlay}")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    rows: list[dict[str, float]] = []
    frame_id = 0
    timestamp_ms = 0
    step_ms = max(1, int(round(1000.0 / fps)))

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
            )
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            timestamp_ms += step_ms

            row: dict[str, float] = {"frame": frame_id, "face_found": 0}
            if result.face_landmarks:
                lm = result.face_landmarks[0]
                row["face_found"] = 1
                row.update(compute_features(lm))
                draw_landmarks(frame, lm)

            rows.append(row)
            writer.write(frame)
            frame_id += 1

    cap.release()
    writer.release()

    columns = ["frame", "face_found", "ear_l", "ear_r", "mar", "brow"]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(out_csv, index=False)

    plot_curves(df, out_curve, args.input.stem)

    total = len(df)
    found = int(df["face_found"].sum())
    mar_valid = df["mar"].dropna()
    print(f"input: {args.input}")
    print(f"frames: {total}, face_found: {found} ({found / max(total, 1):.1%})")
    print(f"mar NaN: {int(df['mar'].isna().sum())} / {total}")
    if len(mar_valid):
        print(f"mar range: {mar_valid.min():.4f} - {mar_valid.max():.4f}")
    else:
        print("mar range: no valid frames")
    print(f"overlay: {out_overlay}")
    print(f"features: {out_csv}")
    print(f"curve: {out_curve}")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
