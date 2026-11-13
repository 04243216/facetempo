"""
FaceTempo W1+W2 · extract_landmarks.py  (MediaPipe Tasks API, 1.x)

用法:
    python src/extract_landmarks.py --input data/raw/samples/adult_frontal.mp4
    python src/extract_landmarks.py --input video.mp4 --blendshapes
    python src/extract_landmarks.py --input data/raw/samples/ --blendshapes   # 批量

输入:
    <input>                       源视频
    models/face_landmarker.task   模型（可用 --model 指定）

输出（默认 output_dir = outputs/）:
    outputs/overlay/overlay.mp4
    outputs/features/features.csv
    outputs/curves/curve.png

特征列:
    frame, face_found, ear_l, ear_r, mar, brow, smile_lift, mouth_width,
    eye_aspect_ratio_geo, face_cx, face_cy, face_w, face_h
    + --blendshapes 时追加 52 个 Blendshape 原始列（51 个动作 + neutral）
    + 对应的 52 个 <name>_smooth 列（3 帧居中滚动中值）

几何特征的坐标系:
    ear/mar/brow 沿用 W1 的归一化坐标比值（与历史数据可比）。
    smile_lift/mouth_width/eye_aspect_ratio_geo 在像素坐标下计算，再除以像素脸宽
    （isotropic，避免 16:9 拉伸）。

加固点:
    - face_found 列：每帧都有一行，未检出为 0，特征列为 NaN
    - 分母下限：眼角/嘴角横向距离 < 1e-3（约 1px）时该帧特征判为 NaN
    - mar 上限截断：clip 到 [0, 2.0]，防止异常值拉平曲线
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

MIN_DEN = 1e-3      # 归一化坐标下的最小分母，低于此值视为关键点退化
MAR_MAX = 2.0       # mar 截断上限
SMOOTH_WINDOW = 3   # Blendshape 中值滤波窗口（帧）

# 关键点索引（与旧 FaceMesh 的公开布局一致）
IDX_LEFT_EYE_OUT, IDX_LEFT_EYE_IN = 33, 133
IDX_LEFT_EYE_UP, IDX_LEFT_EYE_LOW = 159, 145
IDX_RIGHT_EYE_IN, IDX_RIGHT_EYE_OUT = 362, 263
IDX_RIGHT_EYE_UP, IDX_RIGHT_EYE_LOW = 386, 374
IDX_MOUTH_UP, IDX_MOUTH_LOW = 13, 14
IDX_MOUTH_LEFT, IDX_MOUTH_RIGHT = 78, 308
IDX_BROW, IDX_BROW_REF = 107, 159
IDX_FACE_LEFT, IDX_FACE_RIGHT = 234, 454
IDX_MOUTH_CORNER_L, IDX_MOUTH_CORNER_R = 61, 291
IDX_FOREHEAD, IDX_CHIN = 10, 152

BASE_COLUMNS = [
    "frame", "face_found", "ear_l", "ear_r", "mar", "brow",
    "smile_lift", "mouth_width", "eye_aspect_ratio_geo",
    "face_cx", "face_cy", "face_w", "face_h",
]

# 见 KI-003：FaceLandmarker 的析构/close() 在 Windows 上会固定阻塞约 42 秒。
# 用模块级列表保活，避免 main() 返回时局部变量被回收而触发析构。
_LANDMARKER_KEEPALIVE: list = []


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FaceTempo W1/W2: 关键点叠加 + 几何特征 + (可选) Blendshape + 曲线"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="输入视频文件或目录（目录则批量处理其中所有 mp4/mov/avi/mkv）",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("outputs"), help="输出根目录（默认 outputs/）"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/face_landmarker.task"),
        help="face_landmarker.task 模型路径",
    )
    parser.add_argument(
        "--blendshapes",
        action="store_true",
        help="同时输出 52 个 Blendshape 动作系数及其 3 帧中值滤波结果",
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


def compute_geometry(lm, width: int, height: int) -> dict[str, float]:
    """像素坐标下的几何微笑代理，避免归一化坐标在 16:9 下的各向异性拉伸。

    smile_lift : (内眼角中点 y - 嘴角中点 y) / 脸宽；嘴角上提时变大。
    mouth_width: |嘴角左 x - 嘴角右 x| / 脸宽；咧嘴时变大。
    eye_aspect_ratio_geo: 左右眼上下眼睑垂直距离的均值 / 脸宽；睁眼时变大。
    脸宽取左右脸颊点 234 / 454 的欧氏距离。
    """
    pts = [(p.x * width, p.y * height) for p in lm]
    face_w = math.hypot(
        pts[IDX_FACE_LEFT][0] - pts[IDX_FACE_RIGHT][0],
        pts[IDX_FACE_LEFT][1] - pts[IDX_FACE_RIGHT][1],
    )
    if face_w < 1.0:
        return {
            "smile_lift": float("nan"),
            "mouth_width": float("nan"),
            "eye_aspect_ratio_geo": float("nan"),
            "face_cx": float("nan"),
            "face_cy": float("nan"),
            "face_w": float("nan"),
            "face_h": float("nan"),
        }

    eye_mid_y = (pts[IDX_LEFT_EYE_IN][1] + pts[IDX_RIGHT_EYE_IN][1]) / 2.0
    mouth_mid_y = (pts[IDX_MOUTH_CORNER_L][1] + pts[IDX_MOUTH_CORNER_R][1]) / 2.0
    smile_lift = (eye_mid_y - mouth_mid_y) / face_w
    mouth_width = (
        abs(pts[IDX_MOUTH_CORNER_L][0] - pts[IDX_MOUTH_CORNER_R][0]) / face_w
    )
    eye_open = (
        abs(pts[IDX_LEFT_EYE_UP][1] - pts[IDX_LEFT_EYE_LOW][1])
        + abs(pts[IDX_RIGHT_EYE_UP][1] - pts[IDX_RIGHT_EYE_LOW][1])
    ) / 2.0
    eye_aspect_ratio_geo = eye_open / face_w
    face_h = abs(pts[IDX_CHIN][1] - pts[IDX_FOREHEAD][1])
    face_cx = (pts[IDX_FACE_LEFT][0] + pts[IDX_FACE_RIGHT][0]) / 2.0
    face_cy = (pts[IDX_FOREHEAD][1] + pts[IDX_CHIN][1]) / 2.0
    return {
        "smile_lift": smile_lift,
        "mouth_width": mouth_width,
        "eye_aspect_ratio_geo": eye_aspect_ratio_geo,
        "face_cx": face_cx,
        "face_cy": face_cy,
        "face_w": face_w,
        "face_h": face_h,
    }


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

    fig, (ax_mar, ax_ear, ax_brow) = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

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
    ax_mar.set_title(f"FaceTempo features - {source_name}")

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


def add_smoothed_columns(df: pd.DataFrame, bs_names: list[str]) -> pd.DataFrame:
    """对 Blendshape 原始列做 3 帧居中滚动中值，生成 <name>_smooth。

    min_periods=1 保证视频首尾帧也有值；face_found=0 产生的 NaN 不参与中值计算。
    """
    if not bs_names:
        return df
    smoothed = (
        df[bs_names]
        .rolling(window=SMOOTH_WINDOW, center=True, min_periods=1)
        .median()
    )
    smoothed.columns = [f"{name}_smooth" for name in bs_names]
    return pd.concat([df, smoothed], axis=1)


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def collect_videos(path: Path) -> tuple[list[Path], bool]:
    """返回 (视频列表, 是否目录模式)。目录模式按文件名排序。"""
    if path.is_dir():
        videos = sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        )
        if not videos:
            raise FileNotFoundError(f"目录中没有视频文件: {path}")
        return videos, True
    if path.is_file():
        return [path], False
    raise FileNotFoundError(f"输入不存在: {path}")


def process_video(
    video: Path,
    output_dir: Path,
    model: Path,
    blendshapes: bool,
    batch: bool,
) -> dict:
    """处理单个视频，产物落盘后返回摘要（所有文件在返回前已关闭/flush）。

    命名规则：
        目录批量模式 -> <stem>_overlay.mp4 / <stem>_features.csv / <stem>_curve.png
        单文件模式   -> overlay.mp4 / features.csv / curve.png（保持 W1/W2 兼容）
    """
    stem = video.stem
    if batch:
        out_overlay = output_dir / "overlay" / f"{stem}_overlay.mp4"
        out_csv = output_dir / "features" / f"{stem}_features.csv"
        out_curve = output_dir / "curves" / f"{stem}_curve.png"
    else:
        out_overlay = output_dir / "overlay" / "overlay.mp4"
        out_csv = output_dir / "features" / "features.csv"
        out_curve = output_dir / "curves" / "curve.png"

    for folder in (out_overlay.parent, out_csv.parent, out_curve.parent):
        folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"无法读取视频尺寸: {width}x{height}")

    writer = cv2.VideoWriter(
        str(out_overlay), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频: {out_overlay}")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=blendshapes,
        output_facial_transformation_matrixes=False,
    )

    rows: list[dict[str, float]] = []
    bs_names: list[str] = []
    infer_ms: list[float] = []
    frame_id = 0
    timestamp_ms = 0
    step_ms = max(1, int(round(1000.0 / fps)))

    # 注意：这里刻意不用 with，也不调用 landmarker.close()。
    # mediapipe 1.2.2 在 Windows 上 close() 会固定阻塞约 42 秒（与帧数无关），
    # 详见 docs/known_issues.md KI-003。每个视频新建实例并保活，进程退出时统一回收。
    init_start = time.perf_counter()
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    _LANDMARKER_KEEPALIVE.append(landmarker)
    init_s = time.perf_counter() - init_start

    loop_start = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )

        t0 = time.perf_counter()
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        infer_ms.append((time.perf_counter() - t0) * 1000.0)
        timestamp_ms += step_ms

        row: dict[str, float] = {"frame": frame_id, "face_found": 0}
        if result.face_landmarks:
            lm = result.face_landmarks[0]
            row["face_found"] = 1
            row.update(compute_features(lm))
            row.update(compute_geometry(lm, width, height))
            draw_landmarks(frame, lm)

        if blendshapes and result.face_blendshapes:
            shapes = result.face_blendshapes[0]
            if not bs_names:
                bs_names = [
                    "neutral" if c.category_name == "_neutral" else c.category_name
                    for c in shapes
                ]
            for name, item in zip(bs_names, shapes):
                row[name] = item.score

        rows.append(row)
        writer.write(frame)
        frame_id += 1

    loop_ms = (time.perf_counter() - loop_start) * 1000.0
    cap.release()
    writer.release()

    df = pd.DataFrame(rows)
    df = df.reindex(columns=BASE_COLUMNS + bs_names)
    df = add_smoothed_columns(df, bs_names)
    df.to_csv(out_csv, index=False)

    plot_curves(df, out_curve, video.stem)

    mar_valid = df["mar"].dropna()
    return {
        "input": video,
        "frames": len(df),
        "found": int(df["face_found"].sum()),
        "columns": len(df.columns),
        "bs_count": len(bs_names),
        "mar_min": float(mar_valid.min()) if len(mar_valid) else float("nan"),
        "mar_max": float(mar_valid.max()) if len(mar_valid) else float("nan"),
        "init_s": init_s,
        "infer_mean": float(np.mean(infer_ms)) if infer_ms else float("nan"),
        "loop_ms": loop_ms,
        "overlay": out_overlay,
        "features": out_csv,
        "curve": out_curve,
    }


def main(argv: list[str] | None = None) -> int:
    t_main_start = time.perf_counter()
    args = parse_args(argv)

    if not args.model.exists():
        raise FileNotFoundError(f"模型不存在: {args.model}")

    videos, batch = collect_videos(args.input)
    summaries: list[dict] = []
    failures: list[tuple[str, str]] = []

    for i, video in enumerate(videos, start=1):
        if batch:
            print(f"[{i}/{len(videos)}] {video.name}", flush=True)
        else:
            print(f"input: {video}", flush=True)

        try:
            summary = process_video(
                video, args.output_dir, args.model, args.blendshapes, batch
            )
        except Exception as exc:  # 单条失败不阻断整批
            failures.append((video.name, str(exc)))
            print(f"  FAILED: {exc}", flush=True)
            continue

        summaries.append(summary)
        total = summary["frames"]
        print(
            f"  frames: {total}, face_found: {summary['found']} "
            f"({summary['found'] / max(total, 1):.1%}), columns: {summary['columns']}"
        )
        if np.isfinite(summary["mar_min"]):
            print(f"  mar range: {summary['mar_min']:.4f} - {summary['mar_max']:.4f}")
        if args.blendshapes:
            print(f"  blendshapes: {summary['bs_count']} raw + {summary['bs_count']} smooth")
        print(
            f"  timing: init {summary['init_s']:.2f}s, "
            f"infer avg {summary['infer_mean']:.1f} ms/frame, "
            f"loop {summary['loop_ms'] / max(total, 1):.1f} ms/frame "
            f"({1000.0 / (summary['loop_ms'] / max(total, 1)):.1f} fps)"
        )
        print(f"  features: {summary['features']}")

        if not batch:
            print(f"  overlay: {summary['overlay']}")
            print(f"  curve: {summary['curve']}")

    wall = time.perf_counter() - t_main_start
    if batch:
        print(
            f"batch done: {len(summaries)} ok, {len(failures)} failed, wall {wall:.1f}s"
        )
        for name, error in failures:
            print(f"  FAILED {name}: {error}")

    print(f"internal total: {wall:.2f}s")
    print("done")
    return 1 if failures else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # 见 docs/known_issues.md KI-003：mediapipe 1.2.2 在 Windows 上析构/close 会阻塞约 42 秒。
    # 产物此时已全部落盘（CSV/视频/图片均已关闭），直接退出，避免无意义等待。
    os._exit(code)
