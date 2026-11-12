"""W3.2：批量肌肉联动诊断（几何/Blendshape 代理 + 事件级滞后）。

用法:
    python src/plot_dynamics.py                      # 批量读 outputs/features/*_features.csv
    python src/plot_dynamics.py --csv outputs/features/adult_frontal_features.csv
    python src/plot_dynamics.py --csv outputs/features --video_dir data/raw/samples

代理选择（见 docs/metrics_spec.md 的「特征可用性实测」）:
    mouthUpperUpRight_smooth   AU12 近似，未饱和
    smile_lift                 几何微笑代理（内眼角中点与嘴角中点的垂直距离 / 脸宽）
    eyeSquintRight_smooth      AU6 近似
    eye_aspect_ratio_geo       几何眼周代理（上下眼睑距离 / 脸宽）

方法:
    1. 0.5 s 居中滚动中值预平滑
    2. 逐 lag 的 Pearson r；lag>0 表示眼周落后于嘴角
    3. 以 smile_lift 峰值为事件中心，前后各 1.5 s
    4. 仅当 r > 0.6 且 |lag| 未触及搜索边界时才输出滞后结论
    5. 比较首个事件（起势）与后续有效事件（维持）的滞后差，>0.4 s 视为不一致

输出：outputs/curves/<name>_dynamics.png（单文件默认 dynamics_good_proxy.png）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LIFT_COL = "smile_lift"
MOUTH_COL = "mouthUpperUpRight_smooth"
EYE_COL = "eyeSquintRight_smooth"

R_THRESHOLD = 0.6
EVENT_HALF_WINDOW_SEC = 1.5
ONSET_MAINTENANCE_GAP_SEC = 0.4
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W3.2: 批量肌肉联动诊断")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/features"),
        help="单个 features.csv 或包含 *_features.csv 的目录",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="单文件模式的输出图路径；目录模式忽略（自动命名）",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/curves"))
    parser.add_argument("--video_dir", type=Path, default=Path("data/raw/samples"),
                        help="用于自动读取 fps 的源视频目录")
    parser.add_argument("--fps", type=float, default=25.0, help="找不到源视频时的兜底 fps")
    parser.add_argument("--max_lag", type=int, default=25, help="最大搜索滞后帧数（1 s @25fps）")
    parser.add_argument("--smooth_sec", type=float, default=0.5, help="预平滑窗口（秒）")
    return parser.parse_args(argv)


def zscore(values: np.ndarray) -> np.ndarray:
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return np.zeros_like(values, dtype=float)
    return (values - np.nanmean(values)) / std


def lagged_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """逐 lag 的 Pearson r；lag>0 表示 y 落后于 x lag 帧。"""
    x = zscore(np.asarray(x, dtype=float))
    y = zscore(np.asarray(y, dtype=float))
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    rs = np.full(lags.shape, np.nan, dtype=float)

    for i, lag in enumerate(lags):
        if lag >= 0:
            a, b = x[: n - lag], y[lag:]
        else:
            a, b = x[-lag:], y[: n + lag]
        if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
            continue
        rs[i] = float(np.corrcoef(a, b)[0, 1])

    return lags, rs


def analyze(x: np.ndarray, y: np.ndarray, max_lag: int) -> dict:
    lags, rs = lagged_correlation(x, y, max_lag)
    best_idx = int(np.nanargmax(rs))
    lag = int(lags[best_idx])
    r = float(rs[best_idx])
    at_boundary = abs(lag) >= max_lag
    return {"lag": lag, "r": r, "valid": (r > R_THRESHOLD) and not at_boundary,
            "at_boundary": at_boundary}


def describe(lag: int, fps: float) -> str:
    ms = lag / fps * 1000.0
    if lag > 0:
        return f"eye lags mouth by {lag} f ({ms:.0f} ms)"
    if lag < 0:
        return f"eye leads mouth by {abs(lag)} f ({abs(ms):.0f} ms)"
    return "synchronous (0 ms)"


def self_check() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    y = np.concatenate([np.zeros(5), x[:-5]]) + 0.05 * rng.normal(size=300)
    result = analyze(x, y, 20)
    if result["lag"] != 5:
        raise RuntimeError(f"滞后符号自检失败: {result['lag']}")


def pick_peaks(values: np.ndarray, min_distance: int, percentile: float = 75.0) -> list[int]:
    threshold = float(np.nanpercentile(values, percentile))
    order = np.argsort(values)[::-1]
    picked: list[int] = []
    for i in order:
        if values[i] < threshold:
            break
        if all(abs(int(i) - p) >= min_distance for p in picked):
            picked.append(int(i))
    return sorted(picked)


def norm01(values: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(hi - lo) or hi - lo < 1e-9:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def detect_fps(csv_path: Path, video_dir: Path, fallback: float) -> tuple[float, str | None]:
    stem = csv_path.stem
    if stem.endswith("_features"):
        stem = stem[: -len("_features")]
    for ext in VIDEO_EXTS:
        candidate = video_dir / f"{stem}{ext}"
        if candidate.exists():
            cap = cv2.VideoCapture(str(candidate))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps and fps > 1:
                return float(fps), candidate.name
    return fallback, None


def render(summary: dict, fps: float, max_lag: int) -> None:
    frames = summary["frames"]
    lift = summary["lift"]
    mouth = summary["mouth"]
    eye = summary["eye"]
    events = summary["events"]
    overall = summary["overall"]
    at_boundary = overall["at_boundary"]
    trusted = overall["valid"] and summary.get("coverage", 1.0) >= 0.95
    verdict = "valid" if trusted else "WEAK / unreliable"

    fig, ax1 = plt.subplots(figsize=(12, 5.2))
    for ev in events:
        ax1.axvspan(ev["lo"], ev["hi"], color="#ffb74d", alpha=0.12, lw=0)

    ax1.plot(frames, norm01(lift), color="#e65100", lw=1.5,
             label=f"{LIFT_COL} (normalized)")
    ax1.plot(frames, norm01(mouth), color="#2e7d32", lw=1.4, ls="--",
             label=f"{MOUTH_COL} (normalized)")
    ax1.set_xlabel("frame")
    ax1.set_ylabel("smile proxy (0-1, display only)")
    ax1.set_ylim(-0.03, 1.03)

    ax2 = ax1.twinx()
    ax2.plot(frames, eye, color="#1565c0", lw=1.5, label=EYE_COL)
    ax2.set_ylabel("eye squint", color="#1565c0")
    ax2.tick_params(axis="y", labelcolor="#1565c0")

    for ev in events:
        ax1.axvline(ev["peak"], color="0.55", ls=":", lw=1)

    lines = [
        f"Muscle Dynamics: {summary['name']} (W3.2)\n"
        f"overall: {MOUTH_COL} vs {EYE_COL}: lag={overall['lag']:+d} f "
        f"({overall['lag'] / fps * 1000:+.0f} ms), r={overall['r']:.3f} [{verdict}]"
    ]
    for ev in events:
        if ev["valid"]:
            lines.append(f"event @{ev['peak']}: {describe(ev['lag'], fps)}, r={ev['r']:.2f}")
        else:
            lines.append(f"event @{ev['peak']}: r={ev['r']:.2f} -> not reliable")
    ax1.set_title("\n".join(lines), fontsize=9)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower right", fontsize=8)

    fig.tight_layout()
    summary["out"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(summary["out"], dpi=150)
    plt.close(fig)


def analyze_file(csv_path: Path, out_path: Path, fps: float,
                 max_lag: int, smooth_sec: float) -> dict:
    df = pd.read_csv(csv_path)
    required = ["frame", LIFT_COL, MOUTH_COL, EYE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"缺少列 {missing}；请先用 --blendshapes 重新提取特征")

    data = df[required].dropna().reset_index(drop=True)
    if len(data) < 20:
        raise ValueError(f"有效帧太少（{len(data)}）")

    window = max(3, int(round(smooth_sec * fps)))
    smoothed = data[required[1:]].rolling(window=window, center=True, min_periods=1).median()
    frames = data["frame"].to_numpy()
    lift = smoothed[LIFT_COL].to_numpy()
    mouth = smoothed[MOUTH_COL].to_numpy()
    eye = smoothed[EYE_COL].to_numpy()

    overall = analyze(mouth, eye, max_lag)

    half = int(round(EVENT_HALF_WINDOW_SEC * fps))
    peaks = pick_peaks(lift, min_distance=half)
    events = []
    for p in peaks:
        lo, hi = max(0, p - half), min(len(frames), p + half)
        if hi - lo < max(10, max_lag * 2):
            continue
        res = analyze(mouth[lo:hi], eye[lo:hi], max_lag)
        events.append({"peak": int(frames[p]), "lo": int(frames[lo]),
                       "hi": int(frames[hi]), **res})

    name = csv_path.stem[: -len("_features")] if csv_path.stem.endswith("_features") else csv_path.stem
    return {
        "name": name,
        "csv": csv_path,
        "out": out_path,
        "frames": frames,
        "lift": lift,
        "mouth": mouth,
        "eye": eye,
        "overall": overall,
        "events": events,
        "coverage": len(data) / max(len(df), 1),
    }


def print_summary(summary: dict, fps: float, source: str | None) -> dict:
    overall = summary["overall"]
    events = summary["events"]
    total_events = len(events)
    valid_events = [e for e in events if e["valid"]]
    coverage = summary.get("coverage", 1.0)
    quality_ok = coverage >= 0.95

    print(f"=== {summary['name']} ===")
    print(f"  fps: {fps:g}" + (f" (from {source})" if source else " (fallback)"))
    print(f"  frames: {len(summary['frames'])}, events: {total_events} ({len(valid_events)} valid)")
    print(f"  face coverage: {coverage:.1%}")
    if not quality_ok:
        print("  WARNING: low face coverage, features may be unreliable (see KI-001)")

    trusted = overall["valid"] and quality_ok
    print(
        f"  overall: lag={overall['lag']:+d} f ({overall['lag'] / fps * 1000:+.0f} ms), "
        f"r={overall['r']:.3f} -> {'VALID' if trusted else 'WEAK/UNRELIABLE'}"
    )
    if overall["valid"] and not quality_ok:
        print("    (r above threshold, but face coverage too low to trust)")
    if trusted:
        print(f"    {describe(overall['lag'], fps)}")

    for ev in events:
        status = "OK" if ev["valid"] else "weak"
        print(
            f"  event @{ev['peak']} [{ev['lo']}-{ev['hi']}]: "
            f"lag={ev['lag']:+d} ({ev['lag'] / fps * 1000:+.0f} ms), "
            f"r={ev['r']:.3f} [{status}]"
        )

    # 起势 vs 维持
    consistency = "n/a"
    gap_frames = None
    if events and valid_events:
        onset = events[0]
        maintenance = [e["lag"] for e in events[1:] if e["valid"]]
        if onset["valid"] and maintenance:
            median_maint = float(np.median(maintenance))
            gap_frames = onset["lag"] - median_maint
            inconsistent = abs(gap_frames) > ONSET_MAINTENANCE_GAP_SEC * fps
            consistency = "INCONSISTENT" if inconsistent else "consistent"
            print(
                f"  onset vs maintenance: onset {onset['lag']:+d} f vs "
                f"maintenance median {median_maint:+.1f} f -> {consistency} "
                f"(gap {gap_frames:+.1f} f / {gap_frames / fps * 1000:+.0f} ms)"
            )
        else:
            print("  onset vs maintenance: not enough valid events")

    print(f"  figure: {summary['out']}")
    return {
        "name": summary["name"],
        "fps": fps,
        "lag": overall["lag"],
        "r": overall["r"],
        "valid": trusted,
        "events": total_events,
        "valid_events": len(valid_events),
        "consistency": consistency,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.csv.is_dir():
        files = sorted(args.csv.glob("*_features.csv"))
        if not files:
            raise FileNotFoundError(f"目录中没有 *_features.csv: {args.csv}")
        batch = True
    elif args.csv.is_file():
        files = [args.csv]
        batch = False
    else:
        raise FileNotFoundError(f"特征路径不存在: {args.csv}")

    self_check()
    rows = []
    for i, csv_path in enumerate(files, start=1):
        fps, source = detect_fps(csv_path, args.video_dir, args.fps)
        if batch:
            name = csv_path.stem[: -len("_features")] if csv_path.stem.endswith("_features") else csv_path.stem
            out_path = args.out_dir / f"{name}_dynamics.png"
        else:
            out_path = args.out or (args.out_dir / "dynamics_good_proxy.png")
        if batch:
            print(f"[{i}/{len(files)}] {csv_path.name}")
        summary = analyze_file(csv_path, out_path, fps, args.max_lag, args.smooth_sec)
        render(summary, fps, args.max_lag)
        rows.append(print_summary(summary, fps, source))
        print()

    if batch and len(rows) > 1:
        print("=== batch summary ===")
        header = f"{'name':28s} {'fps':>5s} {'lag(f)':>7s} {'lag(ms)':>8s} {'r':>6s} {'valid':>5s} {'events':>6s} consistency"
        print(header)
        for row in rows:
            print(
                f"{row['name'][:28]:28s} {row['fps']:5.1f} {row['lag']:+7d} "
                f"{row['lag'] / row['fps'] * 1000:+8.0f} {row['r']:6.3f} "
                f"{'yes' if row['valid'] else 'no':>5s} "
                f"{row['valid_events']}/{row['events']:<4d} {row['consistency']}"
            )

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
