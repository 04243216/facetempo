"""W4.1：批量肌肉联动诊断 + 平滑度/微颤（Jerk）分析。

用法:
    python src/plot_dynamics.py --csv outputs/w4_clean/features \
        --video_dir data/raw/samples/w4_clean --out_dir outputs/w4_clean/curves

核心改动（W4.1）:
    1. 事件级基线扣除：事件前 1 s 的均值作为基线，事件窗口减去基线后再算相关。
    2. Bootstrap 置信区间：对每个事件的 r 做 1000 次重采样，给 95% CI。
    3. 平滑度指标：事件窗口内原始信号的一阶差分均值（速度）与二阶差分均值（Jerk），
       按 fps 归一化为「每秒」与「每秒²」，可跨帧率比较。
    4. 三道硬门控：画面运动量 < 2.0、眼通道一致性 < -0.5、嘴部动态 > 0.05。
       任一不达标 → 整条视频标记 INVALID。
    5. 输出 jerk 对比图（每个视频的嘴部/眼周微颤能量）。

符号约定：lag > 0 表示眼周落后于嘴角；lag < 0 表示眼周领先。
可靠性：r > 0.6 且 |lag| 未触及搜索边界，才输出滞后结论。
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
MOUTH_RAW = "mouthUpperUpRight"
EYE_RAW = "eyeSquintRight"

R_THRESHOLD = 0.6
EVENT_HALF_WINDOW_SEC = 1.5
BASELINE_SEC = 1.0
ONSET_MAINTENANCE_GAP_SEC = 0.4

MOTION_MAX = 2.0
EYE_AGREEMENT_MAX = -0.5
MOUTH_STD_MIN = 0.05

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W4.1: 批量肌肉联动 + 微颤分析")
    parser.add_argument("--csv", type=Path, default=Path("outputs/features"),
                        help="单个 features.csv 或包含 *_features.csv 的目录")
    parser.add_argument("--out", type=Path, default=None,
                        help="单文件模式输出图路径；目录模式忽略")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/curves"))
    parser.add_argument("--video_dir", type=Path, default=Path("data/raw/samples"))
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--max_lag", type=int, default=25)
    parser.add_argument("--smooth_sec", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args(argv)


def zscore(values: np.ndarray) -> np.ndarray:
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return np.zeros_like(values, dtype=float)
    return (values - np.nanmean(values)) / std


def lagged_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
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


def bootstrap_r_ci(x: np.ndarray, y: np.ndarray, lag: int, n_boot: int,
                   alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """对固定 lag 下的 r 做 bootstrap 95% CI。"""
    if n_boot <= 0:
        return (float("nan"), float("nan"))
    n = len(x)
    if lag >= 0:
        a, b = x[: n - lag], y[lag:]
    else:
        a, b = x[-lag:], y[: n + lag]
    if len(a) < 10:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    m = len(a)
    values = np.empty(n_boot, dtype=float)
    count = 0
    for _ in range(n_boot):
        idx = rng.integers(0, m, m)
        aa, bb = a[idx], b[idx]
        if np.std(aa) == 0 or np.std(bb) == 0:
            continue
        values[count] = np.corrcoef(aa, bb)[0, 1]
        count += 1
    if count == 0:
        return (float("nan"), float("nan"))
    values = values[:count]
    return (float(np.percentile(values, 100 * alpha / 2)),
            float(np.percentile(values, 100 * (1 - alpha / 2))))


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
    if analyze(x, y, 20)["lag"] != 5:
        raise RuntimeError("滞后符号自检失败")


def jitter_stats(signal: np.ndarray, fps: float) -> tuple[float, float, float]:
    """返回 (速度, Jerk, 归一化 Jerk)。

    速度 = |一阶差分|均值 × fps
    Jerk = |二阶差分|均值 × fps²
    归一化 Jerk = Jerk / (std × fps²)，去掉动作幅度的影响，才可比“平滑度”。
    """
    d1 = np.diff(signal)
    d2 = np.diff(signal, n=2)
    speed = float(np.mean(np.abs(d1))) * fps if len(d1) else float("nan")
    jerk = float(np.mean(np.abs(d2))) * fps * fps if len(d2) else float("nan")
    std = float(np.std(signal))
    norm_jerk = jerk / (std * fps * fps) if std > 0 else float("nan")
    return speed, jerk, norm_jerk


def hf_ratio(signal: np.ndarray, fps: float, lo: float = 5.0, hi: float = 12.0) -> float:
    """5–12 Hz 高频能量占总能量比例（去线性趋势后）。"""
    if len(signal) < 16:
        return float("nan")
    x = np.asarray(signal, dtype=float)
    t = np.arange(len(x))
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / fps)
    denom = spec[freqs >= 0.5].sum()
    if denom <= 0:
        return float("nan")
    return float(spec[(freqs >= lo) & (freqs <= hi)].sum() / denom)


def compute_motion(video_path: Path, size: tuple[int, int] = (320, 180)) -> float:
    """画面运动量：相邻帧灰度绝对差均值（越小越静）。"""
    if not video_path.exists():
        return float("nan")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return float("nan")
    prev = None
    diffs: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.mean(np.abs(gray - prev))))
        prev = gray
    cap.release()
    return float(np.mean(diffs)) if diffs else float("nan")


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


def detect_fps(csv_path: Path, video_dir: Path, fallback: float) -> tuple[float, Path | None]:
    stem = csv_path.stem[: -len("_features")] if csv_path.stem.endswith("_features") else csv_path.stem
    for ext in VIDEO_EXTS:
        candidate = video_dir / f"{stem}{ext}"
        if candidate.exists():
            cap = cv2.VideoCapture(str(candidate))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps and fps > 1:
                return float(fps), candidate
    return fallback, None


def render(summary: dict, fps: float, max_lag: int) -> None:
    frames = summary["frames"]
    lift, mouth, eye = summary["lift"], summary["mouth"], summary["eye"]
    events = summary["events"]
    overall = summary["overall"]
    trusted = summary["trusted"]
    verdict = "valid" if trusted else "INVALID / unreliable"

    fig, ax1 = plt.subplots(figsize=(12, 5.4))
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
        f"Muscle Dynamics: {summary['name']} (W4.1)\n"
        f"overall: lag={overall['lag']:+d} f ({overall['lag'] / fps * 1000:+.0f} ms), "
        f"r={overall['r']:.3f} [{verdict}]"
    ]
    for ev in events:
        if ev["valid"]:
            lines.append(
                f"event @{ev['peak']}: {describe(ev['lag'], fps)}, r={ev['r']:.2f} "
                f"CI[{ev['ci_lo']:.2f},{ev['ci_hi']:.2f}]"
            )
        else:
            lines.append(f"event @{ev['peak']}: r={ev['r']:.2f} -> not reliable")
    ax1.set_title("\n".join(lines), fontsize=8.6)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower right", fontsize=8)

    fig.tight_layout()
    summary["out"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(summary["out"], dpi=150)
    plt.close(fig)


def analyze_file(csv_path: Path, out_path: Path, fps: float, video_path: Path | None,
                 max_lag: int, smooth_sec: float, n_boot: int) -> dict:
    df = pd.read_csv(csv_path)
    required = ["frame", LIFT_COL, MOUTH_COL, EYE_COL, MOUTH_RAW, EYE_RAW,
                "eye_aspect_ratio_geo"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"缺少列 {missing}；请用 --blendshapes 重新提取")

    data = df[required].dropna().reset_index(drop=True)
    if len(data) < 20:
        raise ValueError(f"有效帧太少（{len(data)}）")

    window = max(3, int(round(smooth_sec * fps)))
    smoothed = data[[LIFT_COL, MOUTH_COL, EYE_COL]].rolling(
        window, center=True, min_periods=1).median()
    frames = data["frame"].to_numpy()
    lift = smoothed[LIFT_COL].to_numpy()
    mouth = smoothed[MOUTH_COL].to_numpy()
    eye = smoothed[EYE_COL].to_numpy()
    mouth_raw = data[MOUTH_RAW].to_numpy()
    eye_raw = data[EYE_RAW].to_numpy()

    overall = analyze(mouth, eye, max_lag)

    # ---- 三道门控 ----
    motion = compute_motion(video_path) if video_path else float("nan")
    agreement = float(np.corrcoef(data["eye_aspect_ratio_geo"], data[EYE_COL])[0, 1])
    mouth_std = float(np.std(data[MOUTH_COL]))
    gate_reasons = []
    if not np.isfinite(motion) or motion >= MOTION_MAX:
        gate_reasons.append(f"motion={motion:.2f}>={MOTION_MAX}")
    if not np.isfinite(agreement) or agreement >= EYE_AGREEMENT_MAX:
        gate_reasons.append(f"eye_agreement={agreement:.2f}>={EYE_AGREEMENT_MAX}")
    if mouth_std <= MOUTH_STD_MIN:
        gate_reasons.append(f"mouth_std={mouth_std:.3f}<={MOUTH_STD_MIN}")

    # ---- 事件 ----
    half = int(round(EVENT_HALF_WINDOW_SEC * fps))
    baseline_len = int(round(BASELINE_SEC * fps))
    peaks = pick_peaks(lift, min_distance=half)
    events = []
    for p in peaks:
        lo, hi = max(0, p - half), min(len(frames), p + half)
        if hi - lo < max(10, max_lag * 2):
            continue

        base_lo = max(0, lo - baseline_len)
        if lo - base_lo < 3:
            base_lo = max(0, lo - 5)
        base_m = float(np.mean(mouth[base_lo:lo])) if lo > base_lo else 0.0
        base_e = float(np.mean(eye[base_lo:lo])) if lo > base_lo else 0.0
        seg_m = mouth[lo:hi] - base_m
        seg_e = eye[lo:hi] - base_e

        res = analyze(seg_m, seg_e, max_lag)
        ci_lo, ci_hi = bootstrap_r_ci(seg_m, seg_e, res["lag"], n_boot, seed=p)
        speed_m, jerk_m, _ = jitter_stats(mouth_raw[lo:hi], fps)
        speed_e, jerk_e, _ = jitter_stats(eye_raw[lo:hi], fps)

        events.append({
            "peak": int(frames[p]), "lo": int(frames[lo]), "hi": int(frames[hi - 1]),
            "baseline_m": base_m, "baseline_e": base_e,
            "speed_m": speed_m, "jerk_m": jerk_m,
            "speed_e": speed_e, "jerk_e": jerk_e,
            "ci_lo": ci_lo, "ci_hi": ci_hi, **res,
        })

    speed_m_all, jerk_m_all, norm_jerk_m = jitter_stats(mouth_raw, fps)
    speed_e_all, jerk_e_all, norm_jerk_e = jitter_stats(eye_raw, fps)
    hf_m = hf_ratio(mouth_raw, fps)
    hf_e = hf_ratio(eye_raw, fps)

    name = csv_path.stem[: -len("_features")] if csv_path.stem.endswith("_features") else csv_path.stem
    trusted = overall["valid"] and not gate_reasons
    return {
        "name": name, "csv": csv_path, "out": out_path,
        "frames": frames, "lift": lift, "mouth": mouth, "eye": eye,
        "overall": overall, "events": events,
        "coverage": len(data) / max(len(df), 1),
        "motion": motion, "agreement": agreement, "mouth_std": mouth_std,
        "gate_reasons": gate_reasons, "trusted": trusted,
        "speed_m": speed_m_all, "jerk_m": jerk_m_all,
        "speed_e": speed_e_all, "jerk_e": jerk_e_all,
        "norm_jerk_m": norm_jerk_m, "norm_jerk_e": norm_jerk_e,
        "hf_m": hf_m, "hf_e": hf_e,
    }


def print_summary(summary: dict, fps: float, source: str | None) -> dict:
    overall, events = summary["overall"], summary["events"]
    valid_events = [e for e in events if e["valid"]]

    print(f"=== {summary['name']} ===")
    print(f"  fps: {fps:g}" + (f" (from {source})" if source else " (fallback)"))
    print(f"  frames: {len(summary['frames'])}, face coverage: {summary['coverage']:.1%}")
    print(
        f"  gates: motion={summary['motion']:.2f}, "
        f"eye_agreement={summary['agreement']:.2f}, "
        f"mouth_std={summary['mouth_std']:.3f} -> "
        f"{'OK' if not summary['gate_reasons'] else 'INVALID: ' + ', '.join(summary['gate_reasons'])}"
    )
    print(
        f"  jitter: mouth speed={summary['speed_m']:.2f}/s jerk={summary['jerk_m']:.2f}/s^2 "
        f"norm={summary['norm_jerk_m']:.3f} | "
        f"eye speed={summary['speed_e']:.2f}/s jerk={summary['jerk_e']:.2f}/s^2 "
        f"norm={summary['norm_jerk_e']:.3f}"
    )
    print(
        f"  hf energy (5-12 Hz): mouth={summary['hf_m']:.3f}, eye={summary['hf_e']:.3f}"
    )
    print(
        f"  overall: lag={overall['lag']:+d} f ({overall['lag'] / fps * 1000:+.0f} ms), "
        f"r={overall['r']:.3f} -> {'VALID' if summary['trusted'] else 'WEAK/UNRELIABLE'}"
    )

    for ev in events:
        status = "OK" if ev["valid"] else "weak"
        print(
            f"  event @{ev['peak']} [{ev['lo']}-{ev['hi']}]: "
            f"lag={ev['lag']:+d} ({ev['lag'] / fps * 1000:+.0f} ms), r={ev['r']:.3f} "
            f"CI[{ev['ci_lo']:.2f},{ev['ci_hi']:.2f}] [{status}] "
            f"jerk_m={ev['jerk_m']:.2f} jerk_e={ev['jerk_e']:.2f}"
        )

    consistency = "n/a"
    if events and valid_events:
        onset = events[0]
        maintenance = [e["lag"] for e in events[1:] if e["valid"]]
        if onset["valid"] and maintenance:
            median_maint = float(np.median(maintenance))
            gap = onset["lag"] - median_maint
            inconsistent = abs(gap) > ONSET_MAINTENANCE_GAP_SEC * fps
            consistency = "INCONSISTENT" if inconsistent else "consistent"
            print(
                f"  onset vs maintenance: onset {onset['lag']:+d} f vs "
                f"maintenance median {median_maint:+.1f} f -> {consistency} "
                f"(gap {gap:+.1f} f / {gap / fps * 1000:+.0f} ms)"
            )
        else:
            print("  onset vs maintenance: not enough valid events")

    print(f"  figure: {summary['out']}")
    return {
        "name": summary["name"], "fps": fps,
        "lag": overall["lag"], "r": overall["r"],
        "valid": summary["trusted"],
        "events": len(events), "valid_events": len(valid_events),
        "consistency": consistency,
        "jerk_m": summary["jerk_m"], "jerk_e": summary["jerk_e"],
        "norm_jerk_m": summary["norm_jerk_m"], "norm_jerk_e": summary["norm_jerk_e"],
        "hf_m": summary["hf_m"], "hf_e": summary["hf_e"],
    }


def plot_jitter(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    names = [r["name"] for r in rows]
    x = np.arange(len(rows))
    width = 0.38
    fig, (ax_raw, ax_norm) = plt.subplots(
        2, 1, figsize=(max(8, 1.4 * len(rows)), 7.0), sharex=True
    )

    ax_raw.bar(x - width / 2, [r["jerk_m"] for r in rows], width,
               label="mouth jerk", color="#2e7d32")
    ax_raw.bar(x + width / 2, [r["jerk_e"] for r in rows], width,
               label="eye jerk", color="#1565c0")
    ax_raw.set_ylabel("mean |2nd diff| x fps^2")
    ax_raw.set_title("Micro-jitter energy (raw, amplitude-dependent)")
    ax_raw.legend(fontsize=8)

    ax_norm.bar(x - width / 2, [r["norm_jerk_m"] for r in rows], width,
                label="mouth norm jerk", color="#2e7d32")
    ax_norm.bar(x + width / 2, [r["norm_jerk_e"] for r in rows], width,
                label="eye norm jerk", color="#1565c0")
    ax_norm.set_ylabel("jerk / (std x fps^2)")
    ax_norm.set_title("Amplitude-normalized micro-jitter (comparable)")
    ax_norm.set_xticks(x)
    ax_norm.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax_norm.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
        fps, video_path = detect_fps(csv_path, args.video_dir, args.fps)
        name = csv_path.stem[: -len("_features")] if csv_path.stem.endswith("_features") else csv_path.stem
        if batch:
            print(f"[{i}/{len(files)}] {csv_path.name}")
            out_path = args.out_dir / f"{name}_dynamics.png"
        else:
            out_path = args.out or (args.out_dir / "dynamics_good_proxy.png")

        summary = analyze_file(csv_path, out_path, fps, video_path,
                               args.max_lag, args.smooth_sec, args.bootstrap)
        render(summary, fps, args.max_lag)
        rows.append(print_summary(summary, fps, video_path.name if video_path else None))
        print()

    if len(rows) > 1:
        print("=== batch summary ===")
        print(f"{'name':26s} {'fps':>5s} {'lag(f)':>7s} {'lag(ms)':>8s} {'r':>6s} "
              f"{'valid':>5s} {'ev':>5s} {'jerk_m':>7s} {'njer_m':>7s} "
              f"{'jerk_e':>7s} {'njer_e':>7s} consistency")
        for row in rows:
            print(
                f"{row['name'][:26]:26s} {row['fps']:5.1f} {row['lag']:+7d} "
                f"{row['lag'] / row['fps'] * 1000:+8.0f} {row['r']:6.3f} "
                f"{'yes' if row['valid'] else 'no':>5s} "
                f"{row['valid_events']}/{row['events']:<3d} "
                f"{row['jerk_m']:7.2f} {row['norm_jerk_m']:7.3f} "
                f"{row['jerk_e']:7.2f} {row['norm_jerk_e']:7.3f} {row['consistency']}"
            )

    jitter_path = args.out_dir / "jerk_comparison.png"
    plot_jitter(rows, jitter_path)
    print(f"jitter figure: {jitter_path}")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
