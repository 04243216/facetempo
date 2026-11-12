"""W3.1：肌肉联动图（经过有效性筛选的代理）+ 事件级互相关滞后分析。

代理选择（W3 反例教训见 docs/metrics_spec.md）：
    mouthUpperUpRight_smooth   AU12 近似，动态范围 0–0.88，未饱和
    smile_lift                 几何微笑代理（内眼角中点与嘴角中点的垂直距离 / 脸宽）
    eyeSquintRight_smooth      AU6 近似

方法：
    1. 0.5 s 居中滚动中值预平滑（抑制眨眼与高频抖动）
    2. 逐 lag 的 Pearson r；lag>0 表示眼周落后于嘴角
    3. 以 smile_lift 的峰值为事件中心，前后各 1.5 s 作为事件窗口
    4. 仅当 r > 0.6 且 |lag| 未触及搜索边界时，才输出滞后结论

输出：outputs/curves/dynamics_good_proxy.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W3.1: 肌肉联动图与事件级滞后分析")
    parser.add_argument("--csv", type=Path, default=Path("outputs/features/features.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/curves/dynamics_good_proxy.png"))
    parser.add_argument("--fps", type=float, default=25.0)
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
    """返回最佳滞后、相关性与有效性标记（触及边界视为无效）。"""
    lags, rs = lagged_correlation(x, y, max_lag)
    best_idx = int(np.nanargmax(rs))
    lag = int(lags[best_idx])
    r = float(rs[best_idx])
    at_boundary = abs(lag) >= max_lag
    valid = (r > R_THRESHOLD) and not at_boundary
    return {"lag": lag, "r": r, "valid": valid, "at_boundary": at_boundary}


def describe(lag: int, fps: float) -> str:
    ms = lag / fps * 1000.0
    if lag > 0:
        return f"eye squint lags mouth smile by {lag} f ({ms:.0f} ms)"
    if lag < 0:
        return f"eye squint leads mouth smile by {abs(lag)} f ({abs(ms):.0f} ms)"
    return "synchronous (0 ms)"


def self_check() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    y = np.concatenate([np.zeros(5), x[:-5]]) + 0.05 * rng.normal(size=300)
    result = analyze(x, y, 20)
    print(f"self-check: synthetic eye lags mouth by 5 -> estimated {result['lag']:+d}")
    if result["lag"] != 5:
        raise RuntimeError("滞后符号自检失败")


def pick_peaks(values: np.ndarray, min_distance: int, percentile: float = 75.0) -> list[int]:
    """贪心选峰：高于分位阈值，且与已选峰间隔 >= min_distance。"""
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.csv.exists():
        raise FileNotFoundError(f"特征文件不存在: {args.csv}")

    df = pd.read_csv(args.csv)
    required = ["frame", LIFT_COL, MOUTH_COL, EYE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"缺少列 {missing}；请重新运行 extract_landmarks.py --blendshapes 生成新特征"
        )

    data = df[required].dropna().reset_index(drop=True)
    if len(data) < 20:
        raise ValueError(f"有效帧太少（{len(data)}）")

    # 0.5 s 居中滚动中值预平滑
    window = max(3, int(round(args.smooth_sec * args.fps)))
    smoothed = data[required[1:]].rolling(window=window, center=True, min_periods=1).median()
    frames = data["frame"].to_numpy()
    lift = smoothed[LIFT_COL].to_numpy()
    mouth = smoothed[MOUTH_COL].to_numpy()
    eye = smoothed[EYE_COL].to_numpy()

    self_check()

    overall_mouth = analyze(mouth, eye, args.max_lag)

    half = int(round(EVENT_HALF_WINDOW_SEC * args.fps))
    peaks = pick_peaks(lift, min_distance=half)
    events = []
    for p in peaks:
        lo, hi = max(0, p - half), min(len(frames), p + half)
        if hi - lo < max(10, args.max_lag * 2):
            continue
        res = analyze(mouth[lo:hi], eye[lo:hi], args.max_lag)
        events.append({"peak": int(frames[p]), "lo": int(frames[lo]), "hi": int(frames[hi]), **res})

    # ---------- 画图 ----------
    fig, ax1 = plt.subplots(figsize=(12, 5.2))

    for ev in events:
        ax1.axvspan(ev["lo"], ev["hi"], color="#ffb74d", alpha=0.12, lw=0)

    ax1.plot(frames, norm01(lift), color="#e65100", lw=1.5,
             label=f"{LIFT_COL} (normalized)")
    ax1.plot(frames, norm01(mouth), color="#2e7d32", lw=1.4, ls="--",
             label=f"{MOUTH_COL} (normalized)")
    ax1.set_xlabel("frame")
    ax1.set_ylabel("smile proxy (0–1, display only)")
    ax1.set_ylim(-0.03, 1.03)

    ax2 = ax1.twinx()
    ax2.plot(frames, eye, color="#1565c0", lw=1.5, label=EYE_COL)
    ax2.set_ylabel("eye squint", color="#1565c0")
    ax2.tick_params(axis="y", labelcolor="#1565c0")

    for ev in events:
        ax1.axvline(ev["peak"], color="0.55", ls=":", lw=1)

    verdict = "valid" if overall_mouth["valid"] else "WEAK / unreliable"
    title = (
        "Muscle Dynamics: Smile Proxies vs Eye Squint (W3.1)\n"
        f"overall: {MOUTH_COL} vs {EYE_COL}: lag={overall_mouth['lag']:+d} f "
        f"({overall_mouth['lag'] / args.fps * 1000:+.0f} ms), r={overall_mouth['r']:.3f} [{verdict}]"
    )

    lines = [title]
    for ev in events:
        if ev["valid"]:
            lines.append(f"event @{ev['peak']}: {describe(ev['lag'], args.fps)}, r={ev['r']:.2f}")
        else:
            lines.append(f"event @{ev['peak']}: r={ev['r']:.2f}, lag {ev['lag']:+d} -> not reliable")
    ax1.set_title("\n".join(lines), fontsize=9.5)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower right", fontsize=8)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)

    # ---------- 终端输出 ----------
    print(f"frames used: {len(data)} (of {len(df)}), pre-smooth {args.smooth_sec}s ({window} frames)")
    print(
        f"overall {MOUTH_COL} vs {EYE_COL}: lag={overall_mouth['lag']:+d} "
        f"({overall_mouth['lag'] / args.fps * 1000:+.0f} ms), r={overall_mouth['r']:.4f}"
    )
    if overall_mouth["valid"]:
        print(f"  -> {describe(overall_mouth['lag'], args.fps)}")
    else:
        reason = "lag hit search boundary" if overall_mouth["at_boundary"] else f"r <= {R_THRESHOLD}"
        print(f"  -> WEAK correlation ({reason}), no reliable lag conclusion")

    print(f"events (peaks in {LIFT_COL}): {[ev['peak'] for ev in events]}")
    for ev in events:
        status = "OK" if ev["valid"] else "weak"
        print(
            f"  event @{ev['peak']} [{ev['lo']}-{ev['hi']}]: lag={ev['lag']:+d} "
            f"({ev['lag'] / args.fps * 1000:+.0f} ms), r={ev['r']:.3f} [{status}]"
        )

    print(f"output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
