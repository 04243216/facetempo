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
import itertools
import math
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
TARGET_FPS = 25.0          # 微颤指标统一重采样到该帧率，消除采样率混杂

HEAD_SPEED_MAX = 0.5        # 脸宽/秒；超过视为大幅甩头
HEAD_DEV_MAX = 1.5          # 相对中位位置的最大偏移；慢漂移可接受，只拦镜头级位移
COVERAGE_MIN = 0.95         # 人脸覆盖率低于此值直接无效
EYE_STD_MIN = 0.008         # 眼通道标准差低于此值时不计算一致性，标记 N/A
EYE_AGREEMENT_MAX = 0.0     # 几何 EAR 与 Blendshape squint 必须负相关

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


def jitter_stats(signal: np.ndarray, fps: float,
                 mask: np.ndarray | None = None) -> tuple[float, float, float]:
    """返回 (速度, Jerk, 归一化 Jerk)，可屏蔽眨眼等瞬态。

    速度 = |一阶差分|均值 × fps
    Jerk = |二阶差分|均值 × fps²
    归一化 Jerk = Jerk / std，去掉动作幅度的影响（调用前已统一重采样到 TARGET_FPS）。
    屏蔽规则：差分窗口内任意端点被 mask 标记则丢弃该差分。
    """
    signal = np.asarray(signal, dtype=float)
    if mask is None:
        mask = np.zeros(signal.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    d1 = np.diff(signal)
    d2 = np.diff(signal, n=2)
    bad1 = mask[1:] | mask[:-1]
    bad2 = mask[2:] | mask[1:-1] | mask[:-2]

    speed = float(np.mean(np.abs(d1[~bad1]))) * fps if (~bad1).any() else float("nan")
    jerk = float(np.mean(np.abs(d2[~bad2]))) * fps * fps if (~bad2).any() else float("nan")
    std = float(np.std(signal[~mask])) if (~mask).any() else float("nan")
    norm_jerk = jerk / std if np.isfinite(std) and std > 0 else float("nan")
    return speed, jerk, norm_jerk


def resample_signal(signal: np.ndarray, fps: float,
                    mask: np.ndarray | None = None,
                    target_fps: float = TARGET_FPS):
    """线性插值到统一帧率；mask 以 0.5 为阈值同步重采样。"""
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    t = np.arange(n) / fps
    t_new = np.arange(0, t[-1], 1.0 / target_fps)
    xs = np.interp(t_new, t, signal)
    ms = None
    if mask is not None:
        ms = np.interp(t_new, t, np.asarray(mask, dtype=float)) > 0.5
    return xs, ms


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


def group_of(name: str) -> str:
    if name.startswith("ai"):
        return "ai"
    if "fake" in name:
        return "fake"
    return "real"


def permutation_test(a: list[float], b: list[float], n_perm: int = 20000,
                     seed: int = 0) -> tuple[float, float]:
    """单侧置换检验：H1 = mean(a) > mean(b)。

    组合数 <= 200000 时做精确枚举；否则蒙特卡洛（带 1e-9 浮点容差）。
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    observed = float(np.mean(a_arr) - np.mean(b_arr))
    pool = np.concatenate([a_arr, b_arr])
    n = len(pool)
    n_a = len(a_arr)
    total = math.comb(n, n_a)

    if total <= 200_000:
        count = 0
        for idx in itertools.combinations(range(n), n_a):
            mask = np.zeros(n, dtype=bool)
            mask[list(idx)] = True
            diff = float(np.mean(pool[mask]) - np.mean(pool[~mask]))
            if diff >= observed - 1e-9:
                count += 1
        return observed, count / total

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pool)
        if np.mean(perm[:n_a]) - np.mean(perm[n_a:]) >= observed - 1e-9:
            count += 1
    return observed, (count + 1) / (n_perm + 1)


def min_permutation_p(n_a: int, n_b: int) -> float:
    """给定样本量时，单侧置换检验能取得的最小 p 值。"""
    return 1.0 / math.comb(n_a + n_b, n_a)


def group_statistics(rows: list[dict]) -> None:
    groups: dict[str, list[dict]] = {"real": [], "fake": [], "ai": []}
    for row in rows:
        groups[group_of(row["name"])].append(row)

    print("=== group stats ===")
    for group, items in groups.items():
        if not items:
            continue
        jerks = [it["norm_jerk_e"] for it in items]
        hfs = [it["hf_e"] for it in items]
        print(f"  {group:4s} n={len(items)} "
              f"(gates_ok {sum(1 for it in items if it['gates_ok'])}, "
              f"trusted {sum(1 for it in items if it['valid'])}): "
              f"norm_jerk_e median={np.median(jerks):.3f} "
              f"values={[round(v, 3) for v in jerks]}")
        print(f"       hf_e median={np.median(hfs):.3f} values={[round(v, 3) for v in hfs]}")

    for metric, label in [("norm_jerk_e", "normalized eye jerk"),
                          ("hf_e", "eye HF energy (5-12 Hz)")]:
        real = [it[metric] for it in groups["real"]]
        ai = [it[metric] for it in groups["ai"]]
        if len(real) < 2 or len(ai) < 2:
            print(f"  {label}: not enough samples (real={len(real)}, ai={len(ai)})")
            continue
        observed, p_value = permutation_test(real, ai)
        floor = min_permutation_p(len(real), len(ai))
        print(
            f"  {label}: real-ai mean diff={observed:+.3f}, one-sided permutation "
            f"p={p_value:.4f} (minimum possible with n={len(real)}v{len(ai)}: {floor:.3f})"
        )
        print("    -> significant at 0.05" if p_value < 0.05
              else "    -> NOT significant (small n / overlap)")

        real_ok = [it[metric] for it in groups["real"] if it["gates_ok"]]
        ai_ok = [it[metric] for it in groups["ai"] if it["gates_ok"]]
        if len(real_ok) >= 2 and len(ai_ok) >= 2:
            if len(real_ok) != len(real) or len(ai_ok) != len(ai):
                obs_t, p_t = permutation_test(real_ok, ai_ok)
                print(f"    gates_ok-only: real n={len(real_ok)}, ai n={len(ai_ok)}, "
                      f"diff={obs_t:+.3f}, p={p_t:.4f}")
        else:
            print(f"    gates_ok-only: insufficient (real={len(real_ok)}, "
                  f"ai={len(ai_ok)})")



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
                "eye_aspect_ratio_geo", "face_cx", "face_cy", "face_w",
                "eyeBlinkLeft_smooth", "eyeBlinkRight_smooth"]
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

    # 眨眼瞬态会主导 eyeSquint 的高频能量，计算微颤时屏蔽（并膨胀 2 帧）
    blink = (
        (data["eyeBlinkLeft_smooth"].to_numpy() > 0.5)
        | (data["eyeBlinkRight_smooth"].to_numpy() > 0.5)
    )
    blink_mask = blink.copy()
    for shift in (1, 2):
        blink_mask[shift:] |= blink[:-shift]
        blink_mask[:-shift] |= blink[shift:]
    blink_pct = float(np.mean(blink))

    # 统一重采样到 TARGET_FPS，消除 24/25/60 fps 采样率混杂
    mouth_rs, _ = resample_signal(mouth_raw, fps)
    eye_rs, blink_rs = resample_signal(eye_raw, fps, blink_mask)

    overall = analyze(mouth, eye, max_lag)

    # ---- 数据质量门控 ----
    cx = data["face_cx"].to_numpy()
    cy = data["face_cy"].to_numpy()
    fw = data["face_w"].to_numpy()
    head_speed = float(np.mean(np.hypot(np.diff(cx), np.diff(cy)) / fw[:-1]) * fps)
    head_dev = float(np.max(np.hypot(cx - np.median(cx), cy - np.median(cy)) / fw))

    mouth_std = float(np.std(data[MOUTH_COL]))
    geo_std = float(np.std(data["eye_aspect_ratio_geo"]))
    bs_std = float(np.std(data[EYE_COL]))

    gate_reasons = []
    coverage = len(data) / max(len(df), 1)
    if coverage < COVERAGE_MIN:
        gate_reasons.append(f"coverage={coverage:.1%}<{COVERAGE_MIN:.0%}")
    if head_speed >= HEAD_SPEED_MAX:
        gate_reasons.append(f"head_speed={head_speed:.2f}>={HEAD_SPEED_MAX}")
    if head_dev >= HEAD_DEV_MAX:
        gate_reasons.append(f"head_dev={head_dev:.2f}>={HEAD_DEV_MAX}")

    if geo_std < EYE_STD_MIN or bs_std < EYE_STD_MIN:
        agreement = float("nan")
        agreement_status = "N/A (low variance)"
    else:
        agreement = float(np.corrcoef(data["eye_aspect_ratio_geo"], data[EYE_COL])[0, 1])
        agreement_status = "ok"
        if agreement >= EYE_AGREEMENT_MAX:
            gate_reasons.append(f"eye_agreement={agreement:.2f}>={EYE_AGREEMENT_MAX}")

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
        e0 = frames[lo] / fps
        e1 = frames[hi - 1] / fps
        i0 = max(0, int(round(e0 * TARGET_FPS)))
        i1 = min(len(mouth_rs), int(round(e1 * TARGET_FPS)) + 1)
        speed_m, jerk_m, _ = jitter_stats(mouth_rs[i0:i1], TARGET_FPS)
        speed_e, jerk_e, _ = jitter_stats(eye_rs[i0:i1], TARGET_FPS, blink_rs[i0:i1])

        events.append({
            "peak": int(frames[p]), "lo": int(frames[lo]), "hi": int(frames[hi - 1]),
            "baseline_m": base_m, "baseline_e": base_e,
            "speed_m": speed_m, "jerk_m": jerk_m,
            "speed_e": speed_e, "jerk_e": jerk_e,
            "ci_lo": ci_lo, "ci_hi": ci_hi, **res,
        })

    speed_m_all, jerk_m_all, norm_jerk_m = jitter_stats(mouth_rs, TARGET_FPS)
    speed_e_all, jerk_e_all, norm_jerk_e = jitter_stats(eye_rs, TARGET_FPS, blink_rs)
    hf_m = hf_ratio(mouth_rs, TARGET_FPS)
    hf_e = hf_ratio(eye_rs, TARGET_FPS)

    name = csv_path.stem[: -len("_features")] if csv_path.stem.endswith("_features") else csv_path.stem
    gates_ok = not gate_reasons
    trusted = overall["valid"] and gates_ok
    return {
        "name": name, "csv": csv_path, "out": out_path,
        "frames": frames, "lift": lift, "mouth": mouth, "eye": eye,
        "overall": overall, "events": events,
        "coverage": coverage,
        "head_speed": head_speed, "head_dev": head_dev,
        "agreement": agreement, "agreement_status": agreement_status,
        "mouth_std": mouth_std,
        "gate_reasons": gate_reasons, "trusted": trusted, "gates_ok": gates_ok,
        "speed_m": speed_m_all, "jerk_m": jerk_m_all,
        "speed_e": speed_e_all, "jerk_e": jerk_e_all,
        "norm_jerk_m": norm_jerk_m, "norm_jerk_e": norm_jerk_e,
        "hf_m": hf_m, "hf_e": hf_e, "blink_pct": blink_pct,
    }


def print_summary(summary: dict, fps: float, source: str | None) -> dict:
    overall, events = summary["overall"], summary["events"]
    valid_events = [e for e in events if e["valid"]]

    print(f"=== {summary['name']} ===")
    print(f"  fps: {fps:g}" + (f" (from {source})" if source else " (fallback)"))
    print(f"  frames: {len(summary['frames'])}, face coverage: {summary['coverage']:.1%}")
    agree_txt = ("N/A" if not np.isfinite(summary["agreement"])
                 else f"{summary['agreement']:+.2f}")
    print(
        f"  gates: coverage={summary['coverage']:.1%}, "
        f"head_speed={summary['head_speed']:.2f}/s, "
        f"head_dev={summary['head_dev']:.2f}, "
        f"eye_agreement={agree_txt} ({summary['agreement_status']}), "
        f"mouth_std={summary['mouth_std']:.3f} -> "
        f"{'OK' if not summary['gate_reasons'] else 'INVALID: ' + ', '.join(summary['gate_reasons'])}"
    )
    print(
        f"  jitter (@{TARGET_FPS:g}fps): mouth speed={summary['speed_m']:.2f}/s "
        f"jerk={summary['jerk_m']:.2f}/s^2 norm={summary['norm_jerk_m']:.2f} | "
        f"eye speed={summary['speed_e']:.2f}/s jerk={summary['jerk_e']:.2f}/s^2 "
        f"norm={summary['norm_jerk_e']:.2f}"
    )
    print(
        f"  hf energy (5-12 Hz): mouth={summary['hf_m']:.3f}, eye={summary['hf_e']:.3f} "
        f"| blink={summary['blink_pct']:.1%} (excluded from eye jitter)"
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
        "gates_ok": summary["gates_ok"],
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
    ax_norm.set_ylabel("jerk / std (1/s^2)")
    ax_norm.set_title("Amplitude-normalized micro-jitter (comparable)")
    ax_norm.set_xticks(x)
    ax_norm.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax_norm.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_group_comparison(rows: list[dict], out_path: Path) -> None:
    groups: dict[str, list[dict]] = {"real": [], "ai": [], "fake": []}
    for row in rows:
        groups[group_of(row["name"])].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    colors = {"real": "#2e7d32", "ai": "#c62828", "fake": "#f9a825"}

    for ax, metric, title in [
        (axes[0], "norm_jerk_e", "normalized eye jerk"),
        (axes[1], "hf_e", "eye HF energy (5-12 Hz)"),
    ]:
        data, ticklabels, used = [], [], []
        for group in ("real", "ai", "fake"):
            values = [r[metric] for r in groups[group]]
            if not values:
                continue
            data.append(values)
            ticklabels.append(f"{group}\nn={len(values)}")
            used.append(group)
        if not data:
            continue
        box = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False)
        ax.set_xticks(range(1, len(ticklabels) + 1))
        ax.set_xticklabels(ticklabels)
        for patch, group in zip(box["boxes"], used):
            patch.set_facecolor(colors[group])
            patch.set_alpha(0.25)
        rng = np.random.default_rng(1)
        for i, (values, group) in enumerate(zip(data, used), start=1):
            jitter = rng.uniform(-0.12, 0.12, len(values))
            ax.scatter(np.full(len(values), i) + jitter, values,
                       s=48, color=colors[group], edgecolor="black", linewidth=0.4, zorder=3)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("FaceTempo W4.3: micro-jitter, real vs AI")
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
    group_statistics(rows)
    comparison_path = args.out_dir / "final_comparison.png"
    plot_group_comparison(rows, comparison_path)
    print(f"comparison figure: {comparison_path}")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
