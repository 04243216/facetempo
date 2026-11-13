"""FaceTempo 演示界面（W5/W6）。

上传一段正脸视频 → 提取逐帧特征 → 计算归一化眼周 Jerk → 给出参考判断。

启动：
    venv\\Scripts\\python.exe src\\app\\app.py
然后打开 http://127.0.0.1:7860

注意（KI-003）：FaceLandmarker 在 Windows 上 close()/析构会阻塞约 42 秒，
因此这里使用进程级单例，只创建一次、永不 close；请求用锁串行化。
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))

import extract_landmarks as core  # noqa: E402
from plot_dynamics import TARGET_FPS, hf_ratio, jitter_stats, resample_signal  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"
JERK_THRESHOLD = 60.0

_LOCK = threading.Lock()
_LANDMARKER = None
_KEEPALIVE: list = []
_TIMESTAMP_MS = 0


def get_landmarker():
    """进程级单例；不要调用 close()（KI-003）。"""
    global _LANDMARKER
    if _LANDMARKER is None:
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
        _LANDMARKER = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        _KEEPALIVE.append(_LANDMARKER)
    return _LANDMARKER


def _extract_series(video_path: Path) -> pd.DataFrame:
    """逐帧提取演示所需的最小特征集。"""
    global _TIMESTAMP_MS

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("无法打开上传的视频，请确认编码为 H.264/MP4。")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(1000.0 / fps)))
    landmarker = get_landmarker()

    rows = []
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = landmarker.detect_for_video(image, _TIMESTAMP_MS)
        _TIMESTAMP_MS += step

        row = {
            "frame": frame_id,
            "face_found": 0,
            "mar": np.nan,
            "eyeSquintRight": np.nan,
            "eyeBlinkLeft": np.nan,
            "eyeBlinkRight": np.nan,
            "face_cx": np.nan,
            "face_cy": np.nan,
            "face_w": np.nan,
        }
        if result.face_landmarks:
            lm = result.face_landmarks[0]
            row["face_found"] = 1
            row.update(core.compute_features(lm))
            row.update(core.compute_geometry(lm, width, height))
        if result.face_blendshapes:
            for item in result.face_blendshapes[0]:
                if item.category_name in ("eyeSquintRight", "eyeBlinkLeft", "eyeBlinkRight"):
                    row[item.category_name] = item.score

        rows.append(row)
        frame_id += 1

    cap.release()
    if not rows:
        raise RuntimeError("视频中没有可读取的帧。")
    return pd.DataFrame(rows)


def _plot_result(mar: np.ndarray, jerk: float, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.2, 1]}
    )
    ax1.plot(mar, color="#d62728", lw=1.3)
    ax1.set_title("MAR（嘴部开合）")
    ax1.set_xlabel("frame")
    ax1.set_ylabel("MAR")
    ax1.grid(alpha=0.3)

    color = "#2e7d32" if jerk >= JERK_THRESHOLD else "#c62828"
    ax2.barh(["eye jerk"], [jerk], color=color, alpha=0.5, height=0.5)
    ax2.axvline(JERK_THRESHOLD, color="black", ls="--", lw=1)
    ax2.text(JERK_THRESHOLD, 0.32, f"  threshold {JERK_THRESHOLD:g}", fontsize=8)
    ax2.set_xlim(0, max(100.0, jerk * 1.2))
    ax2.set_title("归一化眼周 Jerk")
    ax2.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze(video_path: str):
    """Gradio 回调：返回 (Markdown 结论, 图片路径)。"""
    if not video_path:
        return "请先上传一段正脸视频。", None

    path = Path(video_path)
    with _LOCK:
        try:
            df = _extract_series(path)
        except Exception as exc:  # 给界面返回可读错误
            return f"处理失败：{exc}", None

        valid = df.dropna(subset=["face_cx", "face_cy", "face_w"])
        coverage = len(valid) / len(df)

        eye = df["eyeSquintRight"].to_numpy(dtype=float)
        blink = (df["eyeBlinkLeft"].fillna(0).to_numpy() > 0.5) | (
            df["eyeBlinkRight"].fillna(0).to_numpy() > 0.5
        )
        mask = blink.copy()
        for shift in (1, 2):
            mask[shift:] |= blink[:-shift]
            mask[:-shift] |= blink[shift:]

        # 缺失帧用插值补齐，保证重采样连续
        eye_filled = pd.Series(eye).interpolate(limit_direction="both").to_numpy()
        fps = 25.0
        cap = cv2.VideoCapture(str(path))
        real_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if real_fps and real_fps > 1:
            fps = float(real_fps)

        eye_rs, mask_rs = resample_signal(eye_filled, fps, mask)
        _, _, norm_jerk = jitter_stats(eye_rs, TARGET_FPS, mask_rs)
        hf = hf_ratio(eye_rs, TARGET_FPS)

        if not np.isfinite(norm_jerk):
            return "有效眼周信号不足，无法计算微颤指标。", None

        suspicious = norm_jerk < JERK_THRESHOLD
        verdict = "疑似 AI 生成 / 过度平滑" if suspicious else "符合真人微颤范围"
        warn = ""
        if coverage < 0.9:
            warn = f"\n\n> ⚠️ 人脸覆盖率仅 {coverage:.1%}，结果可能不可靠。"

        result = f"""### 诊断结果

| 指标 | 数值 |
|---|---|
| 归一化眼周 Jerk | **{norm_jerk:.2f}** |
| 眼周高频能量（5–12 Hz） | {hf:.3f} |
| 人脸覆盖率 | {coverage:.1%} |
| 参考阈值 | {JERK_THRESHOLD:g} |

**判断：{verdict}**

说明：该结果是基于本批 4v4 样本的参考阈值（真人最低 67.03，AI 最高 55.03），
仅用于演示，不构成确定性鉴定。低画质、静止不动或严重压缩也会降低该数值。{warn}"""

        out_path = Path(os.environ.get("TEMP", "/tmp")) / "facetempo_result.png"
        _plot_result(df["mar"].fillna(0).to_numpy(), norm_jerk, out_path)
        return result, str(out_path)


def build_demo():
    import gradio as gr

    with gr.Blocks(title="FaceTempo · 微颤诊断") as demo:
        gr.Markdown(
            "# FaceTempo\n"
            "**眼周微颤诊断演示**：上传一段正脸视频，系统计算归一化眼周 Jerk，"
            "并与真人/AI 参考区间比较。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="上传正脸视频（MP4, 3–15 秒）", sources=["upload"])
                button = gr.Button("开始诊断", variant="primary")
                gr.Markdown(
                    "> 参考：真人自然笑中位数 77.29；AI 中位数 35.63；阈值 60。\n"
                    "> 本工具仅用于研究演示，不作为鉴定结论。"
                )
            with gr.Column(scale=1):
                result = gr.Markdown("等待上传视频……")
                plot = gr.Image(label="MAR 曲线与 Jerk 评分", type="filepath")

        button.click(fn=analyze, inputs=video, outputs=[result, plot])
        gr.Markdown(
            "**方法**：MediaPipe FaceLandmarker（478 点 + 52 Blendshape）→ "
            "眨眼屏蔽 → 统一 25 fps 重采样 → 归一化 Jerk = mean|d²x|·fps² / std(x)。"
        )
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue(default_concurrency_limit=1)
    try:
        demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
    except KeyboardInterrupt:
        sys.stdout.flush()
        os._exit(0)
