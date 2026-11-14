"""FaceTempo 演示界面（W5/W6）。

上传一段正脸视频 → 提取逐帧特征 → 计算归一化眼周 Jerk → 给出参考判断。

启动：
    venv\\Scripts\\python.exe src\\app\\app.py
然后打开 http://127.0.0.1:7860

注意（KI-003）：FaceLandmarker 在 Windows 上 close()/析构会阻塞约 42 秒，
因此这里使用进程级单例，只创建一次、永不 close；请求用锁串行化。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from pathlib import Path
from urllib.parse import quote

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
from fastapi.staticfiles import StaticFiles
from starlette.routing import Mount

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))

import extract_landmarks as core  # noqa: E402
from plot_dynamics import TARGET_FPS, hf_ratio, jitter_stats, resample_signal  # noqa: E402
from translator import blendshape_export as exporter  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"
JSON_DIR = PROJECT_ROOT / "outputs" / "app_json"
WEB_PREVIEW_DIR = SRC_DIR / "translator" / "web_preview"
JERK_THRESHOLD = 60.0

_CHANNELS: list[str] = []
# 图表统一使用英文标签，避免浏览器翻译/中文字体导致的乱码
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

_LOCK = threading.Lock()
_LANDMARKER = None
_KEEPALIVE: list = []
_TIMESTAMP_MS = 0
_STATIC_MOUNTED: set[int] = set()
_PREVIEW_PATCHED = False

CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
#hero { text-align: center; padding: 10px 0 0 0; }
#hero h1 { font-size: 2.1rem; letter-spacing: .02em; margin: 0; color: #0b2a4a; }
#hero p { color: #475569; margin: 6px 0 0 0; font-size: .98rem; }
#scanline {
  height: 3px; width: 100%; border-radius: 2px; margin: 14px 0 20px 0;
  background: linear-gradient(90deg, #0b2a4a, #38bdf8, #0b2a4a);
  background-size: 200% 100%;
  animation: scan 3s linear infinite;
}
@keyframes scan { 0% { background-position: 0% 0; } 100% { background-position: 200% 0; } }
#baseline {
  background: #f1f5f9; border-left: 4px solid #0b2a4a;
  padding: 12px 16px; border-radius: 8px; font-size: .9rem; line-height: 1.6;
}
#result-card { min-height: 260px; }
button.primary {
  background: #0b2a4a !important; color: #ffffff !important;
  border-radius: 10px !important; font-weight: 600 !important;
  letter-spacing: .02em;
}
button.primary:hover { background: #123c66 !important; }
footer { display: none !important; }

/* 强制深色时的局部配色（Gradio 自带 dark 变量） */
.dark #hero h1 { color: #e2e8f0; }
.dark #hero p { color: #94a3b8; }
.dark #baseline { background: #132033; color: #cbd5e1; border-left-color: #38bdf8; }
.dark button.primary { background: #1d4ed8 !important; }
.dark button.primary:hover { background: #2563eb !important; }
"""


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


def _extract_series(video_path: Path, progress=None) -> pd.DataFrame:
    """逐帧提取演示所需的最小特征集。"""
    global _TIMESTAMP_MS

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("无法打开上传的视频，请确认编码为 H.264/MP4。")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
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
            shapes = result.face_blendshapes[0]
            names = [
                "neutral" if c.category_name == "_neutral" else c.category_name
                for c in shapes
            ]
            if not _CHANNELS:
                _CHANNELS[:] = names
            for name, item in zip(names, shapes):
                row[name] = item.score

        rows.append(row)
        frame_id += 1

        if progress is not None and total > 0 and frame_id % 10 == 0:
            progress(min(0.88, frame_id / total), desc=f"追踪面部关键点 {frame_id}/{total}")

    cap.release()
    if not rows:
        raise RuntimeError("视频中没有可读取的帧。")
    if progress is not None:
        progress(0.92, desc="计算微颤指标…")
    return pd.DataFrame(rows)


def _plot_result(mar: np.ndarray, jerk: float, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.2, 1]}
    )
    ax1.plot(mar, color="#d62728", lw=1.3)
    ax1.set_title("MAR curve (mouth opening)")
    ax1.set_xlabel("frame")
    ax1.set_ylabel("MAR")
    ax1.grid(alpha=0.3)

    color = "#2e7d32" if jerk >= JERK_THRESHOLD else "#c62828"
    ax2.barh(["eye jerk"], [jerk], color=color, alpha=0.55, height=0.5)
    ax2.axvline(JERK_THRESHOLD, color="#0b2a4a", ls="--", lw=1)
    ax2.set_xlim(0, max(100.0, jerk * 1.2))
    ax2.set_title(f"Normalized eye jerk (threshold {JERK_THRESHOLD:g})")
    ax2.set_xlabel("jerk / std (1/s^2)")
    ax2.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_prompt_fix(jerk_score: float, missing_features: list[str] | None = None) -> str:
    """根据 Jerk 分数与缺失特征，生成可直接复制的 AI 视频优化提示词。"""
    missing_features = missing_features or []
    if jerk_score >= JERK_THRESHOLD:
        return "视频面部动态真实，微颤能量达标，无需优化。"

    issues = "、".join(missing_features) if missing_features else "眼周微颤不足"
    return f"""发现缺陷：{issues}（Jerk = {jerk_score:.2f}，低于参考阈值 {JERK_THRESHOLD:g}）。

建议向 AI 视频工具输入以下优化提示词（可直接复制）：

电影级极高画质，正脸特写，4K，自然光。人物面对镜头自然微笑并轻轻眨眼。
重点要求：
1. 面部肌肉要有真实的微小抽动（micro-tremors），不要整张脸像橡皮膜一样平滑；
2. 微笑时眼周肌肉（眼轮匝肌）自然收缩，出现轻微眯眼与卧蚕；
3. 嘴角上扬与回落要有自然的启动/收尾过渡，避免机械式匀速运动；
4. 保留皮肤纹理、细小表情纹和不规则眨眼节奏；
5. 避免塑料感、过度磨皮、冻结表情和循环播放感。

负面提示：塑料感、橡皮膜、过度平滑、假笑、表情冻结、AI 磨皮。"""


def _export_driving_json(df: pd.DataFrame, video: Path, fps: float) -> str:
    """把本次上传的 52 通道平滑后写成驱动 JSON，返回文件名。"""
    channel_cols = list(_CHANNELS)
    missing = [c for c in channel_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"本次提取缺少通道: {missing}")
    if len(channel_cols) != 52:
        raise RuntimeError(f"期望 52 个 Blendshape 通道，实际 {len(channel_cols)} 个")

    smoothed = df[channel_cols].rolling(3, center=True, min_periods=1).median()
    smoothed.columns = [f"{c}_smooth" for c in channel_cols]
    json_df = pd.concat([df[["frame", "face_found"]], smoothed], axis=1)

    payload = exporter.build_payload(json_df, video, fps)
    exporter.validate_payload(payload)

    JSON_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{video.stem}_{uuid.uuid4().hex[:8]}.json"
    with (JSON_DIR / name).open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)

    old_files = sorted(JSON_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    for old in old_files[:-20]:
        old.unlink(missing_ok=True)
    return name


def _add_static_routes(app) -> None:
    """把 /preview 与 /data 插到路由最前面（Gradio 有 catch-all）。"""
    key = id(app)
    if key in _STATIC_MOUNTED:
        return
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    app.router.routes.insert(
        0,
        Mount("/preview", app=StaticFiles(directory=str(WEB_PREVIEW_DIR), html=True),
              name="preview"),
    )
    app.router.routes.insert(
        0,
        Mount("/data", app=StaticFiles(directory=str(JSON_DIR)), name="data"),
    )
    _STATIC_MOUNTED.add(key)


def _patch_gradio_app_factory() -> None:
    """Gradio 在 launch() 时会重建 App；在工厂出口注入静态路由。"""
    global _PREVIEW_PATCHED
    if _PREVIEW_PATCHED:
        return
    from gradio.routes import App as GradioApp

    original = GradioApp.create_app

    def create_app(blocks, *args, **kwargs):
        app = original(blocks, *args, **kwargs)
        _add_static_routes(app)
        return app

    GradioApp.create_app = staticmethod(create_app)
    _PREVIEW_PATCHED = True


def viewer_html(json_name: str) -> str:
    url = f"/preview/index.html?json={quote(json_name, safe='')}"
    return (
        f'<iframe src="{url}" loading="lazy" '
        'style="width:100%;height:560px;border:1px solid #1e293b;'
        'border-radius:12px;background:#070d19"></iframe>'
    )


def analyze(video_path, progress=None):
    """核心回调：返回 (Markdown 结论, 图片路径, 3D 预览 HTML, 提示词)。"""
    if isinstance(video_path, dict):
        video_path = video_path.get("path") or video_path.get("name")
    if not video_path:
        return "请先上传一段正脸视频。", None, "", ""

    path = Path(video_path)
    with _LOCK:
        try:
            df = _extract_series(path, progress)
        except Exception as exc:  # 给界面返回可读错误
            return f"处理失败：{exc}", None, "", ""

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

        eye_filled = pd.Series(eye).interpolate(limit_direction="both").to_numpy()
        cap = cv2.VideoCapture(str(path))
        real_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        fps = float(real_fps) if real_fps and real_fps > 1 else 25.0

        eye_rs, mask_rs = resample_signal(eye_filled, fps, mask)
        _, _, norm_jerk = jitter_stats(eye_rs, TARGET_FPS, mask_rs)
        hf = hf_ratio(eye_rs, TARGET_FPS)

        if not np.isfinite(norm_jerk):
            return "有效眼周信号不足，无法计算微颤指标。", None, "", ""

        mar_std = float(np.nanstd(df["mar"].to_numpy(dtype=float)))
        missing_features = []
        if norm_jerk < JERK_THRESHOLD:
            missing_features.append("眼周微颤不足")
        if hf < 0.02:
            missing_features.append("眼周高频能量偏低")
        if mar_std < 0.05:
            missing_features.append("嘴部表情幅度偏小")
        if coverage < 0.95:
            missing_features.append("人脸稳定性不足")
        prompt = generate_prompt_fix(norm_jerk, missing_features)

        suspicious = norm_jerk < JERK_THRESHOLD
        verdict = "疑似 AI 生成 / 过度平滑" if suspicious else "符合真人微颤范围"
        badge = "低微颤" if suspicious else "真人区间"
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

**判断：{verdict}**（{badge}）

说明：阈值来自主队列 4v4（真人最低 67.03，AI 最高 55.03，取中点 60），
仅用于演示，不构成确定性鉴定。低画质、静止不动或严重压缩也会降低该数值。{warn}"""

        out_path = Path(os.environ.get("TEMP", "/tmp")) / "facetempo_result.png"
        _plot_result(df["mar"].fillna(0).to_numpy(), norm_jerk, out_path)

        try:
            json_name = _export_driving_json(df, path, fps)
            viewer = viewer_html(json_name)
        except Exception as exc:  # 3D 预览失败不影响诊断结果
            viewer = (
                '<div style="padding:16px;border:1px solid #7f1d1d;border-radius:10px;'
                'color:#fca5a5">3D 预览生成失败：' + str(exc) + "</div>"
            )

        if progress is not None:
            progress(1.0, desc="完成")
        return result, str(out_path), viewer, prompt


def build_demo():
    import gradio as gr

    _patch_gradio_app_factory()

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    with gr.Blocks(title="FaceTempo · 微颤诊断") as demo:
        gr.HTML(
            """
            <div id="hero">
              <h1>面速 FaceTempo</h1>
              <p>面部位移与表情节奏的量化诊断 · 眼周微颤视角下的真人与 AI 人脸</p>
            </div>
            <div id="scanline"></div>
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                video = gr.Video(
                    label="上传正脸视频（MP4，3–15 秒）",
                    sources=["upload"],
                    height=320,
                )
                button = gr.Button("🚀 开始扫描诊断", variant="primary", size="lg")
                gr.HTML(
                    """
                    <div id="baseline">
                      <b>参考基准（主队列 4v4）</b><br>
                      真人自然笑：67.03 – 105.70 ｜ AI 生成：21.63 – 55.03<br>
                      阈值 <b>60</b> = 两组区间中点；仅作演示提示。
                    </div>
                    """
                )
            with gr.Column(scale=5):
                result = gr.Markdown(
                    "### 等待输入\n上传一段正脸视频后点击 **开始扫描诊断**。",
                    elem_id="result-card",
                )

        gr.Markdown("### 信号曲线")
        plot = gr.Image(
            label="MAR 曲线与归一化眼周 Jerk",
            type="filepath",
            height=340,
        )

        gr.Markdown("### 3D 复现")
        viewer = gr.HTML(
            "<div style='padding:26px;text-align:center;color:#94a3b8;"
            "border:1px dashed #1e293b;border-radius:12px'>"
            "诊断完成后，这里会加载 3D 驱动预览（52 通道驱动）。</div>"
        )

        gr.Markdown("### 反向优化提示词")
        prompt_box = gr.Textbox(
            label="🎯 诊断后自动生成（可一键复制）",
            lines=9,
            buttons=["copy"],
            placeholder="若判定为 AI 生成，这里会出现可直接用于重新生成的提示词……",
        )

        gr.Markdown(
            "**方法**：MediaPipe FaceLandmarker（478 点 + 52 Blendshape）→ "
            "眨眼屏蔽 → 统一 25 fps 重采样 → 归一化 Jerk = mean|d²x|·fps² / std(x)。"
        )

        def run_analysis(video_file, progress=gr.Progress()):
            return analyze(video_file, progress)

        button.click(
            fn=run_analysis,
            inputs=video,
            outputs=[result, plot, viewer, prompt_box],
        )

        demo.load(
            fn=None,
            inputs=None,
            outputs=None,
            js="() => { document.documentElement.classList.add('dark'); }",
        )

    # Gradio 6 起 theme/css 通过 launch() 传入
    demo._facetempo_theme = theme
    demo._facetempo_css = CSS
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue(default_concurrency_limit=1)
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            show_error=True,
            theme=demo._facetempo_theme,
            css=demo._facetempo_css,
        )
    except KeyboardInterrupt:
        sys.stdout.flush()
        os._exit(0)
