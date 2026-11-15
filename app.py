"""Hugging Face Spaces 入口。

HF 会导入本文件并寻找 `demo`。若 3D 模型不在仓库里（.gitignore 已排除），
启动时自动下载 three.js 的 ARKit 示例模型；失败也不阻塞诊断功能。
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "src/translator/web_preview/model.glb"


def _ensure_model() -> None:
    if MODEL_PATH.exists():
        return
    try:
        from src.translator import auto_fetch_model

        print("model.glb missing, downloading ARKit sample model ...")
        auto_fetch_model.main([])
    except Exception as exc:  # 网络失败时回退程序化头
        print(f"model auto-fetch failed, 3D preview will fall back: {exc}")


_ensure_model()

os.environ.setdefault("FACETEMPO_CLOUD", "1")

from src.app.app import build_demo, is_cloud  # noqa: E402

demo = build_demo()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    cloud = is_cloud()
    kwargs = {
        "server_name": "0.0.0.0" if cloud else "127.0.0.1",
        "server_port": int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        "show_error": True,
    }
    if not cloud:
        kwargs["theme"] = demo._facetempo_theme
        kwargs["css"] = demo._facetempo_css
    demo.launch(**kwargs)
