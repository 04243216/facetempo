"""W7/W8：3D 驱动预览的本地静态服务器。

浏览器不能用 file:// 读取 JSON，所以必须走 HTTP。

单独使用：
    venv\\Scripts\\python.exe src\\translator\\serve_preview.py
    venv\\Scripts\\python.exe src\\translator\\serve_preview.py --json outputs\\translator\\real_clean_04_driving.json

然后打开：
    http://127.0.0.1:8000/src/translator/web_preview/index.html

Gradio 集成时由 app.py 以线程方式启动，并把该次上传生成的 JSON 放进 json_dir。
"""

from __future__ import annotations

import argparse
import http.server
import os
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = PROJECT_ROOT / "outputs/translator/real_clean_04_driving.json"
PREVIEW_PATH = "/src/translator/web_preview/index.html"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D 驱动预览本地服务器")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument(
        "--json_dir",
        type=Path,
        default=None,
        help="对外提供 /data/<name>.json 的目录；默认取 --json 所在目录",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args(argv)


def make_handler(json_dir: Path, default_json: Path | None = None):
    """json_dir 下的 JSON 通过 /data/<name>.json 暴露。"""
    root = json_dir.resolve()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

        def _send_json(self, path: Path) -> None:
            if not path.is_file():
                self.send_error(404, "Driving JSON not found")
                return
            payload = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/data/driving.json":
                if default_json is not None:
                    self._send_json(default_json.resolve())
                else:
                    self.send_error(404, "No default JSON")
                return
            if path.startswith("/data/"):
                name = unquote(path[len("/data/"):])
                if not name or "/" in name or "\\" in name:
                    self.send_error(400, "Invalid JSON name")
                    return
                target = (root / name).resolve()
                try:
                    target.relative_to(root)
                except ValueError:
                    self.send_error(403, "Forbidden")
                    return
                self._send_json(target)
                return
            if path == "/":
                self.path = PREVIEW_PATH
            super().do_GET()

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, fmt, *args):
            if str(args[1] if len(args) > 1 else "").startswith("4"):
                super().log_message(fmt, *args)

    return Handler


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    json_path = args.json.resolve()
    if not json_path.exists():
        print(f"JSON 不存在: {json_path}")
        print("请先运行: venv\\Scripts\\python.exe src\\translator\\blendshape_export.py")
        return 1

    json_dir = (args.json_dir or json_path.parent).resolve()
    handler = make_handler(json_dir, json_path)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}{PREVIEW_PATH}"

    print(f"serving : {PROJECT_ROOT}")
    print(f"json dir: {json_dir}")
    print(f"url     : {url}")
    print("Ctrl+C 停止。")

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.stdout.flush()
    code = main()
    sys.stdout.flush()
    os._exit(code)
