"""W7：启动 3D 驱动预览的本地静态服务器。

浏览器出于安全限制不能用 file:// 读取 JSON，所以必须走 HTTP。

用法：
    venv\\Scripts\\python.exe src\\translator\\serve_preview.py
    venv\\Scripts\\python.exe src\\translator\\serve_preview.py --json outputs\\translator\\real_clean_04_driving.json

然后打开：
    http://127.0.0.1:8000/src/translator/web_preview/index.html
"""

from __future__ import annotations

import argparse
import http.server
import os
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = PROJECT_ROOT / "outputs/translator/real_clean_04_driving.json"
PREVIEW_PATH = "/src/translator/web_preview/index.html"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D 驱动预览本地服务器")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args(argv)


def make_handler(json_path: Path):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/data/driving.json":
                if not json_path.exists():
                    self.send_error(404, "Driving JSON not found")
                    return
                payload = json_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if path == "/":
                self.path = PREVIEW_PATH
            super().do_GET()

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, fmt, *args):
            # 只打印错误，避免每帧请求刷屏
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

    handler = make_handler(json_path)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}{PREVIEW_PATH}"

    print(f"serving : {PROJECT_ROOT}")
    print(f"data    : {json_path}")
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
