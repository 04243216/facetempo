"""W8.1：获取可用于 3D 预览的 ARKit Blendshape 模型。

默认使用 three.js 官方示例里的 `facecap.glb`（52 个 ARKit 形态键，约 325 KB）。
它由 jsDelivr 提供，可直接下载，适合本地演示；商用/再分发请自行核对 three.js
示例资产的许可。

如果传入 FBX 且本机安装了 Blender，脚本会生成一个 bpy 脚本并在后台静默转换：
导入 FBX → 清理相机/灯光 → 补一个 Principled BSDF 皮肤材质 → 导出 GLB。

用法：
    venv\\Scripts\\python.exe src\\translator\\auto_fetch_model.py
    venv\\Scripts\\python.exe src\\translator\\auto_fetch_model.py --force
    venv\\Scripts\\python.exe src\\translator\\auto_fetch_model.py --url <fbx-or-glb-url>
    venv\\Scripts\\python.exe src\\translator\\auto_fetch_model.py --url local.fbx --blender "C:\\...\\blender.exe"

没有 Blender 时，脚本会打印手动转换指南（也见 src/translator/BLENDER_GUIDE.md）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "src/translator/web_preview/model.glb"
DEFAULT_URL = (
    "https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/"
    "examples/models/gltf/facecap.glb"
)
BLENDER_CANDIDATES = [
    "blender",
    r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
]

BLENDER_SCRIPT = r'''
import sys
import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
if src.lower().endswith(".fbx"):
    bpy.ops.import_scene.fbx(filepath=src)
elif src.lower().endswith((".glb", ".gltf")):
    bpy.ops.import_scene.gltf(filepath=src)
else:
    raise SystemExit("unsupported input: " + src)

# 清理相机、灯光和空物体
for obj in list(bpy.data.objects):
    if obj.type in {"CAMERA", "LIGHT", "EMPTY"}:
        bpy.data.objects.remove(obj, do_unlink=True)

# 如果没有材质，补一个不塑料的皮肤基础材质
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    if obj.data.materials:
        continue
    mat = bpy.data.materials.new("FaceTempo_Skin")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.72, 0.45, 0.34, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.55
        for name in ("Subsurface Weight", "Subsurface"):
            if name in bsdf.inputs:
                bsdf.inputs[name].default_value = 0.05
                break
    obj.data.materials.append(mat)

bpy.ops.export_scene.gltf(
    filepath=dst,
    export_format="GLB",
    export_morph=True,
    export_apply=False,
)
print("exported:", dst)
'''


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="获取/转换 ARKit Blendshape 模型")
    parser.add_argument("--url", default=DEFAULT_URL, help="GLB/FBX 的 URL 或本地路径")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--blender", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="覆盖已存在的 model.glb")
    return parser.parse_args(argv)


def find_blender(explicit: Path | None) -> str | None:
    if explicit:
        return str(explicit) if Path(explicit).exists() else None
    for candidate in BLENDER_CANDIDATES:
        found = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if found:
            return str(found)
    return None


def download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not url.lower().startswith(("http://", "https://")):
        source = Path(url).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"本地文件不存在: {source}")
        shutil.copyfile(source, target)
        return target

    print(f"downloading: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "FaceTempo/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    target.write_bytes(data)
    return target


def inspect_glb(path: Path) -> dict:
    """读取 GLB 的 JSON chunk，统计 morph target 名称。"""
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError("文件太小，不是有效 GLB")
    magic, version, _ = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError("不是 GLB（magic 不匹配）")

    offset = 12
    chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
    offset += 8
    if chunk_type != 0x4E4F534A:
        raise ValueError("首个 chunk 不是 JSON")
    doc = json.loads(data[offset:offset + chunk_len].decode("utf-8"))

    names: list[str] = []
    for mesh in doc.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            target_names = (
                primitive.get("extras", {}).get("targetNames")
                or mesh.get("extras", {}).get("targetNames")
                or []
            )
            if target_names:
                names = target_names
                break
        if names:
            break

    if not names:
        # 退而求其次：用 targets 字典数量
        for mesh in doc.get("meshes", []):
            for primitive in mesh.get("primitives", []):
                if primitive.get("targets"):
                    return {
                        "version": version,
                        "target_count": len(primitive["targets"]),
                        "names": [],
                        "meshes": len(doc.get("meshes", [])),
                    }
    return {
        "version": version,
        "target_count": len(names),
        "names": names,
        "meshes": len(doc.get("meshes", [])),
    }


def convert_with_blender(src: Path, dst: Path, blender: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(BLENDER_SCRIPT)
        script = Path(fh.name)
    try:
        cmd = [blender, "--background", "--factory-startup", "--python", str(script),
               "--", str(src), str(dst)]
        print("running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    finally:
        script.unlink(missing_ok=True)


def print_manual_guide() -> None:
    print("\n未找到 Blender，已跳过 FBX 自动转换。手动方案：")
    print("  1) 下载 three.js 示例模型（52 个 ARKit 形态键，约 325 KB）：")
    print(f"     {DEFAULT_URL}")
    print("     保存为 src/translator/web_preview/model.glb 即可，无需 Blender。")
    print("  2) 或安装 Blender 后重跑本脚本，或参考 src/translator/BLENDER_GUIDE.md。")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out.resolve()

    if out.exists() and not args.force:
        info = inspect_glb(out)
        print(f"已存在: {out} ({out.stat().st_size / 1024:.1f} KB)")
        print(f"morph targets: {info['target_count']}，前 8 个: {info['names'][:8]}")
        print("如需覆盖请加 --force")
        return 0

    suffix = Path(args.url.split("?")[0]).suffix.lower()
    is_fbx = suffix == ".fbx"
    blender = find_blender(args.blender)

    with tempfile.TemporaryDirectory(prefix="facetempo_model_") as tmp:
        raw = Path(tmp) / ("source.fbx" if is_fbx else "source.glb")
        try:
            download(args.url, raw)
        except Exception as exc:
            print(f"下载失败: {exc}")
            print_manual_guide()
            return 1

        if is_fbx:
            if not blender:
                print_manual_guide()
                return 1
            try:
                convert_with_blender(raw, out, blender)
            except Exception as exc:
                print(f"Blender 转换失败: {exc}")
                print_manual_guide()
                return 1
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(raw, out)

    try:
        info = inspect_glb(out)
    except Exception as exc:
        print(f"生成的 GLB 无法解析: {exc}")
        return 1

    print(f"saved: {out} ({out.stat().st_size / 1024:.1f} KB)")
    print(f"morph targets: {info['target_count']}")
    if info["names"]:
        print("names:", ", ".join(info["names"][:10]), "...")
    if info["target_count"] < 40:
        print("警告：形态键数量偏少，3D 预览可能无法覆盖全部 52 通道。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
