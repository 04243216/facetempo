# ARKit 52 Blendshape 模型：获取与 Blender 转换指南

## 方案 A：直接用免安装的 GLB（推荐）

three.js 官方示例包含一个带 **52 个 ARKit 形态键** 的头部模型 `facecap.glb`（约 325 KB）：

```
https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/models/gltf/facecap.glb
```

自动下载并校验：

```powershell
venv\Scripts\python.exe src\translator\auto_fetch_model.py
```

脚本会把模型保存到：

```
src/translator/web_preview/model.glb
```

刷新 3D 预览页面即可自动加载，页面右上角会显示 `GLB · N 通道`。

> 许可提示：该模型来自 three.js 示例资产。three.js 代码为 MIT，但示例模型可能另有许可；商用/再分发前请自行核对。若不能使用，请换成你拥有授权的 ARKit 模型。

## 方案 B：Blender 静默转换 FBX

先安装 Blender（4.1+），并确保 `blender.exe` 在 PATH 或常见安装目录。然后：

```powershell
venv\Scripts\python.exe src\translator\auto_fetch_model.py --url "https://example.com/your_arkit_head.fbx" --force
```

脚本会生成一个临时 bpy 脚本，以 `--background` 模式执行：

1. 导入 FBX；
2. 删除相机、灯光、空物体；
3. 没有材质时创建一个 Principled BSDF 皮肤材质（肤色、粗糙度 0.55、轻微次表面）；
4. 导出 `src/translator/web_preview/model.glb`（保留 morph targets）。

也可以指定 Blender 路径：

```powershell
venv\Scripts\python.exe src\translator\auto_fetch_model.py --url local.fbx --blender "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
```

## 方案 C：完全手动（无网络/无脚本时）

1. 打开 Blender → `File > Import > FBX`（或 glTF）。
2. 选中头部网格，右侧 `Object Data Properties > Shape Keys`，确认存在 52 个形态键。
3. 没有材质时：`Material Properties > New`，使用 Principled BSDF：
   - Base Color ≈ `#C68642`（自然肤色）；
   - Roughness ≈ 0.55；
   - Subsurface ≈ 0.05（Blender 4.x 为 Subsurface Weight）。
4. 删除相机、灯光、空物体，只保留头部与形态键。
5. `File > Export > glTF 2.0 (.glb)`：
   - 勾选 **Morph Targets**；
   - 不应用修改器（`Apply Modifiers` 关闭），否则形态键会丢；
   - 保存为 `src/translator/web_preview/model.glb`。
6. 刷新页面，右上角显示 `GLB · 52 通道` 即成功。

## 形态键命名要求

前端做归一化匹配：忽略大小写、下划线和连接符，并把 `Left/Right` 兼容为 `_L/_R`。
以下写法都能匹配到 `eyeBlinkLeft`：

```
eyeBlinkLeft / eyeBlink_L / eyeBlinkL
```

如果模型使用中文命名或完全不同的命名，需要在 `app.js` 的 `channelCandidates()` 里补充映射。

## 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 页面仍是程序化头 | 未找到 `model.glb` 或没有匹配的 morph target | 看浏览器 Console；确认文件名与路径 |
| 模型全黑/过曝 | 材质是纯色且场景无环境光 | 脚本已加环境贴图；仍异常时检查材质是否使用 Principled BSDF |
| 模型侧脸/倒置 | 导出坐标轴不同 | 在 Blender 导出时调整 `+Y Up`，或给 `model.glb` 外层加旋转 |
| 形态键没动 | 导出时未勾选 Morph Targets | 重新导出并勾选 |
