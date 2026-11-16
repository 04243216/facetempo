# FaceTempo 完整制作流程与 AI 交接报告

> 版本：1.0 ｜ 生成日期：2026-11-16 ｜ 面向对象：后续接手/复现/审计本项目的 AI Agent 与开发者
> 仓库：<https://github.com/04243216/facetempo> ｜ 本地路径：`E:\FACETEMPO`
> 本文是项目的**唯一叙事型事实来源**；代码与原始输出优先于本文，若二者冲突，以代码行为为准并修正本文。

---

## 0. 给 AI 的阅读指南

按以下顺序阅读，可最快建立可执行的世界模型：

1. **§2 机器可读摘要**：版本、路径、指标、门槛、结果，全部是结构化字段。
2. **§3 术语与符号**：避免把 `jerk`、`norm_jerk`、`r`、`lag` 混用。
3. **§10 算法规范**：唯一 canonical 的公式、预处理、门控与统计口径。
4. **§11 实验结果**：最终数字；**§11.6 废弃数字**必须忽略。
5. **§13 失败与修复日志**：不要重复踩坑。
6. **§14 复现步骤**：按命令执行即可重建。
7. **§20 AI 交接规则**：修改代码前必须遵守的不变量。

如果只能读一段：读 §2 + §10.4 + §11.1 + §20。

### 0.1 事实优先级

```
代码实际行为 > outputs/ 下真实产物 > 本文 > docs/ 旧章节 > 聊天记录
```

旧文档中与 §11.6 冲突的数值一律视为**已废弃**。

---

## 1. 项目一句话

**FaceTempo（面速）** 用眼周高频微颤（归一化 Jerk）区分真人视频与 AI 生成人脸，并把真人表演导出为 52 通道 ARKit 风格 JSON，可驱动 3D 头部复现。

关键结论（仅限本项目受控样本）：

- 主队列 4 真人 vs 4 AI：中位数 **77.29 vs 35.63**，完全无重叠，精确单侧置换 **p = 0.0143**。
- 独立盲测 2 真人 vs 2 AI：**零重叠**（40.51 / 96.20 vs 26.65 / 18.31），p = 0.1667 是 2v2 数学下限，只能算方向性支持。
- 真人假笑 4 条：中位数 **59.62**，介于真人与 AI 之间。
- 云端只保留纯诊断版；3D 复现保留在本地。

---

## 2. 机器可读摘要（MACHINE SUMMARY）

```json
{
  "schema": "facetempo.handoff.v1",
  "generated": "2026-11-16",
  "project": "FaceTempo",
  "repo": "https://github.com/04243216/facetempo",
  "git": {
    "baseline_head_at_writing": "cd70c96",
    "note": "report commit comes after this baseline; run `git log -1` for the current head"
  },
  "canonical_metric": {
    "name": "normalized_eye_jerk",
    "signal_column": "eyeSquintRight",
    "blink_mask": {"threshold": 0.5, "dilate_frames": 2, "source": "eyeBlinkLeft_smooth / eyeBlinkRight_smooth"},
    "resample_fps": 25.0,
    "derivative": "second difference per frame",
    "formula": "mean(abs(x[t+2]-2*x[t+1]+x[t])) * fps^2 / std(x)",
    "unit": "1/s^2",
    "higher_means": "more micro-tremor / more human-like in this dataset"
  },
  "runtime": {
    "python": "3.12.10",
    "mediapipe": "1.2.2",
    "opencv": "5.0.0",
    "numpy": "2.5.4",
    "pandas": "3.1.1",
    "matplotlib": "3.11.2",
    "gradio": "6.32.0",
    "platform": "Windows 10/11 + CPU"
  },
  "gates": {
    "coverage_min": 0.95,
    "head_speed_max_face_width_per_sec": 0.5,
    "head_dev_max_face_width": 1.5,
    "eye_agreement": "corr(eye_aspect_ratio_geo, eyeSquintRight_smooth) < 0; N/A if either std < 0.008",
    "mouth_std": "recorded only, not a hard gate"
  },
  "cohorts": {
    "main_real": ["real_clean_02", "real_clean_03", "real_clean_04", "real_clean_05"],
    "main_ai": ["ai_clean_01", "ai_clean_02", "ai_clean_03", "ai_clean_04"],
    "main_fake": ["real_03_fake_smile", "real_fake_01", "real_fake_02", "real_fake_03"],
    "excluded": ["real_clean_01 (coverage 75.1%, head jump)"],
    "holdout_real": ["real_holdout_01", "real_holdout_02"],
    "holdout_ai": ["ai_holdout_01", "ai_holdout_02"]
  },
  "results": {
    "main_real_norm_jerk": [105.70, 67.59, 86.99, 67.03],
    "main_real_median": 77.29,
    "main_ai_norm_jerk": [27.02, 21.63, 44.25, 55.03],
    "main_ai_median": 35.63,
    "main_fake_median": 59.62,
    "main_p_one_sided_exact": 0.0143,
    "main_p_floor_4v4": 0.0142857,
    "main_hf_energy_p": 0.0429,
    "holdout_real_norm_jerk": [40.51, 96.20],
    "holdout_ai_norm_jerk": [26.65, 18.31],
    "holdout_p": 0.1667,
    "holdout_p_floor_2v2": 0.1666667,
    "separation_main": "zero overlap",
    "separation_holdout": "zero overlap"
  },
  "artifacts": {
    "json_sample": "outputs/translator/real_clean_04_driving.json (203 frames, 52 channels, ~331 KB)",
    "technical_report": "README_technical_report.md",
    "portfolio_report": "FaceTempo_Report.html",
    "deploy_guide": "DEPLOY_GUIDE.md",
    "demo": "src/app/app.py",
    "preview": "src/translator/web_preview/index.html",
    "model_task": "models/face_landmarker.task (3,758,596 bytes)",
    "model_glb": "src/translator/web_preview/model.glb (332,808 bytes, 52 morphs, ignored by git)"
  },
  "cloud_mode": {
    "env": "FACETEMPO_CLOUD=1",
    "behavior": "hide 3D, skip JSON export and static routes, launch without theme/css",
    "local_behavior": "FACETEMPO_CLOUD unset or 0 -> full 3D"
  }
}
```

---

## 3. 术语与符号

| 术语 | 英文 / 字段 | 定义 |
|---|---|---|
| 微颤 | micro-tremor / jerk | 单位幅度下的高频加速度能量，本项目主指标 |
| 归一化 Jerk | `norm_jerk` | `mean(|Δ²x|)·fps² / std(x)`，单位 1/s² |
| 眼周通道 | `eyeSquintRight` | MediaPipe Blendshape，值域 0–1 |
| 眨眼屏蔽 | blink mask | `eyeBlink*_smooth > 0.5` 的帧及前后 2 帧 |
| 统一帧率 | resample | 所有微颤信号线性重采样到 25 fps |
| 质量门控 | gates | 覆盖率、头动、眼通道一致性；决定样本是否进入统计 |
| 主队列 | main cohort | 4 真人 + 4 AI（+4 假笑） |
| 盲测 | holdout | 不参与调参的 2 真人 + 2 AI |
| 精确置换 | exact permutation | 枚举全部标签组合的 p 值 |
| 起势 / 维持 | onset / maintenance | 首个微笑事件 / 后续有效事件 |
| Blendshape | 52 channels | MediaPipe 输出，命名兼容 ARKit |

符号：`x` 为原始 `eyeSquintRight` 序列；`fps` 为 25；`std` 为非眨眼帧标准差；`r` 为 Pearson 相关；`lag > 0` 表示眼周落后嘴角。

---

## 4. 问题定义与假设

### 4.1 目标

不是判断单帧“像不像”，而是判断**时间维度上“动得对不对”**：

1. 用消费级视频 + 开源模型量化眼周微颤。
2. 在同一采集协议下区分真人自然表情与 AI 生成表情，并在独立样本上复现方向。
3. 把真人表演转成标准机器语言，驱动 3D 头部。

### 4.2 核心假设

> AI 生成人脸在静态画质上足够好，但缺少真人肌肉控制中的不自主微颤，因此眼周高频加速度能量显著更低。

该假设只在**本批 4v4 + 2v2 受控样本**上成立，不能外推所有工具/人群。

### 4.3 非目标

- 不做身份识别、情绪分类、医学诊断。
- 不宣称“所有 AI 都低于所有真人”。
- 不把 2v2 盲测包装成显著性验证。
- 不承诺 3D 是影视级复刻（当前为程序化头 + 可选 ARKit GLB）。

---

## 5. 系统架构

### 5.1 数据流

```text
视频 (mp4)
  │
  ├─ OpenCV 解码
  └─ MediaPipe FaceLandmarker (Tasks API, VIDEO 模式)
        ├─ 478 normalized landmarks
        └─ 52 blendshapes (含 _neutral)
  │
  ├─ 几何特征（EAR / MAR / smile_lift / mouth_width / eye_aspect_ratio_geo）
  ├─ 52 blendshape × (raw + 3-frame median smooth)
  └─ 人脸框中心/尺寸（face_cx/cy/w/h）
  │
  ├─ 质量门控：覆盖率 / 头动 / 眼通道一致性
  ├─ 眨眼屏蔽 + 25 fps 重采样
  ├─ 归一化眼周 Jerk + 5–12 Hz 高频能量
  └─ 诊断：阈值 60 / 分级 / 提示词
  │
  └─ 52 通道 JSON → Three.js / GLB morph targets → 3D 复现（本地）
```

### 5.2 组件表

| 层 | 文件 | 职责 |
|---|---|---|
| 提取 | `src/extract_landmarks.py` | 单文件/目录批量提取，输出 overlay、features.csv、curve.png |
| 诊断 | `src/plot_dynamics.py` | 批量读取特征，门控、Jerk、置换检验、事件级 lag、图 |
| 导出 | `src/translator/blendshape_export.py` | 特征 CSV → 52 通道 driving JSON |
| 校验 | `src/translator/verify_json.py` | JSON 结构/NaN/时间戳校验 + 曲线 |
| 3D 预览 | `src/translator/web_preview/*` | Three.js 播放器，GLB/程序化头 |
| 模型获取 | `src/translator/auto_fetch_model.py` | 下载/转换 model.glb |
| Demo | `src/app/app.py` | Gradio All-in-One，诊断 + 提示词 + 3D（本地） |
| 云端入口 | `app.py` | HF/ModelScope 入口，`FACETEMPO_CLOUD=1` |
| 部署 | `DEPLOY_GUIDE.md`、`requirements*.txt`、`packages.txt` | 云环境 |

---

## 6. 环境与依赖

### 6.1 版本（canonical）

| 组件 | 版本 | 备注 |
|---|---|---|
| Python | 3.12.10 | mediapipe 无 3.14 wheel |
| mediapipe | 1.2.2 | Tasks API；`mp.solutions` 已移除 |
| opencv-python | 5.0.0 | 本地；云端用 headless |
| numpy | 2.5.4 | |
| pandas | 3.1.1 | |
| matplotlib | 3.11.2 | |
| gradio | 6.32.0 | theme/css 在 `launch()` |
| playwright | 1.65.1 | 仅用于截图/验证，不是运行时依赖 |

### 6.2 安装

```powershell
cd E:\FACETEMPO
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements-dev.txt
# 或者云端
venv\Scripts\python.exe -m pip install -r requirements.txt
```

模型：

```powershell
# 关键点模型（3.58 MB）
# 官方：https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
# 已验证 SHA256：64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff

# 3D 头部（可选，325 KB，52 morphs）
venv\Scripts\python.exe src\translator\auto_fetch_model.py
```

### 6.3 平台坑（必须知道）

1. **Python 3.14 装不了 mediapipe**：没有 wheel，所有流程必须用 3.12。
2. **mediapipe 1.x 没有 `mp.solutions`**：必须用 `mp.tasks.vision.FaceLandmarker`。
3. **Gradio 6 的 `theme/css` 在 `launch()`**：放 `Blocks()` 里会有警告；云端 Gradio 版本低时不要传。
4. **OpenCV GUI 在 Linux 云端不可用**：用 `opencv-python-headless` + `packages.txt` 的 libGL。
5. **`model.glb` 用了 KTX2 + meshopt**：需要 ES module 的 GLTFLoader + KTX2Loader + Basis + MeshoptDecoder；旧 UMD loader 会报 `setKTX2Loader must be called`。

---

## 7. 数据

### 7.1 主队列

| 组 | 文件 | 规格 | 说明 |
|---|---|---|---|
| 真人自然 | `real_clean_02`–`real_clean_05` | 24–60 fps，8–14 s | 正脸、固定机位为主 |
| AI | `ai_clean_01`–`ai_clean_04` | 24 fps，10 s，960×720 / 1108×832 | 不同工具生成，正脸 |
| 真人假笑 | `real_03_fake_smile`、`real_fake_01`–`03` | 25 fps，25–43 s | 刻意挤出的笑 |
| 无效 | `real_clean_01` | 25 fps | 覆盖率 75.1%，镜头级跳变 |

### 7.2 独立盲测

`real_holdout_01/02`、`ai_holdout_01/02`，与主队列不同来源/人物，全部 100% 覆盖且通过门控。

### 7.3 采集协议（后续补数据必须遵守）

- 正脸、单人、固定机位、光线均匀；
- 建议 ≥10 s，至少两次完整微笑（为了起势/维持）；
- AI 片段记录工具、版本、提示词；
- 真人需要授权记录；
- 文件名使用 `real_*` / `ai_*` / `*fake*`，`plot_dynamics` 按前缀分组。

### 7.4 数据与隐私

`data/raw/`、`outputs/` 大文件、`model.glb` 均不入 git。仓库只保留代码、文档、小图和报告。

---

## 8. 代码地图（逐文件）

```text
E:\FACETEMPO
├── app.py                         # 云端入口：设 FACETEMPO_CLOUD=1，build_demo，launch
├── requirements.txt               # 云端精简依赖
├── requirements-dev.txt           # 本地完整 freeze
├── packages.txt                   # libgl1 / libglib2.0-0 / libsm6 / libxext6
├── DEPLOY_GUIDE.md                # HF / ModelScope 部署
├── README_technical_report.md     # 技术报告（W12 素材）
├── FaceTempo_Report.html          # 单文件作品集报告
├── docs/
│   ├── metrics_spec.md            # 指标规范 + 各轮实测
│   ├── known_issues.md            # KI-001..004
│   └── diagnosis_report_template.md
├── src/
│   ├── extract_landmarks.py       # 关键点/Blendshape 提取 + 批量 + 参数化
│   ├── plot_dynamics.py           # 门控/统计/事件级滞后/图
│   ├── translator/
│   │   ├── blendshape_export.py   # 52 通道 JSON
│   │   ├── verify_json.py         # JSON 校验
│   │   ├── auto_fetch_model.py    # 下载/转换 model.glb
│   │   ├── serve_preview.py       # 独立预览服务（本地）
│   │   ├── BLENDER_GUIDE.md       # 手动 Blender 指南
│   │   └── web_preview/           # Three.js 页面 + vendor
│   ├── app/app.py                 # Gradio 控制台 + 提示词 + 3D
│   └── db/schema.sql              # MySQL 三表
└── tests/test_metrics.py          # 占位测试
```

### 8.1 关键函数契约

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `extract_landmarks.process_video(video, output_dir, model, blendshapes, batch)` | 视频路径 | dict(summary) | 写 CSV/overlay/curve |
| `plot_dynamics.analyze_file(csv, out_png, fps, video, max_lag, smooth_sec, n_boot)` | 特征 CSV | dict | 门控 + 事件 + 图 |
| `plot_dynamics.permutation_test(a,b)` | 两组值 | (diff, exact p) | 组合数 ≤200k 精确枚举 |
| `translator.build_payload(df, video, fps)` | 含 `_smooth` 的 DataFrame | dict | 52 通道 JSON |
| `translator.auto_fetch_model.main([])` | 无 | 写 `model.glb` | 下载/转换 |
| `app.build_demo()` | 无 | `gr.Blocks` | 云端自动隐藏 3D |

---

## 9. 制作流程（按时间顺序）

### W0：项目骨架

- 建立 `data/`、`outputs/`、`src/`、`docs/`、`tests/`。
- `.gitignore` 排除视频、模型、venv、大产物。
- 结论：目录结构一次到位，后续只填内容。

### W1：MediaPipe Tasks API 管线

目标：视频 → 关键点 overlay + 几何特征 CSV + 曲线。

过程：

1. 先试 `pip install openface` → PyPI 只有 `openface 0.0.0` 空壳，`__init__.py` 0 字节，wrapper 调用不存在的 `bin/FeatureExtraction`；路线 A 判死。
2. 改用 MediaPipe 1.2.2 Tasks API `FaceLandmarker`，VIDEO 模式，478 点 + 52 blendshapes。
3. 首版 `extract_landmarks.py`：`mp.Image` + `detect_for_video`，逐帧写 overlay、CSV、曲线。
4. 跑正脸样本 `adult_frontal.mp4`：321 帧 100% 检出，`mar` 0.000–0.343，峰值 frame 54。

失败：

- 侧脸样本 `bad_sample_sideface.mp4`（甩头）16 帧未检出，31–36 帧关键点飘到背景，`mar` 冲到 1.506，`ear_l` 1.856。
- 结论：几何比值在侧脸/遮挡时会爆炸，分母下限 `1e-3` 防不住“整组关键点偏移”。记为 **KI-001/KI-002**。

### W2：Blendshape 通道

- 打开 `output_face_blendshapes=True`，52 通道（51 动作 + `_neutral`）。
- 3 帧居中中值平滑 → `_smooth` 列。
- 验证：`mouthSmileLeft/Right` 在微笑帧接近 0.94；眼周 `eyeSquint` 可用。
- 结论：Blendshape 可直接作为 AU 近似，不需要 OpenFace。

### W3：联动分析与反例

- 首版用 `mouthSmileRight_smooth` vs `eyeSquintRight_smooth` 双轴图。
- 发现 `mouthSmile` **饱和**：中位数 0.957，68% 帧 ≥0.95；`cheekSquint` 全 0。
- 得到 640 ms 伪滞后；存档 `dynamics_bad_proxy.png`。
- 改用 `mouthUpperUpRight_smooth` + `smile_lift`，并引入 `r>0.6`、边界、事件窗口。
- 结论：**r/lag 不是好的组间判别特征**；真正有希望的是微颤。

### W4：队列、门控、Jerk、统计

- W4.0：5 条样本，r 无分离；发现素材名不副实（持续微笑、仰头大笑）。
- W4.1：Jerk + 归一化；发现裸 Jerk 被幅度污染。
- W4.2：门控改为头动（face bbox）、眼通道一致性；置换检验。
- W4.3：**眨眼屏蔽**（关键），4v4 p=0.0143/3v4 0.0295。
- W4.4：发现 **60 fps + 公式错误** 导致 `real_clean_05` 被压低；统一 25 fps、`norm_jerk = jerk/std`、精确枚举；最终 4v4 p=0.0143，盲测 2v2 零重叠。
- 结论：主指标定型为归一化眼周 Jerk。

### W5：UI 与提示词

- Gradio 6；`gr.themes.Soft`、深色 CSS、深蓝按钮、进度条、扫描线。
- 图表改英文标签，避免 Edge 翻译/中文字体乱码。
- Jerk < 60 生成反向优化提示词；≥60 返回“无需优化”。
- 结论：从“诊断”变成“可执行反馈”。

### W6：JSON 导出

- `blendshape_export.py` 复用提取链路，输出 203 帧 × 52 通道、331 KB。
- `verify_json.py` 校验：帧数、通道数、NaN、时间戳单调。
- 结论：机器语言层打通。

### W7：3D 预览

- 先做程序化头（Three.js 图元），验证 52 通道驱动。
- 后升级 three.js r160 ES module + GLTFLoader/KTX2Loader/Meshopt，加载 `facecap.glb`（52 morphs）。
- 绑定 51/52（`neutral` 无 target）；加阴影、环境贴图、待机呼吸。
- 失败：headless 虚拟时间截断 KTX2 异步转码；真实浏览器正常。

### W8：All-in-One + 云端

- Gradio 集成 JSON 生成 + iframe。
- Gradio 6 `launch()` 会重建 app，挂载路由被 catch-all 吃掉 → 包装 `App.create_app` 注入 `/preview`、`/data`。
- ModelScope 反向代理导致绝对路径 `/preview` 乱码 → 改相对路径 `preview/index.html`、`../data/`。
- 最终云端改为**纯诊断模式**：`FACETEMPO_CLOUD=1` 隐藏 3D、跳过 JSON/路由、不传 theme/css。
- 本地保留完整 3D。

### W8.1：模型与渲染

- `auto_fetch_model.py` 下载 three.js `facecap.glb`（332,808 字节，52 morphs）。
- 本机无 Blender；FBX 分支提供 bpy 脚本 + 手动指南。
- 头部材质：灰度贴图 + 肤色叠加、roughness 0.68、环境贴图、PCF 软阴影、接触阴影。

### W10/W11：部署与仓库

- 根 `app.py`（HF/ModelScope），云端自动下载 `model.glb`。
- `requirements.txt` 云端精简；`requirements-dev.txt` 本地完整。
- `packages.txt` 系统库；`deploy/hf_README.md` YAML。
- MIT LICENSE、双语 README、`DEPLOY_GUIDE.md`。
- GitHub 远端 `04243216/facetempo`；本地领先 2 个提交（`505ee4d`、`cd70c96`）未推。

### W12：报告

- `README_technical_report.md`：方法、结果、限制。
- `FaceTempo_Report.html`：单文件作品集（原生 SVG 图表、下载按钮、打印样式）。

---

## 10. 算法规范（canonical）

### 10.1 特征定义

| 特征 | 公式 / 定义 | 单位 |
|---|---|---|
| `ear_l` / `ear_r` | 眼睑垂直距离 / 眼角水平距离（归一化坐标） | 无量纲 |
| `mar` | 上下唇 y 差 / 嘴角 x 差，clip 0–2 | 无量纲 |
| `brow` | `|y107 - y159|` | 归一化 |
| `smile_lift` | `(eye_mid_y - mouth_mid_y) / face_w`（像素坐标） | 无量纲 |
| `mouth_width` | `|x61 - x291| / face_w` | 无量纲 |
| `eye_aspect_ratio_geo` | 左右眼上下眼睑距离均值 / face_w | 无量纲 |
| `face_cx/cy/w/h` | 234/454 与 10/152 的框 | 像素 |

注意：

- `smile_lift` 为负值，越接近 0 越上提；张嘴会拉低它，和 `mar` 相关仅 0.038。
- `ear/mar/brow` 沿用 W1 的归一化坐标（历史可比）；其余几何量在像素坐标下计算，避免 16:9 各向异性。

### 10.2 质量门控

```text
coverage      = 有效帧 / 总帧             >= 0.95
head_speed    = mean(||Δ(face_cx,face_cy)|| / face_w) * fps  < 0.5
head_dev      = max(||center - median(center)|| / face_w)    < 1.5
eye_agreement = corr(eye_aspect_ratio_geo, eyeSquintRight_smooth)
                仅当 std(geo)>=0.008 且 std(bs)>=0.008 时计算；必须 < 0，否则 INVALID
               低方差时记 N/A（不参与硬拒绝）
mouth_std     = std(mouthUpperUpRight_smooth)  只记录，不硬门控
```

### 10.3 预处理

```text
1) blink = (eyeBlinkLeft_smooth > 0.5) OR (eyeBlinkRight_smooth > 0.5)
2) mask = dilate(blink, 2 frames)
3) 对 x = eyeSquintRight 线性重采样到 25 fps
4) 对 mask 同步重采样（>0.5 为 True）
5) 计算差分时，若窗口内任一帧被 mask 标记则丢弃该差分
```

### 10.4 主指标（唯一 canonical）

```text
jerk      = mean(|x[t+2] - 2x[t+1] + x[t]|) * fps^2
norm_jerk = jerk / std(x[~mask])
fps       = 25
```

- 单位：1/s²。
- 幅度归一化：除以 `std`，避免“没动”被误判成“平滑”。
- 帧率归一化：先统一到 25 fps，再乘 `fps²`。
- **废弃公式**：`jerk = mean|Δ²x| * fps²` 后再 `/(std*fps²)`（多除一次），会把 60 fps 样本压低约 5.8 倍。

### 10.5 统计

- 单侧置换检验，H1：真人 > AI。
- 组合数 ≤ 200,000 时**精确枚举**；否则蒙特卡洛。
- `p = count(diff >= observed - 1e-9) / total`。
- 4v4 最小 p = 1/70 = 0.0142857；3v4 = 1/35 = 0.02857；2v2 = 1/6 = 0.1667。
- 不做多重比较校正的理由：主指标在 W4.1 预先指定，HF 能量为次要探索指标。

### 10.6 联动分析（次要，不作主结论）

- 0.5 s 居中滚动中值；逐 lag Pearson，±1 s。
- `r > 0.6` 且 `|lag| < max_lag` 才有效。
- 符号：`lag > 0` 眼周落后；`< 0` 领先。
- 事件窗口：`smile_lift` 峰值，±1.5 s。
- 结论：本批 r/lag 无组间判别力；只用于探索。

### 10.7 52 通道 JSON

```json
{
  "meta": {
    "fps": 25.0,
    "total_frames": 203,
    "source": "real_clean_04.mp4",
    "model": "MediaPipe FaceLandmarker",
    "schema_version": "1.0",
    "smoothing": "3-frame centered median",
    "face_found_ratio": 1.0
  },
  "frames": [
    {"frame": 0, "time": 0.0, "blendshapes": {"neutral": 0.0, "eyeBlinkLeft": 0.05, "...": 0.0}}
  ]
}
```

- 52 通道：51 动作 + `neutral`（由 `_neutral` 改名）。
- 掉脸帧逐通道线性插值；整列空补 0。
- 值域 [0,1]，时间戳单调。
- 机器语言命名兼容 ARKit：`mouthSmileLeft` / `mouthSmile_L` 均可在前端匹配。

---

## 11. 实验结果

### 11.1 主队列 4v4（canonical）

| 组 | 归一化眼周 Jerk | 中位数 |
|---|---|---|
| 真人 | 105.70 / 67.59 / 86.99 / 67.03 | **77.29** |
| AI | 27.02 / 21.63 / 44.25 / 55.03 | **35.63** |
| 假笑 | 65.89 / 55.49 / 63.75 / 41.34 | 59.62 |

- 真人最小值 67.03 > AI 最大值 55.03 → **零重叠**。
- 精确单侧 p = **0.0143**（4v4 下限）。
- 眼周高频能量 p = 0.0429（次要）。
- 注意：HF 能量直接对重采样信号算，未额外屏蔽眨眼；若要严格屏蔽，应先对眨眼帧插值再 FFT，这是后续改进项。

### 11.2 独立盲测 2v2

| 组 | 归一化眼周 Jerk |
|---|---|
| 真人 | 40.51 / 96.20 |
| AI | 26.65 / 18.31 |

- 零重叠；p = 0.1667（2v2 下限），只能方向性支持。

### 11.3 假笑

- 中位数 59.62，位于真人（77.29）与 AI（35.63）之间。
- 与 AI 有重叠，不能单独当“假笑检测器”。

### 11.4 敏感性（眨眼屏蔽设置）

阈值 0.3/0.5/0.7 × 膨胀 0/2/4 帧，共 9 组：8 组 p<0.05（0.018–0.030），仅 0.7/0 时 p=0.079。

### 11.5 历史 3v4 结果

W4.3 的 gates_ok 口径 3 真人 vs 4 AI：p=0.0295；4v4 敏感性 0.0148。最终被 W4.4 的 4v4 结果取代。

### 11.6 废弃数字（AI 必须忽略）

| 数值 | 来源 | 为什么废弃 |
|---|---|---|
| 真人 0.332 / AI 0.064（0.086） | W4.1 旧归一化 | 公式多除 fps²，未统一帧率 |
| 真人 0.162 / AI 0.081 | W4.3 | 旧尺度 + 3v4；后被 4v4 重采样取代 |
| 假笑 0.095 / 0.110 | W4.3/W4.1 | 旧尺度 |
| p=0.1685 | W4.2 2v2 | 样本不足，不能当显著 |
| 640 ms 滞后 | W3 饱和 `mouthSmile` | 伪滞后 |
| AI 相关性更低 | W4 早期 | 已被证伪：AI r 不低于真人 |

---

## 12. 统计解释边界

1. **p 值下限**：4v4 完全分离时 p=0.0143，这是数学下限；不要期待更小。
2. **单侧检验**：H1 事先定为真人 > AI；不能事后改方向。
3. **盲测 2v2**：即使零重叠，p=0.1667；只能写“方向复现”，不能写“显著”。
4. **多重比较**：主指标预先指定；HF 能量为次要，不单独下结论。
5. **效应量**：主队列均值差约 +41.7（77.29−35.63），但样本小，应同时报告个体值。
6. **外部效度**：分辨率、压缩、内容、工具版本不同；结论限定本批样本。

---

## 13. 失败与修复日志

### KI-001 侧脸几何爆炸

- 现象：`bad_sample_sideface` 31–36 帧关键点飘到背景，`mar` 1.506、`ear_l` 1.856。
- 原因：侧脸时比值分母缩小；阈值 1e-3 只防重合，不防整体偏移。
- 处理：质量门控 + 覆盖率；W2 计划 visibility/bbox/时序连续性。
- 现状：开放；坏样本保留。

### KI-002 分母下限的边界

- `MIN_DEN=1e-3` 只防除零，不是质量过滤器。
- 结论：质量控制放在关键点层面，不在比值层面。

### KI-003 `FaceLandmarker.close()` 固定卡 42 秒

- 现象：`close()` 42.08–42.11 s，N=1/10/90 一样；`main()` 返回触发析构也卡。
- 原因：mediapipe 1.2.2 Windows 运行时关闭路径。
- 处理：
  - 不用 `with`，不调 `close()`；
  - 模块级 `_LANDMARKER_KEEPALIVE` 持有；
  - CLI 入口 `os._exit(code)`；
  - W10 长驻服务只创建一次实例。
- 验收：321 帧 wall 从 55s → 14s。

### KI-004 帧率/归一化错误

- 现象：`real_clean_05` 59.9 fps，旧口径 0.065，与 AI 重叠。
- 原因：逐帧二阶差分随 fps 缩小；旧公式再除一次 fps²。
- 处理：统一重采样 25fps；`norm_jerk = jerk/std`；精确置换。
- 验收：`real_clean_05` 67.03，与其他真人一致；4v4 p=0.0143。

### 其他失败

| 失败 | 处理 |
|---|---|
| PyPI `openface` 空壳 | 改 MediaPipe Blendshapes |
| `mouthSmile` 饱和 | 改 `mouthUpperUp` + 几何代理 |
| `cheekSquint` 全 0 | 弃用 |
| 全画面帧差门控误杀真人 | 改 face bbox 头动 |
| 60fps 混杂 | 重采样 |
| Gradio `launch()` 重建 app | 包装 `App.create_app` 注入静态路由 |
| ModelScope 子路径 iframe 乱码 | 相对路径；最终云端隐藏 3D |
| `model.glb` KTX2/meshopt | r160 ES modules + loaders |
| facecap 灰度贴图 | 叠加肤色 + 环境光 |
| 云端 Gradio 版本低 | 纯诊断模式不传 theme/css |

---

## 14. 复现步骤

### 14.1 环境

```powershell
cd E:\FACETEMPO
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements-dev.txt
venv\Scripts\python.exe src\translator\auto_fetch_model.py
```

### 14.2 单视频 W1

```powershell
venv\Scripts\python.exe src\extract_landmarks.py `
  --input data\raw\samples\w4_clean\adult_frontal.mp4
```

产物：`outputs/overlay/overlay.mp4`、`outputs/features/features.csv`、`outputs/curves/curve.png`。

### 14.3 批量

```powershell
venv\Scripts\python.exe src\extract_landmarks.py `
  --input data\raw\samples\w4_clean --blendshapes --output_dir outputs\w4_final
```

### 14.4 诊断

```powershell
venv\Scripts\python.exe src\plot_dynamics.py `
  --csv outputs\w4_final\features_curated2 `
  --video_dir data\raw\samples\w4_clean `
  --out_dir outputs\curves\w4_clean
```

注意：`features_curated2` 是 4 真人（02–05）+ 4 AI + 4 假笑的副本；目录不存在时先按下面命令生成，否则组统计会混入旧样本。

```powershell
$dst = "outputs\w4_final\features_curated2"
New-Item -ItemType Directory -Path $dst -Force | Out-Null
$names = @(
  "real_clean_02","real_clean_03","real_clean_04","real_clean_05",
  "ai_clean_01","ai_clean_02","ai_clean_03","ai_clean_04",
  "real_03_fake_smile","real_fake_01","real_fake_02","real_fake_03"
)
foreach ($n in $names) {
  Copy-Item "outputs\w4_final\features\${n}_features.csv" $dst -Force
}
```

### 14.5 盲测

```powershell
venv\Scripts\python.exe src\extract_landmarks.py --input data\raw\samples\holdout --blendshapes --output_dir outputs\holdout
venv\Scripts\python.exe src\plot_dynamics.py --csv outputs\holdout\features --video_dir data\raw\samples\holdout --out_dir outputs\holdout\curves
```

### 14.6 JSON

```powershell
venv\Scripts\python.exe src\translator\blendshape_export.py
venv\Scripts\python.exe src\translator\verify_json.py
```

### 14.7 3D

```powershell
venv\Scripts\python.exe src\translator\serve_preview.py
# http://127.0.0.1:8000/src/translator/web_preview/index.html
```

### 14.8 Demo

```powershell
venv\Scripts\python.exe src\app\app.py
# http://127.0.0.1:7860
```

### 14.9 测试

```powershell
venv\Scripts\python.exe -m pytest -q
```

---

## 15. 产物路径

| 产物 | 路径 | 是否入库 |
|---|---|---|
| 特征 CSV | `outputs/features/*.csv` | 否 |
| overlay | `outputs/overlay/*.mp4` | 否 |
| 诊断图 | `outputs/curves/*.png` | 否 |
| 诊断报告 | `outputs/reports/*.md` | 是 |
| 作品集图 | `outputs/figures/*.png` | 是 |
| driving JSON | `outputs/translator/*.json` | 否 |
| 3D 页面 | `src/translator/web_preview/*` | 是（vendor 本地） |
| 模型 task | `models/face_landmarker.task` | 否 |
| 模型 glb | `src/translator/web_preview/model.glb` | 否 |

---

## 16. Schema

### 16.1 `features.csv`

基础列（13）：

```
frame, face_found, ear_l, ear_r, mar, brow,
smile_lift, mouth_width, eye_aspect_ratio_geo,
face_cx, face_cy, face_w, face_h
```

加 `--blendshapes` 后追加：

- 52 个原始通道：`neutral`, `browDownLeft`, ..., `noseSneerRight`
- 52 个 `_smooth` 列（3 帧居中中值）

合计 **117 列**。

### 16.2 driving JSON

见 §10.7。`meta.total_frames` 等于 `frames` 长度；每帧 52 通道；`time = frame / fps`。

### 16.3 MySQL（骨架）

`src/db/schema.sql`：`videos`、`frame_features`（JSON 特征）、`diagnoses`（JSON 指标 + model_version）。W4 起按实际指标调整。

---

## 17. 部署

### 17.1 云端模式

| 环境 | 行为 |
|---|---|
| 本地 `python src/app/app.py` | 完整 3D，`/preview` + `/data` 同源 |
| 云端 `FACETEMPO_CLOUD=1` | 纯诊断，隐藏 3D，跳过 JSON/静态路由，`launch` 不传 theme/css |

云端入口：`app.py`（根）或 `deploy_bundle/app.py`，均 `setdefault("FACETEMPO_CLOUD","1")`。

### 17.2 Hugging Face

- Space：Gradio SDK，Python 3.12，CPU basic。
- 文件：`app.py`、`requirements.txt`、`packages.txt`、`src/`、`deploy/hf_README.md`。
- 模型自动下载；`model.glb` 可自动获取。

### 17.3 ModelScope

- 使用桌面 `facetempo_deploy/` 或 `facetempo_deploy.zip`（4.02 MB，39 文件）。
- 打开文件夹全选内容拖入根目录，保持 `src/`、`models/`、`outputs/` 结构。
- 启动 `python app.py`，端口 7860。
- 云端默认纯诊断；3D 本地看。

### 17.4 关键坑

- 云端 Gradio 版本可能 <6，不要传 theme/css。
- 反向代理子路径：iframe 必须相对路径；但最终已用纯诊断规避。
- `packages.txt` 不被支持时需自定义镜像装 libGL。

---

## 18. 已知限制与非目标

1. 主队列 n=4，p 触底；盲测 2v2 不能显著。
2. 真人/AI 分辨率、压缩、内容不一致。
3. 眨眼屏蔽是 W4.3 引入的事后修正；敏感性 8/9，但需独立复核。
4. 微颤指标依赖 25 fps 重采样；原始 60 fps 高频被截断。
5. HF 能量未严格屏蔽眨眼（仅 Jerk 屏蔽），只作辅助。
6. 3D 头部是程序化/示例模型，不是影视级；`facecap.glb` 许可需自行核对。
7. 云端纯诊断，不提供 3D。
8. 联动 r/lag 不具判别力。

---

## 19. 后续路线

1. 扩盲测到 4v4，争取独立 p<0.05。
2. 补一条合格真人替换 `real_clean_01`。
3. 严格 blink-mask 的 HF 能量实现（插值后 FFT）。
4. 换真实 ARKit GLB（52 morph targets）并确认许可。
5. 事件级基线 + Bootstrap 已实现，可进一步做组间分布检验。
6. 把 3D 复现放到独立 Space/静态站点，与诊断页解耦。
7. 补 MySQL 入库与批量报告自动化（W4 遗留）。

---

## 20. AI 交接规则

### 20.1 不变量（不可破坏）

1. **统一 25 fps** 后再算任何时序导数/高频指标。
2. **`norm_jerk = jerk/std`**，不要再除 fps²。
3. **眨眼屏蔽**（0.5，膨胀 2 帧）用于 Jerk；嘴部不屏蔽。
4. **质量门控**：覆盖率 ≥95%，head_speed <0.5，head_dev <1.5，eye agreement <0。
5. **p 值**：小样本精确枚举；报告必须写样本量与下限。
6. **不宣称** AI 的 r 更低；只宣称 Jerk 差异。
7. **不把 2v2 的 0.1667 包装成显著**。
8. **不调用** `FaceLandmarker.close()`；CLI 用 `os._exit`，服务用单例。
9. **云端** `FACETEMPO_CLOUD=1` 时不要启用 3D/静态路由/theme/css。
10. **数据与模型不入 git**（`data/raw/`、`outputs/` 大文件、`model.glb`）。

### 20.2 修改代码前检查

- [ ] 改指标公式？同步更新 `docs/metrics_spec.md` 和本文 §10/§11。
- [ ] 改门控？重跑主队列 + 盲测，报告两套口径。
- [ ] 改前端？本地 `src/app/app.py` 验证 3D；云端模拟 `FACETEMPO_CLOUD=1`。
- [ ] 改依赖？同时更新 `requirements.txt`（云）与 `requirements-dev.txt`（本地）。
- [ ] 改模型？验证 morph target 数量 ≥51、命名匹配、许可。
- [ ] 改数据？更新 `data/README.md` 的授权与采集条件。

### 20.3 禁止事项

- 不要用 `mouthSmile` 绝对值当笑强度（饱和）。
- 不要用全画面帧差判头动（表情会污染）。
- 不要用 60fps 原始逐帧差分直接比较。
- 不要把未检出帧填 0（应 NaN/插值）。
- 不要在云端 iframe 写 `127.0.0.1:8000`。
- 不要提交 `venv/`、`data/raw/`、`outputs/overlay|features|curves`、`model.glb`。

---

## 21. 提交历史

```text
cd70c96 chore: ignore local facetempo_deploy folder
505ee4d docs: add single-file visual portfolio report
08a9716 docs: add 3d cloud limitation notice        <- origin/master
ad31a3a deploy: add pure-diagnosis cloud mode with 3d hidden
1090760 fix: use relative preview/data routes behind reverse proxies
f3c59ba deploy: add modelscope bundle guide and ignore local bundle
bab57a2 week10-11: huggingface deploy ready and github cleanup
9502018 week8.1: automated 3d model fetch, blender conversion and premium rendering
ebc9d3b week8: integrated all-in-one console with 3d driver
313e047 week7: 3d web preview driver for blendshape json
bc72139 week6: blendshape translator and standard driving json export
ef629a3 week5: UI polish and prompt reverse-optimization engine
c1ad319 week5: polish gradio ui with theme, progress and english charts
9232f38 week5: final technical report and gradio micro-jitter demo
3e9bb7a week4.4: completed 4v4 main cohort and 2v2 independent holdout validation
7cb15f6 week4.3: final benchmark with 4v4 controlled samples, statistical significance achieved
b8e97b1 week4.2: head-motion gates, variance-aware eye agreement, permutation tests
fbffd96 week4.1: normalized micro-jitter, event baselines, bootstrap CI, and quality gates
1255fa2 week4: clean cohort benchmark, fake vs real smile, reliability gates validated
3fe47f5 fix: clamp event window end index at video boundary
45cd7c6 week3: batch extraction and batch dynamics diagnostics with quality gates
102435e week3: valid smile proxies, event-wise cross-correlation, geo-eye proxy
26cd570 week2: blendshape channel with 3-frame median smoothing, fix KI-003 exit delay
192cc20 week1: tasks api pipeline, feature csv, curves, and bad-sample test
```

---

## 22. 附录：关键外部资源

- MediaPipe FaceLandmarker 模型：`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
- 3D 示例模型：`https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/models/gltf/facecap.glb`
- 魔搭 Demo（用户提供）：`https://modelscope.cn/studios/Guo0704/facetempo-final`
- GitHub：`https://github.com/04243216/facetempo`

> 许可提醒：`facecap.glb` 来自 three.js 示例资产；three.js 代码为 MIT，但示例模型可能另有许可。商用/再分发前必须自行核对。
