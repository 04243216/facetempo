# FaceTempo

**面部位移与表情节奏的量化工具**：把一段人脸视频拆成逐帧关键点几何特征与 AU 强度，按时间对齐后输出可比较的节奏指标、诊断报告与可视化图。

> 当前状态：**W1 完成（2026-11-11）**。MediaPipe Tasks API 管线跑通，正脸样本 100% 检出；侧脸会引发几何特征爆炸，已列为 W2 重点防御对象（见 `docs/known_issues.md`）。

---

## 环境准备

实测环境：**Windows + Python 3.12.10**，依赖版本锁定在 `requirements.txt`。

```powershell
cd E:\FACETEMPO
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

- `requirements.in` 是人读的直接依赖清单（带说明），`requirements.txt` 是 `pip freeze` 生成的锁定清单。
- 不要用系统默认的 Python 3.14：mediapipe 未提供对应 wheel。
- 如果 PowerShell 禁止执行脚本：`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`。

### 模型文件（不入 git）

`models/face_landmarker.task`（约 3.6 MB）。官方地址：

```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

如果 Google 存储不可达，可用镜像（文件需自行校验大小）：

```
https://hf-mirror.com/VLAAADYAAA/face-landmarker-task-model/resolve/main/face_landmarker.task
```

已校验的参考值：`3,758,596` 字节，SHA256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`。

---

## 运行命令

### W1：视频 → 叠加视频 + 特征表 + 曲线

```powershell
venv\Scripts\python.exe src\extract_landmarks.py --input data/raw/samples/adult_frontal.mp4 --output_dir outputs
```

参数：

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `--input` | 是 | — | 输入视频文件**或目录** |
| `--output_dir` | 否 | `outputs` | 输出根目录 |
| `--model` | 否 | `models/face_landmarker.task` | 关键点模型 |
| `--blendshapes` | 否 | 关闭 | 追加 52 个 Blendshape 及 `_smooth` 列 |

批量处理目录（按文件名排序，单条失败不阻断整批）：

```powershell
venv\Scripts\python.exe src\extract_landmarks.py --input data/raw/samples --blendshapes
```

产物命名：

| 模式 | 叠加视频 | 特征表 | 曲线 |
|---|---|---|---|
| 单文件 | `outputs/overlay/overlay.mp4` | `outputs/features/features.csv` | `outputs/curves/curve.png` |
| 目录批量 | `outputs/overlay/<stem>_overlay.mp4` | `outputs/features/<stem>_features.csv` | `outputs/curves/<stem>_curve.png` |

### W3：肌肉联动诊断（批量）

```powershell
venv\Scripts\python.exe src\plot_dynamics.py
```

默认读取 `outputs/features/*_features.csv`，逐条输出：
`outputs/curves/<stem>_dynamics.png`，终端打印每条视频的滞后、r 值，以及起势/维持是否一致。
单文件模式：`--csv outputs/features/adult_frontal_features.csv`。

### W5：演示界面

```powershell
venv\Scripts\python.exe src\app\app.py
```

启动后打开 <http://127.0.0.1:7860>，上传正脸视频即可看到归一化眼周 Jerk、
参考阈值判断与 MAR 曲线。界面使用进程级 `FaceLandmarker` 单例，避免 KI-003 的 42 秒关闭阻塞。

### W6：导出驱动 JSON

```powershell
venv\Scripts\python.exe src\translator\blendshape_export.py
venv\Scripts\python.exe src\translator\verify_json.py
```

默认读取 `data/raw/samples/w4_clean/real_clean_04.mp4`，输出
`outputs/translator/real_clean_04_driving.json`（52 通道 × 逐帧，3 帧中值平滑）。
校验脚本会画出 `mouthSmileLeft` 与 `eyeSquintLeft` 曲线到
`outputs/translator/verify_curve.png`，并检查帧数、通道数、NaN、时间戳单调性。

### W7：3D 驱动预览

```powershell
venv\Scripts\python.exe src\translator\serve_preview.py
```

打开 <http://127.0.0.1:8000/src/translator/web_preview/index.html>。
页面用 Three.js 加载 `driving.json`，把 52 个通道实时映射到程序化 3D 头部
（眼睑开合、嘴角/下颌、眉毛、眼周收缩），并显示逐帧时间轴与通道数值。
截图：`outputs/translator/web_preview_screenshot.png`。

CSV 列为 `frame, face_found, ear_l, ear_r, mar, brow, smile_lift, mouth_width, eye_aspect_ratio_geo`，**每帧一行**；未检出脸时 `face_found=0`、特征列为 NaN。加 `--blendshapes` 后再追加 52 个原始 Blendshape 与 52 个 `_smooth` 列。

### 后续阶段（未实现）

```powershell
venv\Scripts\python.exe src\extract_au.py        # W2-W3：OpenFace AU 强度
venv\Scripts\python.exe src\align_features.py    # W2-W3：几何 + AU 按帧对齐
venv\Scripts\python.exe src\diagnose.py          # W4-W6：批量诊断
venv\Scripts\python.exe src\db\load.py           # W4-W6：CSV → MySQL
venv\Scripts\python.exe src\baseline_convlstm.py # W7-W9：轻量基线
venv\Scripts\python.exe src\app\app.py           # W5+：上传视频出诊断（Gradio）
```

---

## W1 结论

| 项目 | 结果 |
|---|---|
| 检测器 | MediaPipe Tasks API `FaceLandmarker`（VIDEO 模式），478 点，索引与旧 FaceMesh 布局一致 |
| 正脸样本 `adult_frontal.mp4` | 2560×1440 / 25 fps / 321 帧，`face_found` **100%**，`mar` 0.000–0.343 |
| 曲线形态 | 两段微笑平台（峰值 0.343 @ frame 54）+ 6 次同步眨眼下探 |
| 侧脸样本 `bad_sample_sideface.mp4` | 272 帧中 16 帧未检出；31–36 帧关键点飘到背景，`mar` 冲到 1.506、`ear_l` 到 1.856 |
| 结论 | 管线可用；**几何特征在侧脸/遮挡时不可信**，W2 需加可见性/人脸框校验 |

---

## 目录结构

```
FACETEMPO/
├── README.md
├── requirements.in          # 人读依赖清单
├── requirements.txt         # pip freeze 锁定
├── .gitignore
├── models/                  # face_landmarker.task（不入 git）
├── data/
│   ├── raw/                 # 原始视频（不入 git）
│   │   ├── samples/         # adult_frontal.mp4 / bad_sample_sideface.mp4
│   │   ├── real/            # W4–W6 真人对照片段
│   │   └── ai/tool_a|b|c/   # W4–W6 AI 片段
│   ├── processed/           # 对齐后的特征矩阵（不入 git）
│   └── README.md
├── outputs/
│   ├── overlay/             # 关键点叠加视频（不入 git）
│   ├── features/            # 逐帧特征 CSV（不入 git）
│   ├── curves/              # 特征曲线 PNG（不入 git）
│   ├── reports/             # 诊断报告（可提交）
│   ├── figures/             # 作品集/论文配图（可提交）
│   └── README.md
├── src/
│   ├── extract_landmarks.py   # W1 ✅
│   ├── extract_au.py          # W2-W3
│   ├── align_features.py      # W2-W3
│   ├── metrics.py             # W4-W6
│   ├── diagnose.py            # W4-W6
│   ├── baseline_convlstm.py   # W7-W9
│   ├── db/                    # schema.sql + load.py
│   ├── translator/            # blendshape_export.py + verify_json.py
│   └── app/                   # app.py（Gradio 演示）
├── notebooks/
│   └── 01_feature_explore.ipynb
├── docs/
│   ├── 项目手册.html
│   ├── metrics_spec.md        # 六项指标的准确定义与公式
│   ├── known_issues.md        # 已知问题与 W2 计划
│   └── technical_report.md    # 技术报告（W12 定稿）
└── tests/
    └── test_metrics.py
```

---

## 数据与产物的规矩

- **进 git**：代码、文档、`requirements.in`、`outputs/reports/`、`outputs/figures/` 里的小图和报告。
- **不进 git**：`data/raw/`、`data/processed/`、`outputs/overlay/`、`outputs/features/`、`outputs/curves/`、`models/`、`venv/`。
- 空目录用 `.gitkeep` 占位；被整目录忽略的路径里的 `.gitkeep` 不会被跟踪，目录只存在于本地，这是有意为之。
- 视频几百 MB，**不要 `git add .` 之后直接提交**；大文件放本地盘或网盘，仓库只留说明和校验值。

---

## 快速自检

```powershell
venv\Scripts\python.exe -c "import cv2, mediapipe, numpy, pandas, matplotlib; print('deps ok')"
venv\Scripts\python.exe -m pytest -q
```

`tests/` 里的用例目前是占位，W4 实现六项指标后逐条替换。
