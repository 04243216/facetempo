# FaceTempo 部署指南

Hugging Face Spaces（免费 CPU）+ GitHub 的完整上线步骤。

---

## 0. 架构说明（先看这一节）

云端只开 **一个端口**，Gradio 与 3D 预览同源：

```
https://<user>-facetempo.hf.space/
├── /                          Gradio 诊断控制台
├── /preview/index.html        3D 预览页（Three.js）
└── /data/<name>.json          每次上传生成的 52 通道驱动数据
```

- 不再依赖 `127.0.0.1:8000` 的第二个服务；
- `app.py` 启动时若 `src/translator/web_preview/model.glb` 不存在，会自动下载
  three.js 的 ARKit 示例模型（约 325 KB）；
- 下载失败不会阻塞诊断，3D 预览自动回退到程序化头。

---

## 1. 前置条件

| 项目 | 要求 |
|---|---|
| 账号 | Hugging Face、GitHub |
| 本地 | Python 3.12（HF 云端也必须是 3.12） |
| Space 硬件 | **CPU basic（免费）**，2 vCPU / 16 GB |
| 端口 | 7860（HF 自动注入 `GRADIO_SERVER_PORT`） |

> mediapipe 1.2.2 只提供到 cp312 的 wheel。若 Space 使用 Python 3.13，构建会失败。

---

## 2. 本地先验证

```powershell
cd E:\FACETEMPO
.\venv\Scripts\Activate.ps1

# 可选：确认模型能自动下载
python src/translator/auto_fetch_model.py

# 启动完整控制台
python app.py
```

打开：

- 控制台：<http://127.0.0.1:7860/>
- 3D 预览：<http://127.0.0.1:7860/preview/index.html>

上传一段 3–10 秒正脸视频，确认诊断表、提示词、3D 复现都出现。

---

## 3. 创建 Hugging Face Space

1. 打开 <https://huggingface.co/new-space>。
2. 填写：
   - **Space name**：`facetempo`
   - **License**：MIT
   - **SDK**：Gradio
   - **Hardware**：CPU basic（免费）
   - **Visibility**：Public
3. 创建后进入 **Settings**：
   - 确认 Gradio 版本为 `6.32.0`（或与 `requirements.txt` 一致）；
   - 若有 **Python version** 选项，设为 **3.12**。
4. 若界面没有 Python 版本选项，使用 `deploy/hf_README.md` 作为 Space 的
   `README.md`（含 YAML 元数据，指定 `python_version: "3.12"`）。

---

## 4. 上传代码（二选一）

### 方案 A：网页上传（最稳，推荐）

在 Space 的 **Files** 页面，上传以下内容：

```
app.py                        ← 根目录入口（必须）
requirements.txt              ← 云端依赖（已精简）
packages.txt                  ← libGL 等系统包
src/                          ← 整个目录
README.md                     ← 用 deploy/hf_README.md 的内容
```

**不要上传**：

```
venv/
outputs/                      ← 大文件，云端会重新生成
src/translator/web_preview/model.glb   ← 启动时自动下载
data/                         ← 原始视频，涉及隐私
```

网页上传文件夹较麻烦时，用方案 B。

### 方案 B：Git 推送（不覆盖 Space 的 README）

HF Space 的 README 带 YAML 元数据，直接 `git push` 整个仓库会把它覆盖掉。
推荐先克隆 Space 仓库，再拷代码：

```powershell
git clone https://huggingface.co/spaces/<user>/facetempo hf-space
cd hf-space

# 从 GitHub 仓库拷贝（不包含 README / outputs / model.glb）
robocopy E:\FACETEMPO . app.py src requirements.txt packages.txt LICENSE /E /XF model.glb
copy E:\FACETEMPO\deploy\hf_README.md README.md

git add .
git commit -m "deploy: facetempo"
git push
```

推送到 HF 时用户名填 HF 用户名，密码用 **Access Token**（Settings → Access Tokens，Write 权限）。

---

## 5. 环境变量

不需要任何密钥。可选：

| 变量 | 值 | 说明 |
|---|---|---|
| `GRADIO_ANALYTICS_ENABLED` | `False` | 关闭统计 |
| `GRADIO_SERVER_PORT` | 由 HF 自动注入 | `app.py` 已读取 |

---

## 6. 构建与启动过程

1. HF 安装 `requirements.txt`；
2. 安装 `packages.txt` 中的 `libgl1` 等系统库；
3. 运行 `app.py`：
   - 检查 `model.glb`，不存在则下载（约 325 KB）；
   - `build_demo()` 注册 `/preview`、`/data` 静态路由；
   - HF 启动 Gradio，监听 7860。
4. 首次构建约 3–8 分钟；冷启动约 30–60 秒。

---

## 7. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 构建失败：`No matching distribution found for mediapipe` | Python 不是 3.12 | 在 Space 设置或 README 元数据中固定 `python_version: "3.12"` |
| 构建失败：`libGL.so.1: cannot open shared object` | 缺系统库 | 确认 `packages.txt` 已上传，包含 `libgl1`、`libglib2.0-0` |
| 3D 区域空白 | `model.glb` 下载失败 / 浏览器拦截 | 访问 `/preview/index.html` 看是否 200；看浏览器 Console；可手动上传 `model.glb` 到 `src/translator/web_preview/` |
| 页面 404 `/preview/...` | 静态路由未注册 | 确认使用仓库内的 `src/app/app.py`（已注入 Gradio `create_app`） |
| 3D 区域显示二进流/乱码 | 应用挂在域名子路径下，iframe 用了绝对 `/preview` | 已改为相对 `preview/index.html`；拉取最新代码并重新构建 |
| 3D 区域只有文字没有画面 | iframe 跨域或代理拦截 | 点“在新标签打开”直达 `preview/index.html` 排查 |
| 端口冲突 | 7860 被占用 | 本地设置 `GRADIO_SERVER_PORT=7862` 后重试 |
| 上传视频后无响应 | 视频过长/分辨率过高 | 先用 3–10 秒、1080p 以内测试；免费 CPU 处理 1440p 约 15–30 秒 |

---

## 8. GitHub 仓库

```powershell
cd E:\FACETEMPO
git add .
git commit -m "release: facetempo"
git branch -M main
git remote add origin https://github.com/<user>/facetempo.git
git push -u origin main
```

仓库已包含：

- `LICENSE`（MIT）
- `README.md`（双语，含指标与截图）
- `README_technical_report.md`
- `DEPLOY_GUIDE.md`
- `requirements.txt`（云端精简）/ `requirements-dev.txt`（本地完整 freeze）

仓库**不包含**：

- 原始视频（`data/`）
- 生成产物（`outputs/` 下除 reports/figures 外）
- `model.glb`（运行 `auto_fetch_model.py` 获取）
- `venv/`

---

## 9. 上线检查清单

- [ ] Space 状态为 `Running`
- [ ] 首页能打开，诊断按钮可用
- [ ] `/preview/index.html` 返回 200
- [ ] 上传视频后：Jerk 数值、MAR 曲线、提示词、3D 复现都出现
- [ ] 手机浏览器能打开同一个链接
- [ ] GitHub README 中的 Space 链接已更新

---

## 10. ModelScope（魔搭）一键包

如果 HF 网络不可用，可直接用桌面上的部署包：

```
C:\Users\dd279\Desktop\facetempo_deploy\
C:\Users\dd279\Desktop\facetempo_deploy.zip
```

包内已包含（这是与普通源码包的关键区别）：

```
app.py
requirements.txt
packages.txt
models/face_landmarker.task          ← 关键点模型 3.58 MB
src/translator/web_preview/model.glb  ← 3D 头部 325 KB
src/                                 ← 全部运行代码
outputs/app_json / outputs/translator
```

步骤：

1. 魔搭 → 创空间 → 新建 → 选择 **Gradio** 框架、免费 CPU、公开。
2. Python 版本选 **3.10 / 3.11 / 3.12**（mediapipe 需要 ≥3.9）。
3. 把 `facetempo_deploy` 里的**内容**（不是外层文件夹）全选拖入创空间根目录，
   保持 `src/`、`models/`、`outputs/` 的目录结构。
4. 启动命令填 `python app.py`，端口 `7860`。
5. 构建完成后访问：
   - 首页：`https://<space>.modelscope.cn/`
   - 3D：`https://<space>.modelscope.cn/preview/index.html`

若魔搭不支持 `packages.txt` 的 apt 安装、且构建时报 `libGL.so.1`，
需要在自定义镜像中安装 `libgl1 libglib2.0-0 libsm6 libxext6 ffmpeg`，
或改用支持系统包安装的运行环境。
