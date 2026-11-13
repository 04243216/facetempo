# FaceTempo 诊断报告：真人 vs AI 眼周微颤（W4.4）

## 0. 元信息

| 项目 | 内容 |
|---|---|
| 分析日期 | 2026-11-13 |
| 分析版本 | W4.4（统一 25 fps 重采样 + 眨眼屏蔽 + 精确置换检验） |
| 分析人 | FaceTempo 自动诊断流程 |
| 结论状态 | 主队列 4v4 显著（p = 0.0143）；独立盲测 2v2 方向一致、无重叠 |

## 1. 样本与采集

| 项目 | 内容 |
|---|---|
| 主队列 | `data/raw/samples/w4_clean/` 中 4 条真人 + 4 条 AI |
| 真人自然 | `real_clean_02` – `real_clean_05`，24–60 fps，约 8–14 s |
| AI | `ai_clean_01` – `ai_clean_04`，24 fps，10 s，960×720 / 1108×832 |
| 真人假笑 | `real_03_fake_smile`、`real_fake_01` – `real_fake_03` |
| 独立盲测 | `data/raw/samples/holdout/`：`real_holdout_01/02`、`ai_holdout_01/02` |
| 机位与光照 | 真人以固定机位正脸为主；AI 为生成的正脸片段 |
| 授权状态 | 真人素材待补齐书面授权说明；AI 素材需记录工具与提示词 |

**帧率混杂已处理**：`real_clean_05` 为 59.9 fps，其余 24–25 fps。所有微颤信号统一线性重采样到 **25 fps** 后再计算，避免采样率造成系统性偏差。

## 2. 数据质量

| 指标 | 真人自然（4 条） | AI（4 条） | 假笑（4 条） |
|---|---|---|---|
| 人脸覆盖率 | 100% | 100% | 100% |
| 头部运动速度 | 0.06–0.17 脸宽/秒 | 0.01–0.02 | 0.03–0.16 |
| 眼通道一致性 | -0.93 ~ N/A | -0.96 ~ -0.83 | -0.99 ~ -0.75 |
| 门控通过 | 4 / 4 | 4 / 4 | 4 / 4 |

另有 `real_clean_01`（覆盖率 75.1%、镜头级跳变）被判 INVALID，未进入主队列；该结论保留在记录中，未静默剔除。

## 3. 核心指标

眼周归一化 Jerk = `mean|d²x| × fps² / std(x)`，统一 25 fps，屏蔽眨眼帧及前后 2 帧。

| 组 | 各条数值 | 中位数 |
|---|---|---|
| 真人自然 | 105.70 / 67.59 / 86.99 / 67.03 | **77.29** |
| AI | 27.02 / 21.63 / 44.25 / 55.03 | **35.63** |
| 真人假笑 | 65.89 / 55.49 / 63.75 / 41.34 | 59.62 |

真人最小值 67.03 > AI 最大值 55.03，**两组完全不重叠**。

| 指标 | 真人 | AI | p（单侧，精确枚举） |
|---|---|---|---|
| 归一化眼周 Jerk | 67.03–105.70 | 21.63–55.03 | **0.0143**（4v4 下限） |
| 眼周高频能量（5–12 Hz） | 0.014–0.063 | 0.007–0.032 | 0.0429 |

## 4. 联动分析（次要，不作主结论）

- 全片 `mouthUpperUp` vs `eyeSquint` 的相关性在本批样本中普遍偏弱，多条视频的最佳滞后撞 ±1 s 搜索边界。
- 事件级 Bootstrap 在部分强事件上 r 可达 0.9 以上，但事件间滞后不一致。
- 结论：**联动指标（r / lag）目前不具备组间判别力**，本报告不以它下结论。

## 5. 独立盲测（Holdout 2v2）

| 视频 | 归一化眼周 Jerk |
|---|---|
| `real_holdout_01` | 40.51 |
| `real_holdout_02` | 96.20 |
| `ai_holdout_01` | 26.65 |
| `ai_holdout_02` | 18.31 |

- 真人 40.51–96.20，AI 18.31–26.65，**完全没有重叠**；
- 单侧置换检验 p = **0.1667**（2v2 的数学下限，不可能 < 0.05）；
- 高频能量方向一致且同样无重叠（真人 0.016 / 0.073，AI 0.007 / 0.006）。

## 6. 结论

**分级：A. 可信（限本批样本）**

> 在 4 条合格真人与 4 条 AI 正脸视频中，AI 的眼周归一化微颤显著低于真人（中位数 35.63 vs 77.29，精确单侧置换检验 p = 0.0143）。该差异在 2 条真人 + 2 条 AI 的独立盲测集上方向一致且完全无重叠，说明指标不是单一队列的过拟合。

## 7. 限制

- 每组仅 4 条（盲测 2 条），样本量小；主队列 p 已到下界。
- 真人/AI 素材的分辨率、压缩与内容不一致，可能存在画质混杂。
- 微颤指标依赖 25 fps 重采样；原始 60 fps 素材的高频信息被有意截断以匹配其他样本。
- 眨眼屏蔽阈值（0.5）与膨胀帧数（2）为预设中间值；敏感性扫描 9 组中 8 组 p < 0.05。
- 主指标预先指定为归一化眼周 Jerk；高频能量为次要指标。
- 结论仅适用于本批样本，不能直接外推到所有 AI 工具或人群。

## 8. 复现命令

```powershell
venv\Scripts\python.exe src\extract_landmarks.py --input data\raw\samples\w4_clean --blendshapes --output_dir outputs\w4_final
venv\Scripts\python.exe src\plot_dynamics.py --csv outputs\w4_final\features_curated2 --video_dir data\raw\samples\w4_clean --out_dir outputs\curves\w4_clean
venv\Scripts\python.exe src\extract_landmarks.py --input data\raw\samples\holdout --blendshapes --output_dir outputs\holdout
venv\Scripts\python.exe src\plot_dynamics.py --csv outputs\holdout\features --video_dir data\raw\samples\holdout --out_dir outputs\holdout\curves
```

图表：
- 主队列：`outputs/curves/w4_clean/final_comparison.png`
- 盲测：`outputs/holdout/curves/final_comparison.png`
