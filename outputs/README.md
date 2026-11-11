# outputs/

一切跑出来的结果。**只提交 `reports/` 与 `figures/` 里的小文件。**

| 目录 | 内容 | 进 git |
|---|---|---|
| `overlay/` | 关键点叠加视频 | 否（体积大） |
| `features/` | 逐帧特征 CSV | 否（可由脚本重跑） |
| `curves/` | 特征曲线 PNG | 否（中间态） |
| `reports/` | 诊断报告（Markdown/HTML/PDF） | 是 |
| `figures/` | 作品集/论文用的定稿图 | 是 |

## 命名建议

- 单条视频产物：`<视频名>_<阶段>.<ext>`，例如 `test_overlay.mp4`、`test_features.csv`。
- 批量诊断报告：`reports/YYYY-MM-DD_<批次名>.md`。
- 定稿配图：`figures/fig_<序号>_<内容>.png`，正文只引用定稿图，不引用 `curves/`。

## 清理

`overlay/`、`features/`、`curves/` 随时可删，重跑 `src/extract_landmarks.py` 即可重建。
