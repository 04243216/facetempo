"""W10-W12：Gradio 应用——上传视频，出诊断报告。

建议页面结构：
    左：视频上传 + 参数（是否叠加关键点、是否显示曲线）
    中：叠加视频预览
    右：六项指标卡片 + 曲线图 + 报告下载

注意：
    - 单次请求不要跑满 CPU，视频先截断到 N 秒或降采样；
    - 上传文件写到临时目录，请求结束即删；
    - 对外部署前确认数据授权与隐私说明。
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("W10-W12 待实现：Gradio 界面 + 管线调用 + 报告生成。")


if __name__ == "__main__":
    main()
