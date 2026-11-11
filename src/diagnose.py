"""W4-W6：批量诊断入口。

输入：data/processed/ 下对齐后的特征矩阵（或 data/raw/ 视频，管线内部先跑 W1-W3）
输出：outputs/reports/<日期>_<批次>.md + outputs/figures/ 定稿图
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("W4-W6 待实现：遍历批次、调用 metrics、汇总成报告。")


if __name__ == "__main__":
    main()
