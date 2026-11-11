"""W2-W3：把几何特征与 OpenFace AU 按帧对齐，输出特征矩阵。

输入：outputs/features/*.csv、OpenFace AU CSV
输出：data/processed/<视频名>_aligned.parquet（或 .npz）
注意：两套流程帧率/起始帧可能不同，需以时间戳而非行号对齐。
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("W2-W3 待实现：按 timestamp 重采样对齐，处理丢帧与置信度。")


if __name__ == "__main__":
    main()
