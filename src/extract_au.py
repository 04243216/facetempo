"""W2-W3：调用 OpenFace 提取 AU（面部动作单元）强度。

输入：data/raw/ 下的视频
输出：OpenFace 原生 CSV（含 AU01_r ... AU45_r、头部姿态、视线等）
说明：OpenFace 2.x 不是 pip 包，需本地安装并把 FeatureExtraction 加入 PATH。
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("W2-W3 待实现：封装 OpenFace FeatureExtraction 并整理输出目录。")


if __name__ == "__main__":
    main()
