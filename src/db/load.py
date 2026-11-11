"""W4-W6：CSV → MySQL 写入。

流程建议：
    1. 读 outputs/features/*.csv 与诊断结果
    2. upsert videos（按 sha256 去重）
    3. 批量写 frame_features（executemany，分块 5k 行）
    4. 写 diagnoses，记录 report_path 与 model_version

连接信息从环境变量读取，不要写进仓库：
    FACETEMPO_DB_HOST / FACETEMPO_DB_PORT / FACETEMPO_DB_USER /
    FACETEMPO_DB_PASSWORD / FACETEMPO_DB_NAME
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("W4-W6 待实现：连接 MySQL、建表校验、批量 upsert。")


if __name__ == "__main__":
    main()
