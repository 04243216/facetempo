"""自测骨架。W4 实现六项指标后，按 docs/metrics_spec.md 逐条补用例。

运行：
    pip install pytest
    pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def test_metrics_module_importable() -> None:
    """冒烟测试：src/metrics.py 能被导入且有入口。"""
    import metrics

    assert callable(metrics.main)


@pytest.mark.skip(reason="W4 实现六项指标后替换为真实数值用例")
def test_placeholder() -> None:
    ...
