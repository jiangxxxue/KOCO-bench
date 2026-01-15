import sys
import os
from pathlib import Path

# 确保可以导入本实例下的实际代码
THIS_DIR = Path(__file__).resolve().parent
CODE_ROOT = THIS_DIR.parent  # .../ToolBrain/code
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
os.chdir(str(CODE_ROOT))

from toolbrain.prompt import _extract_tool_meta  # type: ignore

# 准备 smolagents 的 Tool 装饰器
def _ensure_smolagents_path():
    try:
        import smolagents  # noqa: F401
        return
    except Exception:
        pass

    # 在工作区根目录下查找 smolagents/src
    test_file = Path(__file__).resolve()
    for up in range(2, 10):
        candidate = test_file.parents[up] / "smolagents" / "src"
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            break

_ensure_smolagents_path()
from smolagents import tool  # type: ignore


@tool
def add(x: int, y: int = 3) -> int:
    """Add two integers.

    Args:
        x (int): First addend.
        y (int): Second addend, default is 3.

    Returns:
        int: sum of x and y.
    """
    return x + y


def test_extract_tool_meta_basic_fields():
    meta = _extract_tool_meta(add)
    assert isinstance(meta, dict)
    assert meta.get("name") == "add"
    # 描述可能来自函数 docstring，至少应存在（或为空字符串）
    assert "description" in meta

def test_extract_tool_meta_parameters_and_docs():
    meta = _extract_tool_meta(add)
    # parameters 从 tool.spec 中抽取，至少应包含 x,y
    params = meta.get("parameters", {})
    assert isinstance(params, dict)
    assert "x" in params and "y" in params
    # Returns/Examples 解析（不强制必须存在，但应允许存在）
    assert "returns" in meta
    assert "examples" in meta

def test_extract_tool_meta_type_guard():
    import pytest  # type: ignore
    with pytest.raises(TypeError):
        _extract_tool_meta("not_a_tool")  # type: ignore