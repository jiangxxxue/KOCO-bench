import sys
import os
from pathlib import Path

# 确保可以导入本实例下的实际代码
THIS_DIR = Path(__file__).resolve().parent
CODE_ROOT = THIS_DIR.parent  # .../ToolBrain/code
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
os.chdir(str(CODE_ROOT))

from toolbrain.prompt import tools_to_card  # type: ignore

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
def add(x: int, y: int) -> int:
    """
    Add two integers.

    Args:
        x (int): First addend.
        y (int): Second addend.

    Returns:
        int: Sum of x and y.
    """
    return x + y


@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two integers.

    Args:
        a (int): First factor.
        b (int): Second factor.

    Returns:
        int: Product of a and b.
    """
    return a * b


def test_tools_to_card_basic_layout_and_numbering():
    card = tools_to_card([add, multiply])
    assert isinstance(card, str)
    assert card.startswith("Available tools (2):"), "页眉应指示工具数量"
    # 编号与卡片内容
    assert "1) Tool: add" in card
    assert "2) Tool: multiply" in card
    # 分隔线存在
    assert "\n---\n" in card


def test_tools_to_card_empty_returns_placeholder():
    card = tools_to_card([])
    assert card == "No tools provided.", "空工具列表应返回占位文本"


def test_tools_to_card_non_tool_raises_type_error():
    import pytest  # type: ignore
    with pytest.raises(TypeError):
        tools_to_card([add, "not_a_tool"])  # type: ignore