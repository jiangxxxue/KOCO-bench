import sys
import os
from pathlib import Path

# 确保可以导入本实例下的实际代码
THIS_DIR = Path(__file__).resolve().parent
CODE_ROOT = THIS_DIR.parent  # .../ToolBrain/code
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
os.chdir(str(CODE_ROOT))

from toolbrain.prompt import tool_to_card  # type: ignore

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


def test_tool_to_card_contains_key_sections():
    card = tool_to_card(multiply)
    assert isinstance(card, str)
    # 基本结构段
    assert "Tool: multiply" in card
    assert "Purpose:" in card
    assert "Args (JSON):" in card
    # 参数说明
    assert "- a:" in card
    assert "- b:" in card
    # 返回说明
    assert "Returns:" in card or "Output type:" in card


def test_tool_to_card_handles_missing_description_gracefully():
    @tool
    def noop() -> None:
        """No-op tool that does nothing.

        Returns:
            None: no output.
        """
        return None

    card = tool_to_card(noop)
    assert "Tool: noop" in card
    # 无描述时应给出占位说明
    assert "Purpose:" in card