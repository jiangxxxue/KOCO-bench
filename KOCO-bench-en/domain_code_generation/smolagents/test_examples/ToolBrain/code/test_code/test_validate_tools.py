import sys
import os
from pathlib import Path

# Ensure actual code under this instance can be imported
THIS_DIR = Path(__file__).resolve().parent
CODE_ROOT = THIS_DIR.parent  # .../ToolBrain/code
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
os.chdir(str(CODE_ROOT))

import pytest  # type: ignore
from toolbrain.prompt import validate_tools, tools_to_card  # type: ignore

# Prepare smolagents Tool decorator
def _ensure_smolagents_path():
    try:
        import smolagents  # noqa: F401
        return
    except Exception:
        pass

    # Search for smolagents/src under workspace root directory
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


def test_validate_tools_rejects_string_type():
    with pytest.raises(TypeError):
        validate_tools("not a sequence of tools")  # type: ignore


def test_validate_tools_empty_sequence_raises_value_error():
    with pytest.raises(ValueError):
        validate_tools([])


def test_validate_tools_non_tool_item_raises_type_error():
    with pytest.raises(TypeError):
        validate_tools([add, "not_a_tool"])  # type: ignore


def test_validate_tools_duplicate_names_raises_value_error():
    # Force set duplicate names to trigger validation
    add.name = "dup"       # type: ignore[attr-defined]
    multiply.name = "dup"  # type: ignore[attr-defined]
    try:
        with pytest.raises(ValueError):
            validate_tools([add, multiply])
    finally:
        # Restore to default names to avoid affecting other tests
        add.name = "add"           # type: ignore[attr-defined]
        multiply.name = "multiply" # type: ignore[attr-defined]


def test_validate_tools_valid_unique_passes():
    # Should not raise exception
    validate_tools([add, multiply])