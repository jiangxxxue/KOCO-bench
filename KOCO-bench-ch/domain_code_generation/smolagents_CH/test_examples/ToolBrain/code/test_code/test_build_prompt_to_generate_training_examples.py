import sys
import os
from pathlib import Path

# 确保可以导入本实例下的实际代码
THIS_DIR = Path(__file__).resolve().parent
CODE_ROOT = THIS_DIR.parent  # .../ToolBrain/code
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
os.chdir(str(CODE_ROOT))

import pytest  # type: ignore
from toolbrain.prompt import (
    build_prompt_to_generate_training_examples,
    tools_to_card,
)  # type: ignore

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
    """Add two integers.

    Args:
        x (int): First addend.
        y (int): Second addend.

    Returns:
        int: Sum of x and y.
    """
    return x + y


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a (int): First factor.
        b (int): Second factor.

    Returns:
        int: Product of a and b.
    """
    return a * b


def test_build_prompt_basic_sections_and_includes_tools_description():
    tools_desc = tools_to_card([add, multiply])
    prompt = build_prompt_to_generate_training_examples(
        tools_description=tools_desc,
        task_description="Perform arithmetic tasks using provided tools.",
        min_tool_calls=2,
        max_words=80,
        guidance_example=None,
    )
    assert isinstance(prompt, str)
    # 必要段落
    assert "Instructions:" in prompt
    assert "Tool description:" in prompt
    assert "Task description:" in prompt
    # 工具说明应被包含
    assert "Available tools (2):" in prompt
    assert "Tool: add" in prompt and "Tool: multiply" in prompt
    # 指令应包含最少工具调用约束
    assert "minimum 2 tools" in prompt


def test_build_prompt_max_words_floor_to_10():
    tools_desc = tools_to_card([add])
    prompt = build_prompt_to_generate_training_examples(
        tools_description=tools_desc,
        task_description=None,
        min_tool_calls=1,
        max_words=1,  # 小于 10，应被钳制到 10
        guidance_example=None,
    )
    assert "maximum 10 words" in prompt


def test_build_prompt_includes_guidance_example_when_provided():
    tools_desc = tools_to_card([add])
    guidance = "Few-shot: compute 3+5 using tool."
    prompt = build_prompt_to_generate_training_examples(
        tools_description=tools_desc,
        task_description="Basic math.",
        min_tool_calls=1,
        max_words=20,
        guidance_example=guidance,
    )
    assert "Few-shot guidance" in prompt
    assert guidance in prompt


def test_build_prompt_contains_gold_answer_requirement():
    tools_desc = tools_to_card([add])
    prompt = build_prompt_to_generate_training_examples(
        tools_description=tools_desc,
        task_description=None,
        min_tool_calls=1,
        max_words=30,
        guidance_example=None,
    )
    # 要求输出金标准答案的规范文本
    assert "[Gold answer]" in prompt