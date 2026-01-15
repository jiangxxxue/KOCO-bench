import sys
import os
from pathlib import Path
import types

# 使测试导入到本实例 code/ 下的实际实现
THIS_DIR = Path(__file__).resolve().parent
CODE_ROOT = THIS_DIR.parent  # .../ToolBrain/code
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
os.chdir(str(CODE_ROOT))

import pytest  # type: ignore

# 直接依赖本实例下的 peft 本地包（code/peft/__init__.py），不再通过 sys.modules 注入假模块
from toolbrain.adapters.smolagent.smolagent_adapter import SmolAgentAdapter  # type: ignore


def test_parse_missing_parts_extracts_thought_and_numeric_final_answer():
    # 不经 __init__ 构造实例
    adapter = object.__new__(SmolAgentAdapter)
    text = (
        "Thought: I will compute the result first.\n"
        "Final Answer: The value is 42."
    )
    parts = adapter._parse_missing_parts(text)
    assert parts.get("thought") == "I will compute the result first."
    # 应提取数字部分
    assert parts.get("final_answer") == "42"


def test_parse_missing_parts_handles_textual_final_answer():
    adapter = object.__new__(SmolAgentAdapter)
    text = "Thought: reasoning...\nFinal Answer: hello world"
    parts = adapter._parse_missing_parts(text)
    assert parts.get("final_answer") == "hello world"


def test_segment_text_with_assistant_splits_and_reassembles():
    adapter = object.__new__(SmolAgentAdapter)
    full = "system hi\nuser ask\nassistant reply\nuser follow\nassistant done"
    messages = [
        {"role": "system", "content": "hi"},
        {"role": "user", "content": "ask"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "follow"},
        {"role": "assistant", "content": "done"},
    ]
    segs = adapter._segment_text_with_assistant(full, messages)
    # 必须覆盖完整文本
    assert "".join(s["text"] for s in segs) == full
    # 至少包含两个 assistant 片段
    assert any(s["role"] == "assistant" and "reply" in s["text"] for s in segs)
    assert any(s["role"] == "assistant" and "done" in s["text"] for s in segs)