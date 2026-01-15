import sys
import os
from pathlib import Path
import types

# Make tests import actual implementation under this instance's code/
THIS_DIR = Path(__file__).resolve().parent
CODE_ROOT = THIS_DIR.parent  # .../ToolBrain/code
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
os.chdir(str(CODE_ROOT))

import pytest  # type: ignore

# Directly rely on local peft package under this instance (code/peft/__init__.py), no longer inject fake modules via sys.modules
from toolbrain.adapters.smolagent.smolagent_adapter import SmolAgentAdapter  # type: ignore


def test_parse_missing_parts_extracts_thought_and_numeric_final_answer():
    # Construct instance without going through __init__
    adapter = object.__new__(SmolAgentAdapter)
    text = (
        "Thought: I will compute the result first.\n"
        "Final Answer: The value is 42."
    )
    parts = adapter._parse_missing_parts(text)
    assert parts.get("thought") == "I will compute the result first."
    # Should extract numeric part
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
    # Must cover complete text
    assert "".join(s["text"] for s in segs) == full
    # Should contain at least two assistant segments
    assert any(s["role"] == "assistant" and "reply" in s["text"] for s in segs)
    assert any(s["role"] == "assistant" and "done" in s["text"] for s in segs)