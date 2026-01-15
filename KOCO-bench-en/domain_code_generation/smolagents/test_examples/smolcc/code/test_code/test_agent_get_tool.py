import pytest
import sys
import os
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from smolcc.agent import ToolAgent


def test_get_tool_returns_instance():
    # Construct object by bypassing __init__ to avoid dependency on external model/tool initialization
    agent = object.__new__(ToolAgent)
    dummy_tool = object()
    agent.tools = {"mytool": dummy_tool}

    result = agent.get_tool("mytool")
    assert result is dummy_tool, "Should return tool instance corresponding to name"


def test_get_tool_raises_value_error_for_missing_tool():
    agent = object.__new__(ToolAgent)
    agent.tools = {}

    with pytest.raises(ValueError) as excinfo:
        agent.get_tool("not_exists")
    assert "Tool not found" in str(excinfo.value), "Missing tool should raise ValueError and indicate tool not found"