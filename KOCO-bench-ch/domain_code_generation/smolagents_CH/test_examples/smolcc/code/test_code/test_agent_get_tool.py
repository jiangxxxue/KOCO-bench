import pytest
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from smolcc.agent import ToolAgent


def test_get_tool_returns_instance():
    # 通过绕过 __init__ 构造对象，避免依赖外部模型/工具初始化
    agent = object.__new__(ToolAgent)
    dummy_tool = object()
    agent.tools = {"mytool": dummy_tool}

    result = agent.get_tool("mytool")
    assert result is dummy_tool, "应返回与名称对应的工具实例"


def test_get_tool_raises_value_error_for_missing_tool():
    agent = object.__new__(ToolAgent)
    agent.tools = {}

    with pytest.raises(ValueError) as excinfo:
        agent.get_tool("not_exists")
    assert "Tool not found" in str(excinfo.value), "缺失工具应抛出 ValueError 并提示工具不存在"