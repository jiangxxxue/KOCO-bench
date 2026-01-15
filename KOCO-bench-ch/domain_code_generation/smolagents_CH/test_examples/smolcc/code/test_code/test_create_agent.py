import sys
import types
import pytest
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from smolcc.agent import create_agent, ToolAgent
from smolcc.system_prompt import get_system_prompt
import smolcc.agent as agent_mod


def test_create_agent_uses_stubbed_llm_and_prompt(monkeypatch, tmp_path):
    # 注入模拟的 smolagents 模块，避免真实依赖与联网
    class FakeLiteLLMModel:
        def __init__(self, model_id, api_key=None, system=None):
            self.model_id = model_id
            self.api_key = api_key
            self.system = system

    class FakeTool:
        pass

    fake_module = types.SimpleNamespace(LiteLLMModel=FakeLiteLLMModel, Tool=FakeTool)
    monkeypatch.setitem(sys.modules, "smolagents", fake_module)

    # 注入模拟的 smolcc.tools，避免真实 BashTool/子进程初始化
    fake_tools = types.SimpleNamespace(
        BashTool=FakeTool(),
        EditTool=FakeTool(),
        GlobTool=FakeTool(),
        GrepTool=FakeTool(),
        LSTool=FakeTool(),
        ReplaceTool=FakeTool(),
        ViewTool=FakeTool(),
        UserInputTool=FakeTool(),
    )
    monkeypatch.setitem(sys.modules, "smolcc.tools", fake_tools)

    # 替换系统提示为可控值
    monkeypatch.setattr("smolcc.system_prompt.get_system_prompt", lambda cwd: "PROMPT_CONTEXT")
    
    # 避免 ToolAgent.__init__ 触发父类初始化（依赖 smolagents.prompts）
    def fake_init(self, tools, model, log_file=None, **kwargs):
        self.tools = tools
        self.model = model
        # 提供最小 logger 以满足属性访问
        self.logger = types.SimpleNamespace(console=None)
        return None
    monkeypatch.setattr(agent_mod.ToolAgent, "__init__", fake_init, raising=True)
    
    # 构造 agent（不依赖环境变量）
    agent = create_agent(cwd=str(tmp_path), log_file=None)

    # 断言：返回类型、模型配置与工具集
    assert isinstance(agent, ToolAgent), "应返回 ToolAgent 实例"
    assert getattr(agent.model, "system", None) == "PROMPT_CONTEXT", "模型应携带系统提示上下文"
    assert getattr(agent.model, "model_id", None) == "claude-3-7-sonnet-20250219", "模型 ID 应为预设值"
    # 至少应注册一个工具
    assert hasattr(agent, "tools"), "Agent 应包含工具集合"
    assert len(agent.tools) >= 1, "应至少加载一个工具"