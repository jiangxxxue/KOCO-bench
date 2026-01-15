import sys
import types
import pytest
import os
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from smolcc.agent import create_agent, ToolAgent
from smolcc.system_prompt import get_system_prompt
import smolcc.agent as agent_mod


def test_create_agent_uses_stubbed_llm_and_prompt(monkeypatch, tmp_path):
    # Inject mocked smolagents module to avoid real dependencies and network access
    class FakeLiteLLMModel:
        def __init__(self, model_id, api_key=None, system=None):
            self.model_id = model_id
            self.api_key = api_key
            self.system = system

    class FakeTool:
        pass

    fake_module = types.SimpleNamespace(LiteLLMModel=FakeLiteLLMModel, Tool=FakeTool)
    monkeypatch.setitem(sys.modules, "smolagents", fake_module)

    # Inject mocked smolcc.tools to avoid real BashTool/subprocess initialization
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

    # Replace system prompt with controllable value
    monkeypatch.setattr("smolcc.system_prompt.get_system_prompt", lambda cwd: "PROMPT_CONTEXT")
    
    # Avoid ToolAgent.__init__ triggering parent class initialization (depends on smolagents.prompts)
    def fake_init(self, tools, model, log_file=None, **kwargs):
        self.tools = tools
        self.model = model
        # Provide minimal logger to satisfy attribute access
        self.logger = types.SimpleNamespace(console=None)
        return None
    monkeypatch.setattr(agent_mod.ToolAgent, "__init__", fake_init, raising=True)
    
    # Construct agent (without depending on environment variables)
    agent = create_agent(cwd=str(tmp_path), log_file=None)

    # Assertions: return type, model configuration, and toolset
    assert isinstance(agent, ToolAgent), "Should return ToolAgent instance"
    assert getattr(agent.model, "system", None) == "PROMPT_CONTEXT", "Model should carry system prompt context"
    assert getattr(agent.model, "model_id", None) == "claude-3-7-sonnet-20250219", "Model ID should be preset value"
    # Should register at least one tool
    assert hasattr(agent, "tools"), "Agent should contain tool collection"
    assert len(agent.tools) >= 1, "Should load at least one tool"