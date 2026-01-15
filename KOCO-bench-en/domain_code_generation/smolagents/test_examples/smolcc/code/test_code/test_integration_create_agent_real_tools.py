import sys
import types
from types import SimpleNamespace
import pytest
import os
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from smolcc.agent import create_agent, ToolAgent
import smolcc.agent as agent_mod
from smolcc.system_prompt import get_system_prompt


def test_create_agent_with_real_tools_and_popen_stub(monkeypatch, tmp_path):
    # 1) Isolate external LLM dependencies
    class FakeLiteLLMModel:
        def __init__(self, model_id, api_key=None, system=None):
            self.model_id = model_id
            self.api_key = api_key
            self.system = system

    monkeypatch.setitem(sys.modules, "smolagents", types.SimpleNamespace(LiteLLMModel=FakeLiteLLMModel, Tool=object))

    # 2) Isolate external processes: make /bin/bash and other subprocess calls no-ops
    def fake_popen(args, **kwargs):
        # Return fake process object with common interfaces
        return SimpleNamespace(
            stdin=SimpleNamespace(write=lambda x: None, flush=lambda: None),
            stdout=SimpleNamespace(read=lambda: "", readline=lambda: "", close=lambda: None),
            stderr=SimpleNamespace(read=lambda: "", readline=lambda: "", close=lambda: None),
            poll=lambda: 0,
            wait=lambda timeout=None: 0,
            terminate=lambda: None,
            kill=lambda: None,
            communicate=lambda input=None, timeout=None: ("", ""),
        )

    monkeypatch.setattr("subprocess.Popen", fake_popen, raising=True)

    # 3) Optional: stabilize system prompt (can also use real implementation, simplified to constant here to avoid file/time dependency differences)
    monkeypatch.setattr("smolcc.system_prompt.get_system_prompt", lambda cwd: "PROMPT_CONTEXT")

    # 4) Force next import to use real smolcc.tools (instead of previously injected stubs)
    monkeypatch.delitem(sys.modules, "smolcc.tools", raising=False)

    # 5) Avoid parent class initialization importing smolagents.prompts, only record passed parameters to verify create_agent assembly
    captured = {}
    def fake_init(self, tools, model, log_file=None, **kwargs):
        captured["tools"] = tools
        captured["model"] = model
        captured["log_file"] = log_file
        # Provide basic attributes for subsequent access compatibility
        self.tools = tools
        self.model = model
        self.logger = types.SimpleNamespace(console=None)
        return None

    monkeypatch.setattr(agent_mod.ToolAgent, "__init__", fake_init, raising=True)

    # 6) Execute create_agent, using real smolcc.tools module and tool assembly logic
    agent = create_agent(cwd=str(tmp_path), log_file=None)

    # 7) Assertion verification: model configuration, toolset type and count
    assert isinstance(agent, ToolAgent), "Should return ToolAgent instance (using stub __init__ but not changing type)"
    assert getattr(captured.get("model"), "system", None) == "PROMPT_CONTEXT", "Model should carry system prompt context"
    assert getattr(captured.get("model"), "model_id", None) == "claude-3-7-sonnet-20250219", "Model ID should be preset value"

    tools = captured.get("tools")
    assert isinstance(tools, list) and len(tools) >= 1, "Real smolcc.tools should assemble at least one tool instance"

    # Verify tool instances come from real smolcc.tools code (class names should be specific tool names, not simple FakeTool)
    tool_class_names = {tool.__class__.__name__ for tool in tools}
    # Only check existence, don't force listing all tools to avoid platform differences
    expected_any = {"BashTool", "EditTool", "GlobTool", "GrepTool", "LSTool", "ReplaceTool", "ViewTool"}
    assert tool_class_names & expected_any, f"Should contain at least one real tool class: {expected_any}"