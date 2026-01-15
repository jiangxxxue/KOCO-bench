import sys
import types
from types import SimpleNamespace
import pytest
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from smolcc.agent import create_agent, ToolAgent
import smolcc.agent as agent_mod
from smolcc.system_prompt import get_system_prompt


def test_create_agent_with_real_tools_and_popen_stub(monkeypatch, tmp_path):
    # 1) 隔离外部 LLM 依赖
    class FakeLiteLLMModel:
        def __init__(self, model_id, api_key=None, system=None):
            self.model_id = model_id
            self.api_key = api_key
            self.system = system

    monkeypatch.setitem(sys.modules, "smolagents", types.SimpleNamespace(LiteLLMModel=FakeLiteLLMModel, Tool=object))

    # 2) 隔离外部进程：让 /bin/bash 等子进程调用成为空操作
    def fake_popen(args, **kwargs):
        # 返回具有常用接口的假进程对象
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

    # 3) 可选：稳定系统提示（也可使用真实实现，这里简化为常量，避免文件/时间依赖差异）
    monkeypatch.setattr("smolcc.system_prompt.get_system_prompt", lambda cwd: "PROMPT_CONTEXT")

    # 4) 强制下次导入使用真实 smolcc.tools（而非此前可能注入的桩）
    monkeypatch.delitem(sys.modules, "smolcc.tools", raising=False)

    # 5) 避免父类初始化导入 smolagents.prompts，仅记录传入参数以验证 create_agent 的装配
    captured = {}
    def fake_init(self, tools, model, log_file=None, **kwargs):
        captured["tools"] = tools
        captured["model"] = model
        captured["log_file"] = log_file
        # 提供基础属性以兼容后续访问
        self.tools = tools
        self.model = model
        self.logger = types.SimpleNamespace(console=None)
        return None

    monkeypatch.setattr(agent_mod.ToolAgent, "__init__", fake_init, raising=True)

    # 6) 执行 create_agent，使用真实 smolcc.tools 模块与工具装配逻辑
    agent = create_agent(cwd=str(tmp_path), log_file=None)

    # 7) 断言验证：模型配置、工具集类型与数量
    assert isinstance(agent, ToolAgent), "应返回 ToolAgent 实例（使用桩 __init__ 但不改变类型）"
    assert getattr(captured.get("model"), "system", None) == "PROMPT_CONTEXT", "模型应携带系统提示上下文"
    assert getattr(captured.get("model"), "model_id", None) == "claude-3-7-sonnet-20250219", "模型 ID 应为预设值"

    tools = captured.get("tools")
    assert isinstance(tools, list) and len(tools) >= 1, "真实 smolcc.tools 应装配出至少一个工具实例"

    # 验证工具实例来自真实 smolcc.tools 代码（类名应为具体工具名，非简单 FakeTool）
    tool_class_names = {tool.__class__.__name__ for tool in tools}
    # 只做存在性检查，不强制列出所有工具，避免平台差异
    expected_any = {"BashTool", "EditTool", "GlobTool", "GrepTool", "LSTool", "ReplaceTool", "ViewTool"}
    assert tool_class_names & expected_any, f"应至少包含一个真实工具类：{expected_any}"