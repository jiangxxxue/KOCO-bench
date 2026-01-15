import sys
import os
from pathlib import Path

# 使测试导入到本实例 code/ 下的实际实现
THIS_DIR = Path(__file__).resolve().parent
CODE_ROOT = THIS_DIR.parent  # .../ToolBrain/code
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
os.chdir(str(CODE_ROOT))

import types

# 导入待测目标
from toolbrain.factory import create_agent  # type: ignore

# 确保 smolagents 可导入（用于 @tool）
def _ensure_smolagents_path():
    try:
        import smolagents  # noqa: F401
        return
    except Exception:
        pass
    test_file = Path(__file__).resolve()
    for up in range(2, 10):
        candidate = test_file.parents[up] / "smolagents" / "src"
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            break

_ensure_smolagents_path()
from smolagents import tool  # type: ignore
import smolagents  # type: ignore


class FakeTokenizer:
    def __init__(self):
        self.chat_template = None


class FakeModel:
    def __init__(self):
        self.tokenizer = FakeTokenizer()


# 用一个可跟踪的 FakeCodeAgent 替换 smolagents.CodeAgent
class FakeCodeAgent:
    def __init__(self, *, model, tools, max_steps=10):
        self.model = model
        # smolagents.CodeAgent 通常接收 dict/mapping，这里保持兼容：允许 list 并转为列表/字典均可
        # 我们只校验传值被传递，不深入 smolagents 内部结构
        self.tools = tools
        self.max_steps = max_steps


# 一个最小 @tool 函数，符合 smolagents JSON schema 规则（含 docstring/Args/Returns）
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


def test_factory_create_agent_injects_chat_template_and_constructs_codeagent(monkeypatch):
    # 1) 打桩 toolbrain.factory.create_optimized_model 返回 FakeModel，避免真实加载模型
    import toolbrain.factory as F  # type: ignore

    def fake_create_optimized_model(*args, **kwargs):
        return FakeModel()

    monkeypatch.setattr(F, "create_optimized_model", fake_create_optimized_model, raising=True)

    # 2) 将 smolagents.CodeAgent 替换为我们的 FakeCodeAgent（factory.create_agent 内部 from smolagents import CodeAgent）
    monkeypatch.setattr(smolagents, "CodeAgent", FakeCodeAgent, raising=True)

    # 3) 调用待测函数
    agent = create_agent(
        "dummy-model",
        tools=[add],
        use_unsloth=False,
        load_in_4bit=False,
        max_steps=7,
    )

    # 4) 断言：返回的是 FakeCodeAgent，且注入了 chat_template
    assert isinstance(agent, FakeCodeAgent), "应构造 smolagents.CodeAgent（此处为 FakeCodeAgent）"
    assert isinstance(agent.model, FakeModel), "应使用工厂创建的模型（此处为 FakeModel）"
    assert isinstance(agent.model.tokenizer, FakeTokenizer)
    assert isinstance(agent.model.tokenizer.chat_template, str) and len(agent.model.tokenizer.chat_template) > 0, \
        "当 tokenizer.chat_template 为空时应填充默认模板"
    # 5) 断言工具与 max_steps 透传
    assert agent.tools and agent.tools[0].name == "add", "工具应透传给 CodeAgent"
    assert agent.max_steps == 7, "max_steps 应正确透传"