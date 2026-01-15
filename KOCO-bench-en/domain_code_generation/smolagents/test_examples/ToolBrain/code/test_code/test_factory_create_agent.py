import sys
import os
from pathlib import Path

# Make tests import actual implementation under this instance's code/
THIS_DIR = Path(__file__).resolve().parent
CODE_ROOT = THIS_DIR.parent  # .../ToolBrain/code
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
os.chdir(str(CODE_ROOT))

import types

# Import test target
from toolbrain.factory import create_agent  # type: ignore

# Ensure smolagents can be imported (for @tool)
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


# Replace smolagents.CodeAgent with a traceable FakeCodeAgent
class FakeCodeAgent:
    def __init__(self, *, model, tools, max_steps=10):
        self.model = model
        # smolagents.CodeAgent typically receives dict/mapping, maintain compatibility here: allow list and convert to list/dict
        # We only verify values are passed, not diving into smolagents internal structure
        self.tools = tools
        self.max_steps = max_steps


# A minimal @tool function conforming to smolagents JSON schema rules (with docstring/Args/Returns)
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
    # 1) Stub toolbrain.factory.create_optimized_model to return FakeModel, avoiding real model loading
    import toolbrain.factory as F  # type: ignore

    def fake_create_optimized_model(*args, **kwargs):
        return FakeModel()

    monkeypatch.setattr(F, "create_optimized_model", fake_create_optimized_model, raising=True)

    # 2) Replace smolagents.CodeAgent with our FakeCodeAgent (factory.create_agent internally does from smolagents import CodeAgent)
    monkeypatch.setattr(smolagents, "CodeAgent", FakeCodeAgent, raising=True)

    # 3) Call function under test
    agent = create_agent(
        "dummy-model",
        tools=[add],
        use_unsloth=False,
        load_in_4bit=False,
        max_steps=7,
    )

    # 4) Assert: returned is FakeCodeAgent and chat_template is injected
    assert isinstance(agent, FakeCodeAgent), "Should construct smolagents.CodeAgent (FakeCodeAgent here)"
    assert isinstance(agent.model, FakeModel), "Should use factory-created model (FakeModel here)"
    assert isinstance(agent.model.tokenizer, FakeTokenizer)
    assert isinstance(agent.model.tokenizer.chat_template, str) and len(agent.model.tokenizer.chat_template) > 0, \
        "Should populate default template when tokenizer.chat_template is empty"
    # 5) Assert tools and max_steps are passed through
    assert agent.tools and agent.tools[0].name == "add", "Tools should be passed through to CodeAgent"
    assert agent.max_steps == 7, "max_steps should be correctly passed through"