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

# Note: This project provides a minimal peft package under code/peft/__init__.py to pass Transformers' optional dependency detection,
# so we no longer inject fake modules via sys.modules, but rely on the local package directly.
# Import test targets
import toolbrain.brain as brain_mod  # type: ignore
import toolbrain.adapters.smolagent.smolagent_adapter as sa_mod  # type: ignore
from toolbrain.adapters.smolagent.smolagent_adapter import SmolAgentAdapter  # type: ignore
from toolbrain.brain import Brain  # type: ignore

# smolagents Tool and ChatMessage may be needed in some tests
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


# ========== Common Fake Types ==========
class FakeTokenizer:
    def __init__(self):
        self.chat_template = None
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, return_tensors=None):
        # Simply concatenate role:content to form complete contex
        # messages can be either smolagents ChatMessage or dict {role, content}
        parts = []
        for m in messages:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "user")
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            if isinstance(content, list):
                # ChatMessage.content may be [{"type": "text", "text": "xxx"}]
                text_items = []
                for c in content:
                    if isinstance(c, dict) and "text" in c:
                        text_items.append(str(c["text"]))
                    else:
                        text_items.append(str(c))
                content = " ".join(text_items)
            parts.append(f"{role}:{content}")
        return "\n".join(parts)

class FakeTransformersModel:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.model = types.SimpleNamespace(parameters=lambda: [])

class FakeMemory:
    def __init__(self, steps):
        self.steps = steps

class FakeActionStep:
    def __init__(self, model_output, observations, action_output, error, code_action, model_input_messages):
        self.model_output = model_output
        self.observations = observations
        self.action_output = action_output
        self.error = error
        self.code_action = code_action
        self.model_input_messages = model_input_messages

class FakeCodeAgent:
    def __init__(self, model=None, tools=None):
        self.model = model if model is not None else FakeTransformersModel()
        self.tools = {} if tools is None else tools
        self.memory = FakeMemory(steps=[])
    def write_memory_to_messages(self):
        # Generate a simplified set of messages
        return [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
    def run(self, query, reset=True):
        # Write a minimal ActionStep
        self.memory.steps = [
            FakeActionStep(
                model_output="Thought: done\nFinal Answer: 123",
                observations="tool-ok",
                action_output="",
                error=None,
                code_action="final_answer('OK')",
                model_input_messages=[
                    types.SimpleNamespace(role="user", content="u1")
                ],
            )
        ]
        return "ok"

# ========== Unit Tests ==========

def test_smolagent_adapter_extract_trace_from_memory_and_run(monkeypatch):
    """
    Verify SmolAgentAdapter._extract_trace_from_memory and run basic parsing:
    - Parse Final Answer
    - Generate formatted_conversation
    - Return key fields of Turn
    """
    # Replace CodeAgent/TransformersModel in adapter module with our Fake
    monkeypatch.setattr(sa_mod, "CodeAgent", FakeCodeAgent, raising=True)
    # TransformersModel used for CodeAgent type checking in sa_mod comes from smolagents package, replace with our Fake
    import smolagents  # type: ignore
    monkeypatch.setattr(smolagents, "TransformersModel", FakeTransformersModel, raising=True)

    agent = FakeCodeAgent()
    adapter = SmolAgentAdapter(agent=agent, config={})

    # Executing run will call agent.run and then extract trace
    trace, rl_input, raw_steps = adapter.run("hello")
    assert isinstance(trace, list) and len(trace) == 1, "Should extract 1 turn"
    turn = trace[0]
    assert "prompt_for_model" in turn and "model_completion" in turn and "parsed_completion" in turn
    # Final Answer may be parsed from code_action or text
    assert turn["parsed_completion"]["final_answer"] in ("OK", "123")
    # formatted_conversation is generated by tokenizer.apply_chat_template
    assert turn.get("formatted_conversation") is not None
    # run should also return rl_input (usable for subsequent training) and raw steps
    assert rl_input is not None
    assert isinstance(raw_steps, list) and len(raw_steps) == 1


def test_brain_is_agent_supported_with_codeagent_and_other():
    """
    Verify Brain.is_agent_supported judgment for smolagents CodeAgent and other objects.
    Since CodeAgent in brain module comes from smolagents, using FakeCodeAgent directly won't be recognized.
    We only verify the 'non-CodeAgent' scenario returns False; positive cases are covered in integration tests。
    """
   
    assert Brain.is_agent_supported(object()) is False, "Non-CodeAgent should return False"


def test_brain_generate_training_examples_with_stubbed_adapter(monkeypatch):
    """
    Verify Brain.generate_training_examples:
    - Use stubbed agent_adapter (provides get_trainable_model and get_callable_tools)
    - Use minimal tool decorated with smolagents.Tool
    - FakeModel.generate returns simple string, ensuring output is list of strings
    """
    class FakeModelForGen:
        def generate(self, messages, *args, **kwargs):
            # Return object shape similar to smolagents common output
            class _Out:
                def __init__(self, text): 
                    self.content = text
            return [_Out("query-ok")]

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

    class StubAdapter:
        def get_trainable_model(self): 
            return FakeModelForGen()
        def get_callable_tools(self):
            return [add]

    # Directly construct a minimal Brain instance, populate necessary attributes
    b = object.__new__(Brain)
    # Necessary attribute: agent_adapter
    b.agent_adapter = StubAdapter()

    # Method internally doesn't depend on self.config/self.reward_func and other training fields
    res = Brain.generate_training_examples(
        b,
        task_description="arith",
        num_examples=2,
        min_tool_calls=1,
        max_words=20,
        guidance_example=None,
        external_model=None,  # Use adapter's model
        external_tools=None,
        self_rank=False
    )
    assert isinstance(res, list) and len(res) == 2
    assert all(isinstance(x, str) for x in res)


def test_brain_get_adapter_for_agent_unsupported_raises(monkeypatch):
    """
    Verify _get_adapter_for_agent raises TypeError for unsupported types.
    Only construct minimal Brain, populate config to satisfy to_dict access.
    """
    b = object.__new__(Brain)
    # Provide minimal config object with to_dict()
    class Cfg:
        def to_dict(self): 
            return {}
    b.config = Cfg()
    with pytest.raises(TypeError):
        # Pass in unsupported object
        Brain._get_adapter_for_agent(b, agent_instance=object(), trainable_model=None)