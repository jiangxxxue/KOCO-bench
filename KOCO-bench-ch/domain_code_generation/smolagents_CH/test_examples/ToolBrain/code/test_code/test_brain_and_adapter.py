import sys
import os
from pathlib import Path
import types

# 使测试导入到本实例 code/ 下的实际实现
THIS_DIR = Path(__file__).resolve().parent
CODE_ROOT = THIS_DIR.parent  # .../ToolBrain/code
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
os.chdir(str(CODE_ROOT))

import pytest  # type: ignore

# 注意：本项目已在 code/peft/__init__.py 下提供最小 peft 包以通过 Transformers 的可选依赖探测，
# 此处不再通过 sys.modules 注入假模块，直接依赖本地包。
# 导入待测目标
import toolbrain.brain as brain_mod  # type: ignore
import toolbrain.adapters.smolagent.smolagent_adapter as sa_mod  # type: ignore
from toolbrain.adapters.smolagent.smolagent_adapter import SmolAgentAdapter  # type: ignore
from toolbrain.brain import Brain  # type: ignore

# smolagents 的 Tool 与 ChatMessage 可能在部分测试中需要
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


# ========== 公用 Fake 类型 ==========
class FakeTokenizer:
    def __init__(self):
        self.chat_template = None
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, return_tensors=None):
        # 简单拼接 role:content 形成“完整上下文”
        # messages 既可能是 smolagents 的 ChatMessage，也可能是字典 {role, content}
        parts = []
        for m in messages:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "user")
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            if isinstance(content, list):
                # ChatMessage.content 可能是 [{"type": "text", "text": "xxx"}]
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
        # 生成一组简化的消息
        return [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
    def run(self, query, reset=True):
        # 写入一个最小的 ActionStep
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

# ========== 单元测试 ==========

def test_smolagent_adapter_extract_trace_from_memory_and_run(monkeypatch):
    """
    验证 SmolAgentAdapter._extract_trace_from_memory 与 run 的基本解析：
    - 解析 Final Answer
    - 生成 formatted_conversation
    - 返回 Turn 的关键字段
    """
    # 将适配器模块中的 CodeAgent/TransformersModel 替换为我们的 Fake
    monkeypatch.setattr(sa_mod, "CodeAgent", FakeCodeAgent, raising=True)
    # sa_mod 中 CodeAgent 类型检查使用的 TransformersModel 来自 smolagents 包，替换为我们的 Fake
    import smolagents  # type: ignore
    monkeypatch.setattr(smolagents, "TransformersModel", FakeTransformersModel, raising=True)

    agent = FakeCodeAgent()
    adapter = SmolAgentAdapter(agent=agent, config={})

    # 执行 run 会调用 agent.run 并随后抽取 trace
    trace, rl_input, raw_steps = adapter.run("hello")
    assert isinstance(trace, list) and len(trace) == 1, "应抽取到 1 个 turn"
    turn = trace[0]
    assert "prompt_for_model" in turn and "model_completion" in turn and "parsed_completion" in turn
    # Final Answer 可能从 code_action 或文本中被解析
    assert turn["parsed_completion"]["final_answer"] in ("OK", "123")
    # formatted_conversation 由 tokenizer.apply_chat_template 生成
    assert turn.get("formatted_conversation") is not None
    # run 还应返回 rl_input（可用于后续训练）与原始 steps
    assert rl_input is not None
    assert isinstance(raw_steps, list) and len(raw_steps) == 1


def test_brain_is_agent_supported_with_codeagent_and_other():
    """
    验证 Brain.is_agent_supported 针对 smolagents CodeAgent 与其他对象的判定。
    由于 brain 模块中 CodeAgent 来自 smolagents，直接使用 FakeCodeAgent 不会被认作。
    我们仅验证“非 CodeAgent”场景返回 False；正向用例在集成测试中覆盖。
    """
    assert Brain.is_agent_supported(object()) is False, "非 CodeAgent 应返回 False"


def test_brain_generate_training_examples_with_stubbed_adapter(monkeypatch):
    """
    验证 Brain.generate_training_examples：
    - 使用 stub 的 agent_adapter（提供 get_trainable_model 与 get_callable_tools）
    - 使用 smolagents.Tool 装饰的最小工具
    - FakeModel.generate 返回简单串，保证输出为字符串列表
    """
    class FakeModelForGen:
        def generate(self, messages, *args, **kwargs):
            # 返回与 smolagents 的常见输出类似的对象形态
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

    # 直接构造一个最小 Brain 实例，填充必要属性
    b = object.__new__(Brain)
    # 必要属性：agent_adapter
    b.agent_adapter = StubAdapter()

    # 方法内部不依赖 self.config/self.reward_func 等训练字段
    res = Brain.generate_training_examples(
        b,
        task_description="arith",
        num_examples=2,
        min_tool_calls=1,
        max_words=20,
        guidance_example=None,
        external_model=None,  # 使用 adapter 的模型
        external_tools=None,
        self_rank=False
    )
    assert isinstance(res, list) and len(res) == 2
    assert all(isinstance(x, str) for x in res)


def test_brain_get_adapter_for_agent_unsupported_raises(monkeypatch):
    """
    验证 _get_adapter_for_agent 对不支持类型抛出 TypeError。
    仅构造最小 Brain，填充 config 以满足 to_dict 访问。
    """
    b = object.__new__(Brain)
    # 提供最小 config 对象，具备 to_dict()
    class Cfg:
        def to_dict(self): 
            return {}
    b.config = Cfg()
    with pytest.raises(TypeError):
        # 传入不支持的对象
        Brain._get_adapter_for_agent(b, agent_instance=object(), trainable_model=None)