import os
import sys
import pytest

# Ensure we can import project src modules relative to this test file
CURRENT_DIR = os.path.dirname(__file__)
CODE_DIR = os.path.dirname(CURRENT_DIR)  # .../code
SRC_DIR = os.path.join(CODE_DIR, "src")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Also add smolagents library source path from knowledge corpus
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../../"))
SMOLAGENTS_SRC = os.path.join(PROJECT_ROOT, "smolagents", "knowledge_corpus", "smolagents", "src")
if SMOLAGENTS_SRC not in sys.path:
    sys.path.insert(0, SMOLAGENTS_SRC)

# Dynamically import web_ui and models with proper package context to support relative imports
import importlib.util as _ilu

WEB_UI_PATH = os.path.join(SRC_DIR, "api", "v2", "web_ui.py")
MODELS_PATH = os.path.join(SRC_DIR, "api", "v2", "models.py")

def _load_module(name: str, path: str, package: str = None):
    spec = _ilu.spec_from_file_location(name, path)
    assert spec and spec.loader is not None
    mod = _ilu.module_from_spec(spec)  # type: ignore[arg-type]
    if package:
        mod.__package__ = package
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# Load models first under 'api.v2.models' so web_ui's relative import can resolve
_models = _load_module("api.v2.models", MODELS_PATH, package="api.v2")
_web = _load_module("api.v2.web_ui", WEB_UI_PATH, package="api.v2")

process_planning_step = _web.process_planning_step  # [api.v2.web_ui.process_planning_step()](KOCO-bench/KOCO-bench-en/domain_code_generation/smolagents/test_examples/DeepSearchAgents/code/src/api/v2/web_ui.py:186)
process_action_step = _web.process_action_step  # [api.v2.web_ui.process_action_step()](KOCO-bench/KOCO-bench-en/domain_code_generation/smolagents/test_examples/DeepSearchAgents/code/src/api/v2/web_ui.py:395)
process_final_answer_step = _web.process_final_answer_step  # [api.v2.web_ui.process_final_answer_step()](KOCO-bench/KOCO-bench-en/domain_code_generation/smolagents/test_examples/DeepSearchAgents/code/src/api/v2/web_ui.py:743)
DSAgentRunMessage = _models.DSAgentRunMessage  # [api.v2.models.DSAgentRunMessage](KOCO-bench/KOCO-bench-en/domain_code_generation/smolagents/test_examples/DeepSearchAgents/code/src/api/v2/models.py:1)


class TokenUsageStub:
    def __init__(self, input_tokens=0, output_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class TimingStub:
    def __init__(self, duration=None):
        self.duration = duration


class PlanningStepStub:
    def __init__(self, plan="", token_usage=None, timing=None, error=None):
        self.plan = plan
        self.token_usage = token_usage
        self.timing = timing
        self.error = error


class ToolCallStub:
    def __init__(self, name, arguments=None, id=None):
        self.name = name
        self.arguments = arguments or {}
        self.id = id or "tool-1"


class ActionStepStub:
    def __init__(
        self,
        step_number=1,
        model_output="",
        tool_calls=None,
        code_action="",
        observations="",
        observations_images=None,
        timing=None,
        error=None,
    ):
        self.step_number = step_number
        self.model_output = model_output
        self.tool_calls = tool_calls or []
        self.code_action = code_action
        self.observations = observations
        self.observations_images = observations_images or []
        self.timing = timing
        self.error = error


class FinalAnswerStepStub:
    def __init__(self, output):
        self.output = output


def collect_messages(gen):
    return list(gen)


def find_messages_by_type(messages, message_type):
    return [m for m in messages if isinstance(m, DSAgentRunMessage) and m.metadata.get("message_type") == message_type]


def test_process_planning_step_initial_non_streaming():
    step = PlanningStepStub(
        plan="```markdown\nPlan line 1\nPlan line 2\n```",
        token_usage=TokenUsageStub(input_tokens=10, output_tokens=5),
        timing=TimingStub(duration=1.23),
    )
    msgs = collect_messages(
        process_planning_step(
            step_log=step,
            step_number=1,
            session_id="sess-1",
            skip_model_outputs=False,
            is_streaming=False,
            planning_interval=2,
        )
    )

    # Expect header, content, footer, separator
    types = [m.metadata.get("message_type") for m in msgs]
    assert "planning_header" in types
    assert "planning_content" in types
    assert "planning_footer" in types
    assert "separator" in types

    # Header assertions
    header = find_messages_by_type(msgs, "planning_header")[0]
    assert header.content == ""
    assert header.metadata["planning_type"] == "initial"
    assert header.metadata["is_update_plan"] is False

    # Content assertions - cleaned inner content
    content_msg = find_messages_by_type(msgs, "planning_content")[0]
    assert "Plan line 1" in content_msg.content
    assert "Plan line 2" in content_msg.content
    assert content_msg.metadata["content_length"] == len(content_msg.content)
    # Token/timing metadata present
    assert "token_usage" in content_msg.metadata
    assert content_msg.metadata["token_usage"]["input_tokens"] == 10
    assert content_msg.metadata["token_usage"]["output_tokens"] == 5
    assert content_msg.metadata["timing"]["duration"] == 1.23


def test_process_action_step_python_interpreter_with_logs():
    code_text = "import math\nresult = search_links(query='ai')\nprint('done')"
    model_output_md = f"```python\n{code_text}\n```"
    tool_call = ToolCallStub(name="python_interpreter", arguments={"code": code_text}, id="t1")
    step = ActionStepStub(
        step_number=2,
        model_output=model_output_md,
        tool_calls=[tool_call],
        code_action=code_text,  # Accurate code for python_interpreter path
        observations="Execution logs:\nLine1\nLine2\nOK",
        timing=TimingStub(duration=2.5),
    )

    msgs = collect_messages(process_action_step(step, session_id="sess-2", skip_model_outputs=False))

    # Expect header, thought, tool_call badge(s), webide/code invocation, terminal logs, footer, separator
    types = [m.metadata.get("message_type") for m in msgs]
    assert "action_header" in types
    assert "action_thought" in types
    assert "tool_call" in types  # at least python_interpreter badge
    assert "tool_invocation" in types
    assert "execution_logs" in types
    assert "action_footer" in types
    assert "separator" in types

    # Thought message
    thought = find_messages_by_type(msgs, "action_thought")[0]
    assert "import math" in thought.content
    assert thought.metadata["is_raw_thought"] is True
    assert thought.metadata["full_thought_length"] >= len(thought.content)

    # WebIDE message
    webide = [m for m in msgs if m.metadata.get("component") == "webide"][0]
    assert "```python" in webide.content
    assert webide.metadata["tool_name"] == "python_interpreter"
    assert webide.metadata["language"] == "python"
    assert "code" in webide.metadata
    assert "search_links" in (webide.metadata["code"] or "")

    # Terminal logs
    terminal = [m for m in msgs if m.metadata.get("component") == "terminal"][0]
    assert "Line1" in terminal.content
    assert "Line2" in terminal.content
    assert terminal.metadata["output_lines"] >= 3
    assert terminal.metadata["has_error"] is False
    assert terminal.metadata.get("execution_duration") == 2.5


def test_process_final_answer_step_dict_structured():
    # Using dict final answer to avoid importing smolagents AgentText types
    final_answer_dict = {
        "title": "My Report",
        "content": "Findings summarized in markdown.",
        "sources": ["https://a.com/x", "https://b.org/y"],
    }
    step = FinalAnswerStepStub(output=final_answer_dict)

    msgs = collect_messages(process_final_answer_step(step, session_id="sess-3", step_number=3))

    # Single final answer message with empty content and structured metadata
    assert len(msgs) == 1
    msg = msgs[0]
    assert msg.metadata["message_type"] == "final_answer"
    assert msg.content == ""  # content empty, frontend uses metadata directly
    assert msg.metadata["has_structured_data"] is True
    assert msg.metadata["answer_title"] == "My Report"
    assert msg.metadata["answer_content"].startswith("Findings summarized")
    assert msg.metadata["answer_sources"] == final_answer_dict["sources"]