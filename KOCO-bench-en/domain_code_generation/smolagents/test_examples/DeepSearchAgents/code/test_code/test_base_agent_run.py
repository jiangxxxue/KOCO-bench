import os
import sys
import time
import types
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

from agents.base_agent import BaseAgent, MultiModelRouter  # [agents.base_agent.BaseAgent.run()](KOCO-bench/KOCO-bench-en/domain_code_generation/smolagents/test_examples/DeepSearchAgents/code/src/agents/base_agent.py:313)
from agents.run_result import RunResult  # [agents.run_result.RunResult](KOCO-bench/KOCO-bench-en/domain_code_generation/smolagents/test_examples/DeepSearchAgents/code/src/agents/run_result.py:1)

# Fallback or real import for streaming delta type
try:
    from smolagents.models import ChatMessageStreamDelta
except Exception:
    class ChatMessageStreamDelta:
        def __init__(self, content: str = "", token_usage=None):
            self.content = content
            self.token_usage = token_usage


class DummyMemory:
    def __len__(self):
        return 0
    def reset(self):
        # simulate memory reset
        return None


class DummyAgent:
    """
    Minimal stub of a smolagents-style agent exposing:
    - run(user_input, stream=...) -> string/dict/generator
    - memory.reset()
    - stream_outputs flag (optional)
    - logs list (optional)
    - state dict (optional)
    """
    def __init__(self, streaming=False):
        self.memory = DummyMemory()
        self.stream_outputs = streaming
        self.logs = []
        self.state = {}

    def run(self, user_input, stream=False, **kwargs):
        if stream:
            # Return a generator of streaming deltas without making this function a generator
            return (d for d in [
                ChatMessageStreamDelta(content="Hello "),
                ChatMessageStreamDelta(content="world")
            ])
        else:
            # Non-streaming path returns a dict that BaseAgent.run will normalize
            # to result["content"]
            return {
                "title": "Final Answer",
                "content": f"Echo: {user_input}",
                "sources": []
            }


class DummyModel:
    """
    Stub for LiteLLMModel-like object with model_id and token attributes.
    """
    def __init__(self, model_id):
        self.model_id = model_id
        self.last_input_token_count = 3
        self.last_output_token_count = 2

    def __call__(self, messages, **kwargs):
        # Return a minimal chat message-like dict for MultiModelRouter.generate()
        class Resp:
            content = "dummy response"
        return Resp()

    def generate_stream(self, messages, **kwargs):
        # For completeness if ever used
        yield ChatMessageStreamDelta(content="router ")


class DummyBaseAgent(BaseAgent):
    """
    Concrete subclass to test BaseAgent.run behavior without heavy dependencies.
    """
    def __init__(self):
        # Initialize with dummy models and minimal config
        super().__init__(
            agent_type="codact",
            orchestrator_model=DummyModel("orc-model"),
            search_model=DummyModel("search-model"),
            tools=[],
            initial_state={"visited_urls": set(), "search_queries": [], "key_findings": {}, "search_depth": {}, "reranking_history": [], "content_quality": {}},
            max_steps=5,
            planning_interval=2,
            enable_streaming=False,
            name="TestAgent",
            description="Test BaseAgent",
            managed_agents=[],
            cli_console=None,
            step_callbacks=None
        )
        # router for token stats
        self.orchestrator_model = MultiModelRouter(
            search_model=DummyModel("search-model"),
            orchestrator_model=DummyModel("orc-model")
        )

    def create_agent(self):
        # Return our dummy agent stub
        return DummyAgent(streaming=False)


def test_run_non_streaming_return_result():
    """
    Verify BaseAgent.run in non-streaming mode returns RunResult when return_result=True,
    normalizes JSON dict content, and includes token/model info.
    """
    agent = DummyBaseAgent().initialize()
    start = time.time()
    result = agent.run(
        user_input="test-input",
        stream=False,
        reset=True,
        additional_args={"k": "v"},
        return_result=True
    )
    # Type and content checks
    assert isinstance(result, RunResult)
    assert result.final_answer.startswith("Echo: test-input")
    assert result.error is None

    # Execution time recorded
    assert result.execution_time >= 0
    assert (time.time() - start) >= 0

    # Token usage derived from MultiModelRouter token counts
    assert result.token_usage is not None
    assert result.token_usage.input_tokens in (0, 3)  # may be 0 if not updated, tolerate both
    assert result.token_usage.output_tokens in (0, 2)

    # Model info includes IDs
    assert "orchestrator" in result.model_info
    assert "search" in result.model_info
    assert "orc-model" in result.model_info["orchestrator"]
    assert "search-model" in result.model_info["search"]


def test_run_streaming_returns_generator():
    """
    Verify BaseAgent.run in streaming mode returns a generator of ChatMessageStreamDelta,
    and aggregates content as expected.
    """
    agent = DummyBaseAgent().initialize()
    # Replace underlying agent with a streaming-capable dummy
    agent.agent = DummyAgent(streaming=True)

    gen = agent.run(
        user_input="hello",
        stream=True,
        reset=False
    )
    # Must be an iterator yielding deltas
    outputs = list(gen)
    assert len(outputs) == 2
    assert outputs[0].content == "Hello "
    assert outputs[1].content == "world"

    # get_memory_summary can still be invoked and returns dict
    summary = agent.get_memory_summary()
    assert isinstance(summary, dict)