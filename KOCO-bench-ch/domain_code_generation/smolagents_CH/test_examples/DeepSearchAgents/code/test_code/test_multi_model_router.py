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
# Path: KOCO-bench/KOCO-bench-ch/domain_code_generation/smolagents/knowledge_corpus/smolagents/src
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../../"))
SMOLAGENTS_SRC = os.path.join(PROJECT_ROOT, "smolagents", "knowledge_corpus", "smolagents", "src")
if SMOLAGENTS_SRC not in sys.path:
    sys.path.insert(0, SMOLAGENTS_SRC)

# Import the router under test
from agents.base_agent import MultiModelRouter  # [agents.base_agent.MultiModelRouter](KOCO-bench/KOCO-bench-ch/domain_code_generation/smolagents/test_examples/DeepSearchAgents/code/src/agents/base_agent.py:39)

# Provide fallback stubs if smolagents.models is unavailable in environment
try:
    from smolagents.models import ChatMessage, ChatMessageStreamDelta  # [smolagents.models.ChatMessage](KOCO-bench/KOCO-bench-ch/domain_code_generation/smolagents/knowledge_corpus/smolagents/src/smolagents/models.py:1)
except Exception:
    class ChatMessage:
        def __init__(self, content: str = ""):
            self.content = content

    class ChatMessageStreamDelta:
        def __init__(self, content: str = "", role: str = ""):
            self.content = content
            self.role = role


class DummyStreamingOrchestratorModel:
    """Streaming-capable orchestrator model stub."""
    def __init__(self):
        self.model_id = "orchestrator-model"
        # Token count attributes expected by router
        self.last_input_token_count = 0
        self.last_output_token_count = 0

    def __call__(self, messages, **kwargs) -> ChatMessage:
        # Return a non-streaming ChatMessage when called via generate()
        return ChatMessage("orchestrator direct reply")

    def generate_stream(self, messages, **kwargs):
        # Simulate streaming deltas
        self.last_input_token_count = 3
        self.last_output_token_count = 0
        yield ChatMessageStreamDelta(content="Plan ")
        yield ChatMessageStreamDelta(content="Step")


class DummyStreamingSearchModel:
    """Streaming-capable search model stub."""
    def __init__(self):
        self.model_id = "search-model"
        self.last_input_token_count = 0
        self.last_output_token_count = 0

    def __call__(self, messages, **kwargs) -> ChatMessage:
        return ChatMessage("search direct reply")

    def generate_stream(self, messages, **kwargs):
        # Simulate streaming deltas
        self.last_input_token_count = 7
        self.last_output_token_count = 0
        yield ChatMessageStreamDelta(content="General ")
        yield ChatMessageStreamDelta(content="Text")


class DummyNonStreamingOrchestratorModel:
    """Non-streaming orchestrator model to trigger fallback path in wrapper."""
    def __init__(self):
        self.model_id = "orchestrator-model"
        self.last_input_token_count = 2
        self.last_output_token_count = 0

    def __call__(self, messages, **kwargs) -> ChatMessage:
        return ChatMessage("orchestrator direct reply")

    def generate(self, messages, **kwargs):
        # Return a ChatMessage-like response
        class Response:
            content = "hello world"
        return Response()


def make_messages_with_planning():
    return [{"role": "user", "content": "Facts survey: create a plan for research"}]


def make_messages_with_final_answer():
    return [{"role": "user", "content": "Provide final answer to the original question"}]


def make_messages_general():
    return [{"role": "user", "content": "General query about weather"}]


def collect_stream_content(deltas):
    return "".join([d.content or "" for d in deltas])


def test_generate_stream_selects_orchestrator_for_planning():
    """Router should use orchestrator model for planning/final answer contexts."""
    router = MultiModelRouter(
        search_model=DummyStreamingSearchModel(),
        orchestrator_model=DummyStreamingOrchestratorModel()
    )

    # Planning content triggers orchestrator path
    deltas = list(router.generate_stream(make_messages_with_planning()))
    content = collect_stream_content(deltas)
    assert content == "Plan Step"

    # Token counts copied from orchestrator model after streaming
    counts = router.get_token_counts()
    assert counts["input"] == 3
    assert counts["output"] == 0


def test_generate_stream_selects_search_for_general_queries():
    """Router should use search model for general queries."""
    router = MultiModelRouter(
        search_model=DummyStreamingSearchModel(),
        orchestrator_model=DummyStreamingOrchestratorModel()
    )

    deltas = list(router.generate_stream(make_messages_general()))
    content = collect_stream_content(deltas)
    assert content == "General Text"

    counts = router.get_token_counts()
    assert counts["input"] == 7
    assert counts["output"] == 0


def test_generate_stream_fallback_non_streaming_orchestrator():
    """When selected model lacks generate_stream, wrapper falls back to non-streaming generate()."""
    router = MultiModelRouter(
        search_model=DummyStreamingSearchModel(),
        orchestrator_model=DummyNonStreamingOrchestratorModel()
    )

    # Force orchestrator selection via final answer content
    deltas = list(router.generate_stream(make_messages_with_final_answer()))
    assert len(deltas) == 1
    assert deltas[0].content == "hello world"

    # Token counts should reflect orchestrator's attributes
    counts = router.get_token_counts()
    assert counts["input"] == 2
    assert counts["output"] == 0