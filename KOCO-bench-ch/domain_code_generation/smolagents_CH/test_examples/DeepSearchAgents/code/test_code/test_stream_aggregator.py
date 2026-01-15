import os
import sys
import pytest

# Ensure we can import project src modules relative to this test file
CURRENT_DIR = os.path.dirname(__file__)
CODE_DIR = os.path.dirname(CURRENT_DIR)  # .../code
SRC_DIR = os.path.join(CODE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Also add smolagents library source path from knowledge corpus
# Path: KOCO-bench/KOCO-bench-ch/domain_code_generation/smolagents/knowledge_corpus/smolagents/src
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../../"))
SMOLAGENTS_SRC = os.path.join(
    PROJECT_ROOT,
    "smolagents",
    "knowledge_corpus",
    "smolagents",
    "src"
)
if SMOLAGENTS_SRC not in sys.path:
    sys.path.insert(0, SMOLAGENTS_SRC)

from agents.stream_aggregator import StreamAggregator, ModelStreamWrapper
# Fallback stub for ChatMessageStreamDelta if smolagents is not importable in environment
try:
    from smolagents.models import ChatMessageStreamDelta
except Exception:
    class ChatMessageStreamDelta:
        def __init__(self, content: str = "", role: str = ""):
            self.content = content
            self.role = role


def make_basic_stream():
    # Two deltas that should concatenate to "Hello world"
    yield ChatMessageStreamDelta(content="Hello ")
    yield ChatMessageStreamDelta(content="world")


def make_error_stream():
    # Yield one chunk then raise an error to trigger error delta generation
    yield ChatMessageStreamDelta(content="start ")
    raise RuntimeError("boom")


class DummyNonStreamingModel:
    def generate(self, messages, **kwargs):
        class Response:
            content = "hello world"
        return Response()


class DummyStreamingModel:
    def generate_stream(self, messages, **kwargs):
        # Two deltas -> "Hello world"
        yield ChatMessageStreamDelta(content="Hello ")
        yield ChatMessageStreamDelta(content="world")


def test_aggregate_stream_basic():
    aggregator = StreamAggregator()
    deltas = list(aggregator.aggregate_stream(make_basic_stream(), track_tokens=True))

    # Assert original deltas are forwarded unchanged
    assert len(deltas) == 2
    assert deltas[0].content == "Hello "
    assert deltas[1].content == "world"

    # Assert content aggregation and token estimation (space-separated)
    assert aggregator.get_aggregated_content() == "Hello world"
    assert aggregator.get_token_count() == 2  # "Hello" + "world"


def test_aggregate_stream_error_handling():
    aggregator = StreamAggregator()
    deltas = list(aggregator.aggregate_stream(make_error_stream(), track_tokens=True))

    # First, the normal delta, then an error delta should be produced
    assert len(deltas) == 2
    assert deltas[0].content == "start "
    assert deltas[1].content.startswith("Error: ")

    # Aggregated content and tokens reflect only successful chunks
    assert aggregator.get_aggregated_content() == "start "
    assert aggregator.get_token_count() == 1  # "start"


def test_model_stream_wrapper_fallback_non_streaming():
    wrapper = ModelStreamWrapper(DummyNonStreamingModel())

    outputs = list(wrapper.generate_stream(messages=[{"role": "user", "content": "hi"}]))
    # Non-streaming fallback yields a single delta from response.content
    assert len(outputs) == 1
    assert outputs[0].content == "hello world"

    summary = wrapper.get_aggregated_response()
    assert summary["content"] == "hello world"
    # Token count: "hello" (1) + "world" (1) => 2
    assert summary["token_count"] == 2


def test_model_stream_wrapper_streaming_path():
    wrapper = ModelStreamWrapper(DummyStreamingModel())

    outputs = list(wrapper.generate_stream(messages=[{"role": "user", "content": "hi"}]))
    assert len(outputs) == 2
    assert outputs[0].content == "Hello "
    assert outputs[1].content == "world"

    summary = wrapper.get_aggregated_response()
    assert summary["content"] == "Hello world"
    assert summary["token_count"] == 2


def test_aggregator_reset_and_add_chunk():
    aggregator = StreamAggregator()
    # Simulate prior aggregation
    list(aggregator.aggregate_stream(make_basic_stream(), track_tokens=True))
    assert aggregator.get_aggregated_content() == "Hello world"
    assert aggregator.get_token_count() == 2

    # Reset and check cleared state
    aggregator.reset()
    assert aggregator.get_aggregated_content() == ""
    assert aggregator.get_token_count() == 0

    # Add chunk manually and check token counting and alias method
    aggregator.add_chunk("new words here")
    assert aggregator.get_aggregated_content() == "new words here"
    assert aggregator.get_full_content() == "new words here"
    assert aggregator.get_token_count() == 3  # "new" "words" "here"