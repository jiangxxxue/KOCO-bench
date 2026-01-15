import os
import sys
import pytest

# Path injection to import project src and smolagents from knowledge corpus
CURRENT_DIR = os.path.dirname(__file__)
CODE_DIR = os.path.dirname(CURRENT_DIR)  # .../code
SRC_DIR = os.path.join(CODE_DIR, "src")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../../"))
SMOLAGENTS_SRC = os.path.join(PROJECT_ROOT, "smolagents", "knowledge_corpus", "smolagents", "src")
if SMOLAGENTS_SRC not in sys.path:
    sys.path.insert(0, SMOLAGENTS_SRC)

# Dynamically import AgentStepCallback to avoid ui_common __init__ side-effects
import importlib.util as _importlib_util
AGENT_STEP_CB_PATH = os.path.join(SRC_DIR, "agents", "ui_common", "agent_step_callback.py")
_spec = _importlib_util.spec_from_file_location("agent_step_callback", AGENT_STEP_CB_PATH)
_agent_step_cb_mod = _importlib_util.module_from_spec(_spec)
assert _spec and _spec.loader is not None
_spec.loader.exec_module(_agent_step_cb_mod)  # type: ignore[attr-defined]
AgentStepCallback = _agent_step_cb_mod.AgentStepCallback
try:
    from smolagents import TokenUsage
except Exception:
    TokenUsage = None  # Fallback to dict-based path


class MemoryStepStub:
    """Minimal stub of smolagents.memory.MemoryStep with token_usage attribute."""
    def __init__(self, token_usage=None):
        self.token_usage = token_usage


def test_extract_token_stats_with_dict_token_usage():
    """Token stats extracted when memory_step.token_usage is a dict."""
    callback = AgentStepCallback(debug_mode=False, model=None)
    callback.step_counter = 1  # Ensure a valid step_id string

    # Prepare dict-based token usage
    memory_step = MemoryStepStub(token_usage={"input_tokens": 12, "output_tokens": 7})
    step_data = {}

    callback._extract_token_stats(memory_step, step_data)

    # Validate step_data enhancements
    assert "token_counts" in step_data
    assert step_data["token_counts"]["input_tokens"] == 12
    assert step_data["token_counts"]["output_tokens"] == 7
    assert step_data["total_tokens"] == 19

    # Validate cumulative totals
    assert "total_token_counts" in step_data
    assert step_data["total_token_counts"]["input"] == 12
    assert step_data["total_token_counts"]["output"] == 7
    assert step_data["total_token_counts"]["total"] == 19


@pytest.mark.skipif(TokenUsage is None, reason="TokenUsage not available from smolagents")
def test_extract_token_stats_with_TokenUsage_class():
    """Token stats extracted when memory_step.token_usage is a TokenUsage instance."""
    callback = AgentStepCallback(debug_mode=False, model=None)
    callback.step_counter = 2

    memory_step = MemoryStepStub(token_usage=TokenUsage(input_tokens=5, output_tokens=3))
    step_data = {}

    callback._extract_token_stats(memory_step, step_data)

    # Validate step_data enhancements
    assert "token_counts" in step_data
    assert step_data["token_counts"]["input_tokens"] == 5
    assert step_data["token_counts"]["output_tokens"] == 3
    assert step_data["total_tokens"] == 8

    # Validate cumulative totals
    assert "total_token_counts" in step_data
    totals = step_data["total_token_counts"]
    assert totals["input"] >= 5
    assert totals["output"] >= 3
    assert totals["total"] == totals["input"] + totals["output"]


def test_extract_token_stats_accumulates_over_multiple_calls():
    """Total token counts accumulate across multiple steps."""
    callback = AgentStepCallback(debug_mode=False, model=None)

    # First step
    callback.step_counter = 1
    step1 = MemoryStepStub(token_usage={"input_tokens": 4, "output_tokens": 6})
    data1 = {}
    callback._extract_token_stats(step1, data1)
    assert data1["total_token_counts"]["input"] == 4
    assert data1["total_token_counts"]["output"] == 6
    assert data1["total_token_counts"]["total"] == 10

    # Second step accumulates
    callback.step_counter = 2
    step2 = MemoryStepStub(token_usage={"input_tokens": 10, "output_tokens": 5})
    data2 = {}
    callback._extract_token_stats(step2, data2)
    assert data2["total_token_counts"]["input"] == 4 + 10
    assert data2["total_token_counts"]["output"] == 6 + 5
    assert data2["total_token_counts"]["total"] == (4 + 10) + (6 + 5)