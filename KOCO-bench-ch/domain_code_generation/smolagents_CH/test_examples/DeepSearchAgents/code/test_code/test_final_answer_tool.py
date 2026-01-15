import os
import sys
import pytest

# Ensure we can import project src modules relative to this test file
CURRENT_DIR = os.path.dirname(__file__)
CODE_DIR = os.path.dirname(CURRENT_DIR)  # .../code
SRC_DIR = os.path.join(CODE_DIR, "src")
# Allow absolute imports like "src.tools.*" by adding the parent of "src" to sys.path
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

# Dynamically import final_answer.py directly to avoid tools package side effects
import importlib.util as _importlib_util

FINAL_ANSWER_PATH = os.path.join(SRC_DIR, "tools", "final_answer.py")
_spec = _importlib_util.spec_from_file_location("final_answer", FINAL_ANSWER_PATH)
_final_answer_mod = _importlib_util.module_from_spec(_spec)
assert _spec and _spec.loader is not None
_spec.loader.exec_module(_final_answer_mod)  # type: ignore[attr-defined]
EnhancedFinalAnswerTool = _final_answer_mod.EnhancedFinalAnswerTool


def has_sources_markdown(content: str, urls: list) -> bool:
    if "## Sources" not in content:
        return False
    for u in urls:
        if u not in content:
            return False
    return True


def test_forward_string_with_urls_extracts_sources():
    tool = EnhancedFinalAnswerTool()
    s = "This is an answer. See https://example.com/path and http://foo.bar/baz."
    out = tool.forward(s)
    assert isinstance(out, dict)
    assert out.get("title") == "Final Answer"
    assert out.get("content") == s
    # Normalize extracted URLs to strip trailing punctuation the regex may capture
    normalized_sources = [u.rstrip('.,)]') for u in out.get("sources", [])]
    assert normalized_sources == ["https://example.com/path", "http://foo.bar/baz"]


def test_forward_json_string_valid():
    tool = EnhancedFinalAnswerTool()
    payload = {
        "title": "My Title",
        "content": "Body content",
        "sources": ["https://a.com/doc", "http://b.org/page"]
    }
    s = __import__("json").dumps(payload)
    out = tool.forward(s)
    assert out["title"] == "My Title"
    assert "Body content" in out["content"]
    assert out["sources"] == payload["sources"]
    assert out.get("type") == "final_answer"
    assert out.get("format") == "markdown"
    assert has_sources_markdown(out["content"], out["sources"])


def test_forward_json_string_invalid_falls_back_to_string():
    tool = EnhancedFinalAnswerTool()
    s = "{ invalid json }"
    out = tool.forward(s)
    assert out["title"] == "Final Answer"
    assert out["content"] == s
    assert out["sources"] == []


def test_forward_dict_with_sources_appends_sources():
    tool = EnhancedFinalAnswerTool()
    d = {
        "title": "Report",
        "content": "Findings here",
        "sources": ["https://x.com/a", "https://y.org/b"]
    }
    out = tool.forward(d)
    assert out["title"] == "Report"
    assert "Findings here" in out["content"]
    assert out["sources"] == d["sources"]
    assert has_sources_markdown(out["content"], out["sources"])


def test_forward_other_type_number():
    tool = EnhancedFinalAnswerTool()
    out = tool.forward(12345)
    assert out["title"] == "Final Answer"
    assert out["content"] == "12345"
    assert out["sources"] == []