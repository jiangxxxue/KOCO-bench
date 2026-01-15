import pytest
from rich.console import Console
import sys
import os
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from smolcc.agent import ToolAgent


class DummyLogger:
    def __init__(self, console: Console):
        self.console = console
        self.errors = []

    def log_error(self, msg: str):
        # Simulate RichConsoleLogger's error printing and recording
        self.errors.append(msg)
        self.console.print(f"[bold red]Error:[/bold red] {msg}")


def test_execute_tool_call_normal_displays_output_and_returns_raw_result():
    console = Console(record=True)

    # Construct ToolAgent without calling __init__, manually populate necessary attributes
    agent = object.__new__(ToolAgent)
    agent.console = console
    agent.logger = DummyLogger(console)
    # Provide a simple tool
    agent.tools = {
        "echo": lambda text: text
    }

    res = agent.execute_tool_call("echo", {"text": "hello"})
    assert res == "hello", "Should return the raw result of tool execution"

    out = console.export_text()
    # Tool call line
    assert "⏺ echo(" in out and "text" in out, "Should print tool call and parameters"
    # Non-final_answer case should display output content
    assert "hello" in out, "Should display tool output text"


def test_execute_tool_call_final_answer_skips_output_display():
    console = Console(record=True)

    agent = object.__new__(ToolAgent)
    agent.console = console
    agent.logger = DummyLogger(console)
    # final_answer tool exists and returns text
    agent.tools = {
        "final_answer": lambda text: "bye"
    }

    res = agent.execute_tool_call("final_answer", {"text": "ignored"})
    assert res == "bye", "Should return raw result"
    out = console.export_text()
    # Should print tool call but not display result content
    assert "⏺ final_answer(" in out, "Should display tool call prompt"
    assert "bye" not in out, "final_answer output should not be displayed here"


def test_execute_tool_call_missing_tool_logs_error_and_returns_error_string():
    console = Console(record=True)

    agent = object.__new__(ToolAgent)
    agent.console = console
    agent.logger = DummyLogger(console)
    agent.tools = {}  # Don't provide tool, trigger error path

    res = agent.execute_tool_call("missing_tool", {"x": 1})
    assert isinstance(res, str) and res.startswith("Error:"), "Should return error string when tool not found"

    out = console.export_text()
    # Should print tool call line
    assert "⏺ missing_tool(" in out, "Should display tool call prompt"
    # Should print error message
    assert "Error:" in out, "Should display error message via Rich"
    # Logger should also capture error
    assert agent.logger.errors, "Logger should collect error messages"