import pytest
from rich.console import Console
import sys
import os
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from smolcc.agent import ToolAgent, RichConsoleLogger
# To avoid dependency on external smolagents package, dynamically get parent class reference (ToolAgent's direct base class)
ParentClass = ToolAgent.__mro__[1]

class DummyLogger:
    def __init__(self, console: Console):
        self.console = console
        self.errors = []

    def log_error(self, msg: str):
        self.errors.append(msg)
        self.console.print(f"[bold red]Error:[/bold red] {msg}")


def test_format_messages_for_llm_calls_super_and_returns_messages(monkeypatch):
    console = Console(record=True)

    # Construct ToolAgent without calling __init__ and inject console
    agent = object.__new__(ToolAgent)
    agent.console = console

    # Record super call count and return expected messages
    call_count = {"n": 0}
    expected = [{"role": "user", "content": "hello"}]

    def fake_super_format(self, prompt):
        call_count["n"] += 1
        return expected

    monkeypatch.setattr(ParentClass, "format_messages_for_llm", fake_super_format, raising=False)

    messages = agent.format_messages_for_llm("hello")
    assert messages == expected, "Should return parent class formatted messages"
    assert call_count["n"] == 1, "Should call parent class method once to complete message formatting"

    # Optional validation: progress indicator text (produced by Rich Progress, may not be retained when transient=True)
    # text = console.export_text()
    # assert "Thinking..." in text


def test_run_displays_assistant_output_and_returns_result(monkeypatch):
    console = Console(record=True)
    agent = object.__new__(ToolAgent)
    agent.console = console
    agent.logger = DummyLogger(console)

    # Make parent class run return fixed result
    monkeypatch.setattr(ParentClass, "run", lambda self, user_input, stream=False: "final result")

    res = agent.run("query")
    assert res == "final result", "run should return parent class result"
    out = console.export_text()
    assert "final result" in out, "Should display final result text as assistant output"


def test_rich_console_logger_log_error_writes_file_and_prints(tmp_path):
    log_file = tmp_path / "log.txt"
    logger = RichConsoleLogger(log_file=str(log_file))
    # Reset console to recordable
    logger.console = Console(record=True)

    logger.log_error("oops")
    out = logger.console.export_text()
    assert "Error: oops" in out, "Console should highlight error message"

    content = log_file.read_text()
    assert "ERROR: oops" in content, "Should write to log file for subsequent tracking"