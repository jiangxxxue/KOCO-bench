import pytest
from rich.console import Console
import sys
import os
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from smolcc.tool_output import (
    ToolOutput,
    TextOutput,
    CodeOutput,
    TableOutput,
    FileListOutput,
    ToolCallOutput,
    AssistantOutput,
    convert_to_tool_output,
)


def test_convert_string_detects_code_output():
    code_str = "def foo():\n    return 1"
    out = convert_to_tool_output(code_str)
    assert isinstance(out, CodeOutput), "String containing code structure should be recognized as CodeOutput"


def test_convert_string_plain_text_output():
    text_str = "hello world"
    out = convert_to_tool_output(text_str)
    assert isinstance(out, TextOutput), "Plain string should be converted to TextOutput"


def test_convert_list_of_lists_table_output():
    table_data = [[1, 2], [3, 4]]
    out = convert_to_tool_output(table_data)
    assert isinstance(out, TableOutput), "List of lists should be converted to TableOutput"


def test_convert_list_of_dicts_file_list_output():
    files = [{"name": "file.txt", "is_dir": False, "size": 100}]
    out = convert_to_tool_output(files)
    assert isinstance(out, FileListOutput), "List of dicts containing 'name' should be converted to FileListOutput"


def test_convert_existing_tool_output_passthrough():
    existing = TextOutput("existing")
    out = convert_to_tool_output(existing)
    assert out is existing, "Existing ToolOutput should be returned as-is"


def test_convert_default_text_output_for_other_types():
    other = {"a": 1}
    out = convert_to_tool_output(other)
    assert isinstance(out, TextOutput), "Default fallback should be TextOutput"


def test_tool_call_output_display_renders_yellow_line():
    console = Console(record=True)
    tco = ToolCallOutput("grep", {"pattern": "foo", "path": "/tmp"})
    tco.display(console)
    text = console.export_text()
    assert "⏺ grep(" in text, "Should print tool call line with circle symbol and function format"
    assert "pattern" in text and "path" in text, "Should include parameter names"
    assert "…" in text, "Should include ellipsis symbol to indicate running/action in progress"


def test_text_output_display_multiline_truncation():
    console = Console(record=True)
    data = "first line\nsecond line\nthird line\nfourth line\nfifth line"
    to = TextOutput(data)
    to.display(console)
    text = console.export_text()
    assert "first line" in text, "Should display first line"
    assert "... (+" in text and "lines)" in text, "Should show truncation message when exceeding three lines"


def test_text_output_display_single_line():
    console = Console(record=True)
    data = "single line"
    to = TextOutput(data)
    to.display(console)
    text = console.export_text()
    assert "single line" in text, "Single line text should be displayed completely"


def test_assistant_output_display_plain_white():
    console = Console(record=True)
    msg = "final answer"
    ao = AssistantOutput(msg)
    ao.display(console)
    text = console.export_text()
    assert "final answer" in text, "Assistant output should display final answer text"


def test_code_output_display_short_code_no_summary():
    console = Console(record=True)
    code = "import os\nprint('x')"
    co = CodeOutput(code, language="python")
    co.display(console)
    text = console.export_text()
    assert "import os" in text and "print('x')" in text, "Short code should be displayed completely with syntax highlighting"


def test_code_output_display_long_code_shows_preview_and_summary():
    console = Console(record=True)
    long_code = "\n".join([f"line {i}" for i in range(20)])
    co = CodeOutput(long_code, language="python")
    co.display(console)
    text = console.export_text()
    assert "line 0" in text and "line 1" in text and "line 2" in text, "Long code should show preview of first few lines"
    assert "more lines" in text, "Long code should display summary of remaining lines"


def test_file_list_output_display_table_and_summary():
    console = Console(record=True)
    files = [{"name": f"file_{i}.txt", "is_dir": False, "size": 1234} for i in range(15)]
    flo = FileListOutput(files, path="/tmp")
    flo.display(console)
    text = console.export_text()
    assert "Directory: /tmp" in text, "Should display directory path"
    assert "file_0.txt" in text and "file_9.txt" in text, "Should display first 10 items"
    assert "more items" in text, "Should show summary of remaining items when exceeding display limit"