import pytest
from rich.console import Console
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
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
    assert isinstance(out, CodeOutput), "字符串包含代码结构时应识别为 CodeOutput"


def test_convert_string_plain_text_output():
    text_str = "hello world"
    out = convert_to_tool_output(text_str)
    assert isinstance(out, TextOutput), "普通字符串应转换为 TextOutput"


def test_convert_list_of_lists_table_output():
    table_data = [[1, 2], [3, 4]]
    out = convert_to_tool_output(table_data)
    assert isinstance(out, TableOutput), "列表的列表应转换为 TableOutput"


def test_convert_list_of_dicts_file_list_output():
    files = [{"name": "file.txt", "is_dir": False, "size": 100}]
    out = convert_to_tool_output(files)
    assert isinstance(out, FileListOutput), "包含 name 的字典列表应转换为 FileListOutput"


def test_convert_existing_tool_output_passthrough():
    existing = TextOutput("existing")
    out = convert_to_tool_output(existing)
    assert out is existing, "已有 ToolOutput 应直接返回自身"


def test_convert_default_text_output_for_other_types():
    other = {"a": 1}
    out = convert_to_tool_output(other)
    assert isinstance(out, TextOutput), "默认兜底应为 TextOutput"


def test_tool_call_output_display_renders_yellow_line():
    console = Console(record=True)
    tco = ToolCallOutput("grep", {"pattern": "foo", "path": "/tmp"})
    tco.display(console)
    text = console.export_text()
    assert "⏺ grep(" in text, "应打印工具调用行，含圆点符号与函数格式"
    assert "pattern" in text and "path" in text, "应包含参数名称"
    assert "…" in text, "应含省略号符号表现运行中/动作提示"


def test_text_output_display_multiline_truncation():
    console = Console(record=True)
    data = "first line\nsecond line\nthird line\nfourth line\nfifth line"
    to = TextOutput(data)
    to.display(console)
    text = console.export_text()
    assert "first line" in text, "应显示首行"
    assert "... (+" in text and "lines)" in text, "超过三行应显示截断提示"


def test_text_output_display_single_line():
    console = Console(record=True)
    data = "single line"
    to = TextOutput(data)
    to.display(console)
    text = console.export_text()
    assert "single line" in text, "单行文本应完整显示"


def test_assistant_output_display_plain_white():
    console = Console(record=True)
    msg = "final answer"
    ao = AssistantOutput(msg)
    ao.display(console)
    text = console.export_text()
    assert "final answer" in text, "助手输出应显示最终答案文本"


def test_code_output_display_short_code_no_summary():
    console = Console(record=True)
    code = "import os\nprint('x')"
    co = CodeOutput(code, language="python")
    co.display(console)
    text = console.export_text()
    assert "import os" in text and "print('x')" in text, "短代码应完整展示并高亮"


def test_code_output_display_long_code_shows_preview_and_summary():
    console = Console(record=True)
    long_code = "\n".join([f"line {i}" for i in range(20)])
    co = CodeOutput(long_code, language="python")
    co.display(console)
    text = console.export_text()
    assert "line 0" in text and "line 1" in text and "line 2" in text, "长代码应展示前几行预览"
    assert "more lines" in text, "长代码应显示剩余行数的摘要说明"


def test_file_list_output_display_table_and_summary():
    console = Console(record=True)
    files = [{"name": f"file_{i}.txt", "is_dir": False, "size": 1234} for i in range(15)]
    flo = FileListOutput(files, path="/tmp")
    flo.display(console)
    text = console.export_text()
    assert "Directory: /tmp" in text, "应显示目录路径"
    assert "file_0.txt" in text and "file_9.txt" in text, "应显示前10项"
    assert "more items" in text, "超过显示限制应有剩余项摘要"