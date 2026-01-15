import pytest
from rich.console import Console
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from smolcc.agent import ToolAgent


class DummyLogger:
    def __init__(self, console: Console):
        self.console = console
        self.errors = []

    def log_error(self, msg: str):
        # 模拟 RichConsoleLogger 的错误打印与记录
        self.errors.append(msg)
        self.console.print(f"[bold red]Error:[/bold red] {msg}")


def test_execute_tool_call_normal_displays_output_and_returns_raw_result():
    console = Console(record=True)

    # 构造不调用 __init__ 的 ToolAgent，手动填充必要属性
    agent = object.__new__(ToolAgent)
    agent.console = console
    agent.logger = DummyLogger(console)
    # 提供一个简单工具
    agent.tools = {
        "echo": lambda text: text
    }

    res = agent.execute_tool_call("echo", {"text": "hello"})
    assert res == "hello", "应返回工具执行的原始结果"

    out = console.export_text()
    # 工具调用行
    assert "⏺ echo(" in out and "text" in out, "应打印工具调用及参数"
    # 非 final_answer 情况应显示输出内容
    assert "hello" in out, "应显示工具输出的文本"


def test_execute_tool_call_final_answer_skips_output_display():
    console = Console(record=True)

    agent = object.__new__(ToolAgent)
    agent.console = console
    agent.logger = DummyLogger(console)
    # final_answer 工具存在并返回文本
    agent.tools = {
        "final_answer": lambda text: "bye"
    }

    res = agent.execute_tool_call("final_answer", {"text": "ignored"})
    assert res == "bye", "应返回原始结果"
    out = console.export_text()
    # 应打印工具调用，但不显示结果内容
    assert "⏺ final_answer(" in out, "应显示工具调用提示"
    assert "bye" not in out, "final_answer 的输出不应在此处显示"


def test_execute_tool_call_missing_tool_logs_error_and_returns_error_string():
    console = Console(record=True)

    agent = object.__new__(ToolAgent)
    agent.console = console
    agent.logger = DummyLogger(console)
    agent.tools = {}  # 不提供工具，触发错误路径

    res = agent.execute_tool_call("missing_tool", {"x": 1})
    assert isinstance(res, str) and res.startswith("Error:"), "找不到工具时应返回错误字符串"

    out = console.export_text()
    # 应打印工具调用行
    assert "⏺ missing_tool(" in out, "应显示工具调用提示"
    # 应打印错误信息
    assert "Error:" in out, "应通过 Rich 显示错误信息"
    # 记录器也应捕获错误
    assert agent.logger.errors, "记录器应收集错误信息"