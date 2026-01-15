import pytest
from rich.console import Console
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from smolcc.agent import ToolAgent, RichConsoleLogger
# 为避免依赖外部 smolagents 包，动态获取父类引用（ToolAgent 的直接基类）
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

    # 构造不调用 __init__ 的 ToolAgent 并注入 console
    agent = object.__new__(ToolAgent)
    agent.console = console

    # 记录 super 调用次数并返回预期消息
    call_count = {"n": 0}
    expected = [{"role": "user", "content": "hello"}]

    def fake_super_format(self, prompt):
        call_count["n"] += 1
        return expected

    monkeypatch.setattr(ParentClass, "format_messages_for_llm", fake_super_format, raising=False)

    messages = agent.format_messages_for_llm("hello")
    assert messages == expected, "应返回父类格式化后的消息"
    assert call_count["n"] == 1, "应调用一次父类方法以完成消息格式化"

    # 可选校验：进度指示文本（由 Rich Progress 产生，transient=True 情况下可能不保留）
    # text = console.export_text()
    # assert "Thinking..." in text


def test_run_displays_assistant_output_and_returns_result(monkeypatch):
    console = Console(record=True)
    agent = object.__new__(ToolAgent)
    agent.console = console
    agent.logger = DummyLogger(console)

    # 让父类 run 返回固定结果
    monkeypatch.setattr(ParentClass, "run", lambda self, user_input, stream=False: "final result")

    res = agent.run("query")
    assert res == "final result", "run 应返回父类结果"
    out = console.export_text()
    assert "final result" in out, "应以助手输出展示最终结果文本"


def test_rich_console_logger_log_error_writes_file_and_prints(tmp_path):
    log_file = tmp_path / "log.txt"
    logger = RichConsoleLogger(log_file=str(log_file))
    # 重置 console 为可记录
    logger.console = Console(record=True)

    logger.log_error("oops")
    out = logger.console.export_text()
    assert "Error: oops" in out, "控制台应突出显示错误信息"

    content = log_file.read_text()
    assert "ERROR: oops" in content, "应写入日志文件以便后续追踪"