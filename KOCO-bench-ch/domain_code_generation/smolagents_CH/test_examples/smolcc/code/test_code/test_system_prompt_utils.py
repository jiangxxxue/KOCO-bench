import os
import pytest  # type: ignore
from types import SimpleNamespace
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

# 为编辑器静态检查提供友好提示；运行期由 tests/conftest.py 动态注入副本包路径
from smolcc.system_prompt import (  # type: ignore
    get_directory_structure,
    is_git_repo,
    get_git_status,
    get_system_prompt,
)
import smolcc.system_prompt as sp  # type: ignore


def test_get_directory_structure_ignores_patterns(tmp_path):
    # 创建目录结构：包括可见目录、node_modules、隐藏目录与一些文件
    visible = tmp_path / "visible"
    visible.mkdir()
    (visible / "a.txt").write_text("content")

    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "lib.js").write_text("// lib")

    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "secret.txt").write_text("secret")

    result = get_directory_structure(str(tmp_path))

    # 顶层应显示起始目录行
    assert result.startswith(f"- {tmp_path.resolve()}/"), "应含起始目录行"
    # 应包含可见目录
    assert "- visible/" in result, "应列出未忽略的可见目录"
    # 应忽略隐藏目录与 node_modules
    assert "node_modules" not in result, "应忽略 node_modules"
    assert ".hidden" not in result, "应忽略以 . 开头的隐藏目录"


def test_is_git_repo_true_and_false(monkeypatch, tmp_path):
    def fake_run(args, capture_output=True, text=True, check=False):
        # 默认返回非仓库
        r = SimpleNamespace(returncode=1, stdout="false\n")
        # 针对 rev-parse --is-inside-work-tree 返回不同结果
        if isinstance(args, list) and "rev-parse" in args:
            # 当路径 basename 为 "repo" 时模拟为 git 仓库
            if "-C" in args:
                idx = args.index("-C")
                path = args[idx + 1]
                if os.path.basename(path) == "repo":
                    r.returncode = 0
                    r.stdout = "true\n"
                    return r
        return r

    monkeypatch.setattr(sp.subprocess, "run", fake_run)

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    non_repo_dir = tmp_path / "nonrepo"
    non_repo_dir.mkdir()
    assert is_git_repo(str(repo_dir)) is True, "应识别为 git 仓库"
    assert is_git_repo(str(non_repo_dir)) is False, "应识别为非 git 仓库"


def test_get_git_status_composes_text(monkeypatch, tmp_path):
    def fake_run(args, capture_output=True, text=True, check=False):
        # 默认空输出
        r = SimpleNamespace(returncode=0, stdout="")
        # 当前分支
        if isinstance(args, list) and "rev-parse" in args and "HEAD" in args:
            r.stdout = "feature\n"
            return r
        # 远程主分支
        if isinstance(args, list) and "remote" in args and "show" in args and "origin" in args:
            r.stdout = "Remote branches:\n  feature tracked\n  HEAD branch: main\n"
            return r
        # 状态
        if isinstance(args, list) and "status" in args and "--porcelain" in args:
            r.stdout = ""  # 干净工作区
            return r
        # 最近提交
        if isinstance(args, list) and "log" in args and "--oneline" in args:
            r.stdout = "abc123 fix bug\n def456 add feature\n"
            return r
        return r

    monkeypatch.setattr(sp.subprocess, "run", fake_run)

    status = get_git_status(str(tmp_path))
    assert "Current branch: feature" in status, "应包含当前分支"
    assert "Main branch (you will usually use this for PRs): main" in status, "应包含主分支"
    assert "(clean)" in status, "干净工作区应显示 (clean)"
    assert "Recent commits:" in status and "abc123" in status, "应包含最近提交摘要"


def test_get_system_prompt_fills_placeholders_and_appends_git_status(monkeypatch, tmp_path):
    # 伪造日期，避免平台对 %-m/%-d/%Y 的兼容问题
    class FakeDateObj:
        def strftime(self, fmt: str) -> str:
            return "1/2/2025"

    class FakeDatetimeClass:
        @staticmethod
        def now():
            return FakeDateObj()
    
    # 系统实现使用 datetime.datetime.now()，因此需在模块 sp.datetime 中覆盖其 datetime 类
    monkeypatch.setattr(sp.datetime, "datetime", FakeDatetimeClass)
    # 强制识别为 git 仓库
    monkeypatch.setattr(sp, "is_git_repo", lambda cwd: True)
    # 控制目录结构输出
    monkeypatch.setattr(sp, "get_directory_structure", lambda cwd: "STRUCTURE")
    # 控制 git 状态输出
    monkeypatch.setattr(sp, "get_git_status", lambda cwd: "GIT_STATUS")

    # 生成系统提示
    prompt = get_system_prompt(str(tmp_path))

    # 工作目录与平台、模型占位符应替换
    assert str(tmp_path) in prompt, "应包含工作目录"
    assert "{working_directory}" not in prompt, "占位符应被替换"
    assert "{directory_structure}" not in prompt and "STRUCTURE" in prompt, "目录结构占位符应被替换为模拟值"
    assert "claude-3-7-sonnet-20250219" in prompt, "应包含模型名"
    # 因为我们模拟 is_git_repo=True，应追加 gitStatus context
    assert '<context name="gitStatus">GIT_STATUS</context>' in prompt, "应附加 Git 状态上下文"