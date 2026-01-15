import os
import pytest  # type: ignore
from types import SimpleNamespace
import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

# Provide friendly hints for editor static checking; at runtime, tests/conftest.py dynamically injects replica package path
from smolcc.system_prompt import (  # type: ignore
    get_directory_structure,
    is_git_repo,
    get_git_status,
    get_system_prompt,
)
import smolcc.system_prompt as sp  # type: ignore


def test_get_directory_structure_ignores_patterns(tmp_path):
    # Create directory structure: including visible directory, node_modules, hidden directory and some files
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

    # Top level should display starting directory line
    assert result.startswith(f"- {tmp_path.resolve()}/"), "Should contain starting directory line"
    # Should include visible directory
    assert "- visible/" in result, "Should list non-ignored visible directory"
    # Should ignore hidden directory and node_modules
    assert "node_modules" not in result, "Should ignore node_modules"
    assert ".hidden" not in result, "Should ignore hidden directories starting with ."


def test_is_git_repo_true_and_false(monkeypatch, tmp_path):
    def fake_run(args, capture_output=True, text=True, check=False):
        # Default return non-repository
        r = SimpleNamespace(returncode=1, stdout="false\n")
        # Return different results for rev-parse --is-inside-work-tree
        if isinstance(args, list) and "rev-parse" in args:
            # Simulate as git repository when path basename is "repo"
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
    assert is_git_repo(str(repo_dir)) is True, "Should recognize as git repository"
    assert is_git_repo(str(non_repo_dir)) is False, "Should recognize as non-git repository"


def test_get_git_status_composes_text(monkeypatch, tmp_path):
    def fake_run(args, capture_output=True, text=True, check=False):
        # Default empty output
        r = SimpleNamespace(returncode=0, stdout="")
        # Current branch
        if isinstance(args, list) and "rev-parse" in args and "HEAD" in args:
            r.stdout = "feature\n"
            return r
        # Remote main branch
        if isinstance(args, list) and "remote" in args and "show" in args and "origin" in args:
            r.stdout = "Remote branches:\n  feature tracked\n  HEAD branch: main\n"
            return r
        # Status
        if isinstance(args, list) and "status" in args and "--porcelain" in args:
            r.stdout = ""  # Clean working directory
            return r
        # Recent commits
        if isinstance(args, list) and "log" in args and "--oneline" in args:
            r.stdout = "abc123 fix bug\n def456 add feature\n"
            return r
        return r

    monkeypatch.setattr(sp.subprocess, "run", fake_run)

    status = get_git_status(str(tmp_path))
    assert "Current branch: feature" in status, "Should contain current branch"
    assert "Main branch (you will usually use this for PRs): main" in status, "Should contain main branch"
    assert "(clean)" in status, "Clean working directory should display (clean)"
    assert "Recent commits:" in status and "abc123" in status, "Should contain recent commit summary"


def test_get_system_prompt_fills_placeholders_and_appends_git_status(monkeypatch, tmp_path):
    # Fake date to avoid platform compatibility issues with %-m/%-d/%Y
    class FakeDateObj:
        def strftime(self, fmt: str) -> str:
            return "1/2/2025"

    class FakeDatetimeClass:
        @staticmethod
        def now():
            return FakeDateObj()
    
    # System implementation uses datetime.datetime.now(), so need to override its datetime class in module sp.datetime
    monkeypatch.setattr(sp.datetime, "datetime", FakeDatetimeClass)
    # Force recognition as git repository
    monkeypatch.setattr(sp, "is_git_repo", lambda cwd: True)
    # Control directory structure output
    monkeypatch.setattr(sp, "get_directory_structure", lambda cwd: "STRUCTURE")
    # Control git status output
    monkeypatch.setattr(sp, "get_git_status", lambda cwd: "GIT_STATUS")

    # Generate system prompt
    prompt = get_system_prompt(str(tmp_path))

    # Working directory, platform, and model placeholders should be replaced
    assert str(tmp_path) in prompt, "Should contain working directory"
    assert "{working_directory}" not in prompt, "Placeholder should be replaced"
    assert "{directory_structure}" not in prompt and "STRUCTURE" in prompt, "Directory structure placeholder should be replaced with mock value"
    assert "claude-3-7-sonnet-20250219" in prompt, "Should contain model name"
    # Because we mock is_git_repo=True, should append gitStatus context
    assert '<context name="gitStatus">GIT_STATUS</context>' in prompt, "Should append Git status context"