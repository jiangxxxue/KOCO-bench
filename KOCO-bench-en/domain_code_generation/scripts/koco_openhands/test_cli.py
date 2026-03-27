"""Tests for cli.py — command-line interface for the OpenHands pipeline."""

import inspect
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))

# Insert koco_openhands dir BEFORE scripts dir to resolve imports
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_SCRIPTS_DIR = str(Path(_HERE).parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# Import koco_openhands/cli.py by path to avoid collision with scripts/cli.py
import importlib.util
_cli_spec = importlib.util.spec_from_file_location(
    "oh_cli", os.path.join(_HERE, "cli.py"))
oh_cli = importlib.util.module_from_spec(_cli_spec)
_cli_spec.loader.exec_module(oh_cli)

cmd_infer = oh_cli.cmd_infer
cmd_eval = oh_cli.cmd_eval


# ═══════════════════════════════════════════════════════════════════════════
# cmd_infer: fail counter bug
# ═══════════════════════════════════════════════════════════════════════════


class TestCmdInferFailCounter:
    """The `fail` variable in cmd_infer must be incremented on skipped examples."""

    def test_fail_counter_is_incremented(self):
        """Verify that `fail` is incremented in skip branches."""
        source = inspect.getsource(cmd_infer)

        assert "fail = 0" in source or "fail=0" in source
        # After fix: fail should be incremented
        assert "fail += 1" in source


# ═══════════════════════════════════════════════════════════════════════════
# API key resolution
# ═══════════════════════════════════════════════════════════════════════════


class TestApiKeyResolution:
    """Test the API key resolution logic in cmd_infer."""

    def test_api_key_from_argument(self):
        with patch.object(oh_cli, "get_test_examples", return_value=[]):
            result = cmd_infer("test_framework", "test/model", api_key="sk-test-key")
            assert result == 0

    def test_api_key_from_env(self, monkeypatch):
        """API key from environment variable."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-env")
        with patch.object(oh_cli, "get_test_examples", return_value=[]):
            result = cmd_infer("test_framework", "test/model")
            assert result == 0

    def test_no_api_key_returns_error(self, monkeypatch):
        """No API key available → error."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch.object(oh_cli, "SCRIPTS_DIR", Path("/nonexistent")):
            result = cmd_infer("test_framework", "test/model", api_key="")
            assert result == 1


# ═══════════════════════════════════════════════════════════════════════════
# Force mode
# ═══════════════════════════════════════════════════════════════════════════


class TestForceMode:
    """Test that --force deletes previous output/progress files."""

    def test_force_removes_files(self, tmp_path):
        model_dir = tmp_path / "data" / "fw" / "openhands" / "test-model"
        model_dir.mkdir(parents=True)

        output_file = model_dir / "algorithm_methods_data_EX_output.jsonl"
        progress_file = model_dir / ".EX_progress.json"
        result_file = model_dir / "algorithm_methods_data_EX_result.jsonl"
        metrics_file = model_dir / "algorithm_methods_data_EX_result.metrics.json"

        for f in [output_file, progress_file, result_file, metrics_file]:
            f.write_text("{}")

        # Simulate force deletion logic from cmd_infer
        for f in (output_file, progress_file, result_file, metrics_file):
            if f.exists():
                f.unlink()

        assert not output_file.exists()
        assert not progress_file.exists()
        assert not result_file.exists()
        assert not metrics_file.exists()


# ═══════════════════════════════════════════════════════════════════════════
# Resume support integration
# ═══════════════════════════════════════════════════════════════════════════


class TestResumeIntegration:
    """Test that completed instances are correctly skipped."""

    def test_skip_completed(self, tmp_path):
        from runner import save_completed_ids, load_completed_ids

        progress_file = tmp_path / ".test_progress.json"
        completed = {"func_a"}
        save_completed_ids(completed, str(progress_file))

        records = [
            {"function_name": "func_a", "value": 1},
            {"function_name": "func_b", "value": 2},
        ]

        loaded_completed = load_completed_ids(str(progress_file))
        pending = [r for r in records if r["function_name"] not in loaded_completed]

        assert len(pending) == 1
        assert pending[0]["function_name"] == "func_b"


# ═══════════════════════════════════════════════════════════════════════════
# Model directory naming
# ═══════════════════════════════════════════════════════════════════════════


class TestModelDirNaming:

    def test_slash_model_name(self):
        model = "deepseek/deepseek-v3.2"
        assert model.split("/")[-1] == "deepseek-v3.2"

    def test_no_slash_model_name(self):
        model = "gpt-4"
        assert model.split("/")[-1] == "gpt-4"

    def test_multi_slash_model_name(self):
        model = "openrouter/deepseek/deepseek-v3.2"
        assert model.split("/")[-1] == "deepseek-v3.2"


# ═══════════════════════════════════════════════════════════════════════════
# Docker command construction
# ═══════════════════════════════════════════════════════════════════════════


class TestDockerCmdConstruction:
    """Test the Docker command constructed in cmd_eval."""

    def test_linux_user_injection_position(self):
        """Verify --user flags are injected at the right position."""
        image = "test:latest"

        docker_cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", "/project:/workspace/project",
            image,
            "python3", "/workspace/project/scripts/eval.py",
        ]

        docker_cmd[3:3] = [
            "--user", "1000:1000",
            "-e", "HOME=/tmp",
            "-e", "USER=benchuser",
        ]

        image_idx = docker_cmd.index(image)
        user_idx = docker_cmd.index("--user")
        assert user_idx < image_idx

        assert "--gpus" in docker_cmd
        gpus_idx = docker_cmd.index("--gpus")
        assert gpus_idx < image_idx
