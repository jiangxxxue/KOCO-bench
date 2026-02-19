"""
Cross-platform configuration module for KOCO-bench CLI.
Replaces the shell-based common.sh with pure Python path resolution and .env loading.
Works on Windows, Linux, and macOS without any platform-specific workarounds.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Core paths (platform-native via pathlib)
# ---------------------------------------------------------------------------
SCRIPTS_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SCRIPTS_DIR.parent  # domain_code_generation/

# ---------------------------------------------------------------------------
# Load .env file
# Uses python-dotenv if available; falls back to a minimal built-in parser
# that correctly handles CRLF line endings (a common Windows pitfall).
# ---------------------------------------------------------------------------
_ENV_FILE = SCRIPTS_DIR / ".env"


def _load_env_builtin(env_path: Path) -> None:
    """Minimal .env loader that is immune to CRLF issues."""
    if not env_path.is_file():
        return
    with open(env_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()  # strips \r\n, \n, and surrounding whitespace
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes (single or double)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            # Only set if not already in environment (env vars take precedence)
            if key not in os.environ:
                os.environ[key] = value


try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE)
except ImportError:
    _load_env_builtin(_ENV_FILE)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_test_examples(framework: str) -> list:
    """Return sorted list of test example names for a framework.

    Replaces: ls -d "$TEST_EXAMPLES_DIR"/*/ | xargs -n 1 basename
    """
    examples_dir = PROJECT_ROOT / framework / "test_examples"
    if not examples_dir.is_dir():
        raise FileNotFoundError(f"Framework test_examples directory not found: {examples_dir}")
    return sorted(d.name for d in examples_dir.iterdir() if d.is_dir())


def get_data_files(framework: str) -> list:
    """Return sorted list of algorithm_methods_data JSONL files (excluding _output / _result).

    Replaces: ls "$DATA_DIR"/algorithm_methods_data_*.jsonl | grep -v output | grep -v result
    """
    data_dir = SCRIPTS_DIR / "data" / framework
    if not data_dir.is_dir():
        return []
    return sorted(
        f for f in data_dir.glob("algorithm_methods_data_*.jsonl")
        if "_output" not in f.name and "_result" not in f.name
    )


def get_metadata_path(framework: str) -> Path:
    """Return the path to a framework's metadata.json."""
    return PROJECT_ROOT / framework / "knowledge_corpus" / "metadata.json"


# ---------------------------------------------------------------------------
# Docker image mapping
# ---------------------------------------------------------------------------
DOCKER_IMAGES = {
    "verl": "kocobench/verl-openr1:v0.4",
    "open-r1": "kocobench/verl-openr1:v0.4",
    "raganything": "raganything-smolagents:test",
    "smolagents": "raganything-smolagents:test",
    "tensorrt_model_optimizer": "tensorrt:latest",
}


def get_docker_image(framework: str) -> str:
    """Return the Docker image name for a given framework.

    Replaces the case statement in run_execution_evaluation_pure.sh.
    """
    image = DOCKER_IMAGES.get(framework, "")
    if not image:
        raise ValueError(f"Unknown framework '{framework}' — cannot select Docker image")
    return image
