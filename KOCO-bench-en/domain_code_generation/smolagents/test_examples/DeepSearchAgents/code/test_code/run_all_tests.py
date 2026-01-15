#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import subprocess
import shlex
from pathlib import Path
import io

# Unify stdout/stderr to UTF-8, ensuring both parent and child processes (pytest subprocess) use UTF-8
try:
    # Wrap current process stdout/stderr
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    # Make child Python processes also use UTF-8
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
except Exception:
    # Avoid test interruption due to terminal characteristic differences
    pass


def main():
    # Current script directory is the tests directory
    tests_dir = Path(__file__).resolve().parent
    repo_root = Path(".").resolve()

    if not tests_dir.exists():
        print(f"[ERROR] Test directory does not exist: {tests_dir}")
        sys.exit(1)

    # Construct pytest command
    filters = [
        # After installing pytest-asyncio, this filter is usually no longer needed; uncomment if still appears:
        # "-W", "ignore::pytest.PytestConfigWarning",

        # Filter Pydantic v2 deprecation warnings (code migration needs to be done separately)
        "-W", "ignore:.*PydanticDeprecatedSince20.*:DeprecationWarning",
        "-W", "ignore:.*json_encoders is deprecated.*:DeprecationWarning",
    ]

    cmd = ["python", "-m", "pytest", str(tests_dir), "-q"] + filters
    print(f"[INFO] Running test command: {' '.join(shlex.quote(c) for c in cmd)}")
    proc = subprocess.run(cmd, cwd=str(repo_root))
    exit_code = proc.returncode

    if exit_code != 0:
        print(f"[ERROR] Tests failed, exit code: {exit_code}")
    else:
        print("[INFO] All tests passed")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()