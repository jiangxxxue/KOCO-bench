#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import subprocess
import shlex
from pathlib import Path
import io

# 统一标准输出/错误输出为 UTF-8，确保父进程与子进程（pytest 子进程）均使用 UTF-8
try:
    # 包装当前进程 stdout/stderr
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    # 让子进程 Python 也使用 UTF-8
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
except Exception:
    # 避免因终端特性差异导致测试中断
    pass


def main():
    # 当前脚本所在目录即 tests 目录
    tests_dir = Path(__file__).resolve().parent
    repo_root = Path(".").resolve()

    if not tests_dir.exists():
        print(f"[ERROR] 测试目录不存在: {tests_dir}")
        sys.exit(1)

    # 构造 pytest 命令
    filters = [
        # 已安装 pytest-asyncio 后通常不再需要过滤此项，如仍出现可取消注释:
        # "-W", "ignore::pytest.PytestConfigWarning",

        # 过滤 Pydantic v2 的弃用提示 (代码迁移需另行进行)
        "-W", "ignore:.*PydanticDeprecatedSince20.*:DeprecationWarning",
        "-W", "ignore:.*json_encoders is deprecated.*:DeprecationWarning",
    ]

    cmd = ["python", "-m", "pytest", str(tests_dir), "-q"] + filters
    print(f"[INFO] 运行测试命令: {' '.join(shlex.quote(c) for c in cmd)}")
    proc = subprocess.run(cmd, cwd=str(repo_root))
    exit_code = proc.returncode

    if exit_code != 0:
        print(f"[ERROR] 测试失败，退出码: {exit_code}")
    else:
        print("[INFO] 测试全部通过")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()