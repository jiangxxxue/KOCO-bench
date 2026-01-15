import os
import sys
import pytest
from types import SimpleNamespace


class SummaryPlugin:
    def __init__(self):
        self.collected = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.xfailed = 0
        self.xpassed = 0

    def pytest_collection_finish(self, session):
        # 收集完成后统计用例总数
        self.collected = len(session.items)

    def pytest_runtest_logreport(self, report):
        # 统计每个测试的执行结果
        outcome = report.outcome
        # 仅统计一次调用阶段的结果；skip 在 setup/call 均可能出现
        if outcome == "passed" and report.when == "call":
            if getattr(report, "wasxfail", False):
                self.xpassed += 1
            else:
                self.passed += 1
        elif outcome == "failed":
            # 失败如果是 xfail 则计入 xfailed 而非失败
            if getattr(report, "wasxfail", False):
                self.xfailed += 1
            else:
                # setup/call/teardown 的失败都计入失败
                self.failed += 1
        elif outcome == "skipped":
            # 任意阶段的跳过均计入 skipped
            self.skipped += 1

    def pytest_sessionfinish(self, session, exitstatus):
        # 会话结束时打印汇总
        total = self.collected
        # 通过数包含正常通过 + xpassed（意外通过）
        total_passed = self.passed + self.xpassed
        total_failed = self.failed  # xfailed（预期失败）不计入失败
        total_skipped = self.skipped
        pass_rate = (total_passed / total * 100.0) if total > 0 else 0.0

        print("\n================= 测试汇总 =================")
        print(f"总用例数: {total}")
        print(f"通过: {total_passed}（正常通过: {self.passed}, 意外通过: {self.xpassed}）")
        print(f"失败: {total_failed}")
        print(f"跳过: {total_skipped}")
        print(f"预期失败(xfailed): {self.xfailed}")
        print(f"通过率: {pass_rate:.2f}%")
        print("===========================================")


def main() -> None:
    # 测试目录（当前文件所在目录）
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    # 项目根目录（测试目录的上一级）
    project_root = os.path.dirname(tests_dir)
    
    # 切换到项目根目录，确保相对导入正确工作
    os.chdir(project_root)
    
    # 确保项目根目录在 Python 路径中
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print(f"Running tests from: {project_root}")
    print(f"Test directory: {tests_dir}")
    print(f"Python path includes: {project_root}")

    # 确保在指定目录执行，不使用 cd，直接传入绝对路径
    args = [tests_dir]

    # 启用更详细的结果输出（不使用 -q）
    # 可附加 -rA 显示所有测试结果的简要概要
    args += ["-rA"]

    plugin = SummaryPlugin()
    exit_code = pytest.main(args=args, plugins=[plugin])

    # 以 pytest 的退出码作为进程退出码
    sys.exit(exit_code)


if __name__ == "__main__":
    main()