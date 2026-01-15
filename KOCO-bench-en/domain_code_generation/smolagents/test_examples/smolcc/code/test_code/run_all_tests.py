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
        # Count total test cases after collection completes
        self.collected = len(session.items)

    def pytest_runtest_logreport(self, report):
        # Count execution results for each test
        outcome = report.outcome
        # Only count results from call phase once; skip may appear in setup/call
        if outcome == "passed" and report.when == "call":
            if getattr(report, "wasxfail", False):
                self.xpassed += 1
            else:
                self.passed += 1
        elif outcome == "failed":
            # If failure is xfail, count as xfailed not failed
            if getattr(report, "wasxfail", False):
                self.xfailed += 1
            else:
                # Failures in setup/call/teardown all count as failed
                self.failed += 1
        elif outcome == "skipped":
            # Skips in any phase count as skipped
            self.skipped += 1

    def pytest_sessionfinish(self, session, exitstatus):
        # Print summary when session ends
        total = self.collected
        # Passed count includes normal passed + xpassed (unexpectedly passed)
        total_passed = self.passed + self.xpassed
        total_failed = self.failed  # xfailed (expected failures) not counted as failed
        total_skipped = self.skipped
        pass_rate = (total_passed / total * 100.0) if total > 0 else 0.0

        print("\n================= Test Summary =================")
        print(f"Total test cases: {total}")
        print(f"Passed: {total_passed} (Normal: {self.passed}, Unexpected: {self.xpassed})")
        print(f"Failed: {total_failed}")
        print(f"Skipped: {total_skipped}")
        print(f"Expected failures (xfailed): {self.xfailed}")
        print(f"Pass rate: {pass_rate:.2f}%")
        print("===============================================")


def main() -> None:
    # Test directory (current file's directory)
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    # Project root directory (parent of test directory)
    project_root = os.path.dirname(tests_dir)
    
    # Change to project root directory to ensure relative imports work correctly
    os.chdir(project_root)
    
    # Ensure project root directory is in Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print(f"Running tests from: {project_root}")
    print(f"Test directory: {tests_dir}")
    print(f"Python path includes: {project_root}")

    # Execute in specified directory, don't use cd, pass absolute path directly
    args = [tests_dir]

    # Enable more verbose result output (don't use -q)
    # Can append -rA to show brief summary of all test results
    args += ["-rA"]

    plugin = SummaryPlugin()
    exit_code = pytest.main(args=args, plugins=[plugin])

    # Use pytest's exit code as process exit code
    sys.exit(exit_code)


if __name__ == "__main__":
    main()