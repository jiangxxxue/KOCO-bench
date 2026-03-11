#!/usr/bin/env python3
"""
DeepSearchAgents Test Runner Script (with statistics)
Runs all test files and generates statistical reports using pytest
"""

import sys
import os
import time

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import pytest
except ImportError:
    print("pytest is required. Install with: pip install pytest")
    sys.exit(1)


class StatsPlugin:
    """Pytest plugin to collect per-file pass/fail/error/skip counts."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = 0
        self.skipped = 0

    def pytest_runtest_logreport(self, report):
        if report.when == 'call':
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
        elif report.when == 'setup' and report.failed:
            self.errors += 1
        if report.skipped:
            self.skipped += 1

    @property
    def total(self):
        return self.passed + self.failed + self.errors + self.skipped

    @property
    def success(self):
        return self.failed == 0 and self.errors == 0


def run_test_file(test_file, description):
    """Run a single test file and return results"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"File: {test_file}")
    print(f"{'='*60}")

    plugin = StatsPlugin()
    start_time = time.time()

    exit_code = pytest.main([test_file, "-v", "--tb=short", "-q"], plugins=[plugin])

    end_time = time.time()

    return {
        'module': description,
        'total': plugin.total,
        'passed': plugin.passed,
        'failed': plugin.failed,
        'errors': plugin.errors,
        'skipped': plugin.skipped,
        'time': end_time - start_time,
        'success': plugin.success and exit_code == 0 and plugin.total > 0
    }


def print_summary(all_stats):
    """Print test summary"""
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}\n")

    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_errors = 0
    total_skipped = 0
    total_time = 0

    print(f"{'Module':<40} {'Total':<8} {'Passed':<8} {'Failed':<8} {'Time(s)':<10}")
    print(f"{'-'*74}")

    for stats in all_stats:
        print(f"{stats['module']:<40} {stats['total']:<8} {stats['passed']:<8} "
              f"{stats['failed']:<8} {stats['time']:<10.2f}")

        total_tests += stats['total']
        total_passed += stats['passed']
        total_failed += stats['failed']
        total_errors += stats['errors']
        total_skipped += stats['skipped']
        total_time += stats['time']

    print(f"{'-'*74}")
    print(f"{'TOTAL':<40} {total_tests:<8} {total_passed:<8} {total_failed:<8} {total_time:<10.2f}")
    print()

    # Print detailed statistics
    print(f"Total Tests Run: {total_tests}")
    print(f"Passed: {total_passed} ({100*total_passed/total_tests if total_tests > 0 else 0:.1f}%)")
    print(f"Failed: {total_failed}")
    print(f"Errors: {total_errors}")
    print(f"Skipped: {total_skipped}")
    print(f"Total Time: {total_time:.2f}s")
    print()

    # Determine overall result
    all_success = all(stats['success'] for stats in all_stats)
    if all_success:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
        failed_modules = [stats['module'] for stats in all_stats if not stats['success']]
        print(f"Failed modules: {', '.join(failed_modules)}")

    print(f"{'='*60}\n")

    return all_success


def main():
    """Main function"""
    print("=" * 60)
    print("DeepSearchAgents Core Function Tests")
    print("=" * 60)

    # Get the directory containing this script
    test_dir = os.path.dirname(os.path.abspath(__file__))

    # Define test files (description, filename)
    test_files = [
        ('agent_step_callback_token_stats', 'test_agent_step_callback_token_stats.py'),
        ('base_agent_run', 'test_base_agent_run.py'),
        ('codact_prompt_templates', 'test_codact_prompt_templates.py'),
        ('final_answer_tool', 'test_final_answer_tool.py'),
        ('multi_model_router', 'test_multi_model_router.py'),
        ('runtime_llm_model', 'test_runtime_llm_model.py'),
        ('stream_aggregator', 'test_stream_aggregator.py'),
        ('web_ui_processing', 'test_web_ui_processing.py'),
    ]

    # Run all tests
    all_stats = []
    for description, test_file_name in test_files:
        test_path = os.path.join(test_dir, test_file_name)
        if not os.path.exists(test_path):
            print(f"\n✗ Test file not found: {test_file_name}")
            all_stats.append({
                'module': description,
                'total': 0,
                'passed': 0,
                'failed': 0,
                'errors': 1,
                'skipped': 0,
                'time': 0,
                'success': False
            })
            continue

        try:
            stats = run_test_file(test_path, description)
            all_stats.append(stats)
        except Exception as e:
            print(f"\n✗ Error running {description} tests: {e}")
            all_stats.append({
                'module': description,
                'total': 0,
                'passed': 0,
                'failed': 0,
                'errors': 1,
                'skipped': 0,
                'time': 0,
                'success': False
            })

    # Print summary
    all_success = print_summary(all_stats)

    # Return exit code
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
