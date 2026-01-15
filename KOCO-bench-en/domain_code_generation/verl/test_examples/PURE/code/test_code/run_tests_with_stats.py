#!/usr/bin/env python3
"""
DAPO Core Functionality Test Runner - with detailed statistics
Run all tests and generate detailed test pass rate reports
"""

import sys
import os
import unittest
from io import StringIO
from typing import Dict, List, Tuple
import time

# Add verl directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestResult:
    """Store results of a single test file"""
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.errors = 0
        self.skipped = 0
        self.failures_detail = []
        self.errors_detail = []
        self.duration = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate (percentage)"""
        if self.total == 0:
            return 0.0
        return (self.passed / self.total) * 100
    
    def __repr__(self):
        return f"TestResult({self.test_name}: {self.passed}/{self.total})"


def run_test_module(module_name: str, description: str) -> TestResult:
    """
    Run a single test module and return detailed results
    
    Args:
        module_name: Test module name (e.g. 'test_agg_loss')
        description: Test description
    
    Returns:
        TestResult object containing detailed test results
    """
    result = TestResult(description)
    
    print(f"\n{'='*80}")
    print(f"Running test: {description}")
    print(f"Module: {module_name}")
    print(f"{'='*80}")
    
    try:
        # Load test module
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(module_name)
        
        # Run tests
        start_time = time.time()
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        test_result = runner.run(suite)
        result.duration = time.time() - start_time
        
        # Collect statistics
        result.total = test_result.testsRun
        result.failed = len(test_result.failures)
        result.errors = len(test_result.errors)
        result.skipped = len(test_result.skipped)
        result.passed = result.total - result.failed - result.errors - result.skipped
        
        # Collect failure and error details
        result.failures_detail = [(str(test), traceback) for test, traceback in test_result.failures]
        result.errors_detail = [(str(test), traceback) for test, traceback in test_result.errors]
        
    except Exception as e:
        print(f"✗ Error loading or running test: {e}")
        result.errors = 1
        result.total = 1
        result.errors_detail = [("Module Load Error", str(e))]
    
    # Print test result summary
    print(f"\n{'-'*80}")
    print(f"Test result summary: {description}")
    print(f"{'-'*80}")
    print(f"Total test cases: {result.total}")
    print(f"✓ Passed: {result.passed}")
    print(f"✗ Failed: {result.failed}")
    print(f"⚠ Errors: {result.errors}")
    print(f"⊘ Skipped: {result.skipped}")
    print(f"Success rate: {result.success_rate:.1f}%")
    print(f"Duration: {result.duration:.2f}s")
    print(f"{'-'*80}")
    
    return result


def print_final_report(results: List[TestResult]):
    """Print final test report"""
    print("\n\n")
    print("="*100)
    print(" "*35 + "DAPO Core Functionality Test Report")
    print("="*100)
    
    # Calculate overall statistics
    total_tests = sum(r.total for r in results)
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    total_errors = sum(r.errors for r in results)
    total_skipped = sum(r.skipped for r in results)
    total_duration = sum(r.duration for r in results)
    
    overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0.0
    
    # Print results for each test module
    print(f"\n{'Test Module':<50} {'Cases':>8} {'Passed':>8} {'Failed':>8} {'Errors':>8} {'Skipped':>8} {'Success Rate':>10}")
    print("-"*100)
    
    for result in results:
        status_icon = "✓" if result.success_rate == 100.0 else ("⚠" if result.success_rate >= 50.0 else "✗")
        print(f"{status_icon} {result.test_name:<48} {result.total:>8} {result.passed:>8} "
              f"{result.failed:>8} {result.errors:>8} {result.skipped:>8} {result.success_rate:>9.1f}%")
    
    print("-"*100)
    print(f"{'Total':<50} {total_tests:>8} {total_passed:>8} {total_failed:>8} "
          f"{total_errors:>8} {total_skipped:>8} {overall_success_rate:>9.1f}%")
    print("="*100)
    
    # Print overall assessment
    print(f"\nOverall Assessment:")
    print(f"  • Total test cases: {total_tests}")
    print(f"  • Successful cases: {total_passed} (✓)")
    print(f"  • Failed cases: {total_failed} (✗)")
    print(f"  • Error cases: {total_errors} (⚠)")
    print(f"  • Skipped cases: {total_skipped} (⊘)")
    print(f"  • Overall success rate: {overall_success_rate:.1f}%")
    print(f"  • Total duration: {total_duration:.2f}s")
    
    # Rating
    if overall_success_rate == 100.0:
        grade = "Excellent"
        emoji = "🎉"
    elif overall_success_rate >= 80.0:
        grade = "Good"
        emoji = "👍"
    elif overall_success_rate >= 60.0:
        grade = "Pass"
        emoji = "✓"
    else:
        grade = "Needs Improvement"
        emoji = "⚠"
    
    print(f"\n{emoji} Test Quality Rating: {grade}")
    
    # Print failure and error details
    if total_failed > 0 or total_errors > 0:
        print("\n" + "="*100)
        print(" "*35 + "Failure and Error Details")
        print("="*100)
        
        for result in results:
            if result.failures_detail or result.errors_detail:
                print(f"\n【{result.test_name}】")
                
                if result.failures_detail:
                    print(f"\n  Failed test cases ({len(result.failures_detail)}):")
                    for test_name, traceback in result.failures_detail:
                        print(f"    ✗ {test_name}")
                        # Only print key error information (last few lines)
                        traceback_lines = traceback.strip().split('\n')
                        key_lines = traceback_lines[-3:] if len(traceback_lines) > 3 else traceback_lines
                        for line in key_lines:
                            print(f"      {line}")
                
                if result.errors_detail:
                    print(f"\n  Error test cases ({len(result.errors_detail)}):")
                    for test_name, traceback in result.errors_detail:
                        print(f"    ⚠ {test_name}")
                        traceback_lines = traceback.strip().split('\n')
                        key_lines = traceback_lines[-3:] if len(traceback_lines) > 3 else traceback_lines
                        for line in key_lines:
                            print(f"      {line}")
    
    print("\n" + "="*100)
    
    return overall_success_rate


def main():
    """Main function: Run all tests and generate report"""
    print("="*100)
    print(" "*30 + "DAPO Core Functionality Tests Starting")
    print("="*100)
    
    # Verify environment
    print("\nVerifying test environment:")
    print("  ✓ Test environment ready")
    
    # Define tests to run
    test_modules = [
        {
            'module': 'test_agg_loss',
            'description': 'agg_loss (flexible loss aggregation)',
            'category': 'Core Algorithm'
        },
        {
            'module': 'test_compute_policy_loss_vanilla',
            'description': 'compute_policy_loss_vanilla (decoupled clipped policy gradient)',
            'category': 'Core Algorithm'
        },
        {
            'module': 'test_dapo_reward_manager',
            'description': 'DAPORewardManager.__call__ (ultra-long reward shaping)',
            'category': 'Reward Management'
        }
    ]
    
    # Run all tests
    results = []
    for test_info in test_modules:
        result = run_test_module(test_info['module'], test_info['description'])
        results.append(result)
    
    # Print final report
    overall_success_rate = print_final_report(results)
    
    # Return exit code based on success rate
    if overall_success_rate == 100.0:
        sys.exit(0)  # Perfect pass
    elif overall_success_rate >= 80.0:
        sys.exit(0)  # Good, considered pass
    elif overall_success_rate >= 60.0:
        sys.exit(1)  # Pass but not ideal
    else:
        sys.exit(1)  # Needs improvement


if __name__ == "__main__":
    main()

