#!/usr/bin/env python3
"""
Run all tests and generate statistical report
"""

import unittest
import sys
import os
from io import StringIO

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_tests_with_stats():
    """Run all tests and generate statistical report"""
    
    # List of test files
    test_modules = [
        'test_forward_process',
        'test_selective_log_softmax',
        'test_get_per_token_logps',
        'test_generate_and_score_completions',
        'test_code_reward',
        'test_code_format_reward',
    ]
    
    print("=" * 80)
    print("ml-diffucoder Test Suite")
    print("=" * 80)
    print()
    
    # Collect all tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for module_name in test_modules:
        try:
            module = __import__(module_name)
            suite.addTests(loader.loadTestsFromModule(module))
        except ImportError as e:
            print(f"Warning: Unable to import {module_name}: {e}")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate statistical report
    print()
    print("=" * 80)
    print("Test Statistics Report")
    print("=" * 80)
    print()
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = total_tests - failures - errors - skipped
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed} ({passed/total_tests*100:.1f}%)")
    print(f"Failed: {failures} ({failures/total_tests*100:.1f}%)")
    print(f"Errors: {errors} ({errors/total_tests*100:.1f}%)")
    print(f"Skipped: {skipped} ({skipped/total_tests*100:.1f}%)")
    print()
    
    # Statistics by test file
    print("Statistics by test file:")
    print("-" * 80)
    
    test_stats = {}
    for test in result.failures + result.errors + [(t, None) for t in result.skipped]:
        test_case = test[0]
        module_name = test_case.__class__.__module__
        if module_name not in test_stats:
            test_stats[module_name] = {'total': 0, 'failed': 0}
        test_stats[module_name]['total'] += 1
        test_stats[module_name]['failed'] += 1
    
    # Count all tests
    for test_module in test_modules:
        if test_module not in test_stats:
            test_stats[test_module] = {'total': 0, 'failed': 0}
    
    # Calculate total test count for each module
    for test in suite:
        module_name = test.__class__.__module__
        if module_name in test_stats:
            test_stats[module_name]['total'] += 1
    
    for module_name in sorted(test_stats.keys()):
        stats = test_stats[module_name]
        total = stats['total']
        failed = stats['failed']
        passed_count = total - failed
        if total > 0:
            print(f"{module_name:40s} {passed_count:3d}/{total:3d} passed ({passed_count/total*100:.1f}%)")
    
    print()
    
    # Display failed tests
    if failures:
        print("Failed tests:")
        print("-" * 80)
        for test, traceback in result.failures:
            print(f"✗ {test}")
            print(f"  {traceback.split(chr(10))[0]}")
        print()
    
    # Display error tests
    if errors:
        print("Error tests:")
        print("-" * 80)
        for test, traceback in result.errors:
            print(f"✗ {test}")
            print(f"  {traceback.split(chr(10))[0]}")
        print()
    
    # Summary
    print("=" * 80)
    if result.wasSuccessful():
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(run_tests_with_stats())
