#!/usr/bin/env python3
# Copyright 2024 ByteMLPerf. All rights reserved.

"""
Script for running a single test file
"""

import unittest
import sys
import os
import time

def run_single_test(test_file):
    """Run a single test file"""
    print(f"Running test: {test_file}")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Import test module
        test_module = __import__(test_file[:-3])  # Remove .py extension
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_module)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "success": result.wasSuccessful(),
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "duration": duration,
            "result": result
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"❌ Error running test: {e}")
        return {
            "success": False,
            "tests_run": 0,
            "failures": 0,
            "errors": 1,
            "duration": duration,
            "error": str(e)
        }

def main():
    """Main function"""
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("ByteMLPerf Single Test Runner")
    print("=" * 50)
    print(f"Test directory: {script_dir}")
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python run_single_test.py <test_file>")
        print("Available test files:")
        for file in os.listdir("."):
            if file.startswith("test_") and file.endswith(".py"):
                print(f"  - {file}")
        sys.exit(1)
    
    test_file = sys.argv[1]
    
    if not os.path.exists(test_file):
        print(f"❌ Test file does not exist: {test_file}")
        sys.exit(1)
    
    # Run test
    result = run_single_test(test_file)
    
    # Print results
    print(f"\n{'='*50}")
    print("Test Results")
    print(f"{'='*50}")
    
    status = "✅ Passed" if result["success"] else "❌ Failed"
    duration = result["duration"]
    tests_run = result["tests_run"]
    failures = result["failures"]
    errors = result["errors"]
    
    print(f"Status: {status}")
    print(f"Tests run: {tests_run}")
    print(f"Failures: {failures}")
    print(f"Errors: {errors}")
    print(f"Duration: {duration:.2f}s")
    
    if not result["success"] and "error" in result:
        print(f"Error: {result['error']}")
    
    # Return appropriate exit code
    if result["success"]:
        print(f"\n🎉 Tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
