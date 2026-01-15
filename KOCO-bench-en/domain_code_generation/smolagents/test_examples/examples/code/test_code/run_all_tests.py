import unittest
import os
import sys
import io

# Unify stdout/stderr to UTF-8, avoiding console character garbling (Windows/Powershell/VSCode)
try:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
except Exception:
    # Does not affect test execution
    pass

def run_all_tests():
    """
    Discovers and runs all tests in the current directory, handling path issues.
    """
    current_dir = os.path.dirname(__file__)
    
    # Add parent code directory to path
    code_dir = os.path.abspath(os.path.join(current_dir, '..'))
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    # Add subdirectories to path for nested modules
    for dir_name in ['async_agent', 'plan_customization', 'server']:
        sub_dir_path = os.path.join(code_dir, dir_name)
        if sub_dir_path not in sys.path:
            sys.path.insert(0, sub_dir_path)

    # Discover all tests in the current directory
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=current_dir, pattern='test_*.py')

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print("\n" + "="*30 + " Test Summary " + "="*30)
    print(f"Total tests run: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    
    if total_tests > 0:
        pass_rate = (passed / total_tests) * 100
        print(f"Pass rate: {pass_rate:.2f}%")
    else:
        print("No tests found.")
    print("="*70)

    if not result.wasSuccessful():
        sys.exit(1)

if __name__ == '__main__':
    run_all_tests()