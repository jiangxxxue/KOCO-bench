import unittest
import os
import sys
import io

# 统一标准输出/错误输出为 UTF-8，避免控制台字符乱码（Windows/Powershell/VSCode）
try:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
except Exception:
    # 不影响测试执行
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
    
    print("\n" + "="*30 + " 测试总结 " + "="*30)
    print(f"总共运行测试: {total_tests}")
    print(f"通过: {passed}")
    print(f"失败: {failures}")
    print(f"错误: {errors}")
    
    if total_tests > 0:
        pass_rate = (passed / total_tests) * 100
        print(f"通过率: {pass_rate:.2f}%")
    else:
        print("没有找到任何测试。")
    print("="*70)

    if not result.wasSuccessful():
        sys.exit(1)

if __name__ == '__main__':
    run_all_tests()