#!/usr/bin/env python3
"""
运行所有测试并生成统计报告
"""

import unittest
import sys
import os
from io import StringIO

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_tests_with_stats():
    """运行所有测试并生成统计报告"""
    
    # 测试文件列表
    test_modules = [
        'test_forward_process',
        'test_selective_log_softmax',
        'test_get_per_token_logps',
        'test_generate_and_score_completions',
        'test_code_reward',
        'test_code_format_reward',
    ]
    
    print("=" * 80)
    print("ml-diffucoder 测试套件")
    print("=" * 80)
    print()
    
    # 收集所有测试
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for module_name in test_modules:
        try:
            module = __import__(module_name)
            suite.addTests(loader.loadTestsFromModule(module))
        except ImportError as e:
            print(f"警告: 无法导入 {module_name}: {e}")
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 生成统计报告
    print()
    print("=" * 80)
    print("测试统计报告")
    print("=" * 80)
    print()
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = total_tests - failures - errors - skipped
    
    print(f"总测试数: {total_tests}")
    print(f"通过: {passed} ({passed/total_tests*100:.1f}%)")
    print(f"失败: {failures} ({failures/total_tests*100:.1f}%)")
    print(f"错误: {errors} ({errors/total_tests*100:.1f}%)")
    print(f"跳过: {skipped} ({skipped/total_tests*100:.1f}%)")
    print()
    
    # 按测试文件统计
    print("按测试文件统计:")
    print("-" * 80)
    
    test_stats = {}
    for test in result.failures + result.errors + [(t, None) for t in result.skipped]:
        test_case = test[0]
        module_name = test_case.__class__.__module__
        if module_name not in test_stats:
            test_stats[module_name] = {'total': 0, 'failed': 0}
        test_stats[module_name]['total'] += 1
        test_stats[module_name]['failed'] += 1
    
    # 统计所有测试
    for test_module in test_modules:
        if test_module not in test_stats:
            test_stats[test_module] = {'total': 0, 'failed': 0}
    
    # 计算每个模块的总测试数
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
            print(f"{module_name:40s} {passed_count:3d}/{total:3d} 通过 ({passed_count/total*100:.1f}%)")
    
    print()
    
    # 显示失败的测试
    if failures:
        print("失败的测试:")
        print("-" * 80)
        for test, traceback in result.failures:
            print(f"✗ {test}")
            print(f"  {traceback.split(chr(10))[0]}")
        print()
    
    # 显示错误的测试
    if errors:
        print("错误的测试:")
        print("-" * 80)
        for test, traceback in result.errors:
            print(f"✗ {test}")
            print(f"  {traceback.split(chr(10))[0]}")
        print()
    
    # 总结
    print("=" * 80)
    if result.wasSuccessful():
        print("✓ 所有测试通过!")
        return 0
    else:
        print("✗ 部分测试失败")
        return 1

if __name__ == '__main__':
    sys.exit(run_tests_with_stats())
