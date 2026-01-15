#!/usr/bin/env python3
"""
CANOE测试运行脚本（带统计信息）
运行所有核心函数的测试并生成统计报告
"""

import sys
import os
import unittest
import time
from io import StringIO

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train', 'src'))


def run_test_module(module_name, test_file):
    """运行单个测试模块并返回结果"""
    print(f"\n{'='*60}")
    print(f"Testing: {module_name}")
    print(f"{'='*60}")
    
    # 导入测试模块
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_file)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # 收集统计信息
    stats = {
        'module': module_name,
        'total': result.testsRun,
        'passed': result.testsRun - len(result.failures) - len(result.errors),
        'failed': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped),
        'time': end_time - start_time,
        'success': result.wasSuccessful()
    }
    
    return stats


def print_summary(all_stats):
    """打印测试摘要"""
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}\n")
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_errors = 0
    total_skipped = 0
    total_time = 0
    
    print(f"{'Module':<30} {'Total':<8} {'Passed':<8} {'Failed':<8} {'Time(s)':<10}")
    print(f"{'-'*70}")
    
    for stats in all_stats:
        print(f"{stats['module']:<30} {stats['total']:<8} {stats['passed']:<8} "
              f"{stats['failed']:<8} {stats['time']:<10.2f}")
        
        total_tests += stats['total']
        total_passed += stats['passed']
        total_failed += stats['failed']
        total_errors += stats['errors']
        total_skipped += stats['skipped']
        total_time += stats['time']
    
    print(f"{'-'*70}")
    print(f"{'TOTAL':<30} {total_tests:<8} {total_passed:<8} {total_failed:<8} {total_time:<10.2f}")
    print()
    
    # 打印详细统计
    print(f"Total Tests Run: {total_tests}")
    print(f"Passed: {total_passed} ({100*total_passed/total_tests if total_tests > 0 else 0:.1f}%)")
    print(f"Failed: {total_failed}")
    print(f"Errors: {total_errors}")
    print(f"Skipped: {total_skipped}")
    print(f"Total Time: {total_time:.2f}s")
    print()
    
    # 判断整体结果
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
    """主函数"""
    print("="*60)
    print("AlphaDrive Core Function Tests")
    print("="*60)
    
    # 定义要测试的模块
    test_modules = [
        ('compute_loss', 'test_compute_loss'),
        ('gen_per_token_logps', 'test_gen_per_token_logps'),
        ('plan_format_reward', 'test_plan_format_reward'),
        ('plan_path_reward', 'test_plan_path_reward'),
        ('plan_speed_reward', 'test_plan_speed_reward'),
    ]
    
    # 运行所有测试
    all_stats = []
    for module_name, test_file in test_modules:
        try:
            stats = run_test_module(module_name, test_file)
            all_stats.append(stats)
        except Exception as e:
            print(f"\n✗ Error running {module_name} tests: {e}")
            all_stats.append({
                'module': module_name,
                'total': 0,
                'passed': 0,
                'failed': 0,
                'errors': 1,
                'skipped': 0,
                'time': 0,
                'success': False
            })
    
    # 打印摘要
    all_success = print_summary(all_stats)
    
    # 返回退出码
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
