#!/usr/bin/env python3
"""
CTRL 核心功能测试运行器 - 带详细统计信息
运行所有测试并生成详细的测试通过率报告
"""

import sys
import os
import unittest
from io import StringIO
from typing import Dict, List, Tuple
import time

# 添加code目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))


class TestResult:
    """存储单个测试文件的结果"""
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
        """计算成功率（百分比）"""
        if self.total == 0:
            return 0.0
        return (self.passed / self.total) * 100
    
    def __repr__(self):
        return f"TestResult({self.test_name}: {self.passed}/{self.total})"


def run_test_module(module_name: str, description: str) -> TestResult:
    """
    运行单个测试模块并返回详细结果
    
    Args:
        module_name: 测试模块名称（如 'test_build_revision_messages'）
        description: 测试描述
    
    Returns:
        TestResult对象，包含详细的测试结果
    """
    result = TestResult(description)
    
    print(f"\n{'='*80}")
    print(f"运行测试: {description}")
    print(f"模块: {module_name}")
    print(f"{'='*80}")
    
    try:
        # 加载测试模块
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(module_name)
        
        # 运行测试
        start_time = time.time()
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        test_result = runner.run(suite)
        result.duration = time.time() - start_time
        
        # 收集统计信息
        result.total = test_result.testsRun
        result.failed = len(test_result.failures)
        result.errors = len(test_result.errors)
        result.skipped = len(test_result.skipped)
        result.passed = result.total - result.failed - result.errors - result.skipped
        
        # 收集失败和错误的详细信息
        result.failures_detail = [(str(test), traceback) for test, traceback in test_result.failures]
        result.errors_detail = [(str(test), traceback) for test, traceback in test_result.errors]
        
    except Exception as e:
        print(f"✗ 加载或运行测试时出错: {e}")
        import traceback
        traceback.print_exc()
        result.errors = 1
        result.total = 1
        result.errors_detail = [("Module Load Error", str(e))]
    
    # 打印测试结果摘要
    print(f"\n{'-'*80}")
    print(f"测试结果摘要: {description}")
    print(f"{'-'*80}")
    print(f"总测试用例数: {result.total}")
    print(f"✓ 通过: {result.passed}")
    print(f"✗ 失败: {result.failed}")
    print(f"⚠ 错误: {result.errors}")
    print(f"⊘ 跳过: {result.skipped}")
    print(f"成功率: {result.success_rate:.1f}%")
    print(f"耗时: {result.duration:.2f}秒")
    print(f"{'-'*80}")
    
    return result


def print_final_report(results: List[TestResult]):
    """打印最终的测试报告"""
    print("\n\n")
    print("="*100)
    print(" "*35 + "CTRL 核心功能测试报告")
    print("="*100)
    
    # 计算总体统计
    total_tests = sum(r.total for r in results)
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    total_errors = sum(r.errors for r in results)
    total_skipped = sum(r.skipped for r in results)
    total_duration = sum(r.duration for r in results)
    
    overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0.0
    
    # 打印每个测试模块的结果
    print(f"\n{'测试模块':<40} {'用例数':>8} {'通过':>8} {'失败':>8} {'错误':>8} {'跳过':>8} {'成功率':>10}")
    print("-"*100)
    
    for result in results:
        status_icon = "✓" if result.success_rate == 100.0 else ("⚠" if result.success_rate >= 50.0 else "✗")
        print(f"{status_icon} {result.test_name:<38} {result.total:>8} {result.passed:>8} "
              f"{result.failed:>8} {result.errors:>8} {result.skipped:>8} {result.success_rate:>9.1f}%")
    
    print("-"*100)
    print(f"{'总计':<40} {total_tests:>8} {total_passed:>8} {total_failed:>8} "
          f"{total_errors:>8} {total_skipped:>8} {overall_success_rate:>9.1f}%")
    print("="*100)
    
    # 打印总体评估
    print(f"\n总体评估:")
    print(f"  • 总测试用例数: {total_tests}")
    print(f"  • 成功用例数: {total_passed} (✓)")
    print(f"  • 失败用例数: {total_failed} (✗)")
    print(f"  • 错误用例数: {total_errors} (⚠)")
    print(f"  • 跳过用例数: {total_skipped} (⊘)")
    print(f"  • 总体成功率: {overall_success_rate:.1f}%")
    print(f"  • 总耗时: {total_duration:.2f}秒")
    
    # 评级
    if overall_success_rate == 100.0:
        grade = "优秀 (Excellent)"
        emoji = "🎉"
    elif overall_success_rate >= 80.0:
        grade = "良好 (Good)"
        emoji = "👍"
    elif overall_success_rate >= 60.0:
        grade = "及格 (Pass)"
        emoji = "✓"
    else:
        grade = "需要改进 (Needs Improvement)"
        emoji = "⚠"
    
    print(f"\n{emoji} 测试质量评级: {grade}")
    
    # 打印失败和错误的详细信息
    if total_failed > 0 or total_errors > 0:
        print("\n" + "="*100)
        print(" "*35 + "失败和错误详情")
        print("="*100)
        
        for result in results:
            if result.failures_detail or result.errors_detail:
                print(f"\n【{result.test_name}】")
                
                if result.failures_detail:
                    print(f"\n  失败的测试用例 ({len(result.failures_detail)}):")
                    for test_name, traceback in result.failures_detail:
                        print(f"    ✗ {test_name}")
                        # 只打印关键的错误信息（最后几行）
                        traceback_lines = traceback.strip().split('\n')
                        key_lines = traceback_lines[-3:] if len(traceback_lines) > 3 else traceback_lines
                        for line in key_lines:
                            print(f"      {line}")
                
                if result.errors_detail:
                    print(f"\n  错误的测试用例 ({len(result.errors_detail)}):")
                    for test_name, traceback in result.errors_detail:
                        print(f"    ⚠ {test_name}")
                        traceback_lines = traceback.strip().split('\n')
                        key_lines = traceback_lines[-3:] if len(traceback_lines) > 3 else traceback_lines
                        for line in key_lines:
                            print(f"      {line}")
    
    print("\n" + "="*100)
    
    return overall_success_rate


def main():
    """主函数：运行所有测试并生成报告"""
    print("="*100)
    print(" "*30 + "CTRL 核心功能测试开始")
    print("="*100)
    
    # 验证测试环境
    print("\n验证测试环境:")
    code_path = os.path.join(os.path.dirname(__file__), '..', 'code')
    if os.path.exists(code_path):
        print(f"  ✓ 代码目录存在: {code_path}")
    else:
        print(f"  ⚠ 代码目录不存在: {code_path}")
    
    critic_rm_path = os.path.join(code_path, 'ctrl', 'rl', 'critic_rm.py')
    if os.path.exists(critic_rm_path):
        print(f"  ✓ critic_rm.py 文件存在")
    else:
        print(f"  ⚠ critic_rm.py 文件不存在，测试可能失败")
    
    # 定义要运行的测试
    test_modules = [
        {
            'module': 'test_build_revision_messages',
            'description': '修订提示构建 (build_revision_messages)',
            'category': '提示构建'
        },
        {
            'module': 'test_get_reward_all',
            'description': '并发沙盒执行与奖励计算 (get_reward_all)',
            'category': '奖励计算'
        }
    ]
    
    # 运行所有测试
    results = []
    for test_info in test_modules:
        result = run_test_module(test_info['module'], test_info['description'])
        results.append(result)
    
    # 打印最终报告
    overall_success_rate = print_final_report(results)
    
    # 根据成功率返回退出码
    if overall_success_rate == 100.0:
        sys.exit(0)  # 完美通过
    elif overall_success_rate >= 80.0:
        sys.exit(0)  # 良好，视为通过
    elif overall_success_rate >= 60.0:
        sys.exit(1)  # 及格但不理想
    else:
        sys.exit(1)  # 需要改进


if __name__ == "__main__":
    main()

