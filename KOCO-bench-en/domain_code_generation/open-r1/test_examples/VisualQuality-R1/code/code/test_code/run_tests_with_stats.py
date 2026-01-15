#!/usr/bin/env python3
"""
运行所有测试并生成详细的统计报告
"""

import unittest
import sys
import time
import os
from io import StringIO
from datetime import datetime

# 测试文件列表
TEST_FILES = [
    'test_fidelity_reward',
    'test_accuracy_reward',
    'test_format_reward',
    'test_generate_and_score_completions',
    'test_compute_loss',
    'test_repeat_random_sampler',
]


class TestResult:
    """存储单个测试的结果"""
    def __init__(self, name, status, duration, error_msg=None):
        self.name = name
        self.status = status  # 'PASS', 'FAIL', 'ERROR', 'SKIP'
        self.duration = duration
        self.error_msg = error_msg


class ColoredTextTestResult(unittest.TextTestResult):
    """自定义测试结果类，支持彩色输出和详细统计"""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_results = []
        self.start_time = None
        
    def startTest(self, test):
        super().startTest(test)
        self.start_time = time.time()
        
    def addSuccess(self, test):
        super().addSuccess(test)
        duration = time.time() - self.start_time
        self.test_results.append(TestResult(
            self._get_test_name(test),
            'PASS',
            duration
        ))
        if self.verbosity >= 1:
            self.stream.write(f"✓ {self._get_test_name(test)} ({duration:.3f}s)\n")
            self.stream.flush()
    
    def addError(self, test, err):
        super().addError(test, err)
        duration = time.time() - self.start_time
        error_msg = self._exc_info_to_string(err, test)
        self.test_results.append(TestResult(
            self._get_test_name(test),
            'ERROR',
            duration,
            error_msg
        ))
        if self.verbosity >= 1:
            self.stream.write(f"✗ {self._get_test_name(test)} - ERROR ({duration:.3f}s)\n")
            self.stream.flush()
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        duration = time.time() - self.start_time
        error_msg = self._exc_info_to_string(err, test)
        self.test_results.append(TestResult(
            self._get_test_name(test),
            'FAIL',
            duration,
            error_msg
        ))
        if self.verbosity >= 1:
            self.stream.write(f"✗ {self._get_test_name(test)} - FAIL ({duration:.3f}s)\n")
            self.stream.flush()
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        duration = time.time() - self.start_time
        self.test_results.append(TestResult(
            self._get_test_name(test),
            'SKIP',
            duration,
            reason
        ))
        if self.verbosity >= 1:
            self.stream.write(f"⊘ {self._get_test_name(test)} - SKIPPED ({duration:.3f}s)\n")
            self.stream.flush()
    
    def _get_test_name(self, test):
        """获取测试的完整名称"""
        return str(test)


def print_header(text):
    """打印带边框的标题"""
    width = 80
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width + "\n")


def print_section(text):
    """打印章节标题"""
    print("\n" + "-" * 80)
    print(text)
    print("-" * 80)


def run_test_module(module_name, verbosity=2):
    """运行单个测试模块"""
    print_section(f"Running {module_name}")
    
    # 动态导入测试模块
    try:
        test_module = __import__(module_name)
    except ImportError as e:
        print(f"✗ Failed to import {module_name}: {e}")
        return None
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_module)
    
    # 运行测试
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        resultclass=ColoredTextTestResult
    )
    result = runner.run(suite)
    
    return result


def generate_statistics(all_results):
    """生成统计报告"""
    print_header("Test Statistics")
    
    # 收集所有测试结果
    all_test_results = []
    for result in all_results:
        if result and hasattr(result, 'test_results'):
            all_test_results.extend(result.test_results)
    
    # 计算统计数据
    total_tests = len(all_test_results)
    passed = sum(1 for r in all_test_results if r.status == 'PASS')
    failed = sum(1 for r in all_test_results if r.status == 'FAIL')
    errors = sum(1 for r in all_test_results if r.status == 'ERROR')
    skipped = sum(1 for r in all_test_results if r.status == 'SKIP')
    
    total_duration = sum(r.duration for r in all_test_results)
    avg_duration = total_duration / total_tests if total_tests > 0 else 0
    
    # 打印总体统计
    print(f"Total Tests:     {total_tests}")
    print(f"Passed:          {passed} ({passed/total_tests*100:.1f}%)" if total_tests > 0 else "Passed:          0")
    print(f"Failed:          {failed} ({failed/total_tests*100:.1f}%)" if total_tests > 0 else "Failed:          0")
    print(f"Errors:          {errors} ({errors/total_tests*100:.1f}%)" if total_tests > 0 else "Errors:          0")
    print(f"Skipped:         {skipped} ({skipped/total_tests*100:.1f}%)" if total_tests > 0 else "Skipped:         0")
    print(f"\nTotal Duration:  {total_duration:.3f}s")
    print(f"Average Duration: {avg_duration:.3f}s")
    
    # 按测试文件分组统计
    print_section("Statistics by Test File")
    
    test_files = {}
    for result in all_test_results:
        # 从测试名称中提取文件名
        test_name = result.name
        if '(' in test_name:
            file_name = test_name.split('(')[1].split('.')[0]
        else:
            file_name = 'unknown'
        
        if file_name not in test_files:
            test_files[file_name] = {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'skipped': 0,
                'duration': 0.0
            }
        
        test_files[file_name]['total'] += 1
        test_files[file_name]['duration'] += result.duration
        
        if result.status == 'PASS':
            test_files[file_name]['passed'] += 1
        elif result.status == 'FAIL':
            test_files[file_name]['failed'] += 1
        elif result.status == 'ERROR':
            test_files[file_name]['errors'] += 1
        elif result.status == 'SKIP':
            test_files[file_name]['skipped'] += 1
    
    # 打印每个文件的统计
    for file_name, stats in sorted(test_files.items()):
        print(f"\n{file_name}:")
        print(f"  Total:    {stats['total']}")
        print(f"  Passed:   {stats['passed']}")
        print(f"  Failed:   {stats['failed']}")
        print(f"  Errors:   {stats['errors']}")
        print(f"  Skipped:  {stats['skipped']}")
        print(f"  Duration: {stats['duration']:.3f}s")
        
        if stats['total'] > 0:
            success_rate = stats['passed'] / stats['total'] * 100
            print(f"  Success Rate: {success_rate:.1f}%")
    
    # 找出最慢的测试
    print_section("Slowest Tests (Top 10)")
    
    sorted_by_duration = sorted(all_test_results, key=lambda r: r.duration, reverse=True)
    for i, result in enumerate(sorted_by_duration[:10], 1):
        status_symbol = "✓" if result.status == 'PASS' else "✗"
        print(f"{i:2d}. {status_symbol} {result.name:<60} {result.duration:.3f}s")
    
    # 打印失败和错误的详细信息
    failed_tests = [r for r in all_test_results if r.status in ['FAIL', 'ERROR']]
    if failed_tests:
        print_section("Failed/Error Tests Details")
        for result in failed_tests:
            print(f"\n{result.status}: {result.name}")
            if result.error_msg:
                print(result.error_msg)
    
    return {
        'total': total_tests,
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'skipped': skipped,
        'duration': total_duration
    }


def save_report(stats, all_results, output_file='test_report.txt'):
    """保存测试报告到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("VisualQuality-R1 Test Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Summary:\n")
        f.write(f"  Total Tests: {stats['total']}\n")
        f.write(f"  Passed:      {stats['passed']}\n")
        f.write(f"  Failed:      {stats['failed']}\n")
        f.write(f"  Errors:      {stats['errors']}\n")
        f.write(f"  Skipped:     {stats['skipped']}\n")
        f.write(f"  Duration:    {stats['duration']:.3f}s\n\n")
        
        if stats['total'] > 0:
            success_rate = stats['passed'] / stats['total'] * 100
            f.write(f"  Success Rate: {success_rate:.1f}%\n")
        
        # 保存详细结果
        f.write("\n" + "=" * 80 + "\n")
        f.write("Detailed Results:\n")
        f.write("=" * 80 + "\n\n")
        
        for result in all_results:
            if result and hasattr(result, 'test_results'):
                for test_result in result.test_results:
                    status_symbol = {
                        'PASS': '✓',
                        'FAIL': '✗',
                        'ERROR': '✗',
                        'SKIP': '⊘'
                    }.get(test_result.status, '?')
                    
                    f.write(f"{status_symbol} {test_result.name} - {test_result.status} ({test_result.duration:.3f}s)\n")
                    
                    if test_result.error_msg:
                        f.write(f"  Error: {test_result.error_msg}\n")
                    f.write("\n")
    
    print(f"\nTest report saved to: {output_file}")


def main():
    """主函数"""
    print_header("VisualQuality-R1 Test Suite with Statistics")
    
    start_time = time.time()
    
    # 运行所有测试
    all_results = []
    for test_file in TEST_FILES:
        result = run_test_module(test_file, verbosity=1)
        all_results.append(result)
    
    total_time = time.time() - start_time
    
    # 生成统计报告
    stats = generate_statistics(all_results)
    
    # 打印总体结果
    print_header("Overall Result")
    print(f"Total execution time: {total_time:.3f}s")
    
    if stats['failed'] == 0 and stats['errors'] == 0:
        print("\n✓ All tests passed!")
        exit_code = 0
    else:
        print(f"\n✗ {stats['failed'] + stats['errors']} test(s) failed!")
        exit_code = 1
    
    # 保存报告
    save_report(stats, all_results)
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
