#!/usr/bin/env python3
"""
纯净模式执行评估脚本 - 修改文件 -> 运行测试 -> 还原文件
"""

import json
import sys
import os
import re
import argparse
import subprocess
import shutil
import tempfile
from typing import List, Dict, Any, Tuple
import numpy as np
import itertools
from datetime import datetime


class PureCodeReplacer:
    """纯净代码替换引擎 - 直接修改文件"""
    
    def parse_location(self, location: str) -> Tuple[int, int]:
        """解析location字符串，返回起始行和结束行"""
        pattern = r".*:line\s+(\d+)-(\d+)"
        match = re.search(pattern, location)
        if not match:
            raise ValueError(f"Invalid location format: {location}")
        return int(match.group(1)), int(match.group(2))
    
    def extract_code_from_markdown(self, completion: str) -> str:
        """从 markdown 格式中提取纯代码"""
        if "```python" in completion:
            start = completion.find("```python") + len("```python")
            end = completion.find("```", start)
            if end == -1:
                end = len(completion)
            code_block = completion[start:end].strip()
        elif "```" in completion:
            start = completion.find("```") + len("```")
            end = completion.find("```", start)
            if end == -1:
                end = len(completion)
            code_block = completion[start:end].strip()
        else:
            code_block = completion.strip()
        
        return code_block
    
    def normalize_indentation(self, code: str) -> str:
        """去除代码的基础缩进"""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        if not non_empty_lines:
            return code
        
        min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
        
        normalized_lines = []
        for line in lines:
            if line.strip():
                normalized_lines.append(line[min_indent:])
            else:
                normalized_lines.append('')
        
        return '\n'.join(normalized_lines)
    
    def apply_indentation(self, code: str, base_indent: int) -> str:
        """给代码添加指定的基础缩进"""
        lines = code.split('\n')
        indented_lines = []
        for line in lines:
            if line.strip():
                indented_lines.append(' ' * base_indent + line)
            else:
                indented_lines.append('')
        return '\n'.join(indented_lines)
    
    def replace_function_in_file(self, file_path: str, location: str, completion: str) -> bool:
        """在文件中替换函数实现"""
        try:
            # 读取原文件
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            start_line, end_line = self.parse_location(location)
            lines = source_code.split('\n')
            
            # 提取代码
            extracted_code = self.extract_code_from_markdown(completion)
            normalized_code = self.normalize_indentation(extracted_code)
            
            # 获取原文件的基础缩进
            original_first_line = lines[start_line - 1]
            base_indent = len(original_first_line) - len(original_first_line.lstrip())
            
            # 添加缩进
            indented_code = self.apply_indentation(normalized_code, base_indent)
            
            # 替换代码
            indented_lines = indented_code.split('\n')
            lines[start_line-1:end_line] = indented_lines
            
            # 写回文件
            modified_code = '\n'.join(lines)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_code)
            
            return True
            
        except Exception as e:
            print(f"Error replacing function: {e}")
            import traceback
            traceback.print_exc()
            return False


class PureTestExecutor:
    """纯净测试执行器 - 支持 unittest 和 pytest"""
    
    def __init__(self, source_dir: str, log_dir: str = None):
        self.source_dir = source_dir
        self.log_dir = log_dir
        
        # 创建日志目录
        if self.log_dir and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
    
    def _detect_test_framework(self, test_file_path: str) -> str:
        """检测测试文件使用的测试框架
        
        Args:
            test_file_path: 测试文件的完整路径
            
        Returns:
            'pytest' 或 'unittest'
        """
        try:
            with open(test_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 检查是否导入了 pytest
            if 'import pytest' in content or 'from pytest' in content:
                return 'pytest'
            
            # 检查是否使用了 pytest 的装饰器
            if '@pytest.' in content:
                return 'pytest'
            
            # 检查是否使用了 pytest 的 fixture
            if 'def test_' in content and ('@pytest.fixture' in content or 'pytest.raises' in content):
                return 'pytest'
            
            # 默认使用 unittest
            return 'unittest'
            
        except Exception as e:
            print(f"    Warning: Failed to detect test framework, defaulting to unittest: {e}")
            return 'unittest'
    
    def run_test_file(self, test_file_path: str, function_name: str = "") -> Tuple[bool, float]:
        """运行测试文件 - 自动检测 unittest 或 pytest
        
        Args:
            test_file_path: 测试文件路径
            function_name: 函数名（用于生成日志文件名）
        
        Returns:
            Tuple[bool, float]: (是否全部通过, 通过比例)
        """
        try:
            full_test_path = os.path.join(self.source_dir, test_file_path)
            if not os.path.exists(full_test_path):
                print(f"Test file not found: {full_test_path}")
                return False, 0.0
            
            # 检测测试框架
            framework = self._detect_test_framework(full_test_path)
            print(f"    🔍 检测到测试框架: {framework}")
            
            # 根据框架选择运行命令
            if framework == 'pytest':
                # 使用 pytest 运行，-v 显示详细信息，-s 显示输出
                cmd = [sys.executable, '-m', 'pytest', full_test_path, '-v', '--tb=short']
            else:
                # 使用 unittest 方式运行（直接执行文件）
                cmd = [sys.executable, full_test_path]
            
            # 运行测试
            result = subprocess.run(
                cmd,
                cwd=self.source_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # 保存日志文件
            if self.log_dir:
                self._save_test_log(result, test_file_path, function_name, framework)
            
            # 检查返回码（注意：即使所有测试都跳过，返回码也是0）
            returncode_ok = result.returncode == 0
            
            # 解析测试结果，传入框架类型以便正确解析
            pass_ratio = self._parse_test_output(result.stdout, result.stderr, returncode_ok, framework)
            
            # 判断是否真正全部通过：通过率必须大于0（排除全部跳过的情况）
            all_passed = returncode_ok and pass_ratio > 0.0
            
            return all_passed, pass_ratio
            
        except subprocess.TimeoutExpired as e:
            print(f"Test timeout")
            # 保存超时日志
            if self.log_dir:
                self._save_timeout_log(test_file_path, function_name)
            return False, 0.0
        except Exception as e:
            print(f"Error running test: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0
    
    def _save_test_log(self, result: subprocess.CompletedProcess, test_file_path: str, function_name: str, framework: str = "unknown"):
        """保存测试运行日志
        
        Args:
            result: subprocess 运行结果
            test_file_path: 测试文件路径
            function_name: 函数名
            framework: 测试框架类型 (pytest/unittest)
        """
        try:
            # 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
            
            # 生成日志文件名：函数名_时间戳.log
            if function_name:
                # 清理函数名中的特殊字符
                safe_function_name = re.sub(r'[^\w\-_.]', '_', function_name)
                log_filename = f"{safe_function_name}_{timestamp}.log"
            else:
                test_basename = os.path.basename(test_file_path).replace('.py', '')
                log_filename = f"{test_basename}_{timestamp}.log"
            
            log_filepath = os.path.join(self.log_dir, log_filename)
            
            # 写入日志内容
            with open(log_filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"函数名: {function_name}\n")
                f.write(f"测试文件: {test_file_path}\n")
                f.write(f"测试框架: {framework}\n")
                f.write(f"返回码: {result.returncode}\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("【STDOUT】\n")
                f.write("-" * 80 + "\n")
                f.write(result.stdout if result.stdout else "(无输出)\n")
                f.write("\n")
                
                f.write("【STDERR】\n")
                f.write("-" * 80 + "\n")
                f.write(result.stderr if result.stderr else "(无错误)\n")
                f.write("\n")
                
                f.write("=" * 80 + "\n")
                f.write("日志结束\n")
            
            print(f"    📝 日志已保存: {log_filename}")
            
        except Exception as e:
            print(f"    ⚠️  保存日志失败: {e}")
    
    def _save_timeout_log(self, test_file_path: str, function_name: str):
        """保存超时日志"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe_function_name = re.sub(r'[^\w\-_.]', '_', function_name) if function_name else 'unknown'
            log_filename = f"{safe_function_name}_{timestamp}_TIMEOUT.log"
            log_filepath = os.path.join(self.log_dir, log_filename)
            
            with open(log_filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"函数名: {function_name}\n")
                f.write(f"测试文件: {test_file_path}\n")
                f.write(f"状态: 超时 (TIMEOUT)\n")
                f.write("=" * 80 + "\n")
            
            print(f"    📝 超时日志已保存: {log_filename}")
        except Exception as e:
            print(f"    ⚠️  保存超时日志失败: {e}")
    
    def _parse_test_output(self, stdout: str, stderr: str, all_passed: bool, framework: str = "unittest") -> float:
        """解析测试输出，提取通过率
        
        支持多种测试框架的输出格式：
        - unittest: "Ran X tests", "FAILED (failures=Y)"
        - pytest: "X passed", "Y failed", "X passed in Xs"
        - 字符统计: 优先统计 E/F/. 字符（最准确）
        
        注意：跳过的测试（skipped）不计入通过
        
        Args:
            stdout: 标准输出
            stderr: 标准错误输出
            all_passed: 返回码是否为0
            framework: 测试框架类型 (pytest/unittest)
        
        Returns:
            float: 通过的测试用例比例 (0.0-1.0)
        """
        combined_output = stdout + stderr
        
        # 如果是 pytest，优先使用 pytest 特定的解析方法
        if framework == 'pytest':
            ratio = self._parse_pytest_output(combined_output, all_passed)
            if ratio is not None:
                return ratio
        
        # 检查是否所有测试都被跳过
        # 例如: "Ran 11 tests in 0.000s\n\nOK (skipped=11)"
        import re
        ran_match = re.search(r'Ran (\d+) test', combined_output)
        skipped_match = re.search(r'skipped=(\d+)', combined_output)
        
        if ran_match and skipped_match:
            total_tests = int(ran_match.group(1))
            skipped_tests = int(skipped_match.group(1))
            
            # 如果所有测试都被跳过，返回0.0
            if total_tests == skipped_tests:
                return 0.0
        
        # 如果全部通过（且不是全部跳过），返回1.0
        if all_passed:
            # 再次检查是否有跳过的测试
            if skipped_match and ran_match:
                total_tests = int(ran_match.group(1))
                skipped_tests = int(skipped_match.group(1))
                # 如果有部分跳过，计算实际通过率
                if skipped_tests > 0 and total_tests > skipped_tests:
                    # 有部分测试通过，部分跳过
                    passed_tests = total_tests - skipped_tests
                    return passed_tests / total_tests
            return 1.0
        
        # 方法1：优先使用字符统计（unittest 实时输出的 E/F/.）
        # 这是最准确的方法，因为每个测试都会输出一个字符
        error_count = combined_output.count('E')
        failure_count = combined_output.count('F')
        passed_count = combined_output.count('.')
        
        # 如果找到了字符统计标记，优先使用这个
        if error_count > 0 or failure_count > 0 or passed_count > 0:
            total_from_chars = error_count + failure_count + passed_count
            if total_from_chars > 0:
                # 验证：检查字符统计是否合理（避免匹配到代码中的字符）
                # 通常测试输出的 E/F/. 会连续出现
                char_pattern = re.search(r'[EF.]{3,}', combined_output)
                if char_pattern:
                    # 只统计连续出现的测试标记字符
                    test_markers = char_pattern.group(0)
                    error_count = test_markers.count('E')
                    failure_count = test_markers.count('F')
                    passed_count = test_markers.count('.')
                    total_from_chars = len(test_markers)
                    
                    pass_ratio = passed_count / total_from_chars
                    return max(0.0, min(1.0, pass_ratio))
        
        # 方法2：解析 unittest 文本格式
        # 例如: "Ran 5 tests in 0.001s" 和 "FAILED (failures=2, errors=1, skipped=1)"
        ran_match = re.search(r'Ran (\d+) test', combined_output)
        if ran_match:
            total_tests = int(ran_match.group(1))
            
            # 匹配失败数量、错误数量、跳过数量
            failures = 0
            errors = 0
            skipped = 0
            
            failure_match = re.search(r'failures=(\d+)', combined_output)
            if failure_match:
                failures = int(failure_match.group(1))
            
            error_match = re.search(r'errors=(\d+)', combined_output)
            if error_match:
                errors = int(error_match.group(1))
            
            skipped_match = re.search(r'skipped=(\d+)', combined_output)
            if skipped_match:
                skipped = int(skipped_match.group(1))
            
            if total_tests > 0:
                # 通过的测试 = 总数 - 失败 - 错误 - 跳过
                # 跳过的测试不应该算作通过
                passed_tests = total_tests - failures - errors - skipped
                
                # 确保通过率在 [0.0, 1.0] 范围内
                passed_tests = max(0, min(passed_tests, total_tests))
                return passed_tests / total_tests
        
        # 尝试解析 pytest 格式
        # 例如: "3 passed, 2 failed in 0.12s"
        pytest_match = re.search(r'(\d+) passed(?:, (\d+) failed)?', combined_output)
        if pytest_match:
            passed = int(pytest_match.group(1))
            failed = int(pytest_match.group(2)) if pytest_match.group(2) else 0
            total = passed + failed
            if total > 0:
                # 确保通过率在 [0.0, 1.0] 范围内
                pass_ratio = passed / total
                return max(0.0, min(1.0, pass_ratio))
        
        # 如果无法解析但测试失败了，返回0.0
        return 0.0
    
    def _parse_pytest_output(self, output: str, all_passed: bool) -> float:
        """解析 pytest 的输出格式
        
        Pytest 输出示例:
        - "3 passed in 0.12s"
        - "1 failed, 2 passed in 0.12s"
        - "3 passed, 1 skipped in 0.12s"
        - "collected 3 items" ... "test_file.py::test_name PASSED"
        
        Args:
            output: pytest 的输出
            all_passed: 返回码是否为0
            
        Returns:
            float or None: 通过率，如果无法解析返回 None
        """
        import re
        
        # 方法1: 解析最后的汇总行 "X passed, Y failed in Zs"
        # 匹配 "5 passed in 0.12s" 或 "2 failed, 3 passed in 0.12s"
        summary_pattern = r'(?:(\d+) failed)?(?:, )?(?:(\d+) passed)?(?:, )?(?:(\d+) skipped)?(?:, )?(?:(\d+) error)?.*in ([\d.]+)s'
        summary_match = re.search(summary_pattern, output)
        
        if summary_match:
            failed = int(summary_match.group(1)) if summary_match.group(1) else 0
            passed = int(summary_match.group(2)) if summary_match.group(2) else 0
            skipped = int(summary_match.group(3)) if summary_match.group(3) else 0
            errors = int(summary_match.group(4)) if summary_match.group(4) else 0
            
            # 计算总测试数（不包括跳过的）
            total = passed + failed + errors
            
            # 如果所有测试都被跳过
            if total == 0 and skipped > 0:
                return 0.0
            
            if total > 0:
                return passed / total
        
        # 方法2: 统计测试结果标记 PASSED / FAILED / ERROR / SKIPPED
        passed_count = len(re.findall(r'\bPASSED\b', output))
        failed_count = len(re.findall(r'\bFAILED\b', output))
        error_count = len(re.findall(r'\bERROR\b', output))
        skipped_count = len(re.findall(r'\bSKIPPED\b', output))
        
        total_from_markers = passed_count + failed_count + error_count
        
        if total_from_markers > 0:
            return passed_count / total_from_markers
        
        # 方法3: 检查 collected 行
        # 例如: "collected 3 items"
        collected_match = re.search(r'collected (\d+) items?', output)
        if collected_match:
            total_tests = int(collected_match.group(1))
            
            # 如果 returncode 为 0 且没有找到失败信息，说明全部通过
            if all_passed and total_tests > 0:
                # 检查是否全部被跳过
                if skipped_count == total_tests:
                    return 0.0
                return 1.0
        
        # 如果返回码为0且找到了 passed，认为全部通过
        if all_passed and 'passed' in output.lower():
            return 1.0
        
        # 无法解析
        return None


def estimate_pass_at_k(num_samples, num_correct, k: int) -> np.ndarray:
    """Estimates pass@k"""
    def estimator(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


class ResultCollector:
    """结果收集器"""
    
    def load_jsonl_data(self, file_path: str) -> List[Dict[str, Any]]:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def save_jsonl_data(self, data: List[Dict[str, Any]], file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    def calculate_pass_at_k(self, data_records: List[Dict[str, Any]], k_values: List[int] = [1, 2, 3, 4]) -> Dict[str, float]:
        total_samples = []
        correct_samples = []
        
        for record in data_records:
            if 'results' in record and record['results']:
                num_samples = len(record['results'])
                num_correct = sum(record['results'])
                total_samples.append(num_samples)
                correct_samples.append(num_correct)
        
        if not total_samples:
            return {}
        
        total_samples = np.array(total_samples)
        correct_samples = np.array(correct_samples)
        
        pass_at_k = {}
        for k in k_values:
            if (total_samples >= k).all():
                pass_k_scores = estimate_pass_at_k(total_samples, correct_samples, k)
                pass_at_k[f"pass@{k}"] = pass_k_scores.mean()
        
        return pass_at_k
    
    def calculate_avg_pass_ratio(self, data_records: List[Dict[str, Any]]) -> float:
        """计算平均通过率 - 更细粒度的指标
        
        对于每个任务(task_id/function_name)，计算其所有样本的通过率的平均值
        然后对所有任务的平均通过率再求平均
        
        Args:
            data_records: 包含测试结果的记录列表
            
        Returns:
            平均通过率 (0.0-1.0)
        """
        group = {}
        for record in data_records:
            # 使用 function_name 作为 task_id
            task_id = record.get('function_name', '')
            if not task_id:
                continue
            
            if task_id not in group:
                group[task_id] = []
                
            # 优先使用 pass_ratios 字段（新版本的细粒度数据）
            if 'pass_ratios' in record and record['pass_ratios']:
                group[task_id].extend(record['pass_ratios'])
            # 其次使用 passed 字段（兼容旧数据格式）
            elif 'passed' in record:
                group[task_id].append(record['passed'])
            # 最后使用 results 字段（布尔值列表），计算通过率
            elif 'results' in record and record['results']:
                # 计算这个 completion 的通过率
                pass_ratio = sum(record['results']) / len(record['results']) if len(record['results']) > 0 else 0.0
                group[task_id].append(pass_ratio)
        
        if not group:
            return 0.0
        
        # 对每个 task_id 计算平均通过率
        task_avg_pass_ratios = []
        for task_id, pass_ratios in group.items():
            task_avg = np.mean(pass_ratios)
            task_avg_pass_ratios.append(task_avg)
        
        # 返回所有任务的平均通过率
        return np.mean(task_avg_pass_ratios)


def main():
    parser = argparse.ArgumentParser(description='Pure mode execution evaluation - modify file, run test, restore')
    parser.add_argument('--source_dir', required=True, help='Source code directory path')
    parser.add_argument('--input_file', required=True, help='Input file path')
    parser.add_argument('--output_file', required=True, help='Output file path')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔬 PURE MODE - Modify file, run test, restore")
    print("=" * 60)
    
    # 创建日志目录（与输出文件同目录）
    output_dir = os.path.dirname(args.output_file)
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    print(f"📁 日志目录: {log_dir}")
    print()
    
    code_replacer = PureCodeReplacer()
    test_executor = PureTestExecutor(args.source_dir, log_dir=log_dir)
    result_collector = ResultCollector()
    
    print(f"Loading data from {args.input_file}...")
    data_records = result_collector.load_jsonl_data(args.input_file)
    print(f"Processing {len(data_records)} records...")
    
    for i, record in enumerate(data_records):
        print(f"Processing record {i+1}/{len(data_records)}: {record['function_name']}")
        
        # Validate that required fields are present
        location = record.get('implementation_location', '')
        test_code_path = record.get('test_code_path', '')
        
        if not location or not location.strip():
            print(f"  ❌ Error: 'implementation_location' field is empty or missing")
            print(f"  This field is required to locate the source file to modify")
            record['results'] = [False] * len(record.get('completions', []))
            record['pass_ratios'] = [0.0] * len(record.get('completions', []))
            continue
        
        if not test_code_path or not test_code_path.strip():
            print(f"  ❌ Error: 'test_code_path' field is empty or missing")
            print(f"  This field is required to locate the test file to run")
            record['results'] = [False] * len(record.get('completions', []))
            record['pass_ratios'] = [0.0] * len(record.get('completions', []))
            continue
        
        source_file = location.split(':')[0]
        
        if source_file.startswith('code/'):
            source_file = source_file[5:]
        
        source_file_path = os.path.join(args.source_dir, source_file)
        
        # Check if source_file_path is a directory (indicates invalid path)
        if os.path.isdir(source_file_path):
            print(f"  ❌ Error: Source path is a directory, not a file: {source_file_path}")
            print(f"  Check that 'implementation_location' contains a valid file path")
            record['results'] = [False] * len(record.get('completions', []))
            record['pass_ratios'] = [0.0] * len(record.get('completions', []))
            continue
        
        if not os.path.exists(source_file_path):
            print(f"  ❌ Source file not found: {source_file_path}")
            record['results'] = [False] * len(record.get('completions', []))
            record['pass_ratios'] = [0.0] * len(record.get('completions', []))
            continue
        
        if test_code_path.startswith('code/'):
            test_code_path = test_code_path[5:]
        
        # 备份原文件
        backup_path = source_file_path + '.backup'
        shutil.copy2(source_file_path, backup_path)
        
        results = []
        pass_ratios = []  # 新增：记录每个completion的通过率
        
        try:
            for j, completion in enumerate(record['completions']):
                print(f"  Testing completion {j+1}/{len(record['completions'])}")
                
                try:
                    # 1. 修改源文件
                    success = code_replacer.replace_function_in_file(
                        source_file_path,
                        record['implementation_location'],
                        completion
                    )
                    
                    if not success:
                        results.append(False)
                        pass_ratios.append(0.0)
                        print(f"    Result: FAIL (replace failed)")
                        # 还原文件
                        shutil.copy2(backup_path, source_file_path)
                        continue
                    
                    # 2. 运行测试
                    test_passed, pass_ratio = test_executor.run_test_file(test_code_path, record['function_name'])
                    
                    results.append(test_passed)
                    pass_ratios.append(pass_ratio)
                    print(f"    Result: {'PASS' if test_passed else 'FAIL'} (通过率: {pass_ratio:.2%})")
                    
                    # 3. 还原文件（为下一个 completion 准备）
                    shutil.copy2(backup_path, source_file_path)
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    results.append(False)
                    pass_ratios.append(0.0)
                    # 确保还原文件
                    shutil.copy2(backup_path, source_file_path)
        
        finally:
            # 确保最后还原文件
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, source_file_path)
                os.remove(backup_path)
        
        record['results'] = results
        record['pass_ratios'] = pass_ratios  # 新增：保存通过率信息
    
    print(f"Saving results to {args.output_file}...")
    result_collector.save_jsonl_data(data_records, args.output_file)
    
    # 计算指标
    print("\nCalculating pass@k metrics...")
    pass_at_k_results = result_collector.calculate_pass_at_k(data_records)
    
    print("Calculating AvgPassRatio metric...")
    avg_pass_ratio = result_collector.calculate_avg_pass_ratio(data_records)
    
    total_tests = sum(len(record['completions']) for record in data_records)
    total_passed = sum(sum(record['results']) for record in data_records)
    
    print(f"\nExecution completed!")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_tests - total_passed}")
    print(f"Success rate: {total_passed/total_tests*100:.1f}%")
    
    if pass_at_k_results:
        print("\nPass@k Results:")
        for metric, value in pass_at_k_results.items():
            print(f"{metric}: {value:.4f}")
    
    print(f"\nAvgPassRatio: {avg_pass_ratio:.4f}")
    
    if pass_at_k_results:
        metrics_file = args.output_file.replace('_result.jsonl', '_result.metrics.json')
        print(f"Saving metrics to {metrics_file}...")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_functions': len(data_records),
                'total_tests': total_tests,
                'total_passed': total_passed,
                'overall_success_rate': total_passed/total_tests if total_tests > 0 else 0.0,
                'pass_at_k': pass_at_k_results,
                'avg_pass_ratio': avg_pass_ratio
            }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
