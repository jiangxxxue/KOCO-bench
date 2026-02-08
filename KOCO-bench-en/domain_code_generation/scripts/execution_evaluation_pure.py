#!/usr/bin/env python3
"""
Pure mode execution evaluation script - modify file -> run test -> restore file
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
    """Pure code replacement engine - directly modify files"""
    
    def parse_location(self, location: str) -> Tuple[int, int]:
        """Parse location string, return start line and end line"""
        pattern = r".*:line\s+(\d+)-(\d+)"
        match = re.search(pattern, location)
        if not match:
            raise ValueError(f"Invalid location format: {location}")
        return int(match.group(1)), int(match.group(2))
    
    def extract_code_from_markdown(self, completion: str) -> str:
        """Extract pure code from markdown format"""
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
        """Remove base indentation from code"""
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
        """Add specified base indentation to code"""
        lines = code.split('\n')
        indented_lines = []
        for line in lines:
            if line.strip():
                indented_lines.append(' ' * base_indent + line)
            else:
                indented_lines.append('')
        return '\n'.join(indented_lines)
    
    def replace_function_in_file(self, file_path: str, location: str, completion: str) -> bool:
        """Replace function implementation in file"""
        try:
            # Read original file
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            start_line, end_line = self.parse_location(location)
            lines = source_code.split('\n')
            
            # Extract code
            extracted_code = self.extract_code_from_markdown(completion)
            normalized_code = self.normalize_indentation(extracted_code)
            
            # Get base indentation of original file
            original_first_line = lines[start_line - 1]
            base_indent = len(original_first_line) - len(original_first_line.lstrip())
            
            # Add indentation
            indented_code = self.apply_indentation(normalized_code, base_indent)
            
            # Replace code
            indented_lines = indented_code.split('\n')
            lines[start_line-1:end_line] = indented_lines
            
            # Write back to file
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
    """Pure test executor - supports unittest and pytest"""
    
    def __init__(self, source_dir: str, log_dir: str = None):
        self.source_dir = source_dir
        self.log_dir = log_dir
        
        # Create log directory
        if self.log_dir and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
    
    def _detect_test_framework(self, test_file_path: str) -> str:
        """Detect test framework used by test file
        
        Args:
            test_file_path: Full path to test file
            
        Returns:
            'pytest' or 'unittest'
        """
        try:
            with open(test_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check if pytest is imported
            if 'import pytest' in content or 'from pytest' in content:
                return 'pytest'
            
            # Check if pytest decorators are used
            if '@pytest.' in content:
                return 'pytest'
            
            # Check if pytest fixtures are used
            if 'def test_' in content and ('@pytest.fixture' in content or 'pytest.raises' in content):
                return 'pytest'
            
            # Default to unittest
            return 'unittest'
            
        except Exception as e:
            print(f"    Warning: Failed to detect test framework, defaulting to unittest: {e}")
            return 'unittest'
    
    def run_test_file(self, test_file_path: str, function_name: str = "") -> Tuple[bool, float]:
        """Run test file - auto-detect unittest or pytest
        
        Args:
            test_file_path: Test file path
            function_name: Function name (for generating log file name)
        
        Returns:
            Tuple[bool, float]: (whether all passed, pass ratio)
        """
        try:
            full_test_path = os.path.join(self.source_dir, test_file_path)
            if not os.path.exists(full_test_path):
                print(f"Test file not found: {full_test_path}")
                return False, 0.0
            
            # Detect test framework
            framework = self._detect_test_framework(full_test_path)
            print(f"    🔍 Detected test framework: {framework}")
            
            # Choose run command based on framework
            if framework == 'pytest':
                # Use pytest to run, -v for verbose, --tb=short for traceback
                cmd = [sys.executable, '-m', 'pytest', full_test_path, '-v', '--tb=short']
            else:
                # Use unittest method (directly execute file)
                cmd = [sys.executable, full_test_path]
            
            # Run test
            result = subprocess.run(
                cmd,
                cwd=self.source_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Save log file
            if self.log_dir:
                self._save_test_log(result, test_file_path, function_name, framework)
            
            # Check return code (note: even if all tests are skipped, return code is 0)
            returncode_ok = result.returncode == 0
            
            # Parse test results, pass framework type for correct parsing
            pass_ratio = self._parse_test_output(result.stdout, result.stderr, returncode_ok, framework)
            
            # Determine if truly all passed: pass ratio must be > 0 (exclude all skipped case)
            all_passed = returncode_ok and pass_ratio > 0.0
            
            return all_passed, pass_ratio
            
        except subprocess.TimeoutExpired as e:
            print(f"Test timeout")
            # Save timeout log
            if self.log_dir:
                self._save_timeout_log(test_file_path, function_name)
            return False, 0.0
        except Exception as e:
            print(f"Error running test: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0
    
    def _save_test_log(self, result: subprocess.CompletedProcess, test_file_path: str, function_name: str, framework: str = "unknown"):
        """Save test run log
        
        Args:
            result: subprocess run result
            test_file_path: Test file path
            function_name: Function name
            framework: Test framework type (pytest/unittest)
        """
        try:
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Precise to milliseconds
            
            # Generate log file name: function_name_timestamp.log
            if function_name:
                # Clean special characters from function name
                safe_function_name = re.sub(r'[^\w\-_.]', '_', function_name)
                log_filename = f"{safe_function_name}_{timestamp}.log"
            else:
                test_basename = os.path.basename(test_file_path).replace('.py', '')
                log_filename = f"{test_basename}_{timestamp}.log"
            
            log_filepath = os.path.join(self.log_dir, log_filename)
            
            # Write log content
            with open(log_filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Function name: {function_name}\n")
                f.write(f"Test file: {test_file_path}\n")
                f.write(f"Test framework: {framework}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("【STDOUT】\n")
                f.write("-" * 80 + "\n")
                f.write(result.stdout if result.stdout else "(no output)\n")
                f.write("\n")
                
                f.write("【STDERR】\n")
                f.write("-" * 80 + "\n")
                f.write(result.stderr if result.stderr else "(no errors)\n")
                f.write("\n")
                
                f.write("=" * 80 + "\n")
                f.write("Log end\n")
            
            print(f"    📝 Log saved: {log_filename}")
            
        except Exception as e:
            print(f"    ⚠️  Failed to save log: {e}")
    
    def _save_timeout_log(self, test_file_path: str, function_name: str):
        """Save timeout log"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe_function_name = re.sub(r'[^\w\-_.]', '_', function_name) if function_name else 'unknown'
            log_filename = f"{safe_function_name}_{timestamp}_TIMEOUT.log"
            log_filepath = os.path.join(self.log_dir, log_filename)
            
            with open(log_filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Function name: {function_name}\n")
                f.write(f"Test file: {test_file_path}\n")
                f.write(f"Status: Timeout (TIMEOUT)\n")
                f.write("=" * 80 + "\n")
            
            print(f"    📝 Timeout log saved: {log_filename}")
        except Exception as e:
            print(f"    ⚠️  Failed to save timeout log: {e}")
    
    def _parse_test_output(self, stdout: str, stderr: str, all_passed: bool, framework: str = "unittest") -> float:
        """Parse test output, extract pass ratio
        
        Supports output formats from multiple test frameworks:
        - unittest: "Ran X tests", "FAILED (failures=Y)"
        - pytest: "X passed", "Y failed", "X passed in Xs"
        - Character statistics: prioritize counting E/F/. characters (most accurate)
        
        Note: Skipped tests are not counted as passed
        
        Args:
            stdout: Standard output
            stderr: Standard error output
            all_passed: Whether return code is 0
            framework: Test framework type (pytest/unittest)
        
        Returns:
            float: Ratio of passed test cases (0.0-1.0)
        """
        combined_output = stdout + stderr
        
        # If pytest, prioritize pytest-specific parsing method
        if framework == 'pytest':
            ratio = self._parse_pytest_output(combined_output, all_passed)
            if ratio is not None:
                return ratio
        
        # Check if all tests are skipped
        # Example: "Ran 11 tests in 0.000s\n\nOK (skipped=11)"
        import re
        ran_match = re.search(r'Ran (\d+) test', combined_output)
        skipped_match = re.search(r'skipped=(\d+)', combined_output)
        
        if ran_match and skipped_match:
            total_tests = int(ran_match.group(1))
            skipped_tests = int(skipped_match.group(1))
            
            # If all tests are skipped, return 0.0
            if total_tests == skipped_tests:
                return 0.0
        
        # If all passed (and not all skipped), return 1.0
        if all_passed:
            # Check again if there are skipped tests
            if skipped_match and ran_match:
                total_tests = int(ran_match.group(1))
                skipped_tests = int(skipped_match.group(1))
                # If partially skipped, calculate actual pass ratio
                if skipped_tests > 0 and total_tests > skipped_tests:
                    # Some tests passed, some skipped
                    passed_tests = total_tests - skipped_tests
                    return passed_tests / total_tests
            return 1.0
        
        # Method 1: Prioritize character statistics (unittest real-time output E/F/.)
        # This is the most accurate method, as each test outputs one character
        error_count = combined_output.count('E')
        failure_count = combined_output.count('F')
        passed_count = combined_output.count('.')
        
        # If character statistics markers are found, prioritize this
        if error_count > 0 or failure_count > 0 or passed_count > 0:
            total_from_chars = error_count + failure_count + passed_count
            if total_from_chars > 0:
                # Validate: Check if character statistics are reasonable (avoid matching characters in code)
                # Usually test output E/F/. appear consecutively
                char_pattern = re.search(r'[EF.]{3,}', combined_output)
                if char_pattern:
                    # Only count consecutively appearing test marker characters
                    test_markers = char_pattern.group(0)
                    error_count = test_markers.count('E')
                    failure_count = test_markers.count('F')
                    passed_count = test_markers.count('.')
                    total_from_chars = len(test_markers)
                    
                    pass_ratio = passed_count / total_from_chars
                    return max(0.0, min(1.0, pass_ratio))
        
        # Method 2: Parse unittest text format
        # Example: "Ran 5 tests in 0.001s" and "FAILED (failures=2, errors=1, skipped=1)"
        ran_match = re.search(r'Ran (\d+) test', combined_output)
        if ran_match:
            total_tests = int(ran_match.group(1))
            
            # Match failure count, error count, skipped count
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
                # Passed tests = total - failures - errors - skipped
                # Skipped tests should not count as passed
                passed_tests = total_tests - failures - errors - skipped
                
                # Ensure pass ratio is in [0.0, 1.0] range
                passed_tests = max(0, min(passed_tests, total_tests))
                return passed_tests / total_tests
        
        # Try to parse pytest format
        # Example: "3 passed, 2 failed in 0.12s"
        pytest_match = re.search(r'(\d+) passed(?:, (\d+) failed)?', combined_output)
        if pytest_match:
            passed = int(pytest_match.group(1))
            failed = int(pytest_match.group(2)) if pytest_match.group(2) else 0
            total = passed + failed
            if total > 0:
                # Ensure pass ratio is in [0.0, 1.0] range
                pass_ratio = passed / total
                return max(0.0, min(1.0, pass_ratio))
        
        # If cannot parse but test failed, return 0.0
        return 0.0
    
    def _parse_pytest_output(self, output: str, all_passed: bool) -> float:
        """Parse pytest output format
        
        Pytest output examples:
        - "3 passed in 0.12s"
        - "1 failed, 2 passed in 0.12s"
        - "3 passed, 1 skipped in 0.12s"
        - "collected 3 items" ... "test_file.py::test_name PASSED"
        
        Args:
            output: pytest output
            all_passed: Whether return code is 0
            
        Returns:
            float or None: Pass ratio, returns None if cannot parse
        """
        import re
        
        # Method 1: Parse final summary line "X passed, Y failed in Zs"
        # Match "5 passed in 0.12s" or "2 failed, 3 passed in 0.12s"
        summary_pattern = r'(?:(\d+) failed)?(?:, )?(?:(\d+) passed)?(?:, )?(?:(\d+) skipped)?(?:, )?(?:(\d+) error)?.*in ([\d.]+)s'
        summary_match = re.search(summary_pattern, output)
        
        if summary_match:
            failed = int(summary_match.group(1)) if summary_match.group(1) else 0
            passed = int(summary_match.group(2)) if summary_match.group(2) else 0
            skipped = int(summary_match.group(3)) if summary_match.group(3) else 0
            errors = int(summary_match.group(4)) if summary_match.group(4) else 0
            
            # Calculate total test count (excluding skipped)
            total = passed + failed + errors
            
            # If all tests are skipped
            if total == 0 and skipped > 0:
                return 0.0
            
            if total > 0:
                return passed / total
        
        # Method 2: Count test result markers PASSED / FAILED / ERROR / SKIPPED
        passed_count = len(re.findall(r'\bPASSED\b', output))
        failed_count = len(re.findall(r'\bFAILED\b', output))
        error_count = len(re.findall(r'\bERROR\b', output))
        skipped_count = len(re.findall(r'\bSKIPPED\b', output))
        
        total_from_markers = passed_count + failed_count + error_count
        
        if total_from_markers > 0:
            return passed_count / total_from_markers
        
        # Method 3: Check collected line
        # Example: "collected 3 items"
        collected_match = re.search(r'collected (\d+) items?', output)
        if collected_match:
            total_tests = int(collected_match.group(1))
            
            # If returncode is 0 and no failure info found, all passed
            if all_passed and total_tests > 0:
                # Check if all are skipped
                if skipped_count == total_tests:
                    return 0.0
                return 1.0
        
        # If return code is 0 and found passed, consider all passed
        if all_passed and 'passed' in output.lower():
            return 1.0
        
        # Cannot parse
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
    """Result collector"""
    
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
        """Calculate average pass ratio - more fine-grained metric
        
        For each task (task_id/function_name), calculate the average of pass ratios of all its samples
        Then calculate the average of all tasks' average pass ratios
        
        Args:
            data_records: List of records containing test results
            
        Returns:
            Average pass ratio (0.0-1.0)
        """
        group = {}
        for record in data_records:
            # Use function_name as task_id
            task_id = record.get('function_name', '')
            if not task_id:
                continue
            
            if task_id not in group:
                group[task_id] = []
                
            # Prioritize pass_ratios field (new version fine-grained data)
            if 'pass_ratios' in record and record['pass_ratios']:
                group[task_id].extend(record['pass_ratios'])
            # Then use passed field (compatible with old data format)
            elif 'passed' in record:
                group[task_id].append(record['passed'])
            # Finally use results field (boolean list), calculate pass ratio
            elif 'results' in record and record['results']:
                # Calculate pass ratio for this completion
                pass_ratio = sum(record['results']) / len(record['results']) if len(record['results']) > 0 else 0.0
                group[task_id].append(pass_ratio)
        
        if not group:
            return 0.0
        
        # Calculate average pass ratio for each task_id
        task_avg_pass_ratios = []
        for task_id, pass_ratios in group.items():
            task_avg = np.mean(pass_ratios)
            task_avg_pass_ratios.append(task_avg)
        
        # Return average pass ratio of all tasks
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
    
    # Create log directory (same directory as output file)
    output_dir = os.path.dirname(args.output_file)
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    print(f"📁 Log directory: {log_dir}")
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
        
        # Backup original file
        backup_path = source_file_path + '.backup'
        shutil.copy2(source_file_path, backup_path)
        
        results = []
        pass_ratios = []  # New: record pass ratio for each completion
        
        try:
            for j, completion in enumerate(record['completions']):
                print(f"  Testing completion {j+1}/{len(record['completions'])}")
                
                try:
                    # 1. Modify source file
                    success = code_replacer.replace_function_in_file(
                        source_file_path,
                        record['implementation_location'],
                        completion
                    )
                    
                    if not success:
                        results.append(False)
                        pass_ratios.append(0.0)
                        print(f"    Result: FAIL (replace failed)")
                        # Restore file
                        shutil.copy2(backup_path, source_file_path)
                        continue
                    
                    # 2. Run test
                    test_passed, pass_ratio = test_executor.run_test_file(test_code_path, record['function_name'])
                    
                    results.append(test_passed)
                    pass_ratios.append(pass_ratio)
                    print(f"    Result: {'PASS' if test_passed else 'FAIL'} (pass ratio: {pass_ratio:.2%})")
                    
                    # 3. Restore file (prepare for next completion)
                    shutil.copy2(backup_path, source_file_path)
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    results.append(False)
                    pass_ratios.append(0.0)
                    # Ensure file is restored
                    shutil.copy2(backup_path, source_file_path)
        
        finally:
            # Ensure file is restored at the end
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, source_file_path)
                os.remove(backup_path)
        
        record['results'] = results
        record['pass_ratios'] = pass_ratios  # New: save pass ratio information
    
    print(f"Saving results to {args.output_file}...")
    result_collector.save_jsonl_data(data_records, args.output_file)
    
    # Calculate metrics
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
