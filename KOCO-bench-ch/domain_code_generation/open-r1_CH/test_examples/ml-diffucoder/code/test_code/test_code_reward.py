import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the actual function we want to test
from open_r1.rewards import code_reward, extract_code


class TestCodeReward(unittest.TestCase):
    """测试code_reward函数 - 评估生成的代码并计算测试用例通过率"""
    
    def setUp(self):
        pass
    
    def test_code_reward_basic_structure(self):
        """测试基本输出结构"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef add(a, b):\n    return a + b\n```\n"}],
            [{"role": "assistant", "content": "```python\ndef multiply(a, b):\n    return a * b\n```\n"}]
        ]
        
        test_cases = [
            ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
            ["assert multiply(2, 3) == 6", "assert multiply(0, 5) == 0"]
        ]
        
        # Mock get_code_format_reward
        with patch('open_r1.rewards.get_code_format_reward') as mock_format:
            mock_format.return_value = lambda completions, **kwargs: [1.0, 1.0]
            
            # Mock get_provider
            with patch('open_r1.rewards.get_provider') as mock_provider:
                mock_executor = MagicMock()
                mock_executor.execute_scripts = lambda scripts, langs: [0.5, 1.0]
                mock_provider.return_value = mock_executor
                
                rewards = code_reward(completions, test_cases=test_cases)
        
        # 验证输出
        self.assertEqual(len(rewards), len(completions))
        self.assertTrue(all(isinstance(r, float) for r in rewards))
    
    def test_code_reward_format_check(self):
        """测试格式检查"""
        # 格式正确的代码
        good_completion = [{"role": "assistant", "content": "```python\ndef test():\n    pass\n```\n"}]
        
        # 格式错误的代码（没有代码块标记）
        bad_completion = [{"role": "assistant", "content": "def test():\n    pass"}]
        
        completions = [good_completion, bad_completion]
        test_cases = [["assert True"], ["assert True"]]
        
        with patch('open_r1.rewards.get_code_format_reward') as mock_format:
            # 第一个格式正确(1.0)，第二个格式错误(0.0)
            mock_format.return_value = lambda completions, **kwargs: [1.0, 0.0]
            
            with patch('open_r1.rewards.get_provider') as mock_provider:
                mock_executor = MagicMock()
                # 只有格式正确的会被执行
                mock_executor.execute_scripts = lambda scripts, langs: [1.0]
                mock_provider.return_value = mock_executor
                
                rewards = code_reward(completions, test_cases=test_cases)
        
        # 验证：格式正确的应该有奖励，格式错误的应该是0.0
        self.assertGreater(rewards[0], 0.0)
        self.assertEqual(rewards[1], 0.0)
    
    def test_code_reward_code_extraction(self):
        """测试代码提取"""
        completion_with_code = "```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n```\n"
        
        extracted = extract_code(completion_with_code, "python")
        
        # 验证提取的代码
        self.assertIn("def factorial", extracted)
        self.assertNotIn("```", extracted)
    
    def test_code_reward_test_case_execution(self):
        """测试测试用例执行"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef is_even(n):\n    return n % 2 == 0\n```\n"}]
        ]
        
        test_cases = [
            ["assert is_even(2) == True", "assert is_even(3) == False", "assert is_even(0) == True"]
        ]
        
        with patch('open_r1.rewards.get_code_format_reward') as mock_format:
            mock_format.return_value = lambda completions, **kwargs: [1.0]
            
            with patch('open_r1.rewards.get_provider') as mock_provider:
                mock_executor = MagicMock()
                # 模拟全部通过
                mock_executor.execute_scripts = lambda scripts, langs: [1.0]
                mock_provider.return_value = mock_executor
                
                rewards = code_reward(completions, test_cases=test_cases)
        
        # 验证奖励
        self.assertEqual(rewards[0], 1.0)
    
    def test_code_reward_partial_pass(self):
        """测试部分测试用例通过的情况"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef buggy_add(a, b):\n    return a + b + 1\n```\n"}]
        ]
        
        test_cases = [
            ["assert buggy_add(1, 1) == 2", "assert buggy_add(0, 0) == 0"]
        ]
        
        with patch('open_r1.rewards.get_code_format_reward') as mock_format:
            mock_format.return_value = lambda completions, **kwargs: [1.0]
            
            with patch('open_r1.rewards.get_provider') as mock_provider:
                mock_executor = MagicMock()
                # 模拟50%通过率
                mock_executor.execute_scripts = lambda scripts, langs: [0.5]
                mock_provider.return_value = mock_executor
                
                rewards = code_reward(completions, test_cases=test_cases)
        
        # 验证部分通过的奖励
        self.assertEqual(rewards[0], 0.5)
        self.assertGreater(rewards[0], 0.0)
        self.assertLess(rewards[0], 1.0)
    
    def test_code_reward_provider_types(self):
        """测试不同的执行提供者类型"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef test():\n    pass\n```\n"}]
        ]
        test_cases = [["assert True"]]
        
        # 测试e2b provider
        with patch('open_r1.rewards.get_code_format_reward') as mock_format:
            mock_format.return_value = lambda completions, **kwargs: [1.0]
            
            with patch('open_r1.rewards.get_provider') as mock_provider:
                mock_executor = MagicMock()
                mock_executor.execute_scripts = lambda scripts, langs: [1.0]
                mock_provider.return_value = mock_executor
                
                rewards_e2b = code_reward(completions, provider_type="e2b", test_cases=test_cases)
                
                # 验证provider被正确调用
                mock_provider.assert_called_once()
                call_kwargs = mock_provider.call_args[1]
                self.assertEqual(call_kwargs['provider_type'], "e2b")
        
        # 测试local provider
        with patch('open_r1.rewards.get_code_format_reward') as mock_format:
            mock_format.return_value = lambda completions, **kwargs: [1.0]
            
            with patch('open_r1.rewards.get_provider') as mock_provider:
                mock_executor = MagicMock()
                mock_executor.execute_scripts = lambda scripts, langs: [1.0]
                mock_provider.return_value = mock_executor
                
                rewards_local = code_reward(completions, provider_type="local", test_cases=test_cases)
                
                call_kwargs = mock_provider.call_args[1]
                self.assertEqual(call_kwargs['provider_type'], "local")
    
    def test_code_reward_parallel_execution(self):
        """测试并行执行"""
        completions = [
            [{"role": "assistant", "content": f"```python\ndef func{i}():\n    return {i}\n```\n"}]
            for i in range(4)
        ]
        test_cases = [[f"assert func{i}() == {i}"] for i in range(4)]
        
        with patch('open_r1.rewards.get_code_format_reward') as mock_format:
            mock_format.return_value = lambda completions, **kwargs: [1.0] * len(completions)
            
            with patch('open_r1.rewards.get_provider') as mock_provider:
                mock_executor = MagicMock()
                mock_executor.execute_scripts = lambda scripts, langs: [1.0] * len(scripts)
                mock_provider.return_value = mock_executor
                
                rewards = code_reward(completions, num_parallel=2, test_cases=test_cases)
        
        # 验证所有样本都被处理
        self.assertEqual(len(rewards), 4)
    
    def test_code_reward_evaluation_script_template(self):
        """测试评估脚本模板的使用"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef square(x):\n    return x * x\n```\n"}]
        ]
        test_cases = [["assert square(2) == 4", "assert square(3) == 9"]]
        
        captured_scripts = []
        
        with patch('open_r1.rewards.get_code_format_reward') as mock_format:
            mock_format.return_value = lambda completions, **kwargs: [1.0]
            
            with patch('open_r1.rewards.get_provider') as mock_provider:
                mock_executor = MagicMock()
                
                def capture_scripts(scripts, langs):
                    captured_scripts.extend(scripts)
                    return [1.0] * len(scripts)
                
                mock_executor.execute_scripts = capture_scripts
                mock_provider.return_value = mock_executor
                
                rewards = code_reward(completions, test_cases=test_cases)
        
        # 验证脚本包含必要的组件
        self.assertEqual(len(captured_scripts), 1)
        script = captured_scripts[0]
        
        # 脚本应该包含代码和测试用例
        self.assertIn("def square", script)
        self.assertIn("assert square(2) == 4", script)
        self.assertIn("__PASS_RATE__", script)
    
    def test_code_reward_edge_cases(self):
        """测试边缘情况"""
        # 测试1: 空代码块
        empty_completion = [{"role": "assistant", "content": "```python\n\n```\n"}]
        
        with patch('open_r1.rewards.get_code_format_reward') as mock_format:
            mock_format.return_value = lambda completions, **kwargs: [0.5]  # 格式正确但语法可能有问题
            
            with patch('open_r1.rewards.get_provider') as mock_provider:
                mock_executor = MagicMock()
                mock_executor.execute_scripts = lambda scripts, langs: [0.0]
                mock_provider.return_value = mock_executor
                
                rewards = code_reward([empty_completion], test_cases=[["assert True"]])
        
        self.assertIsInstance(rewards[0], float)
        
        # 测试2: 多个代码块（只提取第一个）
        multi_block = [{"role": "assistant", "content": 
                       "```python\ndef func1():\n    pass\n```\n```python\ndef func2():\n    pass\n```\n"}]
        
        extracted = extract_code(multi_block[0]["content"], "python")
        self.assertIn("func1", extracted)
        # 第二个代码块不应该被提取
        self.assertNotIn("func2", extracted)
    
    def test_code_reward_output_parsing(self):
        """测试输出解析"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef test():\n    return True\n```\n"}]
        ]
        test_cases = [["assert test() == True"]]
        
        with patch('open_r1.rewards.get_code_format_reward') as mock_format:
            mock_format.return_value = lambda completions, **kwargs: [1.0]
            
            with patch('open_r1.rewards.get_provider') as mock_provider:
                mock_executor = MagicMock()
                # 模拟执行器返回通过率
                mock_executor.execute_scripts = lambda scripts, langs: [0.75]
                mock_provider.return_value = mock_executor
                
                rewards = code_reward(completions, test_cases=test_cases)
        
        # 验证通过率被正确解析
        self.assertEqual(rewards[0], 0.75)


class TestExtractCode(unittest.TestCase):
    """测试extract_code函数 - 从完成中提取代码块"""
    
    def test_extract_code_basic(self):
        """测试基本代码提取"""
        completion = "```python\ndef hello():\n    print('Hello')\n```"
        
        extracted = extract_code(completion, "python")
        
        self.assertEqual(extracted, "def hello():\n    print('Hello')")
    
    def test_extract_code_with_surrounding_text(self):
        """测试带有周围文本的代码提取"""
        completion = "Here is the solution:\n```python\ndef solution():\n    return 42\n```\nThis works!"
        
        extracted = extract_code(completion, "python")
        
        self.assertEqual(extracted, "def solution():\n    return 42")
        self.assertNotIn("Here is", extracted)
        self.assertNotIn("This works", extracted)
    
    def test_extract_code_multiple_blocks(self):
        """测试多个代码块（只提取第一个）"""
        completion = "```python\ncode1\n```\nSome text\n```python\ncode2\n```"
        
        extracted = extract_code(completion, "python")
        
        self.assertEqual(extracted, "code1")
    
    def test_extract_code_no_code_block(self):
        """测试没有代码块的情况"""
        completion = "This is just text without code blocks"
        
        extracted = extract_code(completion, "python")
        
        self.assertEqual(extracted, "")
    
    def test_extract_code_different_languages(self):
        """测试不同编程语言"""
        # Python
        python_completion = "```python\nprint('hello')\n```"
        self.assertEqual(extract_code(python_completion, "python"), "print('hello')")
        
        # JavaScript
        js_completion = "```javascript\nconsole.log('hello');\n```"
        self.assertEqual(extract_code(js_completion, "javascript"), "console.log('hello');")
        
        # C++
        cpp_completion = "```cpp\nstd::cout << \"hello\";\n```"
        self.assertEqual(extract_code(cpp_completion, "cpp"), "std::cout << \"hello\";")
    
    def test_extract_code_empty_block(self):
        """测试空代码块"""
        completion = "```python\n\n```"
        
        extracted = extract_code(completion, "python")
        
        self.assertEqual(extracted, "")
    
    def test_extract_code_with_newlines(self):
        """测试包含多行的代码"""
        completion = """```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
```"""
        
        extracted = extract_code(completion, "python")
        
        self.assertIn("def factorial", extracted)
        self.assertIn("if n <= 1:", extracted)
        self.assertIn("return n * factorial(n-1)", extracted)


class TestCodeRewardIntegration(unittest.TestCase):
    """集成测试：测试code_reward的完整工作流程"""
    
    def test_end_to_end_code_evaluation(self):
        """端到端测试：完整的代码评估流程"""
        # 创建一个完整的测试场景
        completions = [
            [{"role": "assistant", "content": "```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```\n"}],
            [{"role": "assistant", "content": "```python\ndef fibonacci(n):\n    return n * 2\n```\n"}],  # 错误实现
        ]
        
        test_cases = [
            ["assert fibonacci(0) == 0", "assert fibonacci(1) == 1", "assert fibonacci(5) == 5"],
            ["assert fibonacci(0) == 0", "assert fibonacci(1) == 1", "assert fibonacci(5) == 5"],
        ]
        
        with patch('open_r1.rewards.get_code_format_reward') as mock_format:
            # 两个都格式正确
            mock_format.return_value = lambda completions, **kwargs: [1.0, 1.0]
            
            with patch('open_r1.rewards.get_provider') as mock_provider:
                mock_executor = MagicMock()
                # 第一个全部通过，第二个部分通过
                mock_executor.execute_scripts = lambda scripts, langs: [1.0, 0.33]
                mock_provider.return_value = mock_executor
                
                rewards = code_reward(completions, test_cases=test_cases)
        
        # 验证：正确实现应该有更高的奖励
        self.assertGreater(rewards[0], rewards[1])
        self.assertEqual(rewards[0], 1.0)
        self.assertLess(rewards[1], 1.0)
    
    def test_format_and_execution_interaction(self):
        """测试格式检查和执行的交互"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef good():\n    return True\n```\n"}],  # 格式和语法都正确
            [{"role": "assistant", "content": "def bad():\n    return True"}],  # 格式错误
            [{"role": "assistant", "content": "```python\ndef syntax_error(\n```\n"}],  # 格式正确但语法错误
        ]
        
        test_cases = [
            ["assert good() == True"],
            ["assert bad() == True"],
            ["assert syntax_error() == True"],
        ]
        
        with patch('open_r1.rewards.get_code_format_reward') as mock_format:
            # 第一个1.0（完全正确），第二个0.0（格式错误），第三个0.5（语法错误）
            mock_format.return_value = lambda completions, **kwargs: [1.0, 0.0, 0.5]
            
            with patch('open_r1.rewards.get_provider') as mock_provider:
                mock_executor = MagicMock()
                # 只有格式正确的会被执行
                mock_executor.execute_scripts = lambda scripts, langs: [1.0, 0.0]
                mock_provider.return_value = mock_executor
                
                rewards = code_reward(completions, test_cases=test_cases)
        
        # 验证奖励分配
        self.assertEqual(rewards[0], 1.0)  # 完全正确
        self.assertEqual(rewards[1], 0.0)  # 格式错误
        self.assertEqual(rewards[2], 0.0)  # 语法错误也应该是0（因为无法执行）


if __name__ == "__main__":
    unittest.main()
