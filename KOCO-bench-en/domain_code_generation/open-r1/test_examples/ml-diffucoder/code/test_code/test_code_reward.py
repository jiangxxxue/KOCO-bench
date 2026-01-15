import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the actual function we want to test
from open_r1.rewards import code_reward, extract_code


class TestCodeReward(unittest.TestCase):
    """Test code_reward function - evaluate generated code and calculate test case pass rate"""
    
    def setUp(self):
        pass
    
    def test_code_reward_basic_structure(self):
        """Test basic output structure"""
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
        
        # Verify output

        # print(rewards)

        expected_rewards = [0.5, 1.0]

        self.assertEqual(rewards, expected_rewards)
        self.assertEqual(len(rewards), len(completions))
        self.assertTrue(all(isinstance(r, float) for r in rewards))
    
    def test_code_reward_format_check(self):
        """Test format checking"""
        # Correctly formatted code
        good_completion = [{"role": "assistant", "content": "```python\ndef test():\n    pass\n```\n"}]
        
        # Incorrectly formatted code (no code block markers)
        bad_completion = [{"role": "assistant", "content": "def test():\n    pass"}]
        
        completions = [good_completion, bad_completion]
        test_cases = [["assert True"], ["assert True"]]
        
        with patch('open_r1.rewards.get_code_format_reward') as mock_format:
            # First one correct format (1.0), second one incorrect format (0.0)
            mock_format.return_value = lambda completions, **kwargs: [1.0, 0.0]
            
            with patch('open_r1.rewards.get_provider') as mock_provider:
                mock_executor = MagicMock()
                # Only correctly formatted code will be executed
                mock_executor.execute_scripts = lambda scripts, langs: [1.0]
                mock_provider.return_value = mock_executor
                
                rewards = code_reward(completions, test_cases=test_cases)
        
        # Verify: correctly formatted should have reward, incorrectly formatted should be 0.0
        self.assertGreater(rewards[0], 0.0)
        self.assertEqual(rewards[1], 0.0)
    
    def test_code_reward_code_extraction(self):
        """Test code extraction"""
        completion_with_code = "```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n```\n"
        
        extracted = extract_code(completion_with_code, "python")
        
        # Verify extracted code
        self.assertIn("def factorial", extracted)
        self.assertNotIn("```", extracted)
    
    def test_code_reward_test_case_execution(self):
        """Test test case execution"""
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
                # Mock all pass
                mock_executor.execute_scripts = lambda scripts, langs: [1.0]
                mock_provider.return_value = mock_executor
                
                rewards = code_reward(completions, test_cases=test_cases)
        
        # Verify reward
        self.assertEqual(rewards[0], 1.0)
    
    def test_code_reward_partial_pass(self):
        """Test partial test case pass"""
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
                # Mock 50% pass rate
                mock_executor.execute_scripts = lambda scripts, langs: [0.5]
                mock_provider.return_value = mock_executor
                
                rewards = code_reward(completions, test_cases=test_cases)
        
        # Verify partial pass reward
        self.assertEqual(rewards[0], 0.5)
        self.assertGreater(rewards[0], 0.0)
        self.assertLess(rewards[0], 1.0)
    
    def test_code_reward_provider_types(self):
        """Test different provider types"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef test():\n    pass\n```\n"}]
        ]
        test_cases = [["assert True"]]
        
        # Test e2b provider
        with patch('open_r1.rewards.get_code_format_reward') as mock_format:
            mock_format.return_value = lambda completions, **kwargs: [1.0]
            
            with patch('open_r1.rewards.get_provider') as mock_provider:
                mock_executor = MagicMock()
                mock_executor.execute_scripts = lambda scripts, langs: [1.0]
                mock_provider.return_value = mock_executor
                
                rewards_e2b = code_reward(completions, provider_type="e2b", test_cases=test_cases)
                
                # Verify provider is called correctly
                mock_provider.assert_called_once()
                call_kwargs = mock_provider.call_args[1]
                self.assertEqual(call_kwargs['provider_type'], "e2b")
        
        # Test local provider
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
        """Test parallel execution"""
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
        
        # print(rewards)
        # Verify all samples are processed
        self.assertEqual(len(rewards), 4)

        expected_rewards = [1.0, 1.0, 1.0, 1.0]

        self.assertEqual(rewards, expected_rewards)
    
    def test_code_reward_evaluation_script_template(self):
        """Test evaluation script template usage"""
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
        
        # Verify script contains necessary components
        self.assertEqual(len(captured_scripts), 1)
        script = captured_scripts[0]
        
        # Script should contain code and test cases
        self.assertIn("def square", script)
        self.assertIn("assert square(2) == 4", script)
        self.assertIn("__PASS_RATE__", script)
    
    def test_code_reward_edge_cases(self):
        """Test edge cases"""
        # Test 1: Empty code block
        empty_completion = [{"role": "assistant", "content": "```python\n\n```\n"}]
        
        with patch('open_r1.rewards.get_code_format_reward') as mock_format:
            mock_format.return_value = lambda completions, **kwargs: [0.5]  # Correct format but may have syntax issues
            
            with patch('open_r1.rewards.get_provider') as mock_provider:
                mock_executor = MagicMock()
                mock_executor.execute_scripts = lambda scripts, langs: [0.0]
                mock_provider.return_value = mock_executor
                
                rewards = code_reward([empty_completion], test_cases=[["assert True"]])
        
        self.assertIsInstance(rewards[0], float)
        
        # Test 2: Multiple code blocks (only extract first one)
        multi_block = [{"role": "assistant", "content": 
                       "```python\ndef func1():\n    pass\n```\n```python\ndef func2():\n    pass\n```\n"}]
        
        extracted = extract_code(multi_block[0]["content"], "python")
        self.assertIn("func1", extracted)
        # Second code block should not be extracted
        self.assertNotIn("func2", extracted)
    
    def test_code_reward_output_parsing(self):
        """Test output parsing"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef test():\n    return True\n```\n"}]
        ]
        test_cases = [["assert test() == True"]]
        
        with patch('open_r1.rewards.get_code_format_reward') as mock_format:
            mock_format.return_value = lambda completions, **kwargs: [1.0]
            
            with patch('open_r1.rewards.get_provider') as mock_provider:
                mock_executor = MagicMock()
                # Mock executor returns pass rate
                mock_executor.execute_scripts = lambda scripts, langs: [0.75]
                mock_provider.return_value = mock_executor
                
                rewards = code_reward(completions, test_cases=test_cases)
        
        # Verify pass rate is correctly parsed
        self.assertEqual(rewards[0], 0.75)


class TestExtractCode(unittest.TestCase):
    """Test extract_code function - extract code blocks from completion"""
    
    def test_extract_code_basic(self):
        """Test basic code extraction"""
        completion = "```python\ndef hello():\n    print('Hello')\n```"
        
        extracted = extract_code(completion, "python")
        
        self.assertEqual(extracted, "def hello():\n    print('Hello')")
    
    def test_extract_code_with_surrounding_text(self):
        """Test code extraction with surrounding text"""
        completion = "Here is the solution:\n```python\ndef solution():\n    return 42\n```\nThis works!"
        
        extracted = extract_code(completion, "python")
        
        self.assertEqual(extracted, "def solution():\n    return 42")
        self.assertNotIn("Here is", extracted)
        self.assertNotIn("This works", extracted)
    
    def test_extract_code_multiple_blocks(self):
        """Test multiple code blocks (only extract first one)"""
        completion = "```python\ncode1\n```\nSome text\n```python\ncode2\n```"
        
        extracted = extract_code(completion, "python")
        
        self.assertEqual(extracted, "code1")
    
    def test_extract_code_no_code_block(self):
        """Test no code block case"""
        completion = "This is just text without code blocks"
        
        extracted = extract_code(completion, "python")
        
        self.assertEqual(extracted, "")
    
    def test_extract_code_different_languages(self):
        """Test different programming languages"""
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
        """Test empty code block"""
        completion = "```python\n\n```"
        
        extracted = extract_code(completion, "python")
        
        self.assertEqual(extracted, "")
    
    def test_extract_code_with_newlines(self):
        """Test code with multiple lines"""
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
    """Integration test: test complete workflow of code_reward"""
    
    def test_end_to_end_code_evaluation(self):
        """End-to-end test: complete code evaluation workflow"""
        # Create a complete test scenario
        completions = [
            [{"role": "assistant", "content": "```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```\n"}],
            [{"role": "assistant", "content": "```python\ndef fibonacci(n):\n    return n * 2\n```\n"}],  # Incorrect implementation
        ]
        
        test_cases = [
            ["assert fibonacci(0) == 0", "assert fibonacci(1) == 1", "assert fibonacci(5) == 5"],
            ["assert fibonacci(0) == 0", "assert fibonacci(1) == 1", "assert fibonacci(5) == 5"],
        ]
        
        with patch('open_r1.rewards.get_code_format_reward') as mock_format:
            # Both have correct format
            mock_format.return_value = lambda completions, **kwargs: [1.0, 1.0]
            
            with patch('open_r1.rewards.get_provider') as mock_provider:
                mock_executor = MagicMock()
                # First one all pass, second one partial pass
                mock_executor.execute_scripts = lambda scripts, langs: [1.0, 0.33]
                mock_provider.return_value = mock_executor
                
                rewards = code_reward(completions, test_cases=test_cases)
        
        # print(rewards)

        expected_rewards = [1.0, 0.33]

        self.assertEqual(rewards, expected_rewards)
        # Verify: correct implementation should have higher reward
        # self.assertGreater(rewards[0], rewards[1])
        # self.assertEqual(rewards[0], 1.0)
        # self.assertLess(rewards[1], 1.0)
    
    def test_format_and_execution_interaction(self):
        """Test interaction between format checking and execution"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef good():\n    return True\n```\n"}],  # Both format and syntax correct
            [{"role": "assistant", "content": "def bad():\n    return True"}],  # Format error
            [{"role": "assistant", "content": "```python\ndef syntax_error(\n```\n"}],  # Correct format but syntax error
        ]
        
        test_cases = [
            ["assert good() == True"],
            ["assert bad() == True"],
            ["assert syntax_error() == True"],
        ]
        
        with patch('open_r1.rewards.get_code_format_reward') as mock_format:
            # First 1.0 (completely correct), second 0.0 (format error), third 0.5 (syntax error)
            mock_format.return_value = lambda completions, **kwargs: [1.0, 0.0, 0.5]
            
            with patch('open_r1.rewards.get_provider') as mock_provider:
                mock_executor = MagicMock()
                # Only correctly formatted will be executed
                mock_executor.execute_scripts = lambda scripts, langs: [1.0, 0.0]
                mock_provider.return_value = mock_executor
                
                rewards = code_reward(completions, test_cases=test_cases)
        
        # Verify reward allocation
        self.assertEqual(rewards[0], 1.0)  # Completely correct
        self.assertEqual(rewards[1], 0.0)  # Format error
        self.assertEqual(rewards[2], 0.0)  # Syntax error should also be 0 (cannot execute)


if __name__ == "__main__":
    unittest.main()
