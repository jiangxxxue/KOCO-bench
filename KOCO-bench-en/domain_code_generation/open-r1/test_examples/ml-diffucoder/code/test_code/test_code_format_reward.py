import unittest
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Import the actual function we want to test
from open_r1.rewards import get_code_format_reward


class TestCodeFormatReward(unittest.TestCase):
    """Test get_code_format_reward function - check code format and syntax"""
    
    def setUp(self):
        self.format_reward_func = get_code_format_reward(language="python")
    
    def test_code_format_reward_correct_format(self):
        """Test correctly formatted code"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef hello():\n    print('Hello')\n```\n"}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # Correct format and syntax should return 1.0
        self.assertEqual(rewards[0], 1.0)
    
    def test_code_format_reward_missing_code_block(self):
        """Test case with missing code block markers"""
        completions = [
            [{"role": "assistant", "content": "def hello():\n    print('Hello')"}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # No code block markers should return 0.0
        self.assertEqual(rewards[0], 0.0)
    
    def test_code_format_reward_syntax_error(self):
        """Test code with syntax errors"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef hello(\n    print('Hello')\n```\n"}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # Correct format but syntax error should return 0.5
        self.assertEqual(rewards[0], 0.5)
    
    def test_code_format_reward_extra_text_before(self):
        """Test extra text before code block"""
        completions = [
            [{"role": "assistant", "content": "Here is the code:\n```python\ndef test():\n    pass\n```\n"}]
        ]
        
        rewards = self.format_reward_func(completions)
        

        self.assertEqual(rewards[0], 1.0)
    
    def test_code_format_reward_extra_text_after(self):
        """Test extra text after code block"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef test():\n    pass\n```\nThis is the solution."}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # Text after code block (other than newlines) does not meet format requirements
        self.assertEqual(rewards[0], 0.0)
    
    def test_code_format_reward_multiple_code_blocks(self):
        """Test multiple code blocks"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef func1():\n    pass\n```\n```python\ndef func2():\n    pass\n```\n"}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # Multiple code blocks do not meet format requirements
        self.assertEqual(rewards[0], 0.0)
    
    def test_code_format_reward_empty_code_block(self):
        """Test empty code block"""
        completions = [
            [{"role": "assistant", "content": "```python\n\n```\n"}]
        ]
        
        rewards = self.format_reward_func(completions)

        self.assertEqual(rewards[0], 1.0)
    
    def test_code_format_reward_correct_format_complex_code(self):
        """Test correctly formatted complex code"""
        completions = [
            [{"role": "assistant", "content": """```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```
"""}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # Complex code with correct format and syntax should return 1.0
        self.assertEqual(rewards[0], 1.0)
    
    def test_code_format_reward_indentation_error(self):
        """Test indentation error"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef test():\npass\n```\n"}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # Indentation error is a syntax error, should return 0.5
        self.assertEqual(rewards[0], 0.5)
    
    def test_code_format_reward_incomplete_code(self):
        """Test incomplete code"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef test():\n    if True:\n```\n"}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # Incomplete code is a syntax error
        self.assertEqual(rewards[0], 0.5)
    
    def test_code_format_reward_wrong_language_marker(self):
        """Test wrong language marker"""
        completions = [
            [{"role": "assistant", "content": "```javascript\ndef test():\n    pass\n```\n"}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # Language marker mismatch, format is incorrect
        self.assertEqual(rewards[0], 0.0)
    
    def test_code_format_reward_batch_processing(self):
        """Test batch processing"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef func1():\n    return 1\n```\n"}],
            [{"role": "assistant", "content": "def func2():\n    return 2"}],
            [{"role": "assistant", "content": "```python\ndef func3(\n```\n"}],
            [{"role": "assistant", "content": "```python\ndef func4():\n    return 4\n```\n"}],
        ]
        
        rewards = self.format_reward_func(completions)
        
        # Verify batch processing results
        self.assertEqual(len(rewards), 4)
        self.assertEqual(rewards[0], 1.0)  # Correct
        self.assertEqual(rewards[1], 0.0)  # Format error
        self.assertEqual(rewards[2], 0.5)  # Syntax error
        self.assertEqual(rewards[3], 1.0)  # Correct
    
    def test_code_format_reward_with_imports(self):
        """Test code with import statements"""
        completions = [
            [{"role": "assistant", "content": "```python\nimport math\nimport sys\n\ndef calculate():\n    return math.sqrt(16)\n```\n"}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # Correct code with imports should return 1.0
        self.assertEqual(rewards[0], 1.0)
    
    def test_code_format_reward_with_classes(self):
        """Test code with class definitions"""
        completions = [
            [{"role": "assistant", "content": """```python
class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x):
        self.result += x
        return self.result
```
"""}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # Correct code with class definitions should return 1.0
        self.assertEqual(rewards[0], 1.0)
    
    def test_code_format_reward_with_comments(self):
        """Test code with comments"""
        completions = [
            [{"role": "assistant", "content": """```python
# This is a comment
def test():
    # Another comment
    return True  # Inline comment
```
"""}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # Correct code with comments should return 1.0
        self.assertEqual(rewards[0], 1.0)
    
    def test_code_format_reward_edge_cases(self):
        """Test edge cases"""
        # Test 1: Only code block markers, no content
        completions1 = [
            [{"role": "assistant", "content": "```python\n```\n"}]
        ]
        rewards1 = self.format_reward_func(completions1)
        self.assertEqual(rewards1[0], 1.0)
        
        # Test 2: Incomplete code block markers
        completions2 = [
            [{"role": "assistant", "content": "```python\ndef test():\n    pass"}]
        ]
        rewards2 = self.format_reward_func(completions2)
        self.assertEqual(rewards2[0], 0.0)
        
        # Test 3: Only one line of code
        completions3 = [
            [{"role": "assistant", "content": "```python\nprint('hello')\n```\n"}]
        ]
        rewards3 = self.format_reward_func(completions3)
        self.assertEqual(rewards3[0], 1.0)


class TestCodeFormatRewardDifferentLanguages(unittest.TestCase):
    """Test format checking for different programming languages"""
    
    def test_code_format_reward_python(self):
        """Test Python language"""
        format_reward = get_code_format_reward(language="python")
        
        completions = [
            [{"role": "assistant", "content": "```python\ndef test():\n    pass\n```\n"}]
        ]
        
        rewards = format_reward(completions)
        self.assertEqual(rewards[0], 1.0)
    
    def test_code_format_reward_javascript(self):
        """Test JavaScript language (only checks format, not syntax)"""
        format_reward = get_code_format_reward(language="javascript")
        
        completions = [
            [{"role": "assistant", "content": "```javascript\nfunction test() {\n    return true;\n}\n```\n"}]
        ]
        
        rewards = format_reward(completions)
        # JavaScript doesn't check syntax, only format
        self.assertEqual(rewards[0], 1.0)
    
    def test_code_format_reward_cpp(self):
        """Test C++ language (only checks format, not syntax)"""
        format_reward = get_code_format_reward(language="cpp")
        
        completions = [
            [{"role": "assistant", "content": "```cpp\nint main() {\n    return 0;\n}\n```\n"}]
        ]
        
        rewards = format_reward(completions)
        # C++ doesn't check syntax, only format
        self.assertEqual(rewards[0], 1.0)


class TestCodeFormatRewardIntegration(unittest.TestCase):
    """Integration tests: test get_code_format_reward performance in real scenarios"""
    
    def test_format_reward_as_filter(self):
        """Test using format reward as a filter"""
        format_reward = get_code_format_reward(language="python")
        
        # Simulate a batch of generated code
        completions = [
            [{"role": "assistant", "content": "```python\ndef good1():\n    return 1\n```\n"}],
            [{"role": "assistant", "content": "bad code without markers"}],
            [{"role": "assistant", "content": "```python\ndef good2():\n    return 2\n```\n"}],
            [{"role": "assistant", "content": "```python\ndef syntax_error(\n```\n"}],
            [{"role": "assistant", "content": "```python\ndef good3():\n    return 3\n```\n"}],
        ]
        
        rewards = format_reward(completions)
        
        # print(rewards)

        expected_rewards = [1.0, 0.0, 1.0, 0.5, 1.0]

        self.assertEqual(rewards, expected_rewards)
        # Count each type of reward
        perfect_count = sum(1 for r in rewards if r == 1.0)
        partial_count = sum(1 for r in rewards if r == 0.5)
        failed_count = sum(1 for r in rewards if r == 0.0)
        
        self.assertEqual(perfect_count, 3)  # 3 completely correct
        self.assertEqual(partial_count, 1)  # 1 syntax error
        self.assertEqual(failed_count, 1)   # 1 format error
    
    def test_format_reward_consistency(self):
        """Test format reward consistency"""
        format_reward = get_code_format_reward(language="python")
        
        completion = [{"role": "assistant", "content": "```python\ndef test():\n    return True\n```\n"}]
        
        # Multiple calls should return the same result
        rewards1 = format_reward([completion])
        rewards2 = format_reward([completion])
        rewards3 = format_reward([completion])
        
        self.assertEqual(rewards1[0], rewards2[0])
        self.assertEqual(rewards2[0], rewards3[0])
    
    def test_format_reward_with_various_syntax_errors(self):
        """Test detection of various syntax errors"""
        format_reward = get_code_format_reward(language="python")
        
        syntax_errors = [
            "```python\ndef test(\n```\n",  # Unmatched parentheses
            "```python\nif True\n    pass\n```\n",  # Missing colon
            "```python\ndef test():\nreturn 1\n```\n",  # Indentation error
            "```python\nfor i in\n```\n",  # Incomplete for loop
            "```python\nimport\n```\n",  # Incomplete import
        ]
        
        completions = [[{"role": "assistant", "content": code}] for code in syntax_errors]
        rewards = format_reward(completions)
        
        # All syntax errors should return 0.5
        for reward in rewards:
            self.assertEqual(reward, 0.5)
    
    def test_format_reward_real_world_examples(self):
        """Test real-world code examples"""
        format_reward = get_code_format_reward(language="python")
        
        # Real code generation examples
        real_examples = [
            # Example 1: Sorting algorithm
            """```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
```
""",
            # Example 2: Recursive function
            """```python
def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)
```
""",
            # Example 3: Class definition
            """```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        return self.items.pop() if self.items else None
```
""",
        ]
        
        completions = [[{"role": "assistant", "content": code}] for code in real_examples]
        rewards = format_reward(completions)
        
        # All real examples should have correct format and syntax
        for reward in rewards:
            self.assertEqual(reward, 1.0)


if __name__ == "__main__":
    unittest.main()
