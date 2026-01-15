import unittest
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the actual function we want to test
from open_r1.rewards import get_code_format_reward


class TestCodeFormatReward(unittest.TestCase):
    """测试get_code_format_reward函数 - 检查代码格式和语法"""
    
    def setUp(self):
        self.format_reward_func = get_code_format_reward(language="python")
    
    def test_code_format_reward_correct_format(self):
        """测试格式正确的代码"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef hello():\n    print('Hello')\n```\n"}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # 格式正确且语法正确应该返回1.0
        self.assertEqual(rewards[0], 1.0)
    
    def test_code_format_reward_missing_code_block(self):
        """测试缺少代码块标记的情况"""
        completions = [
            [{"role": "assistant", "content": "def hello():\n    print('Hello')"}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # 没有代码块标记应该返回0.0
        self.assertEqual(rewards[0], 0.0)
    
    def test_code_format_reward_syntax_error(self):
        """测试语法错误的代码"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef hello(\n    print('Hello')\n```\n"}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # 格式正确但语法错误应该返回0.5
        self.assertEqual(rewards[0], 0.5)
    
    def test_code_format_reward_extra_text_before(self):
        """测试代码块前有额外文本"""
        completions = [
            [{"role": "assistant", "content": "Here is the code:\n```python\ndef test():\n    pass\n```\n"}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # 代码块前有文本，格式不符合要求
        self.assertEqual(rewards[0], 0.0)
    
    def test_code_format_reward_extra_text_after(self):
        """测试代码块后有额外文本"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef test():\n    pass\n```\nThis is the solution."}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # 代码块后有文本（除了换行符），格式不符合要求
        self.assertEqual(rewards[0], 0.0)
    
    def test_code_format_reward_multiple_code_blocks(self):
        """测试多个代码块"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef func1():\n    pass\n```\n```python\ndef func2():\n    pass\n```\n"}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # 多个代码块不符合格式要求
        self.assertEqual(rewards[0], 0.0)
    
    def test_code_format_reward_empty_code_block(self):
        """测试空代码块"""
        completions = [
            [{"role": "assistant", "content": "```python\n\n```\n"}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # 空代码块应该返回0.0（语法错误或格式问题）
        self.assertLessEqual(rewards[0], 0.5)
    
    def test_code_format_reward_correct_format_complex_code(self):
        """测试格式正确的复杂代码"""
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
        
        # 复杂但格式和语法都正确的代码应该返回1.0
        self.assertEqual(rewards[0], 1.0)
    
    def test_code_format_reward_indentation_error(self):
        """测试缩进错误"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef test():\npass\n```\n"}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # 缩进错误是语法错误，应该返回0.5
        self.assertEqual(rewards[0], 0.5)
    
    def test_code_format_reward_incomplete_code(self):
        """测试不完整的代码"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef test():\n    if True:\n```\n"}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # 不完整的代码是语法错误
        self.assertEqual(rewards[0], 0.5)
    
    def test_code_format_reward_wrong_language_marker(self):
        """测试错误的语言标记"""
        completions = [
            [{"role": "assistant", "content": "```javascript\ndef test():\n    pass\n```\n"}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # 语言标记不匹配，格式不正确
        self.assertEqual(rewards[0], 0.0)
    
    def test_code_format_reward_batch_processing(self):
        """测试批量处理"""
        completions = [
            [{"role": "assistant", "content": "```python\ndef func1():\n    return 1\n```\n"}],
            [{"role": "assistant", "content": "def func2():\n    return 2"}],
            [{"role": "assistant", "content": "```python\ndef func3(\n```\n"}],
            [{"role": "assistant", "content": "```python\ndef func4():\n    return 4\n```\n"}],
        ]
        
        rewards = self.format_reward_func(completions)
        
        # 验证批量处理的结果
        self.assertEqual(len(rewards), 4)
        self.assertEqual(rewards[0], 1.0)  # 正确
        self.assertEqual(rewards[1], 0.0)  # 格式错误
        self.assertEqual(rewards[2], 0.5)  # 语法错误
        self.assertEqual(rewards[3], 1.0)  # 正确
    
    def test_code_format_reward_with_imports(self):
        """测试包含import语句的代码"""
        completions = [
            [{"role": "assistant", "content": "```python\nimport math\nimport sys\n\ndef calculate():\n    return math.sqrt(16)\n```\n"}]
        ]
        
        rewards = self.format_reward_func(completions)
        
        # 包含import的正确代码应该返回1.0
        self.assertEqual(rewards[0], 1.0)
    
    def test_code_format_reward_with_classes(self):
        """测试包含类定义的代码"""
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
        
        # 包含类定义的正确代码应该返回1.0
        self.assertEqual(rewards[0], 1.0)
    
    def test_code_format_reward_with_comments(self):
        """测试包含注释的代码"""
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
        
        # 包含注释的正确代码应该返回1.0
        self.assertEqual(rewards[0], 1.0)
    
    def test_code_format_reward_edge_cases(self):
        """测试边缘情况"""
        # 测试1: 只有代码块标记，没有内容
        completions1 = [
            [{"role": "assistant", "content": "```python\n```\n"}]
        ]
        rewards1 = self.format_reward_func(completions1)
        self.assertLessEqual(rewards1[0], 0.5)
        
        # 测试2: 代码块标记不完整
        completions2 = [
            [{"role": "assistant", "content": "```python\ndef test():\n    pass"}]
        ]
        rewards2 = self.format_reward_func(completions2)
        self.assertEqual(rewards2[0], 0.0)
        
        # 测试3: 只有一行代码
        completions3 = [
            [{"role": "assistant", "content": "```python\nprint('hello')\n```\n"}]
        ]
        rewards3 = self.format_reward_func(completions3)
        self.assertEqual(rewards3[0], 1.0)


class TestCodeFormatRewardDifferentLanguages(unittest.TestCase):
    """测试不同编程语言的格式检查"""
    
    def test_code_format_reward_python(self):
        """测试Python语言"""
        format_reward = get_code_format_reward(language="python")
        
        completions = [
            [{"role": "assistant", "content": "```python\ndef test():\n    pass\n```\n"}]
        ]
        
        rewards = format_reward(completions)
        self.assertEqual(rewards[0], 1.0)
    
    def test_code_format_reward_javascript(self):
        """测试JavaScript语言（只检查格式，不检查语法）"""
        format_reward = get_code_format_reward(language="javascript")
        
        completions = [
            [{"role": "assistant", "content": "```javascript\nfunction test() {\n    return true;\n}\n```\n"}]
        ]
        
        rewards = format_reward(completions)
        # JavaScript不检查语法，只检查格式
        self.assertEqual(rewards[0], 1.0)
    
    def test_code_format_reward_cpp(self):
        """测试C++语言（只检查格式，不检查语法）"""
        format_reward = get_code_format_reward(language="cpp")
        
        completions = [
            [{"role": "assistant", "content": "```cpp\nint main() {\n    return 0;\n}\n```\n"}]
        ]
        
        rewards = format_reward(completions)
        # C++不检查语法，只检查格式
        self.assertEqual(rewards[0], 1.0)


class TestCodeFormatRewardIntegration(unittest.TestCase):
    """集成测试：测试get_code_format_reward在实际场景中的表现"""
    
    def test_format_reward_as_filter(self):
        """测试格式奖励作为过滤器的使用"""
        format_reward = get_code_format_reward(language="python")
        
        # 模拟一批生成的代码
        completions = [
            [{"role": "assistant", "content": "```python\ndef good1():\n    return 1\n```\n"}],
            [{"role": "assistant", "content": "bad code without markers"}],
            [{"role": "assistant", "content": "```python\ndef good2():\n    return 2\n```\n"}],
            [{"role": "assistant", "content": "```python\ndef syntax_error(\n```\n"}],
            [{"role": "assistant", "content": "```python\ndef good3():\n    return 3\n```\n"}],
        ]
        
        rewards = format_reward(completions)
        
        # 统计各类奖励
        perfect_count = sum(1 for r in rewards if r == 1.0)
        partial_count = sum(1 for r in rewards if r == 0.5)
        failed_count = sum(1 for r in rewards if r == 0.0)
        
        self.assertEqual(perfect_count, 3)  # 3个完全正确
        self.assertEqual(partial_count, 1)  # 1个语法错误
        self.assertEqual(failed_count, 1)   # 1个格式错误
    
    def test_format_reward_consistency(self):
        """测试格式奖励的一致性"""
        format_reward = get_code_format_reward(language="python")
        
        completion = [{"role": "assistant", "content": "```python\ndef test():\n    return True\n```\n"}]
        
        # 多次调用应该返回相同结果
        rewards1 = format_reward([completion])
        rewards2 = format_reward([completion])
        rewards3 = format_reward([completion])
        
        self.assertEqual(rewards1[0], rewards2[0])
        self.assertEqual(rewards2[0], rewards3[0])
    
    def test_format_reward_with_various_syntax_errors(self):
        """测试各种语法错误的检测"""
        format_reward = get_code_format_reward(language="python")
        
        syntax_errors = [
            "```python\ndef test(\n```\n",  # 括号不匹配
            "```python\nif True\n    pass\n```\n",  # 缺少冒号
            "```python\ndef test():\nreturn 1\n```\n",  # 缩进错误
            "```python\nfor i in\n```\n",  # 不完整的for循环
            "```python\nimport\n```\n",  # 不完整的import
        ]
        
        completions = [[{"role": "assistant", "content": code}] for code in syntax_errors]
        rewards = format_reward(completions)
        
        # 所有语法错误都应该返回0.5
        for reward in rewards:
            self.assertEqual(reward, 0.5)
    
    def test_format_reward_real_world_examples(self):
        """测试真实世界的代码示例"""
        format_reward = get_code_format_reward(language="python")
        
        # 真实的代码生成示例
        real_examples = [
            # 示例1: 排序算法
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
            # 示例2: 递归函数
            """```python
def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)
```
""",
            # 示例3: 类定义
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
        
        # 所有真实示例都应该是格式和语法正确的
        for reward in rewards:
            self.assertEqual(reward, 1.0)


if __name__ == "__main__":
    unittest.main()
