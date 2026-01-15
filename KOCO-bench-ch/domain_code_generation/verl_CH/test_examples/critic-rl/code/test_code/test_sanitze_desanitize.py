import unittest
import asyncio
import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock
import pandas as pd
import numpy as np

# Mock only external dependencies
sys.modules['verl'] = MagicMock()
sys.modules['sandbox_fusion'] = MagicMock()
sys.modules['tenacity'] = MagicMock()

# Add the code directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the actual functions from critic-rl code
from ctrl.rl.critic_rm import normalize_code, sanitize, desanitize, RewardModelForCritic




class TestSanitizeDesanitize(unittest.TestCase):
    """测试sanitize和desanitize函数 - 代码提取"""
    
    def test_sanitize_basic(self):
        """测试基本的代码块提取"""
        text = """Here is some code:
```python
def func():
    pass
```
More text"""
        
        result = sanitize(text)

        # print(result)

        expected_result = "```python\ndef func():\n    pass\n```"
        
        # 应该提取代码块并重新包装
        self.assertEqual(result, expected_result)
    
    def test_sanitize_no_code_block(self):
        """测试没有代码块的情况"""
        text = "Just plain text without code blocks"
        
        result = sanitize(text)
        
        # 没有代码块时应该返回原文本
        self.assertEqual(result, text)
    
    def test_sanitize_multiple_code_blocks(self):
        """测试多个代码块"""
        text = """First block:
```python
def func1():
    pass
```
Second block:
```python
def func2():
    pass
```"""
        
        result = sanitize(text)
        
        # print(result)

        expected_result = "```python\ndef func1():\n    pass\n```"
        # 应该只提取第一个代码块
        self.assertEqual(result, expected_result)
    
    def test_sanitize_empty_code_block(self):
        """测试空代码块"""
        text = "```python\n\n```"
        result = sanitize(text)
        # 空代码块应该返回原文本
        self.assertEqual(result, text)
    
    def test_sanitize_python_case_insensitive(self):
        """测试python标签大小写不敏感"""
        text1 = "```python\ncode\n```"
        text2 = "```Python\ncode\n```"
        text3 = "```PYTHON\ncode\n```"
        
        result1 = sanitize(text1)
        result2 = sanitize(text2)
        result3 = sanitize(text3)

        # print(result1)
        # print(result2)
        # print(result3)

        expected_result1 = "```python\ncode\n```"
        expected_result2 = "```python\ncode\n```"
        expected_result3 = "```python\ncode\n```"
        self.assertEqual(result1, expected_result1)
        self.assertEqual(result2, expected_result2)
        self.assertEqual(result3, expected_result3)
    
    def test_desanitize_basic(self):
        """测试基本的代码块解包"""
        text = """```python
def func():
    return 42
```"""
        
        result = desanitize(text)

        # print(result)
        
        # 应该去除```标记
        expected_result = "def func():\n    return 42"
        self.assertEqual(result, expected_result)
    
    def test_desanitize_no_code_block(self):
        """测试没有代码块标记的文本"""
        text = "def func():\n    pass"
        
        result = desanitize(text)
        
        # 没有```标记时应该返回原文本
        self.assertEqual(result, text)
    
    def test_desanitize_with_whitespace(self):
        """测试带有空白的代码块"""
        text = """  ```python
        def func():
            pass
        ```  """
        
        result = desanitize(text)
        
        # 应该去除```并strip空白
        self.assertNotIn("```", result)
        self.assertIn("def func():", result)
    
    def test_sanitize_desanitize_roundtrip(self):
        """测试sanitize和desanitize的往返转换"""
        original_code = "def add(x, y):\n    return x + y"
        text_with_code = f"Some text\n```python\n{original_code}\n```\nMore text"
        
        # sanitize -> desanitize
        sanitized = sanitize(text_with_code)
        desanitized = desanitize(sanitized)
        
        # 应该恢复原始代码
        self.assertEqual(desanitized.strip(), original_code)

if __name__ == "__main__":
    unittest.main()