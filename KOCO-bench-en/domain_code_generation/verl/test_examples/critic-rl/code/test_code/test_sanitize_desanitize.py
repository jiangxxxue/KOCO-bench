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
    """Test sanitize and desanitize functions - Code extraction"""
    
    def test_sanitize_basic(self):
        """Test basic code block extraction"""
        text = """Here is some code:
```python
def func():
    pass
```
More text"""
        
        result = sanitize(text)

        # print(result)

        expected_result = "```python\ndef func():\n    pass\n```"
        
        # Should extract code block and rewrap
        self.assertEqual(result, expected_result)
    
    def test_sanitize_no_code_block(self):
        """Test case without code blocks"""
        text = "Just plain text without code blocks"
        
        result = sanitize(text)
        
        # Should return original text when no code block
        self.assertEqual(result, text)
    
    def test_sanitize_multiple_code_blocks(self):
        """Test multiple code blocks"""
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
        # Should only extract the first code block
        self.assertEqual(result, expected_result)
    
    def test_sanitize_empty_code_block(self):
        """Test empty code block"""
        text = "```python\n\n```"
        result = sanitize(text)
        # Empty code block should return original text
        self.assertEqual(result, text)
    
    def test_sanitize_python_case_insensitive(self):
        """Test python tag case insensitivity"""
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
        """Test basic code block unwrapping"""
        text = """```python
def func():
    return 42
```"""
        
        result = desanitize(text)

        # print(result)
        
        # Should remove ``` markers
        expected_result = "def func():\n    return 42"
        self.assertEqual(result, expected_result)
    
    def test_desanitize_no_code_block(self):
        """Test text without code block markers"""
        text = "def func():\n    pass"
        
        result = desanitize(text)
        
        # Should return original text when no ``` markers
        self.assertEqual(result, text)
    
    def test_desanitize_with_whitespace(self):
        """Test code block with whitespace"""
        text = """  ```python
        def func():
            pass
        ```  """
        
        result = desanitize(text)
        
        # Should remove ``` and strip whitespace
        self.assertNotIn("```", result)
        self.assertIn("def func():", result)
    
    def test_sanitize_desanitize_roundtrip(self):
        """Test roundtrip conversion of sanitize and desanitize"""
        original_code = "def add(x, y):\n    return x + y"
        text_with_code = f"Some text\n```python\n{original_code}\n```\nMore text"
        
        # sanitize -> desanitize
        sanitized = sanitize(text_with_code)
        desanitized = desanitize(sanitized)
        
        # Should restore original code
        self.assertEqual(desanitized.strip(), original_code)

if __name__ == "__main__":
    unittest.main()