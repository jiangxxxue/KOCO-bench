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


class TestNormalizeCode(unittest.TestCase):
    """测试normalize_code函数 - 代码规范化用于缓存"""
    
    def test_normalize_code_basic(self):
        """测试基本的代码规范化"""
        code = """def add(x, y):
    return x + y"""
        
        normalized = normalize_code(code)
        
        # print(normalized)

        expected_result = "def add(x, y):\n    return x + y"
        self.assertEqual(normalized, expected_result)
    
    def test_normalize_code_variable_renaming(self):
        """测试变量重命名"""
        code1 = """def func():
    a = 1
    b = 2
    return a + b"""
        
        code2 = """def func():
    x = 1
    y = 2
    return x + y"""
        
        # 两段语义相同但变量名不同的代码
        normalized1 = normalize_code(code1)
        normalized2 = normalize_code(code2)

        # print(normalized1)
        # print(normalized2)

        expected_result1 = "def func():\n    v_0 = 1\n    v_1 = 2\n    return v_0 + v_1"
        expected_result2 = "def func():\n    v_0 = 1\n    v_1 = 2\n    return v_0 + v_1"
        self.assertEqual(normalized1, expected_result1)
        self.assertEqual(normalized2, expected_result2)
    
    def test_normalize_code_syntax_error(self):
        """测试语法错误的代码"""
        invalid_code = "def func(\n    invalid syntax"
        
        # 如果解析失败，应该返回原始代码
        result = normalize_code(invalid_code)
        self.assertEqual(result, invalid_code)
    
    def test_normalize_code_empty_string(self):
        """测试空字符串"""
        result = normalize_code("")
        # 空字符串无法解析，应该返回原字符串
        self.assertEqual(result, "")
    
    def test_normalize_code_same_semantics(self):
        """测试语义相同的代码规范化结果相同"""
        # 不同的变量名但相同的逻辑
        code_v1 = "x = 5"
        code_v2 = "y = 5"
        
        norm_v1 = normalize_code(code_v1)
        norm_v2 = normalize_code(code_v2)
        
        # 两者规范化后应该相同（变量都被重命名为v_0）
        self.assertEqual(norm_v1, norm_v2)

if __name__ == "__main__":
    unittest.main()