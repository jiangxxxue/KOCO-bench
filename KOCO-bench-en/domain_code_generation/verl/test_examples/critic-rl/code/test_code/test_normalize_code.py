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
    """Test normalize_code function - Code normalization for caching"""
    
    def test_normalize_code_basic(self):
        """Test basic code normalization"""
        code = """def add(x, y):
    return x + y"""
        
        normalized = normalize_code(code)
        
        # print(normalized)

        expected_result = "def add(x, y):\n    return x + y"
        self.assertEqual(normalized, expected_result)
    
    def test_normalize_code_variable_renaming(self):
        """Test variable renaming"""
        code1 = """def func():
    a = 1
    b = 2
    return a + b"""
        
        code2 = """def func():
    x = 1
    y = 2
    return x + y"""
        
        # Two code segments with same semantics but different variable names
        normalized1 = normalize_code(code1)
        normalized2 = normalize_code(code2)

        # print(normalized1)
        # print(normalized2)

        expected_result1 = "def func():\n    v_0 = 1\n    v_1 = 2\n    return v_0 + v_1"
        expected_result2 = "def func():\n    v_0 = 1\n    v_1 = 2\n    return v_0 + v_1"
        self.assertEqual(normalized1, expected_result1)
        self.assertEqual(normalized2, expected_result2)
    
    def test_normalize_code_syntax_error(self):
        """Test code with syntax errors"""
        invalid_code = "def func(\n    invalid syntax"
        
        # If parsing fails, should return original code
        result = normalize_code(invalid_code)
        self.assertEqual(result, invalid_code)
    
    def test_normalize_code_empty_string(self):
        """Test empty string"""
        result = normalize_code("")
        # Empty string cannot be parsed, should return original string
        self.assertEqual(result, "")
    
    def test_normalize_code_same_semantics(self):
        """Test that code with same semantics produces same normalization result"""
        # Different variable names but same logic
        code_v1 = "x = 5"
        code_v2 = "y = 5"
        
        norm_v1 = normalize_code(code_v1)
        norm_v2 = normalize_code(code_v2)
        
        # Both should be the same after normalization (variables renamed to v_0)
        self.assertEqual(norm_v1, norm_v2)

if __name__ == "__main__":
    unittest.main()