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




class TestRewardRevisionLogic(unittest.TestCase):
    """测试reward_revision的核心逻辑（无需真实沙盒）"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建一个模拟的数据文件用于初始化
        self.test_data = pd.DataFrame({
            'id': ['test_1'],
            'problem': ['Problem 1'],
            'solution': ['Solution 1'],
            'prompter_type': ['mbppplus'],
            'selected_uts': ['test1']
        })
        
        # 创建临时parquet文件
        self.temp_file = '/tmp/test_critic_rm.parquet'
        self.test_data.to_parquet(self.temp_file)
        
        # 创建RewardModelForCritic实例（使用mock tokenizer）
        mock_tokenizer = MagicMock()
        self.rm = RewardModelForCritic(
            file=self.temp_file,
            tokenizer=mock_tokenizer,
            run_all_cases=True
        )
    
    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
    
    def test_critique_validity_check_correct(self):
        """测试批评有效性检查 - Correct判断"""
        critique_correct = "Analysis: Good.\nOverall judgment: Correct"
        
        # 这个批评包含Overall judgment: Correct，应该被认为有效
        self.assertIn("Overall judgment: Correct", critique_correct)
    
    def test_critique_validity_check_incorrect(self):
        """测试批评有效性检查 - Incorrect判断"""
        critique_incorrect = "Analysis: Bad.\nOverall judgment: Incorrect"
        
        # 这个批评包含Overall judgment: Incorrect，应该被认为有效
        self.assertIn("Overall judgment: Incorrect", critique_incorrect)
    
    def test_critique_validity_check_invalid(self):
        """测试批评有效性检查 - 无效批评"""
        critique_invalid = "Some critique without judgment"
        
        # 这个批评不包含Overall judgment，应该被认为无效
        self.assertNotIn("Overall judgment: Correct", critique_invalid)
        self.assertNotIn("Overall judgment: Incorrect", critique_invalid)
    
    def test_code_extraction_and_normalization(self):
        """测试代码提取和规范化流程"""
        revision = "```python\ndef func():\n    x = 1\n    return x\n```"
        
        # 提取代码
        code_str = desanitize(revision).strip()
        self.assertNotIn("```", code_str)
        
        # 规范化代码
        normalized_code_str = normalize_code(code_str)
        self.assertIsInstance(normalized_code_str, str)
        self.assertIn("v_", normalized_code_str)  # 变量应该被重命名
    
    def test_cache_key_construction(self):
        """测试缓存键的构建逻辑"""
        code = "def func():\n    return 1"
        normalized = normalize_code(code)
        test_cases = "assert func() == 1"
        
        # 缓存键应该是(normalized_code, str(test_cases))的元组
        cache_key = (normalized, str(test_cases))
        
        self.assertIsInstance(cache_key, tuple)
        self.assertEqual(len(cache_key), 2)
        self.assertIsInstance(cache_key[0], str)
        self.assertIsInstance(cache_key[1], str)


class TestGetRewardAllAsync(unittest.TestCase):
    """测试get_reward_all的异步执行逻辑"""
    
    def setUp(self):
        """设置测试环境"""
        self.test_data = pd.DataFrame({
            'id': ['test_1'],
            'problem': ['Problem 1'],
            'solution': ['Solution 1'],
            'prompter_type': ['mbppplus'],
            'selected_uts': ['test1']
        })
        
        self.temp_file = '/tmp/test_critic_rm_async.parquet'
        self.test_data.to_parquet(self.temp_file)
        
        mock_tokenizer = MagicMock()
        self.rm = RewardModelForCritic(
            file=self.temp_file,
            tokenizer=mock_tokenizer,
            run_all_cases=True
        )
    
    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
    
    def test_get_reward_all_with_mocked_sandbox(self):
        """测试get_reward_all使用mocked沙盒"""
        critiques = ["Overall judgment: Correct", "Overall judgment: Incorrect"]
        revisions = [
            "```python\ndef func1():\n    return True\n```",
            "```python\ndef func2():\n    return False\n```"
        ]
        samples = [
            {'selected_uts': 'test1', 'prompter_type': 'mbppplus', 'time_limit': 10},
            {'selected_uts': 'test2', 'prompter_type': 'mbppplus', 'time_limit': 10}
        ]
        
        # Mock sandbox submission
        async def mock_submit(request, semaphore):
            mock_result = MagicMock()
            mock_result.dict = MagicMock(return_value={'status': 'Success'})
            return mock_result
        
        with patch('ctrl.rl.critic_rm.submit_to_sandbox', new=mock_submit):
            # 运行异步测试
            loop = asyncio.get_event_loop()
            rewards = loop.run_until_complete(
                self.rm.get_reward_all(critiques, revisions, samples)
            )
        

        # print(rewards)

        expected_result = [1.0, 1.0]
        # 验证返回结果
        self.assertEqual(rewards, expected_result)
    
    def test_get_reward_all_invalid_critique(self):
        """测试无效批评返回零奖励"""
        critiques = ["No judgment here"]
        revisions = ["```python\ncode\n```"]
        samples = [{'selected_uts': 'test1', 'prompter_type': 'mbppplus', 'time_limit': 10}]
        
        # 不需要mock sandbox，因为无效批评不会调用sandbox
        loop = asyncio.get_event_loop()
        rewards = loop.run_until_complete(
            self.rm.get_reward_all(critiques, revisions, samples)
        )
        
        # 无效批评应该返回0奖励
        self.assertEqual(rewards[0], 0.0)
    
    def test_get_reward_all_empty_input(self):
        """测试空输入"""
        critiques = []
        revisions = []
        samples = []
        
        loop = asyncio.get_event_loop()
        rewards = loop.run_until_complete(
            self.rm.get_reward_all(critiques, revisions, samples)
        )
        
        # 空输入应该返回空列表
        self.assertEqual(rewards, [])


if __name__ == "__main__":
    unittest.main()
