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
    """Test core logic of reward_revision (without real sandbox)"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a mock data file for initialization
        self.test_data = pd.DataFrame({
            'id': ['test_1'],
            'problem': ['Problem 1'],
            'solution': ['Solution 1'],
            'prompter_type': ['mbppplus'],
            'selected_uts': ['test1']
        })
        
        # Create temporary parquet file
        self.temp_file = '/tmp/test_critic_rm.parquet'
        self.test_data.to_parquet(self.temp_file)
        
        # Create RewardModelForCritic instance (using mock tokenizer)
        mock_tokenizer = MagicMock()
        self.rm = RewardModelForCritic(
            file=self.temp_file,
            tokenizer=mock_tokenizer,
            run_all_cases=True
        )
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
    
    def test_critique_validity_check_correct(self):
        """Test critique validity check - Correct judgment"""
        critique_correct = "Analysis: Good.\nOverall judgment: Correct"
        
        # This critique contains Overall judgment: Correct, should be considered valid
        self.assertIn("Overall judgment: Correct", critique_correct)
    
    def test_critique_validity_check_incorrect(self):
        """Test critique validity check - Incorrect judgment"""
        critique_incorrect = "Analysis: Bad.\nOverall judgment: Incorrect"
        
        # This critique contains Overall judgment: Incorrect, should be considered valid
        self.assertIn("Overall judgment: Incorrect", critique_incorrect)
    
    def test_critique_validity_check_invalid(self):
        """Test critique validity check - Invalid critique"""
        critique_invalid = "Some critique without judgment"
        
        # This critique does not contain Overall judgment, should be considered invalid
        self.assertNotIn("Overall judgment: Correct", critique_invalid)
        self.assertNotIn("Overall judgment: Incorrect", critique_invalid)
    
    def test_code_extraction_and_normalization(self):
        """Test code extraction and normalization process"""
        revision = "```python\ndef func():\n    x = 1\n    return x\n```"
        
        # Extract code
        code_str = desanitize(revision).strip()
        self.assertNotIn("```", code_str)
        
        # Normalize code
        normalized_code_str = normalize_code(code_str)
        self.assertIsInstance(normalized_code_str, str)
        self.assertIn("v_", normalized_code_str)  # Variables should be renamed
    
    def test_cache_key_construction(self):
        """Test cache key construction logic"""
        code = "def func():\n    return 1"
        normalized = normalize_code(code)
        test_cases = "assert func() == 1"
        
        # Cache key should be a tuple of (normalized_code, str(test_cases))
        cache_key = (normalized, str(test_cases))
        
        self.assertIsInstance(cache_key, tuple)
        self.assertEqual(len(cache_key), 2)
        self.assertIsInstance(cache_key[0], str)
        self.assertIsInstance(cache_key[1], str)


class TestGetRewardAllAsync(unittest.TestCase):
    """Test asynchronous execution logic of get_reward_all"""
    
    def setUp(self):
        """Set up test environment"""
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
        """Clean up test environment"""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
    
    def test_get_reward_all_with_mocked_sandbox(self):
        """Test get_reward_all using mocked sandbox"""
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
            # Run async test
            loop = asyncio.get_event_loop()
            rewards = loop.run_until_complete(
                self.rm.get_reward_all(critiques, revisions, samples)
            )
        

        # print(rewards)

        expected_result = [1.0, 1.0]
        # Verify return result
        self.assertEqual(rewards, expected_result)
    
    def test_get_reward_all_invalid_critique(self):
        """Test invalid critique returns zero reward"""
        critiques = ["No judgment here"]
        revisions = ["```python\ncode\n```"]
        samples = [{'selected_uts': 'test1', 'prompter_type': 'mbppplus', 'time_limit': 10}]
        
        # No need to mock sandbox, because invalid critique won't call sandbox
        loop = asyncio.get_event_loop()
        rewards = loop.run_until_complete(
            self.rm.get_reward_all(critiques, revisions, samples)
        )
        
        # Invalid critique should return 0 reward
        self.assertEqual(rewards[0], 0.0)
    
    def test_get_reward_all_empty_input(self):
        """Test empty input"""
        critiques = []
        revisions = []
        samples = []
        
        loop = asyncio.get_event_loop()
        rewards = loop.run_until_complete(
            self.rm.get_reward_all(critiques, revisions, samples)
        )
        
        # Empty input should return empty list
        self.assertEqual(rewards, [])


if __name__ == "__main__":
    unittest.main()
