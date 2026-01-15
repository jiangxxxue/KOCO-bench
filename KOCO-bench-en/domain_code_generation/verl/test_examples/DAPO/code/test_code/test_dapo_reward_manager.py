import unittest
import torch
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Add the verl directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the actual class we want to test
try:
    from verl.workers.reward_manager.dapo import DAPORewardManager
    from verl import DataProto
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False
    DAPORewardManager = None
    DataProto = None


class TestDAPORewardManager(unittest.TestCase):
    """Test DAPORewardManager.__call__ function - Overlong reward shaping described in DAPO document Section 3"""
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.batch_size = 2
        self.prompt_len = 10
        self.response_len = 20
    
    def _create_mock_tokenizer(self):
        """Create mock tokenizer"""
        tokenizer = MagicMock()
        tokenizer.eos_token = "</s>"
        
        # Mock decode method
        def mock_decode(ids, skip_special_tokens=False):
            # Simply return a string based on id
            return f"text_{len(ids)}"
        
        tokenizer.decode = MagicMock(side_effect=mock_decode)
        return tokenizer
    
    def _create_mock_compute_score(self, scores):
        """Create mock compute_score function"""
        score_iter = iter(scores)
        def mock_compute_score(data_source, solution_str, ground_truth, extra_info=None):
            return next(score_iter)
        return mock_compute_score
    
    def _create_overlong_buffer_config(self, enable=False, len=10, penalty_factor=1.0, log=False):
        """Create overlong buffer configuration"""
        config = MagicMock()
        config.enable = enable
        config.len = len
        config.penalty_factor = penalty_factor
        config.log = log
        return config
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_dapo_reward_manager_basic(self):
        """Test basic functionality"""
        tokenizer = self._create_mock_tokenizer()
        
        # Create DAPORewardManager instance
        reward_manager = DAPORewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            max_resp_len=50,
            overlong_buffer_cfg=self._create_overlong_buffer_config(enable=False)
        )
        reward_manager.compute_score = self._create_mock_compute_score([1.0, 0.0])
        
        # Create test data
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [378,  14, 210, 954, 231, 572, 315, 295, 567, 706]]),
                "responses": torch.tensor([[749, 876,  73, 111, 899, 213, 541, 769, 287, 219, 372, 880, 475, 329,
         733, 564, 739, 376, 632,  10],
        [186, 822, 577, 519, 707, 123, 143, 294, 693, 677,  70, 709, 370, 739,
          86, 499, 215, 384, 978, 908]]),
                "attention_mask": torch.ones(self.batch_size, self.prompt_len + self.response_len)
            },
            non_tensors={
                "reward_model": np.array([
                    {"ground_truth": "answer1"},
                    {"ground_truth": "answer2"}
                ], dtype=object),
                "data_source": np.array(["gsm8k", "math"], dtype=object)
            }
        )

        # print(data.batch["attention_mask"])
        # print(data.batch["prompts"])
        # print(data.batch["responses"])

        # Call reward manager
        reward_tensor = reward_manager(data, return_dict=False)
        
        # print(reward_tensor)

        expected_reward_tensor = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.]])
        torch.testing.assert_close(reward_tensor, expected_reward_tensor, atol=1e-4, rtol=1e-4) 
        # Verify output shape
        self.assertEqual(reward_tensor.shape, (self.batch_size, self.response_len))
        self.assertTrue(torch.is_tensor(reward_tensor))
        
        # Verify reward is only at the last valid token position
        expected_rewards = [1.0, 0.0]  # Values returned by compute_score
        for i in range(self.batch_size):
            valid_len = int(data.batch["attention_mask"][i, self.prompt_len:].sum().item())
            # The last valid position should have the corresponding reward (position valid_len-1, 0-indexed)
            reward_val = reward_tensor[i, valid_len - 1].item()
            self.assertAlmostEqual(reward_val, expected_rewards[i], places=5,
                                 msg=f"Sample {i} reward at position {valid_len-1}")
            # All other positions except the last valid one should be 0
            for j in range(self.response_len):
                if j != valid_len - 1:
                    self.assertEqual(reward_tensor[i, j].item(), 0.0)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_dapo_reward_manager_return_dict(self):
        """Test output when return_dict=True"""
        tokenizer = self._create_mock_tokenizer()
        
        reward_manager = DAPORewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            max_resp_len=50,
            overlong_buffer_cfg=self._create_overlong_buffer_config(enable=False)
        )
        reward_manager.compute_score = self._create_mock_compute_score([1.0, 0.0])
        
        # Create test data
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [378,  14, 210, 954, 231, 572, 315, 295, 567, 706]]),
                "responses": torch.tensor([[749, 876,  73, 111, 899, 213, 541, 769, 287, 219, 372, 880, 475, 329,
         733, 564, 739, 376, 632,  10],
        [186, 822, 577, 519, 707, 123, 143, 294, 693, 677,  70, 709, 370, 739,
          86, 499, 215, 384, 978, 908]]),
                "attention_mask": torch.ones(self.batch_size, self.prompt_len + self.response_len)
            },
            non_tensors={
                "reward_model": np.array([
                    {"ground_truth": "answer1"},
                    {"ground_truth": "answer2"}
                ], dtype=object),
                "data_source": np.array(["gsm8k", "math"], dtype=object)
            }
        )

        # print(data.batch["prompts"])
        # print(data.batch["responses"])
        
        # Call reward manager
        result = reward_manager(data, return_dict=True)

        # print(result["reward_tensor"])
        expected_reward_tensor = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.]])
        torch.testing.assert_close(result["reward_tensor"], expected_reward_tensor, atol=1e-4, rtol=1e-4) 
        # Verify return dictionary
        self.assertIsInstance(result, dict)
        self.assertIn("reward_tensor", result)
        self.assertIn("reward_extra_info", result)
        
        # Verify reward_tensor
        self.assertEqual(result["reward_tensor"].shape, (self.batch_size, self.response_len))
        
        # Verify reward_extra_info
        self.assertIn("acc", result["reward_extra_info"])
        self.assertEqual(len(result["reward_extra_info"]["acc"]), self.batch_size)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_dapo_reward_manager_overlong_penalty_disabled(self):
        """Test when overlong penalty is disabled"""
        tokenizer = self._create_mock_tokenizer()
        
        reward_manager = DAPORewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            max_resp_len=50,
            overlong_buffer_cfg=self._create_overlong_buffer_config(enable=False)
        )
        reward_manager.compute_score = self._create_mock_compute_score([1.0])
        
        # Create a long response
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113]]),
                "responses": torch.tensor([[378,  14, 210, 954, 231, 572, 315, 295, 567, 706, 749, 876,  73, 111,
         899, 213, 541, 769, 287, 219, 372, 880, 475, 329, 733, 564, 739, 376,
         632,  10, 186, 822, 577, 519, 707, 123, 143, 294, 693, 677,  70, 709,
         370, 739,  86]]),  # Close to max_resp_len
                "attention_mask": torch.ones(1, self.prompt_len + 45)
            },
            non_tensors={
                "reward_model": np.array([{"ground_truth": "answer"}], dtype=object),
                "data_source": np.array(["gsm8k"], dtype=object)
            }
        )

            # print(data.batch["prompts"])
            # print(data.batch["responses"])
        
        result = reward_manager(data, return_dict=True)

        # print(result["reward_tensor"])
        expected_reward_tensor = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 1.]])
        torch.testing.assert_close(result["reward_tensor"], expected_reward_tensor, atol=1e-4, rtol=1e-4) 
        # Verify no overlong penalty (reward should be exactly 1.0)
        valid_len = int(data.batch["attention_mask"][0, self.prompt_len:].sum().item())
        reward = result["reward_tensor"][0, valid_len - 1].item()
        self.assertAlmostEqual(reward, 1.0, places=5)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_dapo_reward_manager_overlong_penalty_enabled_no_exceed(self):
        """Test when overlong penalty is enabled but expected_len is not exceeded"""
        tokenizer = self._create_mock_tokenizer()
        
        max_resp_len = 50
        buffer_len = 10
        expected_len = max_resp_len - buffer_len  # 40
        
        reward_manager = DAPORewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            max_resp_len=max_resp_len,
            overlong_buffer_cfg=self._create_overlong_buffer_config(
                enable=True, len=buffer_len, penalty_factor=1.0
            )
        )
        reward_manager.compute_score = self._create_mock_compute_score([1.0])
        
        # Create a response that doesn't exceed expected_len
        response_len = 30  # < expected_len (40)
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113]]),
                "responses": torch.tensor([[378,  14, 210, 954, 231, 572, 315, 295, 567, 706, 749, 876,  73, 111,
         899, 213, 541, 769, 287, 219, 372, 880, 475, 329, 733, 564, 739, 376,
         632,  10]]),
                "attention_mask": torch.ones(1, self.prompt_len + response_len)
            },
            non_tensors={
                "reward_model": np.array([{"ground_truth": "answer"}], dtype=object),
                "data_source": np.array(["gsm8k"], dtype=object)
            }
        )
        # print(data.batch["prompts"])
        # print(data.batch["responses"])
        result = reward_manager(data, return_dict=True)
        # print(data.batch["attention_mask"])
        # print(result["reward_tensor"])
        expected_attention_mask = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1.]])
        torch.testing.assert_close(data.batch["attention_mask"], expected_attention_mask, atol=1e-4, rtol=1e-4) 
        expected_reward_tensor = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
        torch.testing.assert_close(result["reward_tensor"], expected_reward_tensor, atol=1e-4, rtol=1e-4) 
        # Not exceeding expected_len, should have no penalty
        valid_len = int(data.batch["attention_mask"][0, self.prompt_len:].sum().item())
        reward = result["reward_tensor"][0, valid_len - 1].item()
        self.assertAlmostEqual(reward, 1.0, places=5)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_dapo_reward_manager_overlong_penalty_enabled_partial_exceed(self):
        """Test when overlong penalty is enabled and partially exceeds expected_len"""
        tokenizer = self._create_mock_tokenizer()
        
        max_resp_len = 50
        buffer_len = 10
        expected_len = max_resp_len - buffer_len  # 40
        penalty_factor = 1.0
        
        reward_manager = DAPORewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            max_resp_len=max_resp_len,
            overlong_buffer_cfg=self._create_overlong_buffer_config(
                enable=True, len=buffer_len, penalty_factor=penalty_factor
            )
        )
        reward_manager.compute_score = self._create_mock_compute_score([1.0])
        
        # Create a response that exceeds expected_len but doesn't reach max_resp_len
        response_len = 45  # expected_len (40) < 45 < max_resp_len (50)
        exceed_len = response_len - expected_len  # 5
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113]]),
                "responses": torch.tensor([[378,  14, 210, 954, 231, 572, 315, 295, 567, 706, 749, 876,  73, 111,
         899, 213, 541, 769, 287, 219, 372, 880, 475, 329, 733, 564, 739, 376,
         632,  10, 186, 822, 577, 519, 707, 123, 143, 294, 693, 677,  70, 709,
         370, 739,  86]]),
                "attention_mask": torch.ones(1, self.prompt_len + response_len)
            },
            non_tensors={
                "reward_model": np.array([{"ground_truth": "answer"}], dtype=object),
                "data_source": np.array(["gsm8k"], dtype=object)
            }
        )

        # print(data.batch["prompts"])
        # print(data.batch["responses"])
        # print(data.batch["attention_mask"])
        result = reward_manager(data, return_dict=True)
        # print(result["reward_tensor"])

        expected_reward_tensor = torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5000]])
        torch.testing.assert_close(result["reward_tensor"], expected_reward_tensor, atol=1e-4, rtol=1e-4) 
        # Should have linear penalty
        valid_len = int(data.batch["attention_mask"][0, self.prompt_len:].sum().item())
        reward = result["reward_tensor"][0, valid_len - 1].item()
        

        # Expected penalty: -exceed_len / buffer_len * penalty_factor = -5 / 10 * 1.0 = -0.5
        expected_penalty = -exceed_len / buffer_len * penalty_factor
        expected_reward = 1.0 + expected_penalty
        self.assertAlmostEqual(reward, expected_reward, places=5)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_dapo_reward_manager_overlong_penalty_enabled_full_exceed(self):
        """Test when overlong penalty is enabled and fully exceeds buffer"""
        tokenizer = self._create_mock_tokenizer()
        
        max_resp_len = 50
        buffer_len = 10
        expected_len = max_resp_len - buffer_len  # 40
        penalty_factor = 1.0
        
        reward_manager = DAPORewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            max_resp_len=max_resp_len,
            overlong_buffer_cfg=self._create_overlong_buffer_config(
                enable=True, len=buffer_len, penalty_factor=penalty_factor
            )
        )
        reward_manager.compute_score = self._create_mock_compute_score([1.0])
        
        # Create a response that reaches max_resp_len
        response_len = 50  # == max_resp_len
        exceed_len = response_len - expected_len  # 10 == buffer_len
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113]]),
                "responses": torch.tensor([[378,  14, 210, 954, 231, 572, 315, 295, 567, 706, 749, 876,  73, 111,
         899, 213, 541, 769, 287, 219, 372, 880, 475, 329, 733, 564, 739, 376,
         632,  10, 186, 822, 577, 519, 707, 123, 143, 294, 693, 677,  70, 709,
         370, 739,  86, 499, 215, 384, 978, 908]]),
                "attention_mask": torch.ones(1, self.prompt_len + response_len)
            },
            non_tensors={
                "reward_model": np.array([{"ground_truth": "answer"}], dtype=object),
                "data_source": np.array(["gsm8k"], dtype=object)
            }
        )
        # print(data.batch["prompts"])
        # print(data.batch["responses"])
        # print(data.batch["attention_mask"])
        result = reward_manager(data, return_dict=True)

        expected_reward_tensor = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.]])
        # print(result["reward_tensor"])
        # Penalty should reach maximum value -penalty_factor
        torch.testing.assert_close(result["reward_tensor"], expected_reward_tensor, atol=1e-4, rtol=1e-4) 
        valid_len = int(data.batch["attention_mask"][0, self.prompt_len:].sum().item())
        reward = result["reward_tensor"][0, valid_len - 1].item()
        
        # Expected penalty: min(-10 / 10 * 1.0, 0) = -1.0
        expected_reward = 1.0 - penalty_factor
        self.assertAlmostEqual(reward, expected_reward, places=5)
    

    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_dapo_reward_manager_overlong_penalty_with_log(self):
        """Test when overlong logging is enabled"""
        tokenizer = self._create_mock_tokenizer()
        
        max_resp_len = 50
        buffer_len = 10
        expected_len = max_resp_len - buffer_len  # 40
        
        reward_manager = DAPORewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            max_resp_len=max_resp_len,
            overlong_buffer_cfg=self._create_overlong_buffer_config(
                enable=True, len=buffer_len, penalty_factor=1.0, log=True
            )
        )
        reward_manager.compute_score = self._create_mock_compute_score([1.0, 1.0])
        
        # One not overlong, one overlong
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [378,  14, 210, 954, 231, 572, 315, 295, 567, 706]]),
                "responses": torch.tensor([[749, 876,  73, 111, 899, 213, 541, 769, 287, 219, 372, 880, 475, 329,
         733, 564, 739, 376, 632,  10, 186, 822, 577, 519, 707, 123, 143, 294,
         693, 677,  70, 709, 370, 739,  86, 499, 215, 384, 978, 908, 266, 630,
         540, 560, 470],
        [561, 323, 720,  11, 461, 177, 289, 184, 553, 548, 409, 483, 507, 558,
         491, 214, 691, 636, 903, 582, 490, 389, 528, 555,  33, 527, 547, 865,
         589, 141, 645, 761, 339, 861, 564, 270, 553, 137, 275, 217, 321, 205,
         507, 845, 438]]),
                "attention_mask": torch.cat([
                    torch.cat([torch.ones(1, self.prompt_len + 30), torch.zeros(1, 15)], dim=1),  # Not overlong
                    torch.ones(1, self.prompt_len + 45)  # Overlong
                ], dim=0)
            },
            non_tensors={
                "reward_model": np.array([
                    {"ground_truth": "answer1"},
                    {"ground_truth": "answer2"}
                ], dtype=object),
                "data_source": np.array(["gsm8k", "math"], dtype=object)
            }
        )

        # print(data.batch["prompts"])
        # print(data.batch["responses"])

        
        result = reward_manager(data, return_dict=True)
        # print(result["reward_tensor"])

        expected_reward_tensor = torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5000]])
        torch.testing.assert_close(result["reward_tensor"], expected_reward_tensor, atol=1e-4, rtol=1e-4) 
        # Should record overlong information
        self.assertIn("overlong_reward", result["reward_extra_info"])
        self.assertIn("overlong", result["reward_extra_info"])
        self.assertEqual(len(result["reward_extra_info"]["overlong"]), 2)
        
        # First one should not be overlong
        self.assertFalse(result["reward_extra_info"]["overlong"][0])
        # Second one should be overlong
        self.assertTrue(result["reward_extra_info"]["overlong"][1])
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_dapo_reward_manager_zero_score(self):
        """Test case with incorrect answer (score=0)"""
        tokenizer = self._create_mock_tokenizer()
        
        reward_manager = DAPORewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            max_resp_len=50,
            overlong_buffer_cfg=self._create_overlong_buffer_config(enable=False)
        )
        reward_manager.compute_score = self._create_mock_compute_score([0.0])
        
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113]]),
                "responses": torch.tensor([[378,  14, 210, 954, 231, 572, 315, 295, 567, 706, 749, 876,  73, 111,
         899, 213, 541, 769, 287, 219]]),
                "attention_mask": torch.ones(1, self.prompt_len + self.response_len)
            },
            non_tensors={
                "reward_model": np.array([{"ground_truth": "answer"}], dtype=object),
                "data_source": np.array(["gsm8k"], dtype=object)
            }
        )
        # print(data.batch["prompts"])
        # print(data.batch["responses"])
        # print(data.batch["attention_mask"])
        result = reward_manager(data, return_dict=True)
        # print(result["reward_tensor"])
        expected_reward_tensor = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        torch.testing.assert_close(result["reward_tensor"], expected_reward_tensor, atol=1e-4, rtol=1e-4) 
        # Verify reward is 0.0
        valid_len = int(data.batch["attention_mask"][0, self.prompt_len:].sum().item())
        reward = result["reward_tensor"][0, valid_len - 1].item()
        self.assertAlmostEqual(reward, 0.0, places=5)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_dapo_reward_manager_different_data_sources(self):
        """Test different data sources"""
        tokenizer = self._create_mock_tokenizer()
        
        reward_manager = DAPORewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            max_resp_len=50,
            overlong_buffer_cfg=self._create_overlong_buffer_config(enable=False)
        )
        reward_manager.compute_score = self._create_mock_compute_score([1.0, 0.0, 1.0])
        
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [378,  14, 210, 954, 231, 572, 315, 295, 567, 706],
        [749, 876,  73, 111, 899, 213, 541, 769, 287, 219]]),
                "responses": torch.tensor([[372, 880, 475, 329, 733, 564, 739, 376, 632,  10, 186, 822, 577, 519,
         707, 123, 143, 294, 693, 677],
        [ 70, 709, 370, 739,  86, 499, 215, 384, 978, 908, 266, 630, 540, 560,
         470, 561, 323, 720,  11, 461],
        [177, 289, 184, 553, 548, 409, 483, 507, 558, 491, 214, 691, 636, 903,
         582, 490, 389, 528, 555,  33]]),
                "attention_mask": torch.ones(3, self.prompt_len + self.response_len)
            },
            non_tensors={
                "reward_model": np.array([
                    {"ground_truth": "answer1"},
                    {"ground_truth": "answer2"},
                    {"ground_truth": "answer3"}
                ], dtype=object),
                "data_source": np.array(["gsm8k", "math", "aime"], dtype=object)
            }
        )

        # print(data.batch["prompts"])
        # print(data.batch["responses"])
        # print(data.batch["attention_mask"])
        
        result = reward_manager(data, return_dict=True)
        # print(result["reward_tensor"])

        expected_reward_tensor = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 1.]])
        torch.testing.assert_close(result["reward_tensor"], expected_reward_tensor, atol=1e-4, rtol=1e-4) 
        # Verify that different data sources are handled correctly
        self.assertEqual(result["reward_tensor"].shape[0], 3)
        self.assertEqual(len(result["reward_extra_info"]["acc"]), 3)


if __name__ == '__main__':
    unittest.main()

