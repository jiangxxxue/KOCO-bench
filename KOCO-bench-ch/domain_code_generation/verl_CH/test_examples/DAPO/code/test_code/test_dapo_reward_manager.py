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
    """测试DAPORewardManager.__call__函数 - DAPO文档第3节描述的超长奖励塑形"""
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.batch_size = 2
        self.prompt_len = 10
        self.response_len = 20
    
    def _create_mock_tokenizer(self):
        """创建mock的tokenizer"""
        tokenizer = MagicMock()
        tokenizer.eos_token = "</s>"
        
        # Mock decode方法
        def mock_decode(ids, skip_special_tokens=False):
            # 简单返回一个基于id的字符串
            return f"text_{len(ids)}"
        
        tokenizer.decode = MagicMock(side_effect=mock_decode)
        return tokenizer
    
    def _create_mock_compute_score(self, scores):
        """创建mock的compute_score函数"""
        score_iter = iter(scores)
        def mock_compute_score(data_source, solution_str, ground_truth, extra_info=None):
            return next(score_iter)
        return mock_compute_score
    
    def _create_overlong_buffer_config(self, enable=False, len=10, penalty_factor=1.0, log=False):
        """创建超长buffer配置"""
        config = MagicMock()
        config.enable = enable
        config.len = len
        config.penalty_factor = penalty_factor
        config.log = log
        return config
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_dapo_reward_manager_basic(self):
        """测试基本功能"""
        tokenizer = self._create_mock_tokenizer()
        
        # 创建DAPORewardManager实例
        reward_manager = DAPORewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            max_resp_len=50,
            overlong_buffer_cfg=self._create_overlong_buffer_config(enable=False)
        )
        reward_manager.compute_score = self._create_mock_compute_score([1.0, 0.0])
        
        # 创建测试数据
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

        # 调用reward manager
        reward_tensor = reward_manager(data, return_dict=False)
        
        # print(reward_tensor)

        expected_reward_tensor = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.]])
        torch.testing.assert_close(reward_tensor, expected_reward_tensor, atol=1e-4, rtol=1e-4) 
        # 验证输出形状
        self.assertEqual(reward_tensor.shape, (self.batch_size, self.response_len))
        self.assertTrue(torch.is_tensor(reward_tensor))
        
        # 验证奖励只在最后一个有效token位置
        expected_rewards = [1.0, 0.0]  # compute_score返回的值
        for i in range(self.batch_size):
            valid_len = int(data.batch["attention_mask"][i, self.prompt_len:].sum().item())
            # 最后一个有效位置应该有对应的奖励（位置valid_len-1，索引从0开始）
            reward_val = reward_tensor[i, valid_len - 1].item()
            self.assertAlmostEqual(reward_val, expected_rewards[i], places=5,
                                 msg=f"Sample {i} reward at position {valid_len-1}")
            # 除了最后一个有效位置，其他位置应该是0
            for j in range(self.response_len):
                if j != valid_len - 1:
                    self.assertEqual(reward_tensor[i, j].item(), 0.0)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_dapo_reward_manager_return_dict(self):
        """测试return_dict=True时的输出"""
        tokenizer = self._create_mock_tokenizer()
        
        reward_manager = DAPORewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            max_resp_len=50,
            overlong_buffer_cfg=self._create_overlong_buffer_config(enable=False)
        )
        reward_manager.compute_score = self._create_mock_compute_score([1.0, 0.0])
        
        # 创建测试数据
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
        
        # 调用reward manager
        result = reward_manager(data, return_dict=True)

        # print(result["reward_tensor"])
        expected_reward_tensor = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.]])
        torch.testing.assert_close(result["reward_tensor"], expected_reward_tensor, atol=1e-4, rtol=1e-4) 
        # 验证返回字典
        self.assertIsInstance(result, dict)
        self.assertIn("reward_tensor", result)
        self.assertIn("reward_extra_info", result)
        
        # 验证reward_tensor
        self.assertEqual(result["reward_tensor"].shape, (self.batch_size, self.response_len))
        
        # 验证reward_extra_info
        self.assertIn("acc", result["reward_extra_info"])
        self.assertEqual(len(result["reward_extra_info"]["acc"]), self.batch_size)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_dapo_reward_manager_overlong_penalty_disabled(self):
        """测试超长惩罚关闭时"""
        tokenizer = self._create_mock_tokenizer()
        
        reward_manager = DAPORewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            max_resp_len=50,
            overlong_buffer_cfg=self._create_overlong_buffer_config(enable=False)
        )
        reward_manager.compute_score = self._create_mock_compute_score([1.0])
        
        # 创建一个长响应
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113]]),
                "responses": torch.tensor([[378,  14, 210, 954, 231, 572, 315, 295, 567, 706, 749, 876,  73, 111,
         899, 213, 541, 769, 287, 219, 372, 880, 475, 329, 733, 564, 739, 376,
         632,  10, 186, 822, 577, 519, 707, 123, 143, 294, 693, 677,  70, 709,
         370, 739,  86]]),  # 接近max_resp_len
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
        # 验证没有超长惩罚（奖励应该正好是1.0）
        valid_len = int(data.batch["attention_mask"][0, self.prompt_len:].sum().item())
        reward = result["reward_tensor"][0, valid_len - 1].item()
        self.assertAlmostEqual(reward, 1.0, places=5)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_dapo_reward_manager_overlong_penalty_enabled_no_exceed(self):
        """测试超长惩罚启用但未超过expected_len时"""
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
        
        # 创建一个未超过expected_len的响应
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
        # 未超过expected_len，应该没有惩罚
        valid_len = int(data.batch["attention_mask"][0, self.prompt_len:].sum().item())
        reward = result["reward_tensor"][0, valid_len - 1].item()
        self.assertAlmostEqual(reward, 1.0, places=5)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_dapo_reward_manager_overlong_penalty_enabled_partial_exceed(self):
        """测试超长惩罚启用且部分超过expected_len时"""
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
        
        # 创建一个超过expected_len但未到max_resp_len的响应
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
        # 应该有线性惩罚
        valid_len = int(data.batch["attention_mask"][0, self.prompt_len:].sum().item())
        reward = result["reward_tensor"][0, valid_len - 1].item()
        

        # 预期惩罚: -exceed_len / buffer_len * penalty_factor = -5 / 10 * 1.0 = -0.5
        expected_penalty = -exceed_len / buffer_len * penalty_factor
        expected_reward = 1.0 + expected_penalty
        self.assertAlmostEqual(reward, expected_reward, places=5)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_dapo_reward_manager_overlong_penalty_enabled_full_exceed(self):
        """测试超长惩罚启用且完全超过buffer时"""
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
        
        # 创建一个达到max_resp_len的响应
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
        # 惩罚应该达到最大值 -penalty_factor
        torch.testing.assert_close(result["reward_tensor"], expected_reward_tensor, atol=1e-4, rtol=1e-4) 
        valid_len = int(data.batch["attention_mask"][0, self.prompt_len:].sum().item())
        reward = result["reward_tensor"][0, valid_len - 1].item()
        
        # 预期惩罚: min(-10 / 10 * 1.0, 0) = -1.0
        expected_reward = 1.0 - penalty_factor
        self.assertAlmostEqual(reward, expected_reward, places=5)
    

    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_dapo_reward_manager_overlong_penalty_with_log(self):
        """测试启用overlong日志时"""
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
        
        # 一个不超长，一个超长
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
                    torch.cat([torch.ones(1, self.prompt_len + 30), torch.zeros(1, 15)], dim=1),  # 不超长
                    torch.ones(1, self.prompt_len + 45)  # 超长
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
        # 应该记录overlong信息
        self.assertIn("overlong_reward", result["reward_extra_info"])
        self.assertIn("overlong", result["reward_extra_info"])
        self.assertEqual(len(result["reward_extra_info"]["overlong"]), 2)
        
        # 第一个不应该超长
        self.assertFalse(result["reward_extra_info"]["overlong"][0])
        # 第二个应该超长
        self.assertTrue(result["reward_extra_info"]["overlong"][1])
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_dapo_reward_manager_zero_score(self):
        """测试错误答案（score=0）的情况"""
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
        # 验证奖励为0.0
        valid_len = int(data.batch["attention_mask"][0, self.prompt_len:].sum().item())
        reward = result["reward_tensor"][0, valid_len - 1].item()
        self.assertAlmostEqual(reward, 0.0, places=5)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_dapo_reward_manager_different_data_sources(self):
        """测试不同数据源"""
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
        # 验证不同数据源都能正确处理
        self.assertEqual(result["reward_tensor"].shape[0], 3)
        self.assertEqual(len(result["reward_extra_info"]["acc"]), 3)


if __name__ == '__main__':
    unittest.main()

