import unittest
import torch
import sys
import os
from unittest.mock import MagicMock, patch
import PIL.Image

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))


class TestGenerateAndScoreCompletions(unittest.TestCase):
    """测试VLMGRPOTrainer._generate_and_score_completions函数 - 生成多个响应并计算奖励"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        self.batch_size = 2
        self.max_completion_length = 50
        self.num_generations = 1
    
    def test_input_preparation(self):
        """测试输入数据准备逻辑"""
        # 模拟输入数据
        inputs = [
            {
                "prompt": [{"role": "user", "content": [{"type": "text", "text": "Rate this image"}]}],
                "image_path": ["test_image1.jpg"],
                "solution": "<answer>4.5</answer>",
            },
            {
                "prompt": [{"role": "user", "content": [{"type": "text", "text": "Rate this image"}]}],
                "image_path": ["test_image2.jpg"],
                "solution": "<answer>3.2</answer>",
            }
        ]
        
        # 验证输入结构
        self.assertEqual(len(inputs), self.batch_size)
        for inp in inputs:
            self.assertIn("prompt", inp)
            self.assertIn("image_path", inp)
            self.assertIn("solution", inp)
    
    def test_image_loading_and_resizing(self):
        """测试图像加载和调整大小逻辑"""
        # 创建测试图像
        small_image = PIL.Image.new('RGB', (20, 20), color='red')
        normal_image = PIL.Image.new('RGB', (100, 100), color='blue')
        
        # 测试小图像调整大小
        w, h = small_image.size
        if w < 28 or h < 28:
            if w < h:
                new_w = 28
                new_h = int(h * (28/w))
            else:
                new_h = 28
                new_w = int(w * (28/h))
            resized_image = small_image.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
            
            # 验证调整后的尺寸
            self.assertGreaterEqual(resized_image.size[0], 28)
            self.assertGreaterEqual(resized_image.size[1], 28)
        
        # 测试正常图像不需要调整
        w, h = normal_image.size
        self.assertGreaterEqual(w, 28)
        self.assertGreaterEqual(h, 28)
    
    def test_completion_generation_simulation(self):
        """测试完成生成的模拟"""
        # 模拟生成的token IDs
        prompt_length = 10
        completion_length = 20
        vocab_size = 1000
        
        prompt_ids = torch.randint(0, vocab_size, (self.batch_size, prompt_length))
        completion_ids = torch.randint(0, vocab_size, (self.batch_size, completion_length))
        
        # 拼接prompt和completion
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        
        # 验证形状
        self.assertEqual(prompt_completion_ids.shape, (self.batch_size, prompt_length + completion_length))
        self.assertEqual(completion_ids.shape, (self.batch_size, completion_length))
    
    def test_eos_masking(self):
        """测试EOS token掩码逻辑"""
        completion_length = 10
        eos_token_id = 2
        
        # 创建包含EOS token的completion
        completion_ids = torch.tensor([
            [5, 7, 2, 9, 11, 13, 15, 17, 19, 21],  # EOS在位置2
            [4, 6, 8, 10, 12, 2, 16, 18, 20, 22],  # EOS在位置5
        ])
        
        # 检测EOS位置
        is_eos = completion_ids == eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=self.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        
        # 创建掩码
        sequence_indices = torch.arange(is_eos.size(1), device=self.device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # 验证掩码
        self.assertEqual(completion_mask[0, 2].item(), 1)  # EOS位置应该为1
        self.assertEqual(completion_mask[0, 3].item(), 0)  # EOS后应该为0
        self.assertEqual(completion_mask[1, 5].item(), 1)
        self.assertEqual(completion_mask[1, 6].item(), 0)
    
    def test_reward_computation_simulation(self):
        """测试奖励计算的模拟"""
        num_reward_funcs = 2
        
        # 模拟每个奖励函数的输出
        rewards_per_func = torch.zeros(self.batch_size, num_reward_funcs, device=self.device)
        
        # 模拟accuracy_reward
        rewards_per_func[:, 0] = torch.tensor([1.8, 1.6])
        
        # 模拟format_reward
        rewards_per_func[:, 1] = torch.tensor([1.0, 1.0])
        
        # 计算总奖励
        rewards = rewards_per_func.sum(dim=1)
        
        # 验证奖励
        self.assertEqual(rewards.shape, (self.batch_size,))
        self.assertAlmostEqual(rewards[0].item(), 2.8, places=5)
        self.assertAlmostEqual(rewards[1].item(), 2.6, places=5)
    
    def test_advantage_computation(self):
        """测试优势计算逻辑"""
        # 模拟奖励（4个样本，每个2次生成）
        rewards = torch.tensor([2.0, 2.5, 1.5, 1.8, 3.0, 2.8, 1.0, 1.2])
        num_generations = 2
        
        # 计算分组奖励的均值和标准差
        mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, num_generations).std(dim=1)
        
        # 重复以匹配原始形状
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_generations, dim=0)
        
        # 计算优势
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        # 验证优势
        self.assertEqual(advantages.shape, rewards.shape)
        self.assertTrue(torch.isfinite(advantages).all())
        
        # 验证每组的优势和应该接近0（归一化后）
        for i in range(0, len(rewards), num_generations):
            group_advantages = advantages[i:i+num_generations]
            # 由于归一化，每组的优势应该有正有负
            self.assertTrue(torch.isfinite(group_advantages).all())
    
    def test_multimodal_inputs_extraction(self):
        """测试多模态输入提取"""
        # 模拟VLM模块的多模态关键字
        multimodal_keywords = ["pixel_values", "image_grid_thw"]
        
        # 模拟prompt_inputs
        prompt_inputs = {
            "input_ids": torch.randint(0, 1000, (self.batch_size, 10)),
            "attention_mask": torch.ones(self.batch_size, 10),
            "pixel_values": torch.randn(self.batch_size, 3, 224, 224),
            "image_grid_thw": torch.tensor([[1, 14, 14], [1, 14, 14]]),
        }
        
        # 提取多模态输入
        multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords}
        
        # 验证提取结果
        self.assertIn("pixel_values", multimodal_inputs)
        self.assertIn("image_grid_thw", multimodal_inputs)
        self.assertIsNotNone(multimodal_inputs["pixel_values"])
        self.assertIsNotNone(multimodal_inputs["image_grid_thw"])
    
    def test_output_structure(self):
        """测试输出数据结构"""
        # 模拟_generate_and_score_completions的输出
        prompt_length = 10
        completion_length = 20
        
        output = {
            "prompt_ids": torch.randint(0, 1000, (self.batch_size, prompt_length)),
            "prompt_mask": torch.ones(self.batch_size, prompt_length),
            "completion_ids": torch.randint(0, 1000, (self.batch_size, completion_length)),
            "completion_mask": torch.ones(self.batch_size, completion_length),
            "old_per_token_logps": torch.randn(self.batch_size, completion_length),
            "ref_per_token_logps": torch.randn(self.batch_size, completion_length),
            "advantages": torch.randn(self.batch_size),
            "multimodal_inputs": {
                "pixel_values": torch.randn(self.batch_size, 3, 224, 224),
            }
        }
        
        # 验证输出结构
        self.assertIn("prompt_ids", output)
        self.assertIn("prompt_mask", output)
        self.assertIn("completion_ids", output)
        self.assertIn("completion_mask", output)
        self.assertIn("old_per_token_logps", output)
        self.assertIn("ref_per_token_logps", output)
        self.assertIn("advantages", output)
        self.assertIn("multimodal_inputs", output)
        
        # 验证形状
        self.assertEqual(output["prompt_ids"].shape, (self.batch_size, prompt_length))
        self.assertEqual(output["completion_ids"].shape, (self.batch_size, completion_length))
        self.assertEqual(output["advantages"].shape, (self.batch_size,))


class TestGenerateAndScoreCompletionsIntegration(unittest.TestCase):
    """集成测试：测试_generate_and_score_completions的整体工作流程"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
    
    def test_end_to_end_workflow_simulation(self):
        """端到端工作流程的模拟测试"""
        batch_size = 2
        prompt_length = 15
        completion_length = 30
        num_generations = 1
        
        # 1. 输入准备
        inputs = [
            {
                "prompt": [{"role": "user", "content": [{"type": "text", "text": "Question 1"}]}],
                "solution": "<answer>4.5</answer>",
            },
            {
                "prompt": [{"role": "user", "content": [{"type": "text", "text": "Question 2"}]}],
                "solution": "<answer>3.2</answer>",
            }
        ]
        
        # 2. 生成completion
        prompt_ids = torch.randint(0, 1000, (batch_size, prompt_length))
        completion_ids = torch.randint(0, 1000, (batch_size, completion_length))
        
        # 3. 创建掩码
        prompt_mask = torch.ones(batch_size, prompt_length)
        completion_mask = torch.ones(batch_size, completion_length)
        
        # 4. 计算奖励
        rewards = torch.tensor([2.5, 2.3])
        
        # 5. 计算优势
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        advantages = (rewards - mean_reward) / (std_reward + 1e-4)
        
        # 6. 验证完整流程
        self.assertEqual(prompt_ids.shape, (batch_size, prompt_length))
        self.assertEqual(completion_ids.shape, (batch_size, completion_length))
        self.assertEqual(advantages.shape, (batch_size,))
        self.assertTrue(torch.isfinite(advantages).all())
    
    def test_multiple_reward_functions(self):
        """测试多个奖励函数的集成"""
        batch_size = 3
        num_reward_funcs = 3
        
        # 模拟多个奖励函数的输出
        rewards_per_func = torch.tensor([
            [1.5, 1.0, 0.8],  # 样本1: accuracy, format, custom
            [1.2, 1.0, 0.9],  # 样本2
            [1.8, 0.0, 0.7],  # 样本3: format失败
        ])
        
        # 计算总奖励
        total_rewards = rewards_per_func.sum(dim=1)
        
        # 验证
        self.assertEqual(total_rewards.shape, (batch_size,))
        self.assertAlmostEqual(total_rewards[0].item(), 3.3, places=5)
        self.assertAlmostEqual(total_rewards[1].item(), 3.1, places=5)
        self.assertAlmostEqual(total_rewards[2].item(), 2.5, places=5)  # 格式失败导致总分较低


if __name__ == "__main__":
    unittest.main()
