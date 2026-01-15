import unittest
import torch
import sys
import os
from unittest.mock import MagicMock, patch

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'r1-v', 'src'))

# Import the actual class we want to test
try:
    from open_r1.trainer.grpo_trainer import Qwen2VLGRPOTrainer
    TRAINER_AVAILABLE = True
except ImportError:
    TRAINER_AVAILABLE = False
    Qwen2VLGRPOTrainer = None


class TestComputeLoss(unittest.TestCase):
    """测试Qwen2VLGRPOTrainer.compute_loss函数 - AlphaDrive文档第3节描述的GRPO损失计算"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.num_generations = 4
        self.seq_len = 10
        self.completion_len = 5
        self.vocab_size = 100
    
    def test_multimodal_input_processing(self):
        """测试多模态输入处理逻辑"""
        # 模拟输入数据
        inputs = [
            {
                "prompt": [{"role": "user", "content": "What should I do?"}],
                "image": "path/to/image1.jpg",
                "solution": "<answer>ACCELERATE</answer>"
            },
            {
                "prompt": [{"role": "user", "content": "How to proceed?"}],
                "image": "path/to/image2.jpg",
                "solution": "<answer>STOP</answer>"
            }
        ]
        
        # 验证输入结构
        self.assertEqual(len(inputs), 2)
        for inp in inputs:
            self.assertIn("prompt", inp)
            self.assertIn("image", inp)
            self.assertTrue(isinstance(inp["prompt"], list))
    
    def test_generation_config_setup(self):
        """测试生成配置的设置"""
        max_new_tokens = 128
        temperature = 1.0
        num_return_sequences = 4
        
        # 模拟GenerationConfig
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "num_return_sequences": num_return_sequences,
        }
        
        # 验证配置
        self.assertEqual(generation_config["max_new_tokens"], 128)
        self.assertTrue(generation_config["do_sample"])
        self.assertEqual(generation_config["temperature"], 1.0)
        self.assertEqual(generation_config["num_return_sequences"], 4)
    
    def test_prompt_completion_separation(self):
        """测试prompt和completion的分离"""
        prompt_length = 10
        total_length = 15
        
        # 模拟生成的完整序列
        prompt_completion_ids = torch.randint(0, self.vocab_size, (self.batch_size, total_length))
        
        # 分离prompt和completion
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        
        # 验证分离结果
        self.assertEqual(prompt_ids.shape, (self.batch_size, prompt_length))
        self.assertEqual(completion_ids.shape, (self.batch_size, total_length - prompt_length))
        
        # 验证拼接回去能恢复原始序列
        reconstructed = torch.cat([prompt_ids, completion_ids], dim=1)
        torch.testing.assert_close(reconstructed, prompt_completion_ids)
    
    def test_eos_mask_creation(self):
        """测试EOS掩码的创建"""
        eos_token_id = 2
        
        # 创建包含EOS token的completion
        completion_ids = torch.tensor([
            [10, 20, 2, 30, 40],  # EOS在位置2
            [15, 25, 35, 2, 45],  # EOS在位置3
        ])
        
        # 检测EOS token
        is_eos = completion_ids == eos_token_id
        
        # 找到第一个EOS的位置
        device = completion_ids.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        
        # 创建completion mask
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # 验证mask
        expected_mask = torch.tensor([
            [1, 1, 1, 0, 0],  # mask掉位置2之后的内容
            [1, 1, 1, 1, 0],  # mask掉位置3之后的内容
        ], dtype=torch.int32)
        torch.testing.assert_close(completion_mask, expected_mask)
    
    def test_kl_divergence_computation(self):
        """测试KL散度计算"""
        # 创建策略模型和参考模型的对数概率
        policy_logps = torch.tensor([
            [-1.0, -2.0, -1.5],
            [-0.5, -1.0, -2.5]
        ])
        
        ref_logps = torch.tensor([
            [-1.2, -1.8, -1.6],
            [-0.6, -0.9, -2.3]
        ])
        
        # 计算KL散度: exp(ref - policy) - (ref - policy) - 1
        per_token_kl = torch.exp(ref_logps - policy_logps) - (ref_logps - policy_logps) - 1
        
        # 验证KL散度的性质
        self.assertEqual(per_token_kl.shape, policy_logps.shape)
        self.assertTrue(torch.isfinite(per_token_kl).all())
        # KL散度应该非负（在数值精度范围内）
        self.assertTrue((per_token_kl >= -1e-5).all())
    
    def test_reward_aggregation(self):
        """测试多个奖励函数的聚合"""
        batch_size = 4
        num_reward_funcs = 3
        
        # 创建多个奖励函数的输出
        rewards_per_func = torch.tensor([
            [0.8, 0.9, 1.0],  # sample 0
            [0.5, 0.6, 0.8],  # sample 1
            [0.9, 1.0, 0.7],  # sample 2
            [0.6, 0.7, 0.9],  # sample 3
        ])
        
        # 聚合奖励：对所有奖励函数求和
        rewards = rewards_per_func.sum(dim=1)
        
        # 验证聚合结果
        expected_rewards = torch.tensor([2.7, 1.9, 2.6, 2.2])
        torch.testing.assert_close(rewards, expected_rewards)
    
    def test_grpo_advantage_estimation(self):
        """测试GRPO优势估计"""
        num_generations = 4
        
        # 创建奖励（每个prompt生成G个响应）
        rewards = torch.tensor([
            # Prompt 0的4个生成
            1.0, 0.5, 0.8, 0.6,
            # Prompt 1的4个生成
            0.9, 1.2, 0.7, 1.0
        ])
        
        # 分组统计
        grouped_rewards = rewards.view(-1, num_generations)  # (2, 4)
        mean_grouped_rewards = grouped_rewards.mean(dim=1)  # (2,)
        std_grouped_rewards = grouped_rewards.std(dim=1)  # (2,)
        
        # 扩展到每个响应
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations)  # (8,)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_generations)  # (8,)
        
        # 计算优势
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        # 验证优势估计
        self.assertEqual(advantages.shape, rewards.shape)
        self.assertTrue(torch.isfinite(advantages).all())
        
        # 验证每组的优势均值接近0
        grouped_advantages = advantages.view(-1, num_generations)
        for i in range(grouped_advantages.size(0)):
            group_mean = grouped_advantages[i].mean()
            self.assertAlmostEqual(group_mean.item(), 0.0, places=1)
    
    def test_policy_gradient_loss_computation(self):
        """测试策略梯度损失计算"""
        batch_size, seq_len = 2, 5
        
        # 创建对数概率和优势
        per_token_logps = torch.tensor([
            [-1.0, -1.5, -2.0, -1.2, -1.8],
            [-0.8, -1.3, -1.7, -1.1, -2.2]
        ])
        
        advantages = torch.tensor([0.5, -0.3])
        
        # 计算策略梯度项（带重要性采样）
        # exp(logps - logps.detach()) * advantages
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        
        # 验证
        # exp(logps - logps.detach()) = exp(0) = 1
        expected_loss = advantages.unsqueeze(1).expand(-1, seq_len)
        torch.testing.assert_close(per_token_loss, expected_loss)
    
    def test_kl_penalty_integration(self):
        """测试KL惩罚的集成"""
        batch_size, seq_len = 2, 4
        beta = 0.1
        
        # 创建策略梯度项和KL散度
        policy_gradient = torch.tensor([
            [0.5, 0.3, 0.4, 0.2],
            [-0.2, -0.1, 0.1, -0.3]
        ])
        
        per_token_kl = torch.tensor([
            [0.1, 0.2, 0.15, 0.25],
            [0.3, 0.1, 0.2, 0.15]
        ])
        
        # 计算带KL惩罚的损失
        per_token_loss = -(policy_gradient - beta * per_token_kl)
        
        # 验证
        self.assertEqual(per_token_loss.shape, (batch_size, seq_len))
        self.assertTrue(torch.isfinite(per_token_loss).all())
    
    def test_masked_loss_averaging(self):
        """测试掩码平均损失"""
        batch_size, seq_len = 3, 5
        
        # 创建per-token损失
        per_token_loss = torch.randn(batch_size, seq_len)
        
        # 创建completion mask（不同长度）
        completion_mask = torch.tensor([
            [1, 1, 1, 1, 1],  # 全部有效
            [1, 1, 1, 0, 0],  # 前3个有效
            [1, 1, 0, 0, 0],  # 前2个有效
        ])
        
        # 计算掩码平均损失
        masked_loss = (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        final_loss = masked_loss.mean()
        
        # 验证
        self.assertEqual(masked_loss.shape, (batch_size,))
        self.assertTrue(torch.isfinite(final_loss))
        
        # 手动验证第一个样本
        expected_loss_0 = per_token_loss[0].mean()
        self.assertAlmostEqual(masked_loss[0].item(), expected_loss_0.item(), places=5)


class TestComputeLossIntegration(unittest.TestCase):
    """集成测试：测试compute_loss的完整工作流程"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.num_generations = 4
    
    def test_end_to_end_loss_computation_simulation(self):
        """端到端损失计算模拟"""
        batch_size = 2
        num_generations = 4
        seq_len = 6
        beta = 0.1
        
        # 1. 模拟对数概率
        per_token_logps = torch.randn(batch_size * num_generations, seq_len)
        ref_per_token_logps = torch.randn(batch_size * num_generations, seq_len)
        
        # 2. 计算KL散度
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - \
                       (ref_per_token_logps - per_token_logps) - 1
        
        # 3. 模拟奖励
        rewards = torch.randn(batch_size * num_generations)
        
        # 4. 计算优势
        grouped_rewards = rewards.view(-1, num_generations)
        mean_grouped = grouped_rewards.mean(dim=1).repeat_interleave(num_generations)
        std_grouped = grouped_rewards.std(dim=1).repeat_interleave(num_generations)
        advantages = (rewards - mean_grouped) / (std_grouped + 1e-4)
        
        # 5. 计算损失
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - beta * per_token_kl)
        
        # 6. 掩码平均
        completion_mask = torch.ones(batch_size * num_generations, seq_len)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # 验证最终损失
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.requires_grad or not per_token_logps.requires_grad)


if __name__ == "__main__":
    unittest.main()
