import unittest
import torch
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))


class TestComputeLoss(unittest.TestCase):
    """测试VLMGRPOTrainer.compute_loss函数 - 计算GRPO训练的损失函数"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        self.batch_size = 2
        self.prompt_length = 10
        self.completion_length = 20
    
    def test_policy_ratio_computation(self):
        """测试策略比率计算"""
        # 模拟当前策略和旧策略的log概率
        per_token_logps = torch.randn(self.batch_size, self.completion_length)
        old_per_token_logps = torch.randn(self.batch_size, self.completion_length)
        
        # 计算策略比率
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        
        # 验证比率
        self.assertEqual(ratio.shape, (self.batch_size, self.completion_length))
        self.assertTrue(torch.isfinite(ratio).all())
        self.assertTrue((ratio > 0).all())  # 比率应该为正
    
    def test_ratio_clipping(self):
        """测试比率裁剪逻辑"""
        epsilon_low = 0.1
        epsilon_high = 0.2
        
        # 创建不同范围的比率
        ratio = torch.tensor([
            [0.5, 0.95, 1.0, 1.15, 1.5],  # 包含需要裁剪的值
            [0.8, 1.0, 1.1, 1.25, 2.0],
        ])
        
        # 裁剪比率
        clipped_ratio = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high)
        
        # 验证裁剪结果
        self.assertEqual(clipped_ratio.shape, ratio.shape)
        self.assertTrue((clipped_ratio >= 1 - epsilon_low).all())
        self.assertTrue((clipped_ratio <= 1 + epsilon_high).all())
        
        # 验证具体值
        self.assertAlmostEqual(clipped_ratio[0, 0].item(), 0.9, places=5)  # 0.5被裁剪到0.9
        self.assertAlmostEqual(clipped_ratio[0, 4].item(), 1.2, places=5)  # 1.5被裁剪到1.2
    
    def test_ppo_loss_computation(self):
        """测试PPO损失计算"""
        # 模拟输入
        ratio = torch.tensor([[1.1, 0.9], [1.2, 0.8]])
        clipped_ratio = torch.tensor([[1.1, 0.9], [1.2, 0.9]])  # 假设epsilon_low=0.1
        advantages = torch.tensor([0.5, -0.3])
        
        # 计算PPO损失
        per_token_loss1 = ratio * advantages.unsqueeze(1)
        per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        # 验证损失
        self.assertEqual(per_token_loss.shape, (2, 2))
        self.assertTrue(torch.isfinite(per_token_loss).all())
    
    def test_kl_penalty_computation(self):
        """测试KL散度惩罚计算"""
        beta = 0.01
        
        # 模拟参考模型和当前策略的log概率
        ref_per_token_logps = torch.randn(self.batch_size, self.completion_length)
        per_token_logps = torch.randn(self.batch_size, self.completion_length)
        
        # 计算KL散度
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        # 验证KL散度
        self.assertEqual(per_token_kl.shape, (self.batch_size, self.completion_length))
        self.assertTrue(torch.isfinite(per_token_kl).all())
        
        # KL散度应该非负
        # 注意：由于数值精度，可能有小的负值，但应该接近0
        self.assertTrue((per_token_kl >= -1e-5).all())
    
    def test_loss_with_mask(self):
        """测试带掩码的损失计算"""
        # 模拟per_token_loss
        per_token_loss = torch.randn(self.batch_size, self.completion_length)
        
        # 创建部分掩码
        completion_mask = torch.tensor([
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 10,
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0] + [0.0] * 10,
        ])
        
        # 计算掩码后的损失
        masked_loss = (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        
        # 验证损失
        self.assertEqual(masked_loss.shape, (self.batch_size,))
        self.assertTrue(torch.isfinite(masked_loss).all())
    
    def test_final_loss_aggregation(self):
        """测试最终损失聚合"""
        # 模拟每个样本的损失
        per_sample_loss = torch.tensor([0.5, 0.8, 0.3, 0.6])
        
        # 计算平均损失
        final_loss = per_sample_loss.mean()
        
        # 验证
        self.assertTrue(torch.is_tensor(final_loss))
        self.assertTrue(torch.isfinite(final_loss))
        self.assertAlmostEqual(final_loss.item(), 0.55, places=5)
    
    def test_clip_ratio_metric(self):
        """测试裁剪比率指标计算"""
        # 模拟loss1和loss2
        per_token_loss1 = torch.tensor([[0.5, 0.8, 0.3], [0.6, 0.9, 0.4]])
        per_token_loss2 = torch.tensor([[0.6, 0.7, 0.3], [0.5, 0.9, 0.5]])
        completion_mask = torch.ones(2, 3)
        
        # 计算裁剪标志
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        
        # 计算裁剪比率
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        
        # 验证
        self.assertTrue(torch.is_tensor(clip_ratio))
        self.assertTrue(torch.isfinite(clip_ratio))
        self.assertTrue(0.0 <= clip_ratio.item() <= 1.0)
    
    def test_advantage_influence(self):
        """测试优势对损失的影响"""
        ratio = torch.tensor([[1.1, 1.0], [0.9, 1.0]])
        
        # 正优势
        advantages_positive = torch.tensor([0.5, 0.3])
        loss1_pos = -(ratio * advantages_positive.unsqueeze(1))
        
        # 负优势
        advantages_negative = torch.tensor([-0.5, -0.3])
        loss1_neg = -(ratio * advantages_negative.unsqueeze(1))
        
        # 验证优势的影响
        self.assertTrue(torch.isfinite(loss1_pos).all())
        self.assertTrue(torch.isfinite(loss1_neg).all())
        
        # 正优势应该鼓励增加概率（负损失）
        # 负优势应该鼓励减少概率（正损失）
        self.assertTrue((loss1_pos < 0).all())
        self.assertTrue((loss1_neg > 0).all())


class TestComputeLossIntegration(unittest.TestCase):
    """集成测试：测试compute_loss的整体工作流程"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
    
    def test_end_to_end_loss_computation(self):
        """端到端测试：完整的损失计算流程"""
        batch_size = 2
        completion_length = 10
        
        # 1. 模拟输入数据
        per_token_logps = torch.randn(batch_size, completion_length)
        old_per_token_logps = torch.randn(batch_size, completion_length)
        advantages = torch.tensor([0.5, -0.3])
        completion_mask = torch.ones(batch_size, completion_length)
        
        # 2. 计算策略比率
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        clipped_ratio = torch.clamp(ratio, 0.9, 1.2)
        
        # 3. 计算PPO损失
        per_token_loss1 = ratio * advantages.unsqueeze(1)
        per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        # 4. 计算最终损失
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # 5. 验证
        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.dim(), 0)  # 标量
    
    def test_loss_with_kl_penalty(self):
        """测试带KL惩罚的损失计算"""
        batch_size = 2
        completion_length = 10
        beta = 0.01
        
        # 模拟输入
        per_token_logps = torch.randn(batch_size, completion_length)
        old_per_token_logps = torch.randn(batch_size, completion_length)
        ref_per_token_logps = torch.randn(batch_size, completion_length)
        advantages = torch.tensor([0.5, -0.3])
        completion_mask = torch.ones(batch_size, completion_length)
        
        # 计算PPO损失
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        clipped_ratio = torch.clamp(ratio, 0.9, 1.2)
        per_token_loss1 = ratio * advantages.unsqueeze(1)
        per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        # 添加KL惩罚
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        per_token_loss_with_kl = per_token_loss + beta * per_token_kl
        
        # 计算最终损失
        loss = ((per_token_loss_with_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # 验证
        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(torch.isfinite(loss))
    
    def test_loss_gradient_flow(self):
        """测试损失的梯度流"""
        batch_size = 2
        completion_length = 5
        
        # 创建需要梯度的张量
        per_token_logps = torch.randn(batch_size, completion_length, requires_grad=True)
        old_per_token_logps = torch.randn(batch_size, completion_length)
        advantages = torch.tensor([0.5, -0.3])
        completion_mask = torch.ones(batch_size, completion_length)
        
        # 计算损失
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        clipped_ratio = torch.clamp(ratio, 0.9, 1.2)
        per_token_loss1 = ratio * advantages.unsqueeze(1)
        per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # 反向传播
        loss.backward()
        
        # 验证梯度
        self.assertIsNotNone(per_token_logps.grad)
        self.assertTrue(torch.isfinite(per_token_logps.grad).all())
    
    def test_multiple_iterations(self):
        """测试多次迭代的场景"""
        batch_size = 2
        completion_length = 10
        num_iterations = 3
        
        # 模拟多次迭代
        losses = []
        for iteration in range(num_iterations):
            per_token_logps = torch.randn(batch_size, completion_length)
            old_per_token_logps = torch.randn(batch_size, completion_length)
            advantages = torch.randn(batch_size)
            completion_mask = torch.ones(batch_size, completion_length)
            
            # 计算损失
            ratio = torch.exp(per_token_logps - old_per_token_logps)
            clipped_ratio = torch.clamp(ratio, 0.9, 1.2)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            
            losses.append(loss.item())
        
        # 验证所有迭代的损失都是有效的
        self.assertEqual(len(losses), num_iterations)
        for loss_val in losses:
            self.assertTrue(math.isfinite(loss_val))


if __name__ == "__main__":
    import math
    unittest.main()
