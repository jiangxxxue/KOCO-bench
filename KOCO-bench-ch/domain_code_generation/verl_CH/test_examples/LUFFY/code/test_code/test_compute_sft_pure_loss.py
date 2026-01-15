import unittest
import torch
import sys
import os

# Add the verl directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'luffy', 'verl'))

# Import the actual function we want to test
from verl.mix_src.mix_core_alg import compute_sft_pure_loss


class TestComputeSftPureLoss(unittest.TestCase):
    """测试compute_sft_pure_loss函数 - LUFFY文档第3节描述的纯SFT损失计算"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_basic_functionality(self):
        """测试基本功能：计算负对数似然损失"""
        batch_size, response_length = 2, 4
        
        # 创建log概率
        log_prob = torch.tensor([
            [-0.5, -0.3, -0.4, -0.6],
            [-0.2, -0.5, -0.3, -0.4]
        ])
        eos_mask = torch.ones(batch_size, response_length)
        
        # 调用真实函数
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # 验证输出是标量
        self.assertTrue(torch.is_tensor(sft_loss))
        self.assertEqual(sft_loss.dim(), 0)
        self.assertTrue(torch.isfinite(sft_loss))
        
        # 验证损失为正（因为是负log概率）
        self.assertGreater(sft_loss.item(), 0.0)
        
        # 手动验证计算
        # SFT loss = mean(-log_prob) = mean([0.5, 0.3, 0.4, 0.6, 0.2, 0.5, 0.3, 0.4])
        expected_loss = (0.5 + 0.3 + 0.4 + 0.6 + 0.2 + 0.5 + 0.3 + 0.4) / 8.0
        self.assertAlmostEqual(sft_loss.item(), expected_loss, places=5)
    
    def test_with_eos_mask(self):
        """测试eos_mask的应用：只在有效token位置计算损失"""
        batch_size, response_length = 2, 5
        
        log_prob = torch.tensor([
            [-0.5, -0.3, -0.4, -0.6, -0.8],
            [-0.2, -0.5, -0.3, -0.4, -0.7]
        ])
        
        # 部分mask
        eos_mask = torch.tensor([
            [1.0, 1.0, 1.0, 0.0, 0.0],  # 前3个有效
            [1.0, 1.0, 1.0, 1.0, 0.0]   # 前4个有效
        ])
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # 验证输出
        self.assertTrue(torch.isfinite(sft_loss))
        self.assertGreater(sft_loss.item(), 0.0)
        
        # 手动验证：只计算有效位置
        # 序列1: [0.5, 0.3, 0.4] (3个有效)
        # 序列2: [0.2, 0.5, 0.3, 0.4] (4个有效)
        # 总共7个有效token
        expected_loss = (0.5 + 0.3 + 0.4 + 0.2 + 0.5 + 0.3 + 0.4) / 7.0
        self.assertAlmostEqual(sft_loss.item(), expected_loss, places=5)
    
    def test_uniform_log_prob(self):
        """测试均匀log概率的情况"""
        batch_size, response_length = 3, 4
        
        # 所有位置相同的log概率
        log_prob = torch.full((batch_size, response_length), -0.5)
        eos_mask = torch.ones(batch_size, response_length)
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # 验证输出
        self.assertTrue(torch.isfinite(sft_loss))
        
        # 均匀log概率时，损失应该等于-log_prob的值
        expected_loss = torch.tensor(0.5)
        torch.testing.assert_close(sft_loss, expected_loss, atol=1e-4, rtol=1e-4)

    
    def test_high_confidence_predictions(self):
        """测试高置信度预测（接近0的log概率）"""
        batch_size, response_length = 2, 3
        
        # 高置信度：log概率接近0
        log_prob = torch.tensor([
            [-0.01, -0.02, -0.015],
            [-0.03, -0.01, -0.025]
        ])
        eos_mask = torch.ones(batch_size, response_length)
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)

        # print(sft_loss)
        expected_loss = torch.tensor(0.0183)
        torch.testing.assert_close(sft_loss, expected_loss, atol=1e-4, rtol=1e-4)
        
        # 高置信度应该产生低损失
        self.assertLess(sft_loss.item(), 0.1)
        self.assertGreater(sft_loss.item(), 0.0)
    
    def test_low_confidence_predictions(self):
        """测试低置信度预测（远离0的log概率）"""
        batch_size, response_length = 2, 3
        
        # 低置信度：log概率远离0
        log_prob = torch.tensor([
            [-2.0, -3.0, -2.5],
            [-3.5, -2.0, -2.8]
        ])
        eos_mask = torch.ones(batch_size, response_length)
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # print(sft_loss)
        expected_loss = torch.tensor(2.6333)
        torch.testing.assert_close(sft_loss, expected_loss, atol=1e-4, rtol=1e-4)
    
    def test_single_token(self):
        """测试单个token的情况"""
        batch_size, response_length = 1, 1
        
        log_prob = torch.tensor([[-0.7]])
        eos_mask = torch.ones(batch_size, response_length)
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # 单个token时，损失应该等于-log_prob
        expected_loss = 0.7
        self.assertAlmostEqual(sft_loss.item(), expected_loss, places=5)
    
    def test_all_masked(self):
        """测试所有token都被mask的情况"""
        batch_size, response_length = 2, 3
        
        log_prob = torch.tensor([
            [-0.5, -0.3, -0.4],
            [-0.2, -0.6, -0.3]
        ])
        eos_mask = torch.zeros(batch_size, response_length)  # 全部mask
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # print(sft_loss)
        expected_loss = torch.tensor(float('nan'))
        torch.testing.assert_close(sft_loss, expected_loss, equal_nan=True)
    
    def test_batch_processing(self):
        """测试批处理功能"""
        batch_size, response_length = 4, 6
        
        log_prob = torch.randn(batch_size, response_length) * 0.5 - 1.0
        eos_mask = torch.ones(batch_size, response_length)
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # print(sft_loss)
        expected_loss = torch.tensor(1.1210)
        torch.testing.assert_close(sft_loss, expected_loss, atol=1e-4, rtol=1e-4)
    
    def test_varying_sequence_lengths(self):
        """测试不同序列长度（通过mask实现）"""
        batch_size, response_length = 3, 5
        
        log_prob = torch.tensor([
            [-0.5, -0.3, -0.4, -0.6, -0.7],
            [-0.2, -0.5, -0.3, -0.4, -0.8],
            [-0.4, -0.6, -0.5, -0.3, -0.9]
        ])
        
        # 不同的有效长度
        eos_mask = torch.tensor([
            [1.0, 1.0, 0.0, 0.0, 0.0],  # 长度2
            [1.0, 1.0, 1.0, 1.0, 0.0],  # 长度4
            [1.0, 1.0, 1.0, 0.0, 0.0]   # 长度3
        ])
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # 验证输出
        self.assertTrue(torch.isfinite(sft_loss))
        self.assertGreater(sft_loss.item(), 0.0)
        
        # 手动验证
        # 序列1: [0.5, 0.3] (2个)
        # 序列2: [0.2, 0.5, 0.3, 0.4] (4个)
        # 序列3: [0.4, 0.6, 0.5] (3个)
        # 总共9个有效token
        expected_loss = (0.5 + 0.3 + 0.2 + 0.5 + 0.3 + 0.4 + 0.4 + 0.6 + 0.5) / 9.0
        self.assertAlmostEqual(sft_loss.item(), expected_loss, places=5)
    
    def test_zero_log_prob(self):
        """测试log概率为0的情况（完美预测）"""
        batch_size, response_length = 2, 3
        
        log_prob = torch.zeros(batch_size, response_length)
        eos_mask = torch.ones(batch_size, response_length)
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # log概率为0时，损失应该为0
        self.assertAlmostEqual(sft_loss.item(), 0.0, places=5)
    
    def test_negative_values_handling(self):
        """测试负log概率值的正确处理"""
        batch_size, response_length = 2, 4
        
        # 负log概率（正常情况）
        log_prob = torch.tensor([
            [-1.2, -0.8, -1.5, -0.9],
            [-1.0, -1.3, -0.7, -1.1]
        ])
        eos_mask = torch.ones(batch_size, response_length)
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # 验证损失为正
        self.assertGreater(sft_loss.item(), 0.0)
        
        # 手动验证
        expected_loss = (1.2 + 0.8 + 1.5 + 0.9 + 1.0 + 1.3 + 0.7 + 1.1) / 8.0
        self.assertAlmostEqual(sft_loss.item(), expected_loss, places=5)


class TestComputeSftPureLossIntegration(unittest.TestCase):
    """集成测试：测试compute_sft_pure_loss在实际场景中的应用"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_realistic_training_scenario(self):
        """测试真实训练场景"""
        batch_size, response_length = 8, 12
        
        # 模拟真实的log概率分布
        log_prob = torch.randn(batch_size, response_length) * 0.8 - 1.5
        
        # 模拟真实的mask（不同序列有不同长度）
        eos_mask = torch.ones(batch_size, response_length)
        for i in range(batch_size):
            # 随机mask掉一些后续token
            mask_start = torch.randint(response_length // 2, response_length, (1,)).item()
            eos_mask[i, mask_start:] = 0.0
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # print(sft_loss)
        expected_loss = torch.tensor(1.4558)
        torch.testing.assert_close(sft_loss, expected_loss, atol=1e-4, rtol=1e-4)
    
    def test_comparison_with_manual_calculation(self):
        """测试与手动计算的一致性"""
        batch_size, response_length = 3, 4
        
        log_prob = torch.tensor([
            [-0.5, -0.3, -0.4, -0.6],
            [-0.2, -0.5, -0.3, -0.4],
            [-0.4, -0.6, -0.5, -0.3]
        ])
        eos_mask = torch.ones(batch_size, response_length)
        
        # 使用函数计算
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # 手动计算
        manual_sft_losses = -log_prob
        manual_sft_loss = (manual_sft_losses * eos_mask).sum() / eos_mask.sum()
        
        # 验证一致性
        torch.testing.assert_close(sft_loss, manual_sft_loss, atol=1e-6, rtol=1e-6)
    
    def test_loss_magnitude_with_different_confidences(self):
        """测试不同置信度下的损失大小关系"""
        batch_size, response_length = 2, 3
        
        # 高置信度
        high_confidence_log_prob = torch.tensor([
            [-0.1, -0.05, -0.08],
            [-0.06, -0.09, -0.07]
        ])
        
        # 低置信度
        low_confidence_log_prob = torch.tensor([
            [-2.0, -1.5, -1.8],
            [-1.6, -1.9, -1.7]
        ])
        
        eos_mask = torch.ones(batch_size, response_length)
        
        high_conf_loss = compute_sft_pure_loss(high_confidence_log_prob, eos_mask)
        low_conf_loss = compute_sft_pure_loss(low_confidence_log_prob, eos_mask)
        
        # print(high_conf_loss)
        # print(low_conf_loss)
        # 低置信度应该产生更高的损失
        expected_loss = torch.tensor(0.0750)
        torch.testing.assert_close(high_conf_loss, expected_loss, atol=1e-4, rtol=1e-4)
        expected_loss = torch.tensor(1.7500)
        torch.testing.assert_close(low_conf_loss, expected_loss, atol=1e-4, rtol=1e-4)
    
    def test_gradient_flow(self):
        """测试梯度流动（确保损失可以反向传播）"""
        batch_size, response_length = 2, 4
        
        # 创建需要梯度的log概率
        log_prob = torch.randn(batch_size, response_length, requires_grad=True)
        eos_mask = torch.ones(batch_size, response_length)
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # 反向传播
        sft_loss.backward()

        # print(log_prob.grad)
        expected_grad = torch.tensor([[-0.1250, -0.1250, -0.1250, -0.1250],
        [-0.1250, -0.1250, -0.1250, -0.1250]])
        torch.testing.assert_close(log_prob.grad, expected_grad, atol=1e-4, rtol=1e-4)
    
    def test_multi_task_learning_scenario(self):
        """测试多任务学习场景：SFT作为辅助损失"""
        batch_size, response_length = 4, 6
        
        # 模拟off-policy样本的log概率
        off_policy_log_prob = torch.tensor([[-0.0365, -0.2564, -0.5496, -2.0528, -0.6608, -1.6173],
        [-1.0215, -1.8023, -0.8221, -1.3433, -1.2467, -0.8793],
        [-1.5555, -0.9542, -2.1585, -1.1084, -1.1549, -1.1979],
        [-0.5983, -1.3108, -1.2960, -1.0315, -1.4143, -0.8346]])

        # print(off_policy_log_prob)
        eos_mask = torch.ones(batch_size, response_length)
        
        # 计算SFT损失
        sft_loss = compute_sft_pure_loss(off_policy_log_prob, eos_mask)
        
        # print(sft_loss)
        expected_loss = torch.tensor(1.1210)
        torch.testing.assert_close(sft_loss, expected_loss, atol=1e-4, rtol=1e-4)
        
        # 模拟与其他损失结合
        mock_rl_loss = torch.tensor(0.5)
        combined_loss = sft_loss + 0.1 * mock_rl_loss
        
        self.assertTrue(torch.isfinite(combined_loss))


if __name__ == "__main__":
    unittest.main()
