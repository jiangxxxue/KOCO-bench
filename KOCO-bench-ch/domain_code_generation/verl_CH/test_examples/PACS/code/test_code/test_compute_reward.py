import unittest
import torch
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the actual function we want to test
from pacs.pacs_core_algos import compute_reward


class TestComputeReward(unittest.TestCase):
    """测试compute_reward函数 - PACS文档第1节描述的策略改进奖励计算"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_compute_reward_method_1_basic(self):
        """测试方法1的基本功能：计算对数概率差的总和"""
        batch_size, response_length = 2, 4
        beta = 1.0
        
        # 创建对数概率
        old_log_prob = torch.tensor([
            [-0.5, -0.3, -0.4, -0.6],
            [-0.2, -0.5, -0.3, -0.4]
        ])
        log_prob = torch.tensor([
            [-0.3, -0.2, -0.3, -0.5],
            [-0.1, -0.4, -0.2, -0.3]
        ])
        response_mask = torch.ones(batch_size, response_length)
        
        # 调用真实的compute_reward函数
        reward = compute_reward(old_log_prob, log_prob, response_mask, beta=beta, reward_method="1")
        
        # 验证输出形状和属性
        self.assertEqual(reward.shape, (batch_size, 1))
        self.assertTrue(torch.is_tensor(reward))
        self.assertTrue(torch.isfinite(reward).all())
        
        # 手动验证计算
        # 序列1: sum([(-0.3)-(-0.5), (-0.2)-(-0.3), (-0.3)-(-0.4), (-0.5)-(-0.6)])
        #       = sum([0.2, 0.1, 0.1, 0.1]) = 0.5
        self.assertAlmostEqual(reward[0, 0].item(), 0.5, places=5)
        
        # 序列2: sum([(-0.1)-(-0.2), (-0.4)-(-0.5), (-0.2)-(-0.3), (-0.3)-(-0.4)])
        #       = sum([0.1, 0.1, 0.1, 0.1]) = 0.4
        self.assertAlmostEqual(reward[1, 0].item(), 0.4, places=5)
    
    def test_compute_reward_method_1_with_beta(self):
        """测试方法1中beta参数的影响"""
        batch_size, response_length = 1, 3
        
        old_log_prob = torch.tensor([[-0.5, -0.3, -0.4]])
        log_prob = torch.tensor([[-0.3, -0.2, -0.3]])
        response_mask = torch.ones(batch_size, response_length)
        
        # 测试不同的beta值
        betas = [0.5, 1.0, 2.0]
        
        for beta in betas:
            reward = compute_reward(old_log_prob, log_prob, response_mask, beta=beta, reward_method="1")
            
            # 手动计算: sum([0.2, 0.1, 0.1]) = 0.4
            # 乘以beta后: 0.4 * beta
            expected = torch.tensor([[0.4 * beta]])
            torch.testing.assert_close(reward, expected, atol=1e-5, rtol=1e-5)
    
    def test_compute_reward_method_2_basic(self):
        """测试方法2的基本功能：计算归一化的对数概率总和"""
        batch_size, response_length = 2, 4
        beta = 1.0
        
        # 创建对数概率
        log_prob = torch.tensor([
            [-0.2, -0.3, -0.4, -0.5],
            [-0.1, -0.2, -0.3, -0.4]
        ])
        old_log_prob = torch.zeros_like(log_prob)  # 方法2不使用old_log_prob
        response_mask = torch.ones(batch_size, response_length)
        
        # 调用真实的compute_reward函数
        reward = compute_reward(old_log_prob, log_prob, response_mask, beta=beta, reward_method="2")
        
        
        # 验证输出形状和属性
        self.assertEqual(reward.shape, (batch_size, 1))
        self.assertTrue(torch.is_tensor(reward))
        self.assertTrue(torch.isfinite(reward).all())
        
        # 手动验证计算
        # 序列1: sum([-0.2, -0.3, -0.4, -0.5]) / 4 = -1.4 / 4 = -0.35
        self.assertAlmostEqual(reward[0, 0].item(), -0.35, places=5)
        
        # 序列2: sum([-0.1, -0.2, -0.3, -0.4]) / 4 = -1.0 / 4 = -0.25
        self.assertAlmostEqual(reward[1, 0].item(), -0.25, places=5)
    
    def test_compute_reward_method_2_with_mask(self):
        """测试方法2中response_mask的影响"""
        batch_size, response_length = 2, 5
        beta = 1.0
        
        log_prob = torch.tensor([
            [-0.2, -0.3, -0.4, -0.5, -0.6],
            [-0.1, -0.2, -0.3, -0.4, -0.5]
        ])
        old_log_prob = torch.zeros_like(log_prob)
        
        # 创建部分mask
        response_mask = torch.tensor([
            [1.0, 1.0, 1.0, 0.0, 0.0],  # 序列1：前3个有效
            [1.0, 1.0, 1.0, 1.0, 0.0]   # 序列2：前4个有效
        ])
        
        reward = compute_reward(old_log_prob, log_prob, response_mask, beta=beta, reward_method="2")
        
        # 手动验证计算
        # 方法2计算：sum(所有log_prob) / sum(response_mask)
        # 序列1: sum([-0.2, -0.3, -0.4, -0.5, -0.6]) / 3 = -2.0 / 3 = -0.6666...
        self.assertAlmostEqual(reward[0, 0].item(), -2.0 / 3.0, places=5)
        
        # 序列2: sum([-0.1, -0.2, -0.3, -0.4, -0.5]) / 4 = -1.5 / 4 = -0.375
        self.assertAlmostEqual(reward[1, 0].item(), -1.5 / 4.0, places=5)
    
    def test_compute_reward_method_2_with_beta(self):
        """测试方法2中beta参数的影响"""
        batch_size, response_length = 1, 4
        
        log_prob = torch.tensor([[-0.2, -0.3, -0.4, -0.5]])
        old_log_prob = torch.zeros_like(log_prob)
        response_mask = torch.ones(batch_size, response_length)
        
        # 测试不同的beta值
        betas = [0.5, 1.0, 2.0]
        
        for beta in betas:
            reward = compute_reward(old_log_prob, log_prob, response_mask, beta=beta, reward_method="2")
            
            # 手动计算: sum([-0.2, -0.3, -0.4, -0.5]) / 4 = -1.4 / 4 = -0.35
            # 乘以beta后: -0.35 * beta
            expected = torch.tensor([[-0.35 * beta]])
            torch.testing.assert_close(reward, expected, atol=1e-5, rtol=1e-5)
    
    def test_compute_reward_positive_improvement(self):
        """测试策略改进为正的情况（方法1）"""
        batch_size, response_length = 1, 3
        
        # 当前策略的log_prob更大（更好）
        old_log_prob = torch.tensor([[-1.0, -1.2, -0.8]])
        log_prob = torch.tensor([[-0.5, -0.6, -0.4]])
        response_mask = torch.ones(batch_size, response_length)
        
        reward = compute_reward(old_log_prob, log_prob, response_mask, beta=1.0, reward_method="1")
        
        # 所有位置都有改进，奖励应该为正
        self.assertGreater(reward[0, 0].item(), 0.0)
        
        # 手动验证: sum([0.5, 0.6, 0.4]) = 1.5
        expected = torch.tensor([[1.5]])
        torch.testing.assert_close(reward, expected, atol=1e-5, rtol=1e-5)
    
    def test_compute_reward_negative_improvement(self):
        """测试策略退化为负的情况（方法1）"""
        batch_size, response_length = 1, 3
        
        # 当前策略的log_prob更小（更差）
        old_log_prob = torch.tensor([[-0.5, -0.6, -0.4]])
        log_prob = torch.tensor([[-1.0, -1.2, -0.8]])
        response_mask = torch.ones(batch_size, response_length)
        
        reward = compute_reward(old_log_prob, log_prob, response_mask, beta=1.0, reward_method="1")
        
        # 所有位置都退化，奖励应该为负
        self.assertLess(reward[0, 0].item(), 0.0)
        
        # 手动验证: sum([-0.5, -0.6, -0.4]) = -1.5
        expected = torch.tensor([[-1.5]])
        torch.testing.assert_close(reward, expected, atol=1e-5, rtol=1e-5)
    
    def test_compute_reward_edge_cases(self):
        """测试边缘情况"""
        # 测试样例1: 单个token
        single_token_old = torch.tensor([[-0.5]])
        single_token_new = torch.tensor([[-0.3]])
        single_mask = torch.ones(1, 1)
        
        reward_method_1 = compute_reward(single_token_old, single_token_new, single_mask, 
                                         beta=1.0, reward_method="1")
        expected_1 = torch.tensor([[0.2]])
        torch.testing.assert_close(reward_method_1, expected_1, atol=1e-5, rtol=1e-5)
        
        reward_method_2 = compute_reward(single_token_old, single_token_new, single_mask, 
                                         beta=1.0, reward_method="2")
        expected_2 = torch.tensor([[-0.3]])
        torch.testing.assert_close(reward_method_2, expected_2, atol=1e-5, rtol=1e-5)
        
        # 测试样例2: 零对数概率
        zero_old = torch.zeros(1, 3)
        zero_new = torch.zeros(1, 3)
        zero_mask = torch.ones(1, 3)
        
        reward_zero_1 = compute_reward(zero_old, zero_new, zero_mask, beta=1.0, reward_method="1")
        expected_zero_1 = torch.tensor([[0.0]])
        torch.testing.assert_close(reward_zero_1, expected_zero_1)
        
        reward_zero_2 = compute_reward(zero_old, zero_new, zero_mask, beta=1.0, reward_method="2")
        expected_zero_2 = torch.tensor([[0.0]])
        torch.testing.assert_close(reward_zero_2, expected_zero_2)
    
    def test_compute_reward_batch_processing(self):
        """测试批处理功能"""
        batch_size, response_length = 4, 5
        
        old_log_prob = torch.randn(batch_size, response_length)
        log_prob = torch.randn(batch_size, response_length)
        response_mask = torch.ones(batch_size, response_length)
        
        # 方法1批处理
        reward_1 = compute_reward(old_log_prob, log_prob, response_mask, beta=1.0, reward_method="1")
        self.assertEqual(reward_1.shape, (batch_size, 1))
        self.assertTrue(torch.isfinite(reward_1).all())
        
        # 方法2批处理
        reward_2 = compute_reward(old_log_prob, log_prob, response_mask, beta=1.0, reward_method="2")
        self.assertEqual(reward_2.shape, (batch_size, 1))
        self.assertTrue(torch.isfinite(reward_2).all())
        
        # 验证批处理结果与逐个处理一致
        for i in range(batch_size):
            single_reward_1 = compute_reward(
                old_log_prob[i:i+1], log_prob[i:i+1], response_mask[i:i+1],
                beta=1.0, reward_method="1"
            )
            torch.testing.assert_close(reward_1[i, 0], single_reward_1[0, 0])
    
    def test_compute_reward_invalid_method(self):
        """测试无效的reward_method参数"""
        old_log_prob = torch.tensor([[-0.5, -0.3]])
        log_prob = torch.tensor([[-0.3, -0.2]])
        response_mask = torch.ones(1, 2)
        
        # 应该抛出ValueError
        with self.assertRaises(ValueError) as context:
            compute_reward(old_log_prob, log_prob, response_mask, beta=1.0, reward_method="3")
        
        self.assertIn("Invalid reward method", str(context.exception))


class TestComputeRewardIntegration(unittest.TestCase):
    """集成测试：测试compute_reward在实际场景中的应用"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_reward_method_comparison(self):
        """比较两种奖励计算方法的差异"""
        batch_size, response_length = 2, 4
        
        old_log_prob = torch.tensor([
            [-0.5, -0.3, -0.4, -0.6],
            [-0.2, -0.5, -0.3, -0.4]
        ])
        log_prob = torch.tensor([
            [-0.3, -0.2, -0.3, -0.5],
            [-0.1, -0.4, -0.2, -0.3]
        ])
        response_mask = torch.ones(batch_size, response_length)
        beta = 1.0
        
        # 计算两种方法的奖励
        reward_1 = compute_reward(old_log_prob, log_prob, response_mask, beta=beta, reward_method="1")
        reward_2 = compute_reward(old_log_prob, log_prob, response_mask, beta=beta, reward_method="2")
        # print(reward_1)
        # print(reward_2)

        expected_reward_1 = torch.tensor([[0.5000],
        [0.4000]])
        expected_reward_2 = torch.tensor([[-0.3250],
        [-0.2500]])
        torch.testing.assert_close(reward_1, expected_reward_1, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(reward_2, expected_reward_2, atol=1e-5, rtol=1e-5)
        # 验证两种方法产生不同的结果
        self.assertFalse(torch.equal(reward_1, reward_2))
        
        # 验证两种方法都产生有效输出
        self.assertTrue(torch.isfinite(reward_1).all())
        self.assertTrue(torch.isfinite(reward_2).all())
    
    def test_reward_scaling_with_beta(self):
        """测试beta参数对奖励缩放的影响"""
        old_log_prob = torch.tensor([[-0.5, -0.3, -0.4]])
        log_prob = torch.tensor([[-0.3, -0.2, -0.3]])
        response_mask = torch.ones(1, 3)
        
        # 测试beta的线性缩放效应
        beta_1 = 1.0
        beta_2 = 2.0
        
        reward_beta_1 = compute_reward(old_log_prob, log_prob, response_mask, 
                                       beta=beta_1, reward_method="1")
        reward_beta_2 = compute_reward(old_log_prob, log_prob, response_mask, 
                                       beta=beta_2, reward_method="1")
        
        # beta=2的奖励应该是beta=1的两倍
        expected_ratio = beta_2 / beta_1
        actual_ratio = reward_beta_2[0].item() / reward_beta_1[0].item()
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=5)


if __name__ == "__main__":
    unittest.main()
