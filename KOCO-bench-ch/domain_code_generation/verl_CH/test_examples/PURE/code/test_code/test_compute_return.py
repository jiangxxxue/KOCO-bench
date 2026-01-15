import unittest
import torch
import sys
import os

# Add the verl directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'verl'))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the actual function we want to test
from verl.trainer.ppo.core_algos import compute_return


class TestComputeReturn(unittest.TestCase):
    """测试compute_return函数 - PURE文档第3节描述的token级回报序列计算"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_compute_return_sum_method_basic(self):
        """测试sum方法的基本功能"""
        batch_size, response_length = 2, 4
        gamma = 0.9
        
        # 创建token级奖励
        token_level_rewards = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],    # 序列1
            [0.5, 1.5, 2.5, 3.5]     # 序列2
        ])
        eos_mask = torch.ones(batch_size, response_length)
        
        # 调用真实的compute_return函数
        returns = compute_return(token_level_rewards, eos_mask, method='sum', gamma=gamma)
        
        # 验证输出形状和属性
        self.assertEqual(returns.shape, (batch_size, response_length))
        self.assertTrue(torch.is_tensor(returns))
        self.assertTrue(torch.isfinite(returns).all())
        
        # 手动验证第一个序列的计算
        # Position 3: return = 4.0
        # Position 2: return = 3.0 + 0.9 * 4.0 = 3.0 + 3.6 = 6.6
        # Position 1: return = 2.0 + 0.9 * 6.6 = 2.0 + 5.94 = 7.94
        # Position 0: return = 1.0 + 0.9 * 7.94 = 1.0 + 7.146 = 8.146
        expected_returns_seq1 = torch.tensor([8.146, 7.94, 6.6, 4.0])
        torch.testing.assert_close(returns[0], expected_returns_seq1, atol=1e-3, rtol=1e-3)
    
    def test_compute_return_sum_method_with_gamma(self):
        """测试不同gamma值对sum方法的影响"""
        batch_size, response_length = 1, 3
        token_level_rewards = torch.tensor([[1.0, 1.0, 1.0]])
        eos_mask = torch.ones(batch_size, response_length)
        
        # 测试gamma = 1.0（无折扣）
        returns_gamma_1 = compute_return(token_level_rewards, eos_mask, method='sum', gamma=1.0)
        
        # gamma=1时，每个位置的return应该是后续所有奖励的和
        expected_gamma_1 = torch.tensor([[3.0, 2.0, 1.0]])
        torch.testing.assert_close(returns_gamma_1, expected_gamma_1)
        
        # 测试gamma = 0.0（无传播）
        returns_gamma_0 = compute_return(token_level_rewards, eos_mask, method='sum', gamma=0.0)
        
        # gamma=0时，每个位置的return应该就是当前奖励
        expected_gamma_0 = torch.tensor([[1.0, 1.0, 1.0]])
        torch.testing.assert_close(returns_gamma_0, expected_gamma_0)
        
        # 测试gamma = 0.5（中等折扣）
        returns_gamma_05 = compute_return(token_level_rewards, eos_mask, method='sum', gamma=0.5)
        
        # gamma=0.5时的手动计算
        # Position 2: return = 1.0
        # Position 1: return = 1.0 + 0.5 * 1.0 = 1.5
        # Position 0: return = 1.0 + 0.5 * 1.5 = 1.75
        expected_gamma_05 = torch.tensor([[1.75, 1.5, 1.0]])
        torch.testing.assert_close(returns_gamma_05, expected_gamma_05)
    
    def test_compute_return_min_method_basic(self):
        """测试min方法的基本功能"""
        batch_size, response_length = 2, 4
        
        # 创建token级奖励
        token_level_rewards = torch.tensor([
            [3.0, 1.0, 4.0, 2.0],    # 序列1：最小值是1.0在位置1
            [2.5, 3.5, 1.5, 4.5]     # 序列2：最小值是1.5在位置2
        ])
        eos_mask = torch.ones(batch_size, response_length)
        
        # 调用真实的compute_return函数
        returns = compute_return(token_level_rewards, eos_mask, method='min')
        
        # 验证输出形状和属性
        self.assertEqual(returns.shape, (batch_size, response_length))
        self.assertTrue(torch.is_tensor(returns))
        self.assertTrue(torch.isfinite(returns).all())
        
        # 手动验证第一个序列的计算
        # Position 3: return = 2.0
        # Position 2: return = min(4.0, 2.0) = 2.0
        # Position 1: return = min(1.0, 2.0) = 1.0
        # Position 0: return = min(3.0, 1.0) = 1.0
        expected_returns_seq1 = torch.tensor([1.0, 1.0, 2.0, 2.0])
        torch.testing.assert_close(returns[0], expected_returns_seq1)
        
        # 验证第二个序列
        # Position 3: return = 4.5
        # Position 2: return = min(1.5, 4.5) = 1.5
        # Position 1: return = min(3.5, 1.5) = 1.5
        # Position 0: return = min(2.5, 1.5) = 1.5
        expected_returns_seq2 = torch.tensor([1.5, 1.5, 1.5, 4.5])
        torch.testing.assert_close(returns[1], expected_returns_seq2)
    
    def test_compute_return_with_eos_mask(self):
        """测试eos_mask对计算的影响"""
        batch_size, response_length = 2, 5
        
        # 创建token级奖励
        token_level_rewards = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0],    # 序列1
            [2.0, 1.0, 4.0, 3.0, 2.0]     # 序列2
        ])
        
        # 创建部分mask
        eos_mask = torch.tensor([
            [1.0, 1.0, 1.0, 0.0, 0.0],    # 序列1：前3个有效
            [1.0, 1.0, 1.0, 1.0, 0.0]     # 序列2：前4个有效
        ])
        
        # 测试sum方法与mask
        gamma = 0.8
        returns_sum = compute_return(token_level_rewards, eos_mask, method='sum', gamma=gamma)
        
        # 验证mask位置为0
        self.assertEqual(returns_sum[0, 3].item(), 0.0)
        self.assertEqual(returns_sum[0, 4].item(), 0.0)
        self.assertEqual(returns_sum[1, 4].item(), 0.0)
        
        # 验证有效位置的计算（序列1前3个位置）
        # Position 2: return = 3.0 (最后有效位置)
        # Position 1: return = 2.0 + 0.8 * 3.0 = 2.0 + 2.4 = 4.4
        # Position 0: return = 1.0 + 0.8 * 4.4 = 1.0 + 3.52 = 4.52
        self.assertAlmostEqual(returns_sum[0, 2].item(), 3.0, places=5)
        self.assertAlmostEqual(returns_sum[0, 1].item(), 4.4, places=5)
        self.assertAlmostEqual(returns_sum[0, 0].item(), 4.52, places=5)
        
        # 测试min方法与mask
        returns_min = compute_return(token_level_rewards, eos_mask, method='min')
        
        # 验证mask位置为0
        self.assertEqual(returns_min[0, 3].item(), 0.0)
        self.assertEqual(returns_min[0, 4].item(), 0.0)
        self.assertEqual(returns_min[1, 4].item(), 0.0)
        
        # 验证min计算（序列1前3个位置：[1.0, 2.0, 3.0]）
        # Position 2: return = 3.0
        # Position 1: return = min(2.0, 3.0) = 2.0  
        # Position 0: return = min(1.0, 2.0) = 1.0
        self.assertAlmostEqual(returns_min[0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(returns_min[0, 1].item(), 2.0, places=5)
        self.assertAlmostEqual(returns_min[0, 2].item(), 3.0, places=5)
    
    def test_compute_return_input_output_samples_sum(self):
        """测试sum方法的具体输入输出样例"""
        # 测试样例1: 基本sum计算
        token_rewards = torch.tensor([[2.0, 1.0, 3.0]])
        eos_mask = torch.ones(1, 3)
        gamma = 0.9
        
        # 手动计算expected结果
        # Position 2: return = 3.0
        # Position 1: return = 1.0 + 0.9 * 3.0 = 1.0 + 2.7 = 3.7
        # Position 0: return = 2.0 + 0.9 * 3.7 = 2.0 + 3.33 = 5.33
        
        returns = compute_return(token_rewards, eos_mask, method='sum', gamma=gamma)
        
        expected = torch.tensor([[5.33, 3.7, 3.0]])
        torch.testing.assert_close(returns, expected, atol=1e-2, rtol=1e-2)
        
        # 测试样例2: gamma=1的特殊情况
        token_rewards_2 = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        eos_mask_2 = torch.ones(1, 4)
        gamma_2 = 1.0
        
        returns_2 = compute_return(token_rewards_2, eos_mask_2, method='sum', gamma=gamma_2)
        
        # gamma=1时，每个位置的return是后续所有奖励的和
        expected_2 = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        torch.testing.assert_close(returns_2, expected_2)
        
        # 测试样例3: 带mask的情况
        token_rewards_3 = torch.tensor([[2.0, 3.0, 1.0, 4.0]])
        eos_mask_3 = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # 只有前两个有效
        gamma_3 = 0.8
        
        returns_3 = compute_return(token_rewards_3, eos_mask_3, method='sum', gamma=gamma_3)
        
        # 只计算前两个有效位置
        # Position 1: return = 3.0 (最后有效)
        # Position 0: return = 2.0 + 0.8 * 3.0 = 2.0 + 2.4 = 4.4
        expected_3 = torch.tensor([[4.4, 3.0, 0.0, 0.0]])
        torch.testing.assert_close(returns_3, expected_3)
    
    def test_compute_return_input_output_samples_min(self):
        """测试min方法的具体输入输出样例"""
        # 测试样例1: 基本min计算
        token_rewards = torch.tensor([[5.0, 2.0, 8.0, 3.0]])
        eos_mask = torch.ones(1, 4)
        
        returns = compute_return(token_rewards, eos_mask, method='min')
        
        # 手动计算expected结果
        # Position 3: return = 3.0
        # Position 2: return = min(8.0, 3.0) = 3.0
        # Position 1: return = min(2.0, 3.0) = 2.0
        # Position 0: return = min(5.0, 2.0) = 2.0
        expected = torch.tensor([[2.0, 2.0, 3.0, 3.0]])
        torch.testing.assert_close(returns, expected)
        
        # 测试样例2: 递减序列
        token_rewards_2 = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        eos_mask_2 = torch.ones(1, 4)
        
        returns_2 = compute_return(token_rewards_2, eos_mask_2, method='min')
        
        # 递减序列中，每个位置的min就是最右边的值
        expected_2 = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        torch.testing.assert_close(returns_2, expected_2)
        
        # 测试样例3: 递增序列
        token_rewards_3 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        eos_mask_3 = torch.ones(1, 4)
        
        returns_3 = compute_return(token_rewards_3, eos_mask_3, method='min')
        
        # 递增序列中，每个位置的min就是该位置的值
        expected_3 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        torch.testing.assert_close(returns_3, expected_3)
        
        # 测试样例4: 带mask的min计算
        token_rewards_4 = torch.tensor([[3.0, 1.0, 5.0, 2.0]])
        eos_mask_4 = torch.tensor([[1.0, 1.0, 1.0, 0.0]])  # 最后一个位置masked
        
        returns_4 = compute_return(token_rewards_4, eos_mask_4, method='min')
        
        # 只考虑前3个位置：[3.0, 1.0, 5.0]
        # Position 2: return = 5.0 (最后有效)
        # Position 1: return = min(1.0, 5.0) = 1.0
        # Position 0: return = min(3.0, 1.0) = 1.0
        expected_4 = torch.tensor([[1.0, 1.0, 5.0, 0.0]])
        torch.testing.assert_close(returns_4, expected_4)
    
    def test_compute_return_edge_cases(self):
        """测试边缘情况"""
        # 测试样例1: 单个token
        single_token = torch.tensor([[5.0]])
        single_mask = torch.ones(1, 1)
        
        # sum方法
        returns_sum = compute_return(single_token, single_mask, method='sum')
        expected_sum = torch.tensor([[5.0]])
        torch.testing.assert_close(returns_sum, expected_sum)
        
        # min方法
        returns_min = compute_return(single_token, single_mask, method='min')
        expected_min = torch.tensor([[5.0]])
        torch.testing.assert_close(returns_min, expected_min)
        
        # 测试样例2: 全零奖励
        zero_rewards = torch.zeros(1, 3)
        zero_mask = torch.ones(1, 3)
        gamma = 0.9
        
        # sum方法应该返回全零
        returns_sum_zero = compute_return(zero_rewards, zero_mask, method='sum', gamma=gamma)
        expected_zero_sum = torch.zeros(1, 3)
        torch.testing.assert_close(returns_sum_zero, expected_zero_sum)
        
        # min方法应该返回全零
        returns_min_zero = compute_return(zero_rewards, zero_mask, method='min')
        expected_zero_min = torch.zeros(1, 3)
        torch.testing.assert_close(returns_min_zero, expected_zero_min)
        
        # 测试样例3: 全mask序列
        all_masked_rewards = torch.tensor([[1.0, 2.0, 3.0]])
        all_zero_mask = torch.zeros(1, 3)
        
        # sum方法应该返回全零
        returns_sum_masked = compute_return(all_masked_rewards, all_zero_mask, method='sum', gamma=gamma)
        expected_masked_sum = torch.zeros(1, 3)
        torch.testing.assert_close(returns_sum_masked, expected_masked_sum)
        
        # min方法应该返回全零
        returns_min_masked = compute_return(all_masked_rewards, all_zero_mask, method='min')
        expected_masked_min = torch.zeros(1, 3)
        torch.testing.assert_close(returns_min_masked, expected_masked_min)
    
    def test_compute_return_multiple_sequences(self):
        """测试多序列批处理"""
        batch_size, response_length = 3, 4
        
        # 创建多个不同的序列
        token_rewards = torch.tensor([
            [1.0, 2.0, 1.0, 3.0],    # 序列1
            [2.0, 1.0, 3.0, 1.0],    # 序列2  
            [3.0, 1.0, 2.0, 1.0]     # 序列3
        ])
        eos_mask = torch.ones(batch_size, response_length)
        gamma = 0.8
        
        # sum方法批处理
        returns_sum = compute_return(token_rewards, eos_mask, method='sum', gamma=gamma)
        
        # 验证每个序列的形状
        self.assertEqual(returns_sum.shape, (batch_size, response_length))
        
        # 手动验证第一个序列 [1.0, 2.0, 1.0, 3.0]
        # Position 3: return = 3.0
        # Position 2: return = 1.0 + 0.8 * 3.0 = 1.0 + 2.4 = 3.4
        # Position 1: return = 2.0 + 0.8 * 3.4 = 2.0 + 2.72 = 4.72
        # Position 0: return = 1.0 + 0.8 * 4.72 = 1.0 + 3.776 = 4.776
        expected_seq1 = torch.tensor([4.776, 4.72, 3.4, 3.0])
        torch.testing.assert_close(returns_sum[0], expected_seq1, atol=1e-3, rtol=1e-3)
        
        # min方法批处理
        returns_min = compute_return(token_rewards, eos_mask, method='min')
        
        # 验证每个序列的形状
        self.assertEqual(returns_min.shape, (batch_size, response_length))
        
        # 手动验证第一个序列 [1.0, 2.0, 1.0, 3.0] 的min计算
        # Position 3: return = 3.0
        # Position 2: return = min(1.0, 3.0) = 1.0
        # Position 1: return = min(2.0, 1.0) = 1.0
        # Position 0: return = min(1.0, 1.0) = 1.0
        expected_min_seq1 = torch.tensor([1.0, 1.0, 1.0, 3.0])
        torch.testing.assert_close(returns_min[0], expected_min_seq1)
        
        # 验证第二个序列 [2.0, 1.0, 3.0, 1.0] 的min计算
        # Position 3: return = 1.0
        # Position 2: return = min(3.0, 1.0) = 1.0
        # Position 1: return = min(1.0, 1.0) = 1.0
        # Position 0: return = min(2.0, 1.0) = 1.0
        expected_min_seq2 = torch.tensor([1.0, 1.0, 1.0, 1.0])
        torch.testing.assert_close(returns_min[1], expected_min_seq2)


class TestComputeReturnIntegration(unittest.TestCase):
    """集成测试：测试compute_return与其他组件的集成"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_method_parameter_switching(self):
        """测试method参数切换功能"""
        token_rewards = torch.tensor([[3.0, 1.0, 4.0, 2.0]])
        eos_mask = torch.ones(1, 4)
        
        # 直接使用真实的compute_return函数
        
        # 测试sum方法
        returns_sum = compute_return(token_rewards, eos_mask, method='sum', gamma=0.9)
        self.assertEqual(returns_sum.shape, (1, 4))
        self.assertTrue(torch.isfinite(returns_sum).all())
        
        # 测试min方法
        returns_min = compute_return(token_rewards, eos_mask, method='min')
        self.assertEqual(returns_min.shape, (1, 4))
        self.assertTrue(torch.isfinite(returns_min).all())
        
        # 验证两种方法产生不同结果
        self.assertFalse(torch.equal(returns_sum, returns_min))
        
        # 测试无效方法 - 应该抛出ValueError（参数值无效）
        # 注：使用ValueError而非NotImplementedError更符合语义
        # ValueError: 参数值不在允许的范围内
        # NotImplementedError: 功能尚未实现（这里不适用）
        with self.assertRaises(NotImplementedError) as context:
            compute_return(token_rewards, eos_mask, method='invalid')
    
    def test_integration_with_advantage_computation(self):
        """测试与优势计算的集成"""
        batch_size, response_length = 2, 5
        
        # 创建token级奖励和基线值
        token_rewards = torch.tensor([[3.0, 1.0, 4.0, 2.0, 1.0], [2.0, 1.0, 3.0, 1.0, 2.0]])
        baseline_values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
        eos_mask = torch.ones(batch_size, response_length)
        gamma = 0.95
        
        # 计算returns (sum方法)
        returns = compute_return(token_rewards, eos_mask, method='sum', gamma=gamma)
        
        # 计算优势 = returns - baseline_values
        advantages = returns - baseline_values

        print(advantages)
        expected_advantages = torch.tensor([[ 9.0893,  5.4624,  3.8025, -1.0500, -4.0000],
        [ 7.1439,  4.4672,  2.7550, -1.1000, -3.0000]])

        
        # 验证集成结果
        # self.assertEqual(advantages.shape, (batch_size, response_length))
        # self.assertTrue(torch.isfinite(advantages).all())
        # 最主要一步，测试具体值
        torch.testing.assert_close(advantages, expected_advantages, atol=1e-4, rtol=1e-4)
        # 验证优势计算的合理性
        # 当returns > baseline时，优势应该为正
        positive_advantage_mask = returns > baseline_values
        self.assertTrue((advantages[positive_advantage_mask] >= 0).all())


if __name__ == "__main__":
    unittest.main()
