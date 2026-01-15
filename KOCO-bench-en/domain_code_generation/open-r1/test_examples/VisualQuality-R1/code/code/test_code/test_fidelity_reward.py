import unittest
import torch
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))

# Import the actual function we want to test
from open_r1.grpo_jsonl import fidelity_reward


class TestFidelityReward(unittest.TestCase):
    """测试fidelity_reward函数 - 计算单个样本对之间的排序保真度奖励"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
    
    def test_fidelity_reward_basic(self):
        """测试基本的保真度奖励计算"""
        # 测试样例1: pred1 > pred2, gt = 1.0 (预测正确)
        pred1 = 4.0
        pred2 = 2.0
        var1 = 0.5
        var2 = 0.3
        gt = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        
        reward = fidelity_reward(pred1, pred2, var1, var2, gt, self.device)
        
        # 验证输出
        self.assertTrue(torch.is_tensor(reward))
        self.assertTrue(torch.isfinite(reward))
        # 当预测正确时，奖励应该接近2.0
        self.assertGreater(reward.item(), 1.5)
    
    def test_fidelity_reward_correct_ranking(self):
        """测试预测排序正确的情况"""
        # pred1 > pred2, gt = 1.0
        pred1 = 5.0
        pred2 = 2.0
        var1 = 0.1
        var2 = 0.1
        gt = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        
        reward = fidelity_reward(pred1, pred2, var1, var2, gt, self.device)
        
        # 预测正确，奖励应该高
        self.assertGreater(reward.item(), 1.8)
        self.assertLessEqual(reward.item(), 2.0)
    
    def test_fidelity_reward_incorrect_ranking(self):
        """测试预测排序错误的情况"""
        # pred1 < pred2, 但 gt = 1.0 (预测错误)
        pred1 = 2.0
        pred2 = 5.0
        var1 = 0.1
        var2 = 0.1
        gt = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        
        reward = fidelity_reward(pred1, pred2, var1, var2, gt, self.device)
        
        # 预测错误，奖励应该低
        self.assertLess(reward.item(), 1.0)
    
    def test_fidelity_reward_equal_quality(self):
        """测试质量相等的情况 (gt = 0.5)"""
        pred1 = 3.0
        pred2 = 3.0
        var1 = 0.2
        var2 = 0.2
        gt = torch.tensor(0.5, dtype=torch.float32, device=self.device)
        
        reward = fidelity_reward(pred1, pred2, var1, var2, gt, self.device)
        
        # 当预测相等且真实也相等时，奖励应该接近最大值
        self.assertGreater(reward.item(), 1.4)
        self.assertLessEqual(reward.item(), 2.0)
    
    def test_fidelity_reward_with_variance(self):
        """测试方差对奖励的影响"""
        pred1 = 4.0
        pred2 = 2.0
        gt = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        
        # 小方差
        reward_small_var = fidelity_reward(pred1, pred2, 0.1, 0.1, gt, self.device)
        
        # 大方差
        reward_large_var = fidelity_reward(pred1, pred2, 2.0, 2.0, gt, self.device)
        
        # 方差越小，对差异的置信度越高，奖励差异应该更明显
        self.assertTrue(torch.isfinite(reward_small_var))
        self.assertTrue(torch.isfinite(reward_large_var))
    
    def test_fidelity_reward_tensor_inputs(self):
        """测试张量输入"""
        pred1 = torch.tensor(4.0, dtype=torch.float32, device=self.device)
        pred2 = torch.tensor(2.0, dtype=torch.float32, device=self.device)
        var1 = torch.tensor(0.5, dtype=torch.float32, device=self.device)
        var2 = torch.tensor(0.3, dtype=torch.float32, device=self.device)
        gt = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        
        reward = fidelity_reward(pred1, pred2, var1, var2, gt, self.device)
        
        self.assertTrue(torch.is_tensor(reward))
        self.assertTrue(torch.isfinite(reward))
        self.assertGreater(reward.item(), 0.0)
    
    def test_fidelity_reward_edge_cases(self):
        """测试边缘情况"""
        # 测试样例1: 极小的方差
        pred1 = 3.0
        pred2 = 2.0
        var1 = 1e-8
        var2 = 1e-8
        gt = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        
        reward = fidelity_reward(pred1, pred2, var1, var2, gt, self.device)
        self.assertTrue(torch.isfinite(reward))
        
        # 测试样例2: 零方差（依赖epsilon保护）
        reward_zero_var = fidelity_reward(pred1, pred2, 0.0, 0.0, gt, self.device)
        self.assertTrue(torch.isfinite(reward_zero_var))
    
    def test_fidelity_reward_different_gt_values(self):
        """测试不同的真实排序关系"""
        pred1 = 4.0
        pred2 = 2.0
        var1 = 0.5
        var2 = 0.5
        
        # gt = 1.0: pred1 应该优于 pred2
        gt_1 = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        reward_1 = fidelity_reward(pred1, pred2, var1, var2, gt_1, self.device)
        
        # gt = 0.0: pred2 应该优于 pred1
        gt_0 = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        reward_0 = fidelity_reward(pred1, pred2, var1, var2, gt_0, self.device)
        
        # gt = 0.5: 两者相等
        gt_05 = torch.tensor(0.5, dtype=torch.float32, device=self.device)
        reward_05 = fidelity_reward(pred1, pred2, var1, var2, gt_05, self.device)
        
        # 当pred1 > pred2时，gt=1.0应该给出最高奖励
        self.assertGreater(reward_1.item(), reward_0.item())
        self.assertTrue(torch.isfinite(reward_05))
    
    def test_fidelity_reward_symmetry(self):
        """测试对称性：交换pred1和pred2，gt也相应变化，奖励应该相同"""
        pred1 = 4.0
        pred2 = 2.0
        var1 = 0.5
        var2 = 0.3
        
        # 原始情况
        gt_1 = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        reward_1 = fidelity_reward(pred1, pred2, var1, var2, gt_1, self.device)
        
        # 交换后
        gt_0 = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        reward_2 = fidelity_reward(pred2, pred1, var2, var1, gt_0, self.device)
        
        # 由于对称性，奖励应该相同
        torch.testing.assert_close(reward_1, reward_2, atol=1e-5, rtol=1e-5)


class TestFidelityRewardIntegration(unittest.TestCase):
    """集成测试：测试fidelity_reward在实际场景中的应用"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
    
    def test_batch_fidelity_computation(self):
        """测试批量计算保真度奖励"""
        # 模拟多个样本对的比较
        pred_pairs = [
            (4.0, 2.0, 1.0),  # pred1 > pred2, gt = 1.0
            (2.0, 4.0, 0.0),  # pred1 < pred2, gt = 0.0
            (3.0, 3.0, 0.5),  # pred1 = pred2, gt = 0.5
        ]
        
        rewards = []
        for pred1, pred2, gt_val in pred_pairs:
            var1 = 0.5
            var2 = 0.5
            gt = torch.tensor(gt_val, dtype=torch.float32, device=self.device)
            reward = fidelity_reward(pred1, pred2, var1, var2, gt, self.device)
            rewards.append(reward.item())
        
        # 验证所有奖励都是有效的
        self.assertEqual(len(rewards), 3)
        for reward in rewards:
            self.assertTrue(math.isfinite(reward))
            self.assertGreater(reward, 0.0)
            self.assertLessEqual(reward, 2.0)


if __name__ == "__main__":
    import math
    unittest.main()
