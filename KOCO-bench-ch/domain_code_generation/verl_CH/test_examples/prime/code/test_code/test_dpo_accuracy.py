import unittest
import torch
import sys
import os
from unittest.mock import MagicMock

# Add the recipe directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'recipe'))

# Mock verl modules since they might not be available in test environment
class MockVerlF:
    @staticmethod
    def masked_whiten(advantages, mask):
        masked_adv = advantages[mask.bool()]
        if len(masked_adv) > 0:
            mean = masked_adv.mean()
            std = masked_adv.std()
            if std > 1e-8:
                advantages_normalized = (advantages - mean) / std
            else:
                advantages_normalized = advantages - mean
            return advantages_normalized
        return advantages

mock_verl = MagicMock()
mock_verl.utils = MagicMock()
mock_verl.utils.torch_functional = MockVerlF()

sys.modules['verl'] = mock_verl
sys.modules['verl.utils'] = mock_verl.utils
sys.modules['verl.utils.torch_functional'] = MockVerlF()

# Import ground-truth function
from prime.prime_core_algos import compute_dpo_accuracy


class TestDPOAccuracy(unittest.TestCase):
    """测试compute_dpo_accuracy函数 - 文档第3节描述的DPO成对比较准确率"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_compute_dpo_accuracy_basic(self):
        """测试DPO准确率计算的基本功能"""
        n_samples = 3
        batch_size = 6  # 必须能被n_samples整除
        seq_len = 10
        
        token_level_scores = torch.tensor([[0.1,0.3,1.0,-0.1,0.7,0.5,0.1,0.3,0.6,-0.1], [0.2,0.3,0.2,-0.4,0.9,-0.1,0.2,0.5,0.1,0.3], [-0.1,0.3,0.6,-0.1,0.2,0.7,1.0,0.2,0.5,0.1]])


        acc = torch.tensor([1.0, 0.5, 0.0, 1.0, 0.8, 0.2])
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # print(accuracy)
        expected_accuracy = torch.tensor(0.2500)
        torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-30, rtol=1e-30)
        # 验证输出
        # self.assertEqual(accuracy.dim(), 0)
        # self.assertTrue(torch.is_tensor(accuracy))
        # self.assertTrue(0 <= accuracy.item() <= 1)
        # self.assertFalse(torch.isnan(accuracy))
    
    def test_equal_accuracy_case(self):
        """测试相同准确率的边缘情况"""
        batch_size = 4
        seq_len = 5
        n_samples = 2
        
        # 测试相同准确率的情况
        token_level_scores = torch.tensor([[0.1,0.3,1.0,-0.1,0.7], [0.2,0.3,0.2,-0.4,0.9], [-0.1,0.3,0.6,-0.1,0.2], [0.1,0.8,0.6,-0.1,0.2]])
        acc = torch.ones(batch_size) * 0.5
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        self.assertTrue(torch.is_tensor(accuracy))
        # 对于相同准确率，应该返回0.5
        self.assertEqual(accuracy.item(), 0.5)
    
    def test_pairwise_comparison_logic(self):
        """测试成对比较的逻辑正确性"""
        n_samples = 2
        batch_size = 4  # 2 groups × 2 samples per group
        seq_len = 6
        
        # 创建具有明确差异的数据
        token_level_scores = torch.tensor([
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Group 1, Sample 1: high scores
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Group 1, Sample 2: low scores
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],  # Group 2, Sample 1: very high scores
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],  # Group 2, Sample 2: low scores
        ])
        acc = torch.tensor([0.8, 0.2, 0.9, 0.1])  # 准确率对应分数趋势
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # print(accuracy)
        # 由于分数趋势与准确率一致，准确率应该较高
        # self.assertTrue(0 <= accuracy.item() <= 1)
        # self.assertFalse(torch.isnan(accuracy))
        expected_accuracy = torch.tensor(1.)
        torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-30, rtol=1e-30)
    
    def test_weighted_average_computation(self):
        """测试加权平均准确率计算"""
        n_samples = 2
        batch_size = 6  # 3 groups × 2 samples per group
        seq_len = 4
        
        token_level_scores = torch.tensor([[-0.1, 0.7, 0.1, 0.5], [0.1,0.4,0.7,0.2], [0.5,-0.3,0.2,0.1], [0.2,0.4,0.6,0.8], [-0.2,0.65,-0.44,0.55], [1.0,0.8,0.6,0.4]])
        acc = torch.tensor([1.0, 0.0, 0.8, 0.2, 0.6, 0.4])  # 不同准确率差异
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # print(accuracy)
        # 验证加权平均结果在合理范围内
        expected_accuracy = torch.tensor(0.)
        torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-30, rtol=1e-30)
    

    
    def test_response_mask_effect(self):
        """测试response_mask对准确率计算的影响"""
        n_samples = 2
        batch_size = 4
        seq_len = 6
        
        token_level_scores = torch.tensor([[0.1,0.3,1.0,-0.1,0.7,0.5], [0.2,0.3,0.2,-0.4,0.9,-0.1], [-0.1,0.3,0.6,-0.1,0.2,0.7], [0.1,0.8,0.6,-0.1,0.2,0.5]])
        acc = torch.tensor([1.0, 0.0, 0.8, 0.2])
        
        # 测试全部有效的mask
        full_mask = torch.ones(batch_size, seq_len)
        accuracy_full = compute_dpo_accuracy(token_level_scores, acc, full_mask, n_samples)
        # print(accuracy_full)
        expected_accuracy_full = torch.tensor(0.5)
        torch.testing.assert_close(accuracy_full, expected_accuracy_full, atol=1e-30, rtol=1e-30)
        # 测试部分有效的mask
        partial_mask = torch.ones(batch_size, seq_len)
        partial_mask[:, -2:] = 0  # 最后两个位置masked out
        accuracy_partial = compute_dpo_accuracy(token_level_scores, acc, partial_mask, n_samples)
        expected_accuracy_partial = torch.tensor(0.5)
        torch.testing.assert_close(accuracy_partial, expected_accuracy_partial, atol=1e-30, rtol=1e-30)

    
    def test_score_aggregation_logic(self):
        """测试序列级分数聚合逻辑"""
        n_samples = 2
        batch_size = 4
        seq_len = 3
        
        # 创建可预测的分数
        token_level_scores = torch.tensor([
            [1.0, 2.0, 3.0],    # 序列分数应该是 6.0
            [0.5, 1.0, 1.5],    # 序列分数应该是 3.0
            [2.0, 2.0, 2.0],    # 序列分数应该是 6.0
            [1.0, 1.0, 1.0],    # 序列分数应该是 3.0
        ])
        acc = torch.tensor([0.9, 0.1, 0.8, 0.2])
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # 手动验证：序列分数 [6.0, 3.0, 6.0, 3.0]，准确率 [0.9, 0.1, 0.8, 0.2]
        # Group 1: 分数差异 = 6.0 - 3.0 = 3.0 > 0，准确率差异 = 0.9 - 0.1 = 0.8 > 0，方向一致
        # Group 2: 分数差异 = 6.0 - 3.0 = 3.0 > 0，准确率差异 = 0.8 - 0.2 = 0.6 > 0，方向一致
        # 由于方向完全一致，准确率应该是1.0
        self.assertAlmostEqual(accuracy.item(), 1.0, places=30)

    def test_input_output_samples(self):
        """测试具体的输入输出样例"""
        # 测试样例1: 简单的2样本情况
        token_level_scores = torch.tensor([
            [2.0, 1.0, 3.0],  # 序列分数: 6.0
            [1.0, 0.5, 0.5]   # 序列分数: 2.0
        ])
        acc = torch.tensor([0.8, 0.3])
        response_mask = torch.ones(2, 3)
        n_samples = 2
        
        result = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # 手动计算验证:
        # 分数差异 = 6.0 - 2.0 = 4.0 > 0
        # 准确率差异 = 0.8 - 0.3 = 0.5 > 0
        # 方向一致，权重 = 0.5，准确率 = 1.0
        expected_accuracy = 1.0
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)
        
        # 测试样例2: 方向不一致的情况
        token_level_scores = torch.tensor([
            [1.0, 1.0, 1.0],  # 序列分数: 3.0
            [2.0, 2.0, 2.0]   # 序列分数: 6.0
        ])
        acc = torch.tensor([0.9, 0.1])  # 高准确率对应低分数
        response_mask = torch.ones(2, 3)
        
        result = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # 手动计算验证:
        # 分数差异 = 3.0 - 6.0 = -3.0 < 0
        # 准确率差异 = 0.9 - 0.1 = 0.8 > 0
        # 方向不一致，准确率 = 0.0
        expected_accuracy = 0.0
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)
        
        # 测试样例3: 多组样本
        token_level_scores = torch.tensor([
            [3.0, 2.0],  # Group 1, Sample 1: 5.0
            [1.0, 1.0],  # Group 1, Sample 2: 2.0
            [4.0, 1.0],  # Group 2, Sample 1: 5.0
            [2.0, 0.5]   # Group 2, Sample 2: 2.5
        ])
        acc = torch.tensor([0.8, 0.2, 0.7, 0.4])
        response_mask = torch.ones(4, 2)
        
        result = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # 手动计算验证:
        # Group 1: 分数差异 = 5.0 - 2.0 = 3.0 > 0, 准确率差异 = 0.8 - 0.2 = 0.6 > 0, 一致
        # Group 2: 分数差异 = 5.0 - 2.5 = 2.5 > 0, 准确率差异 = 0.7 - 0.4 = 0.3 > 0, 一致
        # 两组都一致，总体准确率 = (1.0*0.6 + 1.0*0.3) / (0.6 + 0.3) = 0.9 / 0.9 = 1.0
        expected_accuracy = 1.0
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)

    def test_edge_case_input_output_samples(self):
        """测试边缘情况的输入输出样例"""
        n_samples = 2
        
        # 测试样例1: 零分数情况
        token_level_scores = torch.zeros(2, 3)
        acc = torch.tensor([0.6, 0.4])
        response_mask = torch.ones(2, 3)
        
        result = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # 分数差异 = 0.0 - 0.0 = 0.0，不大于0，但准确率差异 > 0
        # 方向不一致，准确率应该是 0.0
        expected_accuracy = 0.0
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)
        
        # 测试样例2: 相同准确率情况
        token_level_scores = torch.tensor([
            [1.0, 2.0],
            [3.0, 1.0]
        ])
        acc = torch.tensor([0.5, 0.5])  # 完全相同的准确率
        response_mask = torch.ones(2, 2)
        
        result = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # 准确率差异为0，应该返回0.5
        expected_accuracy = 0.5
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)
        
        # 测试样例3: 带mask的情况
        token_level_scores = torch.tensor([
            [2.0, 4.0, 1.0],  # 有效分数: 2.0 + 4.0 = 6.0
            [1.0, 2.0, 3.0]   # 有效分数: 1.0 + 2.0 = 3.0 (最后一个被mask)
        ])
        acc = torch.tensor([0.8, 0.3])
        response_mask = torch.tensor([
            [1.0, 1.0, 1.0],  # 全部有效
            [1.0, 1.0, 0.0]   # 最后一个无效
        ])
        
        result = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # 分数差异 = 6.0 - 3.0 = 3.0 > 0
        # 准确率差异 = 0.8 - 0.3 = 0.5 > 0
        # 方向一致，准确率 = 1.0
        expected_accuracy = 1.0
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)
    
    def test_extreme_cases(self):
        """测试极端情况"""
        n_samples = 2
        batch_size = 4
        seq_len = 4
        
        # 测试极端分数值
        extreme_scores = torch.tensor([
            [100.0, 100.0, 100.0, 100.0],
            [-100.0, -100.0, -100.0, -100.0],
            [50.0, 50.0, 50.0, 50.0],
            [-50.0, -50.0, -50.0, -50.0],
        ])
        acc = torch.tensor([1.0, 0.0, 0.8, 0.2])
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_accuracy(extreme_scores, acc, response_mask, n_samples)

        # print(accuracy)
        
        expected_accuracy = torch.tensor(1.)
        torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-30, rtol=1e-30)


if __name__ == "__main__":
    unittest.main()
