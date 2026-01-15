import unittest
import torch
import numpy as np
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the actual function we want to test
from pacs.pacs_core_algos import compute_weight


class TestComputeWeight(unittest.TestCase):
    """测试compute_weight函数 - PACS文档第2节描述的样本权重计算"""
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_compute_weight_question_mode_balanced(self):
        """测试question模式：有正负样本多样性的情况"""
        # 创建一组有多样性的标签（既有0又有1）
        rollout_n = 4
        score = torch.tensor([
            [1.0],  # 正样本
            [0.0],  # 负样本
            [1.0],  # 正样本
            [0.0]   # 负样本
        ])
        
        # 调用真实的compute_weight函数
        weight = compute_weight(score, rollout_n, mode="question")

        # print(weight)

        expected_weight = torch.tensor([[1.],
        [1.],
        [1.],
        [1.]])
        torch.testing.assert_close(weight, expected_weight, atol=1e-5, rtol=1e-5)
        
        # 验证输出形状和属性
        self.assertEqual(weight.shape, (4, 1))
        self.assertTrue(torch.is_tensor(weight))
        self.assertTrue(torch.isfinite(weight).all())
        self.assertTrue((weight >= 0).all())
        
        # 验证平衡权重：正负样本数量相同时，权重应该相同
        positive_weights = weight[score.squeeze(-1) == 1]
        negative_weights = weight[score.squeeze(-1) == 0]
        
        # 所有正样本权重应该相同
        self.assertTrue(torch.allclose(positive_weights, positive_weights[0]))
        # 所有负样本权重应该相同
        self.assertTrue(torch.allclose(negative_weights, negative_weights[0]))
    
    def test_compute_weight_question_mode_imbalanced(self):
        """测试question模式：正负样本不平衡的情况"""
        rollout_n = 5
        # 3个正样本，2个负样本
        score = torch.tensor([
            [1.0],
            [1.0],
            [1.0],
            [0.0],
            [0.0]
        ])
        
        weight = compute_weight(score, rollout_n, mode="question")

        # print(weight)

        expected_weight = torch.tensor([[0.8333],
        [0.8333],
        [0.8333],
        [1.2500],
        [1.2500]])
        torch.testing.assert_close(weight, expected_weight, atol=1e-4, rtol=1e-4)
        # 验证输出
        self.assertEqual(weight.shape, (5, 1))
        self.assertTrue(torch.isfinite(weight).all())
        
        # 负样本（少数类）应该获得更高的权重
        positive_weights = weight[score.squeeze(-1) == 1]
        negative_weights = weight[score.squeeze(-1) == 0]
        
        avg_positive_weight = positive_weights.mean()
        avg_negative_weight = negative_weights.mean()
        
        # 负样本权重应该大于正样本权重
        self.assertGreater(avg_negative_weight.item(), avg_positive_weight.item())
    
    def test_compute_weight_question_mode_no_diversity(self):
        """测试question模式：没有多样性的情况（全是同一类）"""
        rollout_n = 4
        
        # 测试全是正样本
        score_all_positive = torch.ones(4, 1)
        weight_all_positive = compute_weight(score_all_positive, rollout_n, mode="question")
        
        # 没有多样性时，权重应该全为1
        expected_weight = torch.ones(4, 1)
        torch.testing.assert_close(weight_all_positive, expected_weight)
        
        # 测试全是负样本
        score_all_negative = torch.zeros(4, 1)
        weight_all_negative = compute_weight(score_all_negative, rollout_n, mode="question")
        
        # 没有多样性时，权重应该全为1
        torch.testing.assert_close(weight_all_negative, expected_weight)
    
    def test_compute_weight_only_positive_mode(self):
        """测试only_positive模式：只保留正样本"""
        rollout_n = 4
        score = torch.tensor([
            [1.0],
            [0.0],
            [1.0],
            [0.0]
        ])
        
        weight = compute_weight(score, rollout_n, mode="only_positive")
        
        # 验证输出形状
        self.assertEqual(weight.shape, (4, 1))
        
        # 验证正样本权重为1.0
        self.assertEqual(weight[0].item(), 1.0)
        self.assertEqual(weight[2].item(), 1.0)
        
        # 验证负样本权重为0.0
        self.assertEqual(weight[1].item(), 0.0)
        self.assertEqual(weight[3].item(), 0.0)
    
    def test_compute_weight_only_negative_mode(self):
        """测试only_negative模式：只保留负样本"""
        rollout_n = 4
        score = torch.tensor([
            [1.0],
            [0.0],
            [1.0],
            [0.0]
        ])
        
        weight = compute_weight(score, rollout_n, mode="only_negative")
        
        # 验证输出形状
        self.assertEqual(weight.shape, (4, 1))
        
        # 验证负样本权重为1.0
        self.assertEqual(weight[1].item(), 1.0)
        self.assertEqual(weight[3].item(), 1.0)
        
        # 验证正样本权重为0.0
        self.assertEqual(weight[0].item(), 0.0)
        self.assertEqual(weight[2].item(), 0.0)
    
    def test_compute_weight_multiple_groups(self):
        """测试多组样本的权重计算"""
        rollout_n = 3
        # 两组样本，每组3个
        score = torch.tensor([
            [1.0], [0.0], [1.0],  # 第一组：2正1负
            [0.0], [0.0], [1.0]   # 第二组：1正2负
        ])
        
        weight = compute_weight(score, rollout_n, mode="question")
        
        # print(weight)

        expected_weight = torch.tensor([[0.7500],
        [1.5000],
        [0.7500],
        [0.7500],
        [0.7500],
        [1.5000]])
        torch.testing.assert_close(weight, expected_weight, atol=1e-4, rtol=1e-4)
        # 验证输出形状
        self.assertEqual(weight.shape, (6, 1))
        self.assertTrue(torch.isfinite(weight).all())
        
        # 第一组：负样本（少数）应该有更高权重
        group1_weights = weight[:3]
        group1_negative_weight = group1_weights[1]  # index 1是负样本
        group1_positive_weights = torch.stack([group1_weights[0], group1_weights[2]])
        self.assertGreater(group1_negative_weight.item(), group1_positive_weights.mean().item())
        
        # 第二组：正样本（少数）应该有更高权重
        group2_weights = weight[3:]
        group2_positive_weight = group2_weights[2]  # index 2是正样本
        group2_negative_weights = torch.stack([group2_weights[0], group2_weights[1]])
        self.assertGreater(group2_positive_weight.item(), group2_negative_weights.mean().item())
    
    def test_compute_weight_edge_cases(self):
        """测试边缘情况"""
        # 测试样例1: 单个样本
        single_score = torch.tensor([[1.0]])
        single_weight = compute_weight(single_score, rollout_n=1, mode="question")
        expected_single = torch.tensor([[1.0]])
        torch.testing.assert_close(single_weight, expected_single)
        
        # 测试样例2: 大批量
        large_batch_size = 100
        large_rollout_n = 10
        large_score = torch.randint(0, 2, (large_batch_size, 1)).float()
        large_weight = compute_weight(large_score, large_rollout_n, mode="question")

        # print(large_weight)

        expected_weight = torch.tensor([[0.7143],
        [1.6667],
        [0.7143],
        [0.7143],
        [0.7143],
        [1.6667],
        [0.7143],
        [0.7143],
        [0.7143],
        [1.6667],
        [0.8333],
        [0.8333],
        [0.8333],
        [0.8333],
        [1.2500],
        [0.8333],
        [1.2500],
        [1.2500],
        [1.2500],
        [0.8333],
        [0.5556],
        [5.0000],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.8333],
        [0.8333],
        [1.2500],
        [1.2500],
        [1.2500],
        [0.8333],
        [1.2500],
        [0.8333],
        [0.8333],
        [0.8333],
        [1.6667],
        [1.6667],
        [0.7143],
        [0.7143],
        [0.7143],
        [0.7143],
        [0.7143],
        [1.6667],
        [0.7143],
        [0.7143],
        [0.8333],
        [1.2500],
        [0.8333],
        [1.2500],
        [0.8333],
        [1.2500],
        [1.2500],
        [0.8333],
        [0.8333],
        [0.8333],
        [0.8333],
        [0.8333],
        [0.8333],
        [0.8333],
        [0.8333],
        [1.2500],
        [1.2500],
        [0.8333],
        [1.2500],
        [1.2500],
        [0.7143],
        [0.7143],
        [1.6667],
        [0.7143],
        [1.6667],
        [0.7143],
        [0.7143],
        [0.7143],
        [1.6667],
        [0.7143],
        [1.0000],
        [1.0000],
        [1.0000],
        [1.0000],
        [1.0000],
        [1.0000],
        [1.0000],
        [1.0000],
        [1.0000],
        [1.0000],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [5.0000]])
        torch.testing.assert_close(large_weight, expected_weight, atol=1e-4, rtol=1e-4)
    
    def test_compute_weight_invalid_mode(self):
        """测试无效的mode参数"""
        score = torch.tensor([[1.0], [0.0]])
        
        # 应该抛出ValueError
        with self.assertRaises(ValueError) as context:
            compute_weight(score, rollout_n=2, mode="invalid_mode")
        
        self.assertIn("Invalid weight mode", str(context.exception))
    
    def test_compute_weight_batch_consistency(self):
        """测试批处理的一致性"""
        rollout_n = 4
        score = torch.tensor([
            [1.0], [0.0], [1.0], [0.0],  # 第一组
            [1.0], [1.0], [0.0], [0.0]   # 第二组
        ])
        
        # 批量计算
        weight_batch = compute_weight(score, rollout_n, mode="question")
        
        # 分组计算
        weight_group1 = compute_weight(score[:4], rollout_n, mode="question")
        weight_group2 = compute_weight(score[4:], rollout_n, mode="question")
        
        # 验证批量计算与分组计算结果一致
        torch.testing.assert_close(weight_batch[:4], weight_group1)
        torch.testing.assert_close(weight_batch[4:], weight_group2)


class TestComputeWeightIntegration(unittest.TestCase):
    """集成测试：测试compute_weight在实际场景中的应用"""
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_weight_mode_comparison(self):
        """比较三种权重模式的差异"""
        rollout_n = 4
        score = torch.tensor([
            [1.0],
            [0.0],
            [1.0],
            [0.0]
        ])
        
        # 计算三种模式的权重
        weight_question = compute_weight(score, rollout_n, mode="question")
        weight_only_positive = compute_weight(score, rollout_n, mode="only_positive")
        weight_only_negative = compute_weight(score, rollout_n, mode="only_negative")

        # print(weight_question)
        # print(weight_only_positive)
        # print(weight_only_negative)

        expected_weight_question = torch.tensor([[1.],
        [1.],
        [1.],
        [1.]])
        expected_weight_only_positive = torch.tensor([[1.],
        [0.],
        [1.],
        [0.]])
        expected_weight_only_negative = torch.tensor([[0.],
        [1.],
        [0.],
        [1.]])
        torch.testing.assert_close(weight_question, expected_weight_question, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(weight_only_positive, expected_weight_only_positive, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(weight_only_negative, expected_weight_only_negative, atol=1e-4, rtol=1e-4)
        
        # 验证三种模式产生不同的结果
        self.assertFalse(torch.equal(weight_question, weight_only_positive))
        self.assertFalse(torch.equal(weight_question, weight_only_negative))
        self.assertFalse(torch.equal(weight_only_positive, weight_only_negative))
        
        # 验证only_positive和only_negative是互补的
        combined_weight = weight_only_positive + weight_only_negative
        expected_combined = torch.ones(4, 1)
        torch.testing.assert_close(combined_weight, expected_combined)
    
    def test_weight_with_loss_computation(self):
        """测试权重在损失计算中的应用"""
        rollout_n = 4
        score = torch.tensor([
            [1.0],
            [0.0],
            [1.0],
            [0.0]
        ])
        
        # 计算权重
        weight = compute_weight(score, rollout_n, mode="question")
        
        # print(weight)

        expected_weight = torch.tensor([[1.],
        [1.],
        [1.],
        [1.]])
        torch.testing.assert_close(weight, expected_weight, atol=1e-4, rtol=1e-4)
        # 模拟损失值
        loss_per_sample = torch.tensor([
            [0.5],
            [0.8],
            [0.3],
            [0.9]
        ])
        
        # 应用权重
        weighted_loss = loss_per_sample * weight
        
        # 验证加权损失
        self.assertEqual(weighted_loss.shape, (4, 1))
        self.assertTrue(torch.isfinite(weighted_loss).all())
        self.assertTrue((weighted_loss >= 0).all())
        
        # 验证权重对损失的影响
        # 少数类样本的加权损失应该更大
        total_weighted_loss = weighted_loss.sum()
        self.assertGreater(total_weighted_loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
