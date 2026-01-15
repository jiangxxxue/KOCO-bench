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
from prime.prime_core_algos import compute_dpo_abs_accuracy


class TestDPOAbsAccuracy(unittest.TestCase):
    """测试compute_dpo_abs_accuracy函数 - 文档第3节描述的DPO绝对准确率"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_compute_dpo_abs_accuracy_basic(self):
        """测试DPO绝对准确率计算的基本功能"""
        batch_size = 4
        seq_len = 10
        n_samples = 2
        
        token_level_scores = torch.tensor([[0.1,0.3,1.0,-0.1,0.7,0.5,0.1,0.3,0.6,-0.1], [0.2,0.3,0.2,-0.4,0.9,-0.1,0.2,0.5,0.1,0.3], [-0.1,0.3,0.6,-0.1,0.2,0.7,1.0,0.2,0.5,0.1], [0.1,0.8,0.6,-0.1,0.2,0.5,0.1,0.5,0.1,0.3]])
        acc = torch.tensor([0.8, 0.1, 1.0, 0.0])

        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # print(accuracy)
        expected_accuracy = torch.tensor(0.5000)
        torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-30, rtol=1e-30)
        # 验证输出
        # self.assertEqual(accuracy.dim(), 0)
        # self.assertTrue(torch.is_tensor(accuracy))
        # self.assertTrue(0 <= accuracy.item() <= 1)
        # self.assertFalse(torch.isnan(accuracy))
    
    def test_perfect_prediction(self):
        """测试完美预测的情况"""
        batch_size = 4
        seq_len = 6
        n_samples = 2
        
        # 创建明确的分数-准确率对应关系
        token_level_scores = torch.tensor([
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],    # 高分数 -> 应该预测为正例
            [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0],  # 低分数 -> 应该预测为负例
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],    # 正分数 -> 应该预测为正例
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],  # 负分数 -> 应该预测为负例
        ])
        acc = torch.tensor([1.0, 0.0, 1.0, 0.0])  # 与预测完全一致
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # print(accuracy)
        # 完美预测应该得到100%准确率
        expected_accuracy = torch.tensor(1.0000)
        torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-30, rtol=1e-30)
    
    def test_opposite_prediction(self):
        """测试相反预测的情况"""
        batch_size = 4
        seq_len = 4
        n_samples = 2
        
        # 创建与真实标签相反的分数
        token_level_scores = torch.tensor([
            [2.0, 2.0, 2.0, 2.0],    # 高分数但标签为0
            [-2.0, -2.0, -2.0, -2.0],  # 低分数但标签为1
            [1.0, 1.0, 1.0, 1.0],    # 正分数但标签为0
            [-1.0, -1.0, -1.0, -1.0],  # 负分数但标签为1
        ])
        acc = torch.tensor([0.0, 1.0, 0.0, 1.0])  # 与预测相反
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # 完全相反的预测应该得到0%准确率
        self.assertAlmostEqual(accuracy.item(), 0.0, places=5)
    
    def test_binary_classification_logic(self):
        """测试二元分类逻辑的正确性"""
        batch_size = 6
        seq_len = 3
        n_samples = 2
        
        # 测试符号函数的分类逻辑
        token_level_scores = torch.tensor([
            [1.0, 1.0, 1.0],     # 序列分数 = 3.0 > 0 -> 预测为1
            [0.5, 0.5, 0.5],     # 序列分数 = 1.5 > 0 -> 预测为1
            [-1.0, -1.0, -1.0],  # 序列分数 = -3.0 < 0 -> 预测为0
            [-0.5, -0.5, -0.5],  # 序列分数 = -1.5 < 0 -> 预测为0
            [0.0, 0.0, 0.0],     # 序列分数 = 0.0 = 0 -> 预测为0 (符号函数)
            [1.0, -1.0, 0.0],    # 序列分数 = 0.0 = 0 -> 预测为0
        ])
        acc = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # 对应的真实标签
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # print(accuracy)
        expected_accuracy = torch.tensor(0.666666666666666666666666666667)
        torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-30, rtol=1e-30)
        # 验证分类逻辑正确性，准确率应该较高
        # self.assertTrue(accuracy.item() >= 0.5, "Accuracy should be reasonable for correct classification logic")
    
    def test_different_n_samples(self):
        """测试不同n_samples参数的影响"""
        seq_len = 5
        accuracies = []
        for n_samples in [1, 2, 4]:
            batch_size = n_samples * 4  # 确保能被n_samples整除
            
            with self.subTest(n_samples=n_samples):
                token_level_scores = torch.randn(batch_size, seq_len)
                acc = torch.randint(0, 2, (batch_size,)).float()
                response_mask = torch.ones(batch_size, seq_len)
                
                accuracy = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
                accuracies.append(accuracy.item())
                # 验证输出在有效范围内
                # self.assertTrue(0 <= accuracy.item() <= 1)
                # self.assertFalse(torch.isnan(accuracy))

        # print(accuracies)
        expected_accuracies = [0.5, 0.5, 0.6875]
        torch.testing.assert_close(accuracies, expected_accuracies, atol=1e-30, rtol=1e-30)
    
    def test_response_mask_impact(self):
        """测试response_mask对绝对准确率的影响"""
        batch_size = 4
        seq_len = 8
        n_samples = 2
        
        token_level_scores = torch.ones(batch_size, seq_len)  # 所有分数都是正数
        acc = torch.tensor([1.0, 1.0, 0.0, 0.0])
        
        # 测试全部有效的mask
        full_mask = torch.ones(batch_size, seq_len)
        accuracy_full = compute_dpo_abs_accuracy(token_level_scores, acc, full_mask, n_samples)
        # print(accuracy_full)
        # 测试部分mask（影响序列分数的聚合）
        partial_mask = torch.ones(batch_size, seq_len)
        partial_mask[:, -4:] = 0  # 后一半masked out
        accuracy_partial = compute_dpo_abs_accuracy(token_level_scores, acc, partial_mask, n_samples)
        # print(accuracy_partial)
        # 两种情况都应该产生有效准确率
        # self.assertTrue(0 <= accuracy_full.item() <= 1)
        # self.assertTrue(0 <= accuracy_partial.item() <= 1)
        # self.assertFalse(torch.isnan(accuracy_full))
        # self.assertFalse(torch.isnan(accuracy_partial))
        
        # 由于都是正数，前两个样本(acc=1.0)应该预测正确，后两个(acc=0.0)预测错误
        # 对于full_mask: 准确率应该是 2/4 = 0.5
        expected_accuracy_full = torch.tensor(0.5000)
        torch.testing.assert_close(accuracy_full, expected_accuracy_full, atol=1e-30, rtol=1e-30)
        
        # 对于partial_mask: 序列分数仍然是正数（尽管更小），所以结果应该相同
        expected_accuracy_partial = torch.tensor(0.5000)
        torch.testing.assert_close(accuracy_partial, expected_accuracy_partial, atol=1e-30, rtol=1e-30)
    
    def test_zero_scores_edge_case(self):
        """测试零分数的边缘情况"""
        batch_size = 4
        seq_len = 3
        n_samples = 2
        
        # 所有分数都是零
        token_level_scores = torch.zeros(batch_size, seq_len)
        acc = torch.tensor([1.0, 0.0, 1.0, 0.0])
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # print(accuracy)
        expected_accuracy = torch.tensor(0.)
        torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-30, rtol=1e-30)
        # 零分数是边缘情况，验证函数能正确处理
    
    def test_score_aggregation_correctness(self):
        """测试分数聚合的正确性"""
        batch_size = 2
        seq_len = 4
        n_samples = 1
        
        # 明确的分数聚合测试
        token_level_scores = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],    # 聚合分数 = 10.0 > 0 -> 预测1
            [-1.0, -2.0, -3.0, -4.0]  # 聚合分数 = -10.0 < 0 -> 预测0
        ])
        acc = torch.tensor([1.0, 0.0])  # 与预测一致
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # 预测完全正确
        self.assertAlmostEqual(accuracy.item(), 1.0, places=5)
    
    def test_consistency_with_different_seeds(self):
        """测试不同随机种子下的一致性"""
        batch_size = 6
        seq_len = 4
        n_samples = 2
        
        # 固定的输入数据
        token_level_scores = torch.tensor([
            [1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [2.0, 2.0, 2.0, 2.0],
            [-2.0, -2.0, -2.0, -2.0],
            [0.5, 0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5, -0.5],
        ])
        acc = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        response_mask = torch.ones(batch_size, seq_len)
        
        # 测试多次计算的一致性
        accuracies = []
        for seed in [42, 123, 456]:
            torch.manual_seed(seed)
            accuracy = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
            accuracies.append(accuracy.item())
        
            # print(accuracies)
        expected_accuracies = [1.0, 1.0, 1.0]
        torch.testing.assert_close(accuracies, expected_accuracies, atol=1e-30, rtol=1e-30)
        # 由于输入是确定性的，结果应该完全一致
        # self.assertTrue(all(abs(acc - accuracies[0]) < 1e-6 for acc in accuracies))

    def test_input_output_samples(self):
        """测试具体的输入输出样例"""
        # 测试样例1: 简单的二元分类
        token_level_scores = torch.tensor([
            [2.0, 3.0],   # 序列分数: 5.0 > 0 -> 预测标签: 1
            [-1.0, -2.0]  # 序列分数: -3.0 < 0 -> 预测标签: 0
        ])
        acc = torch.tensor([1.0, 0.0])  # 真实标签与预测一致
        response_mask = torch.ones(2, 2)
        n_samples = 1
        
        result = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        # print(result)
        # 手动计算验证:
        # 序列分数 = [5.0, -3.0]
        # 预测标签 = sign([5.0, -3.0]) = [1, -1]  
        # 真实标签转换 = acc * 2 - 1 = [1.0, 0.0] * 2 - 1 = [1.0, -1.0]
        # 匹配情况 = [1==1, -1==-1] = [True, True]
        # 准确率 = 2/2 = 1.0
        expected_accuracy = 1.0
        self.assertAlmostEqual(result.item(), expected_accuracy, places=5)
        
        # 测试样例2: 完全错误的预测
        token_level_scores = torch.tensor([
            [1.0, 1.0, 1.0],   # 序列分数: 3.0 > 0 -> 预测标签: 1
            [-2.0, -1.0, 0.0]  # 序列分数: -3.0 < 0 -> 预测标签: 0
        ])
        acc = torch.tensor([0.0, 1.0])  # 真实标签与预测相反
        response_mask = torch.tensor([
            [1.0, 1.0, 1.0],  
            [1.0, 1.0, 0.0]   # 最后一个token被mask
        ])
        
        result = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # 手动计算验证:
        # 有效序列分数 = [3.0, -3.0]
        # 预测标签 = sign([3.0, -3.0]) = [1, -1]
        # 真实标签转换 = [0.0, 1.0] * 2 - 1 = [-1.0, 1.0]
        # 匹配情况 = [1==-1, -1==1] = [False, False]
        # 准确率 = 0/2 = 0.0
        expected_accuracy = 0.0
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)
        
        # 测试样例3: 混合情况
        token_level_scores = torch.tensor([
            [1.5, 2.5],    # 序列分数: 4.0 > 0 -> 预测: 1
            [-0.5, -1.5],  # 序列分数: -2.0 < 0 -> 预测: 0  
            [2.0, -1.0],   # 序列分数: 1.0 > 0 -> 预测: 1
            [-3.0, 1.0]    # 序列分数: -2.0 < 0 -> 预测: 0
        ])
        acc = torch.tensor([1.0, 0.0, 0.0, 1.0])  # 前两个正确，后两个错误
        response_mask = torch.ones(4, 2)
        
        result = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # 手动计算验证:
        # 序列分数 = [4.0, -2.0, 1.0, -2.0]
        # 预测标签 = sign([4.0, -2.0, 1.0, -2.0]) = [1, -1, 1, -1]
        # 真实标签转换 = [1.0, 0.0, 0.0, 1.0] * 2 - 1 = [1.0, -1.0, -1.0, 1.0]
        # 匹配情况 = [1==1, -1==-1, 1==-1, -1==1] = [True, True, False, False]
        # 准确率 = 2/4 = 0.5
        expected_accuracy = 0.5
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)

    def test_zero_boundary_cases(self):
        """测试零值边界情况的具体输入输出"""
        # 测试样例1: 零分数情况
        token_level_scores = torch.tensor([
            [0.0, 0.0, 0.0],  # 序列分数: 0.0，sign(0.0) = 0
            [1.0, -1.0, 0.0],      # 序列分数: 0.0，sign(0.0) = 0
            [0.5, -0.5, 0.0]  # 序列分数: 0.0，sign(0.0) = 0
        ])
        acc = torch.tensor([0.0, 1.0, 0.0])
        response_mask = torch.ones(3, 3)
        response_mask[1, 2] = 0  # 第二个样本只有两个有效token
        response_mask[2, 2] = 1  # 第三个样本三个token都有效
        n_samples = 1
        
        result = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # 手动计算验证:
        # 有效序列分数 = [0.0, 0.0, 0.0]
        # 预测标签 = sign([0.0, 0.0, 0.0]) = [0, 0, 0] (在PyTorch中，sign(0)=0)
        # 真实标签转换 = [0.0, 1.0, 0.0] * 2 - 1 = [-1.0, 1.0, -1.0]
        # 匹配情况 = [0==-1, 0==1, 0==-1] = [False, False, False]
        # 准确率 = 0/3 = 0.0
        expected_accuracy = 0.0
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)
        
        # 测试样例2: 微小正负值
        token_level_scores = torch.tensor([
            [0.001],      # 序列分数: 0.001 > 0 -> 预测: 1
            [-0.001],     # 序列分数: -0.001 < 0 -> 预测: 0
            [0.0001],     # 序列分数: 0.0001 > 0 -> 预测: 1
            [-0.0001]     # 序列分数: -0.0001 < 0 -> 预测: 0
        ])
        acc = torch.tensor([1.0, 0.0, 1.0, 0.0])  # 与预测一致
        response_mask = torch.ones(4, 1)
        
        result = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # 手动计算验证:
        # 序列分数 = [0.001, -0.001, 0.0001, -0.0001]
        # 预测标签 = sign([0.001, -0.001, 0.0001, -0.0001]) = [1, -1, 1, -1]
        # 真实标签转换 = [1.0, 0.0, 1.0, 0.0] * 2 - 1 = [1.0, -1.0, 1.0, -1.0]
        # 匹配情况 = [1==1, -1==-1, 1==1, -1==-1] = [True, True, True, True]
        # 准确率 = 4/4 = 1.0
        expected_accuracy = 1.0
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)

    def test_mask_effect_input_output(self):
        """测试mask效果的具体输入输出样例"""
        # 测试样例: mask对序列分数聚合的影响
        token_level_scores = torch.tensor([
            [3.0, 2.0, 1.0, -10.0],  # 如果全部有效: 序列分数 = -4.0 < 0 -> 预测: 0
                                      # 如果最后一个被mask: 序列分数 = 6.0 > 0 -> 预测: 1
            [1.0, 1.0, 1.0, -5.0]    # 如果全部有效: 序列分数 = -2.0 < 0 -> 预测: 0
                                      # 如果最后一个被mask: 序列分数 = 3.0 > 0 -> 预测: 1
        ])
        acc = torch.tensor([1.0, 1.0])  # 真实标签都是1
        
        # 测试全部有效的mask
        full_mask = torch.ones(2, 4)
        result_full = compute_dpo_abs_accuracy(token_level_scores, acc, full_mask, 1)
        
        # 手动计算验证(全mask):
        # 序列分数 = [-4.0, -2.0]
        # 预测标签 = sign([-4.0, -2.0]) = [-1, -1]
        # 真实标签转换 = [1.0, 1.0] * 2 - 1 = [1.0, 1.0]
        # 匹配情况 = [-1==1, -1==1] = [False, False]
        # 准确率 = 0/2 = 0.0
        expected_full = 0.0
        self.assertAlmostEqual(result_full.item(), expected_full, places=30)
        
        # 测试部分mask（最后一个token被mask）
        partial_mask = torch.ones(2, 4)
        partial_mask[:, -1] = 0  # 最后一个token被mask
        result_partial = compute_dpo_abs_accuracy(token_level_scores, acc, partial_mask, 1)
        
        # 手动计算验证(部分mask):
        # 有效序列分数 = [6.0, 3.0]
        # 预测标签 = sign([6.0, 3.0]) = [1, 1]
        # 真实标签转换 = [1.0, 1.0] * 2 - 1 = [1.0, 1.0]
        # 匹配情况 = [1==1, 1==1] = [True, True]
        # 准确率 = 2/2 = 1.0
        expected_partial = 1.0
        self.assertAlmostEqual(result_partial.item(), expected_partial, places=30)

    def test_batch_computation_input_output(self):
        """测试批次计算的具体输入输出样例"""
        # 测试样例: 大批次数据
        token_level_scores = torch.tensor([
            [1.0, 1.0],    # 样本1: 2.0 > 0 -> 预测: 1
            [-1.0, -1.0],  # 样本2: -2.0 < 0 -> 预测: 0
            [2.0, 0.0],    # 样本3: 2.0 > 0 -> 预测: 1
            [0.0, -3.0],   # 样本4: -3.0 < 0 -> 预测: 0
            [0.5, 0.5],    # 样本5: 1.0 > 0 -> 预测: 1
            [-0.8, -0.2]   # 样本6: -1.0 < 0 -> 预测: 0
        ])
        acc = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        # 预期匹配: [True, True, False, False, True, True] = 4/6
        response_mask = torch.ones(6, 2)
        n_samples = 1
        
        result = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # 手动计算验证:
        # 序列分数 = [2.0, -2.0, 2.0, -3.0, 1.0, -1.0]
        # 预测标签 = sign([2.0, -2.0, 2.0, -3.0, 1.0, -1.0]) = [1, -1, 1, -1, 1, -1]
        # 真实标签转换 = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0] * 2 - 1 = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0]
        # 匹配情况 = [1==1, -1==-1, 1==-1, -1==1, 1==1, -1==-1] = [T, T, F, F, T, T]
        # 准确率 = 4/6 = 0.6667
        expected_accuracy = 0.6666666865348816
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)
        
        # 测试样例: n_samples > 1的情况
        result_grouped = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples=2)
        
        # 当n_samples=2时，函数内部仍然对所有样本计算准确率，不应该影响结果
        # (因为该函数没有使用n_samples参数进行分组计算)
        self.assertAlmostEqual(result_grouped.item(), expected_accuracy, places=30)


if __name__ == "__main__":
    unittest.main()
 