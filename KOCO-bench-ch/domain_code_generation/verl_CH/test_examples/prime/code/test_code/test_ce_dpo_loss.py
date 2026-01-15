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
from prime.prime_core_algos import compute_ce_dpo_loss_rm


class TestCEDPOLoss(unittest.TestCase):
    """测试compute_ce_dpo_loss_rm函数 - 文档第2节描述的交叉熵DPO损失计算"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_compute_ce_dpo_loss_rm_basic(self):
        """测试交叉熵DPO损失计算的基本功能"""
        batch_size = 4
        seq_len = 10
        
        token_level_scores = torch.tensor([[0.1,0.3,0.6,-0.1,0.2,0.5,0.1,0.3,0.6,-0.1], [0.1,0.3,0.6,-0.4,0.9,-0.1,0.2,0.5,0.1,0.3], [0.1,0.3,0.6,-0.1,0.2,0.7,1.0,0.2,0.5,0.1], [0.1,0.3,0.6,-0.1,0.2,0.5,0.1,0.5,0.1,0.3]])
        acc = torch.tensor([0.8, 0.3, 1.0, 0.0])
        response_mask = torch.ones(batch_size, seq_len)
        beta = 0.1
        
        loss = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta)
        
        # print(f"loss: {loss.item():.30f}")
        expected_loss = torch.tensor(0.684428155422210693359375000000)
        # 验证输出是标量
        # self.assertEqual(loss.dim(), 0)
        # self.assertTrue(torch.is_tensor(loss))
        # self.assertFalse(torch.isnan(loss))
        # self.assertFalse(torch.isinf(loss))
        torch.testing.assert_close(loss, expected_loss, atol=1e-30, rtol=1e-30)


    def test_ce_dpo_loss_mathematical_properties(self):
        """测试CE DPO损失的数学属性和计算正确性"""
        # 简单的确定性测试用例
        token_level_scores = torch.tensor([[1.0, 2.0], [-1.0, -2.0]])
        acc = torch.tensor([1.0, 0.0])
        response_mask = torch.ones(2, 2)
        beta = 1.0
        
        loss = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta)
        
        # print(loss)
        # print(f"loss: {loss.item():.30f}")
        expected_loss = torch.tensor(0.048587348312139511108398437500)
        # 手动计算期望的损失以验证实现
        # seq_scores = (token_level_scores * response_mask).sum(dim=1) * beta  # [3.0, -3.0]
        # probs = torch.sigmoid(seq_scores)  # sigmoid([3.0, -3.0])
        # expected_loss = torch.nn.functional.binary_cross_entropy(probs, acc)
        
        # self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)
        torch.testing.assert_close(loss, expected_loss, atol=1e-30, rtol=1e-30)
    
    def test_different_beta_values(self):
        """测试不同beta参数值的影响"""
        batch_size = 2
        seq_len = 4
        
        token_level_scores = torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
        acc = torch.tensor([1.0, 0.0])
        response_mask = torch.ones(batch_size, seq_len)
        
        # 测试不同的beta值
        betas = [0.01, 0.1, 1.0, 10.0]
        losses = []
        
        for beta in betas:
            loss = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta)
            losses.append(loss.item())
            
            
            # 验证损失都是有效的
            self.assertFalse(torch.isnan(loss))
            self.assertFalse(torch.isinf(loss))
            self.assertGreaterEqual(loss.item(), 0.0)  # BCE损失应该非负
        # print(f"losses: {losses[3]:.30f}")
        expected_losses = [0.703647077083587646484375000000, 0.842057943344116210937500000000, 4.009264469146728515625000000000, 50.0]
        torch.testing.assert_close(losses, expected_losses, atol=1e-30, rtol=1e-30)
    
    def test_response_mask_effect(self):
        """测试response_mask对损失计算的影响"""
        batch_size = 2
        seq_len = 4
        
        token_level_scores = torch.ones(batch_size, seq_len)
        acc = torch.tensor([1.0, 0.0])
        beta = 0.1
        
        # 测试全部有效的mask
        full_mask = torch.ones(batch_size, seq_len)
        loss_full = compute_ce_dpo_loss_rm(token_level_scores, acc, full_mask, beta)
        
        # print(f"loss_full: {loss_full.item():.30f}")
        expected_loss_full = torch.tensor(0.713015258312225341796875000000)
        # 测试部分有效的mask
        partial_mask = torch.ones(batch_size, seq_len)
        partial_mask[:, -1] = 0  # 最后一个位置masked out
        loss_partial = compute_ce_dpo_loss_rm(token_level_scores, acc, partial_mask, beta)
        
        # print(f"loss_partial: {loss_partial.item():.30f}")
        expected_loss_partial = torch.tensor(0.704355239868164062500000000000)

        torch.testing.assert_close(loss_full, expected_loss_full, atol=1e-30, rtol=1e-30)
        torch.testing.assert_close(loss_partial, expected_loss_partial, atol=1e-30, rtol=1e-30)
        # 验证两种情况都产生有效损失
        # self.assertFalse(torch.isnan(loss_full))
        # self.assertFalse(torch.isnan(loss_partial))
        
        # # 由于mask影响了序列分数的计算，损失应该不同
        # self.assertNotEqual(loss_full.item(), loss_partial.item())
    
    def test_edge_cases(self):
        """测试边缘情况"""
        batch_size = 2
        seq_len = 3
        
        # 测试极端的token_level_scores
        extreme_scores = torch.tensor([[100.0, 100.0, 100.0], [-100.0, -100.0, -100.0]])
        acc = torch.tensor([1.0, 0.0])
        response_mask = torch.ones(batch_size, seq_len)
        beta = 0.1
        
        loss = compute_ce_dpo_loss_rm(extreme_scores, acc, response_mask, beta)
        
        # print(f"loss: {loss.item():.30f}")
        expected_loss = torch.tensor(0.000000000000046788117949230976)
        # 即使是极端值，损失也应该是有效的
        # self.assertFalse(torch.isnan(loss))
        # self.assertFalse(torch.isinf(loss))
        # self.assertGreaterEqual(loss.item(), 0.0)
        torch.testing.assert_close(loss, expected_loss, atol=1e-30, rtol=1e-30)
        
        # 测试零分数情况
        zero_scores = torch.zeros(batch_size, seq_len)
        loss_zero = compute_ce_dpo_loss_rm(zero_scores, acc, response_mask, beta)
        
        # print(loss_zero)
        # print(f"loss_zero: {loss_zero.item():.30f}")
        expected_loss_zero = torch.tensor(0.693147182464599609375000000000)
        # self.assertFalse(torch.isnan(loss_zero))
        # self.assertFalse(torch.isinf(loss_zero))
        torch.testing.assert_close(loss_zero, expected_loss_zero, atol=1e-30, rtol=1e-30)
    
    def test_batch_consistency(self):
        """测试批次处理的一致性"""
        seq_len = 5
        beta = 0.1
        
        # 创建单个样本
        single_scores = torch.randn(1, seq_len)
        single_acc = torch.rand(1)
        single_mask = torch.ones(1, seq_len)
        
        single_loss = compute_ce_dpo_loss_rm(single_scores, single_acc, single_mask, beta)
        
        # 创建包含相同样本的批次
        batch_scores = single_scores.repeat(3, 1)
        batch_acc = single_acc.repeat(3)
        batch_mask = single_mask.repeat(3, 1)
        
        batch_loss = compute_ce_dpo_loss_rm(batch_scores, batch_acc, batch_mask, beta)
        
        # 由于所有样本相同，批次损失应该等于单个样本损失
        self.assertAlmostEqual(single_loss.item(), batch_loss.item(), places=30)

    def test_input_output_samples(self):
        """测试具体的输入输出样例"""
        # 测试样例1: 简单确定性案例
        token_level_scores = torch.tensor([[1.0, 2.0, 3.0]])  # 序列分数和: 6.0
        acc = torch.tensor([1.0])
        response_mask = torch.ones(1, 3)
        beta = 1.0
        
        result = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta)
        
        # 手动计算验证:
        # 序列分数 = 6.0 * 1.0 = 6.0
        # 概率 = sigmoid(6.0) ≈ 0.9975
        # BCE损失 = -1.0 * log(0.9975) - 0.0 * log(1-0.9975) ≈ 0.0025
        expected_prob = torch.sigmoid(torch.tensor(6.0))
        expected_loss = torch.nn.functional.binary_cross_entropy(expected_prob.unsqueeze(0), acc)
        self.assertAlmostEqual(result.item(), expected_loss.item(), places=30)
        
        # 测试样例2: 负分数情况
        token_level_scores = torch.tensor([[-1.0, -2.0]])  # 序列分数和: -3.0
        acc = torch.tensor([0.0])  # 期望低概率
        response_mask = torch.ones(1, 2)
        beta = 1.0
        
        result = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta)
        
        # 手动计算验证:
        # 序列分数 = -3.0 * 1.0 = -3.0
        # 概率 = sigmoid(-3.0) ≈ 0.0474
        # BCE损失 = -0.0 * log(0.0474) - 1.0 * log(1-0.0474) ≈ 0.0487
        expected_prob = torch.sigmoid(torch.tensor(-3.0))
        expected_loss = torch.nn.functional.binary_cross_entropy(expected_prob.unsqueeze(0), acc)
        self.assertAlmostEqual(result.item(), expected_loss.item(), places=30)
        
        # 测试样例3: 多样本批次
        token_level_scores = torch.tensor([
            [2.0, 1.0],    # 序列分数: 3.0
            [0.5, -0.5],   # 序列分数: 0.0
            [-1.0, -1.0]   # 序列分数: -2.0
        ])
        acc = torch.tensor([0.8, 0.5, 0.2])
        response_mask = torch.ones(3, 2)
        beta = 0.5
        
        result = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta)
        
        # 手动计算验证:
        # 序列分数 = [3.0*0.5, 0.0*0.5, -2.0*0.5] = [1.5, 0.0, -1.0]
        # 概率 = sigmoid([1.5, 0.0, -1.0])
        seq_scores = torch.tensor([3.0, 0.0, -2.0]) * beta
        expected_probs = torch.sigmoid(seq_scores)
        expected_loss = torch.nn.functional.binary_cross_entropy(expected_probs, acc)
        self.assertAlmostEqual(result.item(), expected_loss.item(), places=30)
        
        # 测试样例4: 带mask的情况
        token_level_scores = torch.tensor([
            [2.0, 3.0, 1.0],  # 有效分数: 2.0 + 3.0 = 5.0 (最后一个被mask)
            [1.0, 1.0, 1.0]   # 有效分数: 1.0 + 1.0 + 1.0 = 3.0 (全部有效)
        ])
        acc = torch.tensor([0.9, 0.7])
        response_mask = torch.tensor([
            [1.0, 1.0, 0.0],  # 最后一个无效
            [1.0, 1.0, 1.0]   # 全部有效
        ])
        beta = 0.2
        
        result = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta)
        
        # 手动计算验证:
        # 有效序列分数 = [5.0*0.2, 3.0*0.2] = [1.0, 0.6]
        # 概率 = sigmoid([1.0, 0.6])
        seq_scores = torch.tensor([5.0, 3.0]) * beta
        expected_probs = torch.sigmoid(seq_scores)
        expected_loss = torch.nn.functional.binary_cross_entropy(expected_probs, acc)
        self.assertAlmostEqual(result.item(), expected_loss.item(), places=30)

    def test_beta_scaling_input_output_samples(self):
        """测试beta参数缩放的具体输入输出样例"""
        token_level_scores = torch.tensor([[4.0, 2.0]])  # 序列分数: 6.0
        acc = torch.tensor([0.8])
        response_mask = torch.ones(1, 2)
        
        # 测试不同beta值的效果
        beta_small = 0.1
        result_small = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta_small)
        
        # 手动计算: 序列分数 = 6.0 * 0.1 = 0.6
        expected_prob_small = torch.sigmoid(torch.tensor(6.0 * beta_small))
        expected_loss_small = torch.nn.functional.binary_cross_entropy(expected_prob_small.unsqueeze(0), acc)
        self.assertAlmostEqual(result_small.item(), expected_loss_small.item(), places=30)
        
        beta_large = 2.0
        result_large = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta_large)
        
        # 手动计算: 序列分数 = 6.0 * 2.0 = 12.0
        expected_prob_large = torch.sigmoid(torch.tensor(6.0 * beta_large))
        expected_loss_large = torch.nn.functional.binary_cross_entropy(expected_prob_large.unsqueeze(0), acc)
        self.assertAlmostEqual(result_large.item(), expected_loss_large.item(), places=30)
        
        # beta越大，序列分数的影响越明显
        # 对于正的序列分数和准确率标签0.8，较大的beta会让预测概率更接近1
        # 但目标是0.8，所以过高的预测概率(接近1)会导致更大的损失
        # 因此较大的beta应该产生较大的损失
        self.assertGreater(result_large.item(), result_small.item())

    def test_perfect_prediction_cases(self):
        """测试完美预测情况的输入输出样例"""
        # 测试样例1: 高分数+高准确率 (应该有很小的损失)
        token_level_scores = torch.tensor([[10.0, 5.0]])  # 很高的分数
        acc = torch.tensor([1.0])  # 完美准确率
        response_mask = torch.ones(1, 2)
        beta = 0.1
        
        result = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta)
        # print(f"result: {result.item():.30f}")
        expected_result = torch.tensor(0.201413318514823913574218750000)
        # 序列分数 = 15.0 * 0.1 = 1.5，sigmoid(1.5) ≈ 0.8176
        # BCE损失应该很小，因为预测概率和目标都很高
        # expected_prob = torch.sigmoid(torch.tensor(15.0 * beta))
        # expected_loss = torch.nn.functional.binary_cross_entropy(expected_prob.unsqueeze(0), acc)
        # self.assertAlmostEqual(result.item(), expected_loss.item(), places=5)
        # self.assertLess(result.item(), 0.25)  # 损失应该相对较小
        torch.testing.assert_close(result, expected_result, atol=1e-30, rtol=1e-30)
        # 测试样例2: 低分数+低准确率 (应该有很小的损失)
        token_level_scores = torch.tensor([[-5.0, -12.0]])  # 很低的分数
        acc = torch.tensor([0.0])  # 零准确率
        response_mask = torch.ones(1, 2)
        beta = 0.1
        
        result = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta)
        # print(f"result: {result.item():.30f}")
        expected_result = torch.tensor(0.167786017060279846191406250000)
        # 序列分数 = -15.0 * 0.1 = -1.5，sigmoid(-1.5) ≈ 0.1824
        # BCE损失应该很小，因为预测概率很低，目标也是0
        # expected_prob = torch.sigmoid(torch.tensor(-15.0 * beta))
        # expected_loss = torch.nn.functional.binary_cross_entropy(expected_prob.unsqueeze(0), acc)
        # self.assertAlmostEqual(result.item(), expected_loss.item(), places=5)
        # self.assertLess(result.item(), 0.25)  # 损失应该相对较小
        torch.testing.assert_close(result, expected_result, atol=1e-30, rtol=1e-30)


if __name__ == "__main__":
    unittest.main()
