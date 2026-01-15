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
from prime.prime_core_algos import compute_detach_dpo_loss_rm


class TestDetachDPOLoss(unittest.TestCase):
    """测试compute_detach_dpo_loss_rm函数 - 文档第2节描述的分离式DPO损失计算"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_compute_detach_dpo_loss_rm_basic(self):
        """测试分离DPO损失计算的基本功能"""
        batch_size = 4
        seq_len = 10
        n_samples = 2
        
        token_level_scores = torch.tensor([[0.1,0.3,0.6,-0.1,0.2,0.5,0.1,0.3,0.6,-0.1], [0.2,0.3,0.6,-0.4,0.9,-0.1,0.2,0.5,0.1,0.3], [-0.1,0.3,0.6,-0.1,0.2,0.7,1.0,0.2,0.5,0.1], [0.1,0.8,0.6,-0.1,0.2,0.5,0.1,0.5,0.1,0.3]])
        acc = torch.tensor([0.8, 0.3, 1.0, 0.0])
        Q_bc = torch.tensor([[0.1,0.3], [0.4,0.5], [0.2,0.6], [0.1,0.9]])
        acc_bc = torch.tensor([[0.8, 0.3], [0.7, 0.2], [0.8, 0.3], [0.6, 0.1]])
        response_mask = torch.ones(batch_size, seq_len)
        beta = 0.1
        
        losses = []
        # 测试不同的bon_mode
        for bon_mode in ["none", "bon_rm", "bon_acc"]:
            with self.subTest(bon_mode=bon_mode):
                loss = compute_detach_dpo_loss_rm(
                    token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode=bon_mode
                )
                losses.append(loss.item())
                # self.assertEqual(loss.dim(), 0)
                # self.assertTrue(torch.is_tensor(loss))
                # self.assertFalse(torch.isnan(loss))
                # self.assertFalse(torch.isinf(loss))
        # print(f"losses: {losses[0]:.30f}")
        # print(f"losses: {losses[1]:.30f}")
        # print(f"losses: {losses[2]:.30f}")
        expected_losses = [0.642190814018249511718750000000, 5.137526512145996093750000000000, 2.880730152130126953125000000000]

        torch.testing.assert_close(losses, expected_losses, atol=1e-30, rtol=1e-30)
    
    def test_dpo_loss_mathematical_properties(self):
        """测试分离DPO损失的数学属性"""
        batch_size = 2
        seq_len = 4
        n_samples = 2
        
        # 创建确定性数据
        token_level_scores = torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
        acc = torch.tensor([1.0, 0.0])
        Q_bc = torch.tensor([[1.5, 0.5], [1.5, 0.5]])  # 批次中的Q值
        acc_bc = torch.tensor([[1.0, 0.0], [1.0, 0.0]])  # 批次中的准确率
        response_mask = torch.ones(batch_size, seq_len)
        beta = 0.1
        
        loss = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )
        # print(f"loss: {loss.item():.30f}")
        expected_loss = torch.tensor(0.801718711853027343750000000000)
        # 手动验证损失计算逻辑
        Q_current = (token_level_scores * response_mask).sum(dim=1) * beta  # [0.4, 0.8]
        
        # 验证损失是有效值
        # self.assertFalse(torch.isnan(loss))
        # self.assertFalse(torch.isinf(loss))
        # self.assertGreaterEqual(loss.item(), 0.0)
        torch.testing.assert_close(loss, expected_loss, atol=1e-30, rtol=1e-30)

    def test_bon_mode_none(self):
        """测试bon_mode="none"的权重计算"""
        batch_size = 4
        seq_len = 6
        n_samples = 2
        
        token_level_scores = torch.tensor([[0.1,0.3,0.6,-0.1,0.2,0.5], [0.2,0.3,0.6,-0.4,0.9,-0.1], [-0.1,0.3,0.6,-0.1,0.2,0.7], [0.1,0.8,0.6,-0.1,0.2,0.5]])
        acc = torch.tensor([0.8, 0.3, 1.0, 0.0])
        Q_bc = torch.tensor([[0.1,0.3], [0.4,0.5], [0.2,0.6], [0.1,0.9]])
        acc_bc = torch.tensor([[0.8, 0.3], [0.7, 0.2], [0.8, 0.3], [0.6, 0.1]])
        # Q_bc = torch.randn(batch_size, n_samples)
        # acc_bc = torch.rand(batch_size, n_samples)
        response_mask = torch.ones(batch_size, seq_len)
        beta = 0.1
        
        loss = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )
        # print(loss)
        # print(f"loss: {loss.item():.30f}")
        expected_loss = torch.tensor(0.671486198902130126953125000000)
        # none模式应该产生有效损失
        # self.assertFalse(torch.isnan(loss))
        # self.assertFalse(torch.isinf(loss))
        torch.testing.assert_close(loss, expected_loss, atol=1e-30, rtol=1e-30)
    
    def test_bon_mode_rm(self):
        """测试bon_mode="bon_rm"的BoN权重计算"""
        batch_size = 4
        seq_len = 6
        n_samples = 3
        
        token_level_scores = torch.tensor([[0.1,0.3,0.6,-0.1,0.2,0.5], [0.2,0.3,0.6,-0.4,0.9,-0.1], [-0.1,0.3,0.6,-0.1,0.2,0.7], [0.1,0.8,0.6,-0.1,0.2,0.5]])
        acc = torch.tensor([0.8, 0.3, 1.0, 0.0])
        Q_bc = torch.tensor([[0.1,0.3], [0.4,0.5], [0.2,0.6], [0.1,0.9]])
        acc_bc = torch.tensor([[0.8, 0.3], [0.7, 0.2], [0.8, 0.3], [0.6, 0.1]])
        response_mask = torch.ones(batch_size, seq_len)
        beta = 0.1
        
        loss = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="bon_rm"
        )
        # print(loss)
        # print(f"loss: {loss.item():.30f}")
        expected_loss = torch.tensor(5.371889591217041015625000000000)

        # bon_rm模式基于Q值排序分配权重
        # self.assertFalse(torch.isnan(loss))
        # self.assertFalse(torch.isinf(loss))
        torch.testing.assert_close(loss, expected_loss, atol=1e-30, rtol=1e-30)
    
    def test_bon_mode_acc(self):
        """测试bon_mode="bon_acc"的BoN权重计算"""
        batch_size = 4
        seq_len = 6
        n_samples = 3
        
        token_level_scores = torch.tensor([[0.1,0.3,0.6,-0.1,0.2,0.7], [0.2,0.5,0.6,-0.4,0.9,-0.1], [-0.1,0.3,0.6,-0.1,0.2,0.7], [0.1,0.8,0.6,-0.1,0.2,0.5]])
        acc = torch.tensor([0.8, 0.3, 1.0, 0.0])
        Q_bc = torch.tensor([[0.1,0.3], [0.4,0.5], [0.2,0.6], [0.1,0.9]])
        acc_bc = torch.tensor([[0.8, 0.3], [0.7, 0.2], [0.8, 0.3], [0.6, 0.1]])
        response_mask = torch.ones(batch_size, seq_len)
        beta = 0.1
        
        loss = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="bon_acc"
        )
        # print(loss)
        # print(f"loss: {loss.item():.30f}")
        expected_loss = torch.tensor(3.146752834320068359375000000000)
        # bon_acc模式基于准确率排序分配权重
        # self.assertFalse(torch.isnan(loss))
        # self.assertFalse(torch.isinf(loss))
        torch.testing.assert_close(loss, expected_loss, atol=1e-30, rtol=1e-30)
    
    def test_different_beta_values(self):
        """测试不同beta参数对分离DPO损失的影响"""
        batch_size = 2
        seq_len = 4
        n_samples = 2
        
        token_level_scores = torch.tensor([[0.6,-0.1,0.2,0.5], [0.2,0.3,0.9,-0.1]])
        acc = torch.tensor([0.8, 1.0])
        Q_bc = torch.tensor([[0.1,0.3], [0.4,0.5]])
        acc_bc = torch.tensor([[0.8, 0.3], [0.7, 0.2]])
        response_mask = torch.ones(batch_size, seq_len)
        
        betas = [0.01, 0.1, 1.0, 10.0]
        losses = []
        
        for beta in betas:
            loss = compute_detach_dpo_loss_rm(
                token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
            )
            losses.append(loss.item())
            
            # 验证损失都是有效的
            # self.assertFalse(torch.isnan(loss))
            # self.assertFalse(torch.isinf(loss))
        
        # print(losses)
        # 不同beta值应该产生不同的损失
        # self.assertTrue(len(set(losses)) > 1, "Different beta values should produce different losses")
        # print(f"losses: {losses[0]:.30f}")
        # print(f"losses: {losses[1]:.30f}")
        # print(f"losses: {losses[2]:.30f}")
        # print(f"losses: {losses[3]:.30f}")
        expected_losses = [0.688781797885894775390625000000, 0.650354683399200439453125000000, 0.348509460687637329101562500000,  0.000163420278113335371017456055]
        torch.testing.assert_close(losses, expected_losses, atol=1e-30, rtol=1e-30)
    
    def test_contrasting_samples_selection(self):
        """测试对比样本选择逻辑"""
        batch_size = 4
        seq_len = 4
        n_samples = 2
        
        # 创建具有不同准确率的样本
        token_level_scores = torch.tensor([[0.5, 0.7, 1.0, 0.1], [0.0, -1.0, 0.6, 0.2], [0.3, 0.0, 0.44, 0.5], [0.0, 0.45, 0.1, -0.3]])
        acc = torch.tensor([1.0, 0.0, 1.0, 0.0])  # 交替的准确率标签
        
        # Q_bc和acc_bc应该包含不同准确率的样本
        Q_bc = torch.tensor([[0.1,0.3], [0.4,0.5], [0.2,0.6], [0.1,0.9]])
        acc_bc = torch.tensor([[0.8, 0.3], [0.7, 0.2], [0.8, 0.3], [0.6, 0.1]])
        
        response_mask = torch.ones(batch_size, seq_len)
        beta = 0.1
        
        loss = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )
        # print(loss)
        # print(f"loss: {loss.item():.30f}")
        expected_loss = torch.tensor( 0.646894752979278564453125000000)
        # 验证对比样本选择逻辑产生有效损失
        # self.assertFalse(torch.isnan(loss))
        # self.assertFalse(torch.isinf(loss))
        torch.testing.assert_close(loss, expected_loss, atol=1e-30, rtol=1e-30)
    
    def test_direction_calculation(self):
        """测试偏好方向计算"""
        batch_size = 4
        seq_len = 3
        n_samples = 2
        
        # 测试不同准确率对偏好方向的影响
        token_level_scores = torch.ones(batch_size, seq_len)
        acc_positive = torch.tensor([0.8, 0.9, 0.7, 0.6])  # 正确样本
        acc_negative = torch.tensor([0.2, 0.1, 0.3, 0.4])  # 错误样本
        
        Q_bc = torch.tensor([[0.1,0.3], [0.4,0.5], [0.2,0.6], [0.1,0.9]])
        acc_bc = torch.tensor([[0.8, 0.3], [0.7, 0.2], [0.8, 0.3], [0.6, 0.1]])
        response_mask = torch.ones(batch_size, seq_len)
        beta = 0.1
        
        # 测试正确样本
        loss_pos = compute_detach_dpo_loss_rm(
            token_level_scores, acc_positive, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )
        
        # 测试错误样本
        loss_neg = compute_detach_dpo_loss_rm(
            token_level_scores, acc_negative, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )
        
        # print(loss_pos)
        # print(loss_neg)
        # print(f"loss_pos: {loss_pos.item():.30f}")
        # print(f"loss_neg: {loss_neg.item():.30f}")
        expected_loss_pos = torch.tensor(0.578741252422332763671875000000)
        expected_loss_neg = torch.tensor(0.564178824424743652343750000000)
        torch.testing.assert_close(loss_pos, expected_loss_pos, atol=1e-30, rtol=1e-30)
        torch.testing.assert_close(loss_neg, expected_loss_neg, atol=1e-30, rtol=1e-30)

    def test_input_output_samples(self):
        """测试具体的输入输出样例"""
        # 测试样例1: 简单的两样本情况
        token_level_scores = torch.tensor([
            [2.0, 1.0],  # 当前Q = 3.0 * beta
            [1.0, 0.5]   # 当前Q = 1.5 * beta
        ])
        acc = torch.tensor([1.0, 0.0])  # 第一个样本高准确率，第二个低准确率
        Q_bc = torch.tensor([
            [2.0, 1.0],  # 批次中的Q值
            [2.0, 1.0]
        ])
        acc_bc = torch.tensor([
            [0.8, 0.2],  # 批次中的准确率
            [0.8, 0.2]
        ])
        response_mask = torch.ones(2, 2)
        beta = 0.1
        
        result = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )

        # print(result)
        # print(f"result: {result.item():.30f}")
        expected_result = torch.tensor(0.657052159309387207031250000000)
        torch.testing.assert_close(result, expected_result, atol=1e-30, rtol=1e-30)
        # 手动计算验证:
        # cur_Q = [3.0*0.1, 1.5*0.1] = [0.3, 0.15]
        # 对于样本1 (acc=1.0>0): 选择 acc_bc < 1.0 的样本，即 Q_bc[0][acc_bc[0] < 1.0] = [2.0, 1.0]
        # other_Q[0] = mean([2.0, 1.0]) * 0.1 = 1.5 * 0.1 = 0.15
        # 对于样本2 (acc=0.0<=0): 选择 acc_bc > 0.0 的样本，即 Q_bc[1][acc_bc[1] > 0.0] = [2.0, 1.0]
        # other_Q[1] = mean([2.0, 1.0]) * 0.1 = 1.5 * 0.1 = 0.15
        # direction = [1, -1] (因为 acc > 0 的判断)
        # dpo_loss = -log(sigmoid(([0.3, 0.15] - [0.15, 0.15]) * [1, -1]))
        #          = -log(sigmoid([0.15, 0.0]))
        # cur_Q = torch.tensor([0.3, 0.15])
        # other_Q = torch.tensor([0.15, 0.15])
        # direction = torch.tensor([1.0, -1.0])
        # expected_dpo_loss = -torch.log(torch.sigmoid((cur_Q - other_Q) * direction))
        # expected_result = expected_dpo_loss.mean()
        
        # self.assertAlmostEqual(result.item(), expected_result.item(), places=4)
        
        # 测试样例2: BoN权重模式
        token_level_scores = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        acc = torch.tensor([0.8, 0.6])
        Q_bc = torch.tensor([[1.0, 2.0], [1.5, 2.5]])  # 不同的Q值分布
        acc_bc = torch.tensor([[0.5, 0.9], [0.4, 0.8]])  # 不同的准确率分布
        response_mask = torch.ones(2, 2)
        beta = 0.1
        
        result = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="bon_acc"
        )
        
        # print(result)
        # print(f"result: {result.item():.30f}")
        expected_result = torch.tensor(1.220335960388183593750000000000)
        torch.testing.assert_close(result, expected_result, atol=1e-30, rtol=1e-30)
        # 验证BoN模式产生有效损失
        #     self.assertFalse(torch.isnan(result))
        #     self.assertFalse(torch.isinf(result))
        #     self.assertGreaterEqual(result.item(), 0.0)
        
        # 测试样例3: 带mask的情况
        token_level_scores = torch.tensor([
            [3.0, 2.0, 1.0],  # 有效分数: 3.0 + 2.0 = 5.0
            [1.0, 1.0, 1.0]   # 有效分数: 1.0 + 1.0 + 1.0 = 3.0
        ])
        acc = torch.tensor([0.9, 0.1])
        Q_bc = torch.tensor([[2.0, 1.0], [2.5, 1.5]])
        acc_bc = torch.tensor([[0.7, 0.3], [0.8, 0.2]])
        response_mask = torch.tensor([
            [1.0, 1.0, 0.0],  # 最后一个位置masked
            [1.0, 1.0, 1.0]   # 全部有效
        ])
        beta = 0.2
        
        result = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )

        # print(result)
        # print(f"result: {result.item():.30f}")
        expected_result = torch.tensor(0.420337051153182983398437500000)
        torch.testing.assert_close(result, expected_result, atol=1e-30, rtol=1e-30)
        
        # 手动计算验证mask效果:
        # cur_Q = [5.0*0.2, 3.0*0.2] = [1.0, 0.6]
        expected_cur_Q = torch.tensor([5.0, 3.0]) * beta
        # 验证损失计算是正确的
        # self.assertFalse(torch.isnan(result))
        # self.assertFalse(torch.isinf(result))

    def test_contrast_sample_selection_input_output(self):
        """测试对比样本选择的具体输入输出"""
        # 测试样例: 明确的对比样本选择
        token_level_scores = torch.tensor([[2.0, 2.0]])  # cur_Q = 4.0 * beta
        acc = torch.tensor([0.8])  # 高准确率样本
        Q_bc = torch.tensor([[3.0, 1.0, 2.0]])  # 批次中的Q值
        acc_bc = torch.tensor([[0.9, 0.3, 0.7]])  # 批次中的准确率，不同的对比样本
        response_mask = torch.ones(1, 2)
        beta = 0.1
        
        result = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )
        # print(result)
        # print(f"result: {result.item():.30f}")
        expected_result = torch.tensor(0.575939357280731201171875000000)
        torch.testing.assert_close(result, expected_result, atol=1e-30, rtol=1e-30)
        # 验证高准确率样本的对比选择产生有效损失
        # self.assertFalse(torch.isnan(result))
        # self.assertFalse(torch.isinf(result))
        # self.assertGreaterEqual(result.item(), 0.0)
        
        # 测试样例: 低准确率样本的对比选择
        acc_low = torch.tensor([0.2])  # 低准确率样本
        result_low = compute_detach_dpo_loss_rm(
            token_level_scores, acc_low, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )
        # print(result_low)
        # print(f"result_low: {result_low.item():.30f}")
        expected_result_low = torch.tensor( 0.513015270233154296875000000000)
        torch.testing.assert_close(result_low, expected_result_low, atol=1e-30, rtol=1e-30)
        # 验证低准确率样本的对比选择产生有效损失
        # self.assertFalse(torch.isnan(result_low))
        # self.assertFalse(torch.isinf(result_low))
        # self.assertGreaterEqual(result_low.item(), 0.0)
        
        # 验证不同准确率会产生不同的损失值（体现对比样本选择的差异）
        # self.assertNotAlmostEqual(result.item(), result_low.item(), places=3)

    def test_bon_weighting_input_output(self):
        """测试BoN权重计算的具体输入输出"""
        # 测试样例: bon_rm模式
        token_level_scores = torch.tensor([
            [1.0, 1.0],  # cur_Q = 2.0 * beta = 0.2
            [2.0, 2.0]   # cur_Q = 4.0 * beta = 0.4
        ])
        acc = torch.tensor([0.6, 0.8])
        Q_bc = torch.tensor([
            [1.0, 2.0, 3.0],  # 与cur_Q[0]=0.2比较: [0.1, 0.2, 0.3]
            [3.0, 4.0, 5.0]   # 与cur_Q[1]=0.4比较: [0.3, 0.4, 0.5]
        ])
        acc_bc = torch.tensor([
            [0.4, 0.7, 0.9],
            [0.6, 0.8, 1.0]
        ])
        response_mask = torch.ones(2, 2)
        beta = 0.1
        n_samples = 3
        
        result = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="bon_rm"
        )
        # print(result)
        # print(f"result: {result.item():.30f}")
        expected_result = torch.tensor(1.718391299247741699218750000000)
        torch.testing.assert_close(result, expected_result, atol=1e-30, rtol=1e-30)
        # 手动验证权重计算:
        # cur_Q = [0.2, 0.4]
        # 对于样本1: Q_bc[0] * beta = [0.1, 0.2, 0.3]，满足 <= 0.2 的比例 = 2/3
        # weight[0] = 3 * (2/3)^(3-1) = 3 * (2/3)^2 = 3 * 4/9 = 4/3
        # 对于样本2: Q_bc[1] * beta = [0.3, 0.4, 0.5]，满足 <= 0.4 的比例 = 2/3
        # weight[1] = 3 * (2/3)^(3-1) = 3 * (2/3)^2 = 3 * 4/9 = 4/3
        
        # 验证结果是有效的
        # self.assertFalse(torch.isnan(result))
        # self.assertFalse(torch.isinf(result))
        # self.assertGreaterEqual(result.item(), 0.0)
        
        # 测试样例: bon_acc模式
        result_acc = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="bon_acc"
        )
        
        # print(result_acc)
        # print(f"result_acc: {result_acc.item():.30f}")
        expected_result_acc = torch.tensor(1.073994517326354980468750000000)
        torch.testing.assert_close(result_acc, expected_result_acc, atol=1e-30, rtol=1e-30)
        # 手动验证准确率权重计算:
        # acc = [0.6, 0.8]
        # 对于样本1: acc_bc[0] = [0.4, 0.7, 0.9]，满足 <= 0.6 的比例 = 2/3
        # 对于样本2: acc_bc[1] = [0.6, 0.8, 1.0]，满足 <= 0.8 的比例 = 2/3
        # 权重计算与bon_rm相同
        
        # self.assertFalse(torch.isnan(result_acc))
        # self.assertFalse(torch.isinf(result_acc))
        # self.assertGreaterEqual(result_acc.item(), 0.0)
        
        # bon_rm和bon_acc应该产生不同的结果（因为权重计算基于不同指标）
        # 这里我们只验证都是有效值，不强制要求不相等，因为在某些情况下可能相等

    def test_edge_cases_input_output(self):
        """测试边缘情况的具体输入输出"""
        # 测试样例1: 没有对比样本的情况
        token_level_scores = torch.tensor([[1.0, 2.0]])
        acc = torch.tensor([0.5])
        Q_bc = torch.tensor([[1.0, 2.0]])
        acc_bc = torch.tensor([[0.6, 0.7]])  # 都大于acc=0.5，对于acc<=0的情况没有对比样本
        response_mask = torch.ones(1, 2)
        beta = 0.1
        
        result = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )
        
        # print(result)
        # print(f"result: {result.item():.30f}")
        expected_result = torch.tensor(0.554355263710021972656250000000)
        torch.testing.assert_close(result, expected_result, atol=1e-30, rtol=1e-30)
        # 当没有满足条件的对比样本时，other_Q应该为0
        # cur_Q = 3.0 * 0.1 = 0.3
        # other_Q = 0.0 (因为没有满足 acc_bc < 0.5 的样本)
        # direction = 1.0
        # dpo_loss = -log(sigmoid((0.3 - 0.0) * 1.0)) = -log(sigmoid(0.3))
        # expected_cur_Q = 3.0 * beta
        # expected_other_Q = 0.0
        # expected_direction = 1.0
        # expected_dpo_loss = -torch.log(torch.sigmoid(torch.tensor((expected_cur_Q - expected_other_Q) * expected_direction)))
        
        # self.assertAlmostEqual(result.item(), expected_dpo_loss.item(), places=4)
        
        # 测试样例2: 零分数情况
        zero_scores = torch.zeros(1, 3)
        acc = torch.tensor([0.8])
        Q_bc = torch.tensor([[1.0, 0.5]])
        acc_bc = torch.tensor([[0.6, 0.4]])
        response_mask = torch.ones(1, 3)
        beta = 0.2
        
        result = compute_detach_dpo_loss_rm(
            zero_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )
        # print(result)
        # print(f"result: {result.item():.30f}")
        expected_result = torch.tensor(0.770957052707672119140625000000)
        torch.testing.assert_close(result, expected_result, atol=1e-30, rtol=1e-30)
        # cur_Q = 0.0 * 0.2 = 0.0
        # 选择 acc_bc < 0.8 的样本: [0.6, 0.4] 都满足
        # other_Q = mean([1.0, 0.5]) * 0.2 = 0.75 * 0.2 = 0.15
        # dpo_loss = -log(sigmoid((0.0 - 0.15) * 1.0)) = -log(sigmoid(-0.15))
        # expected_cur_Q = 0.0
        # expected_other_Q = 0.75 * beta
        # expected_dpo_loss = -torch.log(torch.sigmoid(torch.tensor((expected_cur_Q - expected_other_Q) * 1.0)))
        
        # self.assertAlmostEqual(result.item(), expected_dpo_loss.item(), places=4)


if __name__ == "__main__":
    unittest.main()
