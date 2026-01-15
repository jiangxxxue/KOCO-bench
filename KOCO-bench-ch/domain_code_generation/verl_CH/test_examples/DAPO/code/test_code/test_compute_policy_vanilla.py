import unittest
import torch
import sys
import os
from unittest.mock import MagicMock

# Add the verl directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the actual function we want to test
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla


class TestComputePolicyLossVanilla(unittest.TestCase):
    """测试compute_policy_loss_vanilla函数 - DAPO文档第2节描述的解耦裁剪策略梯度损失"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.response_length = 4
    
    def _create_mock_config(self, clip_ratio=0.2, clip_ratio_low=None, clip_ratio_high=None, clip_ratio_c=3.0):
        """创建mock的config对象"""
        config = MagicMock()
        config.clip_ratio = clip_ratio
        config.clip_ratio_low = clip_ratio_low
        config.clip_ratio_high = clip_ratio_high
        config.tis_imp_ratio_cap = 0  # 默认禁用truncated importance sampling
        config.get = MagicMock(return_value=clip_ratio_c)
        return config
    
    def test_compute_policy_loss_vanilla_basic(self):
        """测试基本功能"""
        old_log_prob = torch.tensor([
            [-1.0, -2.0, -1.5, -1.2],
            [-0.8, -1.8, -1.3, -1.0]
        ])
        log_prob = torch.tensor([
            [-0.9, -1.9, -1.4, -1.1],
            [-0.7, -1.7, -1.2, -0.9]
        ])
        advantages = torch.tensor([
            [1.0, -0.5, 0.3, 0.0],
            [0.5, 1.0, -0.3, 0.2]
        ])
        response_mask = torch.ones(self.batch_size, self.response_length)
        
        config = self._create_mock_config(clip_ratio=0.2)
        
        # 调用真实函数
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_vanilla(
            old_log_prob, log_prob, advantages, response_mask,
            loss_agg_mode="token-mean", config=config
        )

        # print(pg_loss)
        # print(pg_clipfrac)
        # print(ppo_kl)
        # print(pg_clipfrac_lower)
        expected_pg_loss = torch.tensor(-0.3039)
        expected_pg_clipfrac = torch.tensor(0.0000)
        expected_ppo_kl = torch.tensor(-0.1000)
        expected_pg_clipfrac_lower = torch.tensor(0.0000)
        torch.testing.assert_close(pg_loss, expected_pg_loss, atol=1e-4, rtol=1e-4 )
        torch.testing.assert_close(pg_clipfrac, expected_pg_clipfrac, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(ppo_kl, expected_ppo_kl, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(pg_clipfrac_lower, expected_pg_clipfrac_lower, atol=1e-4, rtol=1e-4)

        # # 验证输出
        # self.assertTrue(torch.is_tensor(pg_loss))
        # self.assertTrue(torch.is_tensor(pg_clipfrac))
        # self.assertTrue(torch.is_tensor(ppo_kl))
        # self.assertTrue(torch.is_tensor(pg_clipfrac_lower))
        
        # # 验证都是标量
        # self.assertEqual(pg_loss.shape, torch.Size([]))
        # self.assertEqual(pg_clipfrac.shape, torch.Size([]))
        # self.assertEqual(ppo_kl.shape, torch.Size([]))
        # self.assertEqual(pg_clipfrac_lower.shape, torch.Size([]))
        
        # # 验证值是有限的
        # self.assertTrue(torch.isfinite(pg_loss))
        # self.assertTrue(torch.isfinite(pg_clipfrac))
        # self.assertTrue(torch.isfinite(ppo_kl))
        # self.assertTrue(torch.isfinite(pg_clipfrac_lower))
        
        # # 验证clipfrac在[0, 1]范围内
        # self.assertGreaterEqual(pg_clipfrac.item(), 0.0)
        # self.assertLessEqual(pg_clipfrac.item(), 1.0)
        # self.assertGreaterEqual(pg_clipfrac_lower.item(), 0.0)
        # self.assertLessEqual(pg_clipfrac_lower.item(), 1.0)
    
    def test_compute_policy_loss_vanilla_decoupled_clip(self):
        """测试解耦裁剪（Clip-Higher）机制"""
        # 创建一个正优势的场景
        old_log_prob = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
        log_prob = torch.tensor([[-0.5, -0.5, -0.5, -0.5]])  # 提升很多
        advantages = torch.tensor([[1.0, 1.0, 1.0, 1.0]])  # 全正优势
        response_mask = torch.ones(1, 4)
        
        # 标准PPO配置（对称裁剪）
        config_symmetric = self._create_mock_config(clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.2)
        pg_loss_symmetric, _, _, _ = compute_policy_loss_vanilla(
            old_log_prob, log_prob, advantages, response_mask,
            loss_agg_mode="token-mean", config=config_symmetric
        )
        
        # DAPO配置（非对称裁剪，clip_ratio_high > clip_ratio_low）
        config_dapo = self._create_mock_config(clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.28)
        pg_loss_dapo, _, _, _ = compute_policy_loss_vanilla(
            old_log_prob, log_prob, advantages, response_mask,
            loss_agg_mode="token-mean", config=config_dapo
        )

        # print(pg_loss_dapo)
        # print(pg_loss_symmetric)

        expected_pg_loss_dapo = torch.tensor(-1.2800)
        expected_pg_loss_symmetric = torch.tensor(-1.2000)
        torch.testing.assert_close(pg_loss_dapo, expected_pg_loss_dapo, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(pg_loss_symmetric, expected_pg_loss_symmetric, atol=1e-4, rtol=1e-4)
        
        # 对于正优势，DAPO应该允许更激进的更新（损失更负，即更新幅度更大）
        # 由于是-A*ratio的形式，正优势时loss为负，更激进意味着更负
        self.assertLess(pg_loss_dapo.item(), pg_loss_symmetric.item())
    
    def test_compute_policy_loss_vanilla_negative_advantage_conservatism(self):
        """测试负优势时的保守性"""
        # 创建一个负优势的场景
        old_log_prob = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
        log_prob = torch.tensor([[-1.5, -1.5, -1.5, -1.5]])  # 降低很多（坏行为被减少）
        advantages = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])  # 全负优势
        response_mask = torch.ones(1, 4)
        
        # DAPO配置
        config = self._create_mock_config(clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.28)
        pg_loss, pg_clipfrac, _, pg_clipfrac_lower = compute_policy_loss_vanilla(
            old_log_prob, log_prob, advantages, response_mask,
            loss_agg_mode="token-mean", config=config
        )

        # print(pg_loss)
        # print(pg_clipfrac)
        # print(pg_clipfrac_lower)
        expected_pg_loss = torch.tensor(0.8000)
        expected_pg_clipfrac = torch.tensor(1.0000)
        expected_pg_clipfrac_lower = torch.tensor(0.0000)
        torch.testing.assert_close(pg_loss, expected_pg_loss, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(pg_clipfrac, expected_pg_clipfrac, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(pg_clipfrac_lower, expected_pg_clipfrac_lower, atol=1e-4, rtol=1e-4)
        # 验证负优势时使用clip_ratio_low
        self.assertTrue(torch.isfinite(pg_loss))
        # 负优势时可能触发双裁剪机制
        self.assertGreaterEqual(pg_clipfrac_lower.item(), 0.0)
    
    def test_compute_policy_loss_vanilla_ratio_clipping(self):
        """测试重要性采样比的裁剪"""
        old_log_prob = torch.tensor([[-1.0, -1.0]])
        response_mask = torch.ones(1, 2)
        
        # 测试ratio > 1 + clip_ratio_high的情况
        log_prob_high = torch.tensor([[-0.2, -0.2]])  # 大幅提升
        advantages_pos = torch.tensor([[1.0, 1.0]])
        
        config = self._create_mock_config(clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.2)
        _, pg_clipfrac_high, _, _ = compute_policy_loss_vanilla(
            old_log_prob, log_prob_high, advantages_pos, response_mask,
            loss_agg_mode="token-mean", config=config
        )
        
        # 应该有裁剪发生
        self.assertGreater(pg_clipfrac_high.item(), 0.0)
        
        # 测试ratio < 1 - clip_ratio_low的情况
        log_prob_low = torch.tensor([[-1.8, -1.8]])  # 大幅降低
        advantages_neg = torch.tensor([[-1.0, -1.0]])
        
        _, pg_clipfrac_low, _, _ = compute_policy_loss_vanilla(
            old_log_prob, log_prob_low, advantages_neg, response_mask,
            loss_agg_mode="token-mean", config=config
        )

        # print(pg_clipfrac_high)
        # print(pg_clipfrac_low)
        expected_pg_clipfrac_high = torch.tensor(1.)
        expected_pg_clipfrac_low = torch.tensor(1.)
        torch.testing.assert_close(pg_clipfrac_high, expected_pg_clipfrac_high, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(pg_clipfrac_low, expected_pg_clipfrac_low, atol=1e-4, rtol=1e-4)
        # 应该有裁剪发生
        self.assertGreater(pg_clipfrac_low.item(), 0.0)
    
    def test_compute_policy_loss_vanilla_kl_calculation(self):
        """测试KL散度计算"""
        old_log_prob = torch.tensor([[-1.0, -2.0, -1.5, -1.2]])
        log_prob = torch.tensor([[-0.9, -1.9, -1.4, -1.1]])
        advantages = torch.zeros(1, 4)
        response_mask = torch.ones(1, 4)
        
        config = self._create_mock_config()
        
        _, _, ppo_kl, _ = compute_policy_loss_vanilla(
            old_log_prob, log_prob, advantages, response_mask,
            loss_agg_mode="token-mean", config=config
        )
        
        # print(ppo_kl)

        expected_ppo_kl = torch.tensor(-0.1000)
        torch.testing.assert_close(ppo_kl, expected_ppo_kl, atol=1e-4, rtol=1e-4)
        # 手动计算KL（近似）
        # KL ≈ -(log_prob - old_log_prob)
        # manual_kl = -(log_prob - old_log_prob).mean()
        # torch.testing.assert_close(ppo_kl, manual_kl, atol=1e-4, rtol=1e-4)
    
    def test_compute_policy_loss_vanilla_extreme_log_prob_stability(self):
        """测试极端log_prob值的数值稳定性"""
        # 极端的log_prob值
        old_log_prob = torch.tensor([[-100.0, -1.0]])
        log_prob = torch.tensor([[50.0, -1.1]])  # 差值很大
        advantages = torch.tensor([[1.0, 1.0]])
        response_mask = torch.ones(1, 2)
        
        config = self._create_mock_config()
        
        pg_loss, _, ppo_kl, _ = compute_policy_loss_vanilla(
            old_log_prob, log_prob, advantages, response_mask,
            loss_agg_mode="token-mean", config=config
        )
        
        # print(pg_loss)
        # print(ppo_kl)
        expected_pg_loss = torch.tensor(-1.0524)
        expected_ppo_kl = torch.tensor(-9.9500)
        torch.testing.assert_close(pg_loss, expected_pg_loss, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(ppo_kl, expected_ppo_kl, atol=1e-4, rtol=1e-4)
        # 应该通过clamp保证数值稳定
        self.assertTrue(torch.isfinite(pg_loss))
        self.assertTrue(torch.isfinite(ppo_kl))
    
    def test_compute_policy_loss_vanilla_mixed_advantages(self):
        """测试混合正负优势"""
        old_log_prob = torch.tensor([
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0]
        ])
        log_prob = torch.tensor([
            [-0.8, -0.8, -1.2, -1.2],
            [-0.8, -0.8, -1.2, -1.2]
        ])
        advantages = torch.tensor([
            [1.0, -1.0, 0.5, -0.5],  # 混合
            [2.0, -2.0, 0.0, 0.0]    # 混合
        ])
        response_mask = torch.ones(self.batch_size, self.response_length)
        
        config = self._create_mock_config(clip_ratio_low=0.2, clip_ratio_high=0.28)
        
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_vanilla(
            old_log_prob, log_prob, advantages, response_mask,
            loss_agg_mode="token-mean", config=config
        )

        
        # print(pg_loss)
        # print(pg_clipfrac)
        # print(ppo_kl)
        # print(pg_clipfrac_lower)
        expected_pg_loss = torch.tensor(0.)
        expected_pg_clipfrac = torch.tensor(0.)
        expected_ppo_kl = torch.tensor(2.9802e-08)
        expected_pg_clipfrac_lower = torch.tensor(0.0000)
        torch.testing.assert_close(pg_loss, expected_pg_loss, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(pg_clipfrac, expected_pg_clipfrac, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(ppo_kl, expected_ppo_kl, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(pg_clipfrac_lower, expected_pg_clipfrac_lower, atol=1e-4, rtol=1e-4)
        # 验证所有输出都是有限的
        self.assertTrue(torch.isfinite(pg_loss))
        self.assertTrue(torch.isfinite(pg_clipfrac))
        self.assertTrue(torch.isfinite(ppo_kl))
        self.assertTrue(torch.isfinite(pg_clipfrac_lower))
    
    def test_compute_policy_loss_vanilla_different_agg_modes(self):
        """测试不同的损失聚合模式"""
        old_log_prob = torch.tensor([
            [-1.0, -2.0, -1.5, 0.0],
            [-0.8, -1.8, -1.3, -1.0]
        ])
        log_prob = torch.tensor([
            [-0.9, -1.9, -1.4, 0.0],
            [-0.7, -1.7, -1.2, -0.9]
        ])
        advantages = torch.tensor([
            [1.0, -0.5, 0.3, 0.0],
            [0.5, 1.0, -0.3, 0.2]
        ])
        response_mask = torch.tensor([
            [1.0, 1.0, 1.0, 0.0],  # 最后一个token被mask
            [1.0, 1.0, 1.0, 1.0]
        ])
        
        config = self._create_mock_config()
        
        losses = []
        # 测试三种聚合模式
        for mode in ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean"]:
            pg_loss, _, _, _ = compute_policy_loss_vanilla(
                old_log_prob, log_prob, advantages, response_mask,
                loss_agg_mode=mode, config=config
            )
            
            losses.append(pg_loss)
            # self.assertTrue(torch.isfinite(pg_loss))
            # self.assertEqual(pg_loss.shape, torch.Size([]))
        # print(losses)

        expected_losses = [torch.tensor(-0.3473), torch.tensor(-1.2157), torch.tensor(-0.3408)]
        torch.testing.assert_close(losses[0], expected_losses[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(losses[1], expected_losses[1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(losses[2], expected_losses[2], atol=1e-4, rtol=1e-4)
    
    def test_compute_policy_loss_vanilla_gradient_flow(self):
        """测试梯度反向传播"""
        old_log_prob = torch.tensor([[-1.0, -1.0, -1.0]], requires_grad=False)
        log_prob = torch.tensor([[-0.9, -0.9, -0.9]], requires_grad=True)
        advantages = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=False)
        response_mask = torch.ones(1, 3)
        
        config = self._create_mock_config()
        
        pg_loss, _, _, _ = compute_policy_loss_vanilla(
            old_log_prob, log_prob, advantages, response_mask,
            loss_agg_mode="token-mean", config=config
        )
        
        # 反向传播
        pg_loss.backward()
        
        # print(log_prob.grad)

        expected_log_prob_grad = torch.tensor([[-0.3684, -0.3684, -0.3684]])
        torch.testing.assert_close(log_prob.grad, expected_log_prob_grad, atol=1e-4, rtol=1e-4)
        # 验证梯度存在
        self.assertIsNotNone(log_prob.grad)
        self.assertTrue(torch.isfinite(log_prob.grad).all())
    
    def test_compute_policy_loss_vanilla_zero_advantages(self):
        """测试全零优势"""
        old_log_prob = torch.tensor([[-1.0, -1.0, -1.0]])
        log_prob = torch.tensor([[-0.8, -0.8, -0.8]])
        advantages = torch.zeros(1, 3)
        response_mask = torch.ones(1, 3)
        
        config = self._create_mock_config()
        
        pg_loss, _, _, _ = compute_policy_loss_vanilla(
            old_log_prob, log_prob, advantages, response_mask,
            loss_agg_mode="token-mean", config=config
        )
        
        # 零优势应该导致零损失（或接近零）
        torch.testing.assert_close(pg_loss, torch.tensor(0.0), atol=1e-4, rtol=1e-4)
    
    def test_compute_policy_loss_vanilla_dual_clip(self):
        """测试双裁剪机制（clip_ratio_c）"""
        old_log_prob = torch.tensor([[-1.0, -1.0]])
        log_prob = torch.tensor([[-3.0, -3.0]])  # 大幅降低（极端负优势场景）
        advantages = torch.tensor([[-5.0, -5.0]])  # 极端负优势
        response_mask = torch.ones(1, 2)
        
        config = self._create_mock_config(clip_ratio_c=3.0)
        
        pg_loss, _, _, pg_clipfrac_lower = compute_policy_loss_vanilla(
            old_log_prob, log_prob, advantages, response_mask,
            loss_agg_mode="token-mean", config=config
        )
        
        # print(pg_loss)
        # print(pg_clipfrac_lower)

        expected_pg_loss = torch.tensor(4.)
        expected_pg_clipfrac_lower = torch.tensor(0.0000)
        torch.testing.assert_close(pg_loss, expected_pg_loss, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(pg_clipfrac_lower, expected_pg_clipfrac_lower, atol=1e-4, rtol=1e-4)
        # 双裁剪应该限制极端负优势的影响
        self.assertTrue(torch.isfinite(pg_loss))
        # 可能触发下界裁剪
        self.assertGreaterEqual(pg_clipfrac_lower.item(), 0.0)
    
    def test_compute_policy_loss_vanilla_backward_compatibility(self):
        """测试向后兼容性（未设置clip_ratio_low/high时应退化为标准PPO）"""
        old_log_prob = torch.tensor([[-1.0, -1.0, -1.0]])
        log_prob = torch.tensor([[-0.8, -0.8, -0.8]])
        advantages = torch.tensor([[1.0, 1.0, 1.0]])
        response_mask = torch.ones(1, 3)
        
        # 只设置clip_ratio，不设置low/high
        config = self._create_mock_config(clip_ratio=0.2, clip_ratio_low=None, clip_ratio_high=None)
        
        pg_loss, _, _, _ = compute_policy_loss_vanilla(
            old_log_prob, log_prob, advantages, response_mask,
            loss_agg_mode="token-mean", config=config
        )
        
        # print(pg_loss)
        expected_pg_loss = torch.tensor(-1.2000)
        torch.testing.assert_close(pg_loss, expected_pg_loss, atol=1e-4, rtol=1e-4)
        # 应该使用clip_ratio作为两个方向的裁剪参数
        self.assertTrue(torch.isfinite(pg_loss))


if __name__ == '__main__':
    unittest.main()

