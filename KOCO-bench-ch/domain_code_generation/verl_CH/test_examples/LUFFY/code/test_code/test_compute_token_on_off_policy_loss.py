import unittest
import torch
import sys
import os

# Add the verl directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'luffy', 'verl'))

# Import the actual function we want to test
from verl.mix_src.mix_core_alg import compute_token_on_off_policy_loss


class TestComputeTokenOnOffPolicyLoss(unittest.TestCase):
    """测试compute_token_on_off_policy_loss函数 - LUFFY文档第2节描述的混合策略损失计算"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_basic_on_policy_loss(self):
        """测试基本的on-policy PPO损失计算"""
        batch_size, response_length = 2, 4
        
        old_log_prob = torch.tensor([
            [-0.5, -0.3, -0.4, -0.6],
            [-0.2, -0.5, -0.3, -0.4]
        ])
        log_prob = torch.tensor([
            [-0.3, -0.2, -0.3, -0.5],
            [-0.1, -0.4, -0.2, -0.3]
        ])
        advantages = torch.tensor([
            [1.0, 0.5, -0.5, 1.5],
            [0.8, -0.3, 1.2, 0.6]
        ])
        eos_mask = torch.ones(batch_size, response_length)
        prefix_mask = torch.zeros(batch_size, response_length, dtype=torch.bool)  # 全部on-policy
        
        # 调用真实函数
        result = compute_token_on_off_policy_loss(
            old_log_prob, log_prob, advantages, eos_mask,
            cliprange=0.2, clip_upper_bound=100.0, prefix_mask=prefix_mask,
            off_cliprange=None
        )
        
        # 验证返回字典包含所有必需的键
        expected_keys = ['pg_loss', 'off_pg_loss', 'on_pg_loss', 'off_pg_clipfrac', 
                        'on_pg_clipfrac', 'ppo_kl', 'off_policy_prob', 'on_policy_prob',
                        'off_ratio_mean', 'off_ratio_max_clip_frac', 'off_ratio_min_clip_frac']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # 验证损失值是标量
        self.assertTrue(torch.is_tensor(result['pg_loss']))
        self.assertEqual(result['pg_loss'].dim(), 0)
        self.assertTrue(torch.isfinite(result['pg_loss']))
        
        # 全部on-policy时，off_pg_loss应该为0或很小
        self.assertTrue(result['off_pg_loss'].item() == 0.0 or result['off_pg_loss'].item() < 1e-6)
    
    def test_basic_off_policy_loss(self):
        """测试基本的off-policy损失计算"""
        batch_size, response_length = 2, 4
        
        old_log_prob = torch.tensor([
            [-0.5, -0.3, -0.4, -0.6],
            [-0.2, -0.5, -0.3, -0.4]
        ])
        log_prob = torch.tensor([
            [-0.3, -0.2, -0.3, -0.5],
            [-0.1, -0.4, -0.2, -0.3]
        ])
        advantages = torch.tensor([
            [1.0, 0.5, -0.5, 1.5],
            [0.8, -0.3, 1.2, 0.6]
        ])
        eos_mask = torch.ones(batch_size, response_length)
        prefix_mask = torch.ones(batch_size, response_length, dtype=torch.bool)  # 全部off-policy
        
        result = compute_token_on_off_policy_loss(
            old_log_prob, log_prob, advantages, eos_mask,
            cliprange=0.2, clip_upper_bound=100.0, prefix_mask=prefix_mask,
            off_cliprange=None
        )
        
        # print(result['pg_loss'])
        # print(result['off_pg_loss'])

        expected_pg_loss = torch.tensor(-0.4549)
        expected_off_pg_loss = torch.tensor(-0.4549)
        torch.testing.assert_close(result['pg_loss'], expected_pg_loss, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(result['off_pg_loss'], expected_off_pg_loss, atol=1e-4, rtol=1e-4)
        # 验证输出
        # self.assertTrue(torch.isfinite(result['pg_loss']))
        # self.assertTrue(torch.isfinite(result['off_pg_loss']))
        
        # # 全部off-policy时，on_pg_loss可能不为0（因为仍会计算），但不应该影响总损失
        # # 主要验证off_pg_loss有值
        # self.assertGreater(result['off_pg_loss'].abs().item(), 0.0)
    
    def test_mixed_on_off_policy_loss(self):
        """测试混合on-policy和off-policy损失"""
        batch_size, response_length = 2, 4
        
        old_log_prob = torch.tensor([
            [-0.5, -0.3, -0.4, -0.6],
            [-0.2, -0.5, -0.3, -0.4]
        ])
        log_prob = torch.tensor([
            [-0.3, -0.2, -0.3, -0.5],
            [-0.1, -0.4, -0.2, -0.3]
        ])
        advantages = torch.tensor([
            [1.0, 0.5, -0.5, 1.5],
            [0.8, -0.3, 1.2, 0.6]
        ])
        eos_mask = torch.ones(batch_size, response_length)
        
        # 前两个token是off-policy，后两个是on-policy
        prefix_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]], dtype=torch.bool)
        
        result = compute_token_on_off_policy_loss(
            old_log_prob, log_prob, advantages, eos_mask,
            cliprange=0.2, clip_upper_bound=100.0, prefix_mask=prefix_mask,
            off_cliprange=None
        )

        # print(result['pg_loss'])
        # print(result['on_pg_loss'])
        # print(result['off_pg_loss'])

        expected_pg_loss = torch.tensor(-0.5959)
        expected_on_pg_loss = torch.tensor(-0.7736)
        expected_off_pg_loss = torch.tensor(-0.4182)
        torch.testing.assert_close(result['pg_loss'], expected_pg_loss, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(result['on_pg_loss'], expected_on_pg_loss, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(result['off_pg_loss'], expected_off_pg_loss, atol=1e-4, rtol=1e-4)
        # 验证混合损失
        self.assertTrue(torch.isfinite(result['pg_loss']))
        self.assertTrue(torch.isfinite(result['on_pg_loss']))
        self.assertTrue(torch.isfinite(result['off_pg_loss']))
        
        # 两种损失都应该有贡献
        self.assertGreater(result['on_pg_loss'].abs().item(), 0.0)
        self.assertGreater(result['off_pg_loss'].abs().item(), 0.0)
    
    def test_cliprange_parameter(self):
        """测试cliprange参数对PPO clip的影响"""
        batch_size, response_length = 2, 3
        
        old_log_prob = torch.tensor([
            [-1.0, -0.5, -0.8],
            [-0.6, -0.9, -0.7]
        ])
        log_prob = torch.tensor([
            [-0.2, -0.1, -0.3],  # 大幅改进
            [-0.1, -0.2, -0.15]
        ])
        advantages = torch.ones(batch_size, response_length)
        eos_mask = torch.ones(batch_size, response_length)
        prefix_mask = torch.zeros(batch_size, response_length, dtype=torch.bool)
        
        # 测试不同的cliprange
        result_small_clip = compute_token_on_off_policy_loss(
            old_log_prob, log_prob, advantages, eos_mask,
            cliprange=0.1, clip_upper_bound=100.0, prefix_mask=prefix_mask,
            off_cliprange=None
        )
        
        result_large_clip = compute_token_on_off_policy_loss(
            old_log_prob, log_prob, advantages, eos_mask,
            cliprange=0.5, clip_upper_bound=100.0, prefix_mask=prefix_mask,
            off_cliprange=None
        )

        # print(result_small_clip['on_pg_clipfrac'])
        # print(result_large_clip['on_pg_clipfrac'])

        expected_on_pg_clipfrac_small = torch.tensor(0.)
        expected_on_pg_clipfrac_large = torch.tensor(0.)
        torch.testing.assert_close(result_small_clip['on_pg_clipfrac'], expected_on_pg_clipfrac_small, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(result_large_clip['on_pg_clipfrac'], expected_on_pg_clipfrac_large, atol=1e-4, rtol=1e-4)
        
        # 小的cliprange应该导致更多的clipping
        self.assertGreaterEqual(result_small_clip['on_pg_clipfrac'].item(), 
                               result_large_clip['on_pg_clipfrac'].item())
    
    def test_off_policy_reshape_methods(self):
        """测试不同的off_policy_reshape方法"""
        batch_size, response_length = 2, 3
        
        old_log_prob = torch.zeros(batch_size, response_length)
        log_prob = torch.tensor([
            [-0.5, -0.3, -0.4],
            [-0.2, -0.6, -0.3]
        ])
        advantages = torch.ones(batch_size, response_length)
        eos_mask = torch.ones(batch_size, response_length)
        prefix_mask = torch.ones(batch_size, response_length, dtype=torch.bool)
        
        reshape_methods = ['no_reshape', 'logp', 'square_root', 'pow', 'p_div_p_0.1', 'p_div_p_0.5']

        results = []
        
        for method in reshape_methods:
            result = compute_token_on_off_policy_loss(
                old_log_prob, log_prob, advantages, eos_mask,
                cliprange=0.2, clip_upper_bound=100.0, prefix_mask=prefix_mask,
                off_cliprange=None, off_policy_reshape=method,
                off_policy_reshape_weight=1.0, off_policy_reshape_pow_exp=0.5
            )

            results.append(result)

        # print(results)

        expected_results = [{'pg_loss': torch.tensor(-0.6877), 
        'off_pg_loss': torch.tensor(-0.6877), 
        'on_pg_loss': torch.tensor(float('nan')), 
        'off_pg_clipfrac': torch.tensor(0.), 
        'on_pg_clipfrac': torch.tensor(0.), 
        'ppo_kl': torch.tensor(0.3833), 
        'off_policy_prob': torch.tensor(0.6877), 'on_policy_prob': torch.tensor(0.),
         'off_ratio_mean': torch.tensor(0.6877), 'off_ratio_max_clip_frac': torch.tensor(0.), 
         'off_ratio_min_clip_frac': torch.tensor(0.)}, 
         {'pg_loss': torch.tensor(0.3833), 
         'off_pg_loss': torch.tensor(0.3833),
          'on_pg_loss': torch.tensor(float('nan')), 
          'off_pg_clipfrac': torch.tensor(0.), 
          'on_pg_clipfrac': torch.tensor(0.), 'ppo_kl': torch.tensor(0.3833), 'off_policy_prob': torch.tensor(0.6877), 
          'on_policy_prob': torch.tensor(0.), 'off_ratio_mean': torch.tensor(-0.3833), 'off_ratio_max_clip_frac': torch.tensor(0.), 
          'off_ratio_min_clip_frac': torch.tensor(0.)}, 
          {'pg_loss': torch.tensor(-0.8274), 'off_pg_loss': torch.tensor(-0.8274), 'on_pg_loss': torch.tensor(float('nan')), 
          'off_pg_clipfrac': torch.tensor(0.), 'on_pg_clipfrac': torch.tensor(0.), 'ppo_kl': torch.tensor(0.3833), 
          'off_policy_prob': torch.tensor(0.6877), 'on_policy_prob': torch.tensor(0.), 'off_ratio_mean': torch.tensor(0.8274), 
          'off_ratio_max_clip_frac': torch.tensor(0.), 'off_ratio_min_clip_frac': torch.tensor(0.)}, 
          {'pg_loss': torch.tensor(-0.8274), 'off_pg_loss': torch.tensor(-0.8274), 'on_pg_loss': torch.tensor(float('nan')), 
          'off_pg_clipfrac': torch.tensor(0.), 'on_pg_clipfrac': torch.tensor(0.), 'ppo_kl': torch.tensor(0.3833), 
          'off_policy_prob': torch.tensor(0.6877), 'on_policy_prob': torch.tensor(0.), 'off_ratio_mean': torch.tensor(0.8274), 
          'off_ratio_max_clip_frac': torch.tensor(0.), 'off_ratio_min_clip_frac': torch.tensor(0.)}, {'pg_loss': torch.tensor(-0.8713), 
          'off_pg_loss': torch.tensor(-0.8713), 'on_pg_loss': torch.tensor(float('nan')), 'off_pg_clipfrac': torch.tensor(0.), 
          'on_pg_clipfrac': torch.tensor(0.), 'ppo_kl': torch.tensor(0.3833), 'off_policy_prob': torch.tensor(0.6877), 
          'on_policy_prob': torch.tensor(0.), 'off_ratio_mean': torch.tensor(0.8713), 'off_ratio_max_clip_frac': torch.tensor(0.), 
          'off_ratio_min_clip_frac': torch.tensor(0.)}, {'pg_loss': torch.tensor(-0.5765), 'off_pg_loss': torch.tensor(-0.5765), 
          'on_pg_loss': torch.tensor(float('nan')), 'off_pg_clipfrac': torch.tensor(0.), 'on_pg_clipfrac': torch.tensor(0.), 
          'ppo_kl': torch.tensor(0.3833), 'off_policy_prob': torch.tensor(0.6877), 'on_policy_prob': torch.tensor(0.), 
          'off_ratio_mean': torch.tensor(0.5765), 'off_ratio_max_clip_frac': torch.tensor(0.), 'off_ratio_min_clip_frac': torch.tensor(0.)}]
        
        for i in range(len(results)):
            torch.testing.assert_close(results[i]['pg_loss'], expected_results[i]['pg_loss'], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(results[i]['off_pg_loss'], expected_results[i]['off_pg_loss'], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(results[i]['on_pg_loss'], expected_results[i]['on_pg_loss'], equal_nan=True)
            torch.testing.assert_close(results[i]['off_pg_clipfrac'], expected_results[i]['off_pg_clipfrac'], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(results[i]['on_pg_clipfrac'], expected_results[i]['on_pg_clipfrac'], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(results[i]['ppo_kl'], expected_results[i]['ppo_kl'], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(results[i]['off_policy_prob'], expected_results[i]['off_policy_prob'], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(results[i]['on_policy_prob'], expected_results[i]['on_policy_prob'], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(results[i]['off_ratio_mean'], expected_results[i]['off_ratio_mean'], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(results[i]['off_ratio_max_clip_frac'], expected_results[i]['off_ratio_max_clip_frac'], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(results[i]['off_ratio_min_clip_frac'], expected_results[i]['off_ratio_min_clip_frac'], atol=1e-4, rtol=1e-4)
            # self.assertTrue(torch.isfinite(result['pg_loss']), f"Failed for method: {method}")
            # self.assertTrue(torch.isfinite(result['off_pg_loss']), f"Failed for method: {method}")
    
    def test_on_policy_reshape_methods(self):
        """测试不同的on_policy_reshape方法"""
        batch_size, response_length = 2, 3
        
        old_log_prob = torch.tensor([
            [-0.8, -0.6, -0.7],
            [-0.5, -0.9, -0.6]
        ])
        log_prob = torch.tensor([
            [-0.5, -0.3, -0.4],
            [-0.2, -0.6, -0.3]
        ])
        advantages = torch.ones(batch_size, response_length)
        eos_mask = torch.ones(batch_size, response_length)
        prefix_mask = torch.zeros(batch_size, response_length, dtype=torch.bool)
        
        reshape_methods = ['no_reshape', 'logp', 'p_logp', 'square_root', 'pow', 'p_div_p_0.1', 'p_div_p_0.5']

        results = []
        for method in reshape_methods:
            result = compute_token_on_off_policy_loss(
                old_log_prob, log_prob, advantages, eos_mask,
                cliprange=0.2, clip_upper_bound=100.0, prefix_mask=prefix_mask,
                off_cliprange=None, on_policy_reshape=method,
                on_policy_reshape_weight=1.0, on_policy_reshape_pow_exp=0.5
            )

            results.append(result)

        # print(results)
        expected_results = [{'pg_loss': torch.tensor(-1.3499), 'off_pg_loss': torch.tensor(0.), 
        'on_pg_loss': torch.tensor(-1.3499), 'off_pg_clipfrac': torch.tensor(0.), 'on_pg_clipfrac': torch.tensor(0.), 
        'ppo_kl': torch.tensor(-0.3000), 'off_policy_prob': torch.tensor(0.), 'on_policy_prob': torch.tensor(0.5094), 
        'off_ratio_mean': torch.tensor(0.), 'off_ratio_max_clip_frac': torch.tensor(0.), 'off_ratio_min_clip_frac': torch.tensor(0.)}, 
        {'pg_loss': torch.tensor(-0.3000), 'off_pg_loss': torch.tensor(0.), 'on_pg_loss': torch.tensor(-0.3000), 'off_pg_clipfrac': torch.tensor(0.), 
        'on_pg_clipfrac': torch.tensor(0.), 'ppo_kl': torch.tensor(-0.3000), 'off_policy_prob': torch.tensor(0.), 'on_policy_prob': torch.tensor(0.5094), 
        'off_ratio_mean': torch.tensor(0.), 'off_ratio_max_clip_frac': torch.tensor(0.), 'off_ratio_min_clip_frac': torch.tensor(0.)}, 
        {'pg_loss': torch.tensor(-1.6499), 'off_pg_loss': torch.tensor(0.), 'on_pg_loss': torch.tensor(-1.6499), 
        'off_pg_clipfrac': torch.tensor(0.), 'on_pg_clipfrac': torch.tensor(0.), 'ppo_kl': torch.tensor(-0.3000), 
        'off_policy_prob': torch.tensor(0.), 'on_policy_prob': torch.tensor(0.5094), 'off_ratio_mean': torch.tensor(0.), 
        'off_ratio_max_clip_frac': torch.tensor(0.), 'off_ratio_min_clip_frac': torch.tensor(0.)}, {'pg_loss': torch.tensor(-1.1618), 'off_pg_loss': torch.tensor(0.), 'on_pg_loss': torch.tensor(-1.1618), 'off_pg_clipfrac': torch.tensor(0.), 'on_pg_clipfrac': torch.tensor(0.), 'ppo_kl': torch.tensor(-0.3000), 'off_policy_prob': torch.tensor(0.), 'on_policy_prob': torch.tensor(0.5094), 'off_ratio_mean': torch.tensor(0.), 'off_ratio_max_clip_frac': torch.tensor(0.), 'off_ratio_min_clip_frac': torch.tensor(0.)}, {'pg_loss': torch.tensor(-1.1618), 'off_pg_loss': torch.tensor(0.), 'on_pg_loss': torch.tensor(-1.1618), 'off_pg_clipfrac': torch.tensor(0.), 'on_pg_clipfrac': torch.tensor(0.), 'ppo_kl': torch.tensor(-0.3000), 'off_policy_prob': torch.tensor(0.), 'on_policy_prob': torch.tensor(0.5094), 'off_ratio_mean': torch.tensor(0.), 'off_ratio_max_clip_frac': torch.tensor(0.), 'off_ratio_min_clip_frac': torch.tensor(0.)}, {'pg_loss': torch.tensor(-1.0450), 'off_pg_loss': torch.tensor(0.), 'on_pg_loss': torch.tensor(-1.0450), 'off_pg_clipfrac': torch.tensor(0.), 'on_pg_clipfrac': torch.tensor(0.), 'ppo_kl': torch.tensor(-0.3000), 'off_policy_prob': torch.tensor(0.), 'on_policy_prob': torch.tensor(0.5094), 'off_ratio_mean': torch.tensor(0.), 'off_ratio_max_clip_frac': torch.tensor(0.), 'off_ratio_min_clip_frac': torch.tensor(0.)}, {'pg_loss': torch.tensor(-1.1482), 'off_pg_loss': torch.tensor(0.), 'on_pg_loss': torch.tensor(-1.1482), 'off_pg_clipfrac': torch.tensor(0.), 'on_pg_clipfrac': torch.tensor(0.), 'ppo_kl': torch.tensor(-0.3000), 'off_policy_prob': torch.tensor(0.), 'on_policy_prob': torch.tensor(0.5094), 'off_ratio_mean': torch.tensor(0.), 'off_ratio_max_clip_frac': torch.tensor(0.), 'off_ratio_min_clip_frac': torch.tensor(0.)}]
        for i in range(len(results)):
            torch.testing.assert_close(results[i]['pg_loss'], expected_results[i]['pg_loss'], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(results[i]['off_pg_loss'], expected_results[i]['off_pg_loss'], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(results[i]['on_pg_loss'], expected_results[i]['on_pg_loss'],atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(results[i]['off_pg_clipfrac'], expected_results[i]['off_pg_clipfrac'], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(results[i]['on_pg_clipfrac'], expected_results[i]['on_pg_clipfrac'], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(results[i]['ppo_kl'], expected_results[i]['ppo_kl'], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(results[i]['off_policy_prob'], expected_results[i]['off_policy_prob'], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(results[i]['on_policy_prob'], expected_results[i]['on_policy_prob'], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(results[i]['off_ratio_mean'], expected_results[i]['off_ratio_mean'], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(results[i]['off_ratio_max_clip_frac'], expected_results[i]['off_ratio_max_clip_frac'], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(results[i]['off_ratio_min_clip_frac'], expected_results[i]['off_ratio_min_clip_frac'], atol=1e-4, rtol=1e-4)
    
    def test_off_max_clip_parameter(self):
        """测试off_max_clip参数：限制off-policy ratio上限"""
        batch_size, response_length = 2, 3
        
        old_log_prob = torch.zeros(batch_size, response_length)
        log_prob = torch.tensor([
            [-0.1, -0.05, -0.15],  # 高概率
            [-0.2, -0.1, -0.25]
        ])
        advantages = torch.ones(batch_size, response_length)
        eos_mask = torch.ones(batch_size, response_length)
        prefix_mask = torch.ones(batch_size, response_length, dtype=torch.bool)
        
        result = compute_token_on_off_policy_loss(
            old_log_prob, log_prob, advantages, eos_mask,
            cliprange=0.2, clip_upper_bound=100.0, prefix_mask=prefix_mask,
            off_cliprange=None, off_max_clip=0.5
        )
        
        # print(result['off_ratio_max_clip_frac'])
        expected_off_ratio_max_clip_frac = torch.tensor(1.)
        torch.testing.assert_close(result['off_ratio_max_clip_frac'], expected_off_ratio_max_clip_frac, atol=1e-4, rtol=1e-4)
    
    def test_off_min_clip_parameter(self):
        """测试off_min_clip参数：限制off-policy ratio下限"""
        batch_size, response_length = 2, 3
        
        old_log_prob = torch.zeros(batch_size, response_length)
        log_prob = torch.tensor([
            [-5.0, -4.0, -6.0],  # 低概率
            [-3.0, -5.0, -4.0]
        ])
        advantages = torch.ones(batch_size, response_length)
        eos_mask = torch.ones(batch_size, response_length)
        prefix_mask = torch.ones(batch_size, response_length, dtype=torch.bool)
        
        result = compute_token_on_off_policy_loss(
            old_log_prob, log_prob, advantages, eos_mask,
            cliprange=0.2, clip_upper_bound=100.0, prefix_mask=prefix_mask,
            off_cliprange=None, off_min_clip=0.01
        )
        
        # print(result['off_ratio_min_clip_frac'])
        # 验证有clip发生
        expected_off_ratio_min_clip_frac = torch.tensor(0.5000)
        torch.testing.assert_close(result['off_ratio_min_clip_frac'], expected_off_ratio_min_clip_frac, atol=1e-4, rtol=1e-4)
    
    def test_eos_mask_application(self):
        """测试eos_mask的应用"""
        batch_size, response_length = 2, 5
        
        old_log_prob = torch.randn(batch_size, response_length)
        log_prob = torch.randn(batch_size, response_length)
        advantages = torch.ones(batch_size, response_length)
        
        # 部分mask
        eos_mask = torch.tensor([
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0]
        ])
        prefix_mask = torch.zeros(batch_size, response_length, dtype=torch.bool)
        
        result = compute_token_on_off_policy_loss(
            old_log_prob, log_prob, advantages, eos_mask,
            cliprange=0.2, clip_upper_bound=100.0, prefix_mask=prefix_mask,
            off_cliprange=None
        )

        # print(result['pg_loss'])
        expected_pg_loss = torch.tensor(-1.9608)
        torch.testing.assert_close(result['pg_loss'], expected_pg_loss, atol=1e-4, rtol=1e-4)
    
    def test_ppo_kl_computation(self):
        """测试PPO KL散度的计算"""
        batch_size, response_length = 2, 4
        
        old_log_prob = torch.tensor([
            [-0.5, -0.3, -0.4, -0.6],
            [-0.2, -0.5, -0.3, -0.4]
        ])
        log_prob = torch.tensor([
            [-0.3, -0.2, -0.3, -0.5],
            [-0.1, -0.4, -0.2, -0.3]
        ])
        advantages = torch.ones(batch_size, response_length)
        eos_mask = torch.ones(batch_size, response_length)
        prefix_mask = torch.zeros(batch_size, response_length, dtype=torch.bool)
        
        result = compute_token_on_off_policy_loss(
            old_log_prob, log_prob, advantages, eos_mask,
            cliprange=0.2, clip_upper_bound=100.0, prefix_mask=prefix_mask,
            off_cliprange=None
        )

        # 验证KL散度
        # print(result['ppo_kl'])
        expected_ppo_kl = torch.tensor(-0.1125)
        torch.testing.assert_close(result['ppo_kl'], expected_ppo_kl, atol=1e-4, rtol=1e-4)
        # 注意：PPO使用的是近似KL，可能为负值（当新策略概率低于旧策略时）
        # 只需验证是有限值即可
    
    def test_target_probs_parameter(self):
        """测试target_probs参数：显式行为策略"""
        batch_size, response_length = 2, 3
        
        old_log_prob = torch.zeros(batch_size, response_length)
        log_prob = torch.tensor([
            [-0.5, -0.3, -0.4],
            [-0.2, -0.6, -0.3]
        ])
        target_probs = torch.tensor([
            [0.7, 0.8, 0.6],
            [0.9, 0.5, 0.7]
        ])
        advantages = torch.ones(batch_size, response_length)
        eos_mask = torch.ones(batch_size, response_length)
        prefix_mask = torch.ones(batch_size, response_length, dtype=torch.bool)
        
        result = compute_token_on_off_policy_loss(
            old_log_prob, log_prob, advantages, eos_mask,
            cliprange=0.2, clip_upper_bound=100.0, prefix_mask=prefix_mask,
            off_cliprange=None, target_probs=target_probs
        )

        # print(result['pg_loss'])
        # print(result['off_pg_loss'])
        
        expected_pg_loss = torch.tensor(-0.9959)
        expected_off_pg_loss = torch.tensor(-0.9959)
        torch.testing.assert_close(result['pg_loss'], expected_pg_loss, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(result['off_pg_loss'], expected_off_pg_loss, atol=1e-4, rtol=1e-4)
    
    def test_loss_remove_clip_parameter(self):
        """测试loss_remove_clip参数：移除clip操作"""
        batch_size, response_length = 2, 3
        
        # 测试下界clip：使用策略退化（ratio < 1-cliprange）配合负优势
        # 这样 clipped_loss > unclipped_loss，会触发clipfrac
        old_log_prob = torch.tensor([
            [-0.1, -0.05, -0.15],  # 高概率
            [-0.05, -0.1, -0.08]
        ])
        log_prob = torch.tensor([
            [-3.0, -2.8, -2.9],  # 低概率（策略退化，ratio << 1）
            [-2.7, -3.1, -2.8]
        ])
        # 使用负优势，这样当ratio < 1-cliprange时：
        # unclipped: -(-1) * 0.05 = 0.05
        # clipped: -(-1) * 0.8 = 0.8
        # clipped > unclipped，触发clipfrac
        advantages = -torch.ones(batch_size, response_length)
        eos_mask = torch.ones(batch_size, response_length)
        prefix_mask = torch.zeros(batch_size, response_length, dtype=torch.bool)
        
        # 不移除clip
        result_with_clip = compute_token_on_off_policy_loss(
            old_log_prob, log_prob, advantages, eos_mask,
            cliprange=0.2, clip_upper_bound=100.0, prefix_mask=prefix_mask,
            off_cliprange=None, loss_remove_clip=False
        )
        
        # 移除clip
        result_no_clip = compute_token_on_off_policy_loss(
            old_log_prob, log_prob, advantages, eos_mask,
            cliprange=0.2, clip_upper_bound=100.0, prefix_mask=prefix_mask,
            off_cliprange=None, loss_remove_clip=True
        )

        # print(result_with_clip['on_pg_clipfrac'])
        # print(result_no_clip['on_pg_clipfrac'])

        expected_on_pg_clipfrac_with_clip = torch.tensor(1.)
        expected_on_pg_clipfrac_no_clip = torch.tensor(0.)
        torch.testing.assert_close(result_with_clip['on_pg_clipfrac'], expected_on_pg_clipfrac_with_clip, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(result_no_clip['on_pg_clipfrac'], expected_on_pg_clipfrac_no_clip, atol=1e-4, rtol=1e-4)
        
        # 验证两种模式的clipfrac
        # 有clip时应该有clipfrac > 0（因为ratio远小于1-cliprange，会被clip）
        # self.assertGreater(result_with_clip['on_pg_clipfrac'].item(), 0.0)
        # # 无clip时clipfrac应该为0
        # self.assertEqual(result_no_clip['on_pg_clipfrac'].item(), 0.0)
        
        # 验证损失值不同
        # 有clip时损失应该更大（因为clip限制了ratio的下降）
        self.assertGreater(result_with_clip['pg_loss'].item(), result_no_clip['pg_loss'].item())


class TestComputeTokenOnOffPolicyLossIntegration(unittest.TestCase):
    """集成测试：测试compute_token_on_off_policy_loss在实际场景中的应用"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_realistic_training_scenario(self):
        """测试真实训练场景：混合on-policy和off-policy样本"""
        batch_size, response_length = 8, 10
        
        # 模拟真实的log概率
        old_log_prob = torch.randn(batch_size, response_length) * 0.5 - 1.0
        log_prob = old_log_prob + torch.randn(batch_size, response_length) * 0.2
        
        # 模拟优势估计
        advantages = torch.randn(batch_size, response_length)
        
        # 全部有效
        eos_mask = torch.ones(batch_size, response_length)
        
        # 前半部分是off-policy，后半部分是on-policy
        prefix_mask = torch.zeros(batch_size, response_length, dtype=torch.bool)
        prefix_mask[:, :5] = 1.0
        
        result = compute_token_on_off_policy_loss(
            old_log_prob, log_prob, advantages, eos_mask,
            cliprange=0.2, clip_upper_bound=100.0, prefix_mask=prefix_mask,
            off_cliprange=None
        )
        
        # print(result['pg_loss'])
        # print(result['on_pg_loss'])
        # print(result['off_pg_loss'])
        # print(result['ppo_kl'])
        # print(result['on_policy_prob'])
        # print(result['off_policy_prob'])

        expected_pg_loss = torch.tensor(0.0734)
        expected_on_pg_loss = torch.tensor(0.1314)
        expected_off_pg_loss = torch.tensor(0.0153)
        expected_ppo_kl = torch.tensor(-0.0268)
        expected_on_policy_prob = torch.tensor(0.4355)
        expected_off_policy_prob = torch.tensor(0.4426)
        torch.testing.assert_close(result['pg_loss'], expected_pg_loss, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(result['on_pg_loss'], expected_on_pg_loss, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(result['off_pg_loss'], expected_off_pg_loss, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(result['ppo_kl'], expected_ppo_kl, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(result['on_policy_prob'], expected_on_policy_prob, atol=1e-4, rtol=1e-4)
        # 验证所有输出
        # self.assertTrue(torch.isfinite(result['pg_loss']))
        # self.assertTrue(torch.isfinite(result['on_pg_loss']))
        # self.assertTrue(torch.isfinite(result['off_pg_loss']))
        # self.assertTrue(torch.isfinite(result['ppo_kl']))
        
        # # 验证概率统计
        # self.assertGreater(result['on_policy_prob'].item(), 0.0)
        # self.assertGreater(result['off_policy_prob'].item(), 0.0)
    
    def test_policy_shaping_comparison(self):
        """测试不同策略塑形方法的比较"""
        batch_size, response_length = 4, 6
        
        old_log_prob = torch.tensor([[-0.0365, -0.2564, -0.5496, -2.0528, -0.6608, -1.6173],
        [-1.0215, -1.8023, -0.8221, -1.3433, -1.2467, -0.8793],
        [-1.5555, -0.9542, -2.1585, -1.1084, -1.1549, -1.1979],
        [-0.5983, -1.3108, -1.2960, -1.0315, -1.4143, -0.8346]])
        log_prob = torch.tensor([[-0.9825, -0.8394, -0.2132, -1.4227, -0.3438, -0.6564],
        [-1.5446, -1.1776, -0.2774, -0.5718,  0.1090, -0.7384],
        [-0.8267, -1.0987, -1.5273, -0.3610, -1.0861, -0.7381],
        [-0.9717, -0.7869, -0.7125, -1.3209, -2.1032, -1.3754]])
        advantages = torch.tensor([[ 0.0109, -0.3387, -1.3407, -0.5854,  0.5362,  0.5246],
        [ 1.1412,  0.0516, -0.6788,  0.5743,  0.1877, -0.3576],
        [-0.3165,  0.5886, -0.8905,  0.4098, -0.9864,  0.1233],
        [ 0.3499,  0.6173, -0.1693,  0.2332,  4.0356,  1.2795]])
        eos_mask = torch.ones(batch_size, response_length)

        # print(old_log_prob)
        # print(log_prob)
        # print(advantages)

        prefix_mask = torch.ones(batch_size, response_length, dtype=torch.bool)
        
        # 测试不同的off_policy_reshape方法
        reshape_methods = ['no_reshape', 'logp', 'square_root', 'pow']
        results = {}
        
        for method in reshape_methods:
            result = compute_token_on_off_policy_loss(
                old_log_prob, log_prob, advantages, eos_mask,
                cliprange=0.2, clip_upper_bound=100.0, prefix_mask=prefix_mask,
                off_cliprange=None, off_policy_reshape=method
            )
            results[method] = result['off_pg_loss'].item()
        
        # print(results)

        expected_results = {'no_reshape': -0.019922085106372833, 'logp': 0.4278620481491089, 'square_root': -0.07187823206186295, 'pow': -0.07187823206186295}
        for method in reshape_methods:
            torch.testing.assert_close(results[method], expected_results[method], atol=1e-4, rtol=1e-4)

        # 验证不同方法产生不同结果
        unique_losses = len(set(results.values()))
        self.assertGreater(unique_losses, 1)


if __name__ == "__main__":
    unittest.main()
