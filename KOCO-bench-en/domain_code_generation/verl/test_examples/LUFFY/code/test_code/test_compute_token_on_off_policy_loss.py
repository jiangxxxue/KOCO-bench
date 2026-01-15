import unittest
import torch
import sys
import os

# Add the verl directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'luffy', 'verl'))

# Import the actual function we want to test
from verl.mix_src.mix_core_alg import compute_token_on_off_policy_loss


class TestComputeTokenOnOffPolicyLoss(unittest.TestCase):
    """Test compute_token_on_off_policy_loss function - Mixed policy loss computation described in LUFFY document Section 2"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_basic_on_policy_loss(self):
        """Test basic on-policy PPO loss computation"""
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
        prefix_mask = torch.zeros(batch_size, response_length, dtype=torch.bool)  # All on-policy
        
        # Call the actual function
        result = compute_token_on_off_policy_loss(
            old_log_prob, log_prob, advantages, eos_mask,
            cliprange=0.2, clip_upper_bound=100.0, prefix_mask=prefix_mask,
            off_cliprange=None
        )
        
        # Verify return dictionary contains all required keys
        expected_keys = ['pg_loss', 'off_pg_loss', 'on_pg_loss', 'off_pg_clipfrac', 
                        'on_pg_clipfrac', 'ppo_kl', 'off_policy_prob', 'on_policy_prob',
                        'off_ratio_mean', 'off_ratio_max_clip_frac', 'off_ratio_min_clip_frac']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Verify loss value is a scalar
        self.assertTrue(torch.is_tensor(result['pg_loss']))
        self.assertEqual(result['pg_loss'].dim(), 0)
        self.assertTrue(torch.isfinite(result['pg_loss']))
        
        # When all on-policy, off_pg_loss should be 0 or very small
        self.assertTrue(result['off_pg_loss'].item() == 0.0 or result['off_pg_loss'].item() < 1e-6)
    
    def test_basic_off_policy_loss(self):
        """Test basic off-policy loss computation"""
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
        prefix_mask = torch.ones(batch_size, response_length, dtype=torch.bool)  # All off-policy
        
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
        # Verify output
        # self.assertTrue(torch.isfinite(result['pg_loss']))
        # self.assertTrue(torch.isfinite(result['off_pg_loss']))
        
        # # When all off-policy, on_pg_loss may not be 0 (because it's still calculated), but should not affect total loss
        # # Main verification is that off_pg_loss has a value
        # self.assertGreater(result['off_pg_loss'].abs().item(), 0.0)
    
    def test_mixed_on_off_policy_loss(self):
        """Test mixed on-policy and off-policy loss"""
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
        
        # First two tokens are off-policy, last two are on-policy
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
        # Verify mixed loss
        self.assertTrue(torch.isfinite(result['pg_loss']))
        self.assertTrue(torch.isfinite(result['on_pg_loss']))
        self.assertTrue(torch.isfinite(result['off_pg_loss']))
        
        # Both losses should contribute
        self.assertGreater(result['on_pg_loss'].abs().item(), 0.0)
        self.assertGreater(result['off_pg_loss'].abs().item(), 0.0)
    
    def test_cliprange_parameter(self):
        """Test cliprange parameter's effect on PPO clipping"""
        batch_size, response_length = 2, 3
        
        old_log_prob = torch.tensor([
            [-1.0, -0.5, -0.8],
            [-0.6, -0.9, -0.7]
        ])
        log_prob = torch.tensor([
            [-0.2, -0.1, -0.3],  # Significant improvement
            [-0.1, -0.2, -0.15]
        ])
        advantages = torch.ones(batch_size, response_length)
        eos_mask = torch.ones(batch_size, response_length)
        prefix_mask = torch.zeros(batch_size, response_length, dtype=torch.bool)
        
        # Test different cliprange values
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
        
        # Smaller cliprange should lead to more clipping
        self.assertGreaterEqual(result_small_clip['on_pg_clipfrac'].item(), 
                               result_large_clip['on_pg_clipfrac'].item())
    
    def test_off_policy_reshape_methods(self):
        """Test different off_policy_reshape methods"""
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
        """Test different on_policy_reshape methods"""
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
        """Test off_max_clip parameter: limit off-policy ratio upper bound"""
        batch_size, response_length = 2, 3
        
        old_log_prob = torch.zeros(batch_size, response_length)
        log_prob = torch.tensor([
            [-0.1, -0.05, -0.15],  # High probability
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
        """Test off_min_clip parameter: limit off-policy ratio lower bound"""
        batch_size, response_length = 2, 3
        
        old_log_prob = torch.zeros(batch_size, response_length)
        log_prob = torch.tensor([
            [-5.0, -4.0, -6.0],  # Low probability
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
        # Verify clipping occurred
        expected_off_ratio_min_clip_frac = torch.tensor(0.5000)
        torch.testing.assert_close(result['off_ratio_min_clip_frac'], expected_off_ratio_min_clip_frac, atol=1e-4, rtol=1e-4)
    
    def test_eos_mask_application(self):
        """Test eos_mask application"""
        batch_size, response_length = 2, 5
        
        old_log_prob = torch.randn(batch_size, response_length)
        log_prob = torch.randn(batch_size, response_length)
        advantages = torch.ones(batch_size, response_length)
        
        # Partial mask
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
        """Test PPO KL divergence computation"""
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

        # Verify KL divergence
        # print(result['ppo_kl'])
        expected_ppo_kl = torch.tensor(-0.1125)
        torch.testing.assert_close(result['ppo_kl'], expected_ppo_kl, atol=1e-4, rtol=1e-4)
        # Note: PPO uses approximate KL, which can be negative (when new policy probability is lower than old policy)
        # Just need to verify it's a finite value
    
    def test_target_probs_parameter(self):
        """Test target_probs parameter: explicit behavior policy"""
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
        """Test loss_remove_clip parameter: remove clip operation"""
        batch_size, response_length = 2, 3
        
        # Test lower bound clip: use policy degradation (ratio < 1-cliprange) with negative advantages
        # This way clipped_loss > unclipped_loss, triggering clipfrac
        old_log_prob = torch.tensor([
            [-0.1, -0.05, -0.15],  # High probability
            [-0.05, -0.1, -0.08]
        ])
        log_prob = torch.tensor([
            [-3.0, -2.8, -2.9],  # Low probability (policy degradation, ratio << 1)
            [-2.7, -3.1, -2.8]
        ])
        # Use negative advantages, so when ratio < 1-cliprange:
        # unclipped: -(-1) * 0.05 = 0.05
        # clipped: -(-1) * 0.8 = 0.8
        # clipped > unclipped, triggering clipfrac
        advantages = -torch.ones(batch_size, response_length)
        eos_mask = torch.ones(batch_size, response_length)
        prefix_mask = torch.zeros(batch_size, response_length, dtype=torch.bool)
        
        # Do not remove clip
        result_with_clip = compute_token_on_off_policy_loss(
            old_log_prob, log_prob, advantages, eos_mask,
            cliprange=0.2, clip_upper_bound=100.0, prefix_mask=prefix_mask,
            off_cliprange=None, loss_remove_clip=False
        )
        
        # Remove clip
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
        
        # Verify clipfrac for both modes
        # With clip, clipfrac should be > 0 (because ratio is much less than 1-cliprange, will be clipped)
        # self.assertGreater(result_with_clip['on_pg_clipfrac'].item(), 0.0)
        # # Without clip, clipfrac should be 0
        # self.assertEqual(result_no_clip['on_pg_clipfrac'].item(), 0.0)
        
        # Verify loss values are different
        # With clip, loss should be larger (because clip limits the ratio decrease)
        self.assertGreater(result_with_clip['pg_loss'].item(), result_no_clip['pg_loss'].item())


class TestComputeTokenOnOffPolicyLossIntegration(unittest.TestCase):
    """Integration tests: test compute_token_on_off_policy_loss in real-world scenarios"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_realistic_training_scenario(self):
        """Test realistic training scenario: mixed on-policy and off-policy samples"""
        batch_size, response_length = 8, 10
        
        # Simulate realistic log probabilities
        old_log_prob = torch.randn(batch_size, response_length) * 0.5 - 1.0
        log_prob = old_log_prob + torch.randn(batch_size, response_length) * 0.2
        
        # Simulate advantage estimation
        advantages = torch.randn(batch_size, response_length)
        
        # All valid
        eos_mask = torch.ones(batch_size, response_length)
        
        # First half is off-policy, second half is on-policy
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
        # Verify all outputs
        # self.assertTrue(torch.isfinite(result['pg_loss']))
        # self.assertTrue(torch.isfinite(result['on_pg_loss']))
        # self.assertTrue(torch.isfinite(result['off_pg_loss']))
        # self.assertTrue(torch.isfinite(result['ppo_kl']))
        
        # # Verify probability statistics
        # self.assertGreater(result['on_policy_prob'].item(), 0.0)
        # self.assertGreater(result['off_policy_prob'].item(), 0.0)
    
    def test_policy_shaping_comparison(self):
        """Test comparison of different policy shaping methods"""
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
        
        # Test different off_policy_reshape methods
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

        # Verify different methods produce different results
        unique_losses = len(set(results.values()))
        self.assertGreater(unique_losses, 1)


if __name__ == "__main__":
    unittest.main()
