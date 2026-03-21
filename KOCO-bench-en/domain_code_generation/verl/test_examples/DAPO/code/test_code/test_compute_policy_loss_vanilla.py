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
    """Test compute_policy_loss_vanilla function - Decoupled clipping policy gradient loss described in DAPO document Section 2"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.response_length = 4
    
    def _create_mock_config(self, clip_ratio=0.2, clip_ratio_low=None, clip_ratio_high=None, clip_ratio_c=3.0):
        """Create mock config object"""
        config = MagicMock()
        config.clip_ratio = clip_ratio
        config.clip_ratio_low = clip_ratio_low
        config.clip_ratio_high = clip_ratio_high
        config.tis_imp_ratio_cap = 0  # Disable truncated importance sampling by default
        config.get = MagicMock(return_value=clip_ratio_c)
        return config
    
    def test_compute_policy_loss_vanilla_basic(self):
        """Test basic functionality"""
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
        
        # Call the actual function
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

        # # Verify outputs
        # self.assertTrue(torch.is_tensor(pg_loss))
        # self.assertTrue(torch.is_tensor(pg_clipfrac))
        # self.assertTrue(torch.is_tensor(ppo_kl))
        # self.assertTrue(torch.is_tensor(pg_clipfrac_lower))
        
        # # Verify all are scalars
        # self.assertEqual(pg_loss.shape, torch.Size([]))
        # self.assertEqual(pg_clipfrac.shape, torch.Size([]))
        # self.assertEqual(ppo_kl.shape, torch.Size([]))
        # self.assertEqual(pg_clipfrac_lower.shape, torch.Size([]))
        
        # # Verify values are finite
        # self.assertTrue(torch.isfinite(pg_loss))
        # self.assertTrue(torch.isfinite(pg_clipfrac))
        # self.assertTrue(torch.isfinite(ppo_kl))
        # self.assertTrue(torch.isfinite(pg_clipfrac_lower))
        
        # # Verify clipfrac is in [0, 1] range
        # self.assertGreaterEqual(pg_clipfrac.item(), 0.0)
        # self.assertLessEqual(pg_clipfrac.item(), 1.0)
        # self.assertGreaterEqual(pg_clipfrac_lower.item(), 0.0)
        # self.assertLessEqual(pg_clipfrac_lower.item(), 1.0)
    
    def test_compute_policy_loss_vanilla_decoupled_clip(self):
        """Test decoupled clipping (Clip-Higher) mechanism"""
        # Create a scenario with positive advantages
        old_log_prob = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
        log_prob = torch.tensor([[-0.5, -0.5, -0.5, -0.5]])  # Significant improvement
        advantages = torch.tensor([[1.0, 1.0, 1.0, 1.0]])  # All positive advantages
        response_mask = torch.ones(1, 4)
        
        # Standard PPO configuration (symmetric clipping)
        config_symmetric = self._create_mock_config(clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.2)
        pg_loss_symmetric, _, _, _ = compute_policy_loss_vanilla(
            old_log_prob, log_prob, advantages, response_mask,
            loss_agg_mode="token-mean", config=config_symmetric
        )
        
        # DAPO configuration (asymmetric clipping, clip_ratio_high > clip_ratio_low)
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
        
        # For positive advantages, DAPO should allow more aggressive updates (more negative loss, i.e., larger update magnitude)
        # Since it's in the form of -A*ratio, with positive advantages the loss is negative, more aggressive means more negative
        self.assertLess(pg_loss_dapo.item(), pg_loss_symmetric.item())
    
    def test_compute_policy_loss_vanilla_negative_advantage_conservatism(self):
        """Test conservatism with negative advantages"""
        # Create a scenario with negative advantages
        old_log_prob = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
        log_prob = torch.tensor([[-1.5, -1.5, -1.5, -1.5]])  # Significant decrease (bad behavior reduced)
        advantages = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])  # All negative advantages
        response_mask = torch.ones(1, 4)
        
        # DAPO configuration
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
        # Verify that clip_ratio_low is used for negative advantages
        self.assertTrue(torch.isfinite(pg_loss))
        # Dual clipping mechanism may be triggered with negative advantages
        self.assertGreaterEqual(pg_clipfrac_lower.item(), 0.0)
    
    def test_compute_policy_loss_vanilla_ratio_clipping(self):
        """Test importance sampling ratio clipping"""
        old_log_prob = torch.tensor([[-1.0, -1.0]])
        response_mask = torch.ones(1, 2)
        
        # Test case where ratio > 1 + clip_ratio_high
        log_prob_high = torch.tensor([[-0.2, -0.2]])  # Significant improvement
        advantages_pos = torch.tensor([[1.0, 1.0]])
        
        config = self._create_mock_config(clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.2)
        _, pg_clipfrac_high, _, _ = compute_policy_loss_vanilla(
            old_log_prob, log_prob_high, advantages_pos, response_mask,
            loss_agg_mode="token-mean", config=config
        )
        
        # Clipping should occur
        self.assertGreater(pg_clipfrac_high.item(), 0.0)
        
        # Test case where ratio < 1 - clip_ratio_low
        log_prob_low = torch.tensor([[-1.8, -1.8]])  # Significant decrease
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
        # Clipping should occur
        self.assertGreater(pg_clipfrac_low.item(), 0.0)
    
    def test_compute_policy_loss_vanilla_kl_calculation(self):
        """Test KL divergence calculation"""
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
        # Manually calculate KL (approximation)
        # KL ≈ -(log_prob - old_log_prob)
        # manual_kl = -(log_prob - old_log_prob).mean()
        # torch.testing.assert_close(ppo_kl, manual_kl, atol=1e-4, rtol=1e-4)
    
    def test_compute_policy_loss_vanilla_extreme_log_prob_stability(self):
        """Test numerical stability with extreme log_prob values"""
        # Extreme log_prob values
        old_log_prob = torch.tensor([[-100.0, -1.0]])
        log_prob = torch.tensor([[50.0, -1.1]])  # Large difference
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
        # Should ensure numerical stability through clamp
        self.assertTrue(torch.isfinite(pg_loss))
        self.assertTrue(torch.isfinite(ppo_kl))
    
    def test_compute_policy_loss_vanilla_mixed_advantages(self):
        """Test mixed positive and negative advantages"""
        old_log_prob = torch.tensor([
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0]
        ])
        log_prob = torch.tensor([
            [-0.8, -0.8, -1.2, -1.2],
            [-0.8, -0.8, -1.2, -1.2]
        ])
        advantages = torch.tensor([
            [1.0, -1.0, 0.5, -0.5],  # Mixed
            [2.0, -2.0, 0.0, 0.0]    # Mixed
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
        # Verify all outputs are finite
        self.assertTrue(torch.isfinite(pg_loss))
        self.assertTrue(torch.isfinite(pg_clipfrac))
        self.assertTrue(torch.isfinite(ppo_kl))
        self.assertTrue(torch.isfinite(pg_clipfrac_lower))
    
    def test_compute_policy_loss_vanilla_different_agg_modes(self):
        """Test different loss aggregation modes"""
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
            [1.0, 1.0, 1.0, 0.0],  # Last token is masked
            [1.0, 1.0, 1.0, 1.0]
        ])
        
        config = self._create_mock_config()
        
        losses = []
        # Test three aggregation modes
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
        """Test gradient backpropagation"""
        old_log_prob = torch.tensor([[-1.0, -1.0, -1.0]], requires_grad=False)
        log_prob = torch.tensor([[-0.9, -0.9, -0.9]], requires_grad=True)
        advantages = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=False)
        response_mask = torch.ones(1, 3)
        
        config = self._create_mock_config()
        
        pg_loss, _, _, _ = compute_policy_loss_vanilla(
            old_log_prob, log_prob, advantages, response_mask,
            loss_agg_mode="token-mean", config=config
        )
        
        # Backpropagation
        pg_loss.backward()
        
        # print(log_prob.grad)

        expected_log_prob_grad = torch.tensor([[-0.3684, -0.3684, -0.3684]])
        torch.testing.assert_close(log_prob.grad, expected_log_prob_grad, atol=1e-4, rtol=1e-4)
        # Verify gradient exists
        self.assertIsNotNone(log_prob.grad)
        self.assertTrue(torch.isfinite(log_prob.grad).all())
    
    def test_compute_policy_loss_vanilla_zero_advantages(self):
        """Test all-zero advantages"""
        old_log_prob = torch.tensor([[-1.0, -1.0, -1.0]])
        log_prob = torch.tensor([[-0.8, -0.8, -0.8]])
        advantages = torch.zeros(1, 3)
        response_mask = torch.ones(1, 3)
        
        config = self._create_mock_config()
        
        pg_loss, _, _, _ = compute_policy_loss_vanilla(
            old_log_prob, log_prob, advantages, response_mask,
            loss_agg_mode="token-mean", config=config
        )
        
        # Zero advantages should result in zero loss (or close to zero)
        torch.testing.assert_close(pg_loss, torch.tensor(0.0), atol=1e-4, rtol=1e-4)
    
    def test_compute_policy_loss_vanilla_dual_clip(self):
        """Test dual clipping mechanism (clip_ratio_c)"""
        old_log_prob = torch.tensor([[-1.0, -1.0]])
        log_prob = torch.tensor([[-3.0, -3.0]])  # Significant decrease (extreme negative advantage scenario)
        advantages = torch.tensor([[-5.0, -5.0]])  # Extreme negative advantages
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
        # Dual clipping should limit the impact of extreme negative advantages
        self.assertTrue(torch.isfinite(pg_loss))
        # May trigger lower bound clipping
        self.assertGreaterEqual(pg_clipfrac_lower.item(), 0.0)
    
    def test_compute_policy_loss_vanilla_backward_compatibility(self):
        """Test backward compatibility (should degrade to standard PPO when clip_ratio_low/high are not set)"""
        old_log_prob = torch.tensor([[-1.0, -1.0, -1.0]])
        log_prob = torch.tensor([[-0.8, -0.8, -0.8]])
        advantages = torch.tensor([[1.0, 1.0, 1.0]])
        response_mask = torch.ones(1, 3)
        
        # Only set clip_ratio, don't set low/high
        config = self._create_mock_config(clip_ratio=0.2, clip_ratio_low=None, clip_ratio_high=None)
        
        pg_loss, _, _, _ = compute_policy_loss_vanilla(
            old_log_prob, log_prob, advantages, response_mask,
            loss_agg_mode="token-mean", config=config
        )
        
        # print(pg_loss)
        expected_pg_loss = torch.tensor(-1.2000)
        torch.testing.assert_close(pg_loss, expected_pg_loss, atol=1e-4, rtol=1e-4)
        # Should use clip_ratio as the clipping parameter for both directions
        self.assertTrue(torch.isfinite(pg_loss))


if __name__ == '__main__':
    unittest.main()

