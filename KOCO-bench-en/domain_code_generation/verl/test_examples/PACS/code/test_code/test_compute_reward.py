import unittest
import torch
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the actual function we want to test
from pacs.pacs_core_algos import compute_reward


class TestComputeReward(unittest.TestCase):
    """Test compute_reward function - policy improvement reward calculation described in PACS document Section 1"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_compute_reward_method_1_basic(self):
        """Test basic functionality of method 1: compute sum of log probability differences"""
        batch_size, response_length = 2, 4
        beta = 1.0
        
        # Create log probabilities
        old_log_prob = torch.tensor([
            [-0.5, -0.3, -0.4, -0.6],
            [-0.2, -0.5, -0.3, -0.4]
        ])
        log_prob = torch.tensor([
            [-0.3, -0.2, -0.3, -0.5],
            [-0.1, -0.4, -0.2, -0.3]
        ])
        response_mask = torch.ones(batch_size, response_length)
        
        # Call the actual compute_reward function
        reward = compute_reward(old_log_prob, log_prob, response_mask, beta=beta, reward_method="1")
        
        # Verify output shape and properties
        self.assertEqual(reward.shape, (batch_size, 1))
        self.assertTrue(torch.is_tensor(reward))
        self.assertTrue(torch.isfinite(reward).all())
        
        # Manual verification of calculation
        # Sequence 1: sum([(-0.3)-(-0.5), (-0.2)-(-0.3), (-0.3)-(-0.4), (-0.5)-(-0.6)])
        #       = sum([0.2, 0.1, 0.1, 0.1]) = 0.5
        self.assertAlmostEqual(reward[0, 0].item(), 0.5, places=5)
        
        # Sequence 2: sum([(-0.1)-(-0.2), (-0.4)-(-0.5), (-0.2)-(-0.3), (-0.3)-(-0.4)])
        #       = sum([0.1, 0.1, 0.1, 0.1]) = 0.4
        self.assertAlmostEqual(reward[1, 0].item(), 0.4, places=5)
    
    def test_compute_reward_method_1_with_beta(self):
        """Test the impact of beta parameter in method 1"""
        batch_size, response_length = 1, 3
        
        old_log_prob = torch.tensor([[-0.5, -0.3, -0.4]])
        log_prob = torch.tensor([[-0.3, -0.2, -0.3]])
        response_mask = torch.ones(batch_size, response_length)
        
        # Test different beta values
        betas = [0.5, 1.0, 2.0]
        
        for beta in betas:
            reward = compute_reward(old_log_prob, log_prob, response_mask, beta=beta, reward_method="1")
            
            # Manual calculation: sum([0.2, 0.1, 0.1]) = 0.4
            # After multiplying by beta: 0.4 * beta
            expected = torch.tensor([[0.4 * beta]])
            torch.testing.assert_close(reward, expected, atol=1e-5, rtol=1e-5)
    
    def test_compute_reward_method_2_basic(self):
        """Test basic functionality of method 2: compute normalized sum of log probabilities"""
        batch_size, response_length = 2, 4
        beta = 1.0
        
        # Create log probabilities
        log_prob = torch.tensor([
            [-0.2, -0.3, -0.4, -0.5],
            [-0.1, -0.2, -0.3, -0.4]
        ])
        old_log_prob = torch.zeros_like(log_prob)  # Method 2 does not use old_log_prob
        response_mask = torch.ones(batch_size, response_length)
        
        # Call the actual compute_reward function
        reward = compute_reward(old_log_prob, log_prob, response_mask, beta=beta, reward_method="2")
        
        
        # Verify output shape and properties
        self.assertEqual(reward.shape, (batch_size, 1))
        self.assertTrue(torch.is_tensor(reward))
        self.assertTrue(torch.isfinite(reward).all())
        
        # Manual verification of calculation
        # Sequence 1: sum([-0.2, -0.3, -0.4, -0.5]) / 4 = -1.4 / 4 = -0.35
        self.assertAlmostEqual(reward[0, 0].item(), -0.35, places=5)
        
        # Sequence 2: sum([-0.1, -0.2, -0.3, -0.4]) / 4 = -1.0 / 4 = -0.25
        self.assertAlmostEqual(reward[1, 0].item(), -0.25, places=5)
    
    def test_compute_reward_method_2_with_mask(self):
        """Test the impact of response_mask in method 2"""
        batch_size, response_length = 2, 5
        beta = 1.0
        
        log_prob = torch.tensor([
            [-0.2, -0.3, -0.4, -0.5, -0.6],
            [-0.1, -0.2, -0.3, -0.4, -0.5]
        ])
        old_log_prob = torch.zeros_like(log_prob)
        
        # Create partial mask
        response_mask = torch.tensor([
            [1.0, 1.0, 1.0, 0.0, 0.0],  # Sequence 1: first 3 valid
            [1.0, 1.0, 1.0, 1.0, 0.0]   # Sequence 2: first 4 valid
        ])
        
        reward = compute_reward(old_log_prob, log_prob, response_mask, beta=beta, reward_method="2")
        
        # Manual verification of calculation
        # Method 2 calculation: sum(all log_prob) / sum(response_mask)
        # Sequence 1: sum([-0.2, -0.3, -0.4, -0.5, -0.6]) / 3 = -2.0 / 3 = -0.6666...
        self.assertAlmostEqual(reward[0, 0].item(), -2.0 / 3.0, places=5)
        
        # Sequence 2: sum([-0.1, -0.2, -0.3, -0.4, -0.5]) / 4 = -1.5 / 4 = -0.375
        self.assertAlmostEqual(reward[1, 0].item(), -1.5 / 4.0, places=5)
    
    def test_compute_reward_method_2_with_beta(self):
        """Test the impact of beta parameter in method 2"""
        batch_size, response_length = 1, 4
        
        log_prob = torch.tensor([[-0.2, -0.3, -0.4, -0.5]])
        old_log_prob = torch.zeros_like(log_prob)
        response_mask = torch.ones(batch_size, response_length)
        
        # Test different beta values
        betas = [0.5, 1.0, 2.0]
        
        for beta in betas:
            reward = compute_reward(old_log_prob, log_prob, response_mask, beta=beta, reward_method="2")
            
            # Manual calculation: sum([-0.2, -0.3, -0.4, -0.5]) / 4 = -1.4 / 4 = -0.35
            # After multiplying by beta: -0.35 * beta
            expected = torch.tensor([[-0.35 * beta]])
            torch.testing.assert_close(reward, expected, atol=1e-5, rtol=1e-5)
    
    def test_compute_reward_positive_improvement(self):
        """Test case where policy improvement is positive (method 1)"""
        batch_size, response_length = 1, 3
        
        # Current policy's log_prob is larger (better)
        old_log_prob = torch.tensor([[-1.0, -1.2, -0.8]])
        log_prob = torch.tensor([[-0.5, -0.6, -0.4]])
        response_mask = torch.ones(batch_size, response_length)
        
        reward = compute_reward(old_log_prob, log_prob, response_mask, beta=1.0, reward_method="1")
        
        # All positions have improvement, reward should be positive
        self.assertGreater(reward[0, 0].item(), 0.0)
        
        # Manual verification: sum([0.5, 0.6, 0.4]) = 1.5
        expected = torch.tensor([[1.5]])
        torch.testing.assert_close(reward, expected, atol=1e-5, rtol=1e-5)
    
    def test_compute_reward_negative_improvement(self):
        """Test case where policy degrades to negative (method 1)"""
        batch_size, response_length = 1, 3
        
        # Current policy's log_prob is smaller (worse)
        old_log_prob = torch.tensor([[-0.5, -0.6, -0.4]])
        log_prob = torch.tensor([[-1.0, -1.2, -0.8]])
        response_mask = torch.ones(batch_size, response_length)
        
        reward = compute_reward(old_log_prob, log_prob, response_mask, beta=1.0, reward_method="1")
        
        # All positions degrade, reward should be negative
        self.assertLess(reward[0, 0].item(), 0.0)
        
        # Manual verification: sum([-0.5, -0.6, -0.4]) = -1.5
        expected = torch.tensor([[-1.5]])
        torch.testing.assert_close(reward, expected, atol=1e-5, rtol=1e-5)
    
    def test_compute_reward_edge_cases(self):
        """Test edge cases"""
        # Test case 1: single token
        single_token_old = torch.tensor([[-0.5]])
        single_token_new = torch.tensor([[-0.3]])
        single_mask = torch.ones(1, 1)
        
        reward_method_1 = compute_reward(single_token_old, single_token_new, single_mask, 
                                         beta=1.0, reward_method="1")
        expected_1 = torch.tensor([[0.2]])
        torch.testing.assert_close(reward_method_1, expected_1, atol=1e-5, rtol=1e-5)
        
        reward_method_2 = compute_reward(single_token_old, single_token_new, single_mask, 
                                         beta=1.0, reward_method="2")
        expected_2 = torch.tensor([[-0.3]])
        torch.testing.assert_close(reward_method_2, expected_2, atol=1e-5, rtol=1e-5)
        
        # Test case 2: zero log probabilities
        zero_old = torch.zeros(1, 3)
        zero_new = torch.zeros(1, 3)
        zero_mask = torch.ones(1, 3)
        
        reward_zero_1 = compute_reward(zero_old, zero_new, zero_mask, beta=1.0, reward_method="1")
        expected_zero_1 = torch.tensor([[0.0]])
        torch.testing.assert_close(reward_zero_1, expected_zero_1)
        
        reward_zero_2 = compute_reward(zero_old, zero_new, zero_mask, beta=1.0, reward_method="2")
        expected_zero_2 = torch.tensor([[0.0]])
        torch.testing.assert_close(reward_zero_2, expected_zero_2)
    
    def test_compute_reward_batch_processing(self):
        """Test batch processing functionality"""
        batch_size, response_length = 4, 5
        
        old_log_prob = torch.randn(batch_size, response_length)
        log_prob = torch.randn(batch_size, response_length)
        response_mask = torch.ones(batch_size, response_length)
        
        # Method 1 batch processing
        reward_1 = compute_reward(old_log_prob, log_prob, response_mask, beta=1.0, reward_method="1")
        self.assertEqual(reward_1.shape, (batch_size, 1))
        self.assertTrue(torch.isfinite(reward_1).all())
        
        # Method 2 batch processing
        reward_2 = compute_reward(old_log_prob, log_prob, response_mask, beta=1.0, reward_method="2")
        self.assertEqual(reward_2.shape, (batch_size, 1))
        self.assertTrue(torch.isfinite(reward_2).all())
        
        # Verify batch processing results are consistent with individual processing
        for i in range(batch_size):
            single_reward_1 = compute_reward(
                old_log_prob[i:i+1], log_prob[i:i+1], response_mask[i:i+1],
                beta=1.0, reward_method="1"
            )
            torch.testing.assert_close(reward_1[i, 0], single_reward_1[0, 0])
    
    def test_compute_reward_invalid_method(self):
        """Test invalid reward_method parameter"""
        old_log_prob = torch.tensor([[-0.5, -0.3]])
        log_prob = torch.tensor([[-0.3, -0.2]])
        response_mask = torch.ones(1, 2)
        
        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            compute_reward(old_log_prob, log_prob, response_mask, beta=1.0, reward_method="3")
        
        self.assertIn("Invalid reward method", str(context.exception))


class TestComputeRewardIntegration(unittest.TestCase):
    """Integration tests: test compute_reward in real-world scenarios"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_reward_method_comparison(self):
        """Compare differences between two reward calculation methods"""
        batch_size, response_length = 2, 4
        
        old_log_prob = torch.tensor([
            [-0.5, -0.3, -0.4, -0.6],
            [-0.2, -0.5, -0.3, -0.4]
        ])
        log_prob = torch.tensor([
            [-0.3, -0.2, -0.3, -0.5],
            [-0.1, -0.4, -0.2, -0.3]
        ])
        response_mask = torch.ones(batch_size, response_length)
        beta = 1.0
        
        # Calculate rewards using both methods
        reward_1 = compute_reward(old_log_prob, log_prob, response_mask, beta=beta, reward_method="1")
        reward_2 = compute_reward(old_log_prob, log_prob, response_mask, beta=beta, reward_method="2")
        # print(reward_1)
        # print(reward_2)

        expected_reward_1 = torch.tensor([[0.5000],
        [0.4000]])
        expected_reward_2 = torch.tensor([[-0.3250],
        [-0.2500]])
        torch.testing.assert_close(reward_1, expected_reward_1, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(reward_2, expected_reward_2, atol=1e-5, rtol=1e-5)
        # Verify two methods produce different results
        self.assertFalse(torch.equal(reward_1, reward_2))
        
        # Verify both methods produce valid output
        self.assertTrue(torch.isfinite(reward_1).all())
        self.assertTrue(torch.isfinite(reward_2).all())
    
    def test_reward_scaling_with_beta(self):
        """Test the impact of beta parameter on reward scaling"""
        old_log_prob = torch.tensor([[-0.5, -0.3, -0.4]])
        log_prob = torch.tensor([[-0.3, -0.2, -0.3]])
        response_mask = torch.ones(1, 3)
        
        # Test linear scaling effect of beta
        beta_1 = 1.0
        beta_2 = 2.0
        
        reward_beta_1 = compute_reward(old_log_prob, log_prob, response_mask, 
                                       beta=beta_1, reward_method="1")
        reward_beta_2 = compute_reward(old_log_prob, log_prob, response_mask, 
                                       beta=beta_2, reward_method="1")
        
        # Reward with beta=2 should be twice that of beta=1
        expected_ratio = beta_2 / beta_1
        actual_ratio = reward_beta_2[0].item() / reward_beta_1[0].item()
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=5)


if __name__ == "__main__":
    unittest.main()
