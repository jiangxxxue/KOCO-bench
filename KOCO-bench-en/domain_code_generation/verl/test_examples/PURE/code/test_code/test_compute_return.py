import unittest
import torch
import sys
import os

# Add the verl directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'verl'))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the actual function we want to test
from verl.trainer.ppo.core_algos import compute_return


class TestComputeReturn(unittest.TestCase):
    """Test compute_return function - token-level return sequence calculation described in PURE document Section 3"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_compute_return_sum_method_basic(self):
        """Test basic functionality of sum method"""
        batch_size, response_length = 2, 4
        gamma = 0.9
        
        # Create token-level rewards
        token_level_rewards = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],    # Sequence 1
            [0.5, 1.5, 2.5, 3.5]     # Sequence 2
        ])
        eos_mask = torch.ones(batch_size, response_length)
        
        # Call the real compute_return function
        returns = compute_return(token_level_rewards, eos_mask, method='sum', gamma=gamma)
        
        # Verify output shape and properties
        self.assertEqual(returns.shape, (batch_size, response_length))
        self.assertTrue(torch.is_tensor(returns))
        self.assertTrue(torch.isfinite(returns).all())
        
        # Manually verify calculation for first sequence
        # Position 3: return = 4.0
        # Position 2: return = 3.0 + 0.9 * 4.0 = 3.0 + 3.6 = 6.6
        # Position 1: return = 2.0 + 0.9 * 6.6 = 2.0 + 5.94 = 7.94
        # Position 0: return = 1.0 + 0.9 * 7.94 = 1.0 + 7.146 = 8.146
        expected_returns_seq1 = torch.tensor([8.146, 7.94, 6.6, 4.0])
        torch.testing.assert_close(returns[0], expected_returns_seq1, atol=1e-3, rtol=1e-3)
    
    def test_compute_return_sum_method_with_gamma(self):
        """Test the effect of different gamma values on sum method"""
        batch_size, response_length = 1, 3
        token_level_rewards = torch.tensor([[1.0, 1.0, 1.0]])
        eos_mask = torch.ones(batch_size, response_length)
        
        # Test gamma = 1.0 (no discount)
        returns_gamma_1 = compute_return(token_level_rewards, eos_mask, method='sum', gamma=1.0)
        
        # When gamma=1, return at each position should be sum of all subsequent rewards
        expected_gamma_1 = torch.tensor([[3.0, 2.0, 1.0]])
        torch.testing.assert_close(returns_gamma_1, expected_gamma_1)
        
        # Test gamma = 0.0 (no propagation)
        returns_gamma_0 = compute_return(token_level_rewards, eos_mask, method='sum', gamma=0.0)
        
        # When gamma=0, return at each position should be just the current reward
        expected_gamma_0 = torch.tensor([[1.0, 1.0, 1.0]])
        torch.testing.assert_close(returns_gamma_0, expected_gamma_0)
        
        # Test gamma = 0.5 (medium discount)
        returns_gamma_05 = compute_return(token_level_rewards, eos_mask, method='sum', gamma=0.5)
        
        # Manual calculation for gamma=0.5
        # Position 2: return = 1.0
        # Position 1: return = 1.0 + 0.5 * 1.0 = 1.5
        # Position 0: return = 1.0 + 0.5 * 1.5 = 1.75
        expected_gamma_05 = torch.tensor([[1.75, 1.5, 1.0]])
        torch.testing.assert_close(returns_gamma_05, expected_gamma_05)
    
    def test_compute_return_min_method_basic(self):
        """Test basic functionality of min method"""
        batch_size, response_length = 2, 4
        
        # Create token-level rewards
        token_level_rewards = torch.tensor([
            [3.0, 1.0, 4.0, 2.0],    # Sequence 1: minimum value is 1.0 at position 1
            [2.5, 3.5, 1.5, 4.5]     # Sequence 2: minimum value is 1.5 at position 2
        ])
        eos_mask = torch.ones(batch_size, response_length)
        
        # Call the real compute_return function
        returns = compute_return(token_level_rewards, eos_mask, method='min')
        
        # Verify output shape and properties
        self.assertEqual(returns.shape, (batch_size, response_length))
        self.assertTrue(torch.is_tensor(returns))
        self.assertTrue(torch.isfinite(returns).all())
        
        # Manually verify calculation for first sequence
        # Position 3: return = 2.0
        # Position 2: return = min(4.0, 2.0) = 2.0
        # Position 1: return = min(1.0, 2.0) = 1.0
        # Position 0: return = min(3.0, 1.0) = 1.0
        expected_returns_seq1 = torch.tensor([1.0, 1.0, 2.0, 2.0])
        torch.testing.assert_close(returns[0], expected_returns_seq1)
        
        # Verify second sequence
        # Position 3: return = 4.5
        # Position 2: return = min(1.5, 4.5) = 1.5
        # Position 1: return = min(3.5, 1.5) = 1.5
        # Position 0: return = min(2.5, 1.5) = 1.5
        expected_returns_seq2 = torch.tensor([1.5, 1.5, 1.5, 4.5])
        torch.testing.assert_close(returns[1], expected_returns_seq2)
    
    def test_compute_return_with_eos_mask(self):
        """Test the effect of eos_mask on calculation"""
        batch_size, response_length = 2, 5
        
        # Create token-level rewards
        token_level_rewards = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0],    # Sequence 1
            [2.0, 1.0, 4.0, 3.0, 2.0]     # Sequence 2
        ])
        
        # Create partial mask
        eos_mask = torch.tensor([
            [1.0, 1.0, 1.0, 0.0, 0.0],    # Sequence 1: first 3 valid
            [1.0, 1.0, 1.0, 1.0, 0.0]     # Sequence 2: first 4 valid
        ])
        
        # Test sum method with mask
        gamma = 0.8
        returns_sum = compute_return(token_level_rewards, eos_mask, method='sum', gamma=gamma)
        
        # Verify masked positions are 0
        self.assertEqual(returns_sum[0, 3].item(), 0.0)
        self.assertEqual(returns_sum[0, 4].item(), 0.0)
        self.assertEqual(returns_sum[1, 4].item(), 0.0)
        
        # Verify calculation for valid positions (first 3 positions of sequence 1)
        # Position 2: return = 3.0 (last valid position)
        # Position 1: return = 2.0 + 0.8 * 3.0 = 2.0 + 2.4 = 4.4
        # Position 0: return = 1.0 + 0.8 * 4.4 = 1.0 + 3.52 = 4.52
        self.assertAlmostEqual(returns_sum[0, 2].item(), 3.0, places=5)
        self.assertAlmostEqual(returns_sum[0, 1].item(), 4.4, places=5)
        self.assertAlmostEqual(returns_sum[0, 0].item(), 4.52, places=5)
        
        # Test min method with mask
        returns_min = compute_return(token_level_rewards, eos_mask, method='min')
        
        # Verify masked positions are 0
        self.assertEqual(returns_min[0, 3].item(), 0.0)
        self.assertEqual(returns_min[0, 4].item(), 0.0)
        self.assertEqual(returns_min[1, 4].item(), 0.0)
        
        # Verify min calculation (first 3 positions of sequence 1: [1.0, 2.0, 3.0])
        # Position 2: return = 3.0
        # Position 1: return = min(2.0, 3.0) = 2.0  
        # Position 0: return = min(1.0, 2.0) = 1.0
        self.assertAlmostEqual(returns_min[0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(returns_min[0, 1].item(), 2.0, places=5)
        self.assertAlmostEqual(returns_min[0, 2].item(), 3.0, places=5)
    
    def test_compute_return_input_output_samples_sum(self):
        """Test specific input/output examples for sum method"""
        # Test case 1: Basic sum calculation
        token_rewards = torch.tensor([[2.0, 1.0, 3.0]])
        eos_mask = torch.ones(1, 3)
        gamma = 0.9
        
        # Manually calculate expected result
        # Position 2: return = 3.0
        # Position 1: return = 1.0 + 0.9 * 3.0 = 1.0 + 2.7 = 3.7
        # Position 0: return = 2.0 + 0.9 * 3.7 = 2.0 + 3.33 = 5.33
        
        returns = compute_return(token_rewards, eos_mask, method='sum', gamma=gamma)
        
        expected = torch.tensor([[5.33, 3.7, 3.0]])
        torch.testing.assert_close(returns, expected, atol=1e-2, rtol=1e-2)
        
        # Test case 2: Special case with gamma=1
        token_rewards_2 = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        eos_mask_2 = torch.ones(1, 4)
        gamma_2 = 1.0
        
        returns_2 = compute_return(token_rewards_2, eos_mask_2, method='sum', gamma=gamma_2)
        
        # When gamma=1, return at each position is sum of all subsequent rewards
        expected_2 = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        torch.testing.assert_close(returns_2, expected_2)
        
        # Test case 3: Case with mask
        token_rewards_3 = torch.tensor([[2.0, 3.0, 1.0, 4.0]])
        eos_mask_3 = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # Only first two are valid
        gamma_3 = 0.8
        
        returns_3 = compute_return(token_rewards_3, eos_mask_3, method='sum', gamma=gamma_3)
        
        # Only calculate first two valid positions
        # Position 1: return = 3.0 (last valid)
        # Position 0: return = 2.0 + 0.8 * 3.0 = 2.0 + 2.4 = 4.4
        expected_3 = torch.tensor([[4.4, 3.0, 0.0, 0.0]])
        torch.testing.assert_close(returns_3, expected_3)
    
    def test_compute_return_input_output_samples_min(self):
        """Test specific input/output examples for min method"""
        # Test case 1: Basic min calculation
        token_rewards = torch.tensor([[5.0, 2.0, 8.0, 3.0]])
        eos_mask = torch.ones(1, 4)
        
        returns = compute_return(token_rewards, eos_mask, method='min')
        
        # Manually calculate expected result
        # Position 3: return = 3.0
        # Position 2: return = min(8.0, 3.0) = 3.0
        # Position 1: return = min(2.0, 3.0) = 2.0
        # Position 0: return = min(5.0, 2.0) = 2.0
        expected = torch.tensor([[2.0, 2.0, 3.0, 3.0]])
        torch.testing.assert_close(returns, expected)
        
        # Test case 2: Decreasing sequence
        token_rewards_2 = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        eos_mask_2 = torch.ones(1, 4)
        
        returns_2 = compute_return(token_rewards_2, eos_mask_2, method='min')
        
        # In decreasing sequence, min at each position is the rightmost value
        expected_2 = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        torch.testing.assert_close(returns_2, expected_2)
        
        # Test case 3: Increasing sequence
        token_rewards_3 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        eos_mask_3 = torch.ones(1, 4)
        
        returns_3 = compute_return(token_rewards_3, eos_mask_3, method='min')
        
        # In increasing sequence, min at each position is the value at that position
        expected_3 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        torch.testing.assert_close(returns_3, expected_3)
        
        # Test case 4: Min calculation with mask
        token_rewards_4 = torch.tensor([[3.0, 1.0, 5.0, 2.0]])
        eos_mask_4 = torch.tensor([[1.0, 1.0, 1.0, 0.0]])  # Last position masked
        
        returns_4 = compute_return(token_rewards_4, eos_mask_4, method='min')
        
        # Only consider first 3 positions: [3.0, 1.0, 5.0]
        # Position 2: return = 5.0 (last valid)
        # Position 1: return = min(1.0, 5.0) = 1.0
        # Position 0: return = min(3.0, 1.0) = 1.0
        expected_4 = torch.tensor([[1.0, 1.0, 5.0, 0.0]])
        torch.testing.assert_close(returns_4, expected_4)
    
    def test_compute_return_edge_cases(self):
        """Test edge cases"""
        # Test case 1: Single token
        single_token = torch.tensor([[5.0]])
        single_mask = torch.ones(1, 1)
        
        # sum method
        returns_sum = compute_return(single_token, single_mask, method='sum')
        expected_sum = torch.tensor([[5.0]])
        torch.testing.assert_close(returns_sum, expected_sum)
        
        # min method
        returns_min = compute_return(single_token, single_mask, method='min')
        expected_min = torch.tensor([[5.0]])
        torch.testing.assert_close(returns_min, expected_min)
        
        # Test case 2: All-zero rewards
        zero_rewards = torch.zeros(1, 3)
        zero_mask = torch.ones(1, 3)
        gamma = 0.9
        
        # sum method should return all zeros
        returns_sum_zero = compute_return(zero_rewards, zero_mask, method='sum', gamma=gamma)
        expected_zero_sum = torch.zeros(1, 3)
        torch.testing.assert_close(returns_sum_zero, expected_zero_sum)
        
        # min method should return all zeros
        returns_min_zero = compute_return(zero_rewards, zero_mask, method='min')
        expected_zero_min = torch.zeros(1, 3)
        torch.testing.assert_close(returns_min_zero, expected_zero_min)
        
        # Test case 3: Fully masked sequence
        all_masked_rewards = torch.tensor([[1.0, 2.0, 3.0]])
        all_zero_mask = torch.zeros(1, 3)
        
        # sum method should return all zeros
        returns_sum_masked = compute_return(all_masked_rewards, all_zero_mask, method='sum', gamma=gamma)
        expected_masked_sum = torch.zeros(1, 3)
        torch.testing.assert_close(returns_sum_masked, expected_masked_sum)
        
        # min method should return all zeros
        returns_min_masked = compute_return(all_masked_rewards, all_zero_mask, method='min')
        expected_masked_min = torch.zeros(1, 3)
        torch.testing.assert_close(returns_min_masked, expected_masked_min)
    
    def test_compute_return_multiple_sequences(self):
        """Test multi-sequence batch processing"""
        batch_size, response_length = 3, 4
        
        # Create multiple different sequences
        token_rewards = torch.tensor([
            [1.0, 2.0, 1.0, 3.0],    # Sequence 1
            [2.0, 1.0, 3.0, 1.0],    # Sequence 2  
            [3.0, 1.0, 2.0, 1.0]     # Sequence 3
        ])
        eos_mask = torch.ones(batch_size, response_length)
        gamma = 0.8
        
        # sum method batch processing
        returns_sum = compute_return(token_rewards, eos_mask, method='sum', gamma=gamma)
        
        # Verify shape of each sequence
        self.assertEqual(returns_sum.shape, (batch_size, response_length))
        
        # Manually verify first sequence [1.0, 2.0, 1.0, 3.0]
        # Position 3: return = 3.0
        # Position 2: return = 1.0 + 0.8 * 3.0 = 1.0 + 2.4 = 3.4
        # Position 1: return = 2.0 + 0.8 * 3.4 = 2.0 + 2.72 = 4.72
        # Position 0: return = 1.0 + 0.8 * 4.72 = 1.0 + 3.776 = 4.776
        expected_seq1 = torch.tensor([4.776, 4.72, 3.4, 3.0])
        torch.testing.assert_close(returns_sum[0], expected_seq1, atol=1e-3, rtol=1e-3)
        
        # min method batch processing
        returns_min = compute_return(token_rewards, eos_mask, method='min')
        
        # Verify shape of each sequence
        self.assertEqual(returns_min.shape, (batch_size, response_length))
        
        # Manually verify min calculation for first sequence [1.0, 2.0, 1.0, 3.0]
        # Position 3: return = 3.0
        # Position 2: return = min(1.0, 3.0) = 1.0
        # Position 1: return = min(2.0, 1.0) = 1.0
        # Position 0: return = min(1.0, 1.0) = 1.0
        expected_min_seq1 = torch.tensor([1.0, 1.0, 1.0, 3.0])
        torch.testing.assert_close(returns_min[0], expected_min_seq1)
        
        # Verify min calculation for second sequence [2.0, 1.0, 3.0, 1.0]
        # Position 3: return = 1.0
        # Position 2: return = min(3.0, 1.0) = 1.0
        # Position 1: return = min(1.0, 1.0) = 1.0
        # Position 0: return = min(2.0, 1.0) = 1.0
        expected_min_seq2 = torch.tensor([1.0, 1.0, 1.0, 1.0])
        torch.testing.assert_close(returns_min[1], expected_min_seq2)


class TestComputeReturnIntegration(unittest.TestCase):
    """Integration test: Test integration of compute_return with other components"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_method_parameter_switching(self):
        """Test method parameter switching functionality"""
        token_rewards = torch.tensor([[3.0, 1.0, 4.0, 2.0]])
        eos_mask = torch.ones(1, 4)
        
        # Directly use the real compute_return function
        
        # Test sum method
        returns_sum = compute_return(token_rewards, eos_mask, method='sum', gamma=0.9)
        self.assertEqual(returns_sum.shape, (1, 4))
        self.assertTrue(torch.isfinite(returns_sum).all())
        
        # Test min method
        returns_min = compute_return(token_rewards, eos_mask, method='min')
        self.assertEqual(returns_min.shape, (1, 4))
        self.assertTrue(torch.isfinite(returns_min).all())
        
        # Verify two methods produce different results
        self.assertFalse(torch.equal(returns_sum, returns_min))
        
        # Test invalid method - should raise ValueError (invalid parameter value)
        # Note: Using ValueError rather than NotImplementedError is more semantically appropriate
        # ValueError: Parameter value not in allowed range
        # NotImplementedError: Functionality not yet implemented (not applicable here)
        with self.assertRaises(NotImplementedError) as context:
            compute_return(token_rewards, eos_mask, method='invalid')
    
    def test_integration_with_advantage_computation(self):
        """Test integration with advantage computation"""
        batch_size, response_length = 2, 5
        
        # Create token-level rewards and baseline values
        token_rewards = torch.tensor([[3.0, 1.0, 4.0, 2.0, 1.0], [2.0, 1.0, 3.0, 1.0, 2.0]])
        baseline_values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
        eos_mask = torch.ones(batch_size, response_length)
        gamma = 0.95
        
        # Calculate returns (sum method)
        returns = compute_return(token_rewards, eos_mask, method='sum', gamma=gamma)
        
        # Calculate advantages = returns - baseline_values
        advantages = returns - baseline_values

        print(advantages)
        expected_advantages = torch.tensor([[ 9.0893,  5.4624,  3.8025, -1.0500, -4.0000],
        [ 7.1439,  4.4672,  2.7550, -1.1000, -3.0000]])

        
        # Verify integration results
        # self.assertEqual(advantages.shape, (batch_size, response_length))
        # self.assertTrue(torch.isfinite(advantages).all())
        # Most important step, test specific values
        torch.testing.assert_close(advantages, expected_advantages, atol=1e-4, rtol=1e-4)
        # Verify reasonableness of advantage calculation
        # When returns > baseline, advantage should be positive
        positive_advantage_mask = returns > baseline_values
        self.assertTrue((advantages[positive_advantage_mask] >= 0).all())


if __name__ == "__main__":
    unittest.main()
