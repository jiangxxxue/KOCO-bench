import unittest
import torch
import sys
import os

# Add the verl directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'luffy', 'verl'))

# Import the actual function we want to test
from verl.mix_src.mix_core_alg import compute_grpo_outcome_advantage_split


class TestComputeGrpoOutcomeAdvantageSplit(unittest.TestCase):
    """Test compute_grpo_outcome_advantage_split function - Grouped GRPO advantage estimation described in LUFFY document Section 1"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_basic_functionality(self):
        """Test basic functionality: compute within-group baseline using only on-policy samples"""
        batch_size, response_length = 4, 5
        
        # Create token-level rewards
        token_level_rewards = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 3.0],  # Sequence 1: total score 3.0
            [0.0, 0.0, 0.0, 0.0, 5.0],  # Sequence 2: total score 5.0
            [0.0, 0.0, 0.0, 0.0, 2.0],  # Sequence 3: total score 2.0
            [0.0, 0.0, 0.0, 0.0, 4.0],  # Sequence 4: total score 4.0
        ])
        eos_mask = torch.ones(batch_size, response_length)
        
        # First two samples belong to the same prompt (index=0), last two belong to another prompt (index=1)
        # Note: index needs to be an integer list, not a tensor, as it will be used as dictionary keys
        index = [0, 0, 1, 1]
        
        # All samples are on-policy
        on_policy_mask = torch.tensor([True, True, True, True])
        
        # Call the actual function
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask
        )
        
        # print(advantages)
        # print(returns)

        expected_advantages = torch.tensor([[-0.7071, -0.7071, -0.7071, -0.7071, -0.7071],
        [ 0.7071,  0.7071,  0.7071,  0.7071,  0.7071],
        [-0.7071, -0.7071, -0.7071, -0.7071, -0.7071],
        [ 0.7071,  0.7071,  0.7071,  0.7071,  0.7071]])
        expected_returns = torch.tensor([[-0.7071, -0.7071, -0.7071, -0.7071, -0.7071],
        [ 0.7071,  0.7071,  0.7071,  0.7071,  0.7071],
        [-0.7071, -0.7071, -0.7071, -0.7071, -0.7071],
        [ 0.7071,  0.7071,  0.7071,  0.7071,  0.7071]])
        torch.testing.assert_close(advantages, expected_advantages, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(returns, expected_returns, atol=1e-4, rtol=1e-4)
    
    def test_on_policy_mask_filtering(self):
        """Test on_policy_mask filtering: compute baseline using only on-policy samples"""
        batch_size, response_length = 4, 5
        
        token_level_rewards = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 2.0],  # on-policy
            [0.0, 0.0, 0.0, 0.0, 8.0],  # off-policy (should not affect baseline)
            [0.0, 0.0, 0.0, 0.0, 4.0],  # on-policy
            [0.0, 0.0, 0.0, 0.0, 6.0],  # on-policy
        ])
        eos_mask = torch.ones(batch_size, response_length)
        index = [0, 0, 0, 0]  # All samples from the same prompt
        
        # Second sample is off-policy
        on_policy_mask = torch.tensor([True, False, True, True])
        
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask
        )
        
        # print(advantages)
        # print(returns)

        expected_advantages = torch.tensor([[-1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
        [ 2.0000,  2.0000,  2.0000,  2.0000,  2.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000]])
        expected_returns = torch.tensor([[-1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
        [ 2.0000,  2.0000,  2.0000,  2.0000,  2.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000]])
        torch.testing.assert_close(advantages, expected_advantages, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(returns, expected_returns, atol=1e-4, rtol=1e-4)
        # Verify output
    
    def test_single_on_policy_sample_per_group(self):
        """Test case with only one on-policy sample per group: mean=0, std=1"""
        batch_size, response_length = 2, 4
        
        token_level_rewards = torch.tensor([
            [0.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 3.0],
        ])
        eos_mask = torch.ones(batch_size, response_length)
        index = [0, 1]  # Different prompts
        on_policy_mask = torch.tensor([True, True])
        
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask
        )
        
        # When there's only one sample per group, mean=0, std=1
        # So after normalization: (score - 0) / 1 = score
        # Sequence 1: advantage = 5.0
        # Sequence 2: advantage = 3.0
        self.assertAlmostEqual(advantages[0, -1].item(), 5.0, places=4)
        self.assertAlmostEqual(advantages[1, -1].item(), 3.0, places=4)
    
    def test_use_std_parameter(self):
        """Test use_std parameter: control whether to use standard deviation normalization"""
        batch_size, response_length = 4, 3
        
        token_level_rewards = torch.tensor([
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 6.0],
            [0.0, 0.0, 8.0],
        ])
        eos_mask = torch.ones(batch_size, response_length)
        index = [0, 0, 0, 0]  # Same prompt
        on_policy_mask = torch.tensor([True, True, True, True])
        
        # Use standard deviation normalization
        advantages_with_std, _ = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask, use_std=True
        )
        
        # Do not use standard deviation normalization
        advantages_no_std, _ = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask, use_std=False
        )
        
        # Two methods should produce different results
        self.assertFalse(torch.equal(advantages_with_std, advantages_no_std))
        
        # Verify that use_std=False only subtracts the mean
        # mean([2, 4, 6, 8]) = 5.0
        # Without std: score - mean, then broadcast to all token positions
        # Note: advantage values are broadcast to the entire sequence
        expected_no_std = torch.tensor([
            [-3.0, -3.0, -3.0],  # 2 - 5 = -3, broadcast to all positions
            [-1.0, -1.0, -1.0],  # 4 - 5 = -1, broadcast to all positions
            [1.0, 1.0, 1.0],     # 6 - 5 = 1, broadcast to all positions
            [3.0, 3.0, 3.0],     # 8 - 5 = 3, broadcast to all positions
        ])
        torch.testing.assert_close(advantages_no_std, expected_no_std, atol=1e-4, rtol=1e-4)
    
    def test_epsilon_parameter(self):
        """Test epsilon parameter: numerical stability"""
        batch_size, response_length = 2, 3
        
        token_level_rewards = torch.tensor([
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 3.0],  # Same score, std=0
        ])
        eos_mask = torch.ones(batch_size, response_length)
        index = [0, 0]
        on_policy_mask = torch.tensor([True, True])
        
        # When std=0, it will be replaced with 1.0, so no division by zero
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask, epsilon=1e-6
        )

        # print(advantages)
        # print(returns)
        expected_advantages = torch.tensor([[0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000]])
        expected_returns = torch.tensor([[0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000]])
        torch.testing.assert_close(advantages, expected_advantages, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(returns, expected_returns, atol=1e-4, rtol=1e-4)
        # Verify no NaN or Inf
        # self.assertTrue(torch.isfinite(advantages).all())
        # self.assertTrue(torch.isfinite(returns).all())
    
    def test_eos_mask_application(self):
        """Test eos_mask application: compute advantages only at valid token positions"""
        batch_size, response_length = 2, 5
        
        token_level_rewards = torch.tensor([
            [0.0, 0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 6.0],
        ])
        
        # Partial mask
        eos_mask = torch.tensor([
            [1.0, 1.0, 1.0, 1.0, 0.0],  # Last position is invalid
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])
        
        index = [0, 0]
        on_policy_mask = torch.tensor([True, True])
        
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask
        )
        
        # print(advantages)

        expected_advantages = torch.tensor([[-0.7071, -0.7071, -0.7071, -0.7071, -0.0000],
        [ 0.7071,  0.7071,  0.7071,  0.7071,  0.7071]])
        torch.testing.assert_close(advantages, expected_advantages, atol=1e-4, rtol=1e-4)
        # Verify masked position is 0
        self.assertEqual(advantages[0, -1].item(), 0.0)
        
        # Verify valid positions have values
        self.assertNotEqual(advantages[1, -1].item(), 0.0)
    
    def test_multiple_groups(self):
        """Test handling of multiple prompt groups"""
        batch_size, response_length = 6, 4
        
        token_level_rewards = torch.tensor([
            [0.0, 0.0, 0.0, 2.0],  # group 0
            [0.0, 0.0, 0.0, 4.0],  # group 0
            [0.0, 0.0, 0.0, 3.0],  # group 1
            [0.0, 0.0, 0.0, 5.0],  # group 1
            [0.0, 0.0, 0.0, 1.0],  # group 2
            [0.0, 0.0, 0.0, 3.0],  # group 2
        ])
        eos_mask = torch.ones(batch_size, response_length)
        index = [0, 0, 1, 1, 2, 2]
        on_policy_mask = torch.tensor([True, True, True, True, True, True])
        
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask
        )
        
        # print(advantages)

        expected_advantages = torch.tensor([[-0.7071, -0.7071, -0.7071, -0.7071],
        [ 0.7071,  0.7071,  0.7071,  0.7071],
        [-0.7071, -0.7071, -0.7071, -0.7071],
        [ 0.7071,  0.7071,  0.7071,  0.7071],
        [-0.7071, -0.7071, -0.7071, -0.7071],
        [ 0.7071,  0.7071,  0.7071,  0.7071]])
        torch.testing.assert_close(advantages, expected_advantages, atol=1e-4, rtol=1e-4)
        # Verify output shape
        self.assertEqual(advantages.shape, (batch_size, response_length))
        self.assertTrue(torch.isfinite(advantages).all())
        
        # Samples within each group should be normalized based on their group's statistics
        # Group 0: mean([2, 4]) = 3.0
        # Group 1: mean([3, 5]) = 4.0
        # Group 2: mean([1, 3]) = 2.0
    
    def test_token_level_broadcast(self):
        """Test sequence-level advantage broadcast to token level"""
        batch_size, response_length = 2, 4
        
        token_level_rewards = torch.tensor([
            [0.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 3.0],
        ])
        eos_mask = torch.ones(batch_size, response_length)
        index = [0, 0]
        on_policy_mask = torch.tensor([True, True])
        
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask
        )
        
        # print(advantages)

        expected_advantages = torch.tensor([[ 0.7071,  0.7071,  0.7071,  0.7071],
        [-0.7071, -0.7071, -0.7071, -0.7071]])
        torch.testing.assert_close(advantages, expected_advantages, atol=1e-4, rtol=1e-4)
        # Verify that all valid token positions in each sequence have the same advantage value
        # (because it's outcome reward, broadcast to all positions)
        for i in range(batch_size):
            valid_positions = eos_mask[i].bool()
            unique_values = torch.unique(advantages[i, valid_positions])
            # All valid positions should have the same value
            self.assertEqual(len(unique_values), 1)
    
    def test_edge_case_zero_rewards(self):
        """Test edge case: all zero rewards"""
        batch_size, response_length = 2, 3
        
        token_level_rewards = torch.zeros(batch_size, response_length)
        eos_mask = torch.ones(batch_size, response_length)
        index = [0, 0]
        on_policy_mask = torch.tensor([True, True])
        
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask
        )
        
        # All zero rewards should produce all zero advantages
        self.assertTrue((advantages == 0).all())
        self.assertTrue((returns == 0).all())
    
    def test_edge_case_single_sample(self):
        """Test edge case: single sample"""
        batch_size, response_length = 1, 4
        
        token_level_rewards = torch.tensor([[0.0, 0.0, 0.0, 7.0]])
        eos_mask = torch.ones(batch_size, response_length)
        index = [0]
        on_policy_mask = torch.tensor([True])
        
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask
        )

        # print(advantages)

        expected_advantages = torch.tensor([[7.0000, 7.0000, 7.0000, 7.0000]])
        torch.testing.assert_close(advantages, expected_advantages, atol=1e-4, rtol=1e-4)
        
        # Single sample group: mean=0, std=1, so advantage=score
        self.assertEqual(advantages.shape, (batch_size, response_length))
        self.assertAlmostEqual(advantages[0, -1].item(), 7.0, places=4)


class TestComputeGrpoOutcomeAdvantageSplitIntegration(unittest.TestCase):
    """Integration tests: test compute_grpo_outcome_advantage_split in real-world scenarios"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_mixed_on_off_policy_scenario(self):
        """Test real-world scenario with mixed on-policy and off-policy samples"""
        batch_size, response_length = 8, 6
        
        # Simulate real scenario: each prompt has 4 responses, 2 on-policy and 2 off-policy
        token_level_rewards = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],  # prompt 0, on-policy
            [0.0, 0.0, 0.0, 0.0, 0.0, 7.0],  # prompt 0, off-policy
            [0.0, 0.0, 0.0, 0.0, 0.0, 5.0],  # prompt 0, on-policy
            [0.0, 0.0, 0.0, 0.0, 0.0, 9.0],  # prompt 0, off-policy
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],  # prompt 1, on-policy
            [0.0, 0.0, 0.0, 0.0, 0.0, 8.0],  # prompt 1, off-policy
            [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],  # prompt 1, on-policy
            [0.0, 0.0, 0.0, 0.0, 0.0, 6.0],  # prompt 1, off-policy
        ])
        eos_mask = torch.ones(batch_size, response_length)
        index = [0, 0, 0, 0, 1, 1, 1, 1]
        on_policy_mask = torch.tensor([True, False, True, False, True, False, True, False])
        
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask
        )
        
        # print(advantages)

        expected_advantages = torch.tensor([[-0.7071, -0.7071, -0.7071, -0.7071, -0.7071, -0.7071],
        [ 2.1213,  2.1213,  2.1213,  2.1213,  2.1213,  2.1213],
        [ 0.7071,  0.7071,  0.7071,  0.7071,  0.7071,  0.7071],
        [ 3.5355,  3.5355,  3.5355,  3.5355,  3.5355,  3.5355],
        [-0.7071, -0.7071, -0.7071, -0.7071, -0.7071, -0.7071],
        [ 3.5355,  3.5355,  3.5355,  3.5355,  3.5355,  3.5355],
        [ 0.7071,  0.7071,  0.7071,  0.7071,  0.7071,  0.7071],
        [ 2.1213,  2.1213,  2.1213,  2.1213,  2.1213,  2.1213]])
        torch.testing.assert_close(advantages, expected_advantages, atol=1e-4, rtol=1e-4)
        # Verify output
        self.assertEqual(advantages.shape, (batch_size, response_length))
        # self.assertTrue(torch.isfinite(advantages).all())
        
        # Prompt 0 baseline should be based only on on-policy samples: mean([3, 5]) = 4.0
        # Prompt 1 baseline should be based only on on-policy samples: mean([2, 4]) = 3.0
        # But all samples (including off-policy) will be normalized
    
    def test_normalization_consistency(self):
        """Test normalization consistency"""
        batch_size, response_length = 4, 5
        
        token_level_rewards = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 0.0, 4.0],
        ])
        eos_mask = torch.ones(batch_size, response_length)
        index = [0, 0, 0, 0]
        on_policy_mask = torch.tensor([True, True, True, True])
        
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask, use_std=True
        )
        
        # Verify that normalized advantage mean is close to 0 (for the same group)
        valid_advantages = advantages[:, -1]  # Take the last position
        mean_advantage = valid_advantages.mean()
        
        # After standardization, mean should be close to 0
        self.assertAlmostEqual(mean_advantage.item(), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
