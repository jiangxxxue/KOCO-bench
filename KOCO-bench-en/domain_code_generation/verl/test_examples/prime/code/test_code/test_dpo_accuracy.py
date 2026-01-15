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
from prime.prime_core_algos import compute_dpo_accuracy


class TestDPOAccuracy(unittest.TestCase):
    """Test compute_dpo_accuracy function - DPO pairwise comparison accuracy described in Section 3 of the documentation"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_compute_dpo_accuracy_basic(self):
        """Test basic functionality of DPO accuracy calculation"""
        n_samples = 3
        batch_size = 6  # Must be divisible by n_samples
        seq_len = 10
        
        token_level_scores = torch.tensor([[0.1,0.3,1.0,-0.1,0.7,0.5,0.1,0.3,0.6,-0.1], [0.2,0.3,0.2,-0.4,0.9,-0.1,0.2,0.5,0.1,0.3], [-0.1,0.3,0.6,-0.1,0.2,0.7,1.0,0.2,0.5,0.1]])


        acc = torch.tensor([1.0, 0.5, 0.0, 1.0, 0.8, 0.2])
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # print(accuracy)
        expected_accuracy = torch.tensor(0.2500)
        torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-30, rtol=1e-30)
        # Verify output
        # self.assertEqual(accuracy.dim(), 0)
        # self.assertTrue(torch.is_tensor(accuracy))
        # self.assertTrue(0 <= accuracy.item() <= 1)
        # self.assertFalse(torch.isnan(accuracy))
    
    def test_equal_accuracy_case(self):
        """Test edge case with equal accuracy"""
        batch_size = 4
        seq_len = 5
        n_samples = 2
        
        # Test case with equal accuracy
        token_level_scores = torch.tensor([[0.1,0.3,1.0,-0.1,0.7], [0.2,0.3,0.2,-0.4,0.9], [-0.1,0.3,0.6,-0.1,0.2], [0.1,0.8,0.6,-0.1,0.2]])
        acc = torch.ones(batch_size) * 0.5
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        self.assertTrue(torch.is_tensor(accuracy))
        # For equal accuracy, should return 0.5
        self.assertEqual(accuracy.item(), 0.5)
    
    def test_pairwise_comparison_logic(self):
        """Test correctness of pairwise comparison logic"""
        n_samples = 2
        batch_size = 4  # 2 groups × 2 samples per group
        seq_len = 6
        
        # Create data with clear differences
        token_level_scores = torch.tensor([
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Group 1, Sample 1: high scores
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Group 1, Sample 2: low scores
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],  # Group 2, Sample 1: very high scores
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],  # Group 2, Sample 2: low scores
        ])
        acc = torch.tensor([0.8, 0.2, 0.9, 0.1])  # Accuracy corresponds to score trends
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # print(accuracy)
        # Since score trends match accuracy, accuracy should be high
        # self.assertTrue(0 <= accuracy.item() <= 1)
        # self.assertFalse(torch.isnan(accuracy))
        expected_accuracy = torch.tensor(1.)
        torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-30, rtol=1e-30)
    
    def test_weighted_average_computation(self):
        """Test weighted average accuracy calculation"""
        n_samples = 2
        batch_size = 6  # 3 groups × 2 samples per group
        seq_len = 4
        
        token_level_scores = torch.tensor([[-0.1, 0.7, 0.1, 0.5], [0.1,0.4,0.7,0.2], [0.5,-0.3,0.2,0.1], [0.2,0.4,0.6,0.8], [-0.2,0.65,-0.44,0.55], [1.0,0.8,0.6,0.4]])
        acc = torch.tensor([1.0, 0.0, 0.8, 0.2, 0.6, 0.4])  # Different accuracy differences
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # print(accuracy)
        # Verify weighted average result is within reasonable range
        expected_accuracy = torch.tensor(0.)
        torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-30, rtol=1e-30)
    

    
    def test_response_mask_effect(self):
        """Test impact of response_mask on accuracy calculation"""
        n_samples = 2
        batch_size = 4
        seq_len = 6
        
        token_level_scores = torch.tensor([[0.1,0.3,1.0,-0.1,0.7,0.5], [0.2,0.3,0.2,-0.4,0.9,-0.1], [-0.1,0.3,0.6,-0.1,0.2,0.7], [0.1,0.8,0.6,-0.1,0.2,0.5]])
        acc = torch.tensor([1.0, 0.0, 0.8, 0.2])
        
        # Test fully valid mask
        full_mask = torch.ones(batch_size, seq_len)
        accuracy_full = compute_dpo_accuracy(token_level_scores, acc, full_mask, n_samples)
        # print(accuracy_full)
        expected_accuracy_full = torch.tensor(0.5)
        torch.testing.assert_close(accuracy_full, expected_accuracy_full, atol=1e-30, rtol=1e-30)
        # Test partially valid mask
        partial_mask = torch.ones(batch_size, seq_len)
        partial_mask[:, -2:] = 0  # Last two positions masked out
        accuracy_partial = compute_dpo_accuracy(token_level_scores, acc, partial_mask, n_samples)
        expected_accuracy_partial = torch.tensor(0.5)
        torch.testing.assert_close(accuracy_partial, expected_accuracy_partial, atol=1e-30, rtol=1e-30)

    
    def test_score_aggregation_logic(self):
        """Test sequence-level score aggregation logic"""
        n_samples = 2
        batch_size = 4
        seq_len = 3
        
        # Create predictable scores
        token_level_scores = torch.tensor([
            [1.0, 2.0, 3.0],    # Sequence score should be 6.0
            [0.5, 1.0, 1.5],    # Sequence score should be 3.0
            [2.0, 2.0, 2.0],    # Sequence score should be 6.0
            [1.0, 1.0, 1.0],    # Sequence score should be 3.0
        ])
        acc = torch.tensor([0.9, 0.1, 0.8, 0.2])
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # Manual verification: sequence scores [6.0, 3.0, 6.0, 3.0], accuracy [0.9, 0.1, 0.8, 0.2]
        # Group 1: score difference = 6.0 - 3.0 = 3.0 > 0, accuracy difference = 0.9 - 0.1 = 0.8 > 0, consistent direction
        # Group 2: score difference = 6.0 - 3.0 = 3.0 > 0, accuracy difference = 0.8 - 0.2 = 0.6 > 0, consistent direction
        # Since directions are completely consistent, accuracy should be 1.0
        self.assertAlmostEqual(accuracy.item(), 1.0, places=30)

    def test_input_output_samples(self):
        """Test specific input-output examples"""
        # Test case 1: Simple 2-sample case
        token_level_scores = torch.tensor([
            [2.0, 1.0, 3.0],  # Sequence score: 6.0
            [1.0, 0.5, 0.5]   # Sequence score: 2.0
        ])
        acc = torch.tensor([0.8, 0.3])
        response_mask = torch.ones(2, 3)
        n_samples = 2
        
        result = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # Manual calculation verification:
        # Score difference = 6.0 - 2.0 = 4.0 > 0
        # Accuracy difference = 0.8 - 0.3 = 0.5 > 0
        # Consistent direction, weight = 0.5, accuracy = 1.0
        expected_accuracy = 1.0
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)
        
        # Test case 2: Inconsistent direction case
        token_level_scores = torch.tensor([
            [1.0, 1.0, 1.0],  # Sequence score: 3.0
            [2.0, 2.0, 2.0]   # Sequence score: 6.0
        ])
        acc = torch.tensor([0.9, 0.1])  # High accuracy corresponds to low score
        response_mask = torch.ones(2, 3)
        
        result = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # Manual calculation verification:
        # Score difference = 3.0 - 6.0 = -3.0 < 0
        # Accuracy difference = 0.9 - 0.1 = 0.8 > 0
        # Inconsistent direction, accuracy = 0.0
        expected_accuracy = 0.0
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)
        
        # Test case 3: Multiple groups
        token_level_scores = torch.tensor([
            [3.0, 2.0],  # Group 1, Sample 1: 5.0
            [1.0, 1.0],  # Group 1, Sample 2: 2.0
            [4.0, 1.0],  # Group 2, Sample 1: 5.0
            [2.0, 0.5]   # Group 2, Sample 2: 2.5
        ])
        acc = torch.tensor([0.8, 0.2, 0.7, 0.4])
        response_mask = torch.ones(4, 2)
        
        result = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # Manual calculation verification:
        # Group 1: score difference = 5.0 - 2.0 = 3.0 > 0, accuracy difference = 0.8 - 0.2 = 0.6 > 0, consistent
        # Group 2: score difference = 5.0 - 2.5 = 2.5 > 0, accuracy difference = 0.7 - 0.4 = 0.3 > 0, consistent
        # Both groups consistent, overall accuracy = (1.0*0.6 + 1.0*0.3) / (0.6 + 0.3) = 0.9 / 0.9 = 1.0
        expected_accuracy = 1.0
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)

    def test_edge_case_input_output_samples(self):
        """Test edge case input-output examples"""
        n_samples = 2
        
        # Test case 1: Zero score case
        token_level_scores = torch.zeros(2, 3)
        acc = torch.tensor([0.6, 0.4])
        response_mask = torch.ones(2, 3)
        
        result = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # Score difference = 0.0 - 0.0 = 0.0, not greater than 0, but accuracy difference > 0
        # Inconsistent direction, accuracy should be 0.0
        expected_accuracy = 0.0
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)
        
        # Test case 2: Equal accuracy case
        token_level_scores = torch.tensor([
            [1.0, 2.0],
            [3.0, 1.0]
        ])
        acc = torch.tensor([0.5, 0.5])  # Completely equal accuracy
        response_mask = torch.ones(2, 2)
        
        result = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # Accuracy difference is 0, should return 0.5
        expected_accuracy = 0.5
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)
        
        # Test case 3: With mask
        token_level_scores = torch.tensor([
            [2.0, 4.0, 1.0],  # Valid scores: 2.0 + 4.0 = 6.0
            [1.0, 2.0, 3.0]   # Valid scores: 1.0 + 2.0 = 3.0 (last one masked)
        ])
        acc = torch.tensor([0.8, 0.3])
        response_mask = torch.tensor([
            [1.0, 1.0, 1.0],  # All valid
            [1.0, 1.0, 0.0]   # Last one invalid
        ])
        
        result = compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # Score difference = 6.0 - 3.0 = 3.0 > 0
        # Accuracy difference = 0.8 - 0.3 = 0.5 > 0
        # Consistent direction, accuracy = 1.0
        expected_accuracy = 1.0
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)
    
    def test_extreme_cases(self):
        """Test extreme cases"""
        n_samples = 2
        batch_size = 4
        seq_len = 4
        
        # Test extreme score values
        extreme_scores = torch.tensor([
            [100.0, 100.0, 100.0, 100.0],
            [-100.0, -100.0, -100.0, -100.0],
            [50.0, 50.0, 50.0, 50.0],
            [-50.0, -50.0, -50.0, -50.0],
        ])
        acc = torch.tensor([1.0, 0.0, 0.8, 0.2])
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_accuracy(extreme_scores, acc, response_mask, n_samples)

        # print(accuracy)
        
        expected_accuracy = torch.tensor(1.)
        torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-30, rtol=1e-30)


if __name__ == "__main__":
    unittest.main()
