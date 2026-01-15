import unittest
import torch
import numpy as np
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the actual function we want to test
from pacs.pacs_core_algos import compute_weight


class TestComputeWeight(unittest.TestCase):
    """Test compute_weight function - sample weight calculation described in PACS document Section 2"""
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_compute_weight_question_mode_balanced(self):
        """Test question mode: case with positive and negative sample diversity"""
        # Create a set of diverse labels (both 0 and 1)
        rollout_n = 4
        score = torch.tensor([
            [1.0],  # Positive sample
            [0.0],  # Negative sample
            [1.0],  # Positive sample
            [0.0]   # Negative sample
        ])
        
        # Call the actual compute_weight function
        weight = compute_weight(score, rollout_n, mode="question")

        # print(weight)

        expected_weight = torch.tensor([[1.],
        [1.],
        [1.],
        [1.]])
        torch.testing.assert_close(weight, expected_weight, atol=1e-5, rtol=1e-5)
        
        # Verify output shape and properties
        self.assertEqual(weight.shape, (4, 1))
        self.assertTrue(torch.is_tensor(weight))
        self.assertTrue(torch.isfinite(weight).all())
        self.assertTrue((weight >= 0).all())
        
        # Verify balanced weights: when positive and negative samples are equal, weights should be the same
        positive_weights = weight[score.squeeze(-1) == 1]
        negative_weights = weight[score.squeeze(-1) == 0]
        
        # All positive sample weights should be the same
        self.assertTrue(torch.allclose(positive_weights, positive_weights[0]))
        # All negative sample weights should be the same
        self.assertTrue(torch.allclose(negative_weights, negative_weights[0]))
    
    def test_compute_weight_question_mode_imbalanced(self):
        """Test question mode: case with imbalanced positive and negative samples"""
        rollout_n = 5
        # 3 positive samples, 2 negative samples
        score = torch.tensor([
            [1.0],
            [1.0],
            [1.0],
            [0.0],
            [0.0]
        ])
        
        weight = compute_weight(score, rollout_n, mode="question")

        # print(weight)

        expected_weight = torch.tensor([[0.8333],
        [0.8333],
        [0.8333],
        [1.2500],
        [1.2500]])
        torch.testing.assert_close(weight, expected_weight, atol=1e-4, rtol=1e-4)
        # Verify output
        self.assertEqual(weight.shape, (5, 1))
        self.assertTrue(torch.isfinite(weight).all())
        
        # Negative samples (minority class) should get higher weights
        positive_weights = weight[score.squeeze(-1) == 1]
        negative_weights = weight[score.squeeze(-1) == 0]
        
        avg_positive_weight = positive_weights.mean()
        avg_negative_weight = negative_weights.mean()
        
        # Negative sample weights should be greater than positive sample weights
        self.assertGreater(avg_negative_weight.item(), avg_positive_weight.item())
    
    def test_compute_weight_question_mode_no_diversity(self):
        """Test question mode: case with no diversity (all same class)"""
        rollout_n = 4
        
        # Test all positive samples
        score_all_positive = torch.ones(4, 1)
        weight_all_positive = compute_weight(score_all_positive, rollout_n, mode="question")
        
        # When there's no diversity, weights should all be 1
        expected_weight = torch.ones(4, 1)
        torch.testing.assert_close(weight_all_positive, expected_weight)
        
        # Test all negative samples
        score_all_negative = torch.zeros(4, 1)
        weight_all_negative = compute_weight(score_all_negative, rollout_n, mode="question")
        
        # When there's no diversity, weights should all be 1
        torch.testing.assert_close(weight_all_negative, expected_weight)
    
    def test_compute_weight_only_positive_mode(self):
        """Test only_positive mode: keep only positive samples"""
        rollout_n = 4
        score = torch.tensor([
            [1.0],
            [0.0],
            [1.0],
            [0.0]
        ])
        
        weight = compute_weight(score, rollout_n, mode="only_positive")
        
        # Verify output shape
        self.assertEqual(weight.shape, (4, 1))
        
        # Verify positive sample weights are 1.0
        self.assertEqual(weight[0].item(), 1.0)
        self.assertEqual(weight[2].item(), 1.0)
        
        # Verify negative sample weights are 0.0
        self.assertEqual(weight[1].item(), 0.0)
        self.assertEqual(weight[3].item(), 0.0)
    
    def test_compute_weight_only_negative_mode(self):
        """Test only_negative mode: keep only negative samples"""
        rollout_n = 4
        score = torch.tensor([
            [1.0],
            [0.0],
            [1.0],
            [0.0]
        ])
        
        weight = compute_weight(score, rollout_n, mode="only_negative")
        
        # Verify output shape
        self.assertEqual(weight.shape, (4, 1))
        
        # Verify negative sample weights are 1.0
        self.assertEqual(weight[1].item(), 1.0)
        self.assertEqual(weight[3].item(), 1.0)
        
        # Verify positive sample weights are 0.0
        self.assertEqual(weight[0].item(), 0.0)
        self.assertEqual(weight[2].item(), 0.0)
    
    def test_compute_weight_multiple_groups(self):
        """Test weight calculation for multiple groups of samples"""
        rollout_n = 3
        # Two groups of samples, 3 in each group
        score = torch.tensor([
            [1.0], [0.0], [1.0],  # Group 1: 2 positive, 1 negative
            [0.0], [0.0], [1.0]   # Group 2: 1 positive, 2 negative
        ])
        
        weight = compute_weight(score, rollout_n, mode="question")
        
        # print(weight)

        expected_weight = torch.tensor([[0.7500],
        [1.5000],
        [0.7500],
        [0.7500],
        [0.7500],
        [1.5000]])
        torch.testing.assert_close(weight, expected_weight, atol=1e-4, rtol=1e-4)
        # Verify output shape
        self.assertEqual(weight.shape, (6, 1))
        self.assertTrue(torch.isfinite(weight).all())
        
        # Group 1: negative samples (minority) should have higher weights
        group1_weights = weight[:3]
        group1_negative_weight = group1_weights[1]  # index 1 is negative sample
        group1_positive_weights = torch.stack([group1_weights[0], group1_weights[2]])
        self.assertGreater(group1_negative_weight.item(), group1_positive_weights.mean().item())
        
        # Group 2: positive samples (minority) should have higher weights
        group2_weights = weight[3:]
        group2_positive_weight = group2_weights[2]  # index 2 is positive sample
        group2_negative_weights = torch.stack([group2_weights[0], group2_weights[1]])
        self.assertGreater(group2_positive_weight.item(), group2_negative_weights.mean().item())
    
    def test_compute_weight_edge_cases(self):
        """Test edge cases"""
        # Test case 1: single sample
        single_score = torch.tensor([[1.0]])
        single_weight = compute_weight(single_score, rollout_n=1, mode="question")
        expected_single = torch.tensor([[1.0]])
        torch.testing.assert_close(single_weight, expected_single)
        
        # Test case 2: large batch
        large_batch_size = 100
        large_rollout_n = 10
        large_score = torch.randint(0, 2, (large_batch_size, 1)).float()
        large_weight = compute_weight(large_score, large_rollout_n, mode="question")

        # print(large_weight)

        expected_weight = torch.tensor([[0.7143],
        [1.6667],
        [0.7143],
        [0.7143],
        [0.7143],
        [1.6667],
        [0.7143],
        [0.7143],
        [0.7143],
        [1.6667],
        [0.8333],
        [0.8333],
        [0.8333],
        [0.8333],
        [1.2500],
        [0.8333],
        [1.2500],
        [1.2500],
        [1.2500],
        [0.8333],
        [0.5556],
        [5.0000],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.8333],
        [0.8333],
        [1.2500],
        [1.2500],
        [1.2500],
        [0.8333],
        [1.2500],
        [0.8333],
        [0.8333],
        [0.8333],
        [1.6667],
        [1.6667],
        [0.7143],
        [0.7143],
        [0.7143],
        [0.7143],
        [0.7143],
        [1.6667],
        [0.7143],
        [0.7143],
        [0.8333],
        [1.2500],
        [0.8333],
        [1.2500],
        [0.8333],
        [1.2500],
        [1.2500],
        [0.8333],
        [0.8333],
        [0.8333],
        [0.8333],
        [0.8333],
        [0.8333],
        [0.8333],
        [0.8333],
        [1.2500],
        [1.2500],
        [0.8333],
        [1.2500],
        [1.2500],
        [0.7143],
        [0.7143],
        [1.6667],
        [0.7143],
        [1.6667],
        [0.7143],
        [0.7143],
        [0.7143],
        [1.6667],
        [0.7143],
        [1.0000],
        [1.0000],
        [1.0000],
        [1.0000],
        [1.0000],
        [1.0000],
        [1.0000],
        [1.0000],
        [1.0000],
        [1.0000],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [0.5556],
        [5.0000]])
        torch.testing.assert_close(large_weight, expected_weight, atol=1e-4, rtol=1e-4)
    
    def test_compute_weight_invalid_mode(self):
        """Test invalid mode parameter"""
        score = torch.tensor([[1.0], [0.0]])
        
        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            compute_weight(score, rollout_n=2, mode="invalid_mode")
        
        self.assertIn("Invalid weight mode", str(context.exception))
    
    def test_compute_weight_batch_consistency(self):
        """Test batch processing consistency"""
        rollout_n = 4
        score = torch.tensor([
            [1.0], [0.0], [1.0], [0.0],  # Group 1
            [1.0], [1.0], [0.0], [0.0]   # Group 2
        ])
        
        # Batch calculation
        weight_batch = compute_weight(score, rollout_n, mode="question")
        
        # Group calculation
        weight_group1 = compute_weight(score[:4], rollout_n, mode="question")
        weight_group2 = compute_weight(score[4:], rollout_n, mode="question")
        
        # Verify batch calculation is consistent with group calculation
        torch.testing.assert_close(weight_batch[:4], weight_group1)
        torch.testing.assert_close(weight_batch[4:], weight_group2)


class TestComputeWeightIntegration(unittest.TestCase):
    """Integration tests: test compute_weight in real-world scenarios"""
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_weight_mode_comparison(self):
        """Compare differences between three weight modes"""
        rollout_n = 4
        score = torch.tensor([
            [1.0],
            [0.0],
            [1.0],
            [0.0]
        ])
        
        # Calculate weights for three modes
        weight_question = compute_weight(score, rollout_n, mode="question")
        weight_only_positive = compute_weight(score, rollout_n, mode="only_positive")
        weight_only_negative = compute_weight(score, rollout_n, mode="only_negative")

        # print(weight_question)
        # print(weight_only_positive)
        # print(weight_only_negative)

        expected_weight_question = torch.tensor([[1.],
        [1.],
        [1.],
        [1.]])
        expected_weight_only_positive = torch.tensor([[1.],
        [0.],
        [1.],
        [0.]])
        expected_weight_only_negative = torch.tensor([[0.],
        [1.],
        [0.],
        [1.]])
        torch.testing.assert_close(weight_question, expected_weight_question, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(weight_only_positive, expected_weight_only_positive, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(weight_only_negative, expected_weight_only_negative, atol=1e-4, rtol=1e-4)
        
        # Verify three modes produce different results
        self.assertFalse(torch.equal(weight_question, weight_only_positive))
        self.assertFalse(torch.equal(weight_question, weight_only_negative))
        self.assertFalse(torch.equal(weight_only_positive, weight_only_negative))
        
        # Verify only_positive and only_negative are complementary
        combined_weight = weight_only_positive + weight_only_negative
        expected_combined = torch.ones(4, 1)
        torch.testing.assert_close(combined_weight, expected_combined)
    
    def test_weight_with_loss_computation(self):
        """Test application of weights in loss computation"""
        rollout_n = 4
        score = torch.tensor([
            [1.0],
            [0.0],
            [1.0],
            [0.0]
        ])
        
        # Calculate weights
        weight = compute_weight(score, rollout_n, mode="question")
        
        # print(weight)

        expected_weight = torch.tensor([[1.],
        [1.],
        [1.],
        [1.]])
        torch.testing.assert_close(weight, expected_weight, atol=1e-4, rtol=1e-4)
        # Simulate loss values
        loss_per_sample = torch.tensor([
            [0.5],
            [0.8],
            [0.3],
            [0.9]
        ])
        
        # Apply weights
        weighted_loss = loss_per_sample * weight
        
        # Verify weighted loss
        self.assertEqual(weighted_loss.shape, (4, 1))
        self.assertTrue(torch.isfinite(weighted_loss).all())
        self.assertTrue((weighted_loss >= 0).all())
        
        # Verify the impact of weights on loss
        # Weighted loss for minority class samples should be larger
        total_weighted_loss = weighted_loss.sum()
        self.assertGreater(total_weighted_loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
