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
from prime.prime_core_algos import compute_dpo_abs_accuracy


class TestDPOAbsAccuracy(unittest.TestCase):
    """Test compute_dpo_abs_accuracy function - DPO absolute accuracy described in Section 3 of the documentation"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_compute_dpo_abs_accuracy_basic(self):
        """Test basic functionality of DPO absolute accuracy calculation"""
        batch_size = 4
        seq_len = 10
        n_samples = 2
        
        token_level_scores = torch.tensor([[0.1,0.3,1.0,-0.1,0.7,0.5,0.1,0.3,0.6,-0.1], [0.2,0.3,0.2,-0.4,0.9,-0.1,0.2,0.5,0.1,0.3], [-0.1,0.3,0.6,-0.1,0.2,0.7,1.0,0.2,0.5,0.1], [0.1,0.8,0.6,-0.1,0.2,0.5,0.1,0.5,0.1,0.3]])
        acc = torch.tensor([0.8, 0.1, 1.0, 0.0])

        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # print(accuracy)
        expected_accuracy = torch.tensor(0.5000)
        torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-30, rtol=1e-30)
        # Verify output
        # self.assertEqual(accuracy.dim(), 0)
        # self.assertTrue(torch.is_tensor(accuracy))
        # self.assertTrue(0 <= accuracy.item() <= 1)
        # self.assertFalse(torch.isnan(accuracy))
    
    def test_perfect_prediction(self):
        """Test perfect prediction case"""
        batch_size = 4
        seq_len = 6
        n_samples = 2
        
        # Create clear score-accuracy correspondence
        token_level_scores = torch.tensor([
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],    # High scores -> should predict as positive
            [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0],  # Low scores -> should predict as negative
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],    # Positive scores -> should predict as positive
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],  # Negative scores -> should predict as negative
        ])
        acc = torch.tensor([1.0, 0.0, 1.0, 0.0])  # Completely consistent with predictions
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # print(accuracy)
        # Perfect prediction should achieve 100% accuracy
        expected_accuracy = torch.tensor(1.0000)
        torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-30, rtol=1e-30)
    
    def test_opposite_prediction(self):
        """Test opposite prediction case"""
        batch_size = 4
        seq_len = 4
        n_samples = 2
        
        # Create scores opposite to true labels
        token_level_scores = torch.tensor([
            [2.0, 2.0, 2.0, 2.0],    # High scores but label is 0
            [-2.0, -2.0, -2.0, -2.0],  # Low scores but label is 1
            [1.0, 1.0, 1.0, 1.0],    # Positive scores but label is 0
            [-1.0, -1.0, -1.0, -1.0],  # Negative scores but label is 1
        ])
        acc = torch.tensor([0.0, 1.0, 0.0, 1.0])  # Opposite to predictions
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # Completely opposite predictions should achieve 0% accuracy
        self.assertAlmostEqual(accuracy.item(), 0.0, places=5)
    
    def test_binary_classification_logic(self):
        """Test correctness of binary classification logic"""
        batch_size = 6
        seq_len = 3
        n_samples = 2
        
        # Test sign function classification logic
        token_level_scores = torch.tensor([
            [1.0, 1.0, 1.0],     # Sequence score = 3.0 > 0 -> predict 1
            [0.5, 0.5, 0.5],     # Sequence score = 1.5 > 0 -> predict 1
            [-1.0, -1.0, -1.0],  # Sequence score = -3.0 < 0 -> predict 0
            [-0.5, -0.5, -0.5],  # Sequence score = -1.5 < 0 -> predict 0
            [0.0, 0.0, 0.0],     # Sequence score = 0.0 = 0 -> predict 0 (sign function)
            [1.0, -1.0, 0.0],    # Sequence score = 0.0 = 0 -> predict 0
        ])
        acc = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # Corresponding true labels
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # print(accuracy)
        expected_accuracy = torch.tensor(0.666666666666666666666666666667)
        torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-30, rtol=1e-30)
        # Verify classification logic correctness, accuracy should be reasonably high
        # self.assertTrue(accuracy.item() >= 0.5, "Accuracy should be reasonable for correct classification logic")
    
    def test_different_n_samples(self):
        """Test impact of different n_samples parameters"""
        seq_len = 5
        accuracies = []
        for n_samples in [1, 2, 4]:
            batch_size = n_samples * 4  # Ensure divisible by n_samples
            
            with self.subTest(n_samples=n_samples):
                token_level_scores = torch.randn(batch_size, seq_len)
                acc = torch.randint(0, 2, (batch_size,)).float()
                response_mask = torch.ones(batch_size, seq_len)
                
                accuracy = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
                accuracies.append(accuracy.item())
                # Verify output is within valid range
                # self.assertTrue(0 <= accuracy.item() <= 1)
                # self.assertFalse(torch.isnan(accuracy))

        # print(accuracies)
        expected_accuracies = [0.5, 0.5, 0.6875]
        torch.testing.assert_close(accuracies, expected_accuracies, atol=1e-30, rtol=1e-30)
    
    def test_response_mask_impact(self):
        """Test impact of response_mask on absolute accuracy"""
        batch_size = 4
        seq_len = 8
        n_samples = 2
        
        token_level_scores = torch.ones(batch_size, seq_len)  # All scores are positive
        acc = torch.tensor([1.0, 1.0, 0.0, 0.0])
        
        # Test fully valid mask
        full_mask = torch.ones(batch_size, seq_len)
        accuracy_full = compute_dpo_abs_accuracy(token_level_scores, acc, full_mask, n_samples)
        # print(accuracy_full)
        # Test partial mask (affects sequence score aggregation)
        partial_mask = torch.ones(batch_size, seq_len)
        partial_mask[:, -4:] = 0  # Second half masked out
        accuracy_partial = compute_dpo_abs_accuracy(token_level_scores, acc, partial_mask, n_samples)
        # print(accuracy_partial)
        # Both cases should produce valid accuracy
        # self.assertTrue(0 <= accuracy_full.item() <= 1)
        # self.assertTrue(0 <= accuracy_partial.item() <= 1)
        # self.assertFalse(torch.isnan(accuracy_full))
        # self.assertFalse(torch.isnan(accuracy_partial))
        
        # Since all are positive, first two samples (acc=1.0) should predict correctly, last two (acc=0.0) incorrectly
        # For full_mask: accuracy should be 2/4 = 0.5
        expected_accuracy_full = torch.tensor(0.5000)
        torch.testing.assert_close(accuracy_full, expected_accuracy_full, atol=1e-30, rtol=1e-30)
        
        # For partial_mask: sequence scores are still positive (though smaller), so result should be the same
        expected_accuracy_partial = torch.tensor(0.5000)
        torch.testing.assert_close(accuracy_partial, expected_accuracy_partial, atol=1e-30, rtol=1e-30)
    
    def test_zero_scores_edge_case(self):
        """Test edge case with zero scores"""
        batch_size = 4
        seq_len = 3
        n_samples = 2
        
        # All scores are zero
        token_level_scores = torch.zeros(batch_size, seq_len)
        acc = torch.tensor([1.0, 0.0, 1.0, 0.0])
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # print(accuracy)
        expected_accuracy = torch.tensor(0.)
        torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-30, rtol=1e-30)
        # Zero scores are edge case, verify function handles correctly
    
    def test_score_aggregation_correctness(self):
        """Test correctness of score aggregation"""
        batch_size = 2
        seq_len = 4
        n_samples = 1
        
        # Clear score aggregation test
        token_level_scores = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],    # Aggregated score = 10.0 > 0 -> predict 1
            [-1.0, -2.0, -3.0, -4.0]  # Aggregated score = -10.0 < 0 -> predict 0
        ])
        acc = torch.tensor([1.0, 0.0])  # Consistent with predictions
        response_mask = torch.ones(batch_size, seq_len)
        
        accuracy = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # Predictions are completely correct
        self.assertAlmostEqual(accuracy.item(), 1.0, places=5)
    
    def test_consistency_with_different_seeds(self):
        """Test consistency with different random seeds"""
        batch_size = 6
        seq_len = 4
        n_samples = 2
        
        # Fixed input data
        token_level_scores = torch.tensor([
            [1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [2.0, 2.0, 2.0, 2.0],
            [-2.0, -2.0, -2.0, -2.0],
            [0.5, 0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5, -0.5],
        ])
        acc = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        response_mask = torch.ones(batch_size, seq_len)
        
        # Test consistency across multiple computations
        accuracies = []
        for seed in [42, 123, 456]:
            torch.manual_seed(seed)
            accuracy = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
            accuracies.append(accuracy.item())
        
            # print(accuracies)
        expected_accuracies = [1.0, 1.0, 1.0]
        torch.testing.assert_close(accuracies, expected_accuracies, atol=1e-30, rtol=1e-30)
        # Since input is deterministic, results should be completely consistent
        # self.assertTrue(all(abs(acc - accuracies[0]) < 1e-6 for acc in accuracies))

    def test_input_output_samples(self):
        """Test specific input-output examples"""
        # Test case 1: Simple binary classification
        token_level_scores = torch.tensor([
            [2.0, 3.0],   # Sequence score: 5.0 > 0 -> predicted label: 1
            [-1.0, -2.0]  # Sequence score: -3.0 < 0 -> predicted label: 0
        ])
        acc = torch.tensor([1.0, 0.0])  # True labels consistent with predictions
        response_mask = torch.ones(2, 2)
        n_samples = 1
        
        result = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        # print(result)
        # Manual calculation verification:
        # Sequence scores = [5.0, -3.0]
        # Predicted labels = sign([5.0, -3.0]) = [1, -1]  
        # True label conversion = acc * 2 - 1 = [1.0, 0.0] * 2 - 1 = [1.0, -1.0]
        # Match status = [1==1, -1==-1] = [True, True]
        # Accuracy = 2/2 = 1.0
        expected_accuracy = 1.0
        self.assertAlmostEqual(result.item(), expected_accuracy, places=5)
        
        # Test case 2: Completely incorrect predictions
        token_level_scores = torch.tensor([
            [1.0, 1.0, 1.0],   # Sequence score: 3.0 > 0 -> predicted label: 1
            [-2.0, -1.0, 0.0]  # Sequence score: -3.0 < 0 -> predicted label: 0
        ])
        acc = torch.tensor([0.0, 1.0])  # True labels opposite to predictions
        response_mask = torch.tensor([
            [1.0, 1.0, 1.0],  
            [1.0, 1.0, 0.0]   # Last token masked
        ])
        
        result = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # Manual calculation verification:
        # Valid sequence scores = [3.0, -3.0]
        # Predicted labels = sign([3.0, -3.0]) = [1, -1]
        # True label conversion = [0.0, 1.0] * 2 - 1 = [-1.0, 1.0]
        # Match status = [1==-1, -1==1] = [False, False]
        # Accuracy = 0/2 = 0.0
        expected_accuracy = 0.0
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)
        
        # Test case 3: Mixed case
        token_level_scores = torch.tensor([
            [1.5, 2.5],    # Sequence score: 4.0 > 0 -> predict: 1
            [-0.5, -1.5],  # Sequence score: -2.0 < 0 -> predict: 0  
            [2.0, -1.0],   # Sequence score: 1.0 > 0 -> predict: 1
            [-3.0, 1.0]    # Sequence score: -2.0 < 0 -> predict: 0
        ])
        acc = torch.tensor([1.0, 0.0, 0.0, 1.0])  # First two correct, last two incorrect
        response_mask = torch.ones(4, 2)
        
        result = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # Manual calculation verification:
        # Sequence scores = [4.0, -2.0, 1.0, -2.0]
        # Predicted labels = sign([4.0, -2.0, 1.0, -2.0]) = [1, -1, 1, -1]
        # True label conversion = [1.0, 0.0, 0.0, 1.0] * 2 - 1 = [1.0, -1.0, -1.0, 1.0]
        # Match status = [1==1, -1==-1, 1==-1, -1==1] = [True, True, False, False]
        # Accuracy = 2/4 = 0.5
        expected_accuracy = 0.5
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)

    def test_zero_boundary_cases(self):
        """Test specific input-output for zero boundary cases"""
        # Test case 1: Zero score case
        token_level_scores = torch.tensor([
            [0.0, 0.0, 0.0],  # Sequence score: 0.0, sign(0.0) = 0
            [1.0, -1.0, 0.0],      # Sequence score: 0.0, sign(0.0) = 0
            [0.5, -0.5, 0.0]  # Sequence score: 0.0, sign(0.0) = 0
        ])
        acc = torch.tensor([0.0, 1.0, 0.0])
        response_mask = torch.ones(3, 3)
        response_mask[1, 2] = 0  # Second sample has only two valid tokens
        response_mask[2, 2] = 1  # Third sample has all three tokens valid
        n_samples = 1
        
        result = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # Manual calculation verification:
        # Valid sequence scores = [0.0, 0.0, 0.0]
        # Predicted labels = sign([0.0, 0.0, 0.0]) = [0, 0, 0] (in PyTorch, sign(0)=0)
        # True label conversion = [0.0, 1.0, 0.0] * 2 - 1 = [-1.0, 1.0, -1.0]
        # Match status = [0==-1, 0==1, 0==-1] = [False, False, False]
        # Accuracy = 0/3 = 0.0
        expected_accuracy = 0.0
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)
        
        # Test case 2: Tiny positive/negative values
        token_level_scores = torch.tensor([
            [0.001],      # Sequence score: 0.001 > 0 -> predict: 1
            [-0.001],     # Sequence score: -0.001 < 0 -> predict: 0
            [0.0001],     # Sequence score: 0.0001 > 0 -> predict: 1
            [-0.0001]     # Sequence score: -0.0001 < 0 -> predict: 0
        ])
        acc = torch.tensor([1.0, 0.0, 1.0, 0.0])  # Consistent with predictions
        response_mask = torch.ones(4, 1)
        
        result = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # Manual calculation verification:
        # Sequence scores = [0.001, -0.001, 0.0001, -0.0001]
        # Predicted labels = sign([0.001, -0.001, 0.0001, -0.0001]) = [1, -1, 1, -1]
        # True label conversion = [1.0, 0.0, 1.0, 0.0] * 2 - 1 = [1.0, -1.0, 1.0, -1.0]
        # Match status = [1==1, -1==-1, 1==1, -1==-1] = [True, True, True, True]
        # Accuracy = 4/4 = 1.0
        expected_accuracy = 1.0
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)

    def test_mask_effect_input_output(self):
        """Test specific input-output for mask effect"""
        # Test case: Impact of mask on sequence score aggregation
        token_level_scores = torch.tensor([
            [3.0, 2.0, 1.0, -10.0],  # If all valid: sequence score = -4.0 < 0 -> predict: 0
                                      # If last masked: sequence score = 6.0 > 0 -> predict: 1
            [1.0, 1.0, 1.0, -5.0]    # If all valid: sequence score = -2.0 < 0 -> predict: 0
                                      # If last masked: sequence score = 3.0 > 0 -> predict: 1
        ])
        acc = torch.tensor([1.0, 1.0])  # True labels are both 1
        
        # Test fully valid mask
        full_mask = torch.ones(2, 4)
        result_full = compute_dpo_abs_accuracy(token_level_scores, acc, full_mask, 1)
        
        # Manual calculation verification (full mask):
        # Sequence scores = [-4.0, -2.0]
        # Predicted labels = sign([-4.0, -2.0]) = [-1, -1]
        # True label conversion = [1.0, 1.0] * 2 - 1 = [1.0, 1.0]
        # Match status = [-1==1, -1==1] = [False, False]
        # Accuracy = 0/2 = 0.0
        expected_full = 0.0
        self.assertAlmostEqual(result_full.item(), expected_full, places=30)
        
        # Test partial mask (last token masked)
        partial_mask = torch.ones(2, 4)
        partial_mask[:, -1] = 0  # Last token masked
        result_partial = compute_dpo_abs_accuracy(token_level_scores, acc, partial_mask, 1)
        
        # Manual calculation verification (partial mask):
        # Valid sequence scores = [6.0, 3.0]
        # Predicted labels = sign([6.0, 3.0]) = [1, 1]
        # True label conversion = [1.0, 1.0] * 2 - 1 = [1.0, 1.0]
        # Match status = [1==1, 1==1] = [True, True]
        # Accuracy = 2/2 = 1.0
        expected_partial = 1.0
        self.assertAlmostEqual(result_partial.item(), expected_partial, places=30)

    def test_batch_computation_input_output(self):
        """Test specific input-output for batch computation"""
        # Test case: Large batch data
        token_level_scores = torch.tensor([
            [1.0, 1.0],    # Sample 1: 2.0 > 0 -> predict: 1
            [-1.0, -1.0],  # Sample 2: -2.0 < 0 -> predict: 0
            [2.0, 0.0],    # Sample 3: 2.0 > 0 -> predict: 1
            [0.0, -3.0],   # Sample 4: -3.0 < 0 -> predict: 0
            [0.5, 0.5],    # Sample 5: 1.0 > 0 -> predict: 1
            [-0.8, -0.2]   # Sample 6: -1.0 < 0 -> predict: 0
        ])
        acc = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        # Expected matches: [True, True, False, False, True, True] = 4/6
        response_mask = torch.ones(6, 2)
        n_samples = 1
        
        result = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples)
        
        # Manual calculation verification:
        # Sequence scores = [2.0, -2.0, 2.0, -3.0, 1.0, -1.0]
        # Predicted labels = sign([2.0, -2.0, 2.0, -3.0, 1.0, -1.0]) = [1, -1, 1, -1, 1, -1]
        # True label conversion = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0] * 2 - 1 = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0]
        # Match status = [1==1, -1==-1, 1==-1, -1==1, 1==1, -1==-1] = [T, T, F, F, T, T]
        # Accuracy = 4/6 = 0.6667
        expected_accuracy = 0.6666666865348816
        self.assertAlmostEqual(result.item(), expected_accuracy, places=30)
        
        # Test case: n_samples > 1 situation
        result_grouped = compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples=2)
        
        # When n_samples=2, the function still calculates accuracy for all samples, should not affect result
        # (because this function does not use n_samples parameter for grouped calculation)
        self.assertAlmostEqual(result_grouped.item(), expected_accuracy, places=30)


if __name__ == "__main__":
    unittest.main()
 