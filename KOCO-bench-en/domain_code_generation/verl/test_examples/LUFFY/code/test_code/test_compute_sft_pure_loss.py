import unittest
import torch
import sys
import os

# Add the verl directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'luffy', 'verl'))

# Import the actual function we want to test
from verl.mix_src.mix_core_alg import compute_sft_pure_loss


class TestComputeSftPureLoss(unittest.TestCase):
    """Test compute_sft_pure_loss function - Pure SFT loss computation described in LUFFY document Section 3"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_basic_functionality(self):
        """Test basic functionality: compute negative log likelihood loss"""
        batch_size, response_length = 2, 4
        
        # Create log probabilities
        log_prob = torch.tensor([
            [-0.5, -0.3, -0.4, -0.6],
            [-0.2, -0.5, -0.3, -0.4]
        ])
        eos_mask = torch.ones(batch_size, response_length)
        
        # Call the actual function
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # Verify output is a scalar
        self.assertTrue(torch.is_tensor(sft_loss))
        self.assertEqual(sft_loss.dim(), 0)
        self.assertTrue(torch.isfinite(sft_loss))
        
        # Verify loss is positive (because it's negative log probability)
        self.assertGreater(sft_loss.item(), 0.0)
        
        # Manual verification of calculation
        # SFT loss = mean(-log_prob) = mean([0.5, 0.3, 0.4, 0.6, 0.2, 0.5, 0.3, 0.4])
        expected_loss = (0.5 + 0.3 + 0.4 + 0.6 + 0.2 + 0.5 + 0.3 + 0.4) / 8.0
        self.assertAlmostEqual(sft_loss.item(), expected_loss, places=5)
    
    def test_with_eos_mask(self):
        """Test eos_mask application: compute loss only at valid token positions"""
        batch_size, response_length = 2, 5
        
        log_prob = torch.tensor([
            [-0.5, -0.3, -0.4, -0.6, -0.8],
            [-0.2, -0.5, -0.3, -0.4, -0.7]
        ])
        
        # Partial mask
        eos_mask = torch.tensor([
            [1.0, 1.0, 1.0, 0.0, 0.0],  # First 3 valid
            [1.0, 1.0, 1.0, 1.0, 0.0]   # First 4 valid
        ])
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # Verify output
        self.assertTrue(torch.isfinite(sft_loss))
        self.assertGreater(sft_loss.item(), 0.0)
        
        # Manual verification: only compute valid positions
        # Sequence 1: [0.5, 0.3, 0.4] (3 valid)
        # Sequence 2: [0.2, 0.5, 0.3, 0.4] (4 valid)
        # Total 7 valid tokens
        expected_loss = (0.5 + 0.3 + 0.4 + 0.2 + 0.5 + 0.3 + 0.4) / 7.0
        self.assertAlmostEqual(sft_loss.item(), expected_loss, places=5)
    
    def test_uniform_log_prob(self):
        """Test uniform log probability case"""
        batch_size, response_length = 3, 4
        
        # Same log probability at all positions
        log_prob = torch.full((batch_size, response_length), -0.5)
        eos_mask = torch.ones(batch_size, response_length)
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # Verify output
        self.assertTrue(torch.isfinite(sft_loss))
        
        # With uniform log probability, loss should equal -log_prob value
        expected_loss = torch.tensor(0.5)
        torch.testing.assert_close(sft_loss, expected_loss, atol=1e-4, rtol=1e-4)

    
    def test_high_confidence_predictions(self):
        """Test high confidence predictions (log probability close to 0)"""
        batch_size, response_length = 2, 3
        
        # High confidence: log probability close to 0
        log_prob = torch.tensor([
            [-0.01, -0.02, -0.015],
            [-0.03, -0.01, -0.025]
        ])
        eos_mask = torch.ones(batch_size, response_length)
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)

        # print(sft_loss)
        expected_loss = torch.tensor(0.0183)
        torch.testing.assert_close(sft_loss, expected_loss, atol=1e-4, rtol=1e-4)
        
        # High confidence should produce low loss
        self.assertLess(sft_loss.item(), 0.1)
        self.assertGreater(sft_loss.item(), 0.0)
    
    def test_low_confidence_predictions(self):
        """Test low confidence predictions (log probability far from 0)"""
        batch_size, response_length = 2, 3
        
        # Low confidence: log probability far from 0
        log_prob = torch.tensor([
            [-2.0, -3.0, -2.5],
            [-3.5, -2.0, -2.8]
        ])
        eos_mask = torch.ones(batch_size, response_length)
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # print(sft_loss)
        expected_loss = torch.tensor(2.6333)
        torch.testing.assert_close(sft_loss, expected_loss, atol=1e-4, rtol=1e-4)
    
    def test_single_token(self):
        """Test single token case"""
        batch_size, response_length = 1, 1
        
        log_prob = torch.tensor([[-0.7]])
        eos_mask = torch.ones(batch_size, response_length)
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # For single token, loss should equal -log_prob
        expected_loss = 0.7
        self.assertAlmostEqual(sft_loss.item(), expected_loss, places=5)
    
    def test_all_masked(self):
        """Test case where all tokens are masked"""
        batch_size, response_length = 2, 3
        
        log_prob = torch.tensor([
            [-0.5, -0.3, -0.4],
            [-0.2, -0.6, -0.3]
        ])
        eos_mask = torch.zeros(batch_size, response_length)  # All masked
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # print(sft_loss)
        expected_loss = torch.tensor(float('nan'))
        torch.testing.assert_close(sft_loss, expected_loss, equal_nan=True)
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        batch_size, response_length = 4, 6
        
        log_prob = torch.randn(batch_size, response_length) * 0.5 - 1.0
        eos_mask = torch.ones(batch_size, response_length)
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # print(sft_loss)
        expected_loss = torch.tensor(1.1210)
        torch.testing.assert_close(sft_loss, expected_loss, atol=1e-4, rtol=1e-4)
    
    def test_varying_sequence_lengths(self):
        """Test different sequence lengths (implemented via mask)"""
        batch_size, response_length = 3, 5
        
        log_prob = torch.tensor([
            [-0.5, -0.3, -0.4, -0.6, -0.7],
            [-0.2, -0.5, -0.3, -0.4, -0.8],
            [-0.4, -0.6, -0.5, -0.3, -0.9]
        ])
        
        # Different valid lengths
        eos_mask = torch.tensor([
            [1.0, 1.0, 0.0, 0.0, 0.0],  # Length 2
            [1.0, 1.0, 1.0, 1.0, 0.0],  # Length 4
            [1.0, 1.0, 1.0, 0.0, 0.0]   # Length 3
        ])
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # Verify output
        self.assertTrue(torch.isfinite(sft_loss))
        self.assertGreater(sft_loss.item(), 0.0)
        
        # Manual verification
        # Sequence 1: [0.5, 0.3] (2 tokens)
        # Sequence 2: [0.2, 0.5, 0.3, 0.4] (4 tokens)
        # Sequence 3: [0.4, 0.6, 0.5] (3 tokens)
        # Total 9 valid tokens
        expected_loss = (0.5 + 0.3 + 0.2 + 0.5 + 0.3 + 0.4 + 0.4 + 0.6 + 0.5) / 9.0
        self.assertAlmostEqual(sft_loss.item(), expected_loss, places=5)
    
    def test_zero_log_prob(self):
        """Test case with log probability of 0 (perfect prediction)"""
        batch_size, response_length = 2, 3
        
        log_prob = torch.zeros(batch_size, response_length)
        eos_mask = torch.ones(batch_size, response_length)
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # When log probability is 0, loss should be 0
        self.assertAlmostEqual(sft_loss.item(), 0.0, places=5)
    
    def test_negative_values_handling(self):
        """Test correct handling of negative log probability values"""
        batch_size, response_length = 2, 4
        
        # Negative log probability (normal case)
        log_prob = torch.tensor([
            [-1.2, -0.8, -1.5, -0.9],
            [-1.0, -1.3, -0.7, -1.1]
        ])
        eos_mask = torch.ones(batch_size, response_length)
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # Verify loss is positive
        self.assertGreater(sft_loss.item(), 0.0)
        
        # Manual verification
        expected_loss = (1.2 + 0.8 + 1.5 + 0.9 + 1.0 + 1.3 + 0.7 + 1.1) / 8.0
        self.assertAlmostEqual(sft_loss.item(), expected_loss, places=5)


class TestComputeSftPureLossIntegration(unittest.TestCase):
    """Integration tests: test compute_sft_pure_loss in real-world scenarios"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_realistic_training_scenario(self):
        """Test realistic training scenario"""
        batch_size, response_length = 8, 12
        
        # Simulate realistic log probability distribution
        log_prob = torch.randn(batch_size, response_length) * 0.8 - 1.5
        
        # Simulate realistic mask (different sequences have different lengths)
        eos_mask = torch.ones(batch_size, response_length)
        for i in range(batch_size):
            # Randomly mask some subsequent tokens
            mask_start = torch.randint(response_length // 2, response_length, (1,)).item()
            eos_mask[i, mask_start:] = 0.0
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # print(sft_loss)
        expected_loss = torch.tensor(1.4558)
        torch.testing.assert_close(sft_loss, expected_loss, atol=1e-4, rtol=1e-4)
    
    def test_comparison_with_manual_calculation(self):
        """Test consistency with manual calculation"""
        batch_size, response_length = 3, 4
        
        log_prob = torch.tensor([
            [-0.5, -0.3, -0.4, -0.6],
            [-0.2, -0.5, -0.3, -0.4],
            [-0.4, -0.6, -0.5, -0.3]
        ])
        eos_mask = torch.ones(batch_size, response_length)
        
        # Calculate using function
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # Manual calculation
        manual_sft_losses = -log_prob
        manual_sft_loss = (manual_sft_losses * eos_mask).sum() / eos_mask.sum()
        
        # Verify consistency
        torch.testing.assert_close(sft_loss, manual_sft_loss, atol=1e-6, rtol=1e-6)
    
    def test_loss_magnitude_with_different_confidences(self):
        """Test loss magnitude relationship with different confidences"""
        batch_size, response_length = 2, 3
        
        # High confidence
        high_confidence_log_prob = torch.tensor([
            [-0.1, -0.05, -0.08],
            [-0.06, -0.09, -0.07]
        ])
        
        # Low confidence
        low_confidence_log_prob = torch.tensor([
            [-2.0, -1.5, -1.8],
            [-1.6, -1.9, -1.7]
        ])
        
        eos_mask = torch.ones(batch_size, response_length)
        
        high_conf_loss = compute_sft_pure_loss(high_confidence_log_prob, eos_mask)
        low_conf_loss = compute_sft_pure_loss(low_confidence_log_prob, eos_mask)
        
        # print(high_conf_loss)
        # print(low_conf_loss)
        # Low confidence should produce higher loss
        expected_loss = torch.tensor(0.0750)
        torch.testing.assert_close(high_conf_loss, expected_loss, atol=1e-4, rtol=1e-4)
        expected_loss = torch.tensor(1.7500)
        torch.testing.assert_close(low_conf_loss, expected_loss, atol=1e-4, rtol=1e-4)
    
    def test_gradient_flow(self):
        """Test gradient flow (ensure loss can backpropagate)"""
        batch_size, response_length = 2, 4
        
        # Create log probability that requires gradient
        log_prob = torch.randn(batch_size, response_length, requires_grad=True)
        eos_mask = torch.ones(batch_size, response_length)
        
        sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
        
        # Backpropagation
        sft_loss.backward()

        # print(log_prob.grad)
        expected_grad = torch.tensor([[-0.1250, -0.1250, -0.1250, -0.1250],
        [-0.1250, -0.1250, -0.1250, -0.1250]])
        torch.testing.assert_close(log_prob.grad, expected_grad, atol=1e-4, rtol=1e-4)
    
    def test_multi_task_learning_scenario(self):
        """Test multi-task learning scenario: SFT as auxiliary loss"""
        batch_size, response_length = 4, 6
        
        # Simulate log probability of off-policy samples
        off_policy_log_prob = torch.tensor([[-0.0365, -0.2564, -0.5496, -2.0528, -0.6608, -1.6173],
        [-1.0215, -1.8023, -0.8221, -1.3433, -1.2467, -0.8793],
        [-1.5555, -0.9542, -2.1585, -1.1084, -1.1549, -1.1979],
        [-0.5983, -1.3108, -1.2960, -1.0315, -1.4143, -0.8346]])

        # print(off_policy_log_prob)
        eos_mask = torch.ones(batch_size, response_length)
        
        # Calculate SFT loss
        sft_loss = compute_sft_pure_loss(off_policy_log_prob, eos_mask)
        
        # print(sft_loss)
        expected_loss = torch.tensor(1.1210)
        torch.testing.assert_close(sft_loss, expected_loss, atol=1e-4, rtol=1e-4)
        
        # Simulate combination with other losses
        mock_rl_loss = torch.tensor(0.5)
        combined_loss = sft_loss + 0.1 * mock_rl_loss
        
        self.assertTrue(torch.isfinite(combined_loss))


if __name__ == "__main__":
    unittest.main()
