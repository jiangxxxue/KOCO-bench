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
from prime.prime_core_algos import compute_ce_dpo_loss_rm


class TestCEDPOLoss(unittest.TestCase):
    """Test compute_ce_dpo_loss function - Cross-Entropy DPO loss calculation described in Section 2 of the documentation"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_compute_ce_dpo_loss_rm_basic(self):
        """Test basic functionality of Cross-Entropy DPO loss calculation"""
        batch_size = 4
        seq_len = 10
        
        token_level_scores = torch.tensor([[0.1,0.3,0.6,-0.1,0.2,0.5,0.1,0.3,0.6,-0.1], [0.1,0.3,0.6,-0.4,0.9,-0.1,0.2,0.5,0.1,0.3], [0.1,0.3,0.6,-0.1,0.2,0.7,1.0,0.2,0.5,0.1], [0.1,0.3,0.6,-0.1,0.2,0.5,0.1,0.5,0.1,0.3]])
        acc = torch.tensor([0.8, 0.3, 1.0, 0.0])
        response_mask = torch.ones(batch_size, seq_len)
        beta = 0.1
        
        loss = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta)
        
        # print(f"loss: {loss.item():.30f}")
        expected_loss = torch.tensor(0.684428155422210693359375000000)
        # Verify output is a scalar
        # self.assertEqual(loss.dim(), 0)
        # self.assertTrue(torch.is_tensor(loss))
        # self.assertFalse(torch.isnan(loss))
        # self.assertFalse(torch.isinf(loss))
        torch.testing.assert_close(loss, expected_loss, atol=1e-30, rtol=1e-30)


    def test_ce_dpo_loss_mathematical_properties(self):
        """Test mathematical properties and computational correctness of CE DPO loss"""
        # Simple deterministic test case
        token_level_scores = torch.tensor([[1.0, 2.0], [-1.0, -2.0]])
        acc = torch.tensor([1.0, 0.0])
        response_mask = torch.ones(2, 2)
        beta = 1.0
        
        loss = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta)
        
        # print(loss)
        # print(f"loss: {loss.item():.30f}")
        expected_loss = torch.tensor(0.048587348312139511108398437500)
        # Manually calculate expected loss to verify implementation
        # seq_scores = (token_level_scores * response_mask).sum(dim=1) * beta  # [3.0, -3.0]
        # probs = torch.sigmoid(seq_scores)  # sigmoid([3.0, -3.0])
        # expected_loss = torch.nn.functional.binary_cross_entropy(probs, acc)
        
        # self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)
        torch.testing.assert_close(loss, expected_loss, atol=1e-30, rtol=1e-30)
    
    def test_different_beta_values(self):
        """Test the impact of different beta parameter values"""
        batch_size = 2
        seq_len = 4
        
        token_level_scores = torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
        acc = torch.tensor([1.0, 0.0])
        response_mask = torch.ones(batch_size, seq_len)
        
        # Test different beta values
        betas = [0.01, 0.1, 1.0, 10.0]
        losses = []
        
        for beta in betas:
            loss = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta)
            losses.append(loss.item())
            
            
            # Verify all losses are valid
            self.assertFalse(torch.isnan(loss))
            self.assertFalse(torch.isinf(loss))
            self.assertGreaterEqual(loss.item(), 0.0)  # BCE loss should be non-negative
        # print(f"losses: {losses[3]:.30f}")
        expected_losses = [0.703647077083587646484375000000, 0.842057943344116210937500000000, 4.009264469146728515625000000000, 50.0]
        torch.testing.assert_close(losses, expected_losses, atol=1e-30, rtol=1e-30)
    
    def test_response_mask_effect(self):
        """Test the effect of response_mask on loss calculation"""
        batch_size = 2
        seq_len = 4
        
        token_level_scores = torch.ones(batch_size, seq_len)
        acc = torch.tensor([1.0, 0.0])
        beta = 0.1
        
        # Test fully valid mask
        full_mask = torch.ones(batch_size, seq_len)
        loss_full = compute_ce_dpo_loss_rm(token_level_scores, acc, full_mask, beta)
        
        # print(f"loss_full: {loss_full.item():.30f}")
        expected_loss_full = torch.tensor(0.713015258312225341796875000000)
        # Test partially valid mask
        partial_mask = torch.ones(batch_size, seq_len)
        partial_mask[:, -1] = 0  # Last position masked out
        loss_partial = compute_ce_dpo_loss_rm(token_level_scores, acc, partial_mask, beta)
        
        # print(f"loss_partial: {loss_partial.item():.30f}")
        expected_loss_partial = torch.tensor(0.704355239868164062500000000000)

        torch.testing.assert_close(loss_full, expected_loss_full, atol=1e-30, rtol=1e-30)
        torch.testing.assert_close(loss_partial, expected_loss_partial, atol=1e-30, rtol=1e-30)
        # Verify both cases produce valid losses
        # self.assertFalse(torch.isnan(loss_full))
        # self.assertFalse(torch.isnan(loss_partial))
        
        # # Since mask affects sequence score calculation, losses should be different
        # self.assertNotEqual(loss_full.item(), loss_partial.item())
    
    def test_edge_cases(self):
        """Test edge cases"""
        batch_size = 2
        seq_len = 3
        
        # Test extreme token_level_scores
        extreme_scores = torch.tensor([[100.0, 100.0, 100.0], [-100.0, -100.0, -100.0]])
        acc = torch.tensor([1.0, 0.0])
        response_mask = torch.ones(batch_size, seq_len)
        beta = 0.1
        
        loss = compute_ce_dpo_loss_rm(extreme_scores, acc, response_mask, beta)
        
        # print(f"loss: {loss.item():.30f}")
        expected_loss = torch.tensor(0.000000000000046788117949230976)
        # Even with extreme values, loss should be valid
        # self.assertFalse(torch.isnan(loss))
        # self.assertFalse(torch.isinf(loss))
        # self.assertGreaterEqual(loss.item(), 0.0)
        torch.testing.assert_close(loss, expected_loss, atol=1e-30, rtol=1e-30)
        
        # Test zero scores case
        zero_scores = torch.zeros(batch_size, seq_len)
        loss_zero = compute_ce_dpo_loss_rm(zero_scores, acc, response_mask, beta)
        
        # print(loss_zero)
        # print(f"loss_zero: {loss_zero.item():.30f}")
        expected_loss_zero = torch.tensor(0.693147182464599609375000000000)
        # self.assertFalse(torch.isnan(loss_zero))
        # self.assertFalse(torch.isinf(loss_zero))
        torch.testing.assert_close(loss_zero, expected_loss_zero, atol=1e-30, rtol=1e-30)
    
    def test_batch_consistency(self):
        """Test batch processing consistency"""
        seq_len = 5
        beta = 0.1
        
        # Create single sample
        single_scores = torch.randn(1, seq_len)
        single_acc = torch.rand(1)
        single_mask = torch.ones(1, seq_len)
        
        single_loss = compute_ce_dpo_loss_rm(single_scores, single_acc, single_mask, beta)
        
        # Create batch containing the same sample
        batch_scores = single_scores.repeat(3, 1)
        batch_acc = single_acc.repeat(3)
        batch_mask = single_mask.repeat(3, 1)
        
        batch_loss = compute_ce_dpo_loss_rm(batch_scores, batch_acc, batch_mask, beta)
        
        # Since all samples are identical, batch loss should equal single sample loss
        self.assertAlmostEqual(single_loss.item(), batch_loss.item(), places=30)

    def test_input_output_samples(self):
        """Test specific input-output examples"""
        # Test case 1: Simple deterministic case
        token_level_scores = torch.tensor([[1.0, 2.0, 3.0]])  # Sequence score sum: 6.0
        acc = torch.tensor([1.0])
        response_mask = torch.ones(1, 3)
        beta = 1.0
        
        result = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta)
        
        # Manual calculation verification:
        # Sequence score = 6.0 * 1.0 = 6.0
        # Probability = sigmoid(6.0) ≈ 0.9975
        # BCE loss = -1.0 * log(0.9975) - 0.0 * log(1-0.9975) ≈ 0.0025
        expected_prob = torch.sigmoid(torch.tensor(6.0))
        expected_loss = torch.nn.functional.binary_cross_entropy(expected_prob.unsqueeze(0), acc)
        self.assertAlmostEqual(result.item(), expected_loss.item(), places=30)
        
        # Test case 2: Negative score case
        token_level_scores = torch.tensor([[-1.0, -2.0]])  # Sequence score sum: -3.0
        acc = torch.tensor([0.0])  # Expect low probability
        response_mask = torch.ones(1, 2)
        beta = 1.0
        
        result = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta)
        
        # Manual calculation verification:
        # Sequence score = -3.0 * 1.0 = -3.0
        # Probability = sigmoid(-3.0) ≈ 0.0474
        # BCE loss = -0.0 * log(0.0474) - 1.0 * log(1-0.0474) ≈ 0.0487
        expected_prob = torch.sigmoid(torch.tensor(-3.0))
        expected_loss = torch.nn.functional.binary_cross_entropy(expected_prob.unsqueeze(0), acc)
        self.assertAlmostEqual(result.item(), expected_loss.item(), places=30)
        
        # Test case 3: Multi-sample batch
        token_level_scores = torch.tensor([
            [2.0, 1.0],    # Sequence score: 3.0
            [0.5, -0.5],   # Sequence score: 0.0
            [-1.0, -1.0]   # Sequence score: -2.0
        ])
        acc = torch.tensor([0.8, 0.5, 0.2])
        response_mask = torch.ones(3, 2)
        beta = 0.5
        
        result = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta)
        
        # Manual calculation verification:
        # Sequence scores = [3.0*0.5, 0.0*0.5, -2.0*0.5] = [1.5, 0.0, -1.0]
        # Probabilities = sigmoid([1.5, 0.0, -1.0])
        seq_scores = torch.tensor([3.0, 0.0, -2.0]) * beta
        expected_probs = torch.sigmoid(seq_scores)
        expected_loss = torch.nn.functional.binary_cross_entropy(expected_probs, acc)
        self.assertAlmostEqual(result.item(), expected_loss.item(), places=30)
        
        # Test case 4: With mask
        token_level_scores = torch.tensor([
            [2.0, 3.0, 1.0],  # Valid scores: 2.0 + 3.0 = 5.0 (last one masked)
            [1.0, 1.0, 1.0]   # Valid scores: 1.0 + 1.0 + 1.0 = 3.0 (all valid)
        ])
        acc = torch.tensor([0.9, 0.7])
        response_mask = torch.tensor([
            [1.0, 1.0, 0.0],  # Last one invalid
            [1.0, 1.0, 1.0]   # All valid
        ])
        beta = 0.2
        
        result = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta)
        
        # Manual calculation verification:
        # Valid sequence scores = [5.0*0.2, 3.0*0.2] = [1.0, 0.6]
        # Probabilities = sigmoid([1.0, 0.6])
        seq_scores = torch.tensor([5.0, 3.0]) * beta
        expected_probs = torch.sigmoid(seq_scores)
        expected_loss = torch.nn.functional.binary_cross_entropy(expected_probs, acc)
        self.assertAlmostEqual(result.item(), expected_loss.item(), places=30)

    def test_beta_scaling_input_output_samples(self):
        """Test specific input-output examples for beta parameter scaling"""
        token_level_scores = torch.tensor([[4.0, 2.0]])  # Sequence score: 6.0
        acc = torch.tensor([0.8])
        response_mask = torch.ones(1, 2)
        
        # Test effect of different beta values
        beta_small = 0.1
        result_small = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta_small)
        
        # Manual calculation: Sequence score = 6.0 * 0.1 = 0.6
        expected_prob_small = torch.sigmoid(torch.tensor(6.0 * beta_small))
        expected_loss_small = torch.nn.functional.binary_cross_entropy(expected_prob_small.unsqueeze(0), acc)
        self.assertAlmostEqual(result_small.item(), expected_loss_small.item(), places=30)
        
        beta_large = 2.0
        result_large = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta_large)
        
        # Manual calculation: Sequence score = 6.0 * 2.0 = 12.0
        expected_prob_large = torch.sigmoid(torch.tensor(6.0 * beta_large))
        expected_loss_large = torch.nn.functional.binary_cross_entropy(expected_prob_large.unsqueeze(0), acc)
        self.assertAlmostEqual(result_large.item(), expected_loss_large.item(), places=30)
        
        # Larger beta makes sequence score impact more significant
        # For positive sequence scores and accuracy label 0.8, larger beta makes predicted probability closer to 1
        # But target is 0.8, so overly high predicted probability (close to 1) leads to larger loss
        # Therefore larger beta should produce larger loss
        self.assertGreater(result_large.item(), result_small.item())

    def test_perfect_prediction_cases(self):
        """Test input-output examples for perfect prediction cases"""
        # Test case 1: High scores + high accuracy (should have very small loss)
        token_level_scores = torch.tensor([[10.0, 5.0]])  # Very high scores
        acc = torch.tensor([1.0])  # Perfect accuracy
        response_mask = torch.ones(1, 2)
        beta = 0.1
        
        result = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta)
        # print(f"result: {result.item():.30f}")
        expected_result = torch.tensor(0.201413318514823913574218750000)
        # Sequence score = 15.0 * 0.1 = 1.5, sigmoid(1.5) ≈ 0.8176
        # BCE loss should be small because both predicted probability and target are high
        # expected_prob = torch.sigmoid(torch.tensor(15.0 * beta))
        # expected_loss = torch.nn.functional.binary_cross_entropy(expected_prob.unsqueeze(0), acc)
        # self.assertAlmostEqual(result.item(), expected_loss.item(), places=5)
        # self.assertLess(result.item(), 0.25)  # Loss should be relatively small
        torch.testing.assert_close(result, expected_result, atol=1e-30, rtol=1e-30)
        # Test case 2: Low scores + low accuracy (should have very small loss)
        token_level_scores = torch.tensor([[-5.0, -12.0]])  # Very low scores
        acc = torch.tensor([0.0])  # Zero accuracy
        response_mask = torch.ones(1, 2)
        beta = 0.1
        
        result = compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta)
        # print(f"result: {result.item():.30f}")
        expected_result = torch.tensor(0.167786017060279846191406250000)
        # Sequence score = -15.0 * 0.1 = -1.5, sigmoid(-1.5) ≈ 0.1824
        # BCE loss should be small because predicted probability is low and target is also 0
        # expected_prob = torch.sigmoid(torch.tensor(-15.0 * beta))
        # expected_loss = torch.nn.functional.binary_cross_entropy(expected_prob.unsqueeze(0), acc)
        # self.assertAlmostEqual(result.item(), expected_loss.item(), places=5)
        # self.assertLess(result.item(), 0.25)  # Loss should be relatively small
        torch.testing.assert_close(result, expected_result, atol=1e-30, rtol=1e-30)


if __name__ == "__main__":
    unittest.main()
