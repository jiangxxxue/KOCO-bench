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
from prime.prime_core_algos import compute_detach_dpo_loss_rm


class TestDetachDPOLoss(unittest.TestCase):
    """Test compute_detach_dpo_loss_rm function - Detached DPO loss calculation described in Section 2 of the documentation"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_compute_detach_dpo_loss_rm_basic(self):
        """Test basic functionality of detached DPO loss calculation"""
        batch_size = 4
        seq_len = 10
        n_samples = 2
        
        token_level_scores = torch.tensor([[0.1,0.3,0.6,-0.1,0.2,0.5,0.1,0.3,0.6,-0.1], [0.2,0.3,0.6,-0.4,0.9,-0.1,0.2,0.5,0.1,0.3], [-0.1,0.3,0.6,-0.1,0.2,0.7,1.0,0.2,0.5,0.1], [0.1,0.8,0.6,-0.1,0.2,0.5,0.1,0.5,0.1,0.3]])
        acc = torch.tensor([0.8, 0.3, 1.0, 0.0])
        Q_bc = torch.tensor([[0.1,0.3], [0.4,0.5], [0.2,0.6], [0.1,0.9]])
        acc_bc = torch.tensor([[0.8, 0.3], [0.7, 0.2], [0.8, 0.3], [0.6, 0.1]])
        response_mask = torch.ones(batch_size, seq_len)
        beta = 0.1
        
        losses = []
        # Test different bon_mode
        for bon_mode in ["none", "bon_rm", "bon_acc"]:
            with self.subTest(bon_mode=bon_mode):
                loss = compute_detach_dpo_loss_rm(
                    token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode=bon_mode
                )
                losses.append(loss.item())
                # self.assertEqual(loss.dim(), 0)
                # self.assertTrue(torch.is_tensor(loss))
                # self.assertFalse(torch.isnan(loss))
                # self.assertFalse(torch.isinf(loss))
        # print(f"losses: {losses[0]:.30f}")
        # print(f"losses: {losses[1]:.30f}")
        # print(f"losses: {losses[2]:.30f}")
        expected_losses = [0.642190814018249511718750000000, 5.137526512145996093750000000000, 2.880730152130126953125000000000]

        torch.testing.assert_close(losses, expected_losses, atol=1e-30, rtol=1e-30)
    
    def test_dpo_loss_mathematical_properties(self):
        """Test mathematical properties of detached DPO loss"""
        batch_size = 2
        seq_len = 4
        n_samples = 2
        
        # Create deterministic data
        token_level_scores = torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
        acc = torch.tensor([1.0, 0.0])
        Q_bc = torch.tensor([[1.5, 0.5], [1.5, 0.5]])  # Q values in batch
        acc_bc = torch.tensor([[1.0, 0.0], [1.0, 0.0]])  # Accuracy in batch
        response_mask = torch.ones(batch_size, seq_len)
        beta = 0.1
        
        loss = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )
        # print(f"loss: {loss.item():.30f}")
        expected_loss = torch.tensor(0.801718711853027343750000000000)
        # Manually verify loss calculation logic
        Q_current = (token_level_scores * response_mask).sum(dim=1) * beta  # [0.4, 0.8]
        
        # Verify loss is a valid value
        # self.assertFalse(torch.isnan(loss))
        # self.assertFalse(torch.isinf(loss))
        # self.assertGreaterEqual(loss.item(), 0.0)
        torch.testing.assert_close(loss, expected_loss, atol=1e-30, rtol=1e-30)

    def test_bon_mode_none(self):
        """Test weight calculation for bon_mode='none'"""
        batch_size = 4
        seq_len = 6
        n_samples = 2
        
        token_level_scores = torch.tensor([[0.1,0.3,0.6,-0.1,0.2,0.5], [0.2,0.3,0.6,-0.4,0.9,-0.1], [-0.1,0.3,0.6,-0.1,0.2,0.7], [0.1,0.8,0.6,-0.1,0.2,0.5]])
        acc = torch.tensor([0.8, 0.3, 1.0, 0.0])
        Q_bc = torch.tensor([[0.1,0.3], [0.4,0.5], [0.2,0.6], [0.1,0.9]])
        acc_bc = torch.tensor([[0.8, 0.3], [0.7, 0.2], [0.8, 0.3], [0.6, 0.1]])
        # Q_bc = torch.randn(batch_size, n_samples)
        # acc_bc = torch.rand(batch_size, n_samples)
        response_mask = torch.ones(batch_size, seq_len)
        beta = 0.1
        
        loss = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )
        # print(loss)
        # print(f"loss: {loss.item():.30f}")
        expected_loss = torch.tensor(0.671486198902130126953125000000)
        # none mode should produce valid loss
        # self.assertFalse(torch.isnan(loss))
        # self.assertFalse(torch.isinf(loss))
        torch.testing.assert_close(loss, expected_loss, atol=1e-30, rtol=1e-30)
    
    def test_bon_mode_rm(self):
        """Test BoN weight calculation for bon_mode='bon_rm'"""
        batch_size = 4
        seq_len = 6
        n_samples = 3
        
        token_level_scores = torch.tensor([[0.1,0.3,0.6,-0.1,0.2,0.5], [0.2,0.3,0.6,-0.4,0.9,-0.1], [-0.1,0.3,0.6,-0.1,0.2,0.7], [0.1,0.8,0.6,-0.1,0.2,0.5]])
        acc = torch.tensor([0.8, 0.3, 1.0, 0.0])
        Q_bc = torch.tensor([[0.1,0.3], [0.4,0.5], [0.2,0.6], [0.1,0.9]])
        acc_bc = torch.tensor([[0.8, 0.3], [0.7, 0.2], [0.8, 0.3], [0.6, 0.1]])
        response_mask = torch.ones(batch_size, seq_len)
        beta = 0.1
        
        loss = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="bon_rm"
        )
        # print(loss)
        # print(f"loss: {loss.item():.30f}")
        expected_loss = torch.tensor(5.371889591217041015625000000000)

        # bon_rm mode assigns weights based on Q value ranking
        # self.assertFalse(torch.isnan(loss))
        # self.assertFalse(torch.isinf(loss))
        torch.testing.assert_close(loss, expected_loss, atol=1e-30, rtol=1e-30)
    
    def test_bon_mode_acc(self):
        """Test BoN weight calculation for bon_mode='bon_acc'"""
        batch_size = 4
        seq_len = 6
        n_samples = 3
        
        token_level_scores = torch.tensor([[0.1,0.3,0.6,-0.1,0.2,0.7], [0.2,0.5,0.6,-0.4,0.9,-0.1], [-0.1,0.3,0.6,-0.1,0.2,0.7], [0.1,0.8,0.6,-0.1,0.2,0.5]])
        acc = torch.tensor([0.8, 0.3, 1.0, 0.0])
        Q_bc = torch.tensor([[0.1,0.3], [0.4,0.5], [0.2,0.6], [0.1,0.9]])
        acc_bc = torch.tensor([[0.8, 0.3], [0.7, 0.2], [0.8, 0.3], [0.6, 0.1]])
        response_mask = torch.ones(batch_size, seq_len)
        beta = 0.1
        
        loss = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="bon_acc"
        )
        # print(loss)
        # print(f"loss: {loss.item():.30f}")
        expected_loss = torch.tensor(3.146752834320068359375000000000)
        # bon_acc mode assigns weights based on accuracy ranking
        # self.assertFalse(torch.isnan(loss))
        # self.assertFalse(torch.isinf(loss))
        torch.testing.assert_close(loss, expected_loss, atol=1e-30, rtol=1e-30)
    
    def test_different_beta_values(self):
        """Test the impact of different beta parameters on detached DPO loss"""
        batch_size = 2
        seq_len = 4
        n_samples = 2
        
        token_level_scores = torch.tensor([[0.6,-0.1,0.2,0.5], [0.2,0.3,0.9,-0.1]])
        acc = torch.tensor([0.8, 1.0])
        Q_bc = torch.tensor([[0.1,0.3], [0.4,0.5]])
        acc_bc = torch.tensor([[0.8, 0.3], [0.7, 0.2]])
        response_mask = torch.ones(batch_size, seq_len)
        
        betas = [0.01, 0.1, 1.0, 10.0]
        losses = []
        
        for beta in betas:
            loss = compute_detach_dpo_loss_rm(
                token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
            )
            losses.append(loss.item())
            
            # Verify all losses are valid
            # self.assertFalse(torch.isnan(loss))
            # self.assertFalse(torch.isinf(loss))
        
        # print(losses)
        # Different beta values should produce different losses
        # self.assertTrue(len(set(losses)) > 1, "Different beta values should produce different losses")
        # print(f"losses: {losses[0]:.30f}")
        # print(f"losses: {losses[1]:.30f}")
        # print(f"losses: {losses[2]:.30f}")
        # print(f"losses: {losses[3]:.30f}")
        expected_losses = [0.688781797885894775390625000000, 0.650354683399200439453125000000, 0.348509460687637329101562500000,  0.000163420278113335371017456055]
        torch.testing.assert_close(losses, expected_losses, atol=1e-30, rtol=1e-30)
    
    def test_contrasting_samples_selection(self):
        """Test contrastive sample selection logic"""
        batch_size = 4
        seq_len = 4
        n_samples = 2
        
        # Create samples with different accuracies
        token_level_scores = torch.tensor([[0.5, 0.7, 1.0, 0.1], [0.0, -1.0, 0.6, 0.2], [0.3, 0.0, 0.44, 0.5], [0.0, 0.45, 0.1, -0.3]])
        acc = torch.tensor([1.0, 0.0, 1.0, 0.0])  # Alternating accuracy labels
        
        # Q_bc and acc_bc should contain samples with different accuracies
        Q_bc = torch.tensor([[0.1,0.3], [0.4,0.5], [0.2,0.6], [0.1,0.9]])
        acc_bc = torch.tensor([[0.8, 0.3], [0.7, 0.2], [0.8, 0.3], [0.6, 0.1]])
        
        response_mask = torch.ones(batch_size, seq_len)
        beta = 0.1
        
        loss = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )
        # print(loss)
        # print(f"loss: {loss.item():.30f}")
        expected_loss = torch.tensor( 0.646894752979278564453125000000)
        # Verify contrastive sample selection logic produces valid loss
        # self.assertFalse(torch.isnan(loss))
        # self.assertFalse(torch.isinf(loss))
        torch.testing.assert_close(loss, expected_loss, atol=1e-30, rtol=1e-30)
    
    def test_direction_calculation(self):
        """Test preference direction calculation"""
        batch_size = 4
        seq_len = 3
        n_samples = 2
        
        # Test the impact of different accuracies on preference direction
        token_level_scores = torch.ones(batch_size, seq_len)
        acc_positive = torch.tensor([0.8, 0.9, 0.7, 0.6])  # Correct samples
        acc_negative = torch.tensor([0.2, 0.1, 0.3, 0.4])  # Incorrect samples
        
        Q_bc = torch.tensor([[0.1,0.3], [0.4,0.5], [0.2,0.6], [0.1,0.9]])
        acc_bc = torch.tensor([[0.8, 0.3], [0.7, 0.2], [0.8, 0.3], [0.6, 0.1]])
        response_mask = torch.ones(batch_size, seq_len)
        beta = 0.1
        
        # Test correct samples
        loss_pos = compute_detach_dpo_loss_rm(
            token_level_scores, acc_positive, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )
        
        # Test incorrect samples
        loss_neg = compute_detach_dpo_loss_rm(
            token_level_scores, acc_negative, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )
        
        # print(loss_pos)
        # print(loss_neg)
        # print(f"loss_pos: {loss_pos.item():.30f}")
        # print(f"loss_neg: {loss_neg.item():.30f}")
        expected_loss_pos = torch.tensor(0.578741252422332763671875000000)
        expected_loss_neg = torch.tensor(0.564178824424743652343750000000)
        torch.testing.assert_close(loss_pos, expected_loss_pos, atol=1e-30, rtol=1e-30)
        torch.testing.assert_close(loss_neg, expected_loss_neg, atol=1e-30, rtol=1e-30)

    def test_input_output_samples(self):
        """Test specific input-output examples"""
        # Test case 1: Simple two-sample case
        token_level_scores = torch.tensor([
            [2.0, 1.0],  # Current Q = 3.0 * beta
            [1.0, 0.5]   # Current Q = 1.5 * beta
        ])
        acc = torch.tensor([1.0, 0.0])  # First sample high accuracy, second low accuracy
        Q_bc = torch.tensor([
            [2.0, 1.0],  # Q values in batch
            [2.0, 1.0]
        ])
        acc_bc = torch.tensor([
            [0.8, 0.2],  # Accuracy in batch
            [0.8, 0.2]
        ])
        response_mask = torch.ones(2, 2)
        beta = 0.1
        
        result = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )

        # print(result)
        # print(f"result: {result.item():.30f}")
        expected_result = torch.tensor(0.657052159309387207031250000000)
        torch.testing.assert_close(result, expected_result, atol=1e-30, rtol=1e-30)
        # Manual calculation verification:
        # cur_Q = [3.0*0.1, 1.5*0.1] = [0.3, 0.15]
        # For sample 1 (acc=1.0>0): select samples where acc_bc < 1.0, i.e. Q_bc[0][acc_bc[0] < 1.0] = [2.0, 1.0]
        # other_Q[0] = mean([2.0, 1.0]) * 0.1 = 1.5 * 0.1 = 0.15
        # For sample 2 (acc=0.0<=0): select samples where acc_bc > 0.0, i.e. Q_bc[1][acc_bc[1] > 0.0] = [2.0, 1.0]
        # other_Q[1] = mean([2.0, 1.0]) * 0.1 = 1.5 * 0.1 = 0.15
        # direction = [1, -1] (based on acc > 0 condition)
        # dpo_loss = -log(sigmoid(([0.3, 0.15] - [0.15, 0.15]) * [1, -1]))
        #          = -log(sigmoid([0.15, 0.0]))
        # cur_Q = torch.tensor([0.3, 0.15])
        # other_Q = torch.tensor([0.15, 0.15])
        # direction = torch.tensor([1.0, -1.0])
        # expected_dpo_loss = -torch.log(torch.sigmoid((cur_Q - other_Q) * direction))
        # expected_result = expected_dpo_loss.mean()
        
        # self.assertAlmostEqual(result.item(), expected_result.item(), places=4)
        
        # Test case 2: BoN weight mode
        token_level_scores = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        acc = torch.tensor([0.8, 0.6])
        Q_bc = torch.tensor([[1.0, 2.0], [1.5, 2.5]])  # Different Q value distribution
        acc_bc = torch.tensor([[0.5, 0.9], [0.4, 0.8]])  # Different accuracy distribution
        response_mask = torch.ones(2, 2)
        beta = 0.1
        
        result = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="bon_acc"
        )
        
        # print(result)
        # print(f"result: {result.item():.30f}")
        expected_result = torch.tensor(1.220335960388183593750000000000)
        torch.testing.assert_close(result, expected_result, atol=1e-30, rtol=1e-30)
        # Verify BoN mode produces valid loss
        #     self.assertFalse(torch.isnan(result))
        #     self.assertFalse(torch.isinf(result))
        #     self.assertGreaterEqual(result.item(), 0.0)
        
        # Test case 3: With mask
        token_level_scores = torch.tensor([
            [3.0, 2.0, 1.0],  # Valid scores: 3.0 + 2.0 = 5.0
            [1.0, 1.0, 1.0]   # Valid scores: 1.0 + 1.0 + 1.0 = 3.0
        ])
        acc = torch.tensor([0.9, 0.1])
        Q_bc = torch.tensor([[2.0, 1.0], [2.5, 1.5]])
        acc_bc = torch.tensor([[0.7, 0.3], [0.8, 0.2]])
        response_mask = torch.tensor([
            [1.0, 1.0, 0.0],  # Last position masked
            [1.0, 1.0, 1.0]   # All valid
        ])
        beta = 0.2
        
        result = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )

        # print(result)
        # print(f"result: {result.item():.30f}")
        expected_result = torch.tensor(0.420337051153182983398437500000)
        torch.testing.assert_close(result, expected_result, atol=1e-30, rtol=1e-30)
        
        # Manual calculation verification of mask effect:
        # cur_Q = [5.0*0.2, 3.0*0.2] = [1.0, 0.6]
        expected_cur_Q = torch.tensor([5.0, 3.0]) * beta
        # Verify loss calculation is correct
        # self.assertFalse(torch.isnan(result))
        # self.assertFalse(torch.isinf(result))

    def test_contrast_sample_selection_input_output(self):
        """Test specific input-output for contrastive sample selection"""
        # Test case: Clear contrastive sample selection
        token_level_scores = torch.tensor([[2.0, 2.0]])  # cur_Q = 4.0 * beta
        acc = torch.tensor([0.8])  # High accuracy sample
        Q_bc = torch.tensor([[3.0, 1.0, 2.0]])  # Q values in batch
        acc_bc = torch.tensor([[0.9, 0.3, 0.7]])  # Accuracy in batch, different contrastive samples
        response_mask = torch.ones(1, 2)
        beta = 0.1
        
        result = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )
        # print(result)
        # print(f"result: {result.item():.30f}")
        expected_result = torch.tensor(0.575939357280731201171875000000)
        torch.testing.assert_close(result, expected_result, atol=1e-30, rtol=1e-30)
        # Verify high accuracy sample contrastive selection produces valid loss
        # self.assertFalse(torch.isnan(result))
        # self.assertFalse(torch.isinf(result))
        # self.assertGreaterEqual(result.item(), 0.0)
        
        # Test case: Low accuracy sample contrastive selection
        acc_low = torch.tensor([0.2])  # Low accuracy sample
        result_low = compute_detach_dpo_loss_rm(
            token_level_scores, acc_low, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )
        # print(result_low)
        # print(f"result_low: {result_low.item():.30f}")
        expected_result_low = torch.tensor( 0.513015270233154296875000000000)
        torch.testing.assert_close(result_low, expected_result_low, atol=1e-30, rtol=1e-30)
        # Verify low accuracy sample contrastive selection produces valid loss
        # self.assertFalse(torch.isnan(result_low))
        # self.assertFalse(torch.isinf(result_low))
        # self.assertGreaterEqual(result_low.item(), 0.0)
        
        # Verify different accuracies produce different loss values (reflecting contrastive sample selection differences)
        # self.assertNotAlmostEqual(result.item(), result_low.item(), places=3)

    def test_bon_weighting_input_output(self):
        """Test specific input-output for BoN weight calculation"""
        # Test case: bon_rm mode
        token_level_scores = torch.tensor([
            [1.0, 1.0],  # cur_Q = 2.0 * beta = 0.2
            [2.0, 2.0]   # cur_Q = 4.0 * beta = 0.4
        ])
        acc = torch.tensor([0.6, 0.8])
        Q_bc = torch.tensor([
            [1.0, 2.0, 3.0],  # Compared with cur_Q[0]=0.2: [0.1, 0.2, 0.3]
            [3.0, 4.0, 5.0]   # Compared with cur_Q[1]=0.4: [0.3, 0.4, 0.5]
        ])
        acc_bc = torch.tensor([
            [0.4, 0.7, 0.9],
            [0.6, 0.8, 1.0]
        ])
        response_mask = torch.ones(2, 2)
        beta = 0.1
        n_samples = 3
        
        result = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="bon_rm"
        )
        # print(result)
        # print(f"result: {result.item():.30f}")
        expected_result = torch.tensor(1.718391299247741699218750000000)
        torch.testing.assert_close(result, expected_result, atol=1e-30, rtol=1e-30)
        # Manual weight calculation verification:
        # cur_Q = [0.2, 0.4]
        # For sample 1: Q_bc[0] * beta = [0.1, 0.2, 0.3], proportion satisfying <= 0.2 = 2/3
        # weight[0] = 3 * (2/3)^(3-1) = 3 * (2/3)^2 = 3 * 4/9 = 4/3
        # For sample 2: Q_bc[1] * beta = [0.3, 0.4, 0.5], proportion satisfying <= 0.4 = 2/3
        # weight[1] = 3 * (2/3)^(3-1) = 3 * (2/3)^2 = 3 * 4/9 = 4/3
        
        # Verify result is valid
        # self.assertFalse(torch.isnan(result))
        # self.assertFalse(torch.isinf(result))
        # self.assertGreaterEqual(result.item(), 0.0)
        
        # Test case: bon_acc mode
        result_acc = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="bon_acc"
        )
        
        # print(result_acc)
        # print(f"result_acc: {result_acc.item():.30f}")
        expected_result_acc = torch.tensor(1.073994517326354980468750000000)
        torch.testing.assert_close(result_acc, expected_result_acc, atol=1e-30, rtol=1e-30)
        # Manual accuracy weight calculation verification:
        # acc = [0.6, 0.8]
        # For sample 1: acc_bc[0] = [0.4, 0.7, 0.9], proportion satisfying <= 0.6 = 2/3
        # For sample 2: acc_bc[1] = [0.6, 0.8, 1.0], proportion satisfying <= 0.8 = 2/3
        # Weight calculation same as bon_rm
        
        # self.assertFalse(torch.isnan(result_acc))
        # self.assertFalse(torch.isinf(result_acc))
        # self.assertGreaterEqual(result_acc.item(), 0.0)
        
        # bon_rm and bon_acc should produce different results (because weight calculation based on different metrics)
        # Here we only verify both are valid values, not enforcing inequality as they may be equal in some cases

    def test_edge_cases_input_output(self):
        """Test specific input-output for edge cases"""
        # Test case 1: No contrastive samples case
        token_level_scores = torch.tensor([[1.0, 2.0]])
        acc = torch.tensor([0.5])
        Q_bc = torch.tensor([[1.0, 2.0]])
        acc_bc = torch.tensor([[0.6, 0.7]])  # All greater than acc=0.5, no contrastive samples for acc<=0 case
        response_mask = torch.ones(1, 2)
        beta = 0.1
        
        result = compute_detach_dpo_loss_rm(
            token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )
        
        # print(result)
        # print(f"result: {result.item():.30f}")
        expected_result = torch.tensor(0.554355263710021972656250000000)
        torch.testing.assert_close(result, expected_result, atol=1e-30, rtol=1e-30)
        # When no contrastive samples satisfy the condition, other_Q should be 0
        # cur_Q = 3.0 * 0.1 = 0.3
        # other_Q = 0.0 (because no samples satisfy acc_bc < 0.5)
        # direction = 1.0
        # dpo_loss = -log(sigmoid((0.3 - 0.0) * 1.0)) = -log(sigmoid(0.3))
        # expected_cur_Q = 3.0 * beta
        # expected_other_Q = 0.0
        # expected_direction = 1.0
        # expected_dpo_loss = -torch.log(torch.sigmoid(torch.tensor((expected_cur_Q - expected_other_Q) * expected_direction)))
        
        # self.assertAlmostEqual(result.item(), expected_dpo_loss.item(), places=4)
        
        # Test case 2: Zero scores case
        zero_scores = torch.zeros(1, 3)
        acc = torch.tensor([0.8])
        Q_bc = torch.tensor([[1.0, 0.5]])
        acc_bc = torch.tensor([[0.6, 0.4]])
        response_mask = torch.ones(1, 3)
        beta = 0.2
        
        result = compute_detach_dpo_loss_rm(
            zero_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"
        )
        # print(result)
        # print(f"result: {result.item():.30f}")
        expected_result = torch.tensor(0.770957052707672119140625000000)
        torch.testing.assert_close(result, expected_result, atol=1e-30, rtol=1e-30)
        # cur_Q = 0.0 * 0.2 = 0.0
        # Select samples where acc_bc < 0.8: [0.6, 0.4] both satisfy
        # other_Q = mean([1.0, 0.5]) * 0.2 = 0.75 * 0.2 = 0.15
        # dpo_loss = -log(sigmoid((0.0 - 0.15) * 1.0)) = -log(sigmoid(-0.15))
        # expected_cur_Q = 0.0
        # expected_other_Q = 0.75 * beta
        # expected_dpo_loss = -torch.log(torch.sigmoid(torch.tensor((expected_cur_Q - expected_other_Q) * 1.0)))
        
        # self.assertAlmostEqual(result.item(), expected_dpo_loss.item(), places=4)


if __name__ == "__main__":
    unittest.main()
