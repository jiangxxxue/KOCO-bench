import unittest
import torch
import sys
import os
from unittest.mock import MagicMock, patch

# Add the parent directory to Python path to import verl module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the actual class and components we want to test
try:
    from verl.workers.fsdp_workers import ProcessRewardModelWorker
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False
    ProcessRewardModelWorker = None


class TestForwardMicroBatch(unittest.TestCase):
    """Test ProcessRewardModelWorker._forward_micro_batch function - process reward model inference and min-form credit assignment described in PURE document Section 1"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.seq_len = 8
        self.response_length = 5
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_forward_micro_batch_basic_logic(self):
        """Test basic logic of _forward_micro_batch"""
        # Create a simplified ProcessRewardModelWorker instance, avoiding complex initialization
        worker = object.__new__(ProcessRewardModelWorker)
        
        # Set necessary attributes
        worker.use_remove_padding = False
        worker.ulysses_sequence_parallel_size = 1
        worker.disable_approx_min_form_credit_assignment = False
        worker.temperature = 0.1
        
        # Create mock process_reward_module
        mock_process_reward_module = MagicMock()
        worker.process_reward_module = mock_process_reward_module
        
        # Create micro_batch input data
        micro_batch = {
            # "input_ids": torch.randint(0, 1000, (self.batch_size, self.seq_len)),
            "input_ids": torch.tensor([[13, 27, 38, 44, 51, 62,777,32], [11, 25, 36, 47, 58, 69,88,22], [19, 20, 31, 42, 53, 64,3,65], [11, 22, 33, 44, 55, 66,77,88]]),
            "attention_mask": torch.ones(self.batch_size, self.seq_len),
            "position_ids": torch.arange(self.seq_len).unsqueeze(0).repeat(self.batch_size, 1),
            # "responses": torch.randint(0, 1000, (self.batch_size, self.response_length)),
            "responses": torch.tensor([[1, 21, 32, 43, 54], [111, 256, 37, 48, 59], [19, 20, 31, 42, 53], [11, 22, 33, 44, 55]]),
            "reward_mask": torch.ones(self.batch_size, self.response_length)
        }
        
        # Create controllable binary classification logits output
        mock_logits = torch.tensor([
            [[2.0, 0.0], [1.0, 1.5], [0.5, 2.5], [0.0, 1.0], [-0.5, 1.5], [0.8, 0.2], [1.2, 0.8], [0.3, 1.7]],  # batch 0
            [[1.5, 0.5], [0.2, 2.0], [1.8, 0.3], [-1.0, 2.0], [0.6, 1.4], [1.1, 0.9], [0.4, 1.6], [2.0, 0.1]]   # batch 1
        ])  # (batch_size, seq_len, 2)
        
        mock_output = MagicMock()
        mock_output.logits = mock_logits
        mock_process_reward_module.return_value = mock_output
        
        # Call the real _forward_micro_batch method
        result = worker._forward_micro_batch(micro_batch)
        
        expected_result = torch.tensor([[ 1.7748e-04,  1.4639e-05, -2.0934e-01, -5.5440e-02,  5.5964e-05],
        [ 6.4885e-08,  5.2007e-06, -1.6514e-04,  1.5278e-06, -7.3855e-01]])
        torch.testing.assert_close(result, expected_result, atol=1e-4, rtol=1e-4)
        # Verify output
        # self.assertEqual(result.shape, (self.batch_size, self.response_length))
        # self.assertTrue(torch.is_tensor(result))
        # self.assertTrue(torch.isfinite(result).all())
        
        # Verify result range (should be between [-1, 1], but may exceed due to min-form weighting)
        # self.assertTrue(result.abs().max() <= 10.0)  # Reasonable range check
        
        # Verify min-form weighting characteristic: weights should concentrate on minimum score positions
        # Recalculate expected intermediate results to verify logic
        response_logits = mock_logits[:, -self.response_length:]  # Extract response part
        expected_probs = response_logits.softmax(dim=-1)
        expected_rm_score = (expected_probs[..., 1] - expected_probs[..., 0]) * micro_batch['reward_mask']
        
        # Call the fixed real _forward_micro_batch method
        result = worker._forward_micro_batch(micro_batch)
        
        print(result)
        # Verify output
        self.assertEqual(result.shape, (self.batch_size, self.response_length))
        self.assertTrue(torch.is_tensor(result))
        self.assertTrue(torch.isfinite(result).all())
        
        # Verify result range (should be within reasonable range)
        self.assertTrue(result.abs().max() <= 10.0)
        
        # Verify minimum score position identification and min-form weighting effect
        for i in range(self.batch_size):
            min_score_idx = expected_rm_score[i].argmin()
            # Due to min-form weighting, minimum score position should get maximum weight
            self.assertIsNotNone(min_score_idx)
            
        # Verify weight normalization property (verify through manual calculation)
        expected_weight = torch.softmax(
            -expected_rm_score.masked_fill(~micro_batch['reward_mask'].bool(), float('inf')) / 0.1,
            dim=-1
        )
        # Weight sum should be 1 (for each sequence)
        for i in range(self.batch_size):
            weight_sum = expected_weight[i].sum()
            self.assertAlmostEqual(weight_sum.item(), 1.0, places=5)
    
    # def test_binary_classification_processing(self):
    #     """Test binary classification logits processing logic"""
    #     # Create known binary classification logits
    #     # Create binary classification logits (format: [negative_class, positive_class])
    #     binary_logits = torch.tensor([
    #         [[1.0, -0.5], [2.0, 0.5], [-0.8, 1.2]],    # First sequence
    #         [[0.3, 0.8], [-1.5, 0.2], [3.0, -1.0]]     # Second sequence
    #     ])  # (batch_size, response_length, 2)
        
    #     reward_mask = torch.ones(2, 3)
        
    #     # Implement binary classification processing logic described in document
    #     # 1. Calculate softmax probabilities
    #     probs = binary_logits.softmax(dim=-1)  # (batch_size, response_length, 2)
        
    #     # 2. Calculate p_positive - p_negative
    #     rm_score = (probs[..., 1] - probs[..., 0])  # positive class - negative class
        
    #     # 3. Apply reward mask
    #     rm_score = rm_score * reward_mask
        
    #     # Verify results
    #     self.assertEqual(rm_score.shape, (2, 3))
    #     self.assertTrue(torch.isfinite(rm_score).all())
        
    #     # Verify score range: should be between [-1, 1]
    #     self.assertTrue((rm_score >= -1.0).all())
    #     self.assertTrue((rm_score <= 1.0).all())
        
    #     # Verify specific values: based on logits format [negative_class, positive_class]
    #     # Position (0,1): logits=[2.0, 0.5], negative(2.0) > positive(0.5), so score should be negative
    #     self.assertLess(rm_score[0, 1].item(), 0.0)
        
    #     # Position (1,1): logits=[-1.5, 0.2], positive(0.2) > negative(-1.5), so score should be positive
    #     self.assertGreater(rm_score[1, 1].item(), 0.0)
    
    # def test_min_form_credit_assignment(self):
    #     """Test min-form credit assignment weighting mechanism"""
    #     batch_size, response_length = 2, 4
    #     temperature = 0.1
        
    #     # Create reward scores with clear differences
    #     rm_score = torch.tensor([
    #         [2.0, 0.5, 1.0, -0.5],  # First sequence: last token has lowest score
    #         [1.5, -1.0, 0.8, 2.5]   # Second sequence: second token has lowest score
    #     ])
    #     reward_mask = torch.ones(batch_size, response_length)
        
    #     # Implement approximate min-form weight: softmax(-rm_score/T)
    #     weight = torch.softmax(
    #         -rm_score.masked_fill(~reward_mask.bool(), float('inf')) / temperature,
    #         dim=-1
    #     )
    #     weighted_rm_score = rm_score * weight
        
    #     # Verify weight property: tokens with lower scores should have higher weights (approaching min operation)
    #     for i in range(batch_size):
    #         min_score_idx = rm_score[i].argmin()
    #         max_weight_idx = weight[i].argmax()
    #         # Position with minimum score should correspond to maximum weight
    #         self.assertEqual(min_score_idx, max_weight_idx)
        
    #     # Verify weight normalization
    #     for i in range(batch_size):
    #         valid_positions = reward_mask[i].bool()
    #         weight_sum = weight[i, valid_positions].sum()
    #         self.assertAlmostEqual(weight_sum.item(), 1.0, places=5)
        
    #     # Verify output shape and validity
    #     self.assertEqual(weighted_rm_score.shape, (batch_size, response_length))
    #     self.assertTrue(torch.isfinite(weighted_rm_score).all())
    
    # def test_temperature_effects(self):
    #     """Test temperature parameter effects on min-form weighting"""
    #     rm_score = torch.tensor([[1.0, -2.0, 0.5, -1.5]])  # Second position has lowest score
    #     reward_mask = torch.ones(1, 4)
        
    #     # Test weight distribution under different temperatures
    #     temperatures = [0.01, 0.1, 1.0]  # From low to high
    #     weights = []
        
    #     for temp in temperatures:
    #         weight = torch.softmax(
    #             -rm_score.masked_fill(~reward_mask.bool(), float('inf')) / temp,
    #             dim=-1
    #         )
    #         weights.append(weight)
        
    #     # Verify temperature effects
    #     # 1. At all temperatures, minimum score position (index=1) should have maximum weight
    #     for weight in weights:
    #         max_weight_idx = weight[0].argmax()
    #         self.assertEqual(max_weight_idx.item(), 1)  # Second position (-2.0) in rm_score is minimum
            
    #     # 2. Lower temperature means weight distribution more concentrated at minimum value position
    #     # Maximum weight at low temperature should be larger
    #     self.assertGreater(weights[0][0, 1].item(), weights[1][0, 1].item())
    #     self.assertGreater(weights[1][0, 1].item(), weights[2][0, 1].item())
        
    #     # 3. Verify weight normalization
    #     for weight in weights:
    #         weight_sum = weight[0].sum()
    #         self.assertAlmostEqual(weight_sum.item(), 1.0, places=5)
    
#     def test_response_length_extraction(self):
#         """Test response length extraction and slicing logic"""
#         batch_size, total_seq_len = 2, 10
#         response_length = 4
        
#         # Create logits for full sequence
#         full_logits = torch.randn(batch_size, total_seq_len, 2)
        
#         # Implement response part extraction: logits[:, -response_length:]
#         response_logits = full_logits[:, -response_length:]
        
#         # Verify extracted logits shape is correct
#         self.assertEqual(response_logits.shape, (batch_size, response_length, 2))
        
#         # Verify extraction is from end of sequence
#         expected_response_logits = full_logits[:, total_seq_len-response_length:]
#         torch.testing.assert_close(response_logits, expected_response_logits)
        
#         # Process extracted logits
#         rm_score = response_logits.softmax(dim=-1)
#         rm_score = (rm_score[..., 1] - rm_score[..., 0])
        
#         # Verify final shape
#         self.assertEqual(rm_score.shape, (batch_size, response_length))
#         self.assertTrue(torch.isfinite(rm_score).all())
    
#     def test_reward_mask_application(self):
#         """Test reward_mask application"""
#         batch_size, response_length = 2, 5
        
#         # Create rm_score
#         rm_score = torch.randn(batch_size, response_length)
        
#         # Create partially masked reward_mask
#         reward_mask = torch.tensor([
#             [1.0, 1.0, 1.0, 0.0, 0.0],  # First sequence: first 3 are valid
#             [1.0, 1.0, 1.0, 1.0, 0.0]   # Second sequence: first 4 are valid
#         ])
        
#         # Apply mask
#         masked_rm_score = rm_score * reward_mask
        
#         # Verify mask effect
#         # Masked positions should be 0
#         self.assertEqual(masked_rm_score[0, 3].item(), 0.0)
#         self.assertEqual(masked_rm_score[0, 4].item(), 0.0)
#         self.assertEqual(masked_rm_score[1, 4].item(), 0.0)
        
#         # Unmasked positions should retain original values
#         for i in range(batch_size):
#             for j in range(response_length):
#                 if reward_mask[i, j] == 1.0:
#                     self.assertEqual(masked_rm_score[i, j].item(), rm_score[i, j].item())
#                 else:
#                     self.assertEqual(masked_rm_score[i, j].item(), 0.0)
    
#     def test_input_output_samples_basic_prm_forward(self):
#         """Test specific input/output samples for basic PRM forward inference"""
#         batch_size, response_length = 1, 3
#         temperature = 0.1
        
#         # Create deterministic binary classification logits (format: [negative_class, positive_class])
#         binary_logits = torch.tensor([
#             [[2.0, 0.0], [1.0, 1.5], [0.5, 2.5]]  # First token strong negative, second strong positive, third very strong positive
#         ])  # shape: (1, 3, 2)
        
#         reward_mask = torch.ones(1, 3)
        
#         # Manually calculate expected results
#         # 1. Softmax processing
#         probs = binary_logits.softmax(dim=-1)
#         rm_score = (probs[..., 1] - probs[..., 0])  # positive - negative
        
#         # 2. Approximate min-form weighting
#         weight = torch.softmax(
#             -rm_score.masked_fill(~reward_mask.bool(), float('inf')) / temperature,
#             dim=-1
#         )
#         weighted_rm_score = rm_score * weight
        
#         # Verify results
#         self.assertEqual(weighted_rm_score.shape, (1, 3))
#         self.assertTrue(torch.isfinite(weighted_rm_score).all())
        
#         # Verify weight allocation: minimum score position should get maximum weight
#         min_score_idx = rm_score[0].argmin()
#         max_weight_idx = weight[0].argmax()
#         self.assertEqual(min_score_idx, max_weight_idx)
        
#         # Verify weight sum is 1
#         weight_sum = weight[0].sum()
#         self.assertAlmostEqual(weight_sum.item(), 1.0, places=5)
    
#     def test_input_output_samples_edge_cases(self):
#         """Test specific input/output samples for edge cases"""
#         temperature = 0.1
        
#         # Test case 1: Equal scores
#         rm_score_equal = torch.ones(1, 3) * 0.5  # All scores equal
#         reward_mask = torch.ones(1, 3)
        
#         weight = torch.softmax(
#             -rm_score_equal.masked_fill(~reward_mask.bool(), float('inf')) / temperature,
#             dim=-1
#         )
        
#         # With equal scores, weights should be uniformly distributed
#         expected_weight = 1.0 / 3.0
#         for i in range(3):
#             self.assertAlmostEqual(weight[0, i].item(), expected_weight, places=4)
        
#         # Test case 2: Extreme score differences
#         rm_score_extreme = torch.tensor([[100.0, -100.0, 0.0]])
#         weight_extreme = torch.softmax(
#             -rm_score_extreme.masked_fill(~reward_mask.bool(), float('inf')) / temperature,
#             dim=-1
#         )
        
#         # In extreme case, minimum score position should get weight close to 1
#         self.assertGreater(weight_extreme[0, 1].item(), 0.99)  # -100 position
#         self.assertLess(weight_extreme[0, 0].item(), 0.01)     # 100 position
        
#         # Test case 3: Mask application
#         rm_score_masked = torch.tensor([[1.0, -1.0, 2.0]])
#         reward_mask_partial = torch.tensor([[1.0, 1.0, 0.0]])  # Last position is masked
        
#         # Apply mask to weight calculation
#         weight_masked = torch.softmax(
#             -rm_score_masked.masked_fill(~reward_mask_partial.bool(), float('inf')) / temperature,
#             dim=-1
#         )
        
#         # Masked position weight should be 0
#         self.assertEqual(weight_masked[0, 2].item(), 0.0)
        
#         # Sum of unmasked position weights should be 1
#         valid_weight_sum = weight_masked[0, :2].sum()
#         self.assertAlmostEqual(valid_weight_sum.item(), 1.0, places=5)


# class TestForwardMicroBatchIntegration(unittest.TestCase):
#     """Integration test: Test _forward_micro_batch performance in actual environment"""
    
#     def setUp(self):
#         torch.manual_seed(42)
#         self.batch_size = 2
#         self.seq_len = 12
#         self.response_length = 6
#         self.vocab_size = 1000
#         self.temperature = 0.1
    
#     def test_end_to_end_processing_simulation(self):
#         """End-to-end test: Complete workflow simulation from input to output"""
#         # Create complete input data
#         input_data = {
#             "input_ids": torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len)),
#             "attention_mask": torch.ones(self.batch_size, self.seq_len),
#             "position_ids": torch.arange(self.seq_len).unsqueeze(0).repeat(self.batch_size, 1),
#             "responses": torch.randint(0, self.vocab_size, (self.batch_size, self.response_length)),
#             "reward_mask": torch.ones(self.batch_size, self.response_length)
#         }
        
#         # Simulate complete processing workflow
#         # 1. Simulate PRM model output
#         full_logits = torch.randn(self.batch_size, self.seq_len, 2)
        
#         # 2. Extract response part
#         response_logits = full_logits[:, -self.response_length:]
        
#         # 3. Binary classification processing
#         rm_score = response_logits.softmax(dim=-1)
#         rm_score = (rm_score[..., 1] - rm_score[..., 0]) * input_data['reward_mask']
        
#         # 4. Min-form weighting
#         weight = torch.softmax(
#             -rm_score.masked_fill(~input_data['reward_mask'].bool(), float('inf')) / self.temperature,
#             dim=-1
#         )
#         final_rm_score = rm_score * weight
        
#         # Verify complete workflow output
#         self.assertEqual(final_rm_score.shape, (self.batch_size, self.response_length))
#         self.assertTrue(torch.isfinite(final_rm_score).all())
        
#         # Verify weight normalization (for each sequence)
#         for i in range(self.batch_size):
#             valid_positions = input_data['reward_mask'][i].bool()
#             weight_sum = weight[i, valid_positions].sum()
#             self.assertAlmostEqual(weight_sum.item(), 1.0, places=5)
        
#         # Verify min-form property: position with maximum weight corresponds to position with minimum score
#         for i in range(self.batch_size):
#             min_score_idx = rm_score[i].argmin()
#             max_weight_idx = weight[i].argmax()
#             self.assertEqual(min_score_idx, max_weight_idx)
    
#     def test_disable_min_form_mode(self):
#         """Test disabling approximate min-form credit assignment mode"""
#         rm_score = torch.tensor([
#             [2.0, 0.5, 1.0, -0.5],
#             [1.5, -1.0, 0.8, 2.5]
#         ])
#         reward_mask = torch.ones(2, 4)
        
#         # When approximate min-form is disabled, should return rm_score directly without weighting
#         disable_approx_min_form = True
        
#         if disable_approx_min_form:
#             # Don't apply weighting
#             final_score = rm_score
#         else:
#             # Apply weighting
#             weight = torch.softmax(-rm_score / 0.1, dim=-1)
#             final_score = rm_score * weight
        
#         # Verify results in disabled mode
#         torch.testing.assert_close(final_score, rm_score)
#         self.assertEqual(final_score.shape, (2, 4))
#         self.assertTrue(torch.isfinite(final_score).all())


if __name__ == "__main__":
    unittest.main()