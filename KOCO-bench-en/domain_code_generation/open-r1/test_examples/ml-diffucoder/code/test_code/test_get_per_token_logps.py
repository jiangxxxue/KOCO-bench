import unittest
import torch
import sys
import os
from unittest.mock import MagicMock, patch

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the actual class we want to test
from open_r1.coupled_grpo import DiffuGRPOTrainer


class TestGetPerTokenLogps(unittest.TestCase):
    """Test DiffuGRPOTrainer._get_per_token_logps function - Calculate token-level log probabilities using Coupled Sampling"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 1000
        self.mask_token_id = 50256
    
    def test_get_per_token_logps_input_validation(self):
        """Test input validation"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        # Test incorrect input dimensions
        input_ids_2d = torch.randint(0, self.vocab_size, (4, 20))
        
        with self.assertRaises(ValueError) as context:
            trainer._get_per_token_logps(None, input_ids_2d, 10, [42])
        
        self.assertIn("3 dimensions", str(context.exception))
    
    def test_get_per_token_logps_mask_seeds_validation(self):
        """Test mask_seeds length validation"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        num_iterations = 3
        batch_size = 2
        seq_len = 15
        
        input_ids = torch.randint(0, self.vocab_size, (num_iterations, batch_size, seq_len))
        
        # Provide incorrect number of mask_seeds
        wrong_seeds = [42, 43]  # Should be 3
        
        with self.assertRaises(ValueError) as context:
            trainer._get_per_token_logps(None, input_ids, 10, wrong_seeds)
        
        self.assertIn("mask_seeds length", str(context.exception))
    
    def test_get_per_token_logps_output_shape(self):
        """Test correctness of output shape"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        # Mock forward_process and get_logits methods
        def mock_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            b, l = batch.shape
            noisy_batch = [batch.clone() for _ in range(3)]
            weights = [1.0, 1.5, 2.0]
            completion_mask = torch.ones(b, l, dtype=torch.bool)
            return noisy_batch, weights, completion_mask
        
        trainer.forward_process = mock_forward_process
        
        def mock_get_logits(model, batch):
            b, l = batch.shape
            return torch.randn(b, l, self.vocab_size)
        
        trainer.get_logits = mock_get_logits
        
        # 创建输入
        num_iterations = 2
        batch_size = 3
        seq_len = 20
        logits_to_keep = 10
        
        input_ids = torch.randint(0, self.vocab_size, (num_iterations, batch_size, seq_len))
        mask_seeds = [42, 43]
        
        # print(input_ids)
        # Mock model
        mock_model = MagicMock()
        
        # Call function
        result = trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
        # print(result)
        # Verify output shape
        expected_shape = (batch_size, num_iterations, logits_to_keep)
        self.assertEqual(result.shape, expected_shape,
                        f"Output shape should be {expected_shape}")
        
        # Verify output is finite
        self.assertTrue(torch.isfinite(result).all(),
                       "Output should all be finite values")
    
    def test_get_per_token_logps_prompt_index_creation(self):
        """Test correct creation of prompt_index"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        # Record prompt_index when forward_process is called
        captured_prompt_indices = []
        
        def mock_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            captured_prompt_indices.append(prompt_index.clone())
            b, l = batch.shape
            noisy_batch = [batch.clone() for _ in range(3)]
            weights = [1.0, 1.5, 2.0]
            completion_mask = torch.ones(b, l, dtype=torch.bool)
            return noisy_batch, weights, completion_mask
        
        trainer.forward_process = mock_forward_process
        trainer.get_logits = lambda model, batch: torch.randn(batch.size(0), batch.size(1), self.vocab_size)
        
        num_iterations = 1
        batch_size = 2
        seq_len = 15
        logits_to_keep = 8
        prompt_length = seq_len - logits_to_keep
        
        input_ids = torch.randint(0, self.vocab_size, (num_iterations, batch_size, seq_len))
        mask_seeds = [42]
        
        mock_model = MagicMock()
        trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
        
        # Verify prompt_index
        self.assertEqual(len(captured_prompt_indices), num_iterations)
        prompt_index = captured_prompt_indices[0]
        
        # First prompt_length positions should be True
        self.assertTrue(prompt_index[:prompt_length].all(),
                       "Prompt part should all be marked as True")
        
        # Last logits_to_keep positions should be False
        self.assertFalse(prompt_index[prompt_length:].any(),
                        "Completion part should all be marked as False")
    
    def test_get_per_token_logps_multiple_iterations(self):
        """Test handling of multiple iterations"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        # Record number of times forward_process is called and seeds used
        call_count = [0]
        used_seeds = []
        
        def mock_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            call_count[0] += 1
            used_seeds.append(seed)
            b, l = batch.shape
            noisy_batch = [batch.clone() for _ in range(3)]
            weights = [1.0, 1.5, 2.0]
            completion_mask = torch.ones(b, l, dtype=torch.bool)
            return noisy_batch, weights, completion_mask
        
        trainer.forward_process = mock_forward_process
        trainer.get_logits = lambda model, batch: torch.randn(batch.size(0), batch.size(1), self.vocab_size)
        
        num_iterations = 3
        batch_size = 2
        seq_len = 15
        logits_to_keep = 8
        
        input_ids = torch.randint(0, self.vocab_size, (num_iterations, batch_size, seq_len))
        mask_seeds = [42, 43, 44]
        
        mock_model = MagicMock()
        result = trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
        
        # Verify forward_process was called num_iterations times
        self.assertEqual(call_count[0], num_iterations,
                        f"forward_process should be called {num_iterations} times")
        
        # Verify correct seeds were used
        self.assertEqual(used_seeds, mask_seeds,
                        "Should use provided mask_seeds")
        
        # Verify output shape
        self.assertEqual(result.shape, (batch_size, num_iterations, logits_to_keep))
    
    def test_get_per_token_logps_batch_concatenation(self):
        """Test correctness of batch concatenation"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        # Record batch size passed to get_logits
        get_logits_batch_sizes = []
        
        def mock_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            b, l = batch.shape
            noisy_batch = [batch.clone() for _ in range(3)]
            weights = [1.0, 1.5, 2.0]
            completion_mask = torch.ones(b, l, dtype=torch.bool)
            return noisy_batch, weights, completion_mask
        
        def mock_get_logits(model, batch):
            get_logits_batch_sizes.append(batch.size(0))
            b, l = batch.shape
            return torch.randn(b, l, self.vocab_size)
        
        trainer.forward_process = mock_forward_process
        trainer.get_logits = mock_get_logits
        
        num_iterations = 2
        batch_size = 3
        seq_len = 15
        logits_to_keep = 8
        
        input_ids = torch.randint(0, self.vocab_size, (num_iterations, batch_size, seq_len))
        mask_seeds = [42, 43]
        
        mock_model = MagicMock()
        trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
        
        # Verify batch size received by get_logits
        # Should be num_iterations * 3 * batch_size
        expected_batch_size = num_iterations * 3 * batch_size
        self.assertEqual(get_logits_batch_sizes[0], expected_batch_size,
                        f"get_logits should receive batch of size {expected_batch_size}")
    
    def test_get_per_token_logps_completion_extraction(self):
        """Test correct extraction of completion part"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        def mock_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            b, l = batch.shape
            noisy_batch = [batch.clone() for _ in range(3)]
            weights = [1.0, 1.5, 2.0]
            completion_mask = torch.ones(b, l, dtype=torch.bool)
            return noisy_batch, weights, completion_mask
        
        trainer.forward_process = mock_forward_process
        trainer.get_logits = lambda model, batch: torch.randn(batch.size(0), batch.size(1), self.vocab_size)
        
        num_iterations = 1
        batch_size = 2
        seq_len = 20
        logits_to_keep = 5  # Only keep logits for last 5 tokens
        
        input_ids = torch.randint(0, self.vocab_size, (num_iterations, batch_size, seq_len))
        mask_seeds = [42]
        
        mock_model = MagicMock()
        result = trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
        
        # Verify only logits_to_keep tokens' log probabilities are returned
        self.assertEqual(result.shape[2], logits_to_keep,
                        f"Should only return log probabilities for {logits_to_keep} tokens")
    
    def test_get_per_token_logps_weight_propagation(self):
        """Test correct propagation of weights"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        # Record weights returned by forward_process
        returned_weights = []
        
        def mock_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            b, l = batch.shape
            noisy_batch = [batch.clone() for _ in range(3)]
            weights = [1.0, 2.0, 3.0]  # Use specific weights
            returned_weights.append(weights)
            completion_mask = torch.ones(b, l, dtype=torch.bool)
            return noisy_batch, weights, completion_mask
        
        trainer.forward_process = mock_forward_process
        trainer.get_logits = lambda model, batch: torch.randn(batch.size(0), batch.size(1), self.vocab_size)
        
        num_iterations = 2
        batch_size = 2
        seq_len = 15
        logits_to_keep = 8
        
        input_ids = torch.randint(0, self.vocab_size, (num_iterations, batch_size, seq_len))
        mask_seeds = [42, 43]
        
        mock_model = MagicMock()
        
        # Mock selective_log_softmax to verify weights
        with patch('open_r1.coupled_grpo.selective_log_softmax') as mock_selective:
            mock_selective.return_value = torch.randn(num_iterations * batch_size, logits_to_keep)
            
            trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
            
            # Verify selective_log_softmax was called
            self.assertTrue(mock_selective.called)
            
            # Get weights parameter passed to selective_log_softmax
            call_args = mock_selective.call_args
            weights_arg = call_args[0][2]  # Third positional argument is weights
            
            # Verify length of weights
            expected_weight_length = num_iterations * 3
            self.assertEqual(len(weights_arg), expected_weight_length,
                           f"Weight length should be {expected_weight_length}")
    
    def test_get_per_token_logps_output_permutation(self):
        """Test correct permutation of output dimensions"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        def mock_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            b, l = batch.shape
            noisy_batch = [batch.clone() for _ in range(3)]
            weights = [1.0, 1.5, 2.0]
            completion_mask = torch.ones(b, l, dtype=torch.bool)
            return noisy_batch, weights, completion_mask
        
        trainer.forward_process = mock_forward_process
        trainer.get_logits = lambda model, batch: torch.randn(batch.size(0), batch.size(1), self.vocab_size)
        
        num_iterations = 3
        batch_size = 4
        seq_len = 15
        logits_to_keep = 8
        
        input_ids = torch.randint(0, self.vocab_size, (num_iterations, batch_size, seq_len))
        mask_seeds = [42, 43, 44]
        
        mock_model = MagicMock()
        result = trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
        
        # Verify output dimension order: [batch_size, num_iterations, logits_to_keep]
        self.assertEqual(result.shape[0], batch_size, "First dimension should be batch_size")
        self.assertEqual(result.shape[1], num_iterations, "Second dimension should be num_iterations")
        self.assertEqual(result.shape[2], logits_to_keep, "Third dimension should be logits_to_keep")
    
    def test_get_per_token_logps_dtype_conversion(self):
        """Test output data type conversion"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        def mock_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            b, l = batch.shape
            noisy_batch = [batch.clone() for _ in range(3)]
            weights = [1.0, 1.5, 2.0]
            completion_mask = torch.ones(b, l, dtype=torch.bool)
            return noisy_batch, weights, completion_mask
        
        trainer.forward_process = mock_forward_process
        trainer.get_logits = lambda model, batch: torch.randn(batch.size(0), batch.size(1), self.vocab_size)
        
        num_iterations = 1
        batch_size = 2
        seq_len = 15
        logits_to_keep = 8
        
        input_ids = torch.randint(0, self.vocab_size, (num_iterations, batch_size, seq_len))
        mask_seeds = [42]
        
        mock_model = MagicMock()
        result = trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
        
        # Verify output data type is float32
        self.assertEqual(result.dtype, torch.float32,
                        "Output should be converted to float32 type")
    
    def test_get_per_token_logps_edge_cases(self):
        """Test edge cases"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        def mock_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            b, l = batch.shape
            noisy_batch = [batch.clone() for _ in range(3)]
            weights = [1.0, 1.5, 2.0]
            completion_mask = torch.ones(b, l, dtype=torch.bool)
            return noisy_batch, weights, completion_mask
        
        trainer.forward_process = mock_forward_process
        trainer.get_logits = lambda model, batch: torch.randn(batch.size(0), batch.size(1), self.vocab_size)
        
        mock_model = MagicMock()
        
        # Test 1: Single iteration, single sample
        input_ids_single = torch.randint(0, self.vocab_size, (1, 1, 10))
        result_single = trainer._get_per_token_logps(mock_model, input_ids_single, 5, [42])
        self.assertEqual(result_single.shape, (1, 1, 5))
        
        # Test 2: logits_to_keep equals seq_len (no prompt)
        input_ids_no_prompt = torch.randint(0, self.vocab_size, (1, 2, 10))
        result_no_prompt = trainer._get_per_token_logps(mock_model, input_ids_no_prompt, 10, [42])
        self.assertEqual(result_no_prompt.shape, (2, 1, 10))
        
        # Test 3: logits_to_keep is 1 (only one completion token)
        input_ids_one_token = torch.randint(0, self.vocab_size, (1, 2, 10))
        result_one_token = trainer._get_per_token_logps(mock_model, input_ids_one_token, 1, [42])
        self.assertEqual(result_one_token.shape, (2, 1, 1))


class TestGetPerTokenLogpsIntegration(unittest.TestCase):
    """Integration test: Test _get_per_token_logps performance in actual training workflow"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_end_to_end_workflow(self):
        """End-to-end test: Complete workflow from input to output"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = 50256
        
        # Use real forward_process implementation
        def real_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            import random
            if seed is not None:
                random.seed(seed)
                torch.manual_seed(seed)
            
            b, l = batch.shape
            noisy_batch = []
            mask_ratio = random.uniform(0.2, 0.8)
            random_matrix = torch.rand((b, l), device=batch.device)
            
            # Version 1: mask all completion
            is_mask = ~prompt_index
            noisy_batch.append(torch.where(is_mask, mask_id, batch))
            
            # Version 2: partial mask
            is_mask = ~prompt_index & (random_matrix < mask_ratio)
            completion_mask = is_mask
            noisy_batch.append(torch.where(is_mask, mask_id, batch))
            
            # Version 3: reverse mask
            is_mask = ~prompt_index & (random_matrix > mask_ratio)
            noisy_batch.append(torch.where(is_mask, mask_id, batch))
            
            return noisy_batch, [1, 1/mask_ratio, 1/(1-mask_ratio)], completion_mask
        
        trainer.forward_process = real_forward_process
        trainer.get_logits = lambda model, batch: torch.randn(batch.size(0), batch.size(1), 1000)
        
        # Create input
        num_iterations = 2
        batch_size = 3
        seq_len = 20
        logits_to_keep = 10
        
        input_ids = torch.randint(0, 1000, (num_iterations, batch_size, seq_len))
        mask_seeds = [42, 43]
        
        mock_model = MagicMock()
        
        # Call function
        result = trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
        
        # Verify output
        self.assertEqual(result.shape, (batch_size, num_iterations, logits_to_keep))
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result.dtype, torch.float32)
    
    def test_consistency_across_calls(self):
        """Test consistency when using the same seed"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = 50256
        
        def mock_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            b, l = batch.shape
            noisy_batch = [batch.clone() for _ in range(3)]
            weights = [1.0, 1.5, 2.0]
            completion_mask = torch.ones(b, l, dtype=torch.bool)
            return noisy_batch, weights, completion_mask
        
        trainer.forward_process = mock_forward_process
        
        # Use fixed logits to ensure reproducibility
        def deterministic_get_logits(model, batch):
            torch.manual_seed(100)
            return torch.randn(batch.size(0), batch.size(1), 1000)
        
        trainer.get_logits = deterministic_get_logits
        
        num_iterations = 2
        batch_size = 2
        seq_len = 15
        logits_to_keep = 8
        
        input_ids = torch.randint(0, 1000, (num_iterations, batch_size, seq_len))
        mask_seeds = [42, 43]
        
        mock_model = MagicMock()
        
        # Call twice
        result1 = trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
        result2 = trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
        
        # Verify results are consistent
        torch.testing.assert_close(result1, result2,
                                  msg="Using same input should produce same output")


if __name__ == "__main__":
    unittest.main()
