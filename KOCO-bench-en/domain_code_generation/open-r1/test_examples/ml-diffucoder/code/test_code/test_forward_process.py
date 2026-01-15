import unittest
import torch
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the actual class we want to test
from open_r1.coupled_grpo import DiffuGRPOTrainer


class TestForwardProcess(unittest.TestCase):
    """Test DiffuGRPOTrainer.forward_process function - mask generation for Coupled Sampling scheme"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.seq_len = 20
        self.mask_id = 50256  # Assumed mask token ID
        
    def test_forward_process_basic_output_structure(self):
        """Test basic output structure of forward_process"""
        # Create test data
        batch = torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113, 378,  14, 210, 954,
         231, 572, 315, 295, 567, 706],
        [749, 876,  73, 111, 899, 213, 541, 769, 287, 219, 372, 880, 475, 329,
         733, 564, 739, 376, 632,  10],
        [186, 822, 577, 519, 707, 123, 143, 294, 693, 677,  70, 709, 370, 739,
          86, 499, 215, 384, 978, 908],
        [266, 630, 540, 560, 470, 561, 323, 720,  11, 461, 177, 289, 184, 553,
         548, 409, 483, 507, 558, 491]])
        prompt_length = 10
        prompt_index = torch.zeros(self.seq_len, dtype=torch.bool)
        prompt_index[:prompt_length] = True
        
        # print(batch)
        # Create a simplified trainer instance for testing
        trainer = object.__new__(DiffuGRPOTrainer)
        
        # Call forward_process
        noisy_batch, weights, completion_mask = trainer.forward_process(
            batch, prompt_index, self.mask_id, seed=42
        )
        
        # print(noisy_batch)
        # print(weights)
        # print(completion_mask)

        expected_noisy_batch = [torch.tensor([[  542,    67,   876,   414,    26,   335,   620,   924,   950,   113,
         50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256],
        [  749,   876,    73,   111,   899,   213,   541,   769,   287,   219,
         50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256],
        [  186,   822,   577,   519,   707,   123,   143,   294,   693,   677,
         50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256],
        [  266,   630,   540,   560,   470,   561,   323,   720,    11,   461,
         50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256]]), torch.tensor([[  542,    67,   876,   414,    26,   335,   620,   924,   950,   113,
           378,    14,   210, 50256,   231, 50256,   315, 50256, 50256,   706],
        [  749,   876,    73,   111,   899,   213,   541,   769,   287,   219,
           372, 50256,   475, 50256, 50256,   564, 50256,   376, 50256, 50256],
        [  186,   822,   577,   519,   707,   123,   143,   294,   693,   677,
         50256, 50256, 50256,   739,    86,   499,   215, 50256,   978, 50256],
        [  266,   630,   540,   560,   470,   561,   323,   720,    11,   461,
           177,   289,   184, 50256, 50256, 50256, 50256,   507,   558,   491]]), torch.tensor([[  542,    67,   876,   414,    26,   335,   620,   924,   950,   113,
         50256, 50256, 50256,   954, 50256,   572, 50256,   295,   567, 50256],
        [  749,   876,    73,   111,   899,   213,   541,   769,   287,   219,
         50256,   880, 50256,   329,   733, 50256,   739, 50256,   632,    10],
        [  186,   822,   577,   519,   707,   123,   143,   294,   693,   677,
            70,   709,   370, 50256, 50256, 50256, 50256,   384, 50256,   908],
        [  266,   630,   540,   560,   470,   561,   323,   720,    11,   461,
         50256, 50256, 50256,   553,   548,   409,   483, 50256, 50256, 50256]])]
        
        expected_weights = [1, 1.7133377614866954, 2.4018604565610837]

        expected_completion_mask = torch.tensor([[False, False, False, False, False, False, False, False, False, False,
         False, False, False,  True, False,  True, False,  True,  True, False],
        [False, False, False, False, False, False, False, False, False, False,
         False,  True, False,  True,  True, False,  True, False,  True,  True],
        [False, False, False, False, False, False, False, False, False, False,
          True,  True,  True, False, False, False, False,  True, False,  True],
        [False, False, False, False, False, False, False, False, False, False,
         False, False, False,  True,  True,  True,  True, False, False, False]])

        for i in range(len(noisy_batch)):
            torch.testing.assert_close(noisy_batch[i], expected_noisy_batch[i])

        self.assertAlmostEqual(weights, expected_weights, places=4)

        for i in range(len(completion_mask)):
            torch.testing.assert_close(completion_mask[i], expected_completion_mask[i])

    
    def test_forward_process_version1_masks_all_completion(self):
        """Test Version 1 masks all completion tokens"""
        batch = torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113, 378,  14, 210, 954,
         231, 572, 315, 295, 567, 706],
        [749, 876,  73, 111, 899, 213, 541, 769, 287, 219, 372, 880, 475, 329,
         733, 564, 739, 376, 632,  10],
        [186, 822, 577, 519, 707, 123, 143, 294, 693, 677,  70, 709, 370, 739,
          86, 499, 215, 384, 978, 908],
        [266, 630, 540, 560, 470, 561, 323, 720,  11, 461, 177, 289, 184, 553,
         548, 409, 483, 507, 558, 491]])
        prompt_length = 10
        prompt_index = torch.zeros(self.seq_len, dtype=torch.bool)
        prompt_index[:prompt_length] = True
        
        # print(batch)
        trainer = object.__new__(DiffuGRPOTrainer)
        noisy_batch, weights, completion_mask = trainer.forward_process(
            batch, prompt_index, self.mask_id, seed=42
        )
        
        version1 = noisy_batch[0]
        
        expected_version1 = torch.tensor([[  542,    67,   876,   414,    26,   335,   620,   924,   950,   113,
         50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256],
        [  749,   876,    73,   111,   899,   213,   541,   769,   287,   219,
         50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256],
        [  186,   822,   577,   519,   707,   123,   143,   294,   693,   677,
         50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256],
        [  266,   630,   540,   560,   470,   561,   323,   720,    11,   461,
         50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256]])
        # print(version1)\

        torch.testing.assert_close(version1, expected_version1)
        # Verify Version 1: all completion tokens should be masked
        for i in range(self.batch_size):
            # Prompt part should remain unchanged
            torch.testing.assert_close(version1[i, :prompt_length], batch[i, :prompt_length])
            
            # Completion part should all be masked
            self.assertTrue((version1[i, prompt_length:] == self.mask_id).all(),
                          "Version 1 should mask all completion tokens")
    
    def test_forward_process_version2_partial_masking(self):
        """Test Version 2 randomly masks by mask_ratio"""
        batch = torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113, 378,  14, 210, 954,
         231, 572, 315, 295, 567, 706],
        [749, 876,  73, 111, 899, 213, 541, 769, 287, 219, 372, 880, 475, 329,
         733, 564, 739, 376, 632,  10],
        [186, 822, 577, 519, 707, 123, 143, 294, 693, 677,  70, 709, 370, 739,
          86, 499, 215, 384, 978, 908],
        [266, 630, 540, 560, 470, 561, 323, 720,  11, 461, 177, 289, 184, 553,
         548, 409, 483, 507, 558, 491]])
        prompt_length = 10
        prompt_index = torch.zeros(self.seq_len, dtype=torch.bool)
        prompt_index[:prompt_length] = True
        
        # print(batch)
        trainer = object.__new__(DiffuGRPOTrainer)
        noisy_batch, weights, completion_mask = trainer.forward_process(
            batch, prompt_index, self.mask_id, seed=42
        )
        
        version2 = noisy_batch[1]
        # print(version2)

        expected_version2 = torch.tensor([[  542,    67,   876,   414,    26,   335,   620,   924,   950,   113,
           378,    14,   210, 50256,   231, 50256,   315, 50256, 50256,   706],
        [  749,   876,    73,   111,   899,   213,   541,   769,   287,   219,
           372, 50256,   475, 50256, 50256,   564, 50256,   376, 50256, 50256],
        [  186,   822,   577,   519,   707,   123,   143,   294,   693,   677,
         50256, 50256, 50256,   739,    86,   499,   215, 50256,   978, 50256],
        [  266,   630,   540,   560,   470,   561,   323,   720,    11,   461,
           177,   289,   184, 50256, 50256, 50256, 50256,   507,   558,   491]])
        # print(version2)

        torch.testing.assert_close(version2, expected_version2)
        # Verify Version 2: partial completion tokens are masked
        for i in range(self.batch_size):
            # Prompt part should remain unchanged
            torch.testing.assert_close(version2[i, :prompt_length], batch[i, :prompt_length])
            
            # Completion part should have some masked, some remain original
            completion_tokens = version2[i, prompt_length:]
            masked_count = (completion_tokens == self.mask_id).sum().item()
            unmasked_count = (completion_tokens != self.mask_id).sum().item()
            
            # Should have both masked and unmasked tokens (in most cases)
            self.assertGreater(masked_count, 0, "Version 2 should have masked tokens")
            self.assertGreater(unmasked_count, 0, "Version 2 should have unmasked tokens")
    
    def test_forward_process_version3_reverse_masking(self):
        """Test Version 3 reverse masking (complementary to Version 2)"""
        batch = torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113, 378,  14, 210, 954,
         231, 572, 315, 295, 567, 706],
        [749, 876,  73, 111, 899, 213, 541, 769, 287, 219, 372, 880, 475, 329,
         733, 564, 739, 376, 632,  10],
        [186, 822, 577, 519, 707, 123, 143, 294, 693, 677,  70, 709, 370, 739,
          86, 499, 215, 384, 978, 908],
        [266, 630, 540, 560, 470, 561, 323, 720,  11, 461, 177, 289, 184, 553,
         548, 409, 483, 507, 558, 491]])
        prompt_length = 10
        prompt_index = torch.zeros(self.seq_len, dtype=torch.bool)
        prompt_index[:prompt_length] = True

        # print(batch)
        
        trainer = object.__new__(DiffuGRPOTrainer)
        noisy_batch, weights, completion_mask = trainer.forward_process(
            batch, prompt_index, self.mask_id, seed=42
        )
        
        version2 = noisy_batch[1]
        version3 = noisy_batch[2]
        # print(version2)
        # print(version3)

        expected_version2 = torch.tensor([[  542,    67,   876,   414,    26,   335,   620,   924,   950,   113,
           378,    14,   210, 50256,   231, 50256,   315, 50256, 50256,   706],
        [  749,   876,    73,   111,   899,   213,   541,   769,   287,   219,
           372, 50256,   475, 50256, 50256,   564, 50256,   376, 50256, 50256],
        [  186,   822,   577,   519,   707,   123,   143,   294,   693,   677,
         50256, 50256, 50256,   739,    86,   499,   215, 50256,   978, 50256],
        [  266,   630,   540,   560,   470,   561,   323,   720,    11,   461,
           177,   289,   184, 50256, 50256, 50256, 50256,   507,   558,   491]])

        expected_version3 = torch.tensor([[  542,    67,   876,   414,    26,   335,   620,   924,   950,   113,
         50256, 50256, 50256,   954, 50256,   572, 50256,   295,   567, 50256],
        [  749,   876,    73,   111,   899,   213,   541,   769,   287,   219,
         50256,   880, 50256,   329,   733, 50256,   739, 50256,   632,    10],
        [  186,   822,   577,   519,   707,   123,   143,   294,   693,   677,
            70,   709,   370, 50256, 50256, 50256, 50256,   384, 50256,   908],
        [  266,   630,   540,   560,   470,   561,   323,   720,    11,   461,
         50256, 50256, 50256,   553,   548,   409,   483, 50256, 50256, 50256]])

        torch.testing.assert_close(version2, expected_version2)
        torch.testing.assert_close(version3, expected_version3)
        # Verify complementarity of Version 2 and Version 3
        for i in range(self.batch_size):
            for j in range(prompt_length, self.seq_len):
                # If a position is masked in Version 2, it should not be masked in Version 3
                # If a position is not masked in Version 2, it should be masked in Version 3
                v2_masked = (version2[i, j] == self.mask_id)
                v3_masked = (version3[i, j] == self.mask_id)
                
                # They should be complementary (cannot both be True or both be False)
                self.assertNotEqual(v2_masked.item(), v3_masked.item(),
                                  f"Position {j} should be complementary in Version 2 and 3")
    
    def test_forward_process_weights_calculation(self):
        """Test correctness of weight calculation"""
        batch = torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113, 378,  14, 210, 954,
         231, 572, 315, 295, 567, 706],
        [749, 876,  73, 111, 899, 213, 541, 769, 287, 219, 372, 880, 475, 329,
         733, 564, 739, 376, 632,  10],
        [186, 822, 577, 519, 707, 123, 143, 294, 693, 677,  70, 709, 370, 739,
          86, 499, 215, 384, 978, 908],
        [266, 630, 540, 560, 470, 561, 323, 720,  11, 461, 177, 289, 184, 553,
         548, 409, 483, 507, 558, 491]])
        prompt_length = 10
        prompt_index = torch.zeros(self.seq_len, dtype=torch.bool)
        prompt_index[:prompt_length] = True
        
        # print(batch)
        trainer = object.__new__(DiffuGRPOTrainer)
        noisy_batch, weights, completion_mask = trainer.forward_process(
            batch, prompt_index, self.mask_id, seed=42
        )
        
        # print(weights)

        expected_weights = [1, 1.7133377614866954, 2.4018604565610837]
        self.assertAlmostEqual(weights, expected_weights, places=4)

        # Verify weight structure
        self.assertEqual(weights[0], 1, "Version 1 weight should be 1")
        
        # Version 2 and 3 weights should be reciprocal of mask_ratio
        # weights[1] = 1/mask_ratio, weights[2] = 1/(1-mask_ratio)
        # Therefore 1/weights[1] + 1/weights[2] should equal 1
        mask_ratio = 1.0 / weights[1]
        expected_weight3 = 1.0 / (1.0 - mask_ratio)
        
        self.assertAlmostEqual(weights[2], expected_weight3, places=5,
                              msg="Version 3 weight should be 1/(1-mask_ratio)")
        
        # Verify mask_ratio is in [0.2, 0.8] range
        self.assertGreaterEqual(mask_ratio, 0.2, "mask_ratio should be >= 0.2")
        self.assertLessEqual(mask_ratio, 0.8, "mask_ratio should be <= 0.8")
    
    def test_forward_process_completion_mask_consistency(self):
        """Test consistency of completion_mask with Version 2"""
        batch = torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113, 378,  14, 210, 954,
         231, 572, 315, 295, 567, 706],
        [749, 876,  73, 111, 899, 213, 541, 769, 287, 219, 372, 880, 475, 329,
         733, 564, 739, 376, 632,  10],
        [186, 822, 577, 519, 707, 123, 143, 294, 693, 677,  70, 709, 370, 739,
          86, 499, 215, 384, 978, 908],
        [266, 630, 540, 560, 470, 561, 323, 720,  11, 461, 177, 289, 184, 553,
         548, 409, 483, 507, 558, 491]])
        prompt_length = 10
        prompt_index = torch.zeros(self.seq_len, dtype=torch.bool)
        prompt_index[:prompt_length] = True
        
        # print(batch)
        trainer = object.__new__(DiffuGRPOTrainer)
        noisy_batch, weights, completion_mask = trainer.forward_process(
            batch, prompt_index, self.mask_id, seed=42
        )
        
        version2 = noisy_batch[1]
        # print(version2)
        expected_version2 = torch.tensor([[  542,    67,   876,   414,    26,   335,   620,   924,   950,   113,
           378,    14,   210, 50256,   231, 50256,   315, 50256, 50256,   706],
        [  749,   876,    73,   111,   899,   213,   541,   769,   287,   219,
           372, 50256,   475, 50256, 50256,   564, 50256,   376, 50256, 50256],
        [  186,   822,   577,   519,   707,   123,   143,   294,   693,   677,
         50256, 50256, 50256,   739,    86,   499,   215, 50256,   978, 50256],
        [  266,   630,   540,   560,   470,   561,   323,   720,    11,   461,
           177,   289,   184, 50256, 50256, 50256, 50256,   507,   558,   491]])
        # print(version2)

        torch.testing.assert_close(version2, expected_version2)
        # Verify completion_mask marks masked positions in Version 2
        for i in range(self.batch_size):
            for j in range(self.seq_len):
                is_masked_in_v2 = (version2[i, j] == self.mask_id)
                is_marked_in_mask = completion_mask[i, j].item()
                
                if j < prompt_length:
                    # Prompt part should not be marked
                    self.assertFalse(is_marked_in_mask,
                                   f"Prompt position {j} should not be marked in completion_mask")
                else:
                    # Completion part: completion_mask should be consistent with Version 2 mask status
                    self.assertEqual(is_masked_in_v2, is_marked_in_mask,
                                   f"completion_mask at position {j} should be consistent with Version 2")
    
    def test_forward_process_seed_reproducibility(self):
        """Test reproducibility of random seed"""
        batch = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        prompt_length = 10
        prompt_index = torch.zeros(self.seq_len, dtype=torch.bool)
        prompt_index[:prompt_length] = True
        
        trainer = object.__new__(DiffuGRPOTrainer)
        
        # Generate twice with same seed
        noisy_batch1, weights1, mask1 = trainer.forward_process(
            batch, prompt_index, self.mask_id, seed=123
        )
        noisy_batch2, weights2, mask2 = trainer.forward_process(
            batch, prompt_index, self.mask_id, seed=123
        )
        
        # Verify results are identical
        for i in range(3):
            torch.testing.assert_close(noisy_batch1[i], noisy_batch2[i],
                                      msg=f"With same seed, Version {i+1} should be identical")
        
        self.assertEqual(weights1, weights2, "With same seed, weights should be identical")
        torch.testing.assert_close(mask1, mask2, msg="With same seed, completion_mask should be identical")
    
    def test_forward_process_prompt_preservation(self):
        """Test all versions preserve prompt part"""
        batch = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        prompt_length = 10
        prompt_index = torch.zeros(self.seq_len, dtype=torch.bool)
        prompt_index[:prompt_length] = True
        
        trainer = object.__new__(DiffuGRPOTrainer)
        noisy_batch, weights, completion_mask = trainer.forward_process(
            batch, prompt_index, self.mask_id, seed=42
        )
        
        # Verify all versions preserve prompt part
        for version_idx, noisy_seq in enumerate(noisy_batch):
            for i in range(self.batch_size):
                torch.testing.assert_close(
                    noisy_seq[i, :prompt_length],
                    batch[i, :prompt_length],
                    msg=f"Version {version_idx+1} should preserve prompt part"
                )
    
    def test_forward_process_edge_cases(self):
        """Test edge cases"""
        trainer = object.__new__(DiffuGRPOTrainer)
        
        # Test 1: Only prompt, no completion
        batch_prompt_only = torch.randint(0, 1000, (2, 10))
        prompt_index_all = torch.ones(10, dtype=torch.bool)
        
        noisy_batch, weights, mask = trainer.forward_process(
            batch_prompt_only, prompt_index_all, self.mask_id, seed=42
        )
        
        # All versions should be same as original batch (no completion tokens to mask)
        for version in noisy_batch:
            torch.testing.assert_close(version, batch_prompt_only)
        
        # Test 2: Only completion, no prompt
        batch_completion_only = torch.randint(0, 1000, (2, 10))
        prompt_index_none = torch.zeros(10, dtype=torch.bool)
        
        noisy_batch, weights, mask = trainer.forward_process(
            batch_completion_only, prompt_index_none, self.mask_id, seed=42
        )
        
        # Version 1 should be all masked
        self.assertTrue((noisy_batch[0] == self.mask_id).all(),
                       "When no prompt, Version 1 should be all masked")
        
        # Test 3: Single token
        batch_single = torch.randint(0, 1000, (1, 1))
        prompt_index_single = torch.zeros(1, dtype=torch.bool)
        
        noisy_batch, weights, mask = trainer.forward_process(
            batch_single, prompt_index_single, self.mask_id, seed=42
        )
        
        self.assertEqual(len(noisy_batch), 3)
        self.assertEqual(noisy_batch[0].shape, (1, 1))


class TestForwardProcessIntegration(unittest.TestCase):
    """Integration test: test forward_process performance in actual training workflow"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_coverage_property(self):
        """Test coverage property of Coupled Sampling: each token is computed in at least one version"""
        batch_size, seq_len = 3, 15
        prompt_length = 8
        mask_id = 50256
        
        batch = torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113, 378,  14, 210, 954,
         231],
        [572, 315, 295, 567, 706, 749, 876,  73, 111, 899, 213, 541, 769, 287,
         219],
        [372, 880, 475, 329, 733, 564, 739, 376, 632,  10, 186, 822, 577, 519,
         707]])
        prompt_index = torch.zeros(seq_len, dtype=torch.bool)
        prompt_index[:prompt_length] = True

        # print(batch)
        
        trainer = object.__new__(DiffuGRPOTrainer)
        noisy_batch, weights, completion_mask = trainer.forward_process(
            batch, prompt_index, mask_id, seed=42
        )
        # print(noisy_batch)

        expected_noisy_batch = [torch.tensor([[  542,    67,   876,   414,    26,   335,   620,   924, 50256, 50256,
         50256, 50256, 50256, 50256, 50256],
        [  572,   315,   295,   567,   706,   749,   876,    73, 50256, 50256,
         50256, 50256, 50256, 50256, 50256],
        [  372,   880,   475,   329,   733,   564,   739,   376, 50256, 50256,
         50256, 50256, 50256, 50256, 50256]]), torch.tensor([[  542,    67,   876,   414,    26,   335,   620,   924,   950, 50256,
           378,    14,   210, 50256,   231],
        [  572,   315,   295,   567,   706,   749,   876,    73,   111, 50256,
         50256, 50256, 50256, 50256, 50256],
        [  372,   880,   475,   329,   733,   564,   739,   376, 50256, 50256,
           186, 50256,   577,   519,   707]]), torch.tensor([[  542,    67,   876,   414,    26,   335,   620,   924, 50256,   113,
         50256, 50256, 50256,   954, 50256],
        [  572,   315,   295,   567,   706,   749,   876,    73, 50256,   899,
           213,   541,   769,   287,   219],
        [  372,   880,   475,   329,   733,   564,   739,   376,   632,    10,
         50256,   822, 50256, 50256, 50256]])]

        for i in range(len(noisy_batch)):
            torch.testing.assert_close(noisy_batch[i], expected_noisy_batch[i])
        # For each completion token, check it is unmasked in at least one version
        for i in range(batch_size):
            for j in range(prompt_length, seq_len):
                # Check this token's status in three versions
                v1_unmasked = (noisy_batch[0][i, j] != mask_id)
                v2_unmasked = (noisy_batch[1][i, j] != mask_id)
                v3_unmasked = (noisy_batch[2][i, j] != mask_id)
                
                # At least one version should have this token unmasked
                # Note: Version 1 always masks all completion tokens, so mainly check Version 2 and 3
                self.assertTrue(v2_unmasked or v3_unmasked,
                              f"Token at position {j} should be unmasked in at least one version")


if __name__ == "__main__":
    unittest.main()
