import unittest
import torch
import sys
import os
from unittest.mock import MagicMock, patch

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the actual class we want to test
from open_r1.coupled_grpo import DiffuGRPOTrainer


class TestGenerateAndScoreCompletions(unittest.TestCase):
    """Test DiffuGRPOTrainer._generate_and_score_completions function - generate code completions and calculate rewards"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 1000
        self.eos_token_id = 50256
    
    def _create_mock_trainer(self):
        """Create a configured mock trainer"""
        trainer = object.__new__(DiffuGRPOTrainer)
        
        # Mock processing_class
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = 50257
        trainer.processing_class.encode = lambda x: [self.eos_token_id]
        trainer.processing_class.batch_decode = lambda ids, **kwargs: ["generated code"] * ids.size(0)
        
        # Mock accelerator
        trainer.accelerator = MagicMock()
        trainer.accelerator.device = torch.device('cpu')
        trainer.accelerator.process_index = 0
        trainer.accelerator.gather_for_metrics = lambda x: x
        
        # Mock model
        trainer.model = MagicMock()
        trainer.model_wrapped = MagicMock()
        trainer.model.diffusion_generate = MagicMock()
        
        # Mock config - use real object instead of MagicMock
        class MockArgs:
            max_completion_length = 50
            diffusion_steps = 10
            generation_temperature = 0.7
            generation_batch_size = 4
            random_masking = True
            output_dir = "test_output"
            report_to = []  # Add report_to attribute
        
        trainer.args = MockArgs()
        
        # Mock other attributes
        trainer.max_prompt_length = None
        trainer.num_iterations = 2
        trainer.num_generations = 4
        trainer.beta = 0.0
        trainer.ref_model = None
        trainer.scale_rewards = True
        trainer.reward_funcs = [lambda **kwargs: [1.0] * len(kwargs['prompts'])]
        trainer.reward_processing_classes = [trainer.processing_class]
        trainer.reward_weights = torch.tensor([1.0])
        trainer.reward_func_names = ["test_reward"]
        trainer.control = MagicMock()
        trainer.control.should_evaluate = False
        trainer._step = 0  # Add missing _step attribute
        
        # Mock metrics
        trainer._metrics = {"train": {"completion_length": [], "zero_std_ratio": [], 
                                     "rewards/test_reward": [], "rewards/<lambda>": [], 
                                     "reward": [], "reward_std": []}}
        trainer._textual_logs = {"prompt": [], "completion": [], "rewards": {"test_reward": [], "<lambda>": []}}
        
        return trainer
    
    def test_generate_and_score_completions_basic_structure(self):
        """Test basic output structure"""
        trainer = self._create_mock_trainer()
        
        # Mock input - need to match num_generations
        inputs = [
            {"prompt": "Write a function"},
            {"prompt": "Implement sorting"},
            {"prompt": "Create a class"},
            {"prompt": "Define a method"}
        ]
        
        # Mock generation result
        batch_size = len(inputs)
        seq_len = 30
        vocab_size = self.vocab_size  # Capture to local variable
        trainer.model.diffusion_generate.return_value = MagicMock(
            sequences=torch.randint(0, vocab_size, (batch_size, seq_len))
        )
        
        # Mock _get_per_token_logps
        trainer._get_per_token_logps = lambda model, ids, keep, seeds: torch.randn(
            batch_size, trainer.num_iterations, 10
        )
        
        # Mock maybe_apply_chat_template
        with patch('open_r1.coupled_grpo.maybe_apply_chat_template') as mock_chat:
            mock_chat.return_value = {"prompt": "test"}
            
            # Mock unwrap_model_for_generation
            with patch('open_r1.coupled_grpo.unwrap_model_for_generation') as mock_unwrap:
                mock_unwrap.return_value.__enter__ = lambda self: trainer.model
                mock_unwrap.return_value.__exit__ = lambda self, *args: None
                
                # Mock Trainer._prepare_inputs
                with patch('transformers.Trainer._prepare_inputs') as mock_prepare:
                    mock_prepare.side_effect = lambda self, x: {
                        "input_ids": torch.randint(0, vocab_size, (batch_size, 10)),
                        "attention_mask": torch.ones(batch_size, 10)
                    }
                    
                    with patch('open_r1.coupled_grpo.gather') as mock_gather:
                        mock_gather.side_effect = lambda x: x
                        
                        with patch('open_r1.coupled_grpo.gather_object') as mock_gather_obj:
                            mock_gather_obj.side_effect = lambda x: x
                            
                            result = trainer._generate_and_score_completions(inputs)
        
        # Verify output structure
        self.assertIn("prompt_ids", result)
        self.assertIn("prompt_mask", result)
        self.assertIn("completion_ids", result)
        self.assertIn("completion_mask", result)
        self.assertIn("advantages", result)
        self.assertIn("mask_seeds", result)
    
    def test_generate_and_score_completions_prompt_processing(self):
        """Test prompt processing"""
        trainer = self._create_mock_trainer()
        
        # Need to match num_generations = 4
        inputs = [{"prompt": f"Test prompt {i}"} for i in range(4)]
        batch_size = len(inputs)
        vocab_size = self.vocab_size
        
        # Record processed prompts
        processed_prompts = []
        
        def mock_tokenizer_call(text, **kwargs):
            processed_prompts.extend(text)
            return {
                "input_ids": torch.randint(0, vocab_size, (len(text), 10)),
                "attention_mask": torch.ones(len(text), 10)
            }
        
        # Use side_effect instead of directly assigning __call__
        trainer.processing_class.side_effect = mock_tokenizer_call
        trainer.model.diffusion_generate.return_value = MagicMock(
            sequences=torch.randint(0, vocab_size, (batch_size, 30))
        )
        trainer._get_per_token_logps = lambda model, ids, keep, seeds: torch.randn(batch_size, trainer.num_iterations, 10)
        
        with patch('open_r1.coupled_grpo.maybe_apply_chat_template') as mock_chat:
            mock_chat.side_effect = lambda example, tokenizer: {"prompt": example["prompt"]}
            
            with patch('open_r1.coupled_grpo.unwrap_model_for_generation') as mock_unwrap:
                mock_unwrap.return_value.__enter__ = lambda self: trainer.model
                mock_unwrap.return_value.__exit__ = lambda self, *args: None
                
                with patch('transformers.Trainer._prepare_inputs') as mock_prepare:
                    mock_prepare.side_effect = lambda self, x: x
                    
                    with patch('open_r1.coupled_grpo.gather') as mock_gather:
                        mock_gather.side_effect = lambda x: x
                        
                        with patch('open_r1.coupled_grpo.gather_object') as mock_gather_obj:
                            mock_gather_obj.side_effect = lambda x: x
                            
                            result = trainer._generate_and_score_completions(inputs)
        
        # print(result)
        expected_result = {'prompt_ids': torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [372, 880, 475, 329, 733, 564, 739, 376, 632,  10],
        [266, 630, 540, 560, 470, 561, 323, 720,  11, 461],
        [527, 547, 865, 589, 141, 645, 761, 339, 861, 564]]), 'prompt_mask': torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]), 'completion_ids': torch.tensor([[378,  14, 210, 954, 231, 572, 315, 295, 567, 706, 749, 876,  73, 111,
         899, 213, 541, 769, 287, 219],
        [186, 822, 577, 519, 707, 123, 143, 294, 693, 677,  70, 709, 370, 739,
          86, 499, 215, 384, 978, 908],
        [177, 289, 184, 553, 548, 409, 483, 507, 558, 491, 214, 691, 636, 903,
         582, 490, 389, 528, 555,  33],
        [270, 553, 137, 275, 217, 321, 205, 507, 845, 438, 695, 934, 391, 221,
         990, 139, 720, 429, 111, 268]]), 'completion_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
       dtype=torch.int32), 'old_per_token_logps': torch.tensor([[[ 0.0249, -0.3460,  0.2868, -0.7308,  0.1748, -1.0939,  0.9633,
          -0.3095,  1.2888,  0.0523],
         [-1.5469,  0.7567,  0.7755,  2.0265,  0.9812, -0.6401, -0.8057,
          -0.2076, -0.9319, -1.5910]],

        [[-1.1360, -0.5226,  0.7165,  1.5335, -1.9267,  0.1279,  1.0229,
          -0.5558,  0.7043,  0.7099],
         [-1.5326, -0.7251,  0.9624, -0.3370, -1.1753,  0.3581,  0.4788,
           1.3537, -0.1593, -0.4249]],

        [[-0.5208, -0.9320,  0.1852,  1.0687,  1.3065,  0.4598,  0.2618,
          -0.7599, -0.4949, -0.5923],
         [ 0.1543,  0.4408, -0.1483, -2.3184,  1.3032,  0.4879, -1.7809,
           1.5080,  0.3094, -0.5003]],

        [[ 1.0350,  1.6896,  0.0213, -0.8293,  0.1539, -1.0603, -0.5727,
           0.0836,  0.3999,  1.9892],
         [-0.4611, -0.0639, -2.0487, -1.0811,  0.0176,  0.0782,  0.1932,
           0.4097, -1.5754,  2.2508]]]), 'ref_per_token_logps': [], 'advantages': torch.tensor([0., 0., 0., 0.]), 'mask_seeds': [2182, 200]}
        # Verify prompts are processed
        # self.assertEqual(len(processed_prompts), batch_size)

        for key in result.keys():
            torch.testing.assert_close(result[key], expected_result[key], atol=1e-4, rtol=1e-4)

    
    def test_generate_and_score_completions_eos_masking(self):
        """Test masking after EOS token"""
        trainer = self._create_mock_trainer()
        
        # Need to match num_generations = 4
        inputs = [{"prompt": f"Test {i}"} for i in range(4)]
        batch_size = len(inputs)
        vocab_size = self.vocab_size
        completion_length = 10
        
        # Create completion sequence containing EOS token
        completion_ids = torch.tensor([[  542,    67,   876,   414,    26, 50256,   620,   924,   950,   113],
        [  378,    14,   210,   954,   231,   572,   315,   295,   567,   706],
        [  749,   876,    73,   111,   899,   213,   541,   769,   287,   219],
        [  372,   880,   475,   329,   733,   564,   739,   376,   632,    10]])
        completion_ids[0, 5] = self.eos_token_id  # Place EOS at position 5
        # print(completion_ids)
        prompt_ids = torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [378,  14, 210, 954, 231, 572, 315, 295, 567, 706],
        [749, 876,  73, 111, 899, 213, 541, 769, 287, 219],
        [372, 880, 475, 329, 733, 564, 739, 376, 632,  10]])
        full_sequence = torch.cat([prompt_ids, completion_ids], dim=1)
        # print(prompt_ids)
        trainer.model.diffusion_generate.return_value = MagicMock(sequences=full_sequence)
        trainer._get_per_token_logps = lambda model, ids, keep, seeds: torch.randn(batch_size, trainer.num_iterations, completion_length)
        
        with patch('open_r1.coupled_grpo.maybe_apply_chat_template') as mock_chat:
            mock_chat.return_value = {"prompt": "test"}
            
            with patch('open_r1.coupled_grpo.unwrap_model_for_generation') as mock_unwrap:
                mock_unwrap.return_value.__enter__ = lambda self: trainer.model
                mock_unwrap.return_value.__exit__ = lambda self, *args: None
                
                with patch('transformers.Trainer._prepare_inputs') as mock_prepare:
                    mock_prepare.side_effect = lambda self, x: {"input_ids": prompt_ids, "attention_mask": torch.ones_like(prompt_ids)}
                    
                    with patch('open_r1.coupled_grpo.gather') as mock_gather:
                        mock_gather.side_effect = lambda x: x
                        
                        with patch('open_r1.coupled_grpo.gather_object') as mock_gather_obj:
                            mock_gather_obj.side_effect = lambda x: x
                            
                            result = trainer._generate_and_score_completions(inputs)
        
        # Verify completion_mask - check first sample
        completion_mask = result["completion_mask"]
        # print(completion_mask)
        expected_completion_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int32)

        torch.testing.assert_close(completion_mask, expected_completion_mask)
        # EOS position and before should be 1 (including EOS itself)
        self.assertGreaterEqual(completion_mask[0, :6].sum().item(), 5)
        
        # Verify basic properties of mask
        self.assertEqual(completion_mask.shape[0], batch_size)
    
    def test_generate_and_score_completions_mask_seeds_generation(self):
        """Test mask seeds generation"""
        trainer = self._create_mock_trainer()
        
        # Need to match num_generations = 4
        inputs = [{"prompt": f"Test {i}"} for i in range(4)]
        batch_size = len(inputs)
        vocab_size = self.vocab_size
        
        trainer.model.diffusion_generate.return_value = MagicMock(
            sequences=torch.randint(0, vocab_size, (batch_size, 30))
        )
        trainer._get_per_token_logps = lambda model, ids, keep, seeds: torch.randn(batch_size, trainer.num_iterations, 10)
        
        # Test random_masking=True
        trainer.args.random_masking = True
        
        with patch('open_r1.coupled_grpo.maybe_apply_chat_template') as mock_chat:
            mock_chat.return_value = {"prompt": "test"}
            
            with patch('open_r1.coupled_grpo.unwrap_model_for_generation') as mock_unwrap:
                mock_unwrap.return_value.__enter__ = lambda self: trainer.model
                mock_unwrap.return_value.__exit__ = lambda self, *args: None
                
                with patch('transformers.Trainer._prepare_inputs') as mock_prepare:
                    mock_prepare.side_effect = lambda self, x: {
                        "input_ids": torch.randint(0, vocab_size, (batch_size, 10)),
                        "attention_mask": torch.ones(batch_size, 10)
                    }
                    
                    with patch('open_r1.coupled_grpo.gather') as mock_gather:
                        mock_gather.side_effect = lambda x: x
                        
                        with patch('open_r1.coupled_grpo.gather_object') as mock_gather_obj:
                            mock_gather_obj.side_effect = lambda x: x
                            
                            result = trainer._generate_and_score_completions(inputs)
        
        # Verify mask_seeds
        mask_seeds = result["mask_seeds"]
        self.assertEqual(len(mask_seeds), trainer.num_iterations)
        
        # Test random_masking=False
        trainer.args.random_masking = False
        
        with patch('open_r1.coupled_grpo.maybe_apply_chat_template') as mock_chat:
            mock_chat.return_value = {"prompt": "test"}
            
            with patch('open_r1.coupled_grpo.unwrap_model_for_generation') as mock_unwrap:
                mock_unwrap.return_value.__enter__ = lambda self: trainer.model
                mock_unwrap.return_value.__exit__ = lambda self, *args: None
                
                with patch('transformers.Trainer._prepare_inputs') as mock_prepare:
                    mock_prepare.side_effect = lambda self, x: {
                        "input_ids": torch.randint(0, vocab_size, (batch_size, 10)),
                        "attention_mask": torch.ones(batch_size, 10)
                    }
                    
                    with patch('open_r1.coupled_grpo.gather') as mock_gather:
                        mock_gather.side_effect = lambda x: x
                        
                        with patch('open_r1.coupled_grpo.gather_object') as mock_gather_obj:
                            mock_gather_obj.side_effect = lambda x: x
                            
                            result = trainer._generate_and_score_completions(inputs)
        
        # Verify fixed seeds
        mask_seeds = result["mask_seeds"]
        self.assertEqual(mask_seeds, [42] * trainer.num_iterations)
    
    def test_generate_and_score_completions_advantage_calculation(self):
        """Test advantage calculation (leave-one-out baseline)"""
        trainer = self._create_mock_trainer()
        trainer.num_generations = 4
        
        # Create 4 samples (corresponding to 4 generations of the same prompt)
        inputs = [{"prompt": f"Test {i}"} for i in range(4)]
        batch_size = len(inputs)
        
        # Mock reward function returns known reward values
        rewards = [1.0, 2.0, 3.0, 4.0]
        trainer.reward_funcs = [lambda **kwargs: rewards]
        
        trainer.model.diffusion_generate.return_value = MagicMock(
            sequences=torch.randint(0, self.vocab_size, (batch_size, 30))
        )
        trainer._get_per_token_logps = lambda model, ids, keep, seeds: torch.randn(batch_size, trainer.num_iterations, 10)
        
        with patch('open_r1.coupled_grpo.maybe_apply_chat_template') as mock_chat:
            mock_chat.return_value = {"prompt": "test"}
            
            with patch('open_r1.coupled_grpo.unwrap_model_for_generation') as mock_unwrap:
                mock_unwrap.return_value.__enter__ = lambda self: trainer.model
                mock_unwrap.return_value.__exit__ = lambda self, *args: None
                
                with patch('transformers.Trainer._prepare_inputs') as mock_prepare:
                    mock_prepare.return_value = {
                        "input_ids": torch.randint(0, self.vocab_size, (batch_size, 10)),
                        "attention_mask": torch.ones(batch_size, 10)
                    }
                    
                    with patch('open_r1.coupled_grpo.gather') as mock_gather:
                        mock_gather.side_effect = lambda x: x
                        
                        result = trainer._generate_and_score_completions(inputs)
        
        # Verify advantage calculation
        advantages = result["advantages"]
        
        # Manually calculate expected advantages
        # baseline_i = (sum - reward_i) / (k-1)
        # advantage_i = reward_i - baseline_i
        total_reward = sum(rewards)
        expected_advantages = []
        for r in rewards:
            baseline = (total_reward - r) / (trainer.num_generations - 1)
            advantage = r - baseline
            expected_advantages.append(advantage)
        
        # Since scale_rewards=True, also need to divide by standard deviation
        std = torch.tensor(rewards).std().item()
        expected_advantages = [a / (std + 1e-4) for a in expected_advantages]
        
        # Verify (considering numerical error)
        for i, expected in enumerate(expected_advantages):
            self.assertAlmostEqual(advantages[i].item(), expected, places=4)
    
    def test_generate_and_score_completions_reward_weighting(self):
        """Test weighting of multiple reward functions"""
        trainer = self._create_mock_trainer()
        
        # Set up multiple reward functions
        trainer.reward_funcs = [
            lambda **kwargs: [1.0] * len(kwargs['prompts']),
            lambda **kwargs: [2.0] * len(kwargs['prompts'])
        ]
        trainer.reward_processing_classes = [trainer.processing_class, trainer.processing_class]
        trainer.reward_weights = torch.tensor([0.3, 0.7])
        trainer.reward_func_names = ["reward1", "reward2"]
        trainer._metrics["train"]["rewards/reward1"] = []
        trainer._metrics["train"]["rewards/reward2"] = []
        trainer._textual_logs["rewards"]["reward1"] = []
        trainer._textual_logs["rewards"]["reward2"] = []
        
        # Need to match num_generations = 4
        inputs = [{"prompt": f"Test {i}"} for i in range(4)]
        batch_size = len(inputs)
        vocab_size = self.vocab_size
        
        trainer.model.diffusion_generate.return_value = MagicMock(
            sequences=torch.randint(0, vocab_size, (batch_size, 30))
        )
        trainer._get_per_token_logps = lambda model, ids, keep, seeds: torch.randn(batch_size, trainer.num_iterations, 10)
        
        with patch('open_r1.coupled_grpo.maybe_apply_chat_template') as mock_chat:
            mock_chat.return_value = {"prompt": "test"}
            
            with patch('open_r1.coupled_grpo.unwrap_model_for_generation') as mock_unwrap:
                mock_unwrap.return_value.__enter__ = lambda self: trainer.model
                mock_unwrap.return_value.__exit__ = lambda self, *args: None
                
                with patch('transformers.Trainer._prepare_inputs') as mock_prepare:
                    mock_prepare.side_effect = lambda self, x: {
                        "input_ids": torch.randint(0, vocab_size, (batch_size, 10)),
                        "attention_mask": torch.ones(batch_size, 10)
                    }
                    
                    with patch('open_r1.coupled_grpo.gather') as mock_gather:
                        mock_gather.side_effect = lambda x: x
                        
                        with patch('open_r1.coupled_grpo.gather_object') as mock_gather_obj:
                            mock_gather_obj.side_effect = lambda x: x
                            
                            result = trainer._generate_and_score_completions(inputs)
        
        # Verify weighted reward
        # Expected: 0.3 * 1.0 + 0.7 * 2.0 = 0.3 + 1.4 = 1.7
        # But since it's a single sample, advantage will be 0 (because baseline equals reward)
        self.assertIn("advantages", result)
        # print(result["advantages"])
        expected_advantages = torch.tensor([-0.0012, -0.0012, -0.0012, -0.0012])
        torch.testing.assert_close(result["advantages"], expected_advantages)
    
    def test_generate_and_score_completions_metrics_logging(self):
        """Test metrics logging"""
        trainer = self._create_mock_trainer()
        
        # Need to match num_generations = 4
        inputs = [{"prompt": f"Test {i}"} for i in range(4)]
        batch_size = len(inputs)
        vocab_size = self.vocab_size
        
        trainer.model.diffusion_generate.return_value = MagicMock(
            sequences=torch.randint(0, vocab_size, (batch_size, 30))
        )
        trainer._get_per_token_logps = lambda model, ids, keep, seeds: torch.randn(batch_size, trainer.num_iterations, 10)
        
        with patch('open_r1.coupled_grpo.maybe_apply_chat_template') as mock_chat:
            mock_chat.return_value = {"prompt": "test"}
            
            with patch('open_r1.coupled_grpo.unwrap_model_for_generation') as mock_unwrap:
                mock_unwrap.return_value.__enter__ = lambda self: trainer.model
                mock_unwrap.return_value.__exit__ = lambda self, *args: None
                
                with patch('transformers.Trainer._prepare_inputs') as mock_prepare:
                    mock_prepare.side_effect = lambda self, x: {
                        "input_ids": torch.randint(0, vocab_size, (batch_size, 10)),
                        "attention_mask": torch.ones(batch_size, 10)
                    }
                    
                    with patch('open_r1.coupled_grpo.gather') as mock_gather:
                        mock_gather.side_effect = lambda x: x
                        
                        with patch('open_r1.coupled_grpo.gather_object') as mock_gather_obj:
                            mock_gather_obj.side_effect = lambda x: x
                            
                            result = trainer._generate_and_score_completions(inputs)
        
        # Verify metrics are logged
        self.assertGreater(len(trainer._metrics["train"]["completion_length"]), 0)
        self.assertGreater(len(trainer._metrics["train"]["reward"]), 0)


class TestGenerateAndScoreCompletionsIntegration(unittest.TestCase):
    """Integration test: test complete workflow of _generate_and_score_completions"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_end_to_end_generation_and_scoring(self):
        """End-to-end test: complete generation and scoring workflow"""
        # This test verifies the coherence of the entire workflow
        trainer = object.__new__(DiffuGRPOTrainer)
        
        # Configure all necessary attributes
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = 50257
        trainer.processing_class.encode = lambda x: [50256]
        trainer.processing_class.batch_decode = lambda ids, **kwargs: ["def test(): pass"] * ids.size(0)
        def mock_tokenizer_call(text, **kwargs):
            return {
                "input_ids": torch.randint(0, 1000, (len(text), 10)),
                "attention_mask": torch.ones(len(text), 10)
            }
        # Use side_effect instead of directly assigning __call__
        trainer.processing_class.side_effect = mock_tokenizer_call
        
        trainer.accelerator = MagicMock()
        trainer.accelerator.device = torch.device('cpu')
        trainer.accelerator.process_index = 0
        trainer.accelerator.gather_for_metrics = lambda x: x
        
        trainer.model = MagicMock()
        trainer.model_wrapped = MagicMock()
        trainer.model.diffusion_generate = MagicMock(
            return_value=MagicMock(sequences=torch.randint(0, 1000, (2, 30)))
        )
        
        # Create a real object instead of MagicMock to store configuration
        class MockArgs:
            max_completion_length = 50
            diffusion_steps = 10
            generation_temperature = 0.7
            generation_batch_size = 2
            random_masking = False
            output_dir = "test_output"
            report_to = []  # Add report_to attribute
        
        trainer.args = MockArgs()
        
        trainer.max_prompt_length = None
        trainer.num_iterations = 1
        trainer.num_generations = 2
        trainer.beta = 0.0
        trainer.ref_model = None
        trainer.scale_rewards = False
        trainer.reward_funcs = [lambda **kwargs: [0.8, 0.9]]
        trainer.reward_processing_classes = [trainer.processing_class]
        trainer.reward_weights = torch.tensor([1.0])
        trainer.reward_func_names = ["test_reward"]
        trainer.control = MagicMock()
        trainer.control.should_evaluate = False
        trainer._step = 0  # Add missing _step attribute
        
        trainer._metrics = {"train": {"completion_length": [], "zero_std_ratio": [],
                                     "rewards/test_reward": [], "rewards/<lambda>": [],
                                     "reward": [], "reward_std": []}}
        trainer._textual_logs = {"prompt": [], "completion": [], "rewards": {"test_reward": [], "<lambda>": []}}
        
        trainer._get_per_token_logps = lambda model, ids, keep, seeds: torch.randn(2, 1, 10)
        
        inputs = [{"prompt": "Write a function"}, {"prompt": "Implement sorting"}]
        
        with patch('open_r1.coupled_grpo.maybe_apply_chat_template') as mock_chat:
            mock_chat.side_effect = lambda example, tokenizer: {"prompt": example["prompt"]}
            
            with patch('open_r1.coupled_grpo.unwrap_model_for_generation') as mock_unwrap:
                mock_unwrap.return_value.__enter__ = lambda self: trainer.model
                mock_unwrap.return_value.__exit__ = lambda self, *args: None
                
                with patch('transformers.Trainer._prepare_inputs') as mock_prepare:
                    mock_prepare.side_effect = lambda self, x: x
                    
                    with patch('open_r1.coupled_grpo.gather') as mock_gather:
                        mock_gather.side_effect = lambda x: x
                        
                        with patch('open_r1.coupled_grpo.gather_object') as mock_gather_obj:
                            mock_gather_obj.side_effect = lambda x: x
                            
                            result = trainer._generate_and_score_completions(inputs)
        
        # Verify all necessary outputs exist
        required_keys = ["prompt_ids", "prompt_mask", "completion_ids", "completion_mask",
                        "advantages", "mask_seeds"]
        for key in required_keys:
            self.assertIn(key, result, f"Output should contain {key}")
        
        # Verify output shapes and types
        self.assertEqual(result["prompt_ids"].dim(), 2)
        self.assertEqual(result["completion_ids"].dim(), 2)
        self.assertTrue(torch.is_tensor(result["advantages"]))


if __name__ == "__main__":
    unittest.main()
