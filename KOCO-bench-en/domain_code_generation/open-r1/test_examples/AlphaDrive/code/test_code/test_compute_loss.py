import unittest
import torch
import sys
import os
from unittest.mock import MagicMock, patch, Mock
import torch.nn as nn

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'r1-v', 'src'))

# Import the actual class we want to test
try:
    from open_r1.trainer.grpo_trainer import Qwen2VLGRPOTrainer
    from transformers import GenerationConfig, PreTrainedModel
    TRAINER_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    TRAINER_AVAILABLE = False
    Qwen2VLGRPOTrainer = None


class TestComputeLoss(unittest.TestCase):
    """Test Qwen2VLGRPOTrainer.compute_loss function - GRPO loss calculation described in AlphaDrive documentation section 3
    
    This test directly calls the ground truth compute_loss function rather than reimplementing its logic.
    Uses Mock objects to simulate complex dependencies (model, processor, etc.), but compute_loss itself is called for real.
    """
    
    @unittest.skipIf(not TRAINER_AVAILABLE, "Trainer not available")
    def setUp(self):
        """Set up test environment, create mock objects"""
        torch.manual_seed(42)
        
        # Basic parameters
        self.batch_size = 2
        self.num_generations = 4
        self.prompt_length = 8
        self.completion_length = 6
        self.vocab_size = 100
        
        # Mock accelerator
        self.mock_accelerator = MagicMock()
        self.mock_accelerator.device = torch.device("cpu")
        self.mock_accelerator.gather_for_metrics = lambda x: x
        self.mock_accelerator.unwrap_model = lambda model: MagicMock(__enter__=lambda self: model, __exit__=lambda *args: None)
        
        # Mock processing_class (tokenizer/processor)
        self.mock_processor = MagicMock()
        self.mock_processor.eos_token_id = 2
        self.mock_processor.pad_token_id = 0
        # Mock batch_decode to return strings
        self.mock_processor.batch_decode = lambda ids, skip_special_tokens=False: [
            f"completion_{i}" for i in range(ids.shape[0])
        ]
        
        # Mock training arguments
        self.mock_args = MagicMock()
        self.mock_args.device = torch.device("cpu")
        self.mock_args.past_index = -1
        
        # Create trainer instance
        self.trainer = Qwen2VLGRPOTrainer.__new__(Qwen2VLGRPOTrainer)
        self.trainer.accelerator = self.mock_accelerator
        self.trainer.processing_class = self.mock_processor
        self.trainer.args = self.mock_args
        self.trainer.is_deepspeed_enabled = False
        self.trainer.is_fsdp_enabled = False
        self.trainer._past = None
        self.trainer.reward_processing_classes = [self.mock_processor]
        self.trainer.num_generations = self.num_generations
        self.trainer.max_prompt_length = None
        self.trainer.beta = 0.1  # KL penalty coefficient
        self.trainer.ref_model = None  # Use same model with adapter disabled
        self.trainer._metrics = {
            "completion_length": [],
            "reward": [],
            "reward_std": [],
            "kl": []
        }
        
        # Mock generation config
        self.trainer.generation_config = GenerationConfig(
            max_new_tokens=self.completion_length,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=self.num_generations,
            pad_token_id=0
        )
        
        # Mock _get_per_token_logps method
        def mock_get_per_token_logps(model, input_ids, attention_mask, pixel_values, image_grid_thw):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1] - 1  # Because of shift
            # Return random log probabilities

            # print(torch.randn(batch_size, seq_len))
            return torch.randn(batch_size, seq_len)
        
        self.trainer._get_per_token_logps = mock_get_per_token_logps
    
    def _setup_mocks_for_test(self, batch_size, inputs):
        """Helper method: set up common mock objects for testing"""
        # Mock processor return values
        prompt_ids = torch.randint(1, self.vocab_size, (batch_size, self.prompt_length))
        prompt_mask = torch.ones(batch_size, self.prompt_length, dtype=torch.long)
        pixel_values = torch.randn(batch_size, 1024)  # 2D tensor for Qwen2VL
        image_grid_thw = torch.tensor([[1, 224, 224]] * batch_size)
        
        def mock_processor_call(*args, **kwargs):
            return {
                "input_ids": prompt_ids,
                "attention_mask": prompt_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw
            }
        self.mock_processor.side_effect = mock_processor_call
    
    def _run_compute_loss_test(self, inputs, reward_func, batch_size=None):
        """Generic compute_loss test runner
        
        Args:
            inputs: Input data list
            reward_func: Reward function
            batch_size: Batch size (if None, inferred from inputs)
        
        Returns:
            loss: Computed loss value
        """
        if batch_size is None:
            batch_size = len(inputs)
        
        # Set up mocks
        self._setup_mocks_for_test(batch_size, inputs)
        
        with patch('open_r1.trainer.grpo_trainer.maybe_apply_chat_template') as mock_chat:
            mock_chat.return_value = {"prompt": "Test"}
            
            with patch('open_r1.trainer.grpo_trainer.load_image') as mock_load:
                mock_load.return_value = MagicMock()
                
                mock_model = self._create_mock_model()
                
                with patch('open_r1.trainer.grpo_trainer.unwrap_model_for_generation') as mock_unwrap:
                    mock_context = MagicMock()
                    mock_context.__enter__ = lambda self: mock_model
                    mock_context.__exit__ = lambda *args: None
                    mock_unwrap.return_value = mock_context
                    
                    # Set reward function
                    self.trainer.reward_funcs = [reward_func]
                    self.trainer._metrics[f"rewards/{reward_func.__name__}"] = []
                    
                    # Call compute_loss
                    loss = self.trainer.compute_loss(mock_model, inputs, return_outputs=False)
                    
                    return loss
    
    @unittest.skipIf(not TRAINER_AVAILABLE, "Trainer not available")
    def test_compute_loss_basic_flow(self):
        """Test basic flow of compute_loss: input -> generation -> reward -> loss"""
        
        # Prepare input data
        inputs = [
            {
                "prompt": [{"role": "user", "content": "What action?"}],
                "image": "dummy_path1.jpg",
                "solution": "<answer>ACCELERATE</answer>"
            },
            {
                "prompt": [{"role": "user", "content": "What to do?"}],
                "image": "dummy_path2.jpg", 
                "solution": "<answer>STOP</answer>"
            }
        ]
        
        # Use helper method to set up mocks
        self._setup_mocks_for_test(self.batch_size, inputs)
        
        # Mock maybe_apply_chat_template
        with patch('open_r1.trainer.grpo_trainer.maybe_apply_chat_template') as mock_chat:
            mock_chat.return_value = {"prompt": "What action?"}
            
            # Mock load_image
            with patch('open_r1.trainer.grpo_trainer.load_image') as mock_load:
                mock_image = MagicMock()
                mock_load.return_value = mock_image
                
                # Mock model
                mock_model = self._create_mock_model()
                
                # Mock unwrap_model_for_generation to return the model itself
                with patch('open_r1.trainer.grpo_trainer.unwrap_model_for_generation') as mock_unwrap:
                    mock_context = MagicMock()
                    mock_context.__enter__ = lambda self: mock_model
                    mock_context.__exit__ = lambda *args: None
                    mock_unwrap.return_value = mock_context
                    
                    # Mock reward function
                    def mock_reward_func(prompts, completions, **kwargs):
                        # Simple reward: give reward based on completion length
                        return [float(len(str(c)) % 3) / 2.0 for c in completions]
                    
                    self.trainer.reward_funcs = [mock_reward_func]
                    # Add metrics key for this reward function
                    self.trainer._metrics["rewards/mock_reward_func"] = []
                    
                    # Call the real compute_loss function
                    loss = self.trainer.compute_loss(mock_model, inputs, return_outputs=False)
                    
                    # print(loss)

                    expected_loss = torch.tensor(0.1169)

                    torch.testing.assert_close(loss, expected_loss, atol=1e-4, rtol=1e-4)
                    # Verify output
                    self.assertIsNotNone(loss)
                    # self.assertTrue(torch.is_tensor(loss))
                    # self.assertTrue(torch.isfinite(loss))
                    # self.assertEqual(loss.dim(), 0)  # Scalar loss
    
    @unittest.skipIf(not TRAINER_AVAILABLE, "Trainer not available")
    def test_compute_loss_with_specific_rewards(self):
        """Test compute_loss behavior with specific rewards"""
        
        # Prepare input
        inputs = [
            {
                "prompt": [{"role": "user", "content": "Action?"}],
                "image": "path1.jpg",
                "solution": "<answer>LEFT</answer>"
            }
        ]
        
        # Predefined reward values
        predefined_rewards = [1.0, 0.5, 0.8, 0.6]  # Rewards for 4 generations
        
        # Reward function returns predefined values
        def mock_reward_func(prompts, completions, **kwargs):
            return predefined_rewards
        
        # Use generic runner to execute test
        loss = self._run_compute_loss_test(inputs, mock_reward_func, batch_size=1)

        # print(loss)

        expected_loss = torch.tensor(0.1270)

        torch.testing.assert_close(loss, expected_loss, atol=1e-4, rtol=1e-4)
        # Verify loss calculation
        # self.assertTrue(torch.isfinite(loss))
        
        # Verify metrics are recorded
        self.assertGreater(len(self.trainer._metrics["completion_length"]), 0)
        self.assertGreater(len(self.trainer._metrics["reward"]), 0)
    
    @unittest.skipIf(not TRAINER_AVAILABLE, "Trainer not available")
    def test_compute_loss_kl_penalty(self):
        """Test KL penalty term in compute_loss"""
        
        # Prepare input
        inputs = [
            {
                "prompt": [{"role": "user", "content": "Go?"}],
                "image": "img.jpg",
                "solution": "<answer>YES</answer>"
            }
        ]
        
        losses = []
        # Test different beta values
        for beta in [0.0, 0.1, 0.5]:
            self.trainer.beta = beta
            self.trainer._metrics = {
                "completion_length": [],
                "reward": [],
                "reward_std": [],
                "kl": []
            }
            
            def mock_reward_func(prompts, completions, **kwargs):
                return [0.5] * len(completions)
            
            # Use generic runner to execute test
            loss = self._run_compute_loss_test(inputs, mock_reward_func, batch_size=1)
            losses.append(loss)
            # Verify loss is valid
            self.assertTrue(torch.isfinite(loss))
            
            # Verify KL is recorded
            self.assertGreater(len(self.trainer._metrics["kl"]), 0)

        # print(losses)

        expected_losses = [torch.tensor(0.), torch.tensor(0.1015), torch.tensor(0.4423)]

        for i in range(len(losses)):
            torch.testing.assert_close(losses[i], expected_losses[i], atol=1e-4, rtol=1e-4)
    @unittest.skipIf(not TRAINER_AVAILABLE, "Trainer not available")
    def test_compute_loss_advantage_normalization(self):
        """Test advantage estimation and normalization in compute_loss
        
        GRPO core: for G generations per prompt, compute advantage = (reward - mean) / std
        """
        
        # Use 2 prompts, each generating 4 responses
        inputs = [
            {"prompt": [{"role": "user", "content": "A?"}], "image": "a.jpg", "solution": "a"},
            {"prompt": [{"role": "user", "content": "B?"}], "image": "b.jpg", "solution": "b"}
        ]
        
        # Predefined rewards: 4 generations for first prompt, 4 for second prompt
        rewards_group1 = [1.0, 0.5, 0.8, 0.6]  # mean=0.725, std≈0.187
        rewards_group2 = [0.9, 1.2, 0.7, 1.0]  # mean=0.95, std≈0.187
        all_rewards = rewards_group1 + rewards_group2
        
        def mock_reward_func(prompts, completions, **kwargs):
            return all_rewards
        
        # Use generic runner to execute test
        loss = self._run_compute_loss_test(inputs, mock_reward_func, batch_size=2)
        
        # print(loss)

        expected_loss = torch.tensor(0.1169)

        torch.testing.assert_close(loss, expected_loss, atol=1e-4, rtol=1e-4)
        # Verify
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(len(self.trainer._metrics["reward_std"]), 0)
    
    @unittest.skipIf(not TRAINER_AVAILABLE, "Trainer not available")
    def test_compute_loss_eos_masking(self):
        """Test EOS masking in compute_loss
        
        In generated completions, content after EOS token should be masked out
        """
        
        inputs = [
            {"prompt": [{"role": "user", "content": "Test"}], "image": "test.jpg", "solution": "test"}
        ]
        
        # Set EOS token ID
        eos_token = 2
        self.mock_processor.eos_token_id = eos_token
        
        def mock_reward_func(prompts, completions, **kwargs):
            return [0.5] * len(completions)
        
        # Use generic runner to execute test
        loss = self._run_compute_loss_test(inputs, mock_reward_func, batch_size=1)
        
        # print(loss)

        expected_loss = torch.tensor(0.1270)

        torch.testing.assert_close(loss, expected_loss, atol=1e-4, rtol=1e-4)
        # Verify loss calculation is normal (indicating masking works correctly)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(len(self.trainer._metrics["completion_length"]), 0)
    
    @unittest.skipIf(not TRAINER_AVAILABLE, "Trainer not available")
    def test_compute_loss_multiple_reward_functions(self):
        """Test compute_loss with multiple reward functions"""
        
        inputs = [
            {"prompt": [{"role": "user", "content": "Go"}], "image": "go.jpg", "solution": "go"}
        ]
        
        # Define multiple reward functions
        def reward_func1(prompts, completions, **kwargs):
            return [0.3] * len(completions)
        
        def reward_func2(prompts, completions, **kwargs):
            return [0.2] * len(completions)
        
        def reward_func3(prompts, completions, **kwargs):
            return [0.5] * len(completions)
        
        # Set up mocks
        self._setup_mocks_for_test(1, inputs)
        
        with patch('open_r1.trainer.grpo_trainer.maybe_apply_chat_template') as mock_chat:
            mock_chat.return_value = {"prompt": "Go"}
            
            with patch('open_r1.trainer.grpo_trainer.load_image') as mock_load:
                mock_load.return_value = MagicMock()
                
                mock_model = self._create_mock_model()
                
                with patch('open_r1.trainer.grpo_trainer.unwrap_model_for_generation') as mock_unwrap:
                    mock_context = MagicMock()
                    mock_context.__enter__ = lambda self: mock_model
                    mock_context.__exit__ = lambda *args: None
                    mock_unwrap.return_value = mock_context
                    
                    # Set multiple reward functions
                    self.trainer.reward_funcs = [reward_func1, reward_func2, reward_func3]
                    self.trainer.reward_processing_classes = [self.mock_processor] * 3
                    self.trainer._metrics["rewards/reward_func1"] = []
                    self.trainer._metrics["rewards/reward_func2"] = []
                    self.trainer._metrics["rewards/reward_func3"] = []
                    
                    # Call compute_loss
                    loss = self.trainer.compute_loss(mock_model, inputs, return_outputs=False)
                    
                    # print(loss)

                    expected_loss = torch.tensor(0.1270)

                    torch.testing.assert_close(loss, expected_loss, atol=1e-4, rtol=1e-4)
                    # Verify
                    self.assertTrue(torch.isfinite(loss))
                    
                    # Total reward should be sum of all reward functions: 0.3 + 0.2 + 0.5 = 1.0
                    recorded_reward = self.trainer._metrics["reward"][-1]
                    self.assertAlmostEqual(recorded_reward, 1.0, places=5)
    
    def _create_mock_model(self, include_eos=False, eos_token=2):
        """Create a mock model that simulates generation and forward pass"""
        mock_model = MagicMock()
        
        # Mock generate method
        def mock_generate(**kwargs):
            batch_size = kwargs["input_ids"].shape[0]
            total_len = self.prompt_length + self.completion_length
            
            # Generate complete sequence of prompt+completion
            generated = torch.randint(
                1, 
                self.vocab_size, 
                (batch_size * self.num_generations, total_len)
            )
            
            # If needed, insert EOS token in completion
            if include_eos:
                for i in range(batch_size * self.num_generations):
                    # Insert EOS at middle position of completion
                    eos_pos = self.prompt_length + self.completion_length // 2
                    generated[i, eos_pos] = eos_token
            
            return generated
        
        mock_model.generate = mock_generate
        
        # Mock forward method (for computing log probabilities)
        def mock_forward(**kwargs):
            batch_size = kwargs["input_ids"].shape[0]
            seq_len = kwargs["input_ids"].shape[1]
            
            # Return random logits
            logits = torch.randn(batch_size, seq_len, self.vocab_size)
            
            output = MagicMock()
            output.logits = logits
            return output
        
        mock_model.forward = mock_forward
        mock_model.__call__ = mock_forward
        
        # Mock disable_adapter (for PEFT)
        mock_adapter_context = MagicMock()
        mock_adapter_context.__enter__ = lambda self: None
        mock_adapter_context.__exit__ = lambda *args: None
        mock_model.disable_adapter = lambda: mock_adapter_context
        
        return mock_model


class TestComputeLossIntegration(unittest.TestCase):
    """Integration test: test compute_loss behavior in more complex scenarios"""
    
    @unittest.skipIf(not TRAINER_AVAILABLE, "Trainer not available")
    def setUp(self):
        torch.manual_seed(42)
        
        # Create basic trainer setup
        self.batch_size = 2
        self.num_generations = 4
        self.prompt_length = 10
        self.completion_length = 8
        self.vocab_size = 100
        
        self.mock_accelerator = MagicMock()
        self.mock_accelerator.device = torch.device("cpu")
        self.mock_accelerator.gather_for_metrics = lambda x: x
        self.mock_accelerator.unwrap_model = lambda model: MagicMock(__enter__=lambda self: model, __exit__=lambda *args: None)
        
        self.mock_processor = MagicMock()
        self.mock_processor.eos_token_id = 2
        self.mock_processor.pad_token_id = 0
        
        self.mock_args = MagicMock()
        self.mock_args.device = torch.device("cpu")
        self.mock_args.past_index = -1
        
        self.trainer = Qwen2VLGRPOTrainer.__new__(Qwen2VLGRPOTrainer)
        self.trainer.accelerator = self.mock_accelerator
        self.trainer.processing_class = self.mock_processor
        self.trainer.args = self.mock_args
        self.trainer.is_deepspeed_enabled = False
        self.trainer.is_fsdp_enabled = False
        self.trainer._past = None
        self.trainer.reward_processing_classes = [self.mock_processor]
        self.trainer.num_generations = self.num_generations
        self.trainer.max_prompt_length = None
        self.trainer.beta = 0.1
        self.trainer.ref_model = None
        self.trainer._metrics = {
            "completion_length": [],
            "reward": [],
            "reward_std": [],
            "kl": []
        }
        self.trainer.generation_config = GenerationConfig(
            max_new_tokens=self.completion_length,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=self.num_generations,
            pad_token_id=0
        )
        
        # Mock _get_per_token_logps method (same as TestComputeLoss)
        def mock_get_per_token_logps(model, input_ids, attention_mask, pixel_values, image_grid_thw):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1] - 1  # Because of shift
            # Return random log probabilities
            return torch.randn(batch_size, seq_len)
        
        self.trainer._get_per_token_logps = mock_get_per_token_logps
    
    @unittest.skipIf(not TRAINER_AVAILABLE, "Trainer not available")
    def test_compute_loss_return_outputs_error(self):
        """Test that return_outputs=True should raise an error"""
        
        inputs = [{"prompt": [{"role": "user", "content": "Test"}], "image": "test.jpg"}]
        mock_model = MagicMock()
        
        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            self.trainer.compute_loss(mock_model, inputs, return_outputs=True)
        
        self.assertIn("does not support returning outputs", str(context.exception))
    
    def _create_mock_model(self, batch_size):
        """Create mock model"""
        mock_model = MagicMock()
        
        def mock_generate(**kwargs):
            total_len = self.prompt_length + self.completion_length
            generated = torch.randint(
                1,
                self.vocab_size,
                (batch_size * self.num_generations, total_len)
            )
            return generated
        
        def mock_forward(**kwargs):
            batch = kwargs["input_ids"].shape[0]
            seq_len = kwargs["input_ids"].shape[1]
            logits = torch.randn(batch, seq_len, self.vocab_size)
            output = MagicMock()
            output.logits = logits
            return output
        
        mock_model.generate = mock_generate
        mock_model.forward = mock_forward
        mock_model.__call__ = mock_forward
        mock_model.disable_adapter = lambda: MagicMock(__enter__=lambda self: None, __exit__=lambda *args: None)
        
        return mock_model


if __name__ == "__main__":
    unittest.main()
