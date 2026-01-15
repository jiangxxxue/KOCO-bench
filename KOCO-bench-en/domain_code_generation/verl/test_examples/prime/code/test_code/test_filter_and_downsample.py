import unittest
import torch
import sys
import os
from omegaconf import OmegaConf

# Add the recipe directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'recipe'))

# Mock verl modules
from unittest.mock import MagicMock

# Create a proper base class for RayPRIMETrainer to inherit from
class MockRayPPOTrainer:
    """Mock base class for RayPRIMETrainer"""
    def __init__(self, *args, **kwargs):
        pass

# Setup mocks
mock_verl = MagicMock()
sys.modules['verl'] = mock_verl

# Mock DataProto before other imports
from unittest.mock import Mock
class DataProto:
    """Minimal DataProto implementation for testing"""
    def __init__(self, batch_dict):
        self.batch = batch_dict
    
    @staticmethod
    def from_dict(tensors):
        return DataProto(tensors)
    
    def __len__(self):
        first_key = next(iter(self.batch.keys()))
        return len(self.batch[first_key])
    
    def reorder(self, indices):
        """Reorder batch data according to indices"""
        for key in self.batch:
            self.batch[key] = self.batch[key][indices]

mock_verl.DataProto = DataProto

sys.modules['verl.single_controller'] = MagicMock()
sys.modules['verl.single_controller.ray'] = MagicMock()
sys.modules['verl.trainer'] = MagicMock()
sys.modules['verl.trainer.ppo'] = MagicMock()
sys.modules['verl.trainer.ppo.core_algos'] = MagicMock()
sys.modules['verl.trainer.ppo.metric_utils'] = MagicMock()

# Create a mock module for ray_trainer with our MockRayPPOTrainer
mock_ray_trainer = MagicMock()
mock_ray_trainer.RayPPOTrainer = MockRayPPOTrainer
mock_ray_trainer.ResourcePoolManager = MagicMock
sys.modules['verl.trainer.ppo.ray_trainer'] = mock_ray_trainer

sys.modules['verl.trainer.ppo.utils'] = MagicMock()
sys.modules['verl.utils'] = MagicMock()
sys.modules['verl.utils.torch_functional'] = MagicMock()
sys.modules['verl.utils.checkpoint'] = MagicMock()
sys.modules['verl.utils.checkpoint.checkpoint_manager'] = MagicMock()
sys.modules['verl.utils.dataset'] = MagicMock()
sys.modules['verl.utils.dataset.rl_dataset'] = MagicMock()
sys.modules['verl.utils.metric'] = MagicMock()
sys.modules['verl.utils.profiler'] = MagicMock()
sys.modules['verl.utils.profiler.performance'] = MagicMock()

# Import ground truth class (now it can properly inherit from MockRayPPOTrainer)
from prime.prime_ray_trainer import RayPRIMETrainer


class TestFilterAndDownsample(unittest.TestCase):
    """Test filter_and_downsample function - intelligent sample filtering and priority downsampling described in Section 7 of the documentation"""
    
    def setUp(self):
        torch.manual_seed(42)
        
        # Create minimal config for RayPRIMETrainer
        self.config = OmegaConf.create({
            'actor_rollout_ref': {
                'rollout': {
                    'n': 4  # n_samples
                }
            },
            'data': {
                'oversample_factor': 4.0,
                'filter_accuracy': True,
                'accuracy_lower_bound': 0.2,
                'accuracy_upper_bound': 0.8,
                'filter_truncate': True,
                'max_response_length': 20
            }
        })
    
    def _call_filter_and_downsample(self, scores, batch):
        """Helper method to call the ground truth filter_and_downsample function"""
        # Create a simple object to hold config
        class SimpleTrainer:
            def __init__(self, config):
                self.config = config
        
        trainer = SimpleTrainer(self.config)
        # Call the actual method from RayPRIMETrainer
        return RayPRIMETrainer.filter_and_downsample(trainer, scores, batch)
    
    def test_filter_and_downsample_basic(self):
        """Test basic filtering and downsampling functionality - using ground truth code"""
        n_samples = 4
        num_groups = 4
        total_samples = num_groups * n_samples
        
        # Create test scores: sample groups of different quality
        scores = [
            # Group 1: average 0.5, should pass filtering
            0.6, 0.4, 0.5, 0.5,
            # Group 2: average 0.1, too low, should be filtered
            0.1, 0.1, 0.1, 0.1,
            # Group 3: average 0.9, too high, should be filtered
            0.9, 0.9, 0.9, 0.9,
            # Group 4: average 0.6, should pass filtering
            0.7, 0.5, 0.6, 0.6
        ]
        
        # Create test batch
        batch = DataProto.from_dict(tensors={
            "responses": torch.randint(0, 1000, (total_samples, 10)),
            "attention_mask": torch.ones(total_samples, 15),
            "input_ids": torch.randint(0, 1000, (total_samples, 15))
        })

        
        # Call ground truth function
        result_batch = self._call_filter_and_downsample(scores, batch)
        
        # Verify number of samples after downsampling
        expected_kept_samples = int(total_samples // self.config.data.oversample_factor)
        self.assertEqual(len(result_batch.batch["responses"]), expected_kept_samples)
        
        # Verify returned object is DataProto
        self.assertIsInstance(result_batch, DataProto)
    
    def test_accuracy_filtering(self):
        """Test accuracy filtering - using ground truth code"""
        n_samples = 4
        num_groups = 5
        total_samples = num_groups * n_samples
        
        # Create sample groups with different accuracies
        scores = [
            0.3, 0.4, 0.5, 0.4,  # Group 1: average 0.4, pass
            0.1, 0.1, 0.1, 0.1,  # Group 2: average 0.1, fail (too low)
            0.9, 0.9, 0.9, 0.9,  # Group 3: average 0.9, fail (too high)
            0.6, 0.7, 0.7, 0.6,  # Group 4: average 0.65, pass
            0.2, 0.2, 0.2, 0.2   # Group 5: average 0.2, pass (boundary)
        ]
        
        batch = DataProto.from_dict(tensors={
            "responses": torch.randint(0, 1000, (total_samples, 8)),
            "attention_mask": torch.ones(total_samples, 12),
            "input_ids": torch.randint(0, 1000, (total_samples, 12))
        })
        
        result_batch = self._call_filter_and_downsample(scores, batch)
        
        # Verify downsampling count
        expected_kept_samples = int(total_samples // self.config.data.oversample_factor)
        self.assertEqual(len(result_batch.batch["responses"]), expected_kept_samples)
    
    def test_length_filtering(self):
        """Test length filtering - using ground truth code"""
        n_samples = 4
        num_groups = 3
        total_samples = num_groups * n_samples
        max_response_length = 10
        
        # Update config
        self.config.data.max_response_length = max_response_length
        
        # Create scores with reasonable accuracy
        scores = [0.5, 0.6, 0.5, 0.6] * num_groups
        
        # Create attention_masks with different lengths
        attention_masks = torch.tensor([
            # Group 1: all should pass
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            # Group 2: all should be filtered (length>=10)
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            # Group 3: all should pass
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        ])
        
        batch = DataProto.from_dict(tensors={
            "responses": torch.randint(0, 1000, (total_samples, 12)),
            "attention_mask": attention_masks,
            "input_ids": torch.randint(0, 1000, (total_samples, 12))
        })
        
        result_batch = self._call_filter_and_downsample(scores, batch)
        
        # Verify downsampling count
        expected_kept_samples = int(total_samples // self.config.data.oversample_factor)
        self.assertEqual(len(result_batch.batch["responses"]), expected_kept_samples)
    
    def test_no_groups_pass_filter(self):
        """Test fallback mechanism when no groups pass filtering - using ground truth code"""
        n_samples = 4
        num_groups = 3
        total_samples = num_groups * n_samples
        
        # All groups fail to meet accuracy conditions
        scores = [0.1, 0.05, 0.15, 0.1] * num_groups  # All scores too low
        
        batch = DataProto.from_dict(tensors={
            "responses": torch.randint(0, 1000, (total_samples, 8)),
            "attention_mask": torch.ones(total_samples, 12),
            "input_ids": torch.randint(0, 1000, (total_samples, 12))
        })
        
        result_batch = self._call_filter_and_downsample(scores, batch)
        
        # Even if no groups pass filtering, should still keep proportional samples
        expected_kept_samples = int(total_samples // self.config.data.oversample_factor)
        self.assertEqual(len(result_batch.batch["responses"]), expected_kept_samples)
    
    def test_priority_downsampling(self):
        """Test priority downsampling - using ground truth code"""
        n_samples = 4
        num_groups = 4
        total_samples = num_groups * n_samples
        
        # Create sample groups with partial filtering pass
        scores = [
            0.3, 0.4, 0.35, 0.4,  # Group 1: pass
            0.1, 0.15, 0.1, 0.1,  # Group 2: fail (too low)
            0.5, 0.6, 0.55, 0.5,  # Group 3: pass
            0.9, 0.85, 0.9, 0.9   # Group 4: fail (too high)
        ]
        
        batch = DataProto.from_dict(tensors={
            "responses": torch.randint(0, 1000, (total_samples, 8)),
            "attention_mask": torch.ones(total_samples, 12),
            "input_ids": torch.randint(0, 1000, (total_samples, 12))
        })
        
        result_batch = self._call_filter_and_downsample(scores, batch)
        
        # Verify downsampling count
        expected_kept_samples = int(total_samples // self.config.data.oversample_factor)
        self.assertEqual(len(result_batch.batch["responses"]), expected_kept_samples)
    
    def test_combined_filtering(self):
        """Test combined accuracy and length filtering - using ground truth code"""
        n_samples = 4
        num_groups = 3
        total_samples = num_groups * n_samples
        
        # Create test data
        scores = [
            0.5, 0.6, 0.5, 0.6,  # Group 1: accuracy pass, length pass
            0.4, 0.5, 0.4, 0.5,  # Group 2: accuracy pass, length fail
            0.6, 0.7, 0.6, 0.7   # Group 3: accuracy pass, length pass
        ]
        
        # Create attention_masks with different lengths
        attention_masks = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # Group 1: length 8
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # Group 1: length 9
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # Group 1: length 7
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # Group 1: length 8
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Group 2: length 10 (exceeds max-1)
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Group 2: length 10
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Group 2: length 10
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Group 2: length 10
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # Group 3: length 8
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # Group 3: length 9
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # Group 3: length 7
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]   # Group 3: length 8
        ])
        
        batch = DataProto.from_dict(tensors={
            "responses": torch.randint(0, 1000, (total_samples, 10)),
            "attention_mask": attention_masks,
            "input_ids": torch.randint(0, 1000, (total_samples, 10))
        })
        
        # Set max_response_length
        self.config.data.max_response_length = 10
        
        result_batch = self._call_filter_and_downsample(scores, batch)
        
        # Verify downsampling count
        expected_kept_samples = int(total_samples // self.config.data.oversample_factor)
        self.assertEqual(len(result_batch.batch["responses"]), expected_kept_samples)
    
    def test_different_oversample_factors(self):
        """Test different oversample factors - using ground truth code"""
        n_samples = 4
        num_groups = 6
        total_samples = num_groups * n_samples
        
        # Test different oversample factors
        for oversample_factor in [2.0, 3.0, 6.0]:
            with self.subTest(oversample_factor=oversample_factor):
                # Create new batch for each test (because reorder is in-place operation)
                scores = [0.5, 0.6, 0.4, 0.5] * num_groups  # All groups pass filtering
                
                batch = DataProto.from_dict(tensors={
                    "responses": torch.randint(0, 1000, (total_samples, 8)),
                    "attention_mask": torch.ones(total_samples, 12),
                    "input_ids": torch.randint(0, 1000, (total_samples, 12))
                })
                
                self.config.data.oversample_factor = oversample_factor
                result_batch = self._call_filter_and_downsample(scores, batch)
                
                expected_kept_samples = int(total_samples // oversample_factor)
                self.assertEqual(len(result_batch.batch["responses"]), expected_kept_samples)
    
    def test_filter_disabled(self):
        """Test case with filters disabled - using ground truth code"""
        n_samples = 4
        num_groups = 4
        total_samples = num_groups * n_samples
        
        # Disable all filters
        self.config.data.filter_accuracy = False
        self.config.data.filter_truncate = False
        
        # Use extreme scores (would be filtered if filters enabled)
        scores = [0.05, 0.95, 0.05, 0.95] * num_groups
        
        batch = DataProto.from_dict(tensors={
            "responses": torch.randint(0, 1000, (total_samples, 8)),
            "attention_mask": torch.ones(total_samples, 12),
            "input_ids": torch.randint(0, 1000, (total_samples, 12))
        })
        
        result_batch = self._call_filter_and_downsample(scores, batch)
        
        # Verify downsampling count
        expected_kept_samples = int(total_samples // self.config.data.oversample_factor)
        self.assertEqual(len(result_batch.batch["responses"]), expected_kept_samples)


if __name__ == "__main__":
    unittest.main()
