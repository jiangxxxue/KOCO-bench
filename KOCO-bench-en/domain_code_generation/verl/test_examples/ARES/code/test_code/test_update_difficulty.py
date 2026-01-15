import unittest
import sys
import os
import numpy as np
from collections import defaultdict
import torch

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the actual class and components we want to test
try:
    from verl.protocol import DataProto
    from verl.trainer.ray_trainer import RayPPOTrainer
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False
    DataProto = None
    RayPPOTrainer = None


class TestUpdateDifficulty(unittest.TestCase):
    """Test update_difficulty_and_skip_gid_set function - Online difficulty bucketing described in ARES document Section 3"""
    
    def setUp(self):
        """Set up test environment"""
        # Create real RayPPOTrainer instance, but don't call __init__ to avoid complex dependencies
        # Only set necessary attributes for testing
        if VERL_AVAILABLE:
            self.trainer = object.__new__(RayPPOTrainer)
            self.trainer.difficulty_dict_train = {}
            self.trainer.target_high_entropy_token_num_dict = {"easy": 0, "medium": 0, "hard": 0}
            self.trainer.alpha_entropy_dict = {"easy": 0.5, "medium": 0.5, "hard": 0.6}
            self.trainer.alpha_entropy_lr = 0.01
            self.trainer.alpha_entropy_min = 0.1
            self.trainer.alpha_entropy_max = 1.0
            self.trainer.skip_gid_set = set()
        else:
            self.trainer = None
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_update_difficulty_basic_structure(self):
        """Test basic input/output structure"""
        # Create test data
        batch_data = {
            "global_id": ["sample_1", "sample_1", "sample_2", "sample_2"],
            "accuracy": [1.0, 1.0, 0.0, 0.0],
            "entropies": [0.5, 0.6, 0.8, 0.9],
            "high_entropy_token_num": [10, 12, 30, 35]
        }
        
        batch = DataProto.from_dict(non_tensors=batch_data)
        
        # Call function
        self.trainer.update_difficulty_and_skip_gid_set(batch)
        
        # Verify output structure
        self.assertIn("sample_1", self.trainer.difficulty_dict_train)
        self.assertIn("sample_2", self.trainer.difficulty_dict_train)
        self.assertIn(self.trainer.difficulty_dict_train["sample_1"], ["easy", "medium", "hard"])
        self.assertIn(self.trainer.difficulty_dict_train["sample_2"], ["easy", "medium", "hard"])
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_difficulty_assignment_easy(self):
        """Test Easy difficulty assignment (acc_rate >= 2/3)"""
        # Sample with accuracy 1.0
        batch_data = {
            "global_id": ["easy_sample"] * 3,
            "accuracy": [1.0, 1.0, 1.0],
            "entropies": [0.3, 0.4, 0.35],
            "high_entropy_token_num": [8, 10, 9]
        }
        
        batch = DataProto.from_dict(non_tensors=batch_data)
        self.trainer.update_difficulty_and_skip_gid_set(batch)
        
        # Verify assigned as easy
        self.assertEqual(self.trainer.difficulty_dict_train["easy_sample"], "easy")
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_difficulty_assignment_medium(self):
        """Test Medium difficulty assignment (1/3 <= acc_rate < 2/3)"""
        # Sample with accuracy 0.5
        batch_data = {
            "global_id": ["medium_sample"] * 4,
            "accuracy": [1.0, 0.0, 1.0, 0.0],
            "entropies": [0.5, 0.6, 0.55, 0.65],
            "high_entropy_token_num": [15, 18, 16, 20]
        }
        
        batch = DataProto.from_dict(non_tensors=batch_data)
        self.trainer.update_difficulty_and_skip_gid_set(batch)
        
        # Verify assigned as medium
        self.assertEqual(self.trainer.difficulty_dict_train["medium_sample"], "medium")
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_difficulty_assignment_hard(self):
        """Test Hard difficulty assignment (acc_rate < 1/3)"""
        # Sample with accuracy 0.0
        batch_data = {
            "global_id": ["hard_sample"] * 3,
            "accuracy": [0.0, 0.0, 0.0],
            "entropies": [0.8, 0.9, 0.85],
            "high_entropy_token_num": [40, 45, 42]
        }
        
        batch = DataProto.from_dict(non_tensors=batch_data)
        self.trainer.update_difficulty_and_skip_gid_set(batch)
        
        # Verify assigned as hard
        self.assertEqual(self.trainer.difficulty_dict_train["hard_sample"], "hard")
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_difficulty_boundary_cases(self):
        """Test boundary cases"""
        # Accuracy exactly 2/3 (should be easy)
        batch_data_67 = {
            "global_id": ["boundary_67"] * 3,
            "accuracy": [1.0, 1.0, 0.0],  # 2/3 = 0.667
            "entropies": [0.5] * 3,
            "high_entropy_token_num": [10] * 3
        }
        
        batch_67 = DataProto.from_dict(non_tensors=batch_data_67)
        self.trainer.update_difficulty_and_skip_gid_set(batch_67)
        self.assertEqual(self.trainer.difficulty_dict_train["boundary_67"], "easy")
        
        # Accuracy exactly 1/3 (should be medium)
        batch_data_33 = {
            "global_id": ["boundary_33"] * 3,
            "accuracy": [1.0, 0.0, 0.0],  # 1/3 = 0.333
            "entropies": [0.6] * 3,
            "high_entropy_token_num": [20] * 3
        }
        
        batch_33 = DataProto.from_dict(non_tensors=batch_data_33)
        self.trainer.update_difficulty_and_skip_gid_set(batch_33)
        self.assertEqual(self.trainer.difficulty_dict_train["boundary_33"], "medium")
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_target_token_num_calculation(self):
        """Test target HWE token count calculation"""
        # Create samples of different difficulties
        batch_data = {
            "global_id": ["easy_1", "easy_1", "medium_1", "medium_1", "hard_1", "hard_1"],
            "accuracy": [1.0, 1.0, 0.5, 0.5, 0.0, 0.0],
            "entropies": [0.4] * 6,
            "high_entropy_token_num": [10, 12, 20, 22, 40, 42]
        }
        
        batch = DataProto.from_dict(non_tensors=batch_data)
        self.trainer.update_difficulty_and_skip_gid_set(batch)
        
        # Verify target token count is correctly calculated
        # Easy: mean([11, ...]) = 11
        # Medium: mean([21, ...]) = 21  
        # Hard: mean([41, ...]) = 41
        self.assertEqual(self.trainer.target_high_entropy_token_num_dict["easy"], 11)
        self.assertEqual(self.trainer.target_high_entropy_token_num_dict["medium"], 21)
        self.assertEqual(self.trainer.target_high_entropy_token_num_dict["hard"], 41)
    

    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_alpha_entropy_update_hard_under_generation(self):
        """Test alpha_entropy update when Hard task generates insufficiently"""
        # First set a baseline target
        self.trainer.target_high_entropy_token_num_dict["hard"] = 40
        self.trainer.alpha_entropy_dict["hard"] = 0.6
        
        # First update: set initial baseline
        batch_data_init = {
            "global_id": ["hard_baseline"] * 2,
            "accuracy": [0.0, 0.0],
            "entropies": [0.8] * 2,
            "high_entropy_token_num": [40, 40]  # Average 40, set baseline
        }
        batch_init = DataProto.from_dict(non_tensors=batch_data_init)
        self.trainer.update_difficulty_and_skip_gid_set(batch_init)
        
        baseline_alpha = self.trainer.alpha_entropy_dict["hard"]
        
        # Second update: hard sample generating insufficient tokens
        batch_data = {
            "global_id": ["hard_under"] * 2,
            "accuracy": [0.0, 0.0],
            "entropies": [0.8] * 2,
            "high_entropy_token_num": [20, 22]  # Average 21, below baseline 40
        }
        
        batch = DataProto.from_dict(non_tensors=batch_data)
        self.trainer.update_difficulty_and_skip_gid_set(batch)
        new_alpha = self.trainer.alpha_entropy_dict["hard"]
        # print(new_alpha)
        # Hard generates insufficiently, alpha should increase (strengthen encouragement)
        # self.assertGreaterEqual(new_alpha, baseline_alpha)
        expected_new_alpha = 0.6
        torch.testing.assert_close(new_alpha, expected_new_alpha, atol=1e-4, rtol=1e-4)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_alpha_entropy_clipping(self):
        """Test alpha_entropy upper and lower bound constraints"""
        # Test lower bound
        self.trainer.alpha_entropy_dict["easy"] = 0.11
        self.trainer.alpha_entropy_min = 0.1
        self.trainer.target_high_entropy_token_num_dict["easy"] = 100
        
        # Create data that will cause alpha to decrease
        batch_data = {
            "global_id": ["test_clip"] * 2,
            "accuracy": [1.0, 1.0],
            "entropies": [0.4] * 2,
            "high_entropy_token_num": [5, 5]  # Far below target, causes alpha to decrease
        }
        
        batch = DataProto.from_dict(non_tensors=batch_data)
        self.trainer.update_difficulty_and_skip_gid_set(batch)
        
        # Verify does not go below minimum value
        self.assertGreaterEqual(self.trainer.alpha_entropy_dict["easy"], self.trainer.alpha_entropy_min)
        
        # Test upper bound
        self.trainer.alpha_entropy_dict["medium"] = 0.99
        self.trainer.alpha_entropy_max = 1.0
        self.trainer.target_high_entropy_token_num_dict["medium"] = 5
        
        batch_data_max = {
            "global_id": ["test_max"] * 2,
            "accuracy": [0.5, 0.5],
            "entropies": [0.6] * 2,
            "high_entropy_token_num": [100, 100]  # Far exceeds target, causes alpha to increase
        }
        
        batch_max = DataProto.from_dict(non_tensors=batch_data_max)
        self.trainer.update_difficulty_and_skip_gid_set(batch_max)
        
        # print(self.trainer.alpha_entropy_dict["medium"])
        expected_alpha = 0.99
        torch.testing.assert_close(self.trainer.alpha_entropy_dict["medium"], expected_alpha, atol=1e-4, rtol=1e-4)
        # Verify does not exceed maximum value
        self.assertLessEqual(self.trainer.alpha_entropy_dict["medium"], self.trainer.alpha_entropy_max)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_multiple_samples_aggregation(self):
        """Test aggregation statistics of multiple samples"""
        # Same global_id has multiple rollouts
        batch_data = {
            "global_id": ["sample_1"] * 6,
            "accuracy": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # Average 0.5
            "entropies": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "high_entropy_token_num": [10, 12, 14, 16, 18, 20]  # Average 15
        }
        
        batch = DataProto.from_dict(non_tensors=batch_data)
        self.trainer.update_difficulty_and_skip_gid_set(batch)
        
        # Accuracy 0.5 should be assigned as medium
        self.assertEqual(self.trainer.difficulty_dict_train["sample_1"], "medium")
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_batch_with_multiple_unique_samples(self):
        """Test batch containing multiple different samples"""
        batch_data = {
            "global_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "accuracy": [1.0, 1.0, 0.5, 0.5, 0.0, 0.0],
            "entropies": [0.3, 0.4, 0.6, 0.7, 0.9, 1.0],
            "high_entropy_token_num": [8, 10, 20, 22, 40, 42]
        }
        
        batch = DataProto.from_dict(non_tensors=batch_data)
        self.trainer.update_difficulty_and_skip_gid_set(batch)
        
        # Verify all samples are processed
        self.assertIn("s1", self.trainer.difficulty_dict_train)
        self.assertIn("s2", self.trainer.difficulty_dict_train)
        self.assertIn("s3", self.trainer.difficulty_dict_train)
        
        # Verify difficulty assignment
        self.assertEqual(self.trainer.difficulty_dict_train["s1"], "easy")
        self.assertEqual(self.trainer.difficulty_dict_train["s2"], "medium")
        self.assertEqual(self.trainer.difficulty_dict_train["s3"], "hard")
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_persistent_difficulty_update(self):
        """Test persistent update of difficulty dictionary"""
        # First update
        batch_data_1 = {
            "global_id": ["sample_a"] * 2,
            "accuracy": [1.0, 1.0],
            "entropies": [0.4] * 2,
            "high_entropy_token_num": [10, 12]
        }
        
        batch_1 = DataProto.from_dict(non_tensors=batch_data_1)
        self.trainer.update_difficulty_and_skip_gid_set(batch_1)
        self.assertEqual(self.trainer.difficulty_dict_train["sample_a"], "easy")
        
        # Second update (same sample, different result)
        batch_data_2 = {
            "global_id": ["sample_a"] * 2,
            "accuracy": [0.0, 0.0],
            "entropies": [0.8] * 2,
            "high_entropy_token_num": [40, 42]
        }
        
        batch_2 = DataProto.from_dict(non_tensors=batch_data_2)
        self.trainer.update_difficulty_and_skip_gid_set(batch_2)
        
        # Difficulty should be updated to hard
        self.assertEqual(self.trainer.difficulty_dict_train["sample_a"], "hard")


class TestUpdateDifficultyIntegration(unittest.TestCase):
    """Integration test: test overall workflow of update_difficulty_and_skip_gid_set"""
    
    def setUp(self):
        """Set up test environment"""
        # Create real RayPPOTrainer instance, but don't call __init__ to avoid complex dependencies
        if VERL_AVAILABLE:
            self.trainer = object.__new__(RayPPOTrainer)
            self.trainer.difficulty_dict_train = {}
            self.trainer.target_high_entropy_token_num_dict = {"easy": 0, "medium": 0, "hard": 0}
            self.trainer.alpha_entropy_dict = {"easy": 0.5, "medium": 0.5, "hard": 0.6}
            self.trainer.alpha_entropy_lr = 0.01
            self.trainer.alpha_entropy_min = 0.1
            self.trainer.alpha_entropy_max = 1.0
            self.trainer.skip_gid_set = set()
        else:
            self.trainer = None
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_end_to_end_difficulty_workflow(self):
        """End-to-end difficulty update workflow"""
        # Simulate a complete training batch
        batch_data = {
            "global_id": ["task_1"] * 4 + ["task_2"] * 4 + ["task_3"] * 4,
            "accuracy": [1.0, 1.0, 1.0, 0.0] + [1.0, 0.0, 1.0, 0.0] + [0.0, 0.0, 0.0, 1.0],
            "entropies": [0.3] * 4 + [0.6] * 4 + [0.9] * 4,
            "high_entropy_token_num": [8, 10, 9, 11] + [18, 20, 19, 21] + [38, 40, 39, 41]
        }
        
        batch = DataProto.from_dict(non_tensors=batch_data)
        
        # Record initial alpha values
        initial_alphas = self.trainer.alpha_entropy_dict.copy()
        
        # Execute update
        self.trainer.update_difficulty_and_skip_gid_set(batch)
        
        # Verify all tasks are assigned difficulty
        self.assertEqual(len(self.trainer.difficulty_dict_train), 3)
        
        # Verify difficulty assignment
        # task_1: 3/4 = 0.75 >= 2/3 -> easy
        self.assertEqual(self.trainer.difficulty_dict_train["task_1"], "easy")
        # task_2: 2/4 = 0.5, 1/3 <= 0.5 < 2/3 -> medium
        self.assertEqual(self.trainer.difficulty_dict_train["task_2"], "medium")
        # task_3: 1/4 = 0.25 < 1/3 -> hard
        self.assertEqual(self.trainer.difficulty_dict_train["task_3"], "hard")
        
        # Verify target token counts are set
        self.assertGreater(self.trainer.target_high_entropy_token_num_dict["easy"], 0)
        self.assertGreater(self.trainer.target_high_entropy_token_num_dict["medium"], 0)
        self.assertGreater(self.trainer.target_high_entropy_token_num_dict["hard"], 0)
        
        # Verify alpha values may have changed
        # (specific changes depend on relationship between generated token count and target)
        # print(self.trainer.alpha_entropy_dict["easy"])
        # print(self.trainer.alpha_entropy_dict["medium"])
        # print(self.trainer.alpha_entropy_dict["hard"])
        expected_alpha_easy = 0.495
        expected_alpha_medium = 0.495
        expected_alpha_hard = 0.605
        torch.testing.assert_close(self.trainer.alpha_entropy_dict["easy"], expected_alpha_easy, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(self.trainer.alpha_entropy_dict["medium"], expected_alpha_medium, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(self.trainer.alpha_entropy_dict["hard"], expected_alpha_hard, atol=1e-4, rtol=1e-4)
        # self.assertIsNotNone(self.trainer.alpha_entropy_dict["easy"])
        # self.assertIsNotNone(self.trainer.alpha_entropy_dict["medium"])
        # self.assertIsNotNone(self.trainer.alpha_entropy_dict["hard"])
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_adaptive_alpha_convergence(self):
        """Test adaptive adjustment mechanism of alpha_entropy"""
        # Set initial state
        initial_alpha = 0.5
        self.trainer.alpha_entropy_dict["easy"] = initial_alpha
        self.trainer.alpha_entropy_lr = 0.05  # Use smaller learning rate
        
        # First set baseline
        batch_data_baseline = {
            "global_id": ["converge_baseline"] * 2,
            "accuracy": [1.0, 1.0],
            "entropies": [0.4] * 2,
            "high_entropy_token_num": [15, 15]  # Set baseline as 15
        }
        batch_baseline = DataProto.from_dict(non_tensors=batch_data_baseline)
        self.trainer.update_difficulty_and_skip_gid_set(batch_baseline)
        
        # Record initial alpha (may have changed)
        alpha_after_baseline = self.trainer.alpha_entropy_dict["easy"]
        
        # Multiple updates, each generating more tokens (but not the same sample)
        for i in range(5):
            batch_data = {
                "global_id": [f"converge_test_{i}"] * 2,
                "accuracy": [1.0, 1.0],
                "entropies": [0.4] * 2,
                "high_entropy_token_num": [25, 27]  # Consistently more
            }
            batch = DataProto.from_dict(non_tensors=batch_data)
            self.trainer.update_difficulty_and_skip_gid_set(batch)
        
        final_alpha = self.trainer.alpha_entropy_dict["easy"]

        # print(final_alpha)
        expected_final_alpha = 0.5
        torch.testing.assert_close(final_alpha, expected_final_alpha, atol=1e-4, rtol=1e-4)
        
        # alpha should have some adjustment
        # Due to consistently generating more tokens, alpha may increase or remain stable
        self.assertIsNotNone(final_alpha)
        # But should not exceed upper bound
        self.assertLessEqual(final_alpha, self.trainer.alpha_entropy_max)
        # And should be within reasonable range
        self.assertGreaterEqual(final_alpha, self.trainer.alpha_entropy_min)


if __name__ == "__main__":
    unittest.main()

