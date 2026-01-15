import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock
from PIL import Image
import tempfile

# Add the parent directory to Python path to import open_r1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))

# Import the actual function we want to test
from open_r1.vlm_modules.qwen_module import Qwen2VLModule


class TestIoUReward(unittest.TestCase):
    """Test iou_reward function - IoU reward calculation as described in VLM-R1 documentation"""
    
    def setUp(self):
        """Create temporary test image"""
        self.temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        # Create an 800x600 test image
        img = Image.new('RGB', (800, 600), color='white')
        img.save(self.temp_image.name)
        self.temp_image.close()
        
        # Clear DEBUG_MODE environment variable
        if "DEBUG_MODE" in os.environ:
            del os.environ["DEBUG_MODE"]
    
    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_image.name):
            os.remove(self.temp_image.name)
    
    def test_iou_reward_perfect_match(self):
        """Test perfectly matched bounding box"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><answer>[100, 100, 200, 200]</answer>"}]
        ]
        solution = [f"<answer>{json.dumps([100, 100, 200, 200])}</answer>"]
        
        # image_grid_thw: [1, 43, 57] -> input_height=602, input_width=798
        # Original image: 800x600
        # Scaled bbox should be almost the same as original bbox
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],
            "image_path": [[self.temp_image.name]],
            "problem": ["test problem"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        # print(rewards)
        expected_reward = [0.9826998191532413]
        self.assertEqual(len(rewards), 1)
        # Since scaling factor is close to 1, IoU should be close to 1.0
        self.assertAlmostEqual(rewards[0], expected_reward[0], places=4)
    
    def test_iou_reward_partial_overlap(self):
        """Test partially overlapping bounding box"""
        completions = [
            [{"role": "assistant", "content": "<answer>[100, 100, 200, 200]</answer>"}]
        ]
        # Ground truth box has 50% overlap with predicted box
        solution = [f"<answer>{json.dumps([150, 150, 250, 250])}</answer>"]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],
            "image_path": [[self.temp_image.name]],
            "problem": ["test problem"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        self.assertEqual(len(rewards), 1)
        # Partial overlap should have some IoU value
        expected_reward = [0.14236979677770442]
        self.assertAlmostEqual(rewards[0], expected_reward[0], places=4)
    
    def test_iou_reward_no_overlap(self):
        """Test completely non-overlapping bounding box"""
        completions = [
            [{"role": "assistant", "content": "<answer>[100, 100, 200, 200]</answer>"}]
        ]
        # Ground truth box completely does not overlap with predicted box
        solution = [f"<answer>{json.dumps([300, 300, 400, 400])}</answer>"]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],
            "image_path": [[self.temp_image.name]],
            "problem": ["test problem"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        self.assertEqual(len(rewards), 1)
        # Complete non-overlap, IoU should be 0
        self.assertEqual(rewards[0], 0.0)
    
    def test_iou_reward_bbox_scaling(self):
        """Test bounding box coordinate scaling"""
        # Test different image_grid_thw values
        # Use grid size close to image dimensions (800x600) to reduce scaling error
        completions = [
            [{"role": "assistant", "content": "<answer>[100, 100, 200, 200]</answer>"}]
        ]
        solution = [f"<answer>{json.dumps([100, 100, 200, 200])}</answer>"]
        
        # Use grid close to actual image size: 800/14≈57, 600/14≈43
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],  # input_height=602, input_width=798
            "image_path": [[self.temp_image.name]],
            "problem": ["test problem"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        expected_reward = [0.9826998191532413]
        self.assertEqual(len(rewards), 1)
        # When scaling factor is close to 1, IoU should be very high
        self.assertAlmostEqual(rewards[0], expected_reward[0], places=4)
    
    def test_iou_reward_missing_answer_tag(self):
        """Test missing answer tag case"""
        completions = [
            [{"role": "assistant", "content": "[100, 100, 200, 200]"}]  # No answer tag
        ]
        solution = [f"<answer>{json.dumps([100, 100, 200, 200])}</answer>"]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],
            "image_path": [[self.temp_image.name]],
            "problem": ["test problem"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        self.assertEqual(len(rewards), 1)
        # Cannot extract answer, should return 0.0
        self.assertEqual(rewards[0], 0.0)
    
    def test_iou_reward_invalid_bbox_format(self):
        """Test invalid bounding box format"""
        completions = [
            [{"role": "assistant", "content": "<answer>invalid bbox</answer>"}]
        ]
        solution = [f"<answer>{json.dumps([100, 100, 200, 200])}</answer>"]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],
            "image_path": [[self.temp_image.name]],
            "problem": ["test problem"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        self.assertEqual(len(rewards), 1)
        # Invalid format should return 0.0
        self.assertEqual(rewards[0], 0.0)
    
    def test_iou_reward_batch_processing(self):
        """Test batch processing"""
        completions = [
            [{"role": "assistant", "content": "<answer>[100, 100, 200, 200]</answer>"}],
            [{"role": "assistant", "content": "<answer>[150, 150, 250, 250]</answer>"}],
            [{"role": "assistant", "content": "<answer>[300, 300, 400, 400]</answer>"}]
        ]
        solution = [
            f"<answer>{json.dumps([100, 100, 200, 200])}</answer>",
            f"<answer>{json.dumps([150, 150, 250, 250])}</answer>",
            f"<answer>{json.dumps([100, 100, 200, 200])}</answer>"  # Does not match third prediction
        ]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57], [1, 43, 57], [1, 43, 57]],
            "image_path": [[self.temp_image.name], [self.temp_image.name], [self.temp_image.name]],
            "problem": ["test1", "test2", "test3"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)

        # print(rewards)
        expected_reward = [0.9826998191532413, 0.9770143336194704, 0.0]
        self.assertEqual(len(rewards), 3)
        # First two should have high IoU, third should have low IoU
        self.assertAlmostEqual(rewards[0], expected_reward[0], places=4)
        self.assertAlmostEqual(rewards[1], expected_reward[1], places=4)
        self.assertAlmostEqual(rewards[2], expected_reward[2], places=4)
    
    def test_iou_reward_edge_bbox(self):
        """Test edge bounding box"""
        # Test bounding box at image edge
        completions = [
            [{"role": "assistant", "content": "<answer>[0, 0, 100, 100]</answer>"}]
        ]
        solution = [f"<answer>{json.dumps([0, 0, 100, 100])}</answer>"]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],
            "image_path": [[self.temp_image.name]],
            "problem": ["test problem"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)

        # print(rewards)
        expected_reward = [0.9941943039394016]
        self.assertEqual(len(rewards), 1)
        # Edge box should also correctly calculate IoU
        self.assertAlmostEqual(rewards[0], expected_reward[0], places=4)
    
    def test_iou_reward_large_bbox(self):
        """Test large bounding box"""
        # Test bounding box covering entire image
        completions = [
            [{"role": "assistant", "content": "<answer>[0, 0, 800, 600]</answer>"}]
        ]
        solution = [f"<answer>{json.dumps([0, 0, 800, 600])}</answer>"]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],
            "image_path": [[self.temp_image.name]],
            "problem": ["test problem"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        # print(rewards)
        self.assertEqual(len(rewards), 1)
        expected_reward = [0.9941943039394016]
        self.assertAlmostEqual(rewards[0], expected_reward[0], places=4)
    
    def test_iou_reward_small_bbox(self):
        """Test small bounding box"""
        # Test very small bounding box
        completions = [
            [{"role": "assistant", "content": "<answer>[100, 100, 110, 110]</answer>"}]
        ]
        solution = [f"<answer>{json.dumps([100, 100, 110, 110])}</answer>"]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],
            "image_path": [[self.temp_image.name]],
            "problem": ["test problem"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        self.assertEqual(len(rewards), 1)
        # Small box should also correctly calculate IoU
        expected_reward = [0.8862509721971276]
        self.assertAlmostEqual(rewards[0], expected_reward[0], places=4)
    
    def test_iou_reward_with_think_tag(self):
        """Test complete format with think tag"""
        completions = [
            [{"role": "assistant", "content": "<think>Analyzing image, target at center position</think><answer>[100, 100, 200, 200]</answer>"}]
        ]
        solution = [f"<answer>{json.dumps([100, 100, 200, 200])}</answer>"]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],
            "image_path": [[self.temp_image.name]],
            "problem": ["test problem"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        # print(rewards)
        self.assertEqual(len(rewards), 1)
        expected_reward = [0.9826998191532413]
        self.assertAlmostEqual(rewards[0], expected_reward[0], places=4)

    def test_iou_reward_different_image_sizes(self):
        """Test different image sizes"""
        # Create different sized image
        temp_image2 = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        img2 = Image.new('RGB', (1920, 1080), color='white')
        img2.save(temp_image2.name)
        temp_image2.close()
        
        try:
            # Use grid and bbox matching image dimensions
            # 1920/14≈137, 1080/14≈77
            completions = [
                [{"role": "assistant", "content": "<answer>[100, 100, 200, 200]</answer>"}]
            ]
            solution = [f"<answer>{json.dumps([100, 100, 200, 200])}</answer>"]
            
            kwargs = {
                "image_grid_thw": [[1, 77, 137]],  # Grid matching 1920x1080
                "image_path": [[temp_image2.name]],
                "problem": ["test problem"]
            }
            
            rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
            
            self.assertEqual(len(rewards), 1)
            # Using matching grid size, IoU should be very high
            expected_reward = [0.991357903584749]
            self.assertAlmostEqual(rewards[0], expected_reward[0], places=4)
        finally:
            os.remove(temp_image2.name)


class TestIoURewardIntegration(unittest.TestCase):
    """Integration tests: test iou_reward performance in real scenarios"""
    
    def setUp(self):
        """Create temporary test image"""
        self.temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        img = Image.new('RGB', (800, 600), color='white')
        img.save(self.temp_image.name)
        self.temp_image.close()
    
    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_image.name):
            os.remove(self.temp_image.name)
    
    def test_rec_task_scenario(self):
        """Test REC task scenario"""
        # Simulate a real REC task
        completions = [
            [{"role": "assistant", "content": 
              "<think>Target in image is located center-left, occupying about 1/4 of image area</think>"
              "<answer>[200, 150, 400, 350]</answer>"}]
        ]
        solution = [f"<answer>{json.dumps([200, 150, 400, 350])}</answer>"]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],
            "image_path": [[self.temp_image.name]],
            "problem": ["Please locate the red car in the image"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)

        # print(rewards)
        expected_reward = [0.9843302547573303]
        
        self.assertEqual(len(rewards), 1)
        self.assertAlmostEqual(rewards[0], expected_reward[0], places=4)
    
    def test_quality_control_scenario(self):
        """Test quality control scenario"""
        # Test different quality predictions
        completions = [
            [{"role": "assistant", "content": "<answer>[100, 100, 200, 200]</answer>"}],  # Good prediction
            [{"role": "assistant", "content": "<answer>[90, 90, 210, 210]</answer>"}],   # Slightly worse prediction
            [{"role": "assistant", "content": "<answer>[300, 300, 400, 400]</answer>"}], # Bad prediction
        ]
        solution = [
            f"<answer>{json.dumps([100, 100, 200, 200])}</answer>",
            f"<answer>{json.dumps([100, 100, 200, 200])}</answer>",
            f"<answer>{json.dumps([100, 100, 200, 200])}</answer>"
        ]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57], [1, 43, 57], [1, 43, 57]],
            "image_path": [[self.temp_image.name], [self.temp_image.name], [self.temp_image.name]],
            "problem": ["test1", "test2", "test3"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        expected_reward = [0.9826998191532413, 0.6950173611111112, 0.0]
        # Can be used for filtering: only keep predictions with high IoU
        self.assertAlmostEqual(rewards[0], expected_reward[0], places=4)
        self.assertAlmostEqual(rewards[1], expected_reward[1], places=4)
        self.assertAlmostEqual(rewards[2], expected_reward[2], places=4)


if __name__ == "__main__":
    unittest.main()
