import unittest
import sys
import os

# Add the parent directory to Python path to import open_r1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))

# Import the actual function we want to test
from open_r1.vlm_modules.qwen_module import Qwen2VLModule


class TestFormatRewardRec(unittest.TestCase):
    """Test format_reward_rec function - REC task format validation as described in VLM-R1 documentation"""
    
    def setUp(self):
        """Clear DEBUG_MODE environment variable"""
        if "DEBUG_MODE" in os.environ:
            del os.environ["DEBUG_MODE"]
    
    def test_correct_format(self):
        """Test correct format"""
        completions = [
            [{"role": "assistant", "content": '<think>thinking process</think><answer>{"bbox": [100, 200, 300, 400]}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_correct_format_with_json_object(self):
        """Test correct format with JSON object"""
        completions = [
            [{"role": "assistant", "content": '<think>thinking</think><answer>{"bbox": [100, 200, 300, 400]}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_correct_format_with_whitespace(self):
        """Test correct format with whitespace between tags"""
        completions = [
            [{"role": "assistant", "content": '<think>thinking</think>  <answer>{"bbox": [100, 200, 300, 400]}</answer>'}],
            [{"role": "assistant", "content": '<think>thinking</think>\n<answer>{"bbox": [100, 200, 300, 400]}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 2)
        self.assertTrue(all(r == 1.0 for r in rewards))
    
    def test_missing_think_tag(self):
        """Test missing think tag"""
        completions = [
            [{"role": "assistant", "content": "<answer>[100, 200, 300, 400]</answer>"}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_missing_answer_tag(self):
        """Test missing answer tag"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking process</think>[100, 200, 300, 400]"}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_missing_bbox(self):
        """Test missing bbox coordinates"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><answer>no bbox</answer>"}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_incomplete_bbox(self):
        """Test incomplete bbox (less than 4 numbers)"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><answer>[100, 200]</answer>"}],
            [{"role": "assistant", "content": "<think>thinking</think><answer>[100, 200, 300]</answer>"}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 0.0)
        self.assertEqual(rewards[1], 0.0)
    
    def test_wrong_order(self):
        """Test wrong tag order"""
        completions = [
            [{"role": "assistant", "content": "<answer>[100, 200, 300, 400]</answer><think>thinking</think>"}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_extra_content_before(self):
        """Test extra content before tags"""
        completions = [
            [{"role": "assistant", "content": "extra content<think>thinking</think><answer>[100, 200, 300, 400]</answer>"}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        # print(rewards)

        expected_reward = [0.0]
        self.assertEqual(len(rewards), 1)
        self.assertAlmostEqual(rewards[0], expected_reward[0], places=4)
    
    def test_extra_content_after(self):
        """Test extra content after tags"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><answer>[100, 200, 300, 400]</answer>extra content"}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        # print(rewards)

        expected_reward = [0.0]
        self.assertEqual(len(rewards), 1)
        self.assertAlmostEqual(rewards[0], expected_reward[0], places=4)
    
    def test_multiline_content(self):
        """Test multiline content"""
        completions = [
            [{"role": "assistant", "content": '<think>thinking\nprocess\nmultiline</think><answer>{"bbox": [100, 200, 300, 400]}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_bbox_with_spaces(self):
        """Test bbox coordinates with spaces"""
        completions = [
            [{"role": "assistant", "content": '<think>thinking</think><answer>{"bbox": [100,  200,  300,  400]}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_bbox_without_spaces(self):
        """Test bbox coordinates without spaces"""
        completions = [
            [{"role": "assistant", "content": '<think>thinking</think><answer>{"bbox": [100,200,300,400]}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_json_object_format(self):
        """Test JSON object format"""
        completions = [
            [{"role": "assistant", "content": '<think>thinking</think><answer>{"x1": 100, "y1": 200, "x2": 300, "y2": 400, "bbox": [100, 200, 300, 400]}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_nested_json(self):
        """Test nested JSON"""
        completions = [
            [{"role": "assistant", "content": '<think>thinking</think><answer>{"result": {"bbox": [100, 200, 300, 400]}}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_batch_processing(self):
        """Test batch processing"""
        completions = [
            [{"role": "assistant", "content": '<think>thinking1</think><answer>{"bbox": [100, 200, 300, 400]}</answer>'}],
            [{"role": "assistant", "content": '<think>thinking2</think><answer>{"bbox": [50, 50, 150, 150]}</answer>'}],
            [{"role": "assistant", "content": '<think>thinking3</think><answer>{"bbox": [200, 200, 400, 400]}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 3)
        self.assertTrue(all(r == 1.0 for r in rewards))
    
    def test_mixed_batch(self):
        """Test mixed batch with correct and incorrect formats"""
        completions = [
            [{"role": "assistant", "content": '<think>thinking</think><answer>{"bbox": [100, 200, 300, 400]}</answer>'}],  # Correct
            [{"role": "assistant", "content": '<answer>{"bbox": [100, 200, 300, 400]}</answer>'}],  # Incorrect: missing think
            [{"role": "assistant", "content": '<think>thinking</think><answer>{"bbox": [100, 200]}</answer>'}],  # Incorrect: incomplete bbox
            [{"role": "assistant", "content": '<think>thinking</think><answer>{"bbox": [50, 50, 150, 150]}</answer>'}],  # Correct
            [{"role": "assistant", "content": "<think>thinking</think><answer>no bbox</answer>"}]  # Incorrect: no bbox
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 5)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 0.0)
        self.assertEqual(rewards[2], 0.0)
        self.assertEqual(rewards[3], 1.0)
        self.assertEqual(rewards[4], 0.0)
    
    def test_case_sensitive_tags(self):
        """Test case sensitive tags"""
        completions = [
            [{"role": "assistant", "content": "<Think>thinking</Think><answer>[100, 200, 300, 400]</answer>"}],
            [{"role": "assistant", "content": "<think>thinking</think><Answer>[100, 200, 300, 400]</Answer>"}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 0.0)
        self.assertEqual(rewards[1], 0.0)
    
    def test_float_coordinates(self):
        """Test float coordinates"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><answer>[100.5, 200.5, 300.5, 400.5]</answer>"}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)


        expected_reward = [0.0]
        # Regex requires \d+ (integers), so floats may not match
        # Actual behavior depends on regex implementation
        self.assertAlmostEqual(rewards[0], expected_reward[0], places=4)
    
    def test_negative_coordinates(self):
        """Test negative coordinates"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><answer>[-100, -200, 300, 400]</answer>"}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)

        expected_reward = [0.0]
        # Regex requires \d+ (positive integers), so negatives may not match
        self.assertAlmostEqual(rewards[0], expected_reward[0], places=4)
    
    def test_very_large_coordinates(self):
        """Test very large coordinate values"""
        completions = [
            [{"role": "assistant", "content": '<think>thinking</think><answer>{"bbox": [10000, 20000, 30000, 40000]}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)


class TestFormatRewardRecIntegration(unittest.TestCase):
    """Integration tests: test format_reward_rec performance in real scenarios"""
    
    def test_rec_task_scenario(self):
        """Test REC task scenario"""
        completions = [
            [{"role": "assistant", "content": 
              "<think>Analyzing image, target is located center-left, occupying about 1/4 of the image area</think>"
              '<answer>{"bbox": [200, 150, 400, 350]}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_quality_control_scenario(self):
        """Test quality control scenario"""
        completions = [
            [{"role": "assistant", "content": '<think>thinking</think><answer>{"bbox": [100, 100, 200, 200]}</answer>'}],  # Good generation
            [{"role": "assistant", "content": "target at [100, 100, 200, 200]"}],  # Bad generation
            [{"role": "assistant", "content": '<think>thinking</think><answer>{"bbox": [50, 50, 150, 150]}</answer>'}],  # Good generation
            [{"role": "assistant", "content": '<answer>{"bbox": [200, 200, 300, 300]}</answer>'}]  # Bad generation: missing think
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        # Can be used for filtering: only keep generations with reward=1.0
        good_indices = [i for i, r in enumerate(rewards) if r == 1.0]
        self.assertEqual(good_indices, [0, 2])
    
    def test_multimodal_vlm_scenario(self):
        """Test multimodal VLM scenario"""
        completions = [
            [{"role": "assistant", "content": 
              "<think>The red car in the image is located in the center of the frame, bbox coordinates are...</think>"
              '<answer>{"bbox": [250, 180, 550, 420], "confidence": 0.95}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)


if __name__ == "__main__":
    unittest.main()
