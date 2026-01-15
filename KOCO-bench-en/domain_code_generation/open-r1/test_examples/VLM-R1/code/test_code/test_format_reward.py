import unittest
import sys
import os

# Add the parent directory to Python path to import open_r1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))

# Import the actual function we want to test
from open_r1.grpo_jsonl import format_reward


class TestFormatReward(unittest.TestCase):
    """Test format_reward function - format validation as described in VLM-R1 documentation"""
    
    def setUp(self):
        """Clear DEBUG_MODE environment variable"""
        if "DEBUG_MODE" in os.environ:
            del os.environ["DEBUG_MODE"]
    
    def test_correct_format(self):
        """Test correct format"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking process</think><answer>42</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_correct_format_with_whitespace(self):
        """Test correct format with whitespace between tags"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think>  <answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>thinking</think>\n<answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>thinking</think>\t<answer>42</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 3)
        self.assertTrue(all(r == 1.0 for r in rewards))
    
    def test_missing_think_tag(self):
        """Test missing think tag"""
        completions = [
            [{"role": "assistant", "content": "<answer>42</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_missing_answer_tag(self):
        """Test missing answer tag"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking process</think>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_wrong_order(self):
        """Test wrong tag order"""
        completions = [
            [{"role": "assistant", "content": "<answer>42</answer><think>thinking</think>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_extra_content_before(self):
        """Test extra content before tags"""
        completions = [
            [{"role": "assistant", "content": "extra content<think>thinking</think><answer>42</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_extra_content_after(self):
        """Test extra content after tags"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><answer>42</answer>extra content"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_extra_content_between(self):
        """Test extra content between tags (non-whitespace)"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think>extra content<answer>42</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_empty_tags(self):
        """Test empty tags"""
        completions = [
            [{"role": "assistant", "content": "<think></think><answer></answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_multiline_content(self):
        """Test multiline content"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking\nprocess\nmultiline</think><answer>42</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_nested_tags(self):
        """Test nested tags"""
        completions = [
            [{"role": "assistant", "content": "<think><inner>nested</inner>thinking</think><answer>42</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_case_sensitive_tags(self):
        """Test case sensitive tags"""
        completions = [
            [{"role": "assistant", "content": "<Think>thinking</Think><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>thinking</think><Answer>42</Answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 0.0)
        self.assertEqual(rewards[1], 0.0)
    
    def test_batch_processing(self):
        """Test batch processing"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking1</think><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>thinking2</think><answer>100</answer>"}],
            [{"role": "assistant", "content": "<think>thinking3</think><answer>200</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 3)
        self.assertTrue(all(r == 1.0 for r in rewards))
    
    def test_mixed_batch(self):
        """Test mixed batch with correct and incorrect formats"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><answer>42</answer>"}],  # Correct
            [{"role": "assistant", "content": "<answer>42</answer>"}],  # Incorrect: missing think
            [{"role": "assistant", "content": "<think>thinking</think><answer>100</answer>"}],  # Correct
            [{"role": "assistant", "content": "completely wrong"}],  # Incorrect
            [{"role": "assistant", "content": "<think>thinking</think><answer>200</answer>"}]  # Correct
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 5)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 0.0)
        self.assertEqual(rewards[2], 1.0)
        self.assertEqual(rewards[3], 0.0)
        self.assertEqual(rewards[4], 1.0)
    
    def test_special_characters_in_content(self):
        """Test special characters in content"""
        completions = [
            [{"role": "assistant", "content": "<think>x^2 + y^2 = r^2</think><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>α + β = γ</think><answer>100</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 2)
        self.assertTrue(all(r == 1.0 for r in rewards))
    
    def test_very_long_content(self):
        """Test very long content"""
        long_content = "This is a very long thinking process." * 100
        completions = [
            [{"role": "assistant", "content": f"<think>{long_content}</think><answer>42</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_incomplete_tags(self):
        """Test incomplete tags"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><answer>42"}],  # Missing closing tag
            [{"role": "assistant", "content": "<think>thinking<answer>42</answer>"}]  # Missing think closing tag
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 0.0)
        self.assertEqual(rewards[1], 0.0)
    
    def test_duplicate_tags(self):
        """Test duplicate tags"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking1</think><think>thinking2</think><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>thinking</think><answer>42</answer><answer>100</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        # fullmatch requires exact match, duplicate tags may cause match failure
        self.assertEqual(len(rewards), 2)
        # First may match successfully (because .*? can match content with duplicate tags)
        # Second may match successfully (because .*? can match content with duplicate tags)
        # Actual behavior depends on regex implementation
        expected_reward = [1.0, 1.0]
        self.assertAlmostEqual(rewards[0], expected_reward[0], places=4)
        self.assertAlmostEqual(rewards[1], expected_reward[1], places=4)

class TestFormatRewardIntegration(unittest.TestCase):
    """Integration tests: test format_reward performance in real scenarios"""
    
    def test_realistic_generation_scenario(self):
        """Test realistic generation scenario"""
        completions = [
            [{"role": "assistant", "content": 
              "<think>First analyze the problem, this is a simple math calculation. 2+2=4.</think>"
              "<answer>4</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_quality_control_scenario(self):
        """Test quality control scenario"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><answer>42</answer>"}],  # Good generation
            [{"role": "assistant", "content": "direct answer: 42"}],  # Bad generation
            [{"role": "assistant", "content": "<think>thinking</think><answer>100</answer>"}],  # Good generation
            [{"role": "assistant", "content": "<answer>200</answer>"}]  # Bad generation
        ]
        
        rewards = format_reward(completions)
        
        # Can be used for filtering: only keep generations with reward=1.0
        good_indices = [i for i, r in enumerate(rewards) if r == 1.0]
        self.assertEqual(good_indices, [0, 2])
    
    def test_multimodal_vlm_scenario(self):
        """Test multimodal VLM scenario"""
        completions = [
            [{"role": "assistant", "content": 
              "<think>Analyzing image content, identifying target location and category</think>"
              "<answer>There is a red car in the image</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)


if __name__ == "__main__":
    unittest.main()
