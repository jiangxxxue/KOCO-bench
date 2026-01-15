import unittest
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))

# Import the actual function we want to test
from open_r1.grpo_jsonl import format_reward


class TestFormatReward(unittest.TestCase):
    """Test format_reward function - check if generated responses conform to specified format requirements"""
    
    def test_format_reward_correct_format(self):
        """Test responses with correct format"""
        completions = [
            [{"role": "assistant", "content": "<think>This is reasoning</think><answer>4.5</answer>"}],
            [{"role": "assistant", "content": "<think>Another reasoning</think><answer>3.2</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # Verify output
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 1.0)
    
    def test_format_reward_missing_think_tag(self):
        """Test case with missing think tag"""
        completions = [
            [{"role": "assistant", "content": "<answer>4.5</answer>"}],
            [{"role": "assistant", "content": "Just some text<answer>3.2</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # Missing think tag should return 0.0
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 0.0)
        self.assertEqual(rewards[1], 0.0)
    
    def test_format_reward_missing_answer_tag(self):
        """Test case with missing answer tag"""
        completions = [
            [{"role": "assistant", "content": "<think>This is reasoning</think>"}],
            [{"role": "assistant", "content": "<think>Another reasoning</think>Some text"}],
        ]
        
        rewards = format_reward(completions)
        
        # Missing answer tag should return 0.0
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 0.0)
        self.assertEqual(rewards[1], 0.0)
    
    def test_format_reward_wrong_order(self):
        """Test case with incorrect tag order"""
        completions = [
            [{"role": "assistant", "content": "<answer>4.5</answer><think>This is reasoning</think>"}],
        ]
        
        rewards = format_reward(completions)
        
        # Incorrect tag order should return 0.0
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_format_reward_extra_content_before(self):
        """Test case with extra content before tags"""
        completions = [
            [{"role": "assistant", "content": "Extra text<think>This is reasoning</think><answer>4.5</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # Extra content before tags should return 0.0
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_format_reward_extra_content_after(self):
        """Test case with extra content after tags"""
        completions = [
            [{"role": "assistant", "content": "<think>This is reasoning</think><answer>4.5</answer>Extra text"}],
        ]
        
        rewards = format_reward(completions)
        
        # Extra content after tags should return 0.0
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_format_reward_multiline_content(self):
        """Test format checking with multiline content"""
        completions = [
            [{"role": "assistant", "content": "<think>Line 1\nLine 2\nLine 3</think><answer>4.5</answer>"}],
            [{"role": "assistant", "content": "<think>Reasoning</think>\n<answer>3.2</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # First has correct format, second has newline between tags
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)
        # Second should also match because \s* allows whitespace characters
        self.assertEqual(rewards[1], 1.0)
    
    def test_format_reward_empty_tags(self):
        """Test case with empty tags"""
        completions = [
            [{"role": "assistant", "content": "<think></think><answer></answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # Empty tags should be accepted (as long as format is correct)
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_format_reward_nested_tags(self):
        """Test case with nested tags"""
        completions = [
            [{"role": "assistant", "content": "<think><inner>nested</inner></think><answer>4.5</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # Nested tags should be accepted
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_format_reward_special_characters(self):
        """Test handling of special characters"""
        completions = [
            [{"role": "assistant", "content": "<think>Reasoning with $math$ and symbols!</think><answer>4.5</answer>"}],
            [{"role": "assistant", "content": "<think>中文推理过程</think><answer>3.5</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # Special characters and Chinese should be handled correctly
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 1.0)
    
    def test_format_reward_batch_processing(self):
        """Test batch processing"""
        completions = [
            [{"role": "assistant", "content": "<think>Reasoning 1</think><answer>4.5</answer>"}],
            [{"role": "assistant", "content": "Invalid format"}],
            [{"role": "assistant", "content": "<think>Reasoning 2</think><answer>3.2</answer>"}],
            [{"role": "assistant", "content": "<answer>2.1</answer>"}],
            [{"role": "assistant", "content": "<think>Reasoning 3</think><answer>5.0</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # Verify batch processing results
        self.assertEqual(len(rewards), 5)
        self.assertEqual(rewards[0], 1.0)  # Correct
        self.assertEqual(rewards[1], 0.0)  # Invalid
        self.assertEqual(rewards[2], 1.0)  # Correct
        self.assertEqual(rewards[3], 0.0)  # Missing think
        self.assertEqual(rewards[4], 1.0)  # Correct
    
    def test_format_reward_whitespace_variations(self):
        """Test handling of different whitespace characters"""
        completions = [
            [{"role": "assistant", "content": "<think>Reasoning</think><answer>4.5</answer>"}],
            [{"role": "assistant", "content": "<think>Reasoning</think> <answer>4.5</answer>"}],
            [{"role": "assistant", "content": "<think>Reasoning</think>  <answer>4.5</answer>"}],
            [{"role": "assistant", "content": "<think>Reasoning</think>\n<answer>4.5</answer>"}],
            [{"role": "assistant", "content": "<think>Reasoning</think>\t<answer>4.5</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # All variations should be accepted (\s* matches any whitespace)
        self.assertEqual(len(rewards), 5)
        for reward in rewards:
            self.assertEqual(reward, 1.0)


class TestFormatRewardIntegration(unittest.TestCase):
    """Integration tests: test format_reward in real-world scenarios"""
    
    def test_combined_with_accuracy_reward(self):
        """Test scenario combined with accuracy_reward"""
        # Simulate format checking in GRPO training
        completions = [
            [{"role": "assistant", "content": "<think>Step 1: Analyze\nStep 2: Calculate</think><answer>4.5</answer>"}],
            [{"role": "assistant", "content": "<think>Quick reasoning</think><answer>3.2</answer>"}],
            [{"role": "assistant", "content": "4.1"}],  # No format
            [{"role": "assistant", "content": "<think>Detailed analysis</think><answer>2.8</answer>"}],
        ]
        
        format_rewards = format_reward(completions)
        
        # Verify format rewards
        self.assertEqual(len(format_rewards), 4)
        self.assertEqual(format_rewards[0], 1.0)
        self.assertEqual(format_rewards[1], 1.0)
        self.assertEqual(format_rewards[2], 0.0)
        self.assertEqual(format_rewards[3], 1.0)
        
        # Calculate total rewards (format reward + accuracy reward)
        # In practice, responses without format get 0 format reward
        total_rewards = [fr for fr in format_rewards]
        self.assertEqual(sum(total_rewards), 3.0)
    
    def test_edge_case_empty_completion(self):
        """Test case with empty response"""
        completions = [
            [{"role": "assistant", "content": ""}],
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_multiple_think_answer_pairs(self):
        """Test case with multiple think-answer pairs"""
        completions = [
            [{"role": "assistant", "content": "<think>First</think><answer>1</answer><think>Second</think><answer>2</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # Multiple pairs don't meet format requirements (fullmatch requires exact match)
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)


if __name__ == "__main__":
    unittest.main()
