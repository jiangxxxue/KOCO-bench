import unittest
import sys
import os

# Add the parent directory to Python path to import open_r1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train', 'src'))

# Import the actual function we want to test
from open_r1.grpo import format_reward


class TestFormatReward(unittest.TestCase):
    """Test format_reward function - format validation described in CANOE document section 3"""
    
    def test_format_reward_correct_format(self):
        """Test correct format cases"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking process</think><long_answer>detailed answer</long_answer><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>another thought</think><long_answer>another detailed answer</long_answer><answer>100</answer>"}],
        ]
        
        # Call the actual format_reward function
        rewards = format_reward(completions)
        
        # Verify output
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # Correct format
        self.assertEqual(rewards[1], 1.0)  # Correct format
    
    def test_format_reward_incorrect_format_missing_tags(self):
        """Test cases with missing tags"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking process</think><answer>42</answer>"}],  # Missing long_answer
            [{"role": "assistant", "content": "<long_answer>detailed answer</long_answer><answer>100</answer>"}],  # Missing think
            [{"role": "assistant", "content": "<think>thinking process</think><long_answer>detailed answer</long_answer>"}],  # Missing answer
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 0.0)  # Incorrect format
        self.assertEqual(rewards[1], 0.0)  # Incorrect format
        self.assertEqual(rewards[2], 0.0)  # Incorrect format
    
    def test_format_reward_incorrect_order(self):
        """Test cases with incorrect tag order"""
        completions = [
            [{"role": "assistant", "content": "<answer>42</answer><think>thinking process</think><long_answer>detailed answer</long_answer>"}],  # Wrong order
            [{"role": "assistant", "content": "<think>thinking process</think><answer>42</answer><long_answer>detailed answer</long_answer>"}],  # Wrong order
            [{"role": "assistant", "content": "<long_answer>detailed answer</long_answer><think>thinking process</think><answer>42</answer>"}],  # Wrong order
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 0.0)  # Wrong order
        self.assertEqual(rewards[1], 0.0)  # Wrong order
        self.assertEqual(rewards[2], 0.0)  # Wrong order
    
    def test_format_reward_with_whitespace(self):
        """Test cases with whitespace between tags"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking process</think>  <long_answer>detailed answer</long_answer>  <answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>thinking process</think>\n<long_answer>detailed answer</long_answer>\n<answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>thinking process</think>\t<long_answer>detailed answer</long_answer>\t<answer>42</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # Regular expression allows whitespace between tags
        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 1.0)  # Correct format
        self.assertEqual(rewards[1], 1.0)  # Correct format
        self.assertEqual(rewards[2], 1.0)  # Correct format
    
    def test_format_reward_multiline_content(self):
        """Test cases with multiline content in tags"""
        completions = [
            [{"role": "assistant", "content": 
              "<think>This is a\nmultiline\nthinking process</think>"
              "<long_answer>This is a\nmultiline\ndetailed answer</long_answer>"
              "<answer>42</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # re.DOTALL flag allows . to match newlines
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)  # Correct format
    
    def test_format_reward_empty_tags(self):
        """Test cases with empty tags"""
        completions = [
            [{"role": "assistant", "content": "<think></think><long_answer></long_answer><answer></answer>"}],
            [{"role": "assistant", "content": "<think>thinking</think><long_answer></long_answer><answer>42</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # Empty tags should also be accepted (as long as format is correct)
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # Correct format
        self.assertEqual(rewards[1], 1.0)  # Correct format
    
    def test_format_reward_extra_content_before(self):
        """Test cases with extra content before tags"""
        completions = [
            [{"role": "assistant", "content": "extra content<think>thinking process</think><long_answer>detailed answer</long_answer><answer>42</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # fullmatch requires exact match, extra content before will fail
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)  # Incorrect format
    
    def test_format_reward_extra_content_after(self):
        """Test cases with extra content after tags"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking process</think><long_answer>detailed answer</long_answer><answer>42</answer>extra content"}],
        ]
        
        rewards = format_reward(completions)
        
        # fullmatch requires exact match, extra content after will fail
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)  # Incorrect format
    
    def test_format_reward_extra_content_between(self):
        """Test cases with extra content between tags"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking process</think>extra content<long_answer>detailed answer</long_answer><answer>42</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # Only whitespace is allowed between tags, other content will cause failure
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)  # Incorrect format
    
    def test_format_reward_nested_tags(self):
        """Test cases with nested tags"""
        completions = [
            [{"role": "assistant", "content": "<think><inner>nested</inner>thinking process</think><long_answer>detailed answer</long_answer><answer>42</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # Nested tags should be accepted (.*? will match internal content)
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)  # Correct format
    
    def test_format_reward_duplicate_tags(self):
        """Test cases with duplicate tags"""
        completions = [
            [{"role": "assistant", "content": "<think>thought1</think><think>thought2</think><long_answer>detailed answer</long_answer><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>thinking process</think><long_answer>answer1</long_answer><long_answer>answer2</long_answer><answer>42</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # Note: Because the regex uses .*? (non-greedy matching), <think>.*?</think> can match content containing duplicate tags
        # For example: <think>thought1</think><think>thought2</think> will be matched by .*? because it's followed by <long_answer>
        # So duplicate tag cases will actually still match successfully
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # Will actually match successfully
        self.assertEqual(rewards[1], 1.0)  # Will actually match successfully
    
    def test_format_reward_case_sensitive_tags(self):
        """Test tag case sensitivity"""
        completions = [
            [{"role": "assistant", "content": "<Think>thinking process</Think><long_answer>detailed answer</long_answer><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>thinking process</think><Long_Answer>detailed answer</Long_Answer><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>thinking process</think><long_answer>detailed answer</long_answer><Answer>42</Answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # Tags are case-sensitive
        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 0.0)  # Wrong case
        self.assertEqual(rewards[1], 0.0)  # Wrong case
        self.assertEqual(rewards[2], 0.0)  # Wrong case
    
    def test_format_reward_batch_processing(self):
        """Test batch processing"""
        batch_size = 10
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>detailed answer</long_answer><answer>42</answer>"}]
            for _ in range(batch_size)
        ]
        
        rewards = format_reward(completions)
        
        # All formats should be correct
        self.assertEqual(len(rewards), batch_size)
        self.assertTrue(all(r == 1.0 for r in rewards))
    
    def test_format_reward_mixed_batch(self):
        """Test batch with mixed correct and incorrect formats"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>detailed answer</long_answer><answer>42</answer>"}],  # Correct
            [{"role": "assistant", "content": "<think>thinking</think><answer>42</answer>"}],  # Wrong: missing long_answer
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>detailed answer</long_answer><answer>42</answer>"}],  # Correct
            [{"role": "assistant", "content": "completely wrong format"}],  # Wrong
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>detailed answer</long_answer><answer>42</answer>"}],  # Correct
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 5)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 0.0)
        self.assertEqual(rewards[2], 1.0)
        self.assertEqual(rewards[3], 0.0)
        self.assertEqual(rewards[4], 1.0)
    
    def test_format_reward_special_characters_in_content(self):
        """Test cases with special characters in content"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking: x^2 + y^2 = r^2</think><long_answer>detailed answer: α + β = γ</long_answer><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>thinking & reasoning</think><long_answer>detailed answer < 100</long_answer><answer>42</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # Special characters in tag content should be accepted
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 1.0)
    
    def test_format_reward_very_long_content(self):
        """Test very long content"""
        long_content = "This is a very long content. " * 100
        completions = [
            [{"role": "assistant", "content": f"<think>{long_content}</think><long_answer>{long_content}</long_answer><answer>42</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)  # Correct format


class TestFormatRewardIntegration(unittest.TestCase):
    """Integration tests: test format_reward integration with real-world usage scenarios"""
    
    def test_realistic_generation_scenario(self):
        """Test realistic generation scenario"""
        completions = [
            [{"role": "assistant", "content": 
              "<think>First, I need to understand the problem. This is a simple math problem that asks to calculate 2+2. "
              "According to basic arithmetic rules, the result of adding two positive integers is their sum.</think>"
              "<long_answer>According to basic arithmetic, when we calculate 2 plus 2, we add the two values together. "
              "In the decimal system, 2+2 equals 4. This is a fundamental mathematical fact.</long_answer>"
              "<answer>4</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_quality_control_scenario(self):
        """Test quality control scenario (filtering incorrectly formatted generations)"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>detailed answer</long_answer><answer>42</answer>"}],  # Good generation
            [{"role": "assistant", "content": "Direct answer: 42"}],  # Bad generation
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>detailed answer</long_answer><answer>100</answer>"}],  # Good generation
            [{"role": "assistant", "content": "<answer>200</answer>"}],  # Bad generation
        ]
        
        rewards = format_reward(completions)
        
        # Can be used for filtering: only keep generations with reward=1.0
        good_indices = [i for i, r in enumerate(rewards) if r == 1.0]
        self.assertEqual(good_indices, [0, 2])


if __name__ == "__main__":
    unittest.main()
