import unittest
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'r1-v', 'src'))

# Import the actual function we want to test
from open_r1.grpo import plan_format_reward


class TestPlanFormatReward(unittest.TestCase):
    """Test plan_format_reward function - format validation reward described in AlphaDrive documentation section 3"""
    
    def test_plan_format_reward_correct_format(self):
        """Test response with correct format"""
        completions = [
            [{"content": "<think>This is reasoning</think><answer>ACCELERATE</answer>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # Correct format should return 1.0
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_plan_format_reward_correct_format_with_whitespace(self):
        """Test correct format with whitespace"""
        completions = [
            [{"content": "<think>reasoning</think>  \n  <answer>result</answer>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # Whitespace between tags should be accepted
        self.assertEqual(rewards[0], 1.0)
    
    def test_plan_format_reward_correct_format_multiline(self):
        """Test correct format with multiline content"""
        completions = [
            [{"content": "<think>First line\nSecond line\nThird line</think><answer>Final answer\nwith newline</answer>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # Multiline content within tags should be accepted
        self.assertEqual(rewards[0], 1.0)
    
    def test_plan_format_reward_missing_think_tag(self):
        """Test case with missing think tag"""
        completions = [
            [{"content": "<answer>ACCELERATE</answer>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # Missing think tag should return 0.0
        self.assertEqual(rewards[0], 0.0)
    
    def test_plan_format_reward_missing_answer_tag(self):
        """Test case with missing answer tag"""
        completions = [
            [{"content": "<think>reasoning</think>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # Missing answer tag should return 0.0
        self.assertEqual(rewards[0], 0.0)
    
    def test_plan_format_reward_wrong_order(self):
        """Test case with wrong tag order"""
        completions = [
            [{"content": "<answer>result</answer><think>reasoning</think>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # Wrong tag order should return 0.0
        self.assertEqual(rewards[0], 0.0)
    
    def test_plan_format_reward_extra_content_before(self):
        """Test case with extra content before tags"""
        completions = [
            [{"content": "Extra text <think>reasoning</think><answer>result</answer>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # Extra content before tags should return 0.0
        self.assertEqual(rewards[0], 0.0)
    
    def test_plan_format_reward_extra_content_after(self):
        """Test case with extra content after tags"""
        completions = [
            [{"content": "<think>reasoning</think><answer>result</answer> Extra text"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # Extra content after tags should return 0.0
        self.assertEqual(rewards[0], 0.0)
    
    def test_plan_format_reward_incomplete_tags(self):
        """Test case with incomplete tags"""
        test_cases = [
            [{"content": "<think>reasoning</think><answer>result"}],  # Missing </answer>
            [{"content": "<think>reasoning<answer>result</answer>"}],  # Missing </think>
            [{"content": "think>reasoning</think><answer>result</answer>"}],  # Missing <think
            [{"content": "<think>reasoning</think>answer>result</answer>"}],  # Missing <answer
        ]
        
        for completion in test_cases:
            rewards = plan_format_reward([completion])
            self.assertEqual(rewards[0], 0.0)
    
    def test_plan_format_reward_empty_tags(self):
        """Test case with empty tags"""
        completions = [
            [{"content": "<think></think><answer></answer>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # Empty tags should also be accepted (as long as format is correct)
        self.assertEqual(rewards[0], 1.0)
    
    def test_plan_format_reward_multiple_completions(self):
        """Test batch processing of multiple completions"""
        completions = [
            [{"content": "<think>reasoning1</think><answer>answer1</answer>"}],  # Correct
            [{"content": "<answer>answer2</answer>"}],  # Wrong: missing think
            [{"content": "<think>reasoning3</think><answer>answer3</answer>"}],  # Correct
            [{"content": "wrong format"}],  # Wrong: completely wrong
        ]
        
        rewards = plan_format_reward(completions)
        
        # Verify correct number of rewards returned
        self.assertEqual(len(rewards), 4)
        
        # Verify correctness of each reward
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 0.0)
        self.assertEqual(rewards[2], 1.0)
        self.assertEqual(rewards[3], 0.0)
    
    def test_plan_format_reward_special_characters(self):
        """Test content with special characters"""
        completions = [
            [{"content": "<think>reasoning with $pecial ch@rs!</think><answer>result with #symbols</answer>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # Special characters should not affect format validation
        self.assertEqual(rewards[0], 1.0)
    
    def test_plan_format_reward_nested_tags(self):
        """Test case with nested tags"""
        completions = [
            [{"content": "<think><inner>nested</inner></think><answer>result</answer>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # Nested tags should be accepted (as long as outer format is correct)
        self.assertEqual(rewards[0], 1.0)
    
    def test_plan_format_reward_unicode_content(self):
        """Test Unicode content"""
        completions = [
            [{"content": "<think>Reasoning process Chinese content</think><answer>Answer result</answer>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # Unicode content should be handled correctly
        self.assertEqual(rewards[0], 1.0)


class TestPlanFormatRewardIntegration(unittest.TestCase):
    """Integration test: test plan_format_reward performance in real scenarios"""
    
    def test_format_reward_with_real_content(self):
        """Test format validation with real content"""
        # Simulate real driving decision responses
        completions = [
            [{"content": "<think>Based on the traffic situation, I need to slow down and prepare to stop at the red light.</think><answer>DECELERATE, STOP</answer>"}],
            [{"content": "<think>The road ahead is clear, I can maintain current speed.</think><answer>KEEP</answer>"}],
            [{"content": "<think>Need to change lanes to the left for upcoming turn.</think><answer>LEFT_CHANGE, LEFT_TURN</answer>"}],
        ]
        
        rewards = plan_format_reward(completions)
        
        # All real format responses should pass validation
        self.assertEqual(len(rewards), 3)
        for reward in rewards:
            self.assertEqual(reward, 1.0)
    
    def test_format_reward_consistency(self):
        """Test consistency of format validation"""
        # Different content with same format should get same reward
        completions = [
            [{"content": "<think>A</think><answer>B</answer>"}],
            [{"content": "<think>X</think><answer>Y</answer>"}],
            [{"content": "<think>1</think><answer>2</answer>"}],
        ]
        
        rewards = plan_format_reward(completions)
        
        # All rewards should be the same
        self.assertTrue(all(r == 1.0 for r in rewards))
    
    def test_format_reward_edge_cases(self):
        """Test edge cases"""
        edge_cases = [
            # Minimum valid content
            ([[{"content": "<think>a</think><answer>b</answer>"}]], 1.0),
            # Very long content
            ([[{"content": f"<think>{'x' * 10000}</think><answer>{'y' * 10000}</answer>"}]], 1.0),
            # Tags with only spaces
            ([[{"content": "<think>   </think><answer>   </answer>"}]], 1.0),
            # Multiple whitespace separators
            ([[{"content": "<think>x</think>     \n\n\n     <answer>y</answer>"}]], 1.0),
        ]
        
        for completion, expected_reward in edge_cases:
            rewards = plan_format_reward(completion)
            self.assertEqual(rewards[0], expected_reward)


if __name__ == "__main__":
    unittest.main()
