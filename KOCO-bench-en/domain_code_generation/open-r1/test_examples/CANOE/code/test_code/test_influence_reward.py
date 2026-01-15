import unittest
import sys
import os
from unittest.mock import patch

# Add the parent directory to Python path to import open_r1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train', 'src'))

# Import the actual function we want to test
from open_r1.grpo import influence_reward


class TestInfluenceReward(unittest.TestCase):
    """Test influence_reward function"""
    
    def setUp(self):
        # Clear DEBUG_MODE environment variable
        if "DEBUG_MODE" in os.environ:
            del os.environ["DEBUG_MODE"]
    
    def test_influence_reward_basic_correct(self):
        """Test basic correct influence verification"""
        completions = [
            [{"role": "assistant", "content": "original generated content"}],
            [{"role": "assistant", "content": "original generated content"}],
        ]
        completions_long_answer = [
            [{"role": "assistant", "content": "<answer>42</answer>"}],
            [{"role": "assistant", "content": "<answer>100</answer>"}],
        ]
        solution = ["<answer>42</answer>", "<answer>100</answer>"]
        
        # Call the actual influence_reward function
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        # Verify output
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # Answer generated from long answer is correct
        self.assertEqual(rewards[1], 1.0)  # Answer generated from long answer is correct
    
    def test_influence_reward_incorrect_answer(self):
        """Test incorrect answer case"""
        completions = [
            [{"role": "assistant", "content": "original generated content"}],
            [{"role": "assistant", "content": "original generated content"}],
        ]
        completions_long_answer = [
            [{"role": "assistant", "content": "<answer>42</answer>"}],
            [{"role": "assistant", "content": "<answer>99</answer>"}],  # Incorrect answer
        ]
        solution = ["<answer>42</answer>", "<answer>100</answer>"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # Correct
        self.assertEqual(rewards[1], 0.0)  # Incorrect
    
    def test_influence_reward_none_completion(self):
        """Test case where completions_long_answer is None"""
        completions = [
            [{"role": "assistant", "content": "original generated content"}],
            [{"role": "assistant", "content": "original generated content"}],
        ]
        completions_long_answer = [
            [{"role": "assistant", "content": "<answer>42</answer>"}],
            None,  # No long answer generated
        ]
        solution = ["<answer>42</answer>", "<answer>100</answer>"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # Correct
        self.assertEqual(rewards[1], 0.0)  # None case returns 0.0
    
    def test_influence_reward_all_none(self):
        """Test case where all completions_long_answer are None"""
        completions = [
            [{"role": "assistant", "content": "original generated content"}],
            [{"role": "assistant", "content": "original generated content"}],
        ]
        completions_long_answer = [None, None]
        solution = ["<answer>42</answer>", "<answer>100</answer>"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 0.0)
        self.assertEqual(rewards[1], 0.0)
    
    def test_influence_reward_without_tags(self):
        """Test case without answer tags"""
        completions = [
            [{"role": "assistant", "content": "original generated content"}],
        ]
        completions_long_answer = [
            [{"role": "assistant", "content": "42"}],
        ]
        solution = ["42"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_influence_reward_with_whitespace(self):
        """Test answers with whitespace"""
        completions = [
            [{"role": "assistant", "content": "original generated content"}],
            [{"role": "assistant", "content": "original generated content"}],
        ]
        completions_long_answer = [
            [{"role": "assistant", "content": "<answer>  42  </answer>"}],
            [{"role": "assistant", "content": "<answer>100</answer>"}],
        ]
        solution = ["<answer>42</answer>", "<answer>  100  </answer>"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        # Due to strip() processing, whitespace should be ignored
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 1.0)
    
    def test_influence_reward_mixed_none_and_valid(self):
        """Test mixed None and valid values"""
        completions = [
            [{"role": "assistant", "content": "original generated content"}],
            [{"role": "assistant", "content": "original generated content"}],
            [{"role": "assistant", "content": "original generated content"}],
            [{"role": "assistant", "content": "original generated content"}],
        ]
        completions_long_answer = [
            [{"role": "assistant", "content": "<answer>42</answer>"}],
            None,
            [{"role": "assistant", "content": "<answer>100</answer>"}],
            None,
        ]
        solution = ["<answer>42</answer>", "<answer>50</answer>", "<answer>100</answer>", "<answer>200</answer>"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        self.assertEqual(len(rewards), 4)
        self.assertEqual(rewards[0], 1.0)  # Correct
        self.assertEqual(rewards[1], 0.0)  # None
        self.assertEqual(rewards[2], 1.0)  # Correct
        self.assertEqual(rewards[3], 0.0)  # None
    
    def test_influence_reward_multiline_content(self):
        """Test multiline content"""
        completions = [
            [{"role": "assistant", "content": "original generated content"}],
        ]
        completions_long_answer = [
            [{"role": "assistant", "content": "<think>thinking\nprocess</think><long_answer>detailed\nanswer</long_answer><answer>42</answer>"}],
        ]
        solution = ["<answer>42</answer>"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_influence_reward_empty_answer(self):
        """Test empty answer case"""
        completions = [
            [{"role": "assistant", "content": "original generated content"}],
            [{"role": "assistant", "content": "original generated content"}],
        ]
        completions_long_answer = [
            [{"role": "assistant", "content": "<answer></answer>"}],
            [{"role": "assistant", "content": ""}],
        ]
        solution = ["<answer></answer>", ""]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # Empty answer matches
        self.assertEqual(rewards[1], 1.0)  # Empty string matches
    
    def test_influence_reward_case_sensitive(self):
        """Test case sensitivity"""
        completions = [
            [{"role": "assistant", "content": "original generated content"}],
            [{"role": "assistant", "content": "original generated content"}],
        ]
        completions_long_answer = [
            [{"role": "assistant", "content": "<answer>ABC</answer>"}],
            [{"role": "assistant", "content": "<answer>abc</answer>"}],
        ]
        solution = ["<answer>abc</answer>", "<answer>abc</answer>"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        # String matching is case-sensitive
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 0.0)  # Case mismatch
        self.assertEqual(rewards[1], 1.0)  # Case match
    
    def test_influence_reward_batch_processing(self):
        """Test batch processing"""
        batch_size = 10
        completions = [
            [{"role": "assistant", "content": "original generated content"}]
            for _ in range(batch_size)
        ]
        completions_long_answer = [
            [{"role": "assistant", "content": f"<answer>{i}</answer>"}]
            for i in range(batch_size)
        ]
        solution = [f"<answer>{i}</answer>" for i in range(batch_size)]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        # All answers should be correct
        self.assertEqual(len(rewards), batch_size)
        self.assertTrue(all(r == 1.0 for r in rewards))
    
    def test_influence_reward_debug_mode_logging(self):
        """Test logging in DEBUG_MODE"""
        # Set DEBUG_MODE environment variable
        os.environ["DEBUG_MODE"] = "true"
        
        # Create temporary log file path
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_path = f.name
        
        os.environ["LOG_PATH"] = log_path
        
        try:
            completions = [
                [{"role": "assistant", "content": "original generated content"}],
            ]
            completions_long_answer = [
                [{"role": "assistant", "content": "<answer>42</answer>"}],
            ]
            solution = ["<answer>42</answer>"]
            
            rewards = influence_reward(completions, solution, completions_long_answer)
            
            # Verify log file is created and contains content
            self.assertTrue(os.path.exists(log_path))
            with open(log_path, 'r') as f:
                log_content = f.read()
                self.assertIn("influence_Accuracy reward", log_content)
                self.assertIn("Content:", log_content)
                self.assertIn("Solution:", log_content)
        finally:
            # Cleanup
            if os.path.exists(log_path):
                os.remove(log_path)
            del os.environ["DEBUG_MODE"]
            del os.environ["LOG_PATH"]
    
    def test_influence_reward_exception_handling(self):
        """Test exception handling"""
        completions = [
            [{"role": "assistant", "content": "original generated content"}],
        ]
        # Create a situation that might cause an exception
        completions_long_answer = [
            [{"role": "assistant", "content": "<answer>42</answer>"}],
        ]
        # Use a solution that might cause a regex exception
        solution = ["<answer>42</answer>"]
        
        # Should not raise an exception
        try:
            rewards = influence_reward(completions, solution, completions_long_answer)
            self.assertEqual(len(rewards), 1)
        except Exception as e:
            self.fail(f"influence_reward raised an exception: {e}")


class TestInfluenceRewardIntegration(unittest.TestCase):
    """Integration tests: test influence_reward integration with real-world usage scenarios"""
    
    def test_realistic_dual_grpo_scenario(self):
        """Test realistic Dual-GRPO scenario"""
        # Simulate first stage generation
        completions = [
            [{"role": "assistant", "content": 
              "<think>thinking process</think>"
              "<long_answer>This is a detailed long answer used to replace context.</long_answer>"
              "<answer>42</answer>"}],
        ]
        
        # Simulate second stage generation (regenerate after replacing context with long answer)
        completions_long_answer = [
            [{"role": "assistant", "content": 
              "<think>thinking based on long answer</think>"
              "<long_answer>detailed answer based on long answer</long_answer>"
              "<answer>42</answer>"}],  # Answer is still correct
        ]
        
        solution = ["<answer>42</answer>"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)  # Long answer has positive influence
    
    def test_negative_influence_scenario(self):
        """Test negative influence scenario (long answer leads to incorrect answer)"""
        completions = [
            [{"role": "assistant", "content": 
              "<think>thinking process</think>"
              "<long_answer>This is a potentially misleading long answer.</long_answer>"
              "<answer>42</answer>"}],
        ]
        
        # Long answer leads to incorrect final answer
        completions_long_answer = [
            [{"role": "assistant", "content": 
              "<think>thinking based on misleading long answer</think>"
              "<long_answer>incorrect detailed answer</long_answer>"
              "<answer>99</answer>"}],  # Answer became wrong
        ]
        
        solution = ["<answer>42</answer>"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)  # Long answer has negative influence
    
    def test_mixed_influence_batch(self):
        """Test batch with mixed influences"""
        completions = [
            [{"role": "assistant", "content": "<think>...</think><long_answer>good long answer</long_answer><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>...</think><long_answer>misleading long answer</long_answer><answer>100</answer>"}],
            [{"role": "assistant", "content": "<think>...</think><long_answer>invalid long answer</long_answer><answer>200</answer>"}],
        ]
        
        completions_long_answer = [
            [{"role": "assistant", "content": "<answer>42</answer>"}],   # Correct
            [{"role": "assistant", "content": "<answer>99</answer>"}],   # Incorrect
            None,  # No generation
        ]
        
        solution = ["<answer>42</answer>", "<answer>100</answer>", "<answer>200</answer>"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 1.0)  # Positive influence
        self.assertEqual(rewards[1], 0.0)  # Negative influence
        self.assertEqual(rewards[2], 0.0)  # No influence


if __name__ == "__main__":
    unittest.main()
