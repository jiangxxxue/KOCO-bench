import unittest
import torch
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to Python path to import open_r1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train', 'src'))

# Import the actual function we want to test
from open_r1.grpo import accuracy_reward


class TestAccuracyReward(unittest.TestCase):
    """Test accuracy_reward function"""
    
    def setUp(self):
        # Clear DEBUG_MODE environment variable
        if "DEBUG_MODE" in os.environ:
            del os.environ["DEBUG_MODE"]
    
    def test_accuracy_reward_basic_string_matching(self):
        """Test basic string matching functionality"""
        # Create test data
        completions = [
            [{"role": "assistant", "content": "<think>thinking process</think><long_answer>detailed answer</long_answer><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>thinking process</think><long_answer>detailed answer</long_answer><answer>100</answer>"}],
        ]
        solution = ["<answer>42</answer>", "<answer>100</answer>"]
        
        # Call the actual accuracy_reward function
        rewards = accuracy_reward(completions, solution)
        
        # Verify output
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # Correct answer
        self.assertEqual(rewards[1], 1.0)  # Correct answer
    
    def test_accuracy_reward_incorrect_answer(self):
        """Test incorrect answer case"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking process</think><long_answer>detailed answer</long_answer><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>thinking process</think><long_answer>detailed answer</long_answer><answer>99</answer>"}],
        ]
        solution = ["<answer>42</answer>", "<answer>100</answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # First answer correct
        self.assertEqual(rewards[1], 0.0)  # Second answer incorrect
    
    def test_accuracy_reward_without_tags(self):
        """Test case without answer tags"""
        completions = [
            [{"role": "assistant", "content": "42"}],
            [{"role": "assistant", "content": "100"}],
        ]
        solution = ["42", "99"]
        
        rewards = accuracy_reward(completions, solution)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # Answer correct
        self.assertEqual(rewards[1], 0.0)  # Answer incorrect
    
    def test_accuracy_reward_with_whitespace(self):
        """Test answers with whitespace"""
        completions = [
            [{"role": "assistant", "content": "<answer>  42  </answer>"}],
            [{"role": "assistant", "content": "<answer>100</answer>"}],
        ]
        solution = ["<answer>42</answer>", "<answer>  100  </answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        # Due to strip() processing, whitespace should be ignored
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 1.0)
    
    def test_accuracy_reward_mixed_format(self):
        """Test mixed format (with and without tags)"""
        completions = [
            [{"role": "assistant", "content": "<answer>42</answer>"}],
            [{"role": "assistant", "content": "100"}],
        ]
        solution = ["42", "<answer>100</answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 1.0)
    
    def test_accuracy_reward_multiline_content(self):
        """Test multiline content"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking\nprocess</think><long_answer>detailed\nanswer</long_answer><answer>42</answer>"}],
        ]
        solution = ["<answer>42</answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_accuracy_reward_empty_answer(self):
        """Test empty answer case"""
        completions = [
            [{"role": "assistant", "content": "<answer></answer>"}],
            [{"role": "assistant", "content": ""}],
        ]
        solution = ["<answer></answer>", ""]
        
        rewards = accuracy_reward(completions, solution)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # Empty answer matches
        self.assertEqual(rewards[1], 1.0)  # Empty string matches
    
    def test_accuracy_reward_case_sensitive(self):
        """Test case sensitivity"""
        completions = [
            [{"role": "assistant", "content": "<answer>ABC</answer>"}],
            [{"role": "assistant", "content": "<answer>abc</answer>"}],
        ]
        solution = ["<answer>abc</answer>", "<answer>abc</answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        # String matching is case-sensitive
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 0.0)  # Case mismatch
        self.assertEqual(rewards[1], 1.0)  # Case match
    
    def test_accuracy_reward_numeric_answers(self):
        """Test numeric answers"""
        completions = [
            [{"role": "assistant", "content": "<answer>3.14</answer>"}],
            [{"role": "assistant", "content": "<answer>-5</answer>"}],
            [{"role": "assistant", "content": "<answer>0</answer>"}],
        ]
        solution = ["<answer>3.14</answer>", "<answer>-5</answer>", "<answer>1</answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 1.0)
        self.assertEqual(rewards[2], 0.0)
    
    def test_accuracy_reward_special_characters(self):
        """Test special characters"""
        completions = [
            [{"role": "assistant", "content": "<answer>x^2 + y^2 = r^2</answer>"}],
            [{"role": "assistant", "content": "<answer>α + β = γ</answer>"}],
        ]
        solution = ["<answer>x^2 + y^2 = r^2</answer>", "<answer>α + β = γ</answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 1.0)
    
    @patch('open_r1.grpo.parse')
    @patch('open_r1.grpo.verify')
    def test_accuracy_reward_symbolic_verification_success(self, mock_verify, mock_parse):
        """Test symbolic verification success case"""
        # Mock symbolic verification success
        mock_parse.return_value = MagicMock()
        mock_verify.return_value = 1.0  # Verification success
        
        completions = [
            [{"role": "assistant", "content": "<answer>2*x + 3</answer>"}],
        ]
        solution = ["<answer>3 + 2*x</answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        # Symbolic verification success, should return 1.0
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
        
        # Verify parse and verify were called
        self.assertEqual(mock_parse.call_count, 2)  # Parse generated answer and correct answer
        mock_verify.assert_called_once()
    
    @patch('open_r1.grpo.parse')
    @patch('open_r1.grpo.verify')
    def test_accuracy_reward_symbolic_verification_failure_fallback_to_string(self, mock_verify, mock_parse):
        """Test fallback to string matching after symbolic verification failure"""
        # Mock symbolic verification failure
        mock_parse.side_effect = Exception("Parse error")
        
        completions = [
            [{"role": "assistant", "content": "<answer>42</answer>"}],
        ]
        solution = ["<answer>42</answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        # Symbolic verification failed, but string matching succeeded
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_accuracy_reward_batch_processing(self):
        """Test batch processing"""
        # Create a larger batch
        batch_size = 10
        completions = [
            [{"role": "assistant", "content": f"<answer>{i}</answer>"}]
            for i in range(batch_size)
        ]
        solution = [f"<answer>{i}</answer>" for i in range(batch_size)]
        
        rewards = accuracy_reward(completions, solution)
        
        # All answers should be correct
        self.assertEqual(len(rewards), batch_size)
        self.assertTrue(all(r == 1.0 for r in rewards))
    
    def test_accuracy_reward_debug_mode_logging(self):
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
                [{"role": "assistant", "content": "<answer>42</answer>"}],
            ]
            solution = ["<answer>42</answer>"]
            
            rewards = accuracy_reward(completions, solution)
            
            # Verify log file is created and contains content
            self.assertTrue(os.path.exists(log_path))
            with open(log_path, 'r') as f:
                log_content = f.read()
                self.assertIn("Accuracy reward", log_content)
                self.assertIn("Content:", log_content)
                self.assertIn("Solution:", log_content)
        finally:
            # Cleanup
            if os.path.exists(log_path):
                os.remove(log_path)
            del os.environ["DEBUG_MODE"]
            del os.environ["LOG_PATH"]
    
    def test_accuracy_reward_edge_cases(self):
        """Test edge cases"""
        # Test case 1: Only one sample
        completions_single = [
            [{"role": "assistant", "content": "<answer>42</answer>"}],
        ]
        solution_single = ["<answer>42</answer>"]
        
        rewards = accuracy_reward(completions_single, solution_single)
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
        
        # Test case 2: Answer contains answer tag but format is incomplete
        completions_incomplete = [
            [{"role": "assistant", "content": "<answer>42"}],  # Missing closing tag
        ]
        solution_incomplete = ["<answer>42</answer>"]
        
        rewards = accuracy_reward(completions_incomplete, solution_incomplete)
        self.assertEqual(len(rewards), 1)
        # Since there's no complete tag, the entire string will be used for matching
        self.assertEqual(rewards[0], 0.0)


class TestAccuracyRewardIntegration(unittest.TestCase):
    """Integration tests: test accuracy_reward integration with real-world usage scenarios"""
    
    def test_realistic_math_problem(self):
        """Test realistic math problem scenario"""
        completions = [
            [{"role": "assistant", "content": 
              "<think>This is a simple addition problem. 2 + 2 = 4</think>"
              "<long_answer>According to basic arithmetic, 2 plus 2 equals 4.</long_answer>"
              "<answer>4</answer>"}],
        ]
        solution = ["<answer>4</answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_multiple_samples_mixed_results(self):
        """Test multiple samples with mixed results"""
        completions = [
            [{"role": "assistant", "content": "<think>...</think><long_answer>...</long_answer><answer>correct answer</answer>"}],
            [{"role": "assistant", "content": "<think>...</think><long_answer>...</long_answer><answer>wrong answer</answer>"}],
            [{"role": "assistant", "content": "<think>...</think><long_answer>...</long_answer><answer>another correct answer</answer>"}],
        ]
        solution = ["<answer>correct answer</answer>", "<answer>correct answer</answer>", "<answer>another correct answer</answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 1.0)  # Correct
        self.assertEqual(rewards[1], 0.0)  # Incorrect
        self.assertEqual(rewards[2], 1.0)  # Correct


if __name__ == "__main__":
    unittest.main()
