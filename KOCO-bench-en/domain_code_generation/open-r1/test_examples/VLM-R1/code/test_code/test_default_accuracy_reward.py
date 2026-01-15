import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to Python path to import open_r1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))

# Import the actual function we want to test
from open_r1.grpo_jsonl import default_accuracy_reward


class TestDefaultAccuracyReward(unittest.TestCase):
    """Test default_accuracy_reward function - default accuracy reward function as described in VLM-R1 documentation"""
    
    def test_exact_string_match(self):
        """Test exact string matching"""
        content = "<think>thinking process</think><answer>42</answer>"
        sol = "<answer>42</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)
    
    def test_exact_string_match_without_tags(self):
        """Test exact matching without tags"""
        content = "42"
        sol = "42"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)
    
    def test_incorrect_answer(self):
        """Test incorrect answer"""
        content = "<answer>42</answer>"
        sol = "<answer>100</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 0.0)
    
    def test_numeric_answer_matching(self):
        """Test numeric answer matching"""
        content = "<answer>3.14</answer>"
        sol = "<answer>3.14</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)
    
    def test_numeric_answer_mismatch(self):
        """Test numeric answer mismatch"""
        content = "<answer>3.14</answer>"
        sol = "<answer>2.71</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        # Numeric mismatch, but will use fuzzy matching with low similarity
        self.assertLess(reward, 0.5)
    
    def test_multiple_choice_correct(self):
        """Test multiple choice correct answer"""
        content = "<answer>A</answer>"
        sol = "<answer>A</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)
    
    def test_multiple_choice_incorrect(self):
        """Test multiple choice incorrect answer"""
        content = "<answer>B</answer>"
        sol = "<answer>A</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 0.0)
    
    def test_multiple_choice_with_text(self):
        """Test multiple choice with text"""
        content = "<answer>The answer is A</answer>"
        sol = "<answer>A</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        # Should be able to extract option A and match
        self.assertEqual(reward, 1.0)
    
    def test_text_answer_fuzzy_match(self):
        """Test text answer fuzzy matching"""
        content = "<answer>The capital of France is Paris</answer>"
        sol = "<answer>Paris</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        # print(reward)

        expected_reward = 0.2857142857142857
        self.assertAlmostEqual(reward, expected_reward, places=4)
        # Fuzzy matching should give some similarity score
        self.assertGreater(reward, 0.0)
        self.assertLessEqual(reward, 1.0)
    
    def test_text_answer_exact_fuzzy_match(self):
        """Test exact same text answer"""
        content = "<answer>Paris</answer>"
        sol = "<answer>Paris</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)
    
    def test_whitespace_handling(self):
        """Test whitespace handling"""
        content = "<answer>  42  </answer>"
        sol = "<answer>42</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        # strip() should remove leading and trailing whitespace
        self.assertEqual(reward, 1.0)
    
    def test_multiple_answer_tags(self):
        """Test multiple answer tags (use the last one)"""
        content = "<answer>wrong</answer><think>more thinking</think><answer>42</answer>"
        sol = "<answer>42</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        # Should use the last answer tag
        self.assertEqual(reward, 1.0)
    
    def test_case_sensitivity(self):
        """Test case sensitivity"""
        content = "<answer>ABC</answer>"
        sol = "<answer>abc</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        # clean_text converts to lowercase, so should match
        self.assertEqual(reward, 1.0)
    
    @patch('open_r1.grpo_jsonl.parse')
    @patch('open_r1.grpo_jsonl.verify')
    def test_symbolic_verification_success(self, mock_verify, mock_parse):
        """Test symbolic verification success"""
        mock_parse.return_value = MagicMock()
        mock_verify.return_value = 1.0
        
        content = "<answer>2*x + 3</answer>"
        sol = "<answer>3 + 2*x</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)
        self.assertEqual(mock_parse.call_count, 2)
        mock_verify.assert_called_once()
    
    @patch('open_r1.grpo_jsonl.parse')
    def test_symbolic_verification_failure_fallback(self, mock_parse):
        """Test symbolic verification failure fallback to string matching"""
        mock_parse.side_effect = Exception("Parse error")
        
        content = "<answer>42</answer>"
        sol = "<answer>42</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        # Symbolic verification failed, but string matching succeeded
        self.assertEqual(reward, 1.0)
    
    def test_empty_answer(self):
        """Test empty answer"""
        content = "<answer></answer>"
        sol = "<answer></answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)
    
    def test_special_characters(self):
        """Test special characters"""
        content = "<answer>x^2 + y^2 = r^2</answer>"
        sol = "<answer>x^2 + y^2 = r^2</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)
    
    def test_numeric_with_units(self):
        """Test numeric with units"""
        content = "<answer>42 meters</answer>"
        sol = "<answer>42</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        expected_reward = 1.0
        self.assertAlmostEqual(reward, expected_reward, places=4)
    
    def test_multiline_content(self):
        """Test multiline content"""
        content = "<think>thinking\nprocess</think><answer>42</answer>"
        sol = "<answer>42</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)


class TestDefaultAccuracyRewardIntegration(unittest.TestCase):
    """Integration tests: test default_accuracy_reward performance in real scenarios"""
    
    def test_math_problem_scenario(self):
        """Test math problem scenario"""
        content = "<think>2 + 2 = 4</think><answer>4</answer>"
        sol = "<answer>4</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)
    
    def test_multiple_choice_scenario(self):
        """Test multiple choice scenario"""
        # Since extract_choice function is not defined, will fallback to fuzzy matching
        content = "<think>analyzing options...</think><answer>A</answer>"
        sol = "<answer>A</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        # Exact match should return 1.0
        self.assertEqual(reward, 1.0)
    
    def test_text_answer_scenario(self):
        """Test text answer scenario"""
        content = "<think>thinking...</think><answer>Paris</answer>"
        sol = "<answer>Paris</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)
    
    def test_mixed_verification_methods(self):
        """Test mixed verification methods"""
        # Test numeric answer
        reward1 = default_accuracy_reward("<answer>3.14</answer>", "<answer>3.14</answer>")
        self.assertEqual(reward1, 1.0)
        
        # Test multiple choice
        reward2 = default_accuracy_reward("<answer>B</answer>", "<answer>B</answer>")
        self.assertEqual(reward2, 1.0)
        
        # Test text answer
        reward3 = default_accuracy_reward("<answer>Paris</answer>", "<answer>Paris</answer>")
        self.assertEqual(reward3, 1.0)


if __name__ == "__main__":
    unittest.main()
