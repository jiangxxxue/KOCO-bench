import unittest
import sys
import os

# Add the parent directory to Python path to import open_r1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train', 'src'))

# Import the actual function we want to test
from open_r1.grpo import len_reward


class TestLenReward(unittest.TestCase):
    """Test len_reward function - long answer length constraint described in CANOE document section 4"""
    
    def test_len_reward_correct_length(self):
        """Test case with length in correct range"""
        # context length is 100, long_answer length is 50 (within 20%-80% range)
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "a" * 50 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "x" * 100 + "</context>\n\nQuestion content"]
        
        # Call the actual len_reward function
        rewards = len_reward(completions, problem=problem)
        
        # Verify output
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)  # Length within range
    
    def test_len_reward_too_short(self):
        """Test case where long answer is too short"""
        # context length is 100, long_answer length is 10 (less than 20%)
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "a" * 10 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "x" * 100 + "</context>\n\nQuestion content"]
        
        rewards = len_reward(completions, problem=problem)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)  # Length too short
    
    def test_len_reward_too_long(self):
        """Test case where long answer is too long"""
        # context length is 100, long_answer length is 90 (greater than 80%)
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "a" * 90 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "x" * 100 + "</context>\n\nQuestion content"]
        
        rewards = len_reward(completions, problem=problem)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)  # Length too long
    
    def test_len_reward_boundary_min(self):
        """Test minimum boundary case"""
        # context length is 100, long_answer length is 21 (just above 20%)
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "a" * 21 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "x" * 100 + "</context>\n\nQuestion content"]
        
        rewards = len_reward(completions, problem=problem)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)  # Just within range
    
    def test_len_reward_boundary_max(self):
        """Test maximum boundary case"""
        # context length is 100, long_answer length is 79 (just below 80%)
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "a" * 79 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "x" * 100 + "</context>\n\nQuestion content"]
        
        rewards = len_reward(completions, problem=problem)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)  # Just within range
    
    def test_len_reward_exact_min_boundary(self):
        """Test case exactly at minimum boundary"""
        # context length is 100, long_answer length is 20 (exactly 20%)
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "a" * 20 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "x" * 100 + "</context>\n\nQuestion content"]
        
        rewards = len_reward(completions, problem=problem)
        
        # Since it's strictly greater than, exactly 20% should not satisfy
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_len_reward_exact_max_boundary(self):
        """Test case exactly at maximum boundary"""
        # context length is 100, long_answer length is 80 (exactly 80%)
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "a" * 80 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "x" * 100 + "</context>\n\nQuestion content"]
        
        rewards = len_reward(completions, problem=problem)
        
        # Since it's strictly less than, exactly 80% should not satisfy
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_len_reward_missing_context_tag(self):
        """Test case with missing context tag"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "a" * 50 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["Question without context tag"]
        
        rewards = len_reward(completions, problem=problem)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)  # Cannot extract context, return 0
    
    def test_len_reward_missing_long_answer_tag(self):
        """Test case with missing long_answer tag"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><answer>42</answer>"}],  # Missing long_answer
        ]
        problem = ["<context>" + "x" * 100 + "</context>\n\nQuestion content"]
        
        rewards = len_reward(completions, problem=problem)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)  # Cannot extract long_answer, return 0
    
    def test_len_reward_empty_context(self):
        """Test case with empty context"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "a" * 50 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context></context>\n\nQuestion content"]
        
        rewards = len_reward(completions, problem=problem)
        
        # Empty context has length 0, any length long_answer will exceed 80%
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_len_reward_empty_long_answer(self):
        """Test case with empty long_answer"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><long_answer></long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "x" * 100 + "</context>\n\nQuestion content"]
        
        rewards = len_reward(completions, problem=problem)
        
        # Empty long_answer has length 0, less than 20%
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_len_reward_whitespace_handling(self):
        """Test whitespace handling"""
        # Both context and long_answer contain whitespace
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>  " + "a" * 50 + "  </long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>  " + "x" * 100 + "  </context>\n\nQuestion content"]
        
        rewards = len_reward(completions, problem=problem)
        
        # strip() removes leading/trailing whitespace, but doesn't affect length calculation (length calculated after strip)
        self.assertEqual(len(rewards), 1)
        # 50 is within (0.2*100, 0.8*100) range
        self.assertEqual(rewards[0], 1.0)
    
    def test_len_reward_multiline_content(self):
        """Test multiline content"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "a\n" * 25 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "x\n" * 50 + "</context>\n\nQuestion content"]
        
        rewards = len_reward(completions, problem=problem)
        
        # Note: re.search in actual code doesn't use re.DOTALL flag
        # This means .*? cannot match newlines, so content with newlines cannot be matched by regex
        # Therefore returns 0 instead of 1
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)  # Cannot match multiline content
    
    def test_len_reward_batch_processing(self):
        """Test batch processing"""
        batch_size = 5
        completions = [
            [{"role": "assistant", "content": f"<think>thinking</think><long_answer>{'a' * (30 + i * 10)}</long_answer><answer>42</answer>"}]
            for i in range(batch_size)
        ]
        problem = [f"<context>{'x' * 100}</context>\n\nQuestion {i}" for i in range(batch_size)]
        
        rewards = len_reward(completions, problem=problem)
        
        # Verify batch processing
        self.assertEqual(len(rewards), batch_size)
        # 30, 40, 50, 60, 70 are all within (20, 80) range
        self.assertTrue(all(r == 1.0 for r in rewards))
    
    def test_len_reward_mixed_batch(self):
        """Test batch with mixed results"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "a" * 50 + "</long_answer><answer>42</answer>"}],  # Correct
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "a" * 10 + "</long_answer><answer>42</answer>"}],  # Too short
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "a" * 90 + "</long_answer><answer>42</answer>"}],  # Too long
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "a" * 30 + "</long_answer><answer>42</answer>"}],  # Correct
        ]
        problem = ["<context>" + "x" * 100 + "</context>\n\nQuestion"] * 4
        
        rewards = len_reward(completions, problem=problem)
        
        self.assertEqual(len(rewards), 4)
        self.assertEqual(rewards[0], 1.0)  # Correct
        self.assertEqual(rewards[1], 0.0)  # Too short
        self.assertEqual(rewards[2], 0.0)  # Too long
        self.assertEqual(rewards[3], 1.0)  # Correct
    
    def test_len_reward_different_context_lengths(self):
        """Test cases with different context lengths"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "a" * 25 + "</long_answer><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "a" * 50 + "</long_answer><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "a" * 100 + "</long_answer><answer>42</answer>"}],
        ]
        problem = [
            "<context>" + "x" * 100 + "</context>\n\nQuestion 1",  # context length 100, long_answer 25 within (20, 80)
            "<context>" + "x" * 100 + "</context>\n\nQuestion 2",  # context length 100, long_answer 50 within (20, 80)
            "<context>" + "x" * 200 + "</context>\n\nQuestion 3",  # context length 200, long_answer 100 within (40, 160)
        ]
        
        rewards = len_reward(completions, problem=problem)
        
        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 1.0)  # 25 within (20, 80)
        self.assertEqual(rewards[1], 1.0)  # 50 within (20, 80)
        self.assertEqual(rewards[2], 1.0)  # 100 within (40, 160)
    
    def test_len_reward_unicode_characters(self):
        """Test Unicode characters"""
        # Chinese characters should also be counted correctly
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "中" * 50 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "文" * 100 + "</context>\n\nQuestion content"]
        
        rewards = len_reward(completions, problem=problem)
        
        # 50 Chinese characters within (20, 80) range
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_len_reward_special_characters(self):
        """Test special characters"""
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "!@#$%^&*()" * 5 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "!@#$%^&*()" * 10 + "</context>\n\nQuestion content"]
        
        rewards = len_reward(completions, problem=problem)
        
        # 50 characters within (20, 80) range
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)


class TestLenRewardIntegration(unittest.TestCase):
    """Integration tests: test len_reward integration with real-world usage scenarios"""
    
    def test_realistic_scenario(self):
        """Test realistic scenario"""
        # Simulate a real question and answer
        context_text = """
        In machine learning, overfitting refers to the phenomenon where a model performs well on training data but poorly on test data.
        This is usually because the model is too complex and learns the noise and details in the training data rather than the underlying patterns.
        To prevent overfitting, techniques such as regularization, cross-validation, and early stopping can be used.
        """
        
        long_answer_text = """
        The main cause of overfitting is excessive model complexity. When a model has too many parameters, it can memorize every detail of the training data,
        including noise. Solutions include: 1) Using L1 or L2 regularization to penalize large weights; 2) Using dropout to randomly drop neurons;
        3) Collecting more training data; 4) Using cross-validation to select appropriate model complexity.
        """
        
        completions = [
            [{"role": "assistant", "content": f"<think>Analyzing the problem...</think><long_answer>{long_answer_text}</long_answer><answer>Use regularization and cross-validation</answer>"}],
        ]
        problem = [f"<context>{context_text}</context>\n\nHow to prevent overfitting?"]
        
        rewards = len_reward(completions, problem=problem)
        
        # Verify long answer length is within reasonable range
        self.assertEqual(len(rewards), 1)
        # Judge based on actual length
        context_len = len(context_text.strip())
        long_answer_len = len(long_answer_text.strip())
        if 0.2 * context_len < long_answer_len < 0.8 * context_len:
            self.assertEqual(rewards[0], 1.0)
        else:
            self.assertEqual(rewards[0], 0.0)
    
    def test_quality_control_scenario(self):
        """Test quality control scenario"""
        # Simulate long answers of different quality
        context = "<context>" + "x" * 100 + "</context>\n\nQuestion"
        
        completions = [
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "a" * 50 + "</long_answer><answer>42</answer>"}],  # Good length
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "a" * 5 + "</long_answer><answer>42</answer>"}],   # Too short, poor quality
            [{"role": "assistant", "content": "<think>thinking</think><long_answer>" + "a" * 95 + "</long_answer><answer>42</answer>"}],  # Too long, possibly redundant
        ]
        problem = [context] * 3
        
        rewards = len_reward(completions, problem=problem)
        
        # Can be used for filtering: only keep generations with appropriate length
        good_indices = [i for i, r in enumerate(rewards) if r == 1.0]
        self.assertEqual(good_indices, [0])


if __name__ == "__main__":
    unittest.main()
