import unittest
import torch
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Import the actual function we want to test
from open_r1.grpo_jsonl import accuracy_reward


class TestAccuracyReward(unittest.TestCase):
    """Test accuracy_reward function - Batch calculate fidelity rewards for all samples"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        self.num_generations = 2
    
    def test_accuracy_reward_basic(self):
        """Test basic accuracy reward calculation"""
        # Create test data: 2 samples, 2 generations each
        completions = [
            [{"role": "assistant", "content": "<think>reasoning</think><answer>4.5</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>4.3</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>2.8</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>3.2</answer>"}],
        ]
        
        solution = [
            "<answer>4.5</answer>",
            "<answer>4.5</answer>",
            "<answer>3.0</answer>",
            "<answer>3.0</answer>",
        ]
        
        kwargs = {
            "device": self.device,
            "num_generations": self.num_generations,
        }
        
        rewards = accuracy_reward(completions, solution, **kwargs)
        
        # print(rewards)

        expected_rewards = [torch.tensor(1.0010), torch.tensor(1.0010), torch.tensor(1.0010), torch.tensor(1.0010)]
        # Verify output
        self.assertEqual(len(rewards), 4)  # 2 samples × 2 generations
        for i in range(len(rewards)):
            torch.testing.assert_close(rewards[i], expected_rewards[i],atol=1e-4, rtol=1e-4)
    
    def test_accuracy_reward_correct_predictions(self):
        """Test cases where predictions are correct"""
        completions = [
            [{"role": "assistant", "content": "<think>reasoning</think><answer>5.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>5.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>1.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>1.0</answer>"}],
        ]
        
        solution = [
            "<answer>5.0</answer>",
            "<answer>5.0</answer>",
            "<answer>1.0</answer>",
            "<answer>1.0</answer>",
        ]
        
        kwargs = {
            "device": self.device,
            "num_generations": self.num_generations,
        }
        
        rewards = accuracy_reward(completions, solution, **kwargs)

        expected_rewards = [torch.tensor(1.0010), torch.tensor(1.0010), torch.tensor(1.0010), torch.tensor(1.0010)]
        
        # When predictions are completely correct, rewards should be high
        self.assertEqual(len(rewards), 4)
        for i in range(len(rewards)):
            torch.testing.assert_close(rewards[i], expected_rewards[i],atol=1e-4, rtol=1e-4)
    
    def test_accuracy_reward_with_variance(self):
        """Test impact of variance in different generation results"""
        # First sample: low variance in generation results
        # Second sample: high variance in generation results
        completions = [
            [{"role": "assistant", "content": "<think>reasoning</think><answer>4.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>4.1</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>2.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>5.0</answer>"}],
        ]
        
        solution = [
            "<answer>4.0</answer>",
            "<answer>4.0</answer>",
            "<answer>3.0</answer>",
            "<answer>3.0</answer>",
        ]
        
        kwargs = {
            "device": self.device,
            "num_generations": self.num_generations,
        }
        
        rewards = accuracy_reward(completions, solution, **kwargs)
        
        # print(rewards)
        expected_rewards = [torch.tensor(0.7711), torch.tensor(0.7829), torch.tensor(0.9137), torch.tensor(0.5730)]
        self.assertEqual(len(rewards), 4)
        for i in range(len(rewards)):
            torch.testing.assert_close(rewards[i], expected_rewards[i],atol=1e-4, rtol=1e-4)
    
    def test_accuracy_reward_parsing_formats(self):
        """Test parsing answers in different formats"""
        completions = [
            [{"role": "assistant", "content": "<think>step1</think><answer>3.5</answer>"}],
            [{"role": "assistant", "content": "<think>step2</think><answer>The score is 3.7</answer>"}],
            [{"role": "assistant", "content": "<think>step3</think><answer>2.5</answer>"}],
            [{"role": "assistant", "content": "<think>step4</think><answer>Score: 2.3</answer>"}],
        ]
        
        solution = [
            "<answer>3.5</answer>",
            "<answer>3.5</answer>",
            "<answer>2.5</answer>",
            "<answer>2.5</answer>",
        ]
        
        kwargs = {
            "device": self.device,
            "num_generations": self.num_generations,
        }
        
        rewards = accuracy_reward(completions, solution, **kwargs)
        
        # print(rewards)
        expected_rewards = [torch.tensor(1.0010), torch.tensor(1.0010), torch.tensor(1.0010), torch.tensor(1.0010)]
        self.assertEqual(len(rewards), 4)
        for i in range(len(rewards)):
            torch.testing.assert_close(rewards[i], expected_rewards[i],atol=1e-4, rtol=1e-4)

    
    def test_accuracy_reward_pairwise_comparison(self):
        """Test pairwise comparison logic"""
        # Create clear ranking relationships
        completions = [
            [{"role": "assistant", "content": "<think>reasoning</think><answer>5.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>5.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>3.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>3.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>1.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>1.0</answer>"}],
        ]
        
        solution = [
            "<answer>5.0</answer>",
            "<answer>5.0</answer>",
            "<answer>3.0</answer>",
            "<answer>3.0</answer>",
            "<answer>1.0</answer>",
            "<answer>1.0</answer>",
        ]
        
        kwargs = {
            "device": self.device,
            "num_generations": self.num_generations,
        }
        
        rewards = accuracy_reward(completions, solution, **kwargs)
        # print(rewards)
        expected_rewards = [torch.tensor(1.0010), torch.tensor(1.0010), torch.tensor(1.0010), torch.tensor(1.0010), torch.tensor(1.0010), torch.tensor(1.0010)]
        # Verify rewards from pairwise comparison
        self.assertEqual(len(rewards), 6)  # 3 samples × 2 generations
        for i in range(len(rewards)):
            torch.testing.assert_close(rewards[i], expected_rewards[i],atol=1e-4, rtol=1e-4)
    
    def test_accuracy_reward_error_handling(self):
        """Test error handling: unparseable answers"""
        completions = [
            [{"role": "assistant", "content": "<think>reasoning</think><answer>invalid</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>4.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>3.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>2.0</answer>"}],
        ]
        
        solution = [
            "<answer>4.0</answer>",
            "<answer>4.0</answer>",
            "<answer>3.0</answer>",
            "<answer>3.0</answer>",
        ]
        
        kwargs = {
            "device": self.device,
            "num_generations": self.num_generations,
        }
        
        # Should be able to handle invalid answers (using random values)
        rewards = accuracy_reward(completions, solution, **kwargs)
        
        self.assertEqual(len(rewards), 4)
        for reward in rewards:
            if torch.is_tensor(reward):
                self.assertTrue(torch.isfinite(reward))
    

class TestAccuracyRewardIntegration(unittest.TestCase):
    """Integration test: Test overall workflow of accuracy_reward"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')

    
    def test_ranking_consistency(self):
        """Test ranking consistency"""
        # Create clear quality ranking
        completions = [
            [{"role": "assistant", "content": "<think>reasoning</think><answer>5.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>5.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>1.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>1.0</answer>"}],
        ]
        
        solution = [
            "<answer>5.0</answer>",
            "<answer>5.0</answer>",
            "<answer>1.0</answer>",
            "<answer>1.0</answer>",
        ]
        
        kwargs = {
            "device": self.device,
            "num_generations": 2,
        }
        
        rewards = accuracy_reward(completions, solution, **kwargs)
        # print(rewards)
        expected_rewards = [torch.tensor(1.0010), torch.tensor(1.0010), torch.tensor(1.0010), torch.tensor(1.0010)]
        # Verify consistency of rewards
        self.assertEqual(len(rewards), 4)
        for i in range(len(rewards)):
            torch.testing.assert_close(rewards[i], expected_rewards[i],atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
