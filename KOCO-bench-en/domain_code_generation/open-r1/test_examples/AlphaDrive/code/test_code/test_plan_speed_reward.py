import unittest
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'r1-v', 'src'))

# Import the actual function we want to test
from open_r1.grpo import plan_speed_reward


class TestPlanSpeedReward(unittest.TestCase):
    """Test plan_speed_reward function - Speed planning reward calculation described in AlphaDrive paper Section 3"""
    
    def setUp(self):
        self.default_complexity_weights = {
            "ACCELERATE": 0.9,
            "DECELERATE": 1.0,
            "STOP": 1.0,
            "KEEP": 0.8
        }
    
    def test_plan_speed_reward_perfect_match(self):
        """Test perfect match case"""
        completions = [
            [{"content": "<think>reasoning</think><answer>ACCELERATE, STOP</answer>"}]
        ]
        solution = ["<answer>ACCELERATE, STOP</answer>"]
        
        rewards = plan_speed_reward(completions, solution, diversity_weight=0.4)
        
        # Perfect match: F1=1.0, complexity_factor=(0.9+1.0)/2=0.95, diversity_factor=0.4
        # reward = 1.0 * 0.95 + 0.4 = 1.35
        self.assertEqual(len(rewards), 1)
        self.assertAlmostEqual(rewards[0], 1.35, places=4)
    
    def test_plan_speed_reward_partial_match(self):
        """Test partial match case"""
        completions = [
            [{"content": "<answer>ACCELERATE, KEEP</answer>"}]
        ]
        solution = ["<answer>ACCELERATE, STOP</answer>"]
        
        rewards = plan_speed_reward(completions, solution, diversity_weight=0.4)
        
        # TP=1 (ACCELERATE), FP=1 (KEEP), FN=1 (STOP)
        # precision = 1/2 = 0.5, recall = 1/2 = 0.5, F1 = 0.5
        # complexity_factor = (0.9+0.8)/2 = 0.85
        # diversity_factor = 0.4 (all first occurrence)
        # reward = 0.5 * 0.85 + 0.4 = 0.825
        self.assertEqual(len(rewards), 1)
        self.assertAlmostEqual(rewards[0], 0.825, places=4)
    
    def test_plan_speed_reward_no_match(self):
        """Test complete mismatch case"""
        completions = [
            [{"content": "<answer>KEEP</answer>"}]
        ]
        solution = ["<answer>ACCELERATE, STOP</answer>"]
        
        rewards = plan_speed_reward(completions, solution, diversity_weight=0.4)
        
        # TP=0, FP=1, FN=2
        # precision = 0, recall = 0, F1 = 0
        # complexity_factor = 0.8
        # diversity_factor = 0.4
        # reward = 0 * 0.8 + 0.4 = 0.4
        self.assertEqual(len(rewards), 1)
        self.assertAlmostEqual(rewards[0], 0.4, places=4)
    
    def test_plan_speed_reward_diversity_penalty(self):
        """Test diversity penalty"""
        # Test that repeated decisions in the same batch should be penalized
        completions = [
            [{"content": "<answer>ACCELERATE</answer>"}],
            [{"content": "<answer>ACCELERATE</answer>"}]  # Repeated decision
        ]
        solution = [
            "<answer>ACCELERATE</answer>",
            "<answer>ACCELERATE</answer>"
        ]
        
        rewards = plan_speed_reward(completions, solution, diversity_weight=0.4)
        
        # print(rewards)

        expected_rewards =[1.2999977500033753, 0.4999977500033752]

        for i in range(len(rewards)):
            self.assertAlmostEqual(rewards[i], expected_rewards[i], places=4)
        # First: diversity_factor = +0.4
        # Second: diversity_factor = -0.4 (because ACCELERATE already appeared)
        # Since F1 and complexity are the same, first reward should be greater than second
        self.assertGreater(rewards[0], rewards[1])
    
    def test_plan_speed_reward_complexity_weights(self):
        """Test impact of complexity weights"""
        completions = [
            [{"content": "<answer>STOP</answer>"}],
            [{"content": "<answer>KEEP</answer>"}]
        ]
        solution = ["<answer>STOP</answer>", "<answer>KEEP</answer>"]
        
        rewards = plan_speed_reward(completions, solution, diversity_weight=0.0)
        
        # STOP: F1=1.0, complexity=1.0, reward=1.0
        # KEEP: F1=1.0, complexity=0.8, reward=0.8
        self.assertAlmostEqual(rewards[0], 1.0, places=4)
        self.assertAlmostEqual(rewards[1], 0.8, places=4)
    
    def test_plan_speed_reward_invalid_solution(self):
        """Test handling of invalid solution"""
        completions = [
            [{"content": "<answer>ACCELERATE</answer>"}]
        ]
        solution = ["no answer tag"]
        
        rewards = plan_speed_reward(completions, solution)
        
        # Invalid solution should return 0
        self.assertEqual(rewards[0], 0)
    
    def test_plan_speed_reward_multiple_completions(self):
        """Test batch processing of multiple completions"""
        completions = [
            [{"content": "<answer>ACCELERATE, STOP</answer>"}],
            [{"content": "<answer>DECELERATE</answer>"}],
            [{"content": "<answer>KEEP, ACCELERATE</answer>"}]
        ]
        solution = [
            "<answer>ACCELERATE, STOP</answer>",
            "<answer>DECELERATE</answer>",
            "<answer>KEEP, ACCELERATE</answer>"
        ]
        
        rewards = plan_speed_reward(completions, solution, diversity_weight=0.4)
        
        # Verify correct number of rewards returned
        self.assertEqual(len(rewards), 3)

        expected_rewards = [1.3499985750011874, 1.3999975000037503, 0.4499987250010624]
        for i in range(len(rewards)):
            self.assertAlmostEqual(rewards[i], expected_rewards[i], places=4)
        # print(rewards)
        # # Verify all rewards are finite values
        # for reward in rewards:
        #     self.assertTrue(isinstance(reward, (int, float)))
        #     self.assertFalse(reward != reward)  # Check not NaN
    
    def test_plan_speed_reward_custom_complexity_weights(self):
        """Test custom complexity weights"""
        custom_weights = {
            "ACCELERATE": 1.0,
            "DECELERATE": 0.5,
            "STOP": 0.5,
            "KEEP": 1.0
        }
        
        completions = [
            [{"content": "<answer>ACCELERATE</answer>"}]
        ]
        solution = ["<answer>ACCELERATE</answer>"]
        
        rewards = plan_speed_reward(
            completions, 
            solution, 
            diversity_weight=0.0,
            complexity_weights=custom_weights
        )
        
        # F1=1.0, complexity=1.0, diversity=0.0
        # reward = 1.0 * 1.0 + 0.0 = 1.0
        self.assertAlmostEqual(rewards[0], 1.0, places=4)
    
    def test_plan_speed_reward_format_variations(self):
        """Test different answer formats"""
        # Test case without answer tags
        completions = [
            [{"content": "ACCELERATE, STOP"}]
        ]
        solution = ["<answer>ACCELERATE, STOP</answer>"]
        
        rewards = plan_speed_reward(completions, solution, diversity_weight=0.4)
        
        # print(rewards)

        expected_rewards = [1.3499985750011874]

        for i in range(len(rewards)):
            self.assertAlmostEqual(rewards[i], expected_rewards[i], places=4)
        # Should be able to handle format without tags
        # self.assertEqual(len(rewards), 1)
        # self.assertGreater(rewards[0], 0)


class TestPlanSpeedRewardIntegration(unittest.TestCase):
    """Integration test: Test plan_speed_reward performance in real scenarios"""
    
    def test_f1_score_calculation(self):
        """Test correctness of F1 score calculation"""
        # Test different combinations of TP, FP, FN
        test_cases = [
            # (completions, solution, expected_f1)
            (
                [[{"content": "<answer>ACCELERATE, STOP, KEEP</answer>"}]],
                ["<answer>ACCELERATE, STOP</answer>"],
                # TP=2, FP=1, FN=0
                # precision=2/3, recall=2/2=1.0, F1=2*(2/3*1)/(2/3+1)=0.8
                0.8
            ),
            (
                [[{"content": "<answer>ACCELERATE</answer>"}]],
                ["<answer>ACCELERATE, STOP, DECELERATE</answer>"],
                # TP=1, FP=0, FN=2
                # precision=1/1=1.0, recall=1/3, F1=2*(1*1/3)/(1+1/3)=0.5
                0.5
            ),
        ]
        
        rewards = []
        for completions, solution, expected_f1 in test_cases:
            reward = plan_speed_reward(completions, solution, diversity_weight=0.0)
            # reward = F1 * complexity_factor (without diversity)
            rewards.append(reward[0])
            # Verify impact of F1 score
            # self.assertGreater(reward, 0)

        # print(rewards)
        expected_rewards = [0.71999904000068, 0.4499989875012657]

        for i in range(len(rewards)):
            self.assertAlmostEqual(rewards[i], expected_rewards[i], places=4)
    
    def test_reward_components_interaction(self):
        """Test interaction of reward components"""
        completions = [
            [{"content": "<answer>ACCELERATE, STOP</answer>"}]
        ]
        solution = ["<answer>ACCELERATE, STOP</answer>"]
        
        # Test impact of different diversity_weight values
        reward_low_diversity = plan_speed_reward(completions, solution, diversity_weight=0.1)[0]
        reward_high_diversity = plan_speed_reward(completions, solution, diversity_weight=0.5)[0]
        
        # print(reward_high_diversity, reward_low_diversity)

        expected_reward_high_diversity = 1.4499985750011875
        expected_reward_low_diversity = 1.0499985750011873

        self.assertAlmostEqual(reward_high_diversity, expected_reward_high_diversity, places=4)
        self.assertAlmostEqual(reward_low_diversity, expected_reward_low_diversity, places=4)
        # High diversity_weight should produce higher reward (on first occurrence)
        # self.assertGreater(reward_high_diversity, reward_low_diversity)
        # self.assertAlmostEqual(reward_high_diversity - reward_low_diversity, 0.4, places=2)


if __name__ == "__main__":
    unittest.main()
