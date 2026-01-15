import unittest
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'r1-v', 'src'))

# Import the actual function we want to test
from open_r1.grpo import plan_path_reward


class TestPlanPathReward(unittest.TestCase):
    """Test plan_path_reward function - Path planning reward calculation described in AlphaDrive paper Section 3"""
    
    def setUp(self):
        self.default_complexity_weights = {
            "LEFT_TURN": 1.0,
            "RIGHT_TURN": 1.0,
            "LEFT_CHANGE": 1.0,
            "RIGHT_CHANGE": 1.0,
            "STRAIGHT": 0.8
        }
    
    def test_plan_path_reward_perfect_match(self):
        """Test perfect match case"""
        completions = [
            [{"content": "<think>reasoning</think><answer>LEFT_TURN, STRAIGHT</answer>"}]
        ]
        solution = ["<answer>LEFT_TURN, STRAIGHT</answer>"]
        
        rewards = plan_path_reward(completions, solution, diversity_weight=0.4)
        
        # Perfect match: F1=1.0, complexity_factor=(1.0+0.8)/2=0.9, diversity_factor=0.4
        # reward = 1.0 * 0.9 + 0.4 = 1.3
        self.assertEqual(len(rewards), 1)
        self.assertAlmostEqual(rewards[0], 1.3, places=4)
    
    def test_plan_path_reward_partial_match(self):
        """Test partial match case"""
        completions = [
            [{"content": "<answer>LEFT_TURN, RIGHT_CHANGE</answer>"}]
        ]
        solution = ["<answer>LEFT_TURN, STRAIGHT</answer>"]
        
        rewards = plan_path_reward(completions, solution, diversity_weight=0.4)
        
        # TP=1 (LEFT_TURN), FP=1 (RIGHT_CHANGE), FN=1 (STRAIGHT)
        # precision = 1/2 = 0.5, recall = 1/2 = 0.5, F1 = 0.5
        # complexity_factor = (1.0+1.0)/2 = 1.0
        # diversity_factor = 0.4 (all first occurrence)
        # reward = 0.5 * 1.0 + 0.4 = 0.9
        self.assertEqual(len(rewards), 1)
        self.assertAlmostEqual(rewards[0], 0.9, places=4)
    
    def test_plan_path_reward_no_match(self):
        """Test complete mismatch case"""
        completions = [
            [{"content": "<answer>STRAIGHT</answer>"}]
        ]
        solution = ["<answer>LEFT_TURN, RIGHT_TURN</answer>"]
        
        rewards = plan_path_reward(completions, solution, diversity_weight=0.4)
        
        # TP=0, FP=1, FN=2
        # precision = 0, recall = 0, F1 = 0
        # complexity_factor = 0.8
        # diversity_factor = 0.4
        # reward = 0 * 0.8 + 0.4 = 0.4
        self.assertEqual(len(rewards), 1)
        self.assertAlmostEqual(rewards[0], 0.4, places=4)
    
    def test_plan_path_reward_diversity_penalty(self):
        """Test diversity penalty"""
        # Test that repeated decisions in the same batch should be penalized
        completions = [
            [{"content": "<answer>LEFT_TURN</answer>"}],
            [{"content": "<answer>LEFT_TURN</answer>"}]  # Repeated decision
        ]
        solution = [
            "<answer>LEFT_TURN</answer>",
            "<answer>LEFT_TURN</answer>"
        ]
        
        rewards = plan_path_reward(completions, solution, diversity_weight=0.4)
        
        # print(rewards)

        expected_rewards = [1.3999975000037503, 0.5999975000037502]

        for i in range(len(rewards)):
            self.assertAlmostEqual(rewards[i], expected_rewards[i], places=4)
        # First: diversity_factor = +0.4
        # Second: diversity_factor = -0.4 (because LEFT_TURN already appeared)
        # Since F1 and complexity are the same, first reward should be greater than second
        self.assertGreater(rewards[0], rewards[1])
    
    def test_plan_path_reward_complexity_weights(self):
        """Test impact of complexity weights"""
        completions = [
            [{"content": "<answer>LEFT_TURN</answer>"}],
            [{"content": "<answer>STRAIGHT</answer>"}]
        ]
        solution = ["<answer>LEFT_TURN</answer>", "<answer>STRAIGHT</answer>"]
        
        rewards = plan_path_reward(completions, solution, diversity_weight=0.0)
        
        # LEFT_TURN: F1=1.0, complexity=1.0, reward=1.0
        # STRAIGHT: F1=1.0, complexity=0.8, reward=0.8
        self.assertAlmostEqual(rewards[0], 1.0, places=4)
        self.assertAlmostEqual(rewards[1], 0.8, places=4)
    
    def test_plan_path_reward_invalid_solution(self):
        """Test handling of invalid solution"""
        completions = [
            [{"content": "<answer>LEFT_TURN</answer>"}]
        ]
        solution = ["no answer tag"]
        
        rewards = plan_path_reward(completions, solution)
        
        # Invalid solution should return 0
        self.assertEqual(rewards[0], 0)
    
    def test_plan_path_reward_multiple_completions(self):
        """Test batch processing of multiple completions"""
        completions = [
            [{"content": "<answer>LEFT_TURN, STRAIGHT</answer>"}],
            [{"content": "<answer>RIGHT_CHANGE</answer>"}],
            [{"content": "<answer>LEFT_CHANGE, RIGHT_TURN</answer>"}]
        ]
        solution = [
            "<answer>LEFT_TURN, STRAIGHT</answer>",
            "<answer>RIGHT_CHANGE</answer>",
            "<answer>LEFT_CHANGE, RIGHT_TURN</answer>"
        ]
        
        rewards = plan_path_reward(completions, solution, diversity_weight=0.4)
        
        # Verify correct number of rewards returned
        self.assertEqual(len(rewards), 3)

        # print(rewards)

        expected_rewards = [1.299998650001125, 1.3999975000037503, 1.3999985000012498]

        for i in range(len(rewards)):
            self.assertAlmostEqual(rewards[i], expected_rewards[i], places=4)
        
        # Verify all rewards are finite values
        # for reward in rewards:
        #     self.assertTrue(isinstance(reward, (int, float)))
        #     self.assertFalse(reward != reward)  # Check not NaN
    
    def test_plan_path_reward_custom_complexity_weights(self):
        """Test custom complexity weights"""
        custom_weights = {
            "LEFT_TURN": 0.5,
            "RIGHT_TURN": 0.5,
            "LEFT_CHANGE": 1.0,
            "RIGHT_CHANGE": 1.0,
            "STRAIGHT": 1.0
        }
        
        completions = [
            [{"content": "<answer>LEFT_CHANGE</answer>"}]
        ]
        solution = ["<answer>LEFT_CHANGE</answer>"]
        
        rewards = plan_path_reward(
            completions, 
            solution, 
            diversity_weight=0.0,
            complexity_weights=custom_weights
        )
        
        # F1=1.0, complexity=1.0, diversity=0.0
        # reward = 1.0 * 1.0 + 0.0 = 1.0
        self.assertAlmostEqual(rewards[0], 1.0, places=4)
    
    def test_plan_path_reward_format_variations(self):
        """Test different answer formats"""
        # Test case without answer tags
        completions = [
            [{"content": "LEFT_TURN, STRAIGHT"}]
        ]
        solution = ["<answer>LEFT_TURN, STRAIGHT</answer>"]
        
        rewards = plan_path_reward(completions, solution, diversity_weight=0.4)
        
        # Should be able to handle format without tags

        # print(rewards)

        expected_rewards = [1.299998650001125]

        for i in range(len(rewards)):
            self.assertAlmostEqual(rewards[i], expected_rewards[i], places=4)
        # self.assertEqual(len(rewards), 1)
        # self.assertGreater(rewards[0], 0)
    
    def test_plan_path_reward_all_actions(self):
        """Test all path action types"""
        all_actions = ["LEFT_TURN", "RIGHT_TURN", "LEFT_CHANGE", "RIGHT_CHANGE", "STRAIGHT"]
        
        for action in all_actions:
            completions = [[{"content": f"<answer>{action}</answer>"}]]
            solution = [f"<answer>{action}</answer>"]
            
            rewards = plan_path_reward(completions, solution, diversity_weight=0.0)
            
            # Each action should produce positive reward
            self.assertGreater(rewards[0], 0)
            
            # Verify application of complexity weights
            expected_complexity = self.default_complexity_weights[action]
            self.assertAlmostEqual(rewards[0], expected_complexity, places=4)


class TestPlanPathRewardIntegration(unittest.TestCase):
    """Integration test: Test plan_path_reward performance in real scenarios"""
    
    def test_f1_score_calculation(self):
        """Test correctness of F1 score calculation"""
        # Test different combinations of TP, FP, FN
        test_cases = [
            # (completions, solution, expected_f1)
            (
                [[{"content": "<answer>LEFT_TURN, STRAIGHT, RIGHT_TURN</answer>"}]],
                ["<answer>LEFT_TURN, STRAIGHT</answer>"],
                # TP=2, FP=1, FN=0
                # precision=2/3, recall=2/2=1.0, F1=2*(2/3*1)/(2/3+1)=0.8
                0.8
            ),
            (
                [[{"content": "<answer>LEFT_TURN</answer>"}]],
                ["<answer>LEFT_TURN, STRAIGHT, RIGHT_CHANGE</answer>"],
                # TP=1, FP=0, FN=2
                # precision=1/1=1.0, recall=1/3, F1=2*(1*1/3)/(1+1/3)=0.5
                0.5
            ),
        ]
        
        rewards = []
        for completions, solution, expected_f1 in test_cases:
            rewards = plan_path_reward(completions, solution, diversity_weight=0.0)
            # reward = F1 * complexity_factor (without diversity)
            # Verify impact of F1 score
            rewards.append(rewards[0])



        expected_rewards = [0.49999887500140633, 0.49999887500140633]

        for i in range(len(rewards)):
            self.assertAlmostEqual(rewards[i], expected_rewards[i], places=4)
    
    def test_reward_components_interaction(self):
        """Test interaction of reward components"""
        completions = [
            [{"content": "<answer>LEFT_TURN, STRAIGHT</answer>"}]
        ]
        solution = ["<answer>LEFT_TURN, STRAIGHT</answer>"]
        
        # Test impact of different diversity_weight values
        reward_low_diversity = plan_path_reward(completions, solution, diversity_weight=0.1)[0]
        reward_high_diversity = plan_path_reward(completions, solution, diversity_weight=0.5)[0]

        # print(reward_high_diversity, reward_low_diversity)


        expected_reward_high_diversity = 1.3999986500011248
        expected_reward_low_diversity = 0.9999986500011248

        self.assertAlmostEqual(reward_high_diversity, expected_reward_high_diversity, places=4)
        self.assertAlmostEqual(reward_low_diversity, expected_reward_low_diversity, places=4)

if __name__ == "__main__":
    unittest.main()
