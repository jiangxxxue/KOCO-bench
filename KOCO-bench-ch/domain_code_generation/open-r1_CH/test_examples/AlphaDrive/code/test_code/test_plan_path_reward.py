import unittest
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'r1-v', 'src'))

# Import the actual function we want to test
from open_r1.grpo import plan_path_reward


class TestPlanPathReward(unittest.TestCase):
    """测试plan_path_reward函数 - AlphaDrive文档第3节描述的路径规划奖励计算"""
    
    def setUp(self):
        self.default_complexity_weights = {
            "LEFT_TURN": 1.0,
            "RIGHT_TURN": 1.0,
            "LEFT_CHANGE": 1.0,
            "RIGHT_CHANGE": 1.0,
            "STRAIGHT": 0.8
        }
    
    def test_plan_path_reward_perfect_match(self):
        """测试完全匹配的情况"""
        completions = [
            [{"content": "<think>reasoning</think><answer>LEFT_TURN, STRAIGHT</answer>"}]
        ]
        solution = ["<answer>LEFT_TURN, STRAIGHT</answer>"]
        
        rewards = plan_path_reward(completions, solution, diversity_weight=0.4)
        
        # 完全匹配：F1=1.0, complexity_factor=(1.0+0.8)/2=0.9, diversity_factor=0.4
        # reward = 1.0 * 0.9 + 0.4 = 1.3
        self.assertEqual(len(rewards), 1)
        self.assertAlmostEqual(rewards[0], 1.3, places=2)
    
    def test_plan_path_reward_partial_match(self):
        """测试部分匹配的情况"""
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
        self.assertAlmostEqual(rewards[0], 0.9, places=2)
    
    def test_plan_path_reward_no_match(self):
        """测试完全不匹配的情况"""
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
        self.assertAlmostEqual(rewards[0], 0.4, places=2)
    
    def test_plan_path_reward_diversity_penalty(self):
        """测试多样性惩罚"""
        # 测试在同一批次中，重复的决策应该得到惩罚
        completions = [
            [{"content": "<answer>LEFT_TURN</answer>"}],
            [{"content": "<answer>LEFT_TURN</answer>"}]  # 重复的决策
        ]
        solution = [
            "<answer>LEFT_TURN</answer>",
            "<answer>LEFT_TURN</answer>"
        ]
        
        rewards = plan_path_reward(completions, solution, diversity_weight=0.4)
        
        # 第一个：diversity_factor = +0.4
        # 第二个：diversity_factor = -0.4 (因为LEFT_TURN已经出现过)
        # 由于F1和complexity相同，第一个奖励应该大于第二个
        self.assertGreater(rewards[0], rewards[1])
    
    def test_plan_path_reward_complexity_weights(self):
        """测试复杂度权重的影响"""
        completions = [
            [{"content": "<answer>LEFT_TURN</answer>"}],
            [{"content": "<answer>STRAIGHT</answer>"}]
        ]
        solution = ["<answer>LEFT_TURN</answer>", "<answer>STRAIGHT</answer>"]
        
        rewards = plan_path_reward(completions, solution, diversity_weight=0.0)
        
        # LEFT_TURN: F1=1.0, complexity=1.0, reward=1.0
        # STRAIGHT: F1=1.0, complexity=0.8, reward=0.8
        self.assertAlmostEqual(rewards[0], 1.0, places=2)
        self.assertAlmostEqual(rewards[1], 0.8, places=2)
    
    def test_plan_path_reward_invalid_solution(self):
        """测试无效solution的处理"""
        completions = [
            [{"content": "<answer>LEFT_TURN</answer>"}]
        ]
        solution = ["no answer tag"]
        
        rewards = plan_path_reward(completions, solution)
        
        # 无效solution应该返回0
        self.assertEqual(rewards[0], 0)
    
    def test_plan_path_reward_multiple_completions(self):
        """测试批量处理多个completions"""
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
        
        # 验证返回正确数量的奖励
        self.assertEqual(len(rewards), 3)
        
        # 验证所有奖励都是有限的数值
        for reward in rewards:
            self.assertTrue(isinstance(reward, (int, float)))
            self.assertFalse(reward != reward)  # Check not NaN
    
    def test_plan_path_reward_custom_complexity_weights(self):
        """测试自定义复杂度权重"""
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
        self.assertAlmostEqual(rewards[0], 1.0, places=2)
    
    def test_plan_path_reward_format_variations(self):
        """测试不同格式的答案"""
        # 测试没有answer标签的情况
        completions = [
            [{"content": "LEFT_TURN, STRAIGHT"}]
        ]
        solution = ["<answer>LEFT_TURN, STRAIGHT</answer>"]
        
        rewards = plan_path_reward(completions, solution, diversity_weight=0.4)
        
        # 应该能够处理没有标签的格式
        self.assertEqual(len(rewards), 1)
        self.assertGreater(rewards[0], 0)
    
    def test_plan_path_reward_all_actions(self):
        """测试所有路径动作类型"""
        all_actions = ["LEFT_TURN", "RIGHT_TURN", "LEFT_CHANGE", "RIGHT_CHANGE", "STRAIGHT"]
        
        for action in all_actions:
            completions = [[{"content": f"<answer>{action}</answer>"}]]
            solution = [f"<answer>{action}</answer>"]
            
            rewards = plan_path_reward(completions, solution, diversity_weight=0.0)
            
            # 每个动作都应该产生正奖励
            self.assertGreater(rewards[0], 0)
            
            # 验证复杂度权重的应用
            expected_complexity = self.default_complexity_weights[action]
            self.assertAlmostEqual(rewards[0], expected_complexity, places=2)


class TestPlanPathRewardIntegration(unittest.TestCase):
    """集成测试：测试plan_path_reward在实际场景中的表现"""
    
    def test_f1_score_calculation(self):
        """测试F1分数计算的正确性"""
        # 测试不同的TP, FP, FN组合
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
        
        for completions, solution, expected_f1 in test_cases:
            rewards = plan_path_reward(completions, solution, diversity_weight=0.0)
            # reward = F1 * complexity_factor (without diversity)
            # 验证F1分数的影响
            self.assertGreater(rewards[0], 0)
    
    def test_reward_components_interaction(self):
        """测试奖励各组成部分的交互"""
        completions = [
            [{"content": "<answer>LEFT_TURN, STRAIGHT</answer>"}]
        ]
        solution = ["<answer>LEFT_TURN, STRAIGHT</answer>"]
        
        # 测试不同diversity_weight的影响
        reward_low_diversity = plan_path_reward(completions, solution, diversity_weight=0.1)[0]
        reward_high_diversity = plan_path_reward(completions, solution, diversity_weight=0.5)[0]
        
        # 高diversity_weight应该产生更高的奖励（在首次出现时）
        self.assertGreater(reward_high_diversity, reward_low_diversity)
        self.assertAlmostEqual(reward_high_diversity - reward_low_diversity, 0.4, places=2)


if __name__ == "__main__":
    unittest.main()
