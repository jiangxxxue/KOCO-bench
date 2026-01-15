import unittest
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'r1-v', 'src'))

# Import the actual function we want to test
from open_r1.grpo import plan_format_reward


class TestPlanFormatReward(unittest.TestCase):
    """测试plan_format_reward函数 - AlphaDrive文档第3节描述的格式验证奖励"""
    
    def test_plan_format_reward_correct_format(self):
        """测试正确格式的响应"""
        completions = [
            [{"content": "<think>This is reasoning</think><answer>ACCELERATE</answer>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # 正确格式应该返回1.0
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_plan_format_reward_correct_format_with_whitespace(self):
        """测试带空白字符的正确格式"""
        completions = [
            [{"content": "<think>reasoning</think>  \n  <answer>result</answer>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # 标签之间的空白字符应该被接受
        self.assertEqual(rewards[0], 1.0)
    
    def test_plan_format_reward_correct_format_multiline(self):
        """测试多行内容的正确格式"""
        completions = [
            [{"content": "<think>First line\nSecond line\nThird line</think><answer>Final answer\nwith newline</answer>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # 标签内的多行内容应该被接受
        self.assertEqual(rewards[0], 1.0)
    
    def test_plan_format_reward_missing_think_tag(self):
        """测试缺少think标签的情况"""
        completions = [
            [{"content": "<answer>ACCELERATE</answer>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # 缺少think标签应该返回0.0
        self.assertEqual(rewards[0], 0.0)
    
    def test_plan_format_reward_missing_answer_tag(self):
        """测试缺少answer标签的情况"""
        completions = [
            [{"content": "<think>reasoning</think>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # 缺少answer标签应该返回0.0
        self.assertEqual(rewards[0], 0.0)
    
    def test_plan_format_reward_wrong_order(self):
        """测试标签顺序错误的情况"""
        completions = [
            [{"content": "<answer>result</answer><think>reasoning</think>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # 标签顺序错误应该返回0.0
        self.assertEqual(rewards[0], 0.0)
    
    def test_plan_format_reward_extra_content_before(self):
        """测试标签前有额外内容的情况"""
        completions = [
            [{"content": "Extra text <think>reasoning</think><answer>result</answer>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # 标签前有额外内容应该返回0.0
        self.assertEqual(rewards[0], 0.0)
    
    def test_plan_format_reward_extra_content_after(self):
        """测试标签后有额外内容的情况"""
        completions = [
            [{"content": "<think>reasoning</think><answer>result</answer> Extra text"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # 标签后有额外内容应该返回0.0
        self.assertEqual(rewards[0], 0.0)
    
    def test_plan_format_reward_incomplete_tags(self):
        """测试不完整标签的情况"""
        test_cases = [
            [{"content": "<think>reasoning</think><answer>result"}],  # 缺少</answer>
            [{"content": "<think>reasoning<answer>result</answer>"}],  # 缺少</think>
            [{"content": "think>reasoning</think><answer>result</answer>"}],  # 缺少<think
            [{"content": "<think>reasoning</think>answer>result</answer>"}],  # 缺少<answer
        ]
        
        for completion in test_cases:
            rewards = plan_format_reward([completion])
            self.assertEqual(rewards[0], 0.0)
    
    def test_plan_format_reward_empty_tags(self):
        """测试空标签的情况"""
        completions = [
            [{"content": "<think></think><answer></answer>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # 空标签也应该被接受（只要格式正确）
        self.assertEqual(rewards[0], 1.0)
    
    def test_plan_format_reward_multiple_completions(self):
        """测试批量处理多个completions"""
        completions = [
            [{"content": "<think>reasoning1</think><answer>answer1</answer>"}],  # 正确
            [{"content": "<answer>answer2</answer>"}],  # 错误：缺少think
            [{"content": "<think>reasoning3</think><answer>answer3</answer>"}],  # 正确
            [{"content": "wrong format"}],  # 错误：完全错误
        ]
        
        rewards = plan_format_reward(completions)
        
        # 验证返回正确数量的奖励
        self.assertEqual(len(rewards), 4)
        
        # 验证每个奖励的正确性
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 0.0)
        self.assertEqual(rewards[2], 1.0)
        self.assertEqual(rewards[3], 0.0)
    
    def test_plan_format_reward_special_characters(self):
        """测试包含特殊字符的内容"""
        completions = [
            [{"content": "<think>reasoning with $pecial ch@rs!</think><answer>result with #symbols</answer>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # 特殊字符不应该影响格式验证
        self.assertEqual(rewards[0], 1.0)
    
    def test_plan_format_reward_nested_tags(self):
        """测试嵌套标签的情况"""
        completions = [
            [{"content": "<think><inner>nested</inner></think><answer>result</answer>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # 嵌套标签应该被接受（只要外层格式正确）
        self.assertEqual(rewards[0], 1.0)
    
    def test_plan_format_reward_unicode_content(self):
        """测试Unicode内容"""
        completions = [
            [{"content": "<think>推理过程 中文内容</think><answer>答案 结果</answer>"}]
        ]
        
        rewards = plan_format_reward(completions)
        
        # Unicode内容应该被正确处理
        self.assertEqual(rewards[0], 1.0)


class TestPlanFormatRewardIntegration(unittest.TestCase):
    """集成测试：测试plan_format_reward在实际场景中的表现"""
    
    def test_format_reward_with_real_content(self):
        """测试真实内容的格式验证"""
        # 模拟真实的驾驶决策响应
        completions = [
            [{"content": "<think>Based on the traffic situation, I need to slow down and prepare to stop at the red light.</think><answer>DECELERATE, STOP</answer>"}],
            [{"content": "<think>The road ahead is clear, I can maintain current speed.</think><answer>KEEP</answer>"}],
            [{"content": "<think>Need to change lanes to the left for upcoming turn.</think><answer>LEFT_CHANGE, LEFT_TURN</answer>"}],
        ]
        
        rewards = plan_format_reward(completions)
        
        # 所有真实格式的响应都应该通过验证
        self.assertEqual(len(rewards), 3)
        for reward in rewards:
            self.assertEqual(reward, 1.0)
    
    def test_format_reward_consistency(self):
        """测试格式验证的一致性"""
        # 相同格式的不同内容应该得到相同的奖励
        completions = [
            [{"content": "<think>A</think><answer>B</answer>"}],
            [{"content": "<think>X</think><answer>Y</answer>"}],
            [{"content": "<think>1</think><answer>2</answer>"}],
        ]
        
        rewards = plan_format_reward(completions)
        
        # 所有奖励应该相同
        self.assertTrue(all(r == 1.0 for r in rewards))
    
    def test_format_reward_edge_cases(self):
        """测试边缘情况"""
        edge_cases = [
            # 最小有效内容
            ([[{"content": "<think>a</think><answer>b</answer>"}]], 1.0),
            # 非常长的内容
            ([[{"content": f"<think>{'x' * 10000}</think><answer>{'y' * 10000}</answer>"}]], 1.0),
            # 只有空格的标签
            ([[{"content": "<think>   </think><answer>   </answer>"}]], 1.0),
            # 多个空白字符分隔
            ([[{"content": "<think>x</think>     \n\n\n     <answer>y</answer>"}]], 1.0),
        ]
        
        for completion, expected_reward in edge_cases:
            rewards = plan_format_reward(completion)
            self.assertEqual(rewards[0], expected_reward)


if __name__ == "__main__":
    unittest.main()
