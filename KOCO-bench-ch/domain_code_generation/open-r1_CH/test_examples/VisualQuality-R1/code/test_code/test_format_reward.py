import unittest
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))

# Import the actual function we want to test
from open_r1.grpo_jsonl import format_reward


class TestFormatReward(unittest.TestCase):
    """测试format_reward函数 - 检查生成的响应是否符合指定的格式要求"""
    
    def test_format_reward_correct_format(self):
        """测试正确格式的响应"""
        completions = [
            [{"role": "assistant", "content": "<think>This is reasoning</think><answer>4.5</answer>"}],
            [{"role": "assistant", "content": "<think>Another reasoning</think><answer>3.2</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # 验证输出
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 1.0)
    
    def test_format_reward_missing_think_tag(self):
        """测试缺少think标签的情况"""
        completions = [
            [{"role": "assistant", "content": "<answer>4.5</answer>"}],
            [{"role": "assistant", "content": "Just some text<answer>3.2</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # 缺少think标签，应该返回0.0
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 0.0)
        self.assertEqual(rewards[1], 0.0)
    
    def test_format_reward_missing_answer_tag(self):
        """测试缺少answer标签的情况"""
        completions = [
            [{"role": "assistant", "content": "<think>This is reasoning</think>"}],
            [{"role": "assistant", "content": "<think>Another reasoning</think>Some text"}],
        ]
        
        rewards = format_reward(completions)
        
        # 缺少answer标签，应该返回0.0
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 0.0)
        self.assertEqual(rewards[1], 0.0)
    
    def test_format_reward_wrong_order(self):
        """测试标签顺序错误的情况"""
        completions = [
            [{"role": "assistant", "content": "<answer>4.5</answer><think>This is reasoning</think>"}],
        ]
        
        rewards = format_reward(completions)
        
        # 标签顺序错误，应该返回0.0
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_format_reward_extra_content_before(self):
        """测试标签前有额外内容的情况"""
        completions = [
            [{"role": "assistant", "content": "Extra text<think>This is reasoning</think><answer>4.5</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # 标签前有额外内容，应该返回0.0
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_format_reward_extra_content_after(self):
        """测试标签后有额外内容的情况"""
        completions = [
            [{"role": "assistant", "content": "<think>This is reasoning</think><answer>4.5</answer>Extra text"}],
        ]
        
        rewards = format_reward(completions)
        
        # 标签后有额外内容，应该返回0.0
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_format_reward_multiline_content(self):
        """测试多行内容的格式检查"""
        completions = [
            [{"role": "assistant", "content": "<think>Line 1\nLine 2\nLine 3</think><answer>4.5</answer>"}],
            [{"role": "assistant", "content": "<think>Reasoning</think>\n<answer>3.2</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # 第一个有正确格式，第二个有换行符在标签之间
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)
        # 第二个也应该匹配，因为\s*允许空白字符
        self.assertEqual(rewards[1], 1.0)
    
    def test_format_reward_empty_tags(self):
        """测试空标签的情况"""
        completions = [
            [{"role": "assistant", "content": "<think></think><answer></answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # 空标签也应该被接受（只要格式正确）
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_format_reward_nested_tags(self):
        """测试嵌套标签的情况"""
        completions = [
            [{"role": "assistant", "content": "<think><inner>nested</inner></think><answer>4.5</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # 嵌套标签应该被接受
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_format_reward_special_characters(self):
        """测试特殊字符的处理"""
        completions = [
            [{"role": "assistant", "content": "<think>Reasoning with $math$ and symbols!</think><answer>4.5</answer>"}],
            [{"role": "assistant", "content": "<think>中文推理过程</think><answer>3.5</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # 特殊字符和中文应该被正确处理
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 1.0)
    
    def test_format_reward_batch_processing(self):
        """测试批量处理"""
        completions = [
            [{"role": "assistant", "content": "<think>Reasoning 1</think><answer>4.5</answer>"}],
            [{"role": "assistant", "content": "Invalid format"}],
            [{"role": "assistant", "content": "<think>Reasoning 2</think><answer>3.2</answer>"}],
            [{"role": "assistant", "content": "<answer>2.1</answer>"}],
            [{"role": "assistant", "content": "<think>Reasoning 3</think><answer>5.0</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # 验证批量处理的结果
        self.assertEqual(len(rewards), 5)
        self.assertEqual(rewards[0], 1.0)  # 正确
        self.assertEqual(rewards[1], 0.0)  # 无效
        self.assertEqual(rewards[2], 1.0)  # 正确
        self.assertEqual(rewards[3], 0.0)  # 缺少think
        self.assertEqual(rewards[4], 1.0)  # 正确
    
    def test_format_reward_whitespace_variations(self):
        """测试不同空白字符的处理"""
        completions = [
            [{"role": "assistant", "content": "<think>Reasoning</think><answer>4.5</answer>"}],
            [{"role": "assistant", "content": "<think>Reasoning</think> <answer>4.5</answer>"}],
            [{"role": "assistant", "content": "<think>Reasoning</think>  <answer>4.5</answer>"}],
            [{"role": "assistant", "content": "<think>Reasoning</think>\n<answer>4.5</answer>"}],
            [{"role": "assistant", "content": "<think>Reasoning</think>\t<answer>4.5</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # 所有变体都应该被接受（\s*匹配任意空白字符）
        self.assertEqual(len(rewards), 5)
        for reward in rewards:
            self.assertEqual(reward, 1.0)


class TestFormatRewardIntegration(unittest.TestCase):
    """集成测试：测试format_reward在实际场景中的应用"""
    
    def test_combined_with_accuracy_reward(self):
        """测试与accuracy_reward结合使用的场景"""
        # 模拟GRPO训练中的格式检查
        completions = [
            [{"role": "assistant", "content": "<think>Step 1: Analyze\nStep 2: Calculate</think><answer>4.5</answer>"}],
            [{"role": "assistant", "content": "<think>Quick reasoning</think><answer>3.2</answer>"}],
            [{"role": "assistant", "content": "4.1"}],  # 无格式
            [{"role": "assistant", "content": "<think>Detailed analysis</think><answer>2.8</answer>"}],
        ]
        
        format_rewards = format_reward(completions)
        
        # 验证格式奖励
        self.assertEqual(len(format_rewards), 4)
        self.assertEqual(format_rewards[0], 1.0)
        self.assertEqual(format_rewards[1], 1.0)
        self.assertEqual(format_rewards[2], 0.0)
        self.assertEqual(format_rewards[3], 1.0)
        
        # 计算总奖励（格式奖励 + 准确度奖励）
        # 在实际应用中，无格式的响应会得到0分的格式奖励
        total_rewards = [fr for fr in format_rewards]
        self.assertEqual(sum(total_rewards), 3.0)
    
    def test_edge_case_empty_completion(self):
        """测试空响应的情况"""
        completions = [
            [{"role": "assistant", "content": ""}],
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_multiple_think_answer_pairs(self):
        """测试多个think-answer对的情况"""
        completions = [
            [{"role": "assistant", "content": "<think>First</think><answer>1</answer><think>Second</think><answer>2</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)


if __name__ == "__main__":
    unittest.main()
