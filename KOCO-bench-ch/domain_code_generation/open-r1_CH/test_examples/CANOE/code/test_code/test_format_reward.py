import unittest
import sys
import os

# Add the parent directory to Python path to import open_r1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train', 'src'))

# Import the actual function we want to test
from open_r1.grpo import format_reward


class TestFormatReward(unittest.TestCase):
    """测试format_reward函数 - CANOE文档第3节描述的格式验证"""
    
    def test_format_reward_correct_format(self):
        """测试正确格式的情况"""
        completions = [
            [{"role": "assistant", "content": "<think>思考过程</think><long_answer>详细答案</long_answer><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>另一个思考</think><long_answer>另一个详细答案</long_answer><answer>100</answer>"}],
        ]
        
        # 调用真实的format_reward函数
        rewards = format_reward(completions)
        
        # 验证输出
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # 格式正确
        self.assertEqual(rewards[1], 1.0)  # 格式正确
    
    def test_format_reward_incorrect_format_missing_tags(self):
        """测试缺少标签的情况"""
        completions = [
            [{"role": "assistant", "content": "<think>思考过程</think><answer>42</answer>"}],  # 缺少long_answer
            [{"role": "assistant", "content": "<long_answer>详细答案</long_answer><answer>100</answer>"}],  # 缺少think
            [{"role": "assistant", "content": "<think>思考过程</think><long_answer>详细答案</long_answer>"}],  # 缺少answer
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 0.0)  # 格式错误
        self.assertEqual(rewards[1], 0.0)  # 格式错误
        self.assertEqual(rewards[2], 0.0)  # 格式错误
    
    def test_format_reward_incorrect_order(self):
        """测试标签顺序错误的情况"""
        completions = [
            [{"role": "assistant", "content": "<answer>42</answer><think>思考过程</think><long_answer>详细答案</long_answer>"}],  # 顺序错误
            [{"role": "assistant", "content": "<think>思考过程</think><answer>42</answer><long_answer>详细答案</long_answer>"}],  # 顺序错误
            [{"role": "assistant", "content": "<long_answer>详细答案</long_answer><think>思考过程</think><answer>42</answer>"}],  # 顺序错误
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 0.0)  # 顺序错误
        self.assertEqual(rewards[1], 0.0)  # 顺序错误
        self.assertEqual(rewards[2], 0.0)  # 顺序错误
    
    def test_format_reward_with_whitespace(self):
        """测试标签之间有空白字符的情况"""
        completions = [
            [{"role": "assistant", "content": "<think>思考过程</think>  <long_answer>详细答案</long_answer>  <answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>思考过程</think>\n<long_answer>详细答案</long_answer>\n<answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>思考过程</think>\t<long_answer>详细答案</long_answer>\t<answer>42</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # 正则表达式允许标签之间有空白字符
        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 1.0)  # 格式正确
        self.assertEqual(rewards[1], 1.0)  # 格式正确
        self.assertEqual(rewards[2], 1.0)  # 格式正确
    
    def test_format_reward_multiline_content(self):
        """测试标签内容跨多行的情况"""
        completions = [
            [{"role": "assistant", "content": 
              "<think>这是一个\n多行的\n思考过程</think>"
              "<long_answer>这是一个\n多行的\n详细答案</long_answer>"
              "<answer>42</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # re.DOTALL标志允许.匹配换行符
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)  # 格式正确
    
    def test_format_reward_empty_tags(self):
        """测试空标签的情况"""
        completions = [
            [{"role": "assistant", "content": "<think></think><long_answer></long_answer><answer></answer>"}],
            [{"role": "assistant", "content": "<think>思考</think><long_answer></long_answer><answer>42</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # 空标签也应该被接受（只要格式正确）
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # 格式正确
        self.assertEqual(rewards[1], 1.0)  # 格式正确
    
    def test_format_reward_extra_content_before(self):
        """测试标签前有额外内容的情况"""
        completions = [
            [{"role": "assistant", "content": "额外内容<think>思考过程</think><long_answer>详细答案</long_answer><answer>42</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # fullmatch要求完全匹配，前面有额外内容会失败
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)  # 格式错误
    
    def test_format_reward_extra_content_after(self):
        """测试标签后有额外内容的情况"""
        completions = [
            [{"role": "assistant", "content": "<think>思考过程</think><long_answer>详细答案</long_answer><answer>42</answer>额外内容"}],
        ]
        
        rewards = format_reward(completions)
        
        # fullmatch要求完全匹配，后面有额外内容会失败
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)  # 格式错误
    
    def test_format_reward_extra_content_between(self):
        """测试标签之间有额外内容的情况"""
        completions = [
            [{"role": "assistant", "content": "<think>思考过程</think>额外内容<long_answer>详细答案</long_answer><answer>42</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # 标签之间只允许空白字符，其他内容会导致失败
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)  # 格式错误
    
    def test_format_reward_nested_tags(self):
        """测试嵌套标签的情况"""
        completions = [
            [{"role": "assistant", "content": "<think><inner>嵌套</inner>思考过程</think><long_answer>详细答案</long_answer><answer>42</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # 嵌套标签应该被接受（.*?会匹配内部内容）
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)  # 格式正确
    
    def test_format_reward_duplicate_tags(self):
        """测试重复标签的情况"""
        completions = [
            [{"role": "assistant", "content": "<think>思考1</think><think>思考2</think><long_answer>详细答案</long_answer><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>思考过程</think><long_answer>答案1</long_answer><long_answer>答案2</long_answer><answer>42</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # 注意：由于正则表达式使用.*?（非贪婪匹配），<think>.*?</think>可以匹配包含重复标签的内容
        # 例如：<think>思考1</think><think>思考2</think>会被.*?匹配，因为后面紧跟着<long_answer>
        # 所以重复标签的情况实际上仍然会匹配成功
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # 实际上会匹配成功
        self.assertEqual(rewards[1], 1.0)  # 实际上会匹配成功
    
    def test_format_reward_case_sensitive_tags(self):
        """测试标签大小写敏感性"""
        completions = [
            [{"role": "assistant", "content": "<Think>思考过程</Think><long_answer>详细答案</long_answer><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>思考过程</think><Long_Answer>详细答案</Long_Answer><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>思考过程</think><long_answer>详细答案</long_answer><Answer>42</Answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # 标签是大小写敏感的
        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 0.0)  # 大小写错误
        self.assertEqual(rewards[1], 0.0)  # 大小写错误
        self.assertEqual(rewards[2], 0.0)  # 大小写错误
    
    def test_format_reward_batch_processing(self):
        """测试批量处理"""
        batch_size = 10
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><long_answer>详细答案</long_answer><answer>42</answer>"}]
            for _ in range(batch_size)
        ]
        
        rewards = format_reward(completions)
        
        # 所有格式都应该正确
        self.assertEqual(len(rewards), batch_size)
        self.assertTrue(all(r == 1.0 for r in rewards))
    
    def test_format_reward_mixed_batch(self):
        """测试混合正确和错误格式的批次"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><long_answer>详细答案</long_answer><answer>42</answer>"}],  # 正确
            [{"role": "assistant", "content": "<think>思考</think><answer>42</answer>"}],  # 错误：缺少long_answer
            [{"role": "assistant", "content": "<think>思考</think><long_answer>详细答案</long_answer><answer>42</answer>"}],  # 正确
            [{"role": "assistant", "content": "完全错误的格式"}],  # 错误
            [{"role": "assistant", "content": "<think>思考</think><long_answer>详细答案</long_answer><answer>42</answer>"}],  # 正确
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 5)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 0.0)
        self.assertEqual(rewards[2], 1.0)
        self.assertEqual(rewards[3], 0.0)
        self.assertEqual(rewards[4], 1.0)
    
    def test_format_reward_special_characters_in_content(self):
        """测试内容中包含特殊字符的情况"""
        completions = [
            [{"role": "assistant", "content": "<think>思考: x^2 + y^2 = r^2</think><long_answer>详细答案: α + β = γ</long_answer><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>思考 & 推理</think><long_answer>详细答案 < 100</long_answer><answer>42</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        # 特殊字符在标签内容中应该被接受
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 1.0)
    
    def test_format_reward_very_long_content(self):
        """测试非常长的内容"""
        long_content = "这是一个非常长的内容。" * 100
        completions = [
            [{"role": "assistant", "content": f"<think>{long_content}</think><long_answer>{long_content}</long_answer><answer>42</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)  # 格式正确


class TestFormatRewardIntegration(unittest.TestCase):
    """集成测试：测试format_reward与实际使用场景的集成"""
    
    def test_realistic_generation_scenario(self):
        """测试真实的生成场景"""
        completions = [
            [{"role": "assistant", "content": 
              "<think>首先，我需要理解问题。这是一个简单的数学问题，要求计算2+2。"
              "根据基本的算术运算规则，两个正整数相加的结果是它们的和。</think>"
              "<long_answer>根据基本的算术运算，当我们计算2加2时，我们将两个数值相加。"
              "在十进制系统中，2+2等于4。这是一个基础的数学事实。</long_answer>"
              "<answer>4</answer>"}],
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_quality_control_scenario(self):
        """测试质量控制场景（过滤格式不正确的生成）"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><long_answer>详细答案</long_answer><answer>42</answer>"}],  # 好的生成
            [{"role": "assistant", "content": "直接给出答案：42"}],  # 差的生成
            [{"role": "assistant", "content": "<think>思考</think><long_answer>详细答案</long_answer><answer>100</answer>"}],  # 好的生成
            [{"role": "assistant", "content": "<answer>200</answer>"}],  # 差的生成
        ]
        
        rewards = format_reward(completions)
        
        # 可以用于过滤：只保留reward=1.0的生成
        good_indices = [i for i, r in enumerate(rewards) if r == 1.0]
        self.assertEqual(good_indices, [0, 2])


if __name__ == "__main__":
    unittest.main()
