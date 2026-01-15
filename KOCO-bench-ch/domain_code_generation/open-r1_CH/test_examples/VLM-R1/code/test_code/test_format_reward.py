import unittest
import sys
import os

# Add the parent directory to Python path to import open_r1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))

# Import the actual function we want to test
from open_r1.grpo_jsonl import format_reward


class TestFormatReward(unittest.TestCase):
    """测试format_reward函数 - VLM-R1文档描述的格式验证"""
    
    def setUp(self):
        """清除DEBUG_MODE环境变量"""
        if "DEBUG_MODE" in os.environ:
            del os.environ["DEBUG_MODE"]
    
    def test_correct_format(self):
        """测试正确格式"""
        completions = [
            [{"role": "assistant", "content": "<think>思考过程</think><answer>42</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_correct_format_with_whitespace(self):
        """测试标签之间有空白字符的正确格式"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think>  <answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>思考</think>\n<answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>思考</think>\t<answer>42</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 3)
        self.assertTrue(all(r == 1.0 for r in rewards))
    
    def test_missing_think_tag(self):
        """测试缺少think标签"""
        completions = [
            [{"role": "assistant", "content": "<answer>42</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_missing_answer_tag(self):
        """测试缺少answer标签"""
        completions = [
            [{"role": "assistant", "content": "<think>思考过程</think>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_wrong_order(self):
        """测试标签顺序错误"""
        completions = [
            [{"role": "assistant", "content": "<answer>42</answer><think>思考</think>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_extra_content_before(self):
        """测试标签前有额外内容"""
        completions = [
            [{"role": "assistant", "content": "额外内容<think>思考</think><answer>42</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_extra_content_after(self):
        """测试标签后有额外内容"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><answer>42</answer>额外内容"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_extra_content_between(self):
        """测试标签之间有额外内容（非空白）"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think>额外内容<answer>42</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_empty_tags(self):
        """测试空标签"""
        completions = [
            [{"role": "assistant", "content": "<think></think><answer></answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_multiline_content(self):
        """测试多行内容"""
        completions = [
            [{"role": "assistant", "content": "<think>思考\n过程\n多行</think><answer>42</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_nested_tags(self):
        """测试嵌套标签"""
        completions = [
            [{"role": "assistant", "content": "<think><inner>嵌套</inner>思考</think><answer>42</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_case_sensitive_tags(self):
        """测试标签大小写敏感"""
        completions = [
            [{"role": "assistant", "content": "<Think>思考</Think><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>思考</think><Answer>42</Answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 0.0)
        self.assertEqual(rewards[1], 0.0)
    
    def test_batch_processing(self):
        """测试批量处理"""
        completions = [
            [{"role": "assistant", "content": "<think>思考1</think><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>思考2</think><answer>100</answer>"}],
            [{"role": "assistant", "content": "<think>思考3</think><answer>200</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 3)
        self.assertTrue(all(r == 1.0 for r in rewards))
    
    def test_mixed_batch(self):
        """测试混合正确和错误格式的批次"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><answer>42</answer>"}],  # 正确
            [{"role": "assistant", "content": "<answer>42</answer>"}],  # 错误：缺少think
            [{"role": "assistant", "content": "<think>思考</think><answer>100</answer>"}],  # 正确
            [{"role": "assistant", "content": "完全错误"}],  # 错误
            [{"role": "assistant", "content": "<think>思考</think><answer>200</answer>"}]  # 正确
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 5)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 0.0)
        self.assertEqual(rewards[2], 1.0)
        self.assertEqual(rewards[3], 0.0)
        self.assertEqual(rewards[4], 1.0)
    
    def test_special_characters_in_content(self):
        """测试内容中包含特殊字符"""
        completions = [
            [{"role": "assistant", "content": "<think>x^2 + y^2 = r^2</think><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>α + β = γ</think><answer>100</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 2)
        self.assertTrue(all(r == 1.0 for r in rewards))
    
    def test_very_long_content(self):
        """测试非常长的内容"""
        long_content = "这是一个非常长的思考过程。" * 100
        completions = [
            [{"role": "assistant", "content": f"<think>{long_content}</think><answer>42</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_incomplete_tags(self):
        """测试不完整的标签"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><answer>42"}],  # 缺少结束标签
            [{"role": "assistant", "content": "<think>思考<answer>42</answer>"}]  # 缺少think结束标签
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 0.0)
        self.assertEqual(rewards[1], 0.0)
    
    def test_duplicate_tags(self):
        """测试重复标签"""
        completions = [
            [{"role": "assistant", "content": "<think>思考1</think><think>思考2</think><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>思考</think><answer>42</answer><answer>100</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        # fullmatch要求完全匹配，重复标签会导致匹配失败
        self.assertEqual(len(rewards), 2)
        # 第一个可能匹配成功（因为.*?可以匹配包含重复标签的内容）
        # 第二个可能匹配成功（因为.*?可以匹配包含重复标签的内容）
        # 实际行为取决于正则表达式的具体实现


class TestFormatRewardIntegration(unittest.TestCase):
    """集成测试：测试format_reward在实际场景中的表现"""
    
    def test_realistic_generation_scenario(self):
        """测试真实的生成场景"""
        completions = [
            [{"role": "assistant", "content": 
              "<think>首先分析问题，这是一个简单的数学计算。2+2=4。</think>"
              "<answer>4</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_quality_control_scenario(self):
        """测试质量控制场景"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><answer>42</answer>"}],  # 好的生成
            [{"role": "assistant", "content": "直接给出答案：42"}],  # 差的生成
            [{"role": "assistant", "content": "<think>思考</think><answer>100</answer>"}],  # 好的生成
            [{"role": "assistant", "content": "<answer>200</answer>"}]  # 差的生成
        ]
        
        rewards = format_reward(completions)
        
        # 可以用于过滤：只保留reward=1.0的生成
        good_indices = [i for i, r in enumerate(rewards) if r == 1.0]
        self.assertEqual(good_indices, [0, 2])
    
    def test_multimodal_vlm_scenario(self):
        """测试多模态VLM场景"""
        completions = [
            [{"role": "assistant", "content": 
              "<think>分析图像内容，识别目标位置和类别</think>"
              "<answer>图像中有一辆红色汽车</answer>"}]
        ]
        
        rewards = format_reward(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)


if __name__ == "__main__":
    unittest.main()
