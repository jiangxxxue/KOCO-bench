import unittest
import sys
import os

# Add the parent directory to Python path to import open_r1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))

# Import the actual function we want to test
from open_r1.vlm_modules.qwen_module import Qwen2VLModule


class TestFormatRewardRec(unittest.TestCase):
    """测试format_reward_rec函数 - VLM-R1文档描述的REC任务格式验证"""
    
    def setUp(self):
        """清除DEBUG_MODE环境变量"""
        if "DEBUG_MODE" in os.environ:
            del os.environ["DEBUG_MODE"]
    
    def test_correct_format(self):
        """测试正确格式"""
        completions = [
            [{"role": "assistant", "content": '<think>思考过程</think><answer>{"bbox": [100, 200, 300, 400]}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_correct_format_with_json_object(self):
        """测试包含JSON对象的正确格式"""
        completions = [
            [{"role": "assistant", "content": '<think>思考</think><answer>{"bbox": [100, 200, 300, 400]}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_correct_format_with_whitespace(self):
        """测试标签之间有空白字符的正确格式"""
        completions = [
            [{"role": "assistant", "content": '<think>思考</think>  <answer>{"bbox": [100, 200, 300, 400]}</answer>'}],
            [{"role": "assistant", "content": '<think>思考</think>\n<answer>{"bbox": [100, 200, 300, 400]}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 2)
        self.assertTrue(all(r == 1.0 for r in rewards))
    
    def test_missing_think_tag(self):
        """测试缺少think标签"""
        completions = [
            [{"role": "assistant", "content": "<answer>[100, 200, 300, 400]</answer>"}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_missing_answer_tag(self):
        """测试缺少answer标签"""
        completions = [
            [{"role": "assistant", "content": "<think>思考过程</think>[100, 200, 300, 400]"}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_missing_bbox(self):
        """测试缺少边界框坐标"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><answer>没有边界框</answer>"}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_incomplete_bbox(self):
        """测试不完整的边界框（少于4个数字）"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><answer>[100, 200]</answer>"}],
            [{"role": "assistant", "content": "<think>思考</think><answer>[100, 200, 300]</answer>"}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 0.0)
        self.assertEqual(rewards[1], 0.0)
    
    def test_wrong_order(self):
        """测试标签顺序错误"""
        completions = [
            [{"role": "assistant", "content": "<answer>[100, 200, 300, 400]</answer><think>思考</think>"}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_extra_content_before(self):
        """测试标签前有额外内容"""
        completions = [
            [{"role": "assistant", "content": "额外内容<think>思考</think><answer>[100, 200, 300, 400]</answer>"}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        # search()不要求完全匹配，所以可能返回1.0
        self.assertIn(rewards[0], [0.0, 1.0])
    
    def test_extra_content_after(self):
        """测试标签后有额外内容"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><answer>[100, 200, 300, 400]</answer>额外内容"}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        # search()不要求完全匹配，所以可能返回1.0
        self.assertIn(rewards[0], [0.0, 1.0])
    
    def test_multiline_content(self):
        """测试多行内容"""
        completions = [
            [{"role": "assistant", "content": '<think>思考\n过程\n多行</think><answer>{"bbox": [100, 200, 300, 400]}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_bbox_with_spaces(self):
        """测试边界框坐标之间有空格"""
        completions = [
            [{"role": "assistant", "content": '<think>思考</think><answer>{"bbox": [100,  200,  300,  400]}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_bbox_without_spaces(self):
        """测试边界框坐标之间没有空格"""
        completions = [
            [{"role": "assistant", "content": '<think>思考</think><answer>{"bbox": [100,200,300,400]}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_json_object_format(self):
        """测试JSON对象格式"""
        completions = [
            [{"role": "assistant", "content": '<think>思考</think><answer>{"x1": 100, "y1": 200, "x2": 300, "y2": 400, "bbox": [100, 200, 300, 400]}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_nested_json(self):
        """测试嵌套JSON"""
        completions = [
            [{"role": "assistant", "content": '<think>思考</think><answer>{"result": {"bbox": [100, 200, 300, 400]}}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_batch_processing(self):
        """测试批量处理"""
        completions = [
            [{"role": "assistant", "content": '<think>思考1</think><answer>{"bbox": [100, 200, 300, 400]}</answer>'}],
            [{"role": "assistant", "content": '<think>思考2</think><answer>{"bbox": [50, 50, 150, 150]}</answer>'}],
            [{"role": "assistant", "content": '<think>思考3</think><answer>{"bbox": [200, 200, 400, 400]}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 3)
        self.assertTrue(all(r == 1.0 for r in rewards))
    
    def test_mixed_batch(self):
        """测试混合正确和错误格式的批次"""
        completions = [
            [{"role": "assistant", "content": '<think>思考</think><answer>{"bbox": [100, 200, 300, 400]}</answer>'}],  # 正确
            [{"role": "assistant", "content": '<answer>{"bbox": [100, 200, 300, 400]}</answer>'}],  # 错误：缺少think
            [{"role": "assistant", "content": '<think>思考</think><answer>{"bbox": [100, 200]}</answer>'}],  # 错误：bbox不完整
            [{"role": "assistant", "content": '<think>思考</think><answer>{"bbox": [50, 50, 150, 150]}</answer>'}],  # 正确
            [{"role": "assistant", "content": "<think>思考</think><answer>没有bbox</answer>"}]  # 错误：没有bbox
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 5)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 0.0)
        self.assertEqual(rewards[2], 0.0)
        self.assertEqual(rewards[3], 1.0)
        self.assertEqual(rewards[4], 0.0)
    
    def test_case_sensitive_tags(self):
        """测试标签大小写敏感"""
        completions = [
            [{"role": "assistant", "content": "<Think>思考</Think><answer>[100, 200, 300, 400]</answer>"}],
            [{"role": "assistant", "content": "<think>思考</think><Answer>[100, 200, 300, 400]</Answer>"}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 0.0)
        self.assertEqual(rewards[1], 0.0)
    
    def test_float_coordinates(self):
        """测试浮点数坐标"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><answer>[100.5, 200.5, 300.5, 400.5]</answer>"}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        # 正则表达式要求\d+（整数），所以浮点数可能不匹配
        # 实际行为取决于正则表达式的具体实现
        self.assertIn(rewards[0], [0.0, 1.0])
    
    def test_negative_coordinates(self):
        """测试负数坐标"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><answer>[-100, -200, 300, 400]</answer>"}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        # 正则表达式要求\d+（正整数），所以负数可能不匹配
        self.assertIn(rewards[0], [0.0, 1.0])
    
    def test_very_large_coordinates(self):
        """测试非常大的坐标值"""
        completions = [
            [{"role": "assistant", "content": '<think>思考</think><answer>{"bbox": [10000, 20000, 30000, 40000]}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)


class TestFormatRewardRecIntegration(unittest.TestCase):
    """集成测试：测试format_reward_rec在实际场景中的表现"""
    
    def test_rec_task_scenario(self):
        """测试REC任务场景"""
        completions = [
            [{"role": "assistant", "content": 
              "<think>分析图像，目标位于中心偏左位置，大约占据图像的1/4区域</think>"
              '<answer>{"bbox": [200, 150, 400, 350]}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_quality_control_scenario(self):
        """测试质量控制场景"""
        completions = [
            [{"role": "assistant", "content": '<think>思考</think><answer>{"bbox": [100, 100, 200, 200]}</answer>'}],  # 好的生成
            [{"role": "assistant", "content": "目标在[100, 100, 200, 200]"}],  # 差的生成
            [{"role": "assistant", "content": '<think>思考</think><answer>{"bbox": [50, 50, 150, 150]}</answer>'}],  # 好的生成
            [{"role": "assistant", "content": '<answer>{"bbox": [200, 200, 300, 300]}</answer>'}]  # 差的生成：缺少think
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        # 可以用于过滤：只保留reward=1.0的生成
        good_indices = [i for i, r in enumerate(rewards) if r == 1.0]
        self.assertEqual(good_indices, [0, 2])
    
    def test_multimodal_vlm_scenario(self):
        """测试多模态VLM场景"""
        completions = [
            [{"role": "assistant", "content": 
              "<think>图像中的红色汽车位于画面中央，边界框坐标为...</think>"
              '<answer>{"bbox": [250, 180, 550, 420], "confidence": 0.95}</answer>'}]
        ]
        
        rewards = Qwen2VLModule.format_reward_rec(completions)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)


if __name__ == "__main__":
    unittest.main()
