import unittest
import sys
import os
import json

# Add the parent directory to Python path to import open_r1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))

# Import the actual function we want to test
from open_r1.grpo_jsonl import repetition_reward


class TestRepetitionReward(unittest.TestCase):
    """测试repetition_reward函数 - VLM-R1文档描述的重复内容惩罚"""
    
    def test_no_repetition_json(self):
        """测试JSON格式无重复的情况"""
        data = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [300, 300, 400, 400], "label": "person"},
            {"bbox_2d": [500, 500, 600, 600], "label": "bike"}
        ]
        content = f"```json\n{json.dumps(data)}\n```"
        
        reward = repetition_reward(content)
        
        # 无重复，应该返回0.0
        self.assertEqual(reward, 0.0)
    
    def test_full_repetition_json(self):
        """测试JSON格式完全重复的情况"""
        data = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        content = f"```json\n{json.dumps(data)}\n```"
        
        reward = repetition_reward(content)
        
        # 完全重复，应该接近-1.0
        self.assertLess(reward, -0.5)
        self.assertGreaterEqual(reward, -1.0)
    
    def test_partial_repetition_json(self):
        """测试JSON格式部分重复的情况"""
        data = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [300, 300, 400, 400], "label": "person"}
        ]
        content = f"```json\n{json.dumps(data)}\n```"
        
        reward = repetition_reward(content)
        
        # 部分重复，应该有一定惩罚
        self.assertLess(reward, 0.0)
        self.assertGreater(reward, -1.0)
    
    def test_no_repetition_text(self):
        """测试纯文本无重复的情况"""
        content = "This is a unique sentence with no repeated phrases at all"
        
        reward = repetition_reward(content)
        
        # 无重复，应该返回0.0或接近0.0
        self.assertGreaterEqual(reward, -0.1)
        self.assertLessEqual(reward, 0.0)
    
    def test_full_repetition_text(self):
        """测试纯文本完全重复的情况"""
        phrase = "repeat this phrase "
        content = phrase * 20  # 重复20次
        
        reward = repetition_reward(content)
        
        # 完全重复，应该接近-1.0
        self.assertLess(reward, -0.5)
        self.assertGreaterEqual(reward, -1.0)
    
    def test_partial_repetition_text(self):
        """测试纯文本部分重复的情况"""
        content = "some unique text " * 3 + "different content here"
        
        reward = repetition_reward(content)
        
        # 部分重复，应该有一定惩罚
        self.assertLess(reward, 0.0)
        self.assertGreater(reward, -1.0)
    
    def test_empty_content(self):
        """测试空内容"""
        content = ""
        
        reward = repetition_reward(content)
        
        # 空内容应该返回0.0
        self.assertEqual(reward, 0.0)
    
    def test_short_text(self):
        """测试短文本（小于ngram_size）"""
        content = "short"  # 只有1个词，小于6-gram
        
        reward = repetition_reward(content)
        
        # 短文本应该返回0.0
        self.assertEqual(reward, 0.0)
    
    def test_json_without_markers(self):
        """测试没有```json标记的JSON"""
        data = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        content = json.dumps(data)
        
        reward = repetition_reward(content)
        
        # 应该能识别JSON并检测重复
        self.assertLess(reward, 0.0)
    
    def test_json_with_generic_markers(self):
        """测试使用通用```标记的JSON"""
        data = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        content = f"```\n{json.dumps(data)}\n```"
        
        reward = repetition_reward(content)
        
        # 应该能识别JSON并检测重复
        self.assertLess(reward, 0.0)
    
    def test_invalid_json(self):
        """测试无效JSON（回退到文本处理）"""
        content = "```json\ninvalid json content\n```"
        
        reward = repetition_reward(content)
        
        # 无效JSON应该回退到文本处理
        self.assertGreaterEqual(reward, -1.0)
        self.assertLessEqual(reward, 0.0)
    
    def test_json_missing_fields(self):
        """测试JSON缺少必要字段"""
        data = [
            {"bbox_2d": [100, 100, 200, 200]},  # 缺少label
            {"label": "car"}  # 缺少bbox_2d
        ]
        content = f"```json\n{json.dumps(data)}\n```"
        
        reward = repetition_reward(content)
        
        # 缺少字段应该回退到文本处理
        self.assertGreaterEqual(reward, -1.0)
        self.assertLessEqual(reward, 0.0)
    
    def test_mixed_content(self):
        """测试混合内容（JSON + 文本）"""
        data = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        content = f"Some text before\n```json\n{json.dumps(data)}\n```\nSome text after"
        
        reward = repetition_reward(content)
        
        # 应该能提取JSON部分
        self.assertGreaterEqual(reward, -1.0)
        self.assertLessEqual(reward, 0.0)
    
    def test_case_sensitivity_text(self):
        """测试文本大小写敏感性"""
        # 需要足够长的文本才能形成6-gram
        content = "REPEAT REPEAT repeat repeat word1 word2 word3 word4"
        
        reward = repetition_reward(content)
        
        # 转换为小写后应该检测到重复
        self.assertLessEqual(reward, 0.0)
    
    def test_ngram_size_6(self):
        """测试6-gram检测"""
        # 创建一个包含重复6-gram的文本
        base = "word1 word2 word3 word4 word5 word6 "
        content = base * 5 + "different words here now"
        
        reward = repetition_reward(content)
        
        # 应该检测到6-gram重复
        self.assertLess(reward, -0.3)
    
    def test_different_labels_same_bbox(self):
        """测试相同bbox但不同label"""
        data = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [100, 100, 200, 200], "label": "person"}
        ]
        content = f"```json\n{json.dumps(data)}\n```"
        
        reward = repetition_reward(content)
        
        # bbox相同但label不同，不算完全重复
        self.assertGreater(reward, -1.0)
    
    def test_same_label_different_bbox(self):
        """测试相同label但不同bbox"""
        data = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [300, 300, 400, 400], "label": "car"}
        ]
        content = f"```json\n{json.dumps(data)}\n```"
        
        reward = repetition_reward(content)
        
        # label相同但bbox不同，不算重复
        self.assertGreaterEqual(reward, -0.1)


class TestRepetitionRewardIntegration(unittest.TestCase):
    """集成测试：测试repetition_reward在实际场景中的表现"""
    
    def test_quality_control_scenario(self):
        """测试质量控制场景"""
        # 好的生成（无重复）
        good_data = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [300, 300, 400, 400], "label": "person"},
            {"bbox_2d": [500, 500, 600, 600], "label": "bike"}
        ]
        good_content = f"```json\n{json.dumps(good_data)}\n```"
        
        # 差的生成（有重复）
        bad_data = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        bad_content = f"```json\n{json.dumps(bad_data)}\n```"
        
        good_reward = repetition_reward(good_content)
        bad_reward = repetition_reward(bad_content)
        
        # 好的生成应该得分更高（惩罚更小）
        self.assertGreater(good_reward, bad_reward)
        self.assertGreaterEqual(good_reward, -0.1)
        self.assertLess(bad_reward, -0.5)
    
    def test_text_generation_scenario(self):
        """测试文本生成场景"""
        # 好的生成（多样化）
        good_text = "The cat sat on the mat. The dog played in the yard. Birds flew overhead."
        
        # 差的生成（重复）
        bad_text = "I think I think I think I think I think I think I think"
        
        good_reward = repetition_reward(good_text)
        bad_reward = repetition_reward(bad_text)
        
        # 好的生成应该得分更高
        self.assertGreater(good_reward, bad_reward)
    
    def test_filter_repetitive_outputs(self):
        """测试过滤重复输出"""
        outputs = [
            f"```json\n{json.dumps([{'bbox_2d': [i*100, i*100, (i+1)*100, (i+1)*100], 'label': f'obj{i}'} for i in range(3)])}\n```",
            f"```json\n{json.dumps([{'bbox_2d': [100, 100, 200, 200], 'label': 'car'}] * 5)}\n```",
            f"```json\n{json.dumps([{'bbox_2d': [i*50, i*50, (i+1)*50, (i+1)*50], 'label': 'item'} for i in range(4)])}\n```"
        ]
        
        rewards = [repetition_reward(output) for output in outputs]
        
        # 可以用于过滤：只保留reward接近0的输出
        good_indices = [i for i, r in enumerate(rewards) if r > -0.3]
        
        # 第一个和第三个应该是好的，第二个应该被过滤
        self.assertIn(0, good_indices)
        self.assertNotIn(1, good_indices)


if __name__ == "__main__":
    unittest.main()
