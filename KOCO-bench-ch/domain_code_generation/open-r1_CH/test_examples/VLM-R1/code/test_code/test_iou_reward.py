import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock
from PIL import Image
import tempfile

# Add the parent directory to Python path to import open_r1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))

# Import the actual function we want to test
from open_r1.vlm_modules.qwen_module import Qwen2VLModule


class TestIoUReward(unittest.TestCase):
    """测试iou_reward函数 - VLM-R1文档描述的IoU奖励计算"""
    
    def setUp(self):
        """创建测试用的临时图像"""
        self.temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        # 创建一个800x600的测试图像
        img = Image.new('RGB', (800, 600), color='white')
        img.save(self.temp_image.name)
        self.temp_image.close()
        
        # 清除DEBUG_MODE环境变量
        if "DEBUG_MODE" in os.environ:
            del os.environ["DEBUG_MODE"]
    
    def tearDown(self):
        """清理临时文件"""
        if os.path.exists(self.temp_image.name):
            os.remove(self.temp_image.name)
    
    def test_iou_reward_perfect_match(self):
        """测试完全匹配的边界框"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><answer>[100, 100, 200, 200]</answer>"}]
        ]
        solution = [f"<answer>{json.dumps([100, 100, 200, 200])}</answer>"]
        
        # image_grid_thw: [1, 43, 57] -> input_height=602, input_width=798
        # 原始图像: 800x600
        # 缩放后的bbox应该与原始bbox几乎相同
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],
            "image_path": [[self.temp_image.name]],
            "problem": ["test problem"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        self.assertEqual(len(rewards), 1)
        # 由于缩放因子接近1，IoU应该接近1.0
        self.assertGreater(rewards[0], 0.95)
    
    def test_iou_reward_partial_overlap(self):
        """测试部分重叠的边界框"""
        completions = [
            [{"role": "assistant", "content": "<answer>[100, 100, 200, 200]</answer>"}]
        ]
        # 真实框与预测框有50%重叠
        solution = [f"<answer>{json.dumps([150, 150, 250, 250])}</answer>"]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],
            "image_path": [[self.temp_image.name]],
            "problem": ["test problem"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        self.assertEqual(len(rewards), 1)
        # 部分重叠应该有一定的IoU值
        self.assertGreater(rewards[0], 0.0)
        self.assertLess(rewards[0], 1.0)
    
    def test_iou_reward_no_overlap(self):
        """测试完全不重叠的边界框"""
        completions = [
            [{"role": "assistant", "content": "<answer>[100, 100, 200, 200]</answer>"}]
        ]
        # 真实框与预测框完全不重叠
        solution = [f"<answer>{json.dumps([300, 300, 400, 400])}</answer>"]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],
            "image_path": [[self.temp_image.name]],
            "problem": ["test problem"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        self.assertEqual(len(rewards), 1)
        # 完全不重叠，IoU应该为0
        self.assertEqual(rewards[0], 0.0)
    
    def test_iou_reward_bbox_scaling(self):
        """测试边界框坐标缩放"""
        # 测试不同的image_grid_thw值
        # 使用与图像尺寸(800x600)接近的grid尺寸以减少缩放误差
        completions = [
            [{"role": "assistant", "content": "<answer>[100, 100, 200, 200]</answer>"}]
        ]
        solution = [f"<answer>{json.dumps([100, 100, 200, 200])}</answer>"]
        
        # 使用接近实际图像尺寸的grid: 800/14≈57, 600/14≈43
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],  # input_height=602, input_width=798
            "image_path": [[self.temp_image.name]],
            "problem": ["test problem"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        self.assertEqual(len(rewards), 1)
        # 缩放因子接近1时，IoU应该很高
        self.assertGreater(rewards[0], 0.9)
    
    def test_iou_reward_missing_answer_tag(self):
        """测试缺少answer标签的情况"""
        completions = [
            [{"role": "assistant", "content": "[100, 100, 200, 200]"}]  # 没有answer标签
        ]
        solution = [f"<answer>{json.dumps([100, 100, 200, 200])}</answer>"]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],
            "image_path": [[self.temp_image.name]],
            "problem": ["test problem"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        self.assertEqual(len(rewards), 1)
        # 无法提取answer，应该返回0.0
        self.assertEqual(rewards[0], 0.0)
    
    def test_iou_reward_invalid_bbox_format(self):
        """测试无效的边界框格式"""
        completions = [
            [{"role": "assistant", "content": "<answer>invalid bbox</answer>"}]
        ]
        solution = [f"<answer>{json.dumps([100, 100, 200, 200])}</answer>"]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],
            "image_path": [[self.temp_image.name]],
            "problem": ["test problem"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        self.assertEqual(len(rewards), 1)
        # 无效格式应该返回0.0
        self.assertEqual(rewards[0], 0.0)
    
    def test_iou_reward_batch_processing(self):
        """测试批量处理"""
        completions = [
            [{"role": "assistant", "content": "<answer>[100, 100, 200, 200]</answer>"}],
            [{"role": "assistant", "content": "<answer>[150, 150, 250, 250]</answer>"}],
            [{"role": "assistant", "content": "<answer>[300, 300, 400, 400]</answer>"}]
        ]
        solution = [
            f"<answer>{json.dumps([100, 100, 200, 200])}</answer>",
            f"<answer>{json.dumps([150, 150, 250, 250])}</answer>",
            f"<answer>{json.dumps([100, 100, 200, 200])}</answer>"  # 与第三个预测不匹配
        ]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57], [1, 43, 57], [1, 43, 57]],
            "image_path": [[self.temp_image.name], [self.temp_image.name], [self.temp_image.name]],
            "problem": ["test1", "test2", "test3"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        self.assertEqual(len(rewards), 3)
        # 前两个应该有高IoU，第三个应该有低IoU
        self.assertGreater(rewards[0], 0.9)
        self.assertGreater(rewards[1], 0.9)
        self.assertLess(rewards[2], 0.5)
    
    def test_iou_reward_edge_bbox(self):
        """测试边缘边界框"""
        # 测试在图像边缘的边界框
        completions = [
            [{"role": "assistant", "content": "<answer>[0, 0, 100, 100]</answer>"}]
        ]
        solution = [f"<answer>{json.dumps([0, 0, 100, 100])}</answer>"]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],
            "image_path": [[self.temp_image.name]],
            "problem": ["test problem"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        self.assertEqual(len(rewards), 1)
        # 边缘框也应该正确计算IoU
        self.assertGreater(rewards[0], 0.9)
    
    def test_iou_reward_large_bbox(self):
        """测试大边界框"""
        # 测试覆盖整个图像的边界框
        completions = [
            [{"role": "assistant", "content": "<answer>[0, 0, 800, 600]</answer>"}]
        ]
        solution = [f"<answer>{json.dumps([0, 0, 800, 600])}</answer>"]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],
            "image_path": [[self.temp_image.name]],
            "problem": ["test problem"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        self.assertEqual(len(rewards), 1)
        self.assertGreater(rewards[0], 0.9)
    
    def test_iou_reward_small_bbox(self):
        """测试小边界框"""
        # 测试非常小的边界框
        completions = [
            [{"role": "assistant", "content": "<answer>[100, 100, 110, 110]</answer>"}]
        ]
        solution = [f"<answer>{json.dumps([100, 100, 110, 110])}</answer>"]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],
            "image_path": [[self.temp_image.name]],
            "problem": ["test problem"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        self.assertEqual(len(rewards), 1)
        # 小框也应该正确计算IoU
        self.assertGreater(rewards[0], 0.8)
    
    def test_iou_reward_with_think_tag(self):
        """测试包含think标签的完整格式"""
        completions = [
            [{"role": "assistant", "content": "<think>分析图像，目标在中心位置</think><answer>[100, 100, 200, 200]</answer>"}]
        ]
        solution = [f"<answer>{json.dumps([100, 100, 200, 200])}</answer>"]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],
            "image_path": [[self.temp_image.name]],
            "problem": ["test problem"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        self.assertEqual(len(rewards), 1)
        self.assertGreater(rewards[0], 0.9)
    
    def test_iou_reward_different_image_sizes(self):
        """测试不同图像尺寸"""
        # 创建不同尺寸的图像
        temp_image2 = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        img2 = Image.new('RGB', (1920, 1080), color='white')
        img2.save(temp_image2.name)
        temp_image2.close()
        
        try:
            # 使用与图像尺寸匹配的grid和bbox
            # 1920/14≈137, 1080/14≈77
            completions = [
                [{"role": "assistant", "content": "<answer>[100, 100, 200, 200]</answer>"}]
            ]
            solution = [f"<answer>{json.dumps([100, 100, 200, 200])}</answer>"]
            
            kwargs = {
                "image_grid_thw": [[1, 77, 137]],  # 匹配1920x1080的grid
                "image_path": [[temp_image2.name]],
                "problem": ["test problem"]
            }
            
            rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
            
            self.assertEqual(len(rewards), 1)
            # 使用匹配的grid尺寸，IoU应该很高
            self.assertGreater(rewards[0], 0.9)
        finally:
            os.remove(temp_image2.name)


class TestIoURewardIntegration(unittest.TestCase):
    """集成测试：测试iou_reward在实际场景中的表现"""
    
    def setUp(self):
        """创建测试用的临时图像"""
        self.temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        img = Image.new('RGB', (800, 600), color='white')
        img.save(self.temp_image.name)
        self.temp_image.close()
    
    def tearDown(self):
        """清理临时文件"""
        if os.path.exists(self.temp_image.name):
            os.remove(self.temp_image.name)
    
    def test_rec_task_scenario(self):
        """测试REC任务场景"""
        # 模拟一个真实的REC任务
        completions = [
            [{"role": "assistant", "content": 
              "<think>图像中的目标位于中心偏左位置，大约占据图像的1/4区域</think>"
              "<answer>[200, 150, 400, 350]</answer>"}]
        ]
        solution = [f"<answer>{json.dumps([200, 150, 400, 350])}</answer>"]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57]],
            "image_path": [[self.temp_image.name]],
            "problem": ["请在图像中定位红色的汽车"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        self.assertEqual(len(rewards), 1)
        self.assertGreater(rewards[0], 0.9)
    
    def test_quality_control_scenario(self):
        """测试质量控制场景"""
        # 测试不同质量的预测
        completions = [
            [{"role": "assistant", "content": "<answer>[100, 100, 200, 200]</answer>"}],  # 好的预测
            [{"role": "assistant", "content": "<answer>[90, 90, 210, 210]</answer>"}],   # 稍差的预测
            [{"role": "assistant", "content": "<answer>[300, 300, 400, 400]</answer>"}], # 差的预测
        ]
        solution = [
            f"<answer>{json.dumps([100, 100, 200, 200])}</answer>",
            f"<answer>{json.dumps([100, 100, 200, 200])}</answer>",
            f"<answer>{json.dumps([100, 100, 200, 200])}</answer>"
        ]
        
        kwargs = {
            "image_grid_thw": [[1, 43, 57], [1, 43, 57], [1, 43, 57]],
            "image_path": [[self.temp_image.name], [self.temp_image.name], [self.temp_image.name]],
            "problem": ["test1", "test2", "test3"]
        }
        
        rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        
        # 可以用于过滤：只保留IoU高的预测
        good_indices = [i for i, r in enumerate(rewards) if r > 0.7]
        self.assertIn(0, good_indices)  # 第一个应该是好的
        self.assertNotIn(2, good_indices)  # 第三个应该被过滤


if __name__ == "__main__":
    unittest.main()
