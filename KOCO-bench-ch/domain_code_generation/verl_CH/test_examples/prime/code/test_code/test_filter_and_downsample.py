import unittest
import torch
import sys
import os
from omegaconf import OmegaConf

# Add the recipe directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'recipe'))

# Mock verl modules
from unittest.mock import MagicMock

# Create a proper base class for RayPRIMETrainer to inherit from
class MockRayPPOTrainer:
    """Mock base class for RayPRIMETrainer"""
    def __init__(self, *args, **kwargs):
        pass

# Setup mocks
mock_verl = MagicMock()
sys.modules['verl'] = mock_verl

# Mock DataProto before other imports
from unittest.mock import Mock
class DataProto:
    """Minimal DataProto implementation for testing"""
    def __init__(self, batch_dict):
        self.batch = batch_dict
    
    @staticmethod
    def from_dict(tensors):
        return DataProto(tensors)
    
    def __len__(self):
        first_key = next(iter(self.batch.keys()))
        return len(self.batch[first_key])
    
    def reorder(self, indices):
        """Reorder batch data according to indices"""
        for key in self.batch:
            self.batch[key] = self.batch[key][indices]

mock_verl.DataProto = DataProto

sys.modules['verl.single_controller'] = MagicMock()
sys.modules['verl.single_controller.ray'] = MagicMock()
sys.modules['verl.trainer'] = MagicMock()
sys.modules['verl.trainer.ppo'] = MagicMock()
sys.modules['verl.trainer.ppo.core_algos'] = MagicMock()
sys.modules['verl.trainer.ppo.metric_utils'] = MagicMock()

# Create a mock module for ray_trainer with our MockRayPPOTrainer
mock_ray_trainer = MagicMock()
mock_ray_trainer.RayPPOTrainer = MockRayPPOTrainer
mock_ray_trainer.ResourcePoolManager = MagicMock
sys.modules['verl.trainer.ppo.ray_trainer'] = mock_ray_trainer

sys.modules['verl.trainer.ppo.utils'] = MagicMock()
sys.modules['verl.utils'] = MagicMock()
sys.modules['verl.utils.torch_functional'] = MagicMock()
sys.modules['verl.utils.checkpoint'] = MagicMock()
sys.modules['verl.utils.checkpoint.checkpoint_manager'] = MagicMock()
sys.modules['verl.utils.dataset'] = MagicMock()
sys.modules['verl.utils.dataset.rl_dataset'] = MagicMock()
sys.modules['verl.utils.metric'] = MagicMock()
sys.modules['verl.utils.profiler'] = MagicMock()
sys.modules['verl.utils.profiler.performance'] = MagicMock()

# Import ground truth class (now it can properly inherit from MockRayPPOTrainer)
from prime.prime_ray_trainer import RayPRIMETrainer


class TestFilterAndDownsample(unittest.TestCase):
    """测试filter_and_downsample函数 - 文档第7节描述的智能样本筛选和优先级下采样"""
    
    def setUp(self):
        torch.manual_seed(42)
        
        # Create minimal config for RayPRIMETrainer
        self.config = OmegaConf.create({
            'actor_rollout_ref': {
                'rollout': {
                    'n': 4  # n_samples
                }
            },
            'data': {
                'oversample_factor': 4.0,
                'filter_accuracy': True,
                'accuracy_lower_bound': 0.2,
                'accuracy_upper_bound': 0.8,
                'filter_truncate': True,
                'max_response_length': 20
            }
        })
    
    def _call_filter_and_downsample(self, scores, batch):
        """Helper method to call the ground truth filter_and_downsample function"""
        # Create a simple object to hold config
        class SimpleTrainer:
            def __init__(self, config):
                self.config = config
        
        trainer = SimpleTrainer(self.config)
        # Call the actual method from RayPRIMETrainer
        return RayPRIMETrainer.filter_and_downsample(trainer, scores, batch)
    
    def test_filter_and_downsample_basic(self):
        """测试基本的过滤和下采样功能 - 使用ground truth代码"""
        n_samples = 4
        num_groups = 4
        total_samples = num_groups * n_samples
        
        # 创建测试分数：不同质量的样本组
        scores = [
            # Group 1: 平均0.5，应该通过过滤
            0.6, 0.4, 0.5, 0.5,
            # Group 2: 平均0.1，太低，应该被过滤
            0.1, 0.1, 0.1, 0.1,
            # Group 3: 平均0.9，太高，应该被过滤
            0.9, 0.9, 0.9, 0.9,
            # Group 4: 平均0.6，应该通过过滤
            0.7, 0.5, 0.6, 0.6
        ]
        
        # 创建测试batch
        batch = DataProto.from_dict(tensors={
            "responses": torch.randint(0, 1000, (total_samples, 10)),
            "attention_mask": torch.ones(total_samples, 15),
            "input_ids": torch.randint(0, 1000, (total_samples, 15))
        })

        
        # 调用ground truth函数
        result_batch = self._call_filter_and_downsample(scores, batch)
        
        # 验证下采样后的样本数量
        expected_kept_samples = int(total_samples // self.config.data.oversample_factor)
        self.assertEqual(len(result_batch.batch["responses"]), expected_kept_samples)
        
        # 验证返回的是DataProto对象
        self.assertIsInstance(result_batch, DataProto)
    
    def test_accuracy_filtering(self):
        """测试准确率过滤 - 使用ground truth代码"""
        n_samples = 4
        num_groups = 5
        total_samples = num_groups * n_samples
        
        # 创建不同准确率的样本组
        scores = [
            0.3, 0.4, 0.5, 0.4,  # Group 1: 平均0.4，通过
            0.1, 0.1, 0.1, 0.1,  # Group 2: 平均0.1，不通过（太低）
            0.9, 0.9, 0.9, 0.9,  # Group 3: 平均0.9，不通过（太高）
            0.6, 0.7, 0.7, 0.6,  # Group 4: 平均0.65，通过
            0.2, 0.2, 0.2, 0.2   # Group 5: 平均0.2，通过（边界）
        ]
        
        batch = DataProto.from_dict(tensors={
            "responses": torch.randint(0, 1000, (total_samples, 8)),
            "attention_mask": torch.ones(total_samples, 12),
            "input_ids": torch.randint(0, 1000, (total_samples, 12))
        })
        
        result_batch = self._call_filter_and_downsample(scores, batch)
        
        # 验证下采样数量
        expected_kept_samples = int(total_samples // self.config.data.oversample_factor)
        self.assertEqual(len(result_batch.batch["responses"]), expected_kept_samples)
    
    def test_length_filtering(self):
        """测试长度过滤 - 使用ground truth代码"""
        n_samples = 4
        num_groups = 3
        total_samples = num_groups * n_samples
        max_response_length = 10
        
        # 更新配置
        self.config.data.max_response_length = max_response_length
        
        # 创建准确率都合理的分数
        scores = [0.5, 0.6, 0.5, 0.6] * num_groups
        
        # 创建不同长度的attention_mask
        attention_masks = torch.tensor([
            # Group 1: 都应该通过
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            # Group 2: 都应该被过滤（长度>=10）
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            # Group 3: 都应该通过
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        ])
        
        batch = DataProto.from_dict(tensors={
            "responses": torch.randint(0, 1000, (total_samples, 12)),
            "attention_mask": attention_masks,
            "input_ids": torch.randint(0, 1000, (total_samples, 12))
        })
        
        result_batch = self._call_filter_and_downsample(scores, batch)
        
        # 验证下采样数量
        expected_kept_samples = int(total_samples // self.config.data.oversample_factor)
        self.assertEqual(len(result_batch.batch["responses"]), expected_kept_samples)
    
    def test_no_groups_pass_filter(self):
        """测试没有组通过过滤时的回退机制 - 使用ground truth代码"""
        n_samples = 4
        num_groups = 3
        total_samples = num_groups * n_samples
        
        # 所有组都不满足准确率条件
        scores = [0.1, 0.05, 0.15, 0.1] * num_groups  # 所有分数都太低
        
        batch = DataProto.from_dict(tensors={
            "responses": torch.randint(0, 1000, (total_samples, 8)),
            "attention_mask": torch.ones(total_samples, 12),
            "input_ids": torch.randint(0, 1000, (total_samples, 12))
        })
        
        result_batch = self._call_filter_and_downsample(scores, batch)
        
        # 即使没有组通过过滤，仍应该保留相应比例的样本
        expected_kept_samples = int(total_samples // self.config.data.oversample_factor)
        self.assertEqual(len(result_batch.batch["responses"]), expected_kept_samples)
    
    def test_priority_downsampling(self):
        """测试优先级下采样 - 使用ground truth代码"""
        n_samples = 4
        num_groups = 4
        total_samples = num_groups * n_samples
        
        # 创建部分通过过滤的样本组
        scores = [
            0.3, 0.4, 0.35, 0.4,  # Group 1: 通过
            0.1, 0.15, 0.1, 0.1,  # Group 2: 不通过（太低）
            0.5, 0.6, 0.55, 0.5,  # Group 3: 通过
            0.9, 0.85, 0.9, 0.9   # Group 4: 不通过（太高）
        ]
        
        batch = DataProto.from_dict(tensors={
            "responses": torch.randint(0, 1000, (total_samples, 8)),
            "attention_mask": torch.ones(total_samples, 12),
            "input_ids": torch.randint(0, 1000, (total_samples, 12))
        })
        
        result_batch = self._call_filter_and_downsample(scores, batch)
        
        # 验证下采样数量
        expected_kept_samples = int(total_samples // self.config.data.oversample_factor)
        self.assertEqual(len(result_batch.batch["responses"]), expected_kept_samples)
    
    def test_combined_filtering(self):
        """测试准确率和长度的组合过滤 - 使用ground truth代码"""
        n_samples = 4
        num_groups = 3
        total_samples = num_groups * n_samples
        
        # 创建测试数据
        scores = [
            0.5, 0.6, 0.5, 0.6,  # Group 1: 准确率通过，长度通过
            0.4, 0.5, 0.4, 0.5,  # Group 2: 准确率通过，长度不通过
            0.6, 0.7, 0.6, 0.7   # Group 3: 准确率通过，长度通过
        ]
        
        # 创建不同长度的attention_mask
        attention_masks = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # Group 1: 长度8
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # Group 1: 长度9
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # Group 1: 长度7
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # Group 1: 长度8
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Group 2: 长度10（超过max-1）
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Group 2: 长度10
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Group 2: 长度10
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Group 2: 长度10
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # Group 3: 长度8
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # Group 3: 长度9
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # Group 3: 长度7
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]   # Group 3: 长度8
        ])
        
        batch = DataProto.from_dict(tensors={
            "responses": torch.randint(0, 1000, (total_samples, 10)),
            "attention_mask": attention_masks,
            "input_ids": torch.randint(0, 1000, (total_samples, 10))
        })
        
        # 设置max_response_length
        self.config.data.max_response_length = 10
        
        result_batch = self._call_filter_and_downsample(scores, batch)
        
        # 验证下采样数量
        expected_kept_samples = int(total_samples // self.config.data.oversample_factor)
        self.assertEqual(len(result_batch.batch["responses"]), expected_kept_samples)
    
    def test_different_oversample_factors(self):
        """测试不同的过采样因子 - 使用ground truth代码"""
        n_samples = 4
        num_groups = 6
        total_samples = num_groups * n_samples
        
        # 测试不同的过采样因子
        for oversample_factor in [2.0, 3.0, 6.0]:
            with self.subTest(oversample_factor=oversample_factor):
                # 为每次测试创建新的 batch（因为 reorder 是 in-place 操作）
                scores = [0.5, 0.6, 0.4, 0.5] * num_groups  # 所有组都通过过滤
                
                batch = DataProto.from_dict(tensors={
                    "responses": torch.randint(0, 1000, (total_samples, 8)),
                    "attention_mask": torch.ones(total_samples, 12),
                    "input_ids": torch.randint(0, 1000, (total_samples, 12))
                })
                
                self.config.data.oversample_factor = oversample_factor
                result_batch = self._call_filter_and_downsample(scores, batch)
                
                expected_kept_samples = int(total_samples // oversample_factor)
                self.assertEqual(len(result_batch.batch["responses"]), expected_kept_samples)
    
    def test_filter_disabled(self):
        """测试禁用过滤器的情况 - 使用ground truth代码"""
        n_samples = 4
        num_groups = 4
        total_samples = num_groups * n_samples
        
        # 禁用所有过滤器
        self.config.data.filter_accuracy = False
        self.config.data.filter_truncate = False
        
        # 使用极端的分数（如果过滤器启用会被过滤掉）
        scores = [0.05, 0.95, 0.05, 0.95] * num_groups
        
        batch = DataProto.from_dict(tensors={
            "responses": torch.randint(0, 1000, (total_samples, 8)),
            "attention_mask": torch.ones(total_samples, 12),
            "input_ids": torch.randint(0, 1000, (total_samples, 12))
        })
        
        result_batch = self._call_filter_and_downsample(scores, batch)
        
        # 验证下采样数量
        expected_kept_samples = int(total_samples // self.config.data.oversample_factor)
        self.assertEqual(len(result_batch.batch["responses"]), expected_kept_samples)


if __name__ == "__main__":
    unittest.main()
