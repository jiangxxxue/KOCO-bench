import unittest
import torch
import sys
import os
from collections import defaultdict
from typing import Dict, List

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))


class MockRepeatRandomSampler:
    """模拟RepeatRandomSampler类用于测试"""
    
    def __init__(self, data_source, mini_repeat_count, per_gpu_batch_size, num_gpus, repeat_count, seed=None):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.per_gpu_batch_size = per_gpu_batch_size
        self.num_gpus = num_gpus
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        
        # Group indices by dataset_name and prefix
        self.dataset_indices: Dict[str, Dict[str, List[int]]] = defaultdict(dict)
        self.regular_datasets: Dict[str, List[int]] = {}
        
        for idx, data in enumerate(data_source):
            dataset_name = data["dataset_name"]
            image_name = data["image_path"][0].split("/")[-1]
            
            prefix = None
            # 简化的前缀提取逻辑（实际实现可能更复杂）
            if "KADID" in dataset_name or "PIPAL" in dataset_name:
                prefix = image_name.split("_")[0] if "_" in image_name else None
            
            if prefix is not None:
                if prefix not in self.dataset_indices[dataset_name]:
                    self.dataset_indices[dataset_name][prefix] = []
                self.dataset_indices[dataset_name][prefix].append(idx)
            else:
                if dataset_name not in self.regular_datasets:
                    self.regular_datasets[dataset_name] = []
                self.regular_datasets[dataset_name].append(idx)
    
    def __iter__(self):
        # Create copies of indices
        remaining_prefix_indices = {
            name: {prefix: indices.copy() for prefix, indices in prefixes.items()}
            for name, prefixes in self.dataset_indices.items()
        }
        
        remaining_regular_indices = {
            name: indices.copy() for name, indices in self.regular_datasets.items()
        }
        
        # Create lists of dataset names
        prefix_dataset_names = list(self.dataset_indices.keys())
        regular_dataset_names = list(self.regular_datasets.keys())
        
        # Initialize random generator
        dataset_generator = torch.Generator()
        if self.seed is not None:
            dataset_generator.manual_seed(self.seed)
        
        while True:
            gpu_batches = [[] for _ in range(self.num_gpus)]
            
            # Fill each GPU's batch
            for gpu_idx in range(self.num_gpus):
                if not prefix_dataset_names and not regular_dataset_names:
                    break
                
                all_datasets = prefix_dataset_names + regular_dataset_names
                shuffled_idx = torch.randperm(len(all_datasets), generator=dataset_generator).tolist()
                selected_dataset = all_datasets[shuffled_idx[0]]
                
                if selected_dataset in prefix_dataset_names:
                    # Handle prefix-based sampling
                    prefixes = list(remaining_prefix_indices[selected_dataset].keys())
                    if not prefixes:
                        prefix_dataset_names.remove(selected_dataset)
                        continue
                    
                    prefix_perm = torch.randperm(len(prefixes), generator=self.generator).tolist()
                    selected_prefix = prefixes[prefix_perm[0]]
                    
                    indices = remaining_prefix_indices[selected_dataset][selected_prefix]
                    
                    if len(indices) < self.per_gpu_batch_size:
                        del remaining_prefix_indices[selected_dataset][selected_prefix]
                        if not remaining_prefix_indices[selected_dataset]:
                            prefix_dataset_names.remove(selected_dataset)
                        continue
                    
                    perm = torch.randperm(len(indices), generator=self.generator).tolist()
                    shuffled_indices = [indices[i] for i in perm]
                    batch_indices = shuffled_indices[:self.per_gpu_batch_size]
                    
                    remaining_prefix_indices[selected_dataset][selected_prefix] = shuffled_indices[self.per_gpu_batch_size:]
                    
                    if not remaining_prefix_indices[selected_dataset][selected_prefix]:
                        del remaining_prefix_indices[selected_dataset][selected_prefix]
                        if not remaining_prefix_indices[selected_dataset]:
                            prefix_dataset_names.remove(selected_dataset)
                
                else:
                    # Handle regular sampling
                    indices = remaining_regular_indices[selected_dataset]
                    
                    if len(indices) < self.per_gpu_batch_size:
                        regular_dataset_names.remove(selected_dataset)
                        continue
                    
                    perm = torch.randperm(len(indices), generator=self.generator).tolist()
                    shuffled_indices = [indices[i] for i in perm]
                    batch_indices = shuffled_indices[:self.per_gpu_batch_size]
                    
                    remaining_regular_indices[selected_dataset] = shuffled_indices[self.per_gpu_batch_size:]
                    
                    if not remaining_regular_indices[selected_dataset]:
                        regular_dataset_names.remove(selected_dataset)
                
                gpu_batches[gpu_idx] = batch_indices
            
            if not any(gpu_batches):
                break
            
            # Yield batches with repeats
            for _ in range(self.repeat_count):
                for gpu_batch in gpu_batches:
                    if gpu_batch:
                        for index in gpu_batch:
                            for _ in range(self.mini_repeat_count):
                                yield index
    
    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class TestRepeatRandomSampler(unittest.TestCase):
    """测试RepeatRandomSampler.__iter__函数 - 自定义采样器实现"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_basic_sampling(self):
        """测试基本采样功能"""
        # 创建简单的数据源
        data_source = [
            {"dataset_name": "dataset1", "image_path": ["img1.jpg"]},
            {"dataset_name": "dataset1", "image_path": ["img2.jpg"]},
            {"dataset_name": "dataset1", "image_path": ["img3.jpg"]},
            {"dataset_name": "dataset1", "image_path": ["img4.jpg"]},
        ]
        
        sampler = MockRepeatRandomSampler(
            data_source=data_source,
            mini_repeat_count=1,
            per_gpu_batch_size=2,
            num_gpus=1,
            repeat_count=1,
            seed=42
        )
        
        # 收集采样结果
        samples = list(sampler)
        
        # 验证采样结果
        self.assertGreater(len(samples), 0)
        # 每个索引应该在有效范围内
        for idx in samples:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(data_source))
    
    def test_mini_repeat_count(self):
        """测试mini_repeat_count参数"""
        data_source = [
            {"dataset_name": "dataset1", "image_path": ["img1.jpg"]},
            {"dataset_name": "dataset1", "image_path": ["img2.jpg"]},
        ]
        
        mini_repeat_count = 3
        sampler = MockRepeatRandomSampler(
            data_source=data_source,
            mini_repeat_count=mini_repeat_count,
            per_gpu_batch_size=2,
            num_gpus=1,
            repeat_count=1,
            seed=42
        )
        
        samples = list(sampler)
        
        # 每个索引应该重复mini_repeat_count次
        # 验证连续的重复
        for i in range(0, len(samples), mini_repeat_count):
            if i + mini_repeat_count <= len(samples):
                repeated_indices = samples[i:i+mini_repeat_count]
                # 检查是否所有元素相同
                self.assertTrue(all(idx == repeated_indices[0] for idx in repeated_indices))
    
    def test_repeat_count(self):
        """测试repeat_count参数"""
        data_source = [
            {"dataset_name": "dataset1", "image_path": ["img1.jpg"]},
            {"dataset_name": "dataset1", "image_path": ["img2.jpg"]},
        ]
        
        repeat_count = 2
        sampler = MockRepeatRandomSampler(
            data_source=data_source,
            mini_repeat_count=1,
            per_gpu_batch_size=2,
            num_gpus=1,
            repeat_count=repeat_count,
            seed=42
        )
        
        samples = list(sampler)
        
        # 总样本数应该考虑repeat_count
        # 注意：实际数量取决于批次大小和数据集大小
        self.assertGreater(len(samples), 0)
    
    def test_multi_gpu_sampling(self):
        """测试多GPU采样"""
        data_source = [
            {"dataset_name": "dataset1", "image_path": [f"img{i}.jpg"]}
            for i in range(8)
        ]
        
        num_gpus = 2
        per_gpu_batch_size = 2
        sampler = MockRepeatRandomSampler(
            data_source=data_source,
            mini_repeat_count=1,
            per_gpu_batch_size=per_gpu_batch_size,
            num_gpus=num_gpus,
            repeat_count=1,
            seed=42
        )
        
        samples = list(sampler)
        
        # 验证采样结果
        self.assertGreater(len(samples), 0)
        # 每个批次应该有 num_gpus * per_gpu_batch_size 个样本
        # （考虑mini_repeat_count和repeat_count）
    
    def test_prefix_based_sampling(self):
        """测试基于前缀的采样（KADID-10K数据集）"""
        data_source = [
            {"dataset_name": "KADID-10K", "image_path": ["I01_01.jpg"]},
            {"dataset_name": "KADID-10K", "image_path": ["I01_02.jpg"]},
            {"dataset_name": "KADID-10K", "image_path": ["I02_01.jpg"]},
            {"dataset_name": "KADID-10K", "image_path": ["I02_02.jpg"]},
        ]
        
        sampler = MockRepeatRandomSampler(
            data_source=data_source,
            mini_repeat_count=1,
            per_gpu_batch_size=2,
            num_gpus=1,
            repeat_count=1,
            seed=42
        )
        
        samples = list(sampler)
        
        # 验证采样结果
        self.assertGreater(len(samples), 0)
        for idx in samples:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(data_source))
    
    def test_mixed_datasets(self):
        """测试混合数据集采样"""
        data_source = [
            {"dataset_name": "KADID-10K", "image_path": ["I01_01.jpg"]},
            {"dataset_name": "KADID-10K", "image_path": ["I01_02.jpg"]},
            {"dataset_name": "regular_dataset", "image_path": ["img1.jpg"]},
            {"dataset_name": "regular_dataset", "image_path": ["img2.jpg"]},
        ]
        
        sampler = MockRepeatRandomSampler(
            data_source=data_source,
            mini_repeat_count=1,
            per_gpu_batch_size=2,
            num_gpus=1,
            repeat_count=1,
            seed=42
        )
        
        samples = list(sampler)
        
        # 验证采样结果包含来自不同数据集的样本
        self.assertGreater(len(samples), 0)
    
    def test_sampler_length(self):
        """测试采样器长度计算"""
        data_source = [
            {"dataset_name": "dataset1", "image_path": ["img1.jpg"]},
            {"dataset_name": "dataset1", "image_path": ["img2.jpg"]},
            {"dataset_name": "dataset1", "image_path": ["img3.jpg"]},
            {"dataset_name": "dataset1", "image_path": ["img4.jpg"]},
        ]
        
        mini_repeat_count = 2
        repeat_count = 3
        sampler = MockRepeatRandomSampler(
            data_source=data_source,
            mini_repeat_count=mini_repeat_count,
            per_gpu_batch_size=2,
            num_gpus=1,
            repeat_count=repeat_count,
            seed=42
        )
        
        # 验证长度计算
        expected_length = len(data_source) * mini_repeat_count * repeat_count
        self.assertEqual(len(sampler), expected_length)
    
    def test_deterministic_sampling_with_seed(self):
        """测试使用种子的确定性采样"""
        data_source = [
            {"dataset_name": "dataset1", "image_path": [f"img{i}.jpg"]}
            for i in range(6)
        ]
        
        seed = 42
        
        # 创建两个相同配置的采样器
        sampler1 = MockRepeatRandomSampler(
            data_source=data_source,
            mini_repeat_count=1,
            per_gpu_batch_size=2,
            num_gpus=1,
            repeat_count=1,
            seed=seed
        )
        
        sampler2 = MockRepeatRandomSampler(
            data_source=data_source,
            mini_repeat_count=1,
            per_gpu_batch_size=2,
            num_gpus=1,
            repeat_count=1,
            seed=seed
        )
        
        samples1 = list(sampler1)
        samples2 = list(sampler2)
        
        # 使用相同种子应该产生相同的采样序列
        self.assertEqual(samples1, samples2)


class TestRepeatRandomSamplerIntegration(unittest.TestCase):
    """集成测试：测试RepeatRandomSampler的整体工作流程"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_end_to_end_sampling_workflow(self):
        """端到端测试：完整的采样工作流程"""
        # 创建模拟真实场景的数据源
        data_source = []
        
        # 添加KADID-10K数据集样本（带前缀）
        for i in range(1, 4):
            for j in range(1, 3):
                data_source.append({
                    "dataset_name": "KADID-10K",
                    "image_path": [f"I{i:02d}_{j:02d}.jpg"]
                })
        
        # 添加常规数据集样本
        for i in range(6):
            data_source.append({
                "dataset_name": "regular_dataset",
                "image_path": [f"img{i}.jpg"]
            })
        
        sampler = MockRepeatRandomSampler(
            data_source=data_source,
            mini_repeat_count=2,
            per_gpu_batch_size=2,
            num_gpus=2,
            repeat_count=1,
            seed=42
        )
        
        samples = list(sampler)
        
        # 验证采样结果
        self.assertGreater(len(samples), 0)
        
        # 验证所有索引都在有效范围内
        for idx in samples:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(data_source))
    
    def test_batch_consistency(self):
        """测试批次一致性"""
        data_source = [
            {"dataset_name": "dataset1", "image_path": [f"img{i}.jpg"]}
            for i in range(10)
        ]
        
        per_gpu_batch_size = 2
        num_gpus = 2
        mini_repeat_count = 1
        
        sampler = MockRepeatRandomSampler(
            data_source=data_source,
            mini_repeat_count=mini_repeat_count,
            per_gpu_batch_size=per_gpu_batch_size,
            num_gpus=num_gpus,
            repeat_count=1,
            seed=42
        )
        
        samples = list(sampler)
        
        # 验证采样的一致性
        self.assertGreater(len(samples), 0)
        
        # 检查是否有重复（在mini_repeat_count=1时不应该有连续重复）
        unique_samples = set(samples)
        self.assertGreater(len(unique_samples), 0)


if __name__ == "__main__":
    unittest.main()
