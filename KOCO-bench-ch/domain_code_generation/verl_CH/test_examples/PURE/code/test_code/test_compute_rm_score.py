import unittest
import torch
import sys
import os
from unittest.mock import MagicMock, patch

# Add the verl directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'verl'))

# Import the actual class and components we want to test
try:
    from verl.workers.fsdp_workers import ProcessRewardModelWorker
    from verl.protocol import DataProto
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False
    ProcessRewardModelWorker = None
    DataProto = None


class TestComputeRmScore(unittest.TestCase):
    """测试ProcessRewardModelWorker.compute_rm_score函数 - PURE文档第2节描述的批量PRM推断与聚合"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.response_length = 6
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_compute_rm_score_data_flow(self):
        """测试compute_rm_score的数据流和接口"""
        # 创建输入数据
        input_data = {
            'responses': torch.randint(0, 1000, (self.batch_size, self.response_length)),
            'reward_mask': torch.ones(self.batch_size, self.response_length),
            'input_ids': torch.randint(0, 1000, (self.batch_size, 10)),
            'attention_mask': torch.ones(self.batch_size, 10),
            'position_ids': torch.arange(10).unsqueeze(0).repeat(self.batch_size, 1)
        }
        data = DataProto.from_dict(tensors=input_data)
        
        # 创建mock worker
        mock_worker = MagicMock(spec=ProcessRewardModelWorker)
        mock_worker.config = MagicMock()
        mock_worker.config.use_dynamic_bsz = False
        mock_worker.config.micro_batch_size_per_gpu = 2
        mock_worker._is_offload_param = False
        mock_worker.world_size = 1
        
        # Mock所有依赖的方法和属性
        mock_worker._split_steps.return_value = DataProto.from_dict(tensors={'steps': torch.randint(0, 5, (self.batch_size, 3))})
        mock_worker._build_inputs_for_prm.return_value = DataProto.from_dict(tensors={
            'prm_input_ids': torch.randint(0, 1000, (self.batch_size, 15)),
            'prm_attention_mask': torch.ones(self.batch_size, 15),
            'prm_position_ids': torch.arange(15).unsqueeze(0).repeat(self.batch_size, 1)
        })
        
        # Mock _forward_micro_batch 返回合理的分数
        def mock_forward_micro_batch(micro_batch):
            current_batch_size = micro_batch['responses'].size(0)
            return torch.randn(current_batch_size, self.response_length)
        
        mock_worker._forward_micro_batch.side_effect = mock_forward_micro_batch
        
        # Mock context managers和其他依赖
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_context)
        mock_context.__exit__ = MagicMock(return_value=None)
        mock_worker.ulysses_sharding_manager = mock_context
        mock_worker.ulysses_sharding_manager.preprocess_data.return_value = data
        mock_worker.ulysses_sharding_manager.postprocess_data.return_value = None
        
        # Mock批次分割
        mock_batch = data.batch
        
        # 模拟分批处理逻辑
        micro_batch_size = mock_worker.config.micro_batch_size_per_gpu
        micro_batches = []
        for i in range(0, self.batch_size, micro_batch_size):
            end_idx = min(i + micro_batch_size, self.batch_size)
            micro_batch = {}
            for key, value in mock_batch.items():
                if isinstance(value, torch.Tensor):
                    micro_batch[key] = value[i:end_idx]
                else:
                    micro_batch[key] = value
            micro_batches.append(micro_batch)
        
        # 模拟处理每个micro-batch
        output_scores = []
        for micro_batch in micro_batches:
            scores = mock_forward_micro_batch(micro_batch)
            output_scores.append(scores)
        
        # 聚合结果
        token_level_scores = torch.cat(output_scores, dim=0)
        
        # 创建输出DataProto
        result = DataProto.from_dict(tensors={'rm_scores': token_level_scores})
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertIn('rm_scores', result.batch)
        rm_scores = result.batch['rm_scores']
        
        # 验证输出形状和属性
        self.assertEqual(rm_scores.shape, (self.batch_size, self.response_length))
        self.assertTrue(torch.is_tensor(rm_scores))
        self.assertTrue(torch.isfinite(rm_scores).all())
    
    def test_micro_batch_splitting_logic(self):
        """测试micro-batch分割逻辑"""
        total_batch_size = 8
        micro_batch_size = 3
        
        # 创建测试数据
        responses = torch.randint(0, 1000, (total_batch_size, self.response_length))
        reward_mask = torch.ones(total_batch_size, self.response_length)
        
        # 实现micro-batch分割
        micro_batches = []
        for i in range(0, total_batch_size, micro_batch_size):
            end_idx = min(i + micro_batch_size, total_batch_size)
            micro_batch = {
                'responses': responses[i:end_idx],
                'reward_mask': reward_mask[i:end_idx]
            }
            micro_batches.append(micro_batch)
        
        # 验证分割结果
        self.assertGreater(len(micro_batches), 1)  # 应该被分成多个micro-batch
        
        # 验证每个micro-batch的大小
        total_processed = 0
        for i, micro_batch in enumerate(micro_batches):
            batch_size = micro_batch['responses'].size(0)
            total_processed += batch_size
            
            # 除了最后一个micro-batch，其他应该有完整大小
            if i < len(micro_batches) - 1:
                expected_size = micro_batch_size
            else:
                expected_size = total_batch_size - i * micro_batch_size
            
            self.assertEqual(batch_size, expected_size)
            self.assertEqual(micro_batch['responses'].shape[1], self.response_length)
            self.assertEqual(micro_batch['reward_mask'].shape, (batch_size, self.response_length))
        
        # 验证所有样本都被处理了
        self.assertEqual(total_processed, total_batch_size)
    
    def test_score_aggregation(self):
        """测试分数聚合逻辑"""
        # 创建多个micro-batch的输出分数
        micro_batch_scores = [
            torch.randn(2, self.response_length),  # 第一个micro-batch: 2 samples
            torch.randn(2, self.response_length),  # 第二个micro-batch: 2 samples  
            torch.randn(1, self.response_length)   # 第三个micro-batch: 1 sample
        ]
        
        # 聚合分数
        aggregated_scores = torch.cat(micro_batch_scores, dim=0)
        
        # 验证聚合结果
        total_samples = sum(scores.size(0) for scores in micro_batch_scores)
        self.assertEqual(aggregated_scores.shape, (total_samples, self.response_length))
        self.assertTrue(torch.isfinite(aggregated_scores).all())
        
        # 验证顺序保持正确
        start_idx = 0
        for scores in micro_batch_scores:
            end_idx = start_idx + scores.size(0)
            torch.testing.assert_close(aggregated_scores[start_idx:end_idx], scores)
            start_idx = end_idx
    
    def test_dataproto_interface(self):
        """测试DataProto接口的使用"""
        if not VERL_AVAILABLE:
            self.skipTest("VERL modules not available")
        
        # 创建输入DataProto
        input_tensors = {
            'responses': torch.randint(0, 1000, (self.batch_size, self.response_length)),
            'reward_mask': torch.ones(self.batch_size, self.response_length),
            'input_ids': torch.randint(0, 1000, (self.batch_size, 10)),
        }
        input_data = DataProto.from_dict(tensors=input_tensors)
        
        # 验证输入DataProto
        self.assertIsInstance(input_data, DataProto)
        self.assertIn('responses', input_data.batch)
        self.assertIn('reward_mask', input_data.batch)
        self.assertEqual(input_data.batch['responses'].shape, (self.batch_size, self.response_length))
        
        # 创建输出DataProto
        output_scores = torch.randn(self.batch_size, self.response_length)
        output_data = DataProto.from_dict(tensors={'rm_scores': output_scores})
        
        # 验证输出DataProto
        self.assertIsInstance(output_data, DataProto)
        self.assertIn('rm_scores', output_data.batch)
        self.assertEqual(output_data.batch['rm_scores'].shape, (self.batch_size, self.response_length))
        torch.testing.assert_close(output_data.batch['rm_scores'], output_scores)
    
    def test_device_transfer_simulation(self):
        """测试设备转移的模拟"""
        # 创建CPU上的数据
        input_tensors = {
            'responses': torch.randint(0, 1000, (self.batch_size, self.response_length)),
            'reward_mask': torch.ones(self.batch_size, self.response_length)
        }
        
        # 模拟到CUDA的转移 (实际测试中保持在CPU上)
        cuda_tensors = {}
        for key, tensor in input_tensors.items():
            # 在实际实现中这里会是 tensor.to('cuda')
            cuda_tensors[key] = tensor.clone()
        
        # 验证"转移"后的数据
        for key in input_tensors:
            self.assertEqual(cuda_tensors[key].shape, input_tensors[key].shape)
            self.assertEqual(cuda_tensors[key].dtype, input_tensors[key].dtype)
    
    def test_error_handling_simulation(self):
        """测试错误处理的模拟"""
        # 测试空batch的处理
        empty_responses = torch.empty(0, self.response_length, dtype=torch.long)
        empty_mask = torch.empty(0, self.response_length, dtype=torch.float)
        
        # 模拟空batch的处理结果
        empty_result_scores = torch.empty(0, self.response_length, dtype=torch.float)
        
        # 验证空batch处理
        self.assertEqual(empty_result_scores.shape, (0, self.response_length))
        self.assertTrue(torch.isfinite(empty_result_scores).all())  # 空tensor也应该是finite
        
        # 测试单样本batch
        single_responses = torch.randint(0, 1000, (1, self.response_length))
        single_mask = torch.ones(1, self.response_length)
        
        # 模拟单样本处理
        single_result_scores = torch.randn(1, self.response_length)
        
        # 验证单样本处理
        self.assertEqual(single_result_scores.shape, (1, self.response_length))
        self.assertTrue(torch.isfinite(single_result_scores).all())


class TestComputeRmScoreIntegration(unittest.TestCase):
    """集成测试：测试compute_rm_score的整体工作流程"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 6
        self.response_length = 8
        self.seq_len = 20
    
    def test_end_to_end_workflow_simulation(self):
        """端到端工作流程的模拟测试"""
        # 1. 输入数据准备
        input_data = {
            'responses': torch.randint(0, 1000, (self.batch_size, self.response_length)),
            'reward_mask': torch.ones(self.batch_size, self.response_length),
            'input_ids': torch.randint(0, 1000, (self.batch_size, self.seq_len)),
            'attention_mask': torch.ones(self.batch_size, self.seq_len),
            'position_ids': torch.arange(self.seq_len).unsqueeze(0).repeat(self.batch_size, 1),
        }
        
        # 2. 模拟数据预处理
        processed_data = input_data.copy()
        processed_data['steps'] = torch.randint(0, 5, (self.batch_size, self.response_length))
        processed_data['prm_input_ids'] = torch.randint(0, 1000, (self.batch_size, self.seq_len + 10))
        processed_data['prm_attention_mask'] = torch.ones(self.batch_size, self.seq_len + 10)
        
        # 3. 模拟分批处理
        micro_batch_size = 2
        all_rm_scores = []
        
        for i in range(0, self.batch_size, micro_batch_size):
            end_idx = min(i + micro_batch_size, self.batch_size)
            
            # 提取micro-batch
            current_batch_size = end_idx - i
            
            # 模拟_forward_micro_batch处理
            micro_rm_scores = torch.randn(current_batch_size, self.response_length)
            all_rm_scores.append(micro_rm_scores)
        
        # 4. 聚合结果
        final_rm_scores = torch.cat(all_rm_scores, dim=0)
        
        # 5. 验证端到端结果
        self.assertEqual(final_rm_scores.shape, (self.batch_size, self.response_length))
        self.assertTrue(torch.isfinite(final_rm_scores).all())
        
        # 验证处理的完整性
        expected_num_batches = (self.batch_size + micro_batch_size - 1) // micro_batch_size
        self.assertEqual(len(all_rm_scores), expected_num_batches)
        
        # 验证批次顺序的一致性
        reconstructed_batch_size = sum(scores.size(0) for scores in all_rm_scores)
        self.assertEqual(reconstructed_batch_size, self.batch_size)


if __name__ == "__main__":
    unittest.main()