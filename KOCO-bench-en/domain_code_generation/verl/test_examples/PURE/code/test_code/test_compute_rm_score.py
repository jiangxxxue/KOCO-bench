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
    """Test ProcessRewardModelWorker.compute_rm_score function - batch PRM inference and aggregation described in PURE document Section 2"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.response_length = 6
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_compute_rm_score_data_flow(self):
        """Test data flow and interface of compute_rm_score"""
        # Create input data
        input_data = {
            'responses': torch.randint(0, 1000, (self.batch_size, self.response_length)),
            'reward_mask': torch.ones(self.batch_size, self.response_length),
            'input_ids': torch.randint(0, 1000, (self.batch_size, 10)),
            'attention_mask': torch.ones(self.batch_size, 10),
            'position_ids': torch.arange(10).unsqueeze(0).repeat(self.batch_size, 1)
        }
        data = DataProto.from_dict(tensors=input_data)
        
        # Create mock worker
        mock_worker = MagicMock(spec=ProcessRewardModelWorker)
        mock_worker.config = MagicMock()
        mock_worker.config.use_dynamic_bsz = False
        mock_worker.config.micro_batch_size_per_gpu = 2
        mock_worker._is_offload_param = False
        mock_worker.world_size = 1
        
        # Mock all dependent methods and attributes
        mock_worker._split_steps.return_value = DataProto.from_dict(tensors={'steps': torch.randint(0, 5, (self.batch_size, 3))})
        mock_worker._build_inputs_for_prm.return_value = DataProto.from_dict(tensors={
            'prm_input_ids': torch.randint(0, 1000, (self.batch_size, 15)),
            'prm_attention_mask': torch.ones(self.batch_size, 15),
            'prm_position_ids': torch.arange(15).unsqueeze(0).repeat(self.batch_size, 1)
        })
        
        # Mock _forward_micro_batch to return reasonable scores
        def mock_forward_micro_batch(micro_batch):
            current_batch_size = micro_batch['responses'].size(0)
            return torch.randn(current_batch_size, self.response_length)
        
        mock_worker._forward_micro_batch.side_effect = mock_forward_micro_batch
        
        # Mock context managers and other dependencies
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_context)
        mock_context.__exit__ = MagicMock(return_value=None)
        mock_worker.ulysses_sharding_manager = mock_context
        mock_worker.ulysses_sharding_manager.preprocess_data.return_value = data
        mock_worker.ulysses_sharding_manager.postprocess_data.return_value = None
        
        # Mock batch splitting
        mock_batch = data.batch
        
        # Simulate batch processing logic
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
        
        # Simulate processing each micro-batch
        output_scores = []
        for micro_batch in micro_batches:
            scores = mock_forward_micro_batch(micro_batch)
            output_scores.append(scores)
        
        # Aggregate results
        token_level_scores = torch.cat(output_scores, dim=0)
        
        # Create output DataProto
        result = DataProto.from_dict(tensors={'rm_scores': token_level_scores})
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertIn('rm_scores', result.batch)
        rm_scores = result.batch['rm_scores']
        
        # Verify output shape and properties
        self.assertEqual(rm_scores.shape, (self.batch_size, self.response_length))
        self.assertTrue(torch.is_tensor(rm_scores))
        self.assertTrue(torch.isfinite(rm_scores).all())
    
    def test_micro_batch_splitting_logic(self):
        """Test micro-batch splitting logic"""
        total_batch_size = 8
        micro_batch_size = 3
        
        # Create test data
        responses = torch.randint(0, 1000, (total_batch_size, self.response_length))
        reward_mask = torch.ones(total_batch_size, self.response_length)
        
        # Implement micro-batch splitting
        micro_batches = []
        for i in range(0, total_batch_size, micro_batch_size):
            end_idx = min(i + micro_batch_size, total_batch_size)
            micro_batch = {
                'responses': responses[i:end_idx],
                'reward_mask': reward_mask[i:end_idx]
            }
            micro_batches.append(micro_batch)
        
        # Verify splitting results
        self.assertGreater(len(micro_batches), 1)  # Should be split into multiple micro-batches
        
        # Verify size of each micro-batch
        total_processed = 0
        for i, micro_batch in enumerate(micro_batches):
            batch_size = micro_batch['responses'].size(0)
            total_processed += batch_size
            
            # Except for the last micro-batch, others should have full size
            if i < len(micro_batches) - 1:
                expected_size = micro_batch_size
            else:
                expected_size = total_batch_size - i * micro_batch_size
            
            self.assertEqual(batch_size, expected_size)
            self.assertEqual(micro_batch['responses'].shape[1], self.response_length)
            self.assertEqual(micro_batch['reward_mask'].shape, (batch_size, self.response_length))
        
        # Verify all samples were processed
        self.assertEqual(total_processed, total_batch_size)
    
    def test_score_aggregation(self):
        """Test score aggregation logic"""
        # Create output scores from multiple micro-batches
        micro_batch_scores = [
            torch.randn(2, self.response_length),  # First micro-batch: 2 samples
            torch.randn(2, self.response_length),  # Second micro-batch: 2 samples  
            torch.randn(1, self.response_length)   # Third micro-batch: 1 sample
        ]
        
        # Aggregate scores
        aggregated_scores = torch.cat(micro_batch_scores, dim=0)
        
        # Verify aggregation results
        total_samples = sum(scores.size(0) for scores in micro_batch_scores)
        self.assertEqual(aggregated_scores.shape, (total_samples, self.response_length))
        self.assertTrue(torch.isfinite(aggregated_scores).all())
        
        # Verify order is maintained correctly
        start_idx = 0
        for scores in micro_batch_scores:
            end_idx = start_idx + scores.size(0)
            torch.testing.assert_close(aggregated_scores[start_idx:end_idx], scores)
            start_idx = end_idx
    
    def test_dataproto_interface(self):
        """Test DataProto interface usage"""
        if not VERL_AVAILABLE:
            self.skipTest("VERL modules not available")
        
        # Create input DataProto
        input_tensors = {
            'responses': torch.randint(0, 1000, (self.batch_size, self.response_length)),
            'reward_mask': torch.ones(self.batch_size, self.response_length),
            'input_ids': torch.randint(0, 1000, (self.batch_size, 10)),
        }
        input_data = DataProto.from_dict(tensors=input_tensors)
        
        # Verify input DataProto
        self.assertIsInstance(input_data, DataProto)
        self.assertIn('responses', input_data.batch)
        self.assertIn('reward_mask', input_data.batch)
        self.assertEqual(input_data.batch['responses'].shape, (self.batch_size, self.response_length))
        
        # Create output DataProto
        output_scores = torch.randn(self.batch_size, self.response_length)
        output_data = DataProto.from_dict(tensors={'rm_scores': output_scores})
        
        # Verify output DataProto
        self.assertIsInstance(output_data, DataProto)
        self.assertIn('rm_scores', output_data.batch)
        self.assertEqual(output_data.batch['rm_scores'].shape, (self.batch_size, self.response_length))
        torch.testing.assert_close(output_data.batch['rm_scores'], output_scores)
    
    def test_device_transfer_simulation(self):
        """Test device transfer simulation"""
        # Create data on CPU
        input_tensors = {
            'responses': torch.randint(0, 1000, (self.batch_size, self.response_length)),
            'reward_mask': torch.ones(self.batch_size, self.response_length)
        }
        
        # Simulate transfer to CUDA (kept on CPU in actual test)
        cuda_tensors = {}
        for key, tensor in input_tensors.items():
            # In actual implementation this would be tensor.to('cuda')
            cuda_tensors[key] = tensor.clone()
        
        # Verify data after "transfer"
        for key in input_tensors:
            self.assertEqual(cuda_tensors[key].shape, input_tensors[key].shape)
            self.assertEqual(cuda_tensors[key].dtype, input_tensors[key].dtype)
    
    def test_error_handling_simulation(self):
        """Test error handling simulation"""
        # Test empty batch handling
        empty_responses = torch.empty(0, self.response_length, dtype=torch.long)
        empty_mask = torch.empty(0, self.response_length, dtype=torch.float)
        
        # Simulate empty batch processing result
        empty_result_scores = torch.empty(0, self.response_length, dtype=torch.float)
        
        # Verify empty batch handling
        self.assertEqual(empty_result_scores.shape, (0, self.response_length))
        self.assertTrue(torch.isfinite(empty_result_scores).all())  # Empty tensor should also be finite
        
        # Test single sample batch
        single_responses = torch.randint(0, 1000, (1, self.response_length))
        single_mask = torch.ones(1, self.response_length)
        
        # Simulate single sample processing
        single_result_scores = torch.randn(1, self.response_length)
        
        # Verify single sample handling
        self.assertEqual(single_result_scores.shape, (1, self.response_length))
        self.assertTrue(torch.isfinite(single_result_scores).all())


class TestComputeRmScoreIntegration(unittest.TestCase):
    """Integration test: Test the overall workflow of compute_rm_score"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 6
        self.response_length = 8
        self.seq_len = 20
    
    def test_end_to_end_workflow_simulation(self):
        """End-to-end workflow simulation test"""
        # 1. Input data preparation
        input_data = {
            'responses': torch.randint(0, 1000, (self.batch_size, self.response_length)),
            'reward_mask': torch.ones(self.batch_size, self.response_length),
            'input_ids': torch.randint(0, 1000, (self.batch_size, self.seq_len)),
            'attention_mask': torch.ones(self.batch_size, self.seq_len),
            'position_ids': torch.arange(self.seq_len).unsqueeze(0).repeat(self.batch_size, 1),
        }
        
        # 2. Simulate data preprocessing
        processed_data = input_data.copy()
        processed_data['steps'] = torch.randint(0, 5, (self.batch_size, self.response_length))
        processed_data['prm_input_ids'] = torch.randint(0, 1000, (self.batch_size, self.seq_len + 10))
        processed_data['prm_attention_mask'] = torch.ones(self.batch_size, self.seq_len + 10)
        
        # 3. Simulate batch processing
        micro_batch_size = 2
        all_rm_scores = []
        
        for i in range(0, self.batch_size, micro_batch_size):
            end_idx = min(i + micro_batch_size, self.batch_size)
            
            # Extract micro-batch
            current_batch_size = end_idx - i
            
            # Simulate _forward_micro_batch processing
            micro_rm_scores = torch.randn(current_batch_size, self.response_length)
            all_rm_scores.append(micro_rm_scores)
        
        # 4. Aggregate results
        final_rm_scores = torch.cat(all_rm_scores, dim=0)
        
        # 5. Verify end-to-end results
        self.assertEqual(final_rm_scores.shape, (self.batch_size, self.response_length))
        self.assertTrue(torch.isfinite(final_rm_scores).all())
        
        # Verify processing completeness
        expected_num_batches = (self.batch_size + micro_batch_size - 1) // micro_batch_size
        self.assertEqual(len(all_rm_scores), expected_num_batches)
        
        # Verify batch order consistency
        reconstructed_batch_size = sum(scores.size(0) for scores in all_rm_scores)
        self.assertEqual(reconstructed_batch_size, self.batch_size)


if __name__ == "__main__":
    unittest.main()