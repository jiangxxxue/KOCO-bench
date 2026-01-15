import unittest
import torch
import sys
import os
from unittest.mock import MagicMock, patch

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the actual class we want to test
from open_r1.coupled_grpo import DiffuGRPOTrainer


class TestGetPerTokenLogps(unittest.TestCase):
    """测试DiffuGRPOTrainer._get_per_token_logps函数 - 使用Coupled Sampling计算token级对数概率"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 1000
        self.mask_token_id = 50256
    
    def test_get_per_token_logps_input_validation(self):
        """测试输入验证"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        # 测试错误的输入维度
        input_ids_2d = torch.randint(0, self.vocab_size, (4, 20))
        
        with self.assertRaises(ValueError) as context:
            trainer._get_per_token_logps(None, input_ids_2d, 10, [42])
        
        self.assertIn("3 dimensions", str(context.exception))
    
    def test_get_per_token_logps_mask_seeds_validation(self):
        """测试mask_seeds长度验证"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        num_iterations = 3
        batch_size = 2
        seq_len = 15
        
        input_ids = torch.randint(0, self.vocab_size, (num_iterations, batch_size, seq_len))
        
        # 提供错误数量的mask_seeds
        wrong_seeds = [42, 43]  # 应该是3个
        
        with self.assertRaises(ValueError) as context:
            trainer._get_per_token_logps(None, input_ids, 10, wrong_seeds)
        
        self.assertIn("mask_seeds length", str(context.exception))
    
    def test_get_per_token_logps_output_shape(self):
        """测试输出形状的正确性"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        # Mock forward_process和get_logits方法
        def mock_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            b, l = batch.shape
            noisy_batch = [batch.clone() for _ in range(3)]
            weights = [1.0, 1.5, 2.0]
            completion_mask = torch.ones(b, l, dtype=torch.bool)
            return noisy_batch, weights, completion_mask
        
        trainer.forward_process = mock_forward_process
        
        def mock_get_logits(model, batch):
            b, l = batch.shape
            return torch.randn(b, l, self.vocab_size)
        
        trainer.get_logits = mock_get_logits
        
        # 创建输入
        num_iterations = 2
        batch_size = 3
        seq_len = 20
        logits_to_keep = 10
        
        input_ids = torch.randint(0, self.vocab_size, (num_iterations, batch_size, seq_len))
        mask_seeds = [42, 43]
        
        # Mock model
        mock_model = MagicMock()
        
        # 调用函数
        result = trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
        
        # 验证输出形状
        expected_shape = (batch_size, num_iterations, logits_to_keep)
        self.assertEqual(result.shape, expected_shape,
                        f"输出形状应该为{expected_shape}")
        
        # 验证输出是有限的
        self.assertTrue(torch.isfinite(result).all(),
                       "输出应该全部是有限值")
    
    def test_get_per_token_logps_prompt_index_creation(self):
        """测试prompt_index的正确创建"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        # 记录forward_process被调用时的prompt_index
        captured_prompt_indices = []
        
        def mock_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            captured_prompt_indices.append(prompt_index.clone())
            b, l = batch.shape
            noisy_batch = [batch.clone() for _ in range(3)]
            weights = [1.0, 1.5, 2.0]
            completion_mask = torch.ones(b, l, dtype=torch.bool)
            return noisy_batch, weights, completion_mask
        
        trainer.forward_process = mock_forward_process
        trainer.get_logits = lambda model, batch: torch.randn(batch.size(0), batch.size(1), self.vocab_size)
        
        num_iterations = 1
        batch_size = 2
        seq_len = 15
        logits_to_keep = 8
        prompt_length = seq_len - logits_to_keep
        
        input_ids = torch.randint(0, self.vocab_size, (num_iterations, batch_size, seq_len))
        mask_seeds = [42]
        
        mock_model = MagicMock()
        trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
        
        # 验证prompt_index
        self.assertEqual(len(captured_prompt_indices), num_iterations)
        prompt_index = captured_prompt_indices[0]
        
        # 前prompt_length个位置应该为True
        self.assertTrue(prompt_index[:prompt_length].all(),
                       "提示部分应该全部标记为True")
        
        # 后logits_to_keep个位置应该为False
        self.assertFalse(prompt_index[prompt_length:].any(),
                        "完成部分应该全部标记为False")
    
    def test_get_per_token_logps_multiple_iterations(self):
        """测试多次迭代的处理"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        # 记录forward_process被调用的次数和使用的种子
        call_count = [0]
        used_seeds = []
        
        def mock_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            call_count[0] += 1
            used_seeds.append(seed)
            b, l = batch.shape
            noisy_batch = [batch.clone() for _ in range(3)]
            weights = [1.0, 1.5, 2.0]
            completion_mask = torch.ones(b, l, dtype=torch.bool)
            return noisy_batch, weights, completion_mask
        
        trainer.forward_process = mock_forward_process
        trainer.get_logits = lambda model, batch: torch.randn(batch.size(0), batch.size(1), self.vocab_size)
        
        num_iterations = 3
        batch_size = 2
        seq_len = 15
        logits_to_keep = 8
        
        input_ids = torch.randint(0, self.vocab_size, (num_iterations, batch_size, seq_len))
        mask_seeds = [42, 43, 44]
        
        mock_model = MagicMock()
        result = trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
        
        # 验证forward_process被调用了num_iterations次
        self.assertEqual(call_count[0], num_iterations,
                        f"forward_process应该被调用{num_iterations}次")
        
        # 验证使用了正确的种子
        self.assertEqual(used_seeds, mask_seeds,
                        "应该使用提供的mask_seeds")
        
        # 验证输出形状
        self.assertEqual(result.shape, (batch_size, num_iterations, logits_to_keep))
    
    def test_get_per_token_logps_batch_concatenation(self):
        """测试批次拼接的正确性"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        # 记录传递给get_logits的batch大小
        get_logits_batch_sizes = []
        
        def mock_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            b, l = batch.shape
            noisy_batch = [batch.clone() for _ in range(3)]
            weights = [1.0, 1.5, 2.0]
            completion_mask = torch.ones(b, l, dtype=torch.bool)
            return noisy_batch, weights, completion_mask
        
        def mock_get_logits(model, batch):
            get_logits_batch_sizes.append(batch.size(0))
            b, l = batch.shape
            return torch.randn(b, l, self.vocab_size)
        
        trainer.forward_process = mock_forward_process
        trainer.get_logits = mock_get_logits
        
        num_iterations = 2
        batch_size = 3
        seq_len = 15
        logits_to_keep = 8
        
        input_ids = torch.randint(0, self.vocab_size, (num_iterations, batch_size, seq_len))
        mask_seeds = [42, 43]
        
        mock_model = MagicMock()
        trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
        
        # 验证get_logits接收的batch大小
        # 应该是 num_iterations * 3 * batch_size
        expected_batch_size = num_iterations * 3 * batch_size
        self.assertEqual(get_logits_batch_sizes[0], expected_batch_size,
                        f"get_logits应该接收大小为{expected_batch_size}的batch")
    
    def test_get_per_token_logps_completion_extraction(self):
        """测试完成部分的正确提取"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        def mock_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            b, l = batch.shape
            noisy_batch = [batch.clone() for _ in range(3)]
            weights = [1.0, 1.5, 2.0]
            completion_mask = torch.ones(b, l, dtype=torch.bool)
            return noisy_batch, weights, completion_mask
        
        trainer.forward_process = mock_forward_process
        trainer.get_logits = lambda model, batch: torch.randn(batch.size(0), batch.size(1), self.vocab_size)
        
        num_iterations = 1
        batch_size = 2
        seq_len = 20
        logits_to_keep = 5  # 只保留最后5个token的logits
        
        input_ids = torch.randint(0, self.vocab_size, (num_iterations, batch_size, seq_len))
        mask_seeds = [42]
        
        mock_model = MagicMock()
        result = trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
        
        # 验证只返回了logits_to_keep个token的对数概率
        self.assertEqual(result.shape[2], logits_to_keep,
                        f"应该只返回{logits_to_keep}个token的对数概率")
    
    def test_get_per_token_logps_weight_propagation(self):
        """测试权重的正确传播"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        # 记录forward_process返回的权重
        returned_weights = []
        
        def mock_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            b, l = batch.shape
            noisy_batch = [batch.clone() for _ in range(3)]
            weights = [1.0, 2.0, 3.0]  # 使用特定的权重
            returned_weights.append(weights)
            completion_mask = torch.ones(b, l, dtype=torch.bool)
            return noisy_batch, weights, completion_mask
        
        trainer.forward_process = mock_forward_process
        trainer.get_logits = lambda model, batch: torch.randn(batch.size(0), batch.size(1), self.vocab_size)
        
        num_iterations = 2
        batch_size = 2
        seq_len = 15
        logits_to_keep = 8
        
        input_ids = torch.randint(0, self.vocab_size, (num_iterations, batch_size, seq_len))
        mask_seeds = [42, 43]
        
        mock_model = MagicMock()
        
        # Mock selective_log_softmax来验证权重
        with patch('open_r1.coupled_grpo.selective_log_softmax') as mock_selective:
            mock_selective.return_value = torch.randn(num_iterations * batch_size, logits_to_keep)
            
            trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
            
            # 验证selective_log_softmax被调用
            self.assertTrue(mock_selective.called)
            
            # 获取传递给selective_log_softmax的权重参数
            call_args = mock_selective.call_args
            weights_arg = call_args[0][2]  # 第三个位置参数是weights
            
            # 验证权重的长度
            expected_weight_length = num_iterations * 3
            self.assertEqual(len(weights_arg), expected_weight_length,
                           f"权重长度应该为{expected_weight_length}")
    
    def test_get_per_token_logps_output_permutation(self):
        """测试输出维度的正确排列"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        def mock_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            b, l = batch.shape
            noisy_batch = [batch.clone() for _ in range(3)]
            weights = [1.0, 1.5, 2.0]
            completion_mask = torch.ones(b, l, dtype=torch.bool)
            return noisy_batch, weights, completion_mask
        
        trainer.forward_process = mock_forward_process
        trainer.get_logits = lambda model, batch: torch.randn(batch.size(0), batch.size(1), self.vocab_size)
        
        num_iterations = 3
        batch_size = 4
        seq_len = 15
        logits_to_keep = 8
        
        input_ids = torch.randint(0, self.vocab_size, (num_iterations, batch_size, seq_len))
        mask_seeds = [42, 43, 44]
        
        mock_model = MagicMock()
        result = trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
        
        # 验证输出维度顺序：[batch_size, num_iterations, logits_to_keep]
        self.assertEqual(result.shape[0], batch_size, "第一维应该是batch_size")
        self.assertEqual(result.shape[1], num_iterations, "第二维应该是num_iterations")
        self.assertEqual(result.shape[2], logits_to_keep, "第三维应该是logits_to_keep")
    
    def test_get_per_token_logps_dtype_conversion(self):
        """测试输出数据类型转换"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        def mock_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            b, l = batch.shape
            noisy_batch = [batch.clone() for _ in range(3)]
            weights = [1.0, 1.5, 2.0]
            completion_mask = torch.ones(b, l, dtype=torch.bool)
            return noisy_batch, weights, completion_mask
        
        trainer.forward_process = mock_forward_process
        trainer.get_logits = lambda model, batch: torch.randn(batch.size(0), batch.size(1), self.vocab_size)
        
        num_iterations = 1
        batch_size = 2
        seq_len = 15
        logits_to_keep = 8
        
        input_ids = torch.randint(0, self.vocab_size, (num_iterations, batch_size, seq_len))
        mask_seeds = [42]
        
        mock_model = MagicMock()
        result = trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
        
        # 验证输出数据类型为float32
        self.assertEqual(result.dtype, torch.float32,
                        "输出应该转换为float32类型")
    
    def test_get_per_token_logps_edge_cases(self):
        """测试边缘情况"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = self.mask_token_id
        
        def mock_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            b, l = batch.shape
            noisy_batch = [batch.clone() for _ in range(3)]
            weights = [1.0, 1.5, 2.0]
            completion_mask = torch.ones(b, l, dtype=torch.bool)
            return noisy_batch, weights, completion_mask
        
        trainer.forward_process = mock_forward_process
        trainer.get_logits = lambda model, batch: torch.randn(batch.size(0), batch.size(1), self.vocab_size)
        
        mock_model = MagicMock()
        
        # 测试1: 单个迭代，单个样本
        input_ids_single = torch.randint(0, self.vocab_size, (1, 1, 10))
        result_single = trainer._get_per_token_logps(mock_model, input_ids_single, 5, [42])
        self.assertEqual(result_single.shape, (1, 1, 5))
        
        # 测试2: logits_to_keep等于seq_len（没有提示）
        input_ids_no_prompt = torch.randint(0, self.vocab_size, (1, 2, 10))
        result_no_prompt = trainer._get_per_token_logps(mock_model, input_ids_no_prompt, 10, [42])
        self.assertEqual(result_no_prompt.shape, (2, 1, 10))
        
        # 测试3: logits_to_keep为1（只有一个完成token）
        input_ids_one_token = torch.randint(0, self.vocab_size, (1, 2, 10))
        result_one_token = trainer._get_per_token_logps(mock_model, input_ids_one_token, 1, [42])
        self.assertEqual(result_one_token.shape, (2, 1, 1))


class TestGetPerTokenLogpsIntegration(unittest.TestCase):
    """集成测试：测试_get_per_token_logps在实际训练流程中的表现"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_end_to_end_workflow(self):
        """端到端测试：从输入到输出的完整流程"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = 50256
        
        # 使用真实的forward_process实现
        def real_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            import random
            if seed is not None:
                random.seed(seed)
                torch.manual_seed(seed)
            
            b, l = batch.shape
            noisy_batch = []
            mask_ratio = random.uniform(0.2, 0.8)
            random_matrix = torch.rand((b, l), device=batch.device)
            
            # Version 1: mask all completion
            is_mask = ~prompt_index
            noisy_batch.append(torch.where(is_mask, mask_id, batch))
            
            # Version 2: partial mask
            is_mask = ~prompt_index & (random_matrix < mask_ratio)
            completion_mask = is_mask
            noisy_batch.append(torch.where(is_mask, mask_id, batch))
            
            # Version 3: reverse mask
            is_mask = ~prompt_index & (random_matrix > mask_ratio)
            noisy_batch.append(torch.where(is_mask, mask_id, batch))
            
            return noisy_batch, [1, 1/mask_ratio, 1/(1-mask_ratio)], completion_mask
        
        trainer.forward_process = real_forward_process
        trainer.get_logits = lambda model, batch: torch.randn(batch.size(0), batch.size(1), 1000)
        
        # 创建输入
        num_iterations = 2
        batch_size = 3
        seq_len = 20
        logits_to_keep = 10
        
        input_ids = torch.randint(0, 1000, (num_iterations, batch_size, seq_len))
        mask_seeds = [42, 43]
        
        mock_model = MagicMock()
        
        # 调用函数
        result = trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
        
        # 验证输出
        self.assertEqual(result.shape, (batch_size, num_iterations, logits_to_keep))
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result.dtype, torch.float32)
    
    def test_consistency_across_calls(self):
        """测试使用相同种子时的一致性"""
        trainer = object.__new__(DiffuGRPOTrainer)
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = 50256
        
        def mock_forward_process(batch, prompt_index, mask_id, seed=None, accumulate=False):
            b, l = batch.shape
            noisy_batch = [batch.clone() for _ in range(3)]
            weights = [1.0, 1.5, 2.0]
            completion_mask = torch.ones(b, l, dtype=torch.bool)
            return noisy_batch, weights, completion_mask
        
        trainer.forward_process = mock_forward_process
        
        # 使用固定的logits来确保可重复性
        def deterministic_get_logits(model, batch):
            torch.manual_seed(100)
            return torch.randn(batch.size(0), batch.size(1), 1000)
        
        trainer.get_logits = deterministic_get_logits
        
        num_iterations = 2
        batch_size = 2
        seq_len = 15
        logits_to_keep = 8
        
        input_ids = torch.randint(0, 1000, (num_iterations, batch_size, seq_len))
        mask_seeds = [42, 43]
        
        mock_model = MagicMock()
        
        # 调用两次
        result1 = trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
        result2 = trainer._get_per_token_logps(mock_model, input_ids, logits_to_keep, mask_seeds)
        
        # 验证结果一致
        torch.testing.assert_close(result1, result2,
                                  msg="使用相同输入应该产生相同输出")


if __name__ == "__main__":
    unittest.main()
