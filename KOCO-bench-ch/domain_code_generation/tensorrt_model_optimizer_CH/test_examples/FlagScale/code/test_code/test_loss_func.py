#!/usr/bin/env python3
# Copyright 2024 FlagScale. All rights reserved.

"""
测试 loss_func 函数
"""

import torch
import unittest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from functools import partial

# Add the parent directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Setup minimal mocks before importing
from test_code._mock_megatron import MinimalMock
MinimalMock.setup_megatron_mocks()

# Import after mocks are set up
try:
    from flagscale.train.train_gpt import loss_func
    IMPORT_SUCCESS = True
except Exception as e:
    print(f"警告: 无法导入实现代码: {e}")
    IMPORT_SUCCESS = False


class TestLossFunc(unittest.TestCase):
    """测试 loss_func 函数"""
    
    def setUp(self):
        """设置测试环境"""
        torch.manual_seed(42)
        
        # Mock args
        self.mock_args = Mock()
        self.mock_args.check_for_nan_in_loss_and_grad = False
        self.mock_args.check_for_spiky_loss = False
        
        # Mock rerun_state_machine
        self.mock_rerun_state_machine = Mock()
        self.mock_rerun_state_machine.validate_result = Mock()
        self.mock_rerun_state_machine.is_unexpectedly_large = Mock(return_value=False)
        
        # 创建测试数据
        self.loss_mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])  # 3个有效token
        self.output_tensor = torch.tensor([2.5, 3.0, 1.0, 4.0, 0.5])  # 损失值
        
        # Mock model
        self.mock_model = Mock()
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_rerun_state_machine')
    @patch('flagscale.train.train_gpt.has_nvidia_modelopt', False)
    def test_applies_loss_mask(self, mock_get_rerun, mock_get_args):
        """
        测试样例1: 应用loss_mask过滤损失
        
        关键验证：损失计算应该使用loss_mask进行过滤
        """
        # Setup mocks
        mock_get_args.return_value = self.mock_args
        mock_get_rerun.return_value = self.mock_rerun_state_machine
        
        # Execute
        loss, num_tokens, metrics = loss_func(self.loss_mask, self.output_tensor, self.mock_model)
        
        # Verify: loss应该是masked losses的和
        # 有效token: indices 0, 1, 3 -> losses: 2.5, 3.0, 4.0 -> sum = 9.5
        expected_loss = torch.tensor(9.5)
        torch.testing.assert_close(loss, expected_loss, rtol=1e-5)
        
        # Verify: num_tokens应该是loss_mask的和
        expected_num_tokens = torch.tensor(3, dtype=torch.int)
        torch.testing.assert_close(num_tokens, expected_num_tokens)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_rerun_state_machine')
    @patch('flagscale.train.train_gpt.has_nvidia_modelopt', False)
    def test_computes_num_tokens_correctly(self, mock_get_rerun, mock_get_args):
        """
        测试样例2: 正确计算token数量
        
        关键验证：num_tokens应该是loss_mask.sum()
        """
        # Setup mocks
        mock_get_args.return_value = self.mock_args
        mock_get_rerun.return_value = self.mock_rerun_state_machine
        
        # Execute
        loss, num_tokens, metrics = loss_func(self.loss_mask, self.output_tensor, self.mock_model)
        
        # Verify: num_tokens = loss_mask.sum()
        expected_num_tokens = self.loss_mask.sum().to(torch.int)
        torch.testing.assert_close(num_tokens, expected_num_tokens)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_rerun_state_machine')
    @patch('flagscale.train.train_gpt.has_nvidia_modelopt', False)
    def test_returns_correct_structure(self, mock_get_rerun, mock_get_args):
        """
        测试样例3: 返回正确的数据结构
        
        关键验证：应该返回(loss, num_tokens, metrics字典)
        """
        # Setup mocks
        mock_get_args.return_value = self.mock_args
        mock_get_rerun.return_value = self.mock_rerun_state_machine
        
        # Execute
        result = loss_func(self.loss_mask, self.output_tensor, self.mock_model)
        
        # Verify: 返回元组
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        
        loss, num_tokens, metrics = result
        
        # Verify: loss是标量tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # 标量
        
        # Verify: num_tokens是标量tensor
        self.assertIsInstance(num_tokens, torch.Tensor)
        self.assertEqual(num_tokens.dtype, torch.int)
        
        # Verify: metrics是字典，包含'lm loss'
        self.assertIsInstance(metrics, dict)
        self.assertIn('lm loss', metrics)
        self.assertIsInstance(metrics['lm loss'], torch.Tensor)
        self.assertEqual(metrics['lm loss'].shape[0], 2)  # [loss, num_tokens]
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_rerun_state_machine')
    @patch('flagscale.train.train_gpt.has_nvidia_modelopt', True)
    @patch('flagscale.train.train_gpt.modelopt_args_enabled')
    @patch('flagscale.train.train_gpt.loss_func_modelopt')
    def test_calls_modelopt_loss_when_enabled(self, mock_loss_func_modelopt, mock_modelopt_enabled, 
                                               mock_get_rerun, mock_get_args):
        """
        测试样例4: ModelOpt启用时调用modelopt的loss_func
        
        关键验证：当has_nvidia_modelopt=True且modelopt_args_enabled=True时，应该调用loss_func_modelopt
        """
        # Setup mocks
        mock_get_args.return_value = self.mock_args
        mock_modelopt_enabled.return_value = True
        mock_get_rerun.return_value = self.mock_rerun_state_machine
        
        mock_loss_func_modelopt.return_value = (
            torch.tensor(5.0), 
            torch.tensor(3), 
            {'lm loss': torch.tensor([5.0, 3.0])}
        )
        
        # Execute
        result = loss_func(self.loss_mask, self.output_tensor, self.mock_model)
        
        # Verify: 调用了loss_func_modelopt
        mock_loss_func_modelopt.assert_called_once_with(
            self.loss_mask, 
            self.output_tensor, 
            model=self.mock_model
        )
        
        # Verify: 返回了modelopt的结果
        self.assertEqual(result[0].item(), 5.0)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_rerun_state_machine')
    @patch('flagscale.train.train_gpt.has_nvidia_modelopt', False)
    def test_checks_for_nan_when_enabled(self, mock_get_rerun, mock_get_args):
        """
        测试样例5: 启用NaN检查时调用validate_result
        
        关键验证：当check_for_nan_in_loss_and_grad=True时，应该调用validate_result检查NaN
        """
        # Setup mocks
        self.mock_args.check_for_nan_in_loss_and_grad = True
        mock_get_args.return_value = self.mock_args
        mock_get_rerun.return_value = self.mock_rerun_state_machine
        
        # Execute
        loss_func(self.loss_mask, self.output_tensor, self.mock_model)
        
        # Verify: 调用了validate_result检查NaN
        self.assertEqual(mock_get_rerun_state_machine.validate_result.call_count, 2)  # NaN和Inf检查
        
        # 验证NaN检查调用
        nan_call = mock_get_rerun_state_machine.validate_result.call_args_list[0]
        self.assertEqual(nan_call[1]['rejection_func'], torch.isnan)
        self.assertEqual(nan_call[1]['message'], "found NaN in local forward loss calculation")
        self.assertEqual(nan_call[1]['fatal'], True)
        
        # 验证Inf检查调用
        inf_call = mock_get_rerun_state_machine.validate_result.call_args_list[1]
        self.assertEqual(inf_call[1]['rejection_func'], torch.isinf)
        self.assertEqual(inf_call[1]['message'], "found Inf in local forward loss calculation")
        self.assertEqual(inf_call[1]['fatal'], True)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_rerun_state_machine')
    @patch('flagscale.train.train_gpt.has_nvidia_modelopt', False)
    def test_checks_for_spiky_loss_when_enabled(self, mock_get_rerun, mock_get_args):
        """
        测试样例6: 启用spiky loss检查时调用validate_result
        
        关键验证：当check_for_spiky_loss=True时，应该调用validate_result检查spiky loss
        """
        # Setup mocks
        self.mock_args.check_for_spiky_loss = True
        mock_get_args.return_value = self.mock_args
        mock_get_rerun.return_value = self.mock_rerun_state_machine
        
        # Execute
        loss_func(self.loss_mask, self.output_tensor, self.mock_model)
        
        # Verify: 调用了validate_result检查spiky loss
        # 应该至少有一次调用（可能还有NaN检查）
        call_count = mock_get_rerun_state_machine.validate_result.call_count
        self.assertGreaterEqual(call_count, 1)
        
        # 查找spiky loss检查调用
        spiky_calls = [
            call for call in mock_get_rerun_state_machine.validate_result.call_args_list
            if call[1].get('message') == 'Spiky loss'
        ]
        self.assertGreater(len(spiky_calls), 0, "应该调用validate_result检查spiky loss")
        
        spiky_call = spiky_calls[0]
        self.assertEqual(spiky_call[1]['message'], "Spiky loss")
        self.assertEqual(spiky_call[1]['fatal'], False)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_rerun_state_machine')
    @patch('flagscale.train.train_gpt.has_nvidia_modelopt', False)
    def test_loss_mask_view_operation(self, mock_get_rerun, mock_get_args):
        """
        测试样例7: loss_mask和output_tensor的view操作
        
        关键验证：应该对loss_mask和output_tensor进行view(-1)操作
        """
        # Setup mocks
        mock_get_args.return_value = self.mock_args
        mock_get_rerun.return_value = self.mock_rerun_state_machine
        
        # 使用2D tensor测试
        loss_mask_2d = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
        output_tensor_2d = torch.tensor([[2.0, 3.0], [1.0, 4.0]])
        
        # Execute
        loss, num_tokens, metrics = loss_func(loss_mask_2d, output_tensor_2d, self.mock_model)
        
        # Verify: 计算结果正确（应该flatten后计算）
        # 有效值: [2.0, 3.0, 4.0] -> sum = 9.0
        expected_loss = torch.tensor(9.0)
        torch.testing.assert_close(loss, expected_loss, rtol=1e-5)
        
        expected_num_tokens = torch.tensor(3, dtype=torch.int)
        torch.testing.assert_close(num_tokens, expected_num_tokens)


if __name__ == "__main__":
    unittest.main()
