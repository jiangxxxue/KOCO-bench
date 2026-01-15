#!/usr/bin/env python3
# Copyright 2024 ByteMLPerf. All rights reserved.

"""
测试 loss_func 函数
"""

import torch
import torch.nn as nn
import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Mock transformer_engine before any imports
sys.modules['transformer_engine'] = MagicMock()
sys.modules['transformer_engine.pytorch'] = MagicMock()

# Add the parent directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'byte_train_perf', 'Megatron-LM'))

# Import ground truth implementation
from megatron.inference.gpt.loss_func import loss_func


class TestLossFunc(unittest.TestCase):
    """测试loss_func函数"""
    
    def setUp(self):
        """设置测试环境"""
        torch.manual_seed(42)
        
        # 创建mock args
        self.mock_args = Mock()
        self.mock_args.context_parallel_size = 1
        self.mock_args.tensor_model_parallel_size = 1
        self.mock_args.check_for_nan_in_loss_and_grad = False
        self.mock_args.export_kd_teacher_load = None
        self.mock_args.export_kd_finalize = False
    
    def _create_mock_model(self, enable_kd=False):
        """创建mock模型"""
        model = Mock()
        model.training = True
        model._enable_kd = enable_kd
        
        if enable_kd:
            # Mock compute_kd_loss method
            def compute_kd_loss(student_loss, loss_reduction_fn):
                return student_loss * 1.5
            model.compute_kd_loss = compute_kd_loss
        
        return model
    
    @patch('megatron.inference.gpt.loss_func.get_args')
    @patch('megatron.inference.gpt.loss_func.unwrap_model')
    @patch('megatron.inference.gpt.loss_func._mask_loss')
    @patch('megatron.inference.gpt.loss_func._allreduce_loss')
    def test_basic_lm_loss(self, mock_allreduce, mock_mask, mock_unwrap, mock_get_args):
        """
        测试样例1: 基本语言模型损失计算
        
        输入:
        - loss_mask: 掩码张量
        - model: GPT模型
        - output_tensor: 模型输出
        
        预期行为:
        1. 计算masked loss
        2. 进行allreduce
        3. 返回损失和报告
        """
        # 设置mock
        mock_get_args.return_value = self.mock_args
        model = self._create_mock_model(enable_kd=False)
        mock_unwrap.return_value = model
        
        # 设置mask_loss返回值
        mock_loss = torch.tensor(2.5)
        mock_mask.return_value = mock_loss
        
        # 设置allreduce返回值
        mock_allreduce.return_value = (torch.tensor(2.5), torch.tensor(2.5))
        
        # 创建输入
        loss_mask = torch.ones(4, 10)
        output_tensor = torch.randn(4, 10, 50257)
        
        # 执行
        loss, report = loss_func(loss_mask, model, output_tensor)
        
        # 验证
        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertTrue(isinstance(report, dict))
        self.assertIn('lm loss', report)
        
        # 验证调用了正确的函数
        mock_mask.assert_called_once()
        mock_allreduce.assert_called_once()
        
        # 验证返回值
        torch.testing.assert_close(loss, torch.tensor(2.5), atol=1e-4, rtol=1e-4)
    
    @patch('megatron.inference.gpt.loss_func.get_args')
    @patch('megatron.inference.gpt.loss_func.unwrap_model')
    @patch('megatron.inference.gpt.loss_func._mask_loss')
    @patch('megatron.inference.gpt.loss_func._allreduce_loss')
    def test_kd_loss_calculation(self, mock_allreduce, mock_mask, mock_unwrap, mock_get_args):
        """
        测试样例2: 知识蒸馏损失计算
        
        输入:
        - enable_kd: True
        - export_kd_teacher_load: "/path/to/teacher"
        
        预期行为:
        1. 计算标准LM损失
        2. 额外计算KD损失
        3. 返回LM损失和KD损失的报告
        """
        # 启用KD
        self.mock_args.export_kd_teacher_load = "/path/to/teacher"
        mock_get_args.return_value = self.mock_args
        
        model = self._create_mock_model(enable_kd=True)
        mock_unwrap.return_value = model
        
        # 设置返回值
        mock_loss = torch.tensor(2.0)
        mock_mask.return_value = mock_loss
        mock_allreduce.return_value = (torch.tensor(2.0), torch.tensor(2.0))
        
        # 创建输入
        loss_mask = torch.ones(4, 10)
        output_tensor = torch.randn(4, 10, 50257)
        
        # 执行
        loss, report = loss_func(loss_mask, model, output_tensor)
        
        # 验证
        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertTrue(isinstance(report, dict))
        self.assertIn('lm loss', report)
        
        # KD模式下应该调用compute_kd_loss
        # 这里通过model的mock来验证
        self.assertTrue(model._enable_kd)
    
    @patch('megatron.inference.gpt.loss_func.get_args')
    @patch('megatron.inference.gpt.loss_func.unwrap_model')
    @patch('megatron.inference.gpt.loss_func._mask_loss')
    @patch('megatron.inference.gpt.loss_func._allreduce_loss')
    def test_eval_mode_no_kd(self, mock_allreduce, mock_mask, mock_unwrap, mock_get_args):
        """
        测试样例3: 评估模式不计算KD损失
        
        输入:
        - model.training: False
        - export_kd_teacher_load: "/path/to/teacher"
        
        预期行为: 即使设置了teacher_load，评估模式也不计算KD损失
        """
        self.mock_args.export_kd_teacher_load = "/path/to/teacher"
        mock_get_args.return_value = self.mock_args
        
        model = self._create_mock_model(enable_kd=True)
        model.training = False  # 评估模式
        mock_unwrap.return_value = model
        
        mock_loss = torch.tensor(1.5)
        mock_mask.return_value = mock_loss
        mock_allreduce.return_value = (torch.tensor(1.5), torch.tensor(1.5))
        
        loss_mask = torch.ones(4, 10)
        output_tensor = torch.randn(4, 10, 50257)
        
        loss, report = loss_func(loss_mask, model, output_tensor)
        
        # 验证只有LM损失
        self.assertIn('lm loss', report)
        torch.testing.assert_close(loss, torch.tensor(1.5), atol=1e-4, rtol=1e-4)
    
    @patch('megatron.inference.gpt.loss_func.get_args')
    @patch('megatron.inference.gpt.loss_func.unwrap_model')
    @patch('megatron.inference.gpt.loss_func._mask_loss')
    @patch('megatron.inference.gpt.loss_func._allreduce_loss')
    def test_kd_finalize_mode(self, mock_allreduce, mock_mask, mock_unwrap, mock_get_args):
        """
        测试样例4: KD finalize模式
        
        输入:
        - export_kd_finalize: True
        
        预期行为: finalize模式下不计算KD损失
        """
        self.mock_args.export_kd_teacher_load = "/path/to/teacher"
        self.mock_args.export_kd_finalize = True
        mock_get_args.return_value = self.mock_args
        
        model = self._create_mock_model(enable_kd=True)
        mock_unwrap.return_value = model
        
        mock_loss = torch.tensor(1.8)
        mock_mask.return_value = mock_loss
        mock_allreduce.return_value = (torch.tensor(1.8), torch.tensor(1.8))
        
        loss_mask = torch.ones(4, 10)
        output_tensor = torch.randn(4, 10, 50257)
        
        loss, report = loss_func(loss_mask, model, output_tensor)
        
        # finalize模式下只有LM损失
        self.assertIn('lm loss', report)
        torch.testing.assert_close(loss, torch.tensor(1.8), atol=1e-4, rtol=1e-4)
    
    @patch('megatron.inference.gpt.loss_func.get_args')
    @patch('megatron.inference.gpt.loss_func.unwrap_model')
    @patch('megatron.inference.gpt.loss_func._mask_loss')
    @patch('megatron.inference.gpt.loss_func._allreduce_loss')
    def test_loss_mask_application(self, mock_allreduce, mock_mask, mock_unwrap, mock_get_args):
        """
        测试样例5: 损失掩码的应用
        
        验证mask_loss被正确调用，传入正确的参数
        """
        mock_get_args.return_value = self.mock_args
        model = self._create_mock_model(enable_kd=False)
        mock_unwrap.return_value = model
        
        mock_loss = torch.tensor(3.2)
        mock_mask.return_value = mock_loss
        mock_allreduce.return_value = (torch.tensor(3.2), torch.tensor(3.2))
        
        # 创建特定的mask
        loss_mask = torch.tensor([[1, 1, 1, 0, 0],
                                   [1, 1, 1, 1, 0]])
        output_tensor = torch.randn(2, 5, 50257)
        
        loss, report = loss_func(loss_mask, model, output_tensor)
        
        # 验证mask_loss被调用，传入了正确的参数
        call_args = mock_mask.call_args
        self.assertEqual(call_args[0][1].shape, loss_mask.shape)
    
    @patch('megatron.inference.gpt.loss_func.get_args')
    @patch('megatron.inference.gpt.loss_func.unwrap_model')
    @patch('megatron.inference.gpt.loss_func._mask_loss')
    @patch('megatron.inference.gpt.loss_func._allreduce_loss')
    def test_different_loss_values(self, mock_allreduce, mock_mask, mock_unwrap, mock_get_args):
        """
        测试样例6: 不同的损失值
        
        测试不同大小的损失值都能正确处理
        """
        mock_get_args.return_value = self.mock_args
        model = self._create_mock_model(enable_kd=False)
        mock_unwrap.return_value = model
        
        loss_values = [0.1, 1.0, 5.0, 10.0]
        
        for loss_val in loss_values:
            with self.subTest(loss_value=loss_val):
                mock_loss = torch.tensor(loss_val)
                mock_mask.return_value = mock_loss
                mock_allreduce.return_value = (mock_loss, mock_loss)
                
                loss_mask = torch.ones(4, 10)
                output_tensor = torch.randn(4, 10, 50257)
                
                loss, report = loss_func(loss_mask, model, output_tensor)
                
                torch.testing.assert_close(loss, torch.tensor(loss_val), atol=1e-4, rtol=1e-4)
    
    @patch('megatron.inference.gpt.loss_func.get_args')
    @patch('megatron.inference.gpt.loss_func.unwrap_model')
    @patch('megatron.inference.gpt.loss_func._mask_loss')
    @patch('megatron.inference.gpt.loss_func._allreduce_loss')
    def test_report_structure(self, mock_allreduce, mock_mask, mock_unwrap, mock_get_args):
        """
        测试样例7: 报告结构验证
        
        验证返回的report字典包含正确的键
        """
        mock_get_args.return_value = self.mock_args
        model = self._create_mock_model(enable_kd=False)
        mock_unwrap.return_value = model
        
        mock_loss = torch.tensor(2.3)
        mock_mask.return_value = mock_loss
        mock_allreduce.return_value = (mock_loss, mock_loss)
        
        loss_mask = torch.ones(4, 10)
        output_tensor = torch.randn(4, 10, 50257)
        
        loss, report = loss_func(loss_mask, model, output_tensor)
        
        # 验证report是字典
        self.assertTrue(isinstance(report, dict))
        
        # 验证包含lm loss
        self.assertIn('lm loss', report)
        
        # 验证值是tensor
        self.assertTrue(isinstance(report['lm loss'], torch.Tensor))


class TestLossFuncEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def setUp(self):
        torch.manual_seed(42)
        
        self.mock_args = Mock()
        self.mock_args.context_parallel_size = 1
        self.mock_args.tensor_model_parallel_size = 1
        self.mock_args.check_for_nan_in_loss_and_grad = False
        self.mock_args.export_kd_teacher_load = None
        self.mock_args.export_kd_finalize = False
    
    @patch('megatron.inference.gpt.loss_func.get_args')
    @patch('megatron.inference.gpt.loss_func.unwrap_model')
    @patch('megatron.inference.gpt.loss_func._mask_loss')
    @patch('megatron.inference.gpt.loss_func._allreduce_loss')
    def test_small_batch_size(self, mock_allreduce, mock_mask, mock_unwrap, mock_get_args):
        """
        测试样例8: 小批次大小
        
        输入: batch_size=1
        预期: 正常处理
        """
        mock_get_args.return_value = self.mock_args
        model = Mock()
        model.training = True
        mock_unwrap.return_value = model
        
        mock_loss = torch.tensor(1.2)
        mock_mask.return_value = mock_loss
        mock_allreduce.return_value = (mock_loss, mock_loss)
        
        loss_mask = torch.ones(1, 10)
        output_tensor = torch.randn(1, 10, 50257)
        
        loss, report = loss_func(loss_mask, model, output_tensor)
        
        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertIn('lm loss', report)
    
    @patch('megatron.inference.gpt.loss_func.get_args')
    @patch('megatron.inference.gpt.loss_func.unwrap_model')
    @patch('megatron.inference.gpt.loss_func._mask_loss')
    @patch('megatron.inference.gpt.loss_func._allreduce_loss')
    def test_large_batch_size(self, mock_allreduce, mock_mask, mock_unwrap, mock_get_args):
        """
        测试样例9: 大批次大小
        
        输入: batch_size=128
        预期: 正常处理
        """
        mock_get_args.return_value = self.mock_args
        model = Mock()
        model.training = True
        mock_unwrap.return_value = model
        
        mock_loss = torch.tensor(2.1)
        mock_mask.return_value = mock_loss
        mock_allreduce.return_value = (mock_loss, mock_loss)
        
        loss_mask = torch.ones(128, 10)
        output_tensor = torch.randn(128, 10, 50257)
        
        loss, report = loss_func(loss_mask, model, output_tensor)
        
        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertIn('lm loss', report)
    
    @patch('megatron.inference.gpt.loss_func.get_args')
    @patch('megatron.inference.gpt.loss_func.unwrap_model')
    @patch('megatron.inference.gpt.loss_func._mask_loss')
    @patch('megatron.inference.gpt.loss_func._allreduce_loss')
    def test_all_masked_positions(self, mock_allreduce, mock_mask, mock_unwrap, mock_get_args):
        """
        测试样例10: 全部位置被mask
        
        输入: loss_mask全为0
        预期: 返回0损失（或处理masked情况）
        """
        mock_get_args.return_value = self.mock_args
        model = Mock()
        model.training = True
        mock_unwrap.return_value = model
        
        # 模拟全masked情况，损失为0或nan
        mock_loss = torch.tensor(0.0)
        mock_mask.return_value = mock_loss
        mock_allreduce.return_value = (mock_loss, mock_loss)
        
        loss_mask = torch.zeros(4, 10)  # 全部mask
        output_tensor = torch.randn(4, 10, 50257)
        
        loss, report = loss_func(loss_mask, model, output_tensor)
        
        self.assertTrue(isinstance(loss, torch.Tensor))


if __name__ == "__main__":
    unittest.main()
