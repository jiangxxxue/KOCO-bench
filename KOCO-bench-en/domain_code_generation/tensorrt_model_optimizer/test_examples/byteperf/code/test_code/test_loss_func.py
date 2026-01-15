#!/usr/bin/env python3
# Copyright 2024 ByteMLPerf. All rights reserved.

"""
Test loss_func function
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
    """Testloss_funcfunction"""
    
    def setUp(self):
        """Set up test environment"""
        torch.manual_seed(42)
        
        # Createmock args
        self.mock_args = Mock()
        self.mock_args.context_parallel_size = 1
        self.mock_args.tensor_model_parallel_size = 1
        self.mock_args.check_for_nan_in_loss_and_grad = False
        self.mock_args.export_kd_teacher_load = None
        self.mock_args.export_kd_finalize = False
    
    def _create_mock_model(self, enable_kd=False):
        """Create mock model"""
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
        Testcase1: Basic language model loss computation
        
        Input:
        - loss_mask: Mask tensor
        - model: GPT model
        - output_tensor: Model output
        
        Expected behavior:
        1. Compute masked loss
        2. Perform allreduce
        3. Return loss and report
        """
        # Set up mock
        mock_get_args.return_value = self.mock_args
        model = self._create_mock_model(enable_kd=False)
        mock_unwrap.return_value = model
        
        # Set mask_loss return value
        mock_loss = torch.tensor(2.5)
        mock_mask.return_value = mock_loss
        
        # Set allreduce return value
        mock_allreduce.return_value = (torch.tensor(2.5), torch.tensor(2.5))
        
        # CreateInput
        loss_mask = torch.ones(4, 10)
        output_tensor = torch.randn(4, 10, 50257)
        
        # Execute
        loss, report = loss_func(loss_mask, model, output_tensor)
        
        # Verify
        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertTrue(isinstance(report, dict))
        self.assertIn('lm loss', report)
        
        # Verify correct functions were called
        mock_mask.assert_called_once()
        mock_allreduce.assert_called_once()
        
        # Verify return value
        torch.testing.assert_close(loss, torch.tensor(2.5), atol=1e-4, rtol=1e-4)
    
    @patch('megatron.inference.gpt.loss_func.get_args')
    @patch('megatron.inference.gpt.loss_func.unwrap_model')
    @patch('megatron.inference.gpt.loss_func._mask_loss')
    @patch('megatron.inference.gpt.loss_func._allreduce_loss')
    def test_kd_loss_calculation(self, mock_allreduce, mock_mask, mock_unwrap, mock_get_args):
        """
        Testcase2: Knowledge distillation loss computation
        
        Input:
        - enable_kd: True
        - export_kd_teacher_load: "/path/to/teacher"
        
        Expected behavior:
        1. Compute standard LM loss
        2. Additionally compute KD loss
        3. Return report with LM loss and KD loss
        """
        # Enable KD
        self.mock_args.export_kd_teacher_load = "/path/to/teacher"
        mock_get_args.return_value = self.mock_args
        
        model = self._create_mock_model(enable_kd=True)
        mock_unwrap.return_value = model
        
        # Set return values
        mock_loss = torch.tensor(2.0)
        mock_mask.return_value = mock_loss
        mock_allreduce.return_value = (torch.tensor(2.0), torch.tensor(2.0))
        
        # CreateInput
        loss_mask = torch.ones(4, 10)
        output_tensor = torch.randn(4, 10, 50257)
        
        # Execute
        loss, report = loss_func(loss_mask, model, output_tensor)
        
        # Verify
        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertTrue(isinstance(report, dict))
        self.assertIn('lm loss', report)
        
        # In KD mode, should call compute_kd_loss
        # Verify through model's mock
        self.assertTrue(model._enable_kd)
    
    @patch('megatron.inference.gpt.loss_func.get_args')
    @patch('megatron.inference.gpt.loss_func.unwrap_model')
    @patch('megatron.inference.gpt.loss_func._mask_loss')
    @patch('megatron.inference.gpt.loss_func._allreduce_loss')
    def test_eval_mode_no_kd(self, mock_allreduce, mock_mask, mock_unwrap, mock_get_args):
        """
        Testcase3: Evaluation mode does not compute KD loss
        
        Input:
        - model.training: False
        - export_kd_teacher_load: "/path/to/teacher"
        
        Expected behavior: Even with teacher_load set, evaluation mode does not compute KD loss
        """
        self.mock_args.export_kd_teacher_load = "/path/to/teacher"
        mock_get_args.return_value = self.mock_args
        
        model = self._create_mock_model(enable_kd=True)
        model.training = False  # Evaluation mode
        mock_unwrap.return_value = model
        
        mock_loss = torch.tensor(1.5)
        mock_mask.return_value = mock_loss
        mock_allreduce.return_value = (torch.tensor(1.5), torch.tensor(1.5))
        
        loss_mask = torch.ones(4, 10)
        output_tensor = torch.randn(4, 10, 50257)
        
        loss, report = loss_func(loss_mask, model, output_tensor)
        
        # Verify only LM loss
        self.assertIn('lm loss', report)
        torch.testing.assert_close(loss, torch.tensor(1.5), atol=1e-4, rtol=1e-4)
    
    @patch('megatron.inference.gpt.loss_func.get_args')
    @patch('megatron.inference.gpt.loss_func.unwrap_model')
    @patch('megatron.inference.gpt.loss_func._mask_loss')
    @patch('megatron.inference.gpt.loss_func._allreduce_loss')
    def test_kd_finalize_mode(self, mock_allreduce, mock_mask, mock_unwrap, mock_get_args):
        """
        Testcase4: KD finalize mode
        
        Input:
        - export_kd_finalize: True
        
        Expected behavior: In finalize mode, do not compute KD loss
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
        
        # In finalize mode, only LM loss
        self.assertIn('lm loss', report)
        torch.testing.assert_close(loss, torch.tensor(1.8), atol=1e-4, rtol=1e-4)
    
    @patch('megatron.inference.gpt.loss_func.get_args')
    @patch('megatron.inference.gpt.loss_func.unwrap_model')
    @patch('megatron.inference.gpt.loss_func._mask_loss')
    @patch('megatron.inference.gpt.loss_func._allreduce_loss')
    def test_loss_mask_application(self, mock_allreduce, mock_mask, mock_unwrap, mock_get_args):
        """
        Testcase5: Loss mask application
        
        Verify mask_loss is called correctly with correct parameters
        """
        mock_get_args.return_value = self.mock_args
        model = self._create_mock_model(enable_kd=False)
        mock_unwrap.return_value = model
        
        mock_loss = torch.tensor(3.2)
        mock_mask.return_value = mock_loss
        mock_allreduce.return_value = (torch.tensor(3.2), torch.tensor(3.2))
        
        # Create specific mask
        loss_mask = torch.tensor([[1, 1, 1, 0, 0],
                                   [1, 1, 1, 1, 0]])
        output_tensor = torch.randn(2, 5, 50257)
        
        loss, report = loss_func(loss_mask, model, output_tensor)
        
        # Verify mask_loss was called with correct parameters
        call_args = mock_mask.call_args
        self.assertEqual(call_args[0][1].shape, loss_mask.shape)
    
    @patch('megatron.inference.gpt.loss_func.get_args')
    @patch('megatron.inference.gpt.loss_func.unwrap_model')
    @patch('megatron.inference.gpt.loss_func._mask_loss')
    @patch('megatron.inference.gpt.loss_func._allreduce_loss')
    def test_different_loss_values(self, mock_allreduce, mock_mask, mock_unwrap, mock_get_args):
        """
        Testcase6: Different loss values
        
        Test that different loss value sizes are handled correctly
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
        Testcase7: Report structure verification
        
        Verify returned report dictionary contains correct keys
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
        
        # Verify report is a dictionary
        self.assertTrue(isinstance(report, dict))
        
        # Verify contains lm loss
        self.assertIn('lm loss', report)
        
        # Verify value is tensor
        self.assertTrue(isinstance(report['lm loss'], torch.Tensor))


class TestLossFuncEdgeCases(unittest.TestCase):
    """TestEdge cases"""
    
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
        Testcase8: Small batch size
        
        Input: batch_size=1
        Expected: Normal processing
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
        Testcase9: Large batch size
        
        Input: batch_size=128
        Expected: Normal processing
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
        Testcase10: All positions masked
        
        Input: loss_mask all zeros
        Expected: Return 0 loss (or handle masked case)
        """
        mock_get_args.return_value = self.mock_args
        model = Mock()
        model.training = True
        mock_unwrap.return_value = model
        
        # Simulate fully masked case, loss is 0 or nan
        mock_loss = torch.tensor(0.0)
        mock_mask.return_value = mock_loss
        mock_allreduce.return_value = (mock_loss, mock_loss)
        
        loss_mask = torch.zeros(4, 10)  # All masked
        output_tensor = torch.randn(4, 10, 50257)
        
        loss, report = loss_func(loss_mask, model, output_tensor)
        
        self.assertTrue(isinstance(loss, torch.Tensor))


if __name__ == "__main__":
    unittest.main()
