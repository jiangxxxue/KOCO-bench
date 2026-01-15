#!/usr/bin/env python3
# Copyright 2024 FlagScale. All rights reserved.

"""
Test loss_func function
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
    print(f"Warning: Unable to import implementation code: {e}")
    IMPORT_SUCCESS = False


class TestLossFunc(unittest.TestCase):
    """Test loss_func function"""
    
    def setUp(self):
        """Setup test environment"""
        torch.manual_seed(42)
        
        # Mock args
        self.mock_args = Mock()
        self.mock_args.check_for_nan_in_loss_and_grad = False
        self.mock_args.check_for_spiky_loss = False
        
        # Mock rerun_state_machine
        self.mock_rerun_state_machine = Mock()
        self.mock_rerun_state_machine.validate_result = Mock()
        self.mock_rerun_state_machine.is_unexpectedly_large = Mock(return_value=False)
        
        # Create test data
        self.loss_mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])  # 3 valid tokens
        self.output_tensor = torch.tensor([2.5, 3.0, 1.0, 4.0, 0.5])  # loss values
        
        # Mock model
        self.mock_model = Mock()
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_rerun_state_machine')
    @patch('flagscale.train.train_gpt.has_nvidia_modelopt', False)
    def test_applies_loss_mask(self, mock_get_rerun, mock_get_args):
        """
        Test case 1: Apply loss_mask to filter losses
        
        Key verification: Loss calculation should use loss_mask for filtering
        """
        # Setup mocks
        mock_get_args.return_value = self.mock_args
        mock_get_rerun.return_value = self.mock_rerun_state_machine
        
        # Execute
        loss, num_tokens, metrics = loss_func(self.loss_mask, self.output_tensor, self.mock_model)
        
        # Verify: loss should be sum of masked losses
        # Valid tokens: indices 0, 1, 3 -> losses: 2.5, 3.0, 4.0 -> sum = 9.5
        expected_loss = torch.tensor(9.5)
        torch.testing.assert_close(loss, expected_loss, rtol=1e-5)
        
        # Verify: num_tokens should be sum of loss_mask
        expected_num_tokens = torch.tensor(3, dtype=torch.int)
        torch.testing.assert_close(num_tokens, expected_num_tokens)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_rerun_state_machine')
    @patch('flagscale.train.train_gpt.has_nvidia_modelopt', False)
    def test_computes_num_tokens_correctly(self, mock_get_rerun, mock_get_args):
        """
        Test case 2: Correctly compute token count
        
        Key verification: num_tokens should be loss_mask.sum()
        """
        # Setup mocks
        mock_get_args.return_value = self.mock_args
        mock_get_rerun.return_value = self.mock_rerun_state_machine
        
        # Execute
        loss, num_tokens, metrics = loss_func(self.loss_mask, self.output_tensor, self.mock_model)
        
        # Verify: num_tokens = loss_mask.sum()
        expected_num_tokens = self.loss_mask.sum().to(torch.int)
        torch.testing.assert_close(num_tokens, expected_num_tokens)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_rerun_state_machine')
    @patch('flagscale.train.train_gpt.has_nvidia_modelopt', False)
    def test_returns_correct_structure(self, mock_get_rerun, mock_get_args):
        """
        Test case 3: Return correct data structure
        
        Key verification: Should return (loss, num_tokens, metrics dict)
        """
        # Setup mocks
        mock_get_args.return_value = self.mock_args
        mock_get_rerun.return_value = self.mock_rerun_state_machine
        
        # Execute
        result = loss_func(self.loss_mask, self.output_tensor, self.mock_model)
        
        # Verify: Returns tuple
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        
        loss, num_tokens, metrics = result
        
        # Verify: loss is scalar tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # scalar
        
        # Verify: num_tokens is scalar tensor
        self.assertIsInstance(num_tokens, torch.Tensor)
        self.assertEqual(num_tokens.dtype, torch.int)
        
        # Verify: metrics is dict containing 'lm loss'
        self.assertIsInstance(metrics, dict)
        self.assertIn('lm loss', metrics)
        self.assertIsInstance(metrics['lm loss'], torch.Tensor)
        self.assertEqual(metrics['lm loss'].shape[0], 2)  # [loss, num_tokens]
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_rerun_state_machine')
    @patch('flagscale.train.train_gpt.has_nvidia_modelopt', True)
    @patch('flagscale.train.train_gpt.modelopt_args_enabled')
    @patch('flagscale.train.train_gpt.loss_func_modelopt')
    def test_calls_modelopt_loss_when_enabled(self, mock_loss_func_modelopt, mock_modelopt_enabled, 
                                               mock_get_rerun, mock_get_args):
        """
        Test case 4: Call modelopt's loss_func when ModelOpt is enabled
        
        Key verification: When has_nvidia_modelopt=True and modelopt_args_enabled=True, should call loss_func_modelopt
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
        
        # Verify: Called loss_func_modelopt
        mock_loss_func_modelopt.assert_called_once_with(
            self.loss_mask, 
            self.output_tensor, 
            model=self.mock_model
        )
        
        # Verify: Returned modelopt's result
        self.assertEqual(result[0].item(), 5.0)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_rerun_state_machine')
    @patch('flagscale.train.train_gpt.has_nvidia_modelopt', False)
    def test_checks_for_nan_when_enabled(self, mock_get_rerun, mock_get_args):
        """
        Test case 5: Call validate_result when NaN check is enabled
        
        Key verification: When check_for_nan_in_loss_and_grad=True, should call validate_result to check NaN
        """
        # Setup mocks
        self.mock_args.check_for_nan_in_loss_and_grad = True
        mock_get_args.return_value = self.mock_args
        mock_get_rerun.return_value = self.mock_rerun_state_machine
        
        # Execute
        loss_func(self.loss_mask, self.output_tensor, self.mock_model)
        
        # Verify: Called validate_result to check NaN
        self.assertEqual(mock_get_rerun_state_machine.validate_result.call_count, 2)  # NaN and Inf checks
        
        # Verify NaN check call
        nan_call = mock_get_rerun_state_machine.validate_result.call_args_list[0]
        self.assertEqual(nan_call[1]['rejection_func'], torch.isnan)
        self.assertEqual(nan_call[1]['message'], "found NaN in local forward loss calculation")
        self.assertEqual(nan_call[1]['fatal'], True)
        
        # Verify Inf check call
        inf_call = mock_get_rerun_state_machine.validate_result.call_args_list[1]
        self.assertEqual(inf_call[1]['rejection_func'], torch.isinf)
        self.assertEqual(inf_call[1]['message'], "found Inf in local forward loss calculation")
        self.assertEqual(inf_call[1]['fatal'], True)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_rerun_state_machine')
    @patch('flagscale.train.train_gpt.has_nvidia_modelopt', False)
    def test_checks_for_spiky_loss_when_enabled(self, mock_get_rerun, mock_get_args):
        """
        Test case 6: Call validate_result when spiky loss check is enabled
        
        Key verification: When check_for_spiky_loss=True, should call validate_result to check spiky loss
        """
        # Setup mocks
        self.mock_args.check_for_spiky_loss = True
        mock_get_args.return_value = self.mock_args
        mock_get_rerun.return_value = self.mock_rerun_state_machine
        
        # Execute
        loss_func(self.loss_mask, self.output_tensor, self.mock_model)
        
        # Verify: Called validate_result to check spiky loss
        # Should have at least one call (may also have NaN check)
        call_count = mock_get_rerun_state_machine.validate_result.call_count
        self.assertGreaterEqual(call_count, 1)
        
        # Find spiky loss check call
        spiky_calls = [
            call for call in mock_get_rerun_state_machine.validate_result.call_args_list
            if call[1].get('message') == 'Spiky loss'
        ]
        self.assertGreater(len(spiky_calls), 0, "Should call validate_result to check spiky loss")
        
        spiky_call = spiky_calls[0]
        self.assertEqual(spiky_call[1]['message'], "Spiky loss")
        self.assertEqual(spiky_call[1]['fatal'], False)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_rerun_state_machine')
    @patch('flagscale.train.train_gpt.has_nvidia_modelopt', False)
    def test_loss_mask_view_operation(self, mock_get_rerun, mock_get_args):
        """
        Test case 7: view operation on loss_mask and output_tensor
        
        Key verification: Should perform view(-1) operation on loss_mask and output_tensor
        """
        # Setup mocks
        mock_get_args.return_value = self.mock_args
        mock_get_rerun.return_value = self.mock_rerun_state_machine
        
        # Test with 2D tensor
        loss_mask_2d = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
        output_tensor_2d = torch.tensor([[2.0, 3.0], [1.0, 4.0]])
        
        # Execute
        loss, num_tokens, metrics = loss_func(loss_mask_2d, output_tensor_2d, self.mock_model)
        
        # Verify: Calculation result is correct (should flatten before calculation)
        # Valid values: [2.0, 3.0, 4.0] -> sum = 9.0
        expected_loss = torch.tensor(9.0)
        torch.testing.assert_close(loss, expected_loss, rtol=1e-5)
        
        expected_num_tokens = torch.tensor(3, dtype=torch.int)
        torch.testing.assert_close(num_tokens, expected_num_tokens)


if __name__ == "__main__":
    unittest.main()
