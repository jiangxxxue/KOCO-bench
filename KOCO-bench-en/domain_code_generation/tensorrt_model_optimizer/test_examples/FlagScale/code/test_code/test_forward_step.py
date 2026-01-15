#!/usr/bin/env python3
# Copyright 2024 FlagScale. All rights reserved.

"""
Test forward_step function
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
    from flagscale.train.train_gpt import forward_step, get_batch, loss_func
    IMPORT_SUCCESS = True
except Exception as e:
    print(f"Warning: Unable to import implementation code: {e}")
    IMPORT_SUCCESS = False


class TestForwardStep(unittest.TestCase):
    """Test forward_step function"""
    
    def setUp(self):
        """Setup test environment"""
        torch.manual_seed(42)
        
        # Mock args
        self.mock_args = Mock()
        self.mock_args.use_legacy_models = False
        
        # Mock timers
        self.mock_timer = Mock()
        self.mock_timer.start = Mock()
        self.mock_timer.stop = Mock()
        self.mock_timers = Mock(return_value=self.mock_timer)
        
        # Mock get_batch return value
        self.tokens = torch.randint(0, 1000, (10, 8))  # seq_len=10, batch_size=8
        self.labels = torch.randint(0, 1000, (10, 8))
        self.loss_mask = torch.ones(10, 8, dtype=torch.float)
        self.attention_mask = torch.ones(10, 8, dtype=torch.bool)
        self.position_ids = torch.arange(10).unsqueeze(0).expand(8, -1)
        
        # Mock model
        self.mock_model = Mock()
        self.mock_output_tensor = torch.randn(10, 8, 1000)  # seq_len, batch, vocab_size
        self.mock_model.return_value = self.mock_output_tensor
        self.mock_model.__call__ = Mock(return_value=self.mock_output_tensor)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_timers')
    @patch('flagscale.train.train_gpt.get_batch')
    @patch('flagscale.train.train_gpt.stimer')
    def test_calls_get_batch(self, mock_stimer, mock_get_batch, mock_get_timers, mock_get_args):
        """
        Test case 1: Call get_batch to obtain data
        
        Key verification: Should call get_batch(data_iterator) to get training data
        """
        # Setup mocks
        mock_get_args.return_value = self.mock_args
        mock_get_timers.return_value = self.mock_timers
        mock_get_batch.return_value = (
            self.tokens, self.labels, self.loss_mask, 
            self.attention_mask, self.position_ids
        )
        mock_stimer.__enter__ = Mock()
        mock_stimer.__exit__ = Mock(return_value=False)
        
        # Mock data_iterator
        data_iterator = Mock()
        
        # Execute
        output_tensor, partial_loss_func = forward_step(data_iterator, self.mock_model)
        
        # Verify: get_batch was called
        mock_get_batch.assert_called_once_with(data_iterator)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_timers')
    @patch('flagscale.train.train_gpt.get_batch')
    @patch('flagscale.train.train_gpt.stimer')
    def test_core_model_forward_with_loss_mask(self, mock_stimer, mock_get_batch, mock_get_timers, mock_get_args):
        """
        Test case 2: Core model forward call includes loss_mask parameter
        
        Key verification: When use_legacy_models=False, should pass loss_mask parameter
        """
        # Setup mocks
        self.mock_args.use_legacy_models = False
        mock_get_args.return_value = self.mock_args
        mock_get_timers.return_value = self.mock_timers
        mock_get_batch.return_value = (
            self.tokens, self.labels, self.loss_mask, 
            self.attention_mask, self.position_ids
        )
        mock_stimer.__enter__ = Mock()
        mock_stimer.__exit__ = Mock(return_value=False)
        
        data_iterator = Mock()
        
        # Execute
        output_tensor, partial_loss_func = forward_step(data_iterator, self.mock_model)
        
        # Verify: model was called with loss_mask parameter
        self.mock_model.assert_called_once()
        call_kwargs = self.mock_model.call_args[1]
        self.assertIn('loss_mask', call_kwargs, "Core model forward should receive loss_mask parameter")
        self.assertEqual(call_kwargs['loss_mask'], self.loss_mask)
        self.assertEqual(call_kwargs['labels'], self.labels)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_timers')
    @patch('flagscale.train.train_gpt.get_batch')
    @patch('flagscale.train.train_gpt.stimer')
    def test_legacy_model_forward_without_loss_mask(self, mock_stimer, mock_get_batch, mock_get_timers, mock_get_args):
        """
        Test case 3: Legacy model forward call does not include loss_mask parameter
        
        Key verification: When use_legacy_models=True, should not pass loss_mask parameter
        """
        # Setup mocks
        self.mock_args.use_legacy_models = True
        mock_get_args.return_value = self.mock_args
        mock_get_timers.return_value = self.mock_timers
        mock_get_batch.return_value = (
            self.tokens, self.labels, self.loss_mask, 
            self.attention_mask, self.position_ids
        )
        mock_stimer.__enter__ = Mock()
        mock_stimer.__exit__ = Mock(return_value=False)
        
        data_iterator = Mock()
        
        # Execute
        output_tensor, partial_loss_func = forward_step(data_iterator, self.mock_model)
        
        # Verify: model was called but without loss_mask parameter
        self.mock_model.assert_called_once()
        call_kwargs = self.mock_model.call_args[1]
        self.assertNotIn('loss_mask', call_kwargs, "Legacy model forward should not receive loss_mask parameter")
        self.assertEqual(call_kwargs['labels'], self.labels)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_timers')
    @patch('flagscale.train.train_gpt.get_batch')
    @patch('flagscale.train.train_gpt.stimer')
    def test_returns_output_tensor_and_partial_loss_func(self, mock_stimer, mock_get_batch, mock_get_timers, mock_get_args):
        """
        Test case 4: Return output_tensor and partial loss_func
        
        Key verification: Should return (output_tensor, partial(loss_func, loss_mask, model=model))
        """
        # Setup mocks
        mock_get_args.return_value = self.mock_args
        mock_get_timers.return_value = self.mock_timers
        mock_get_batch.return_value = (
            self.tokens, self.labels, self.loss_mask, 
            self.attention_mask, self.position_ids
        )
        mock_stimer.__enter__ = Mock()
        mock_stimer.__exit__ = Mock(return_value=False)
        
        data_iterator = Mock()
        
        # Execute
        output_tensor, partial_loss_func = forward_step(data_iterator, self.mock_model)
        
        # Verify: Returns output_tensor
        self.assertIsNotNone(output_tensor)
        torch.testing.assert_close(output_tensor, self.mock_output_tensor)
        
        # Verify: Returns partial function
        self.assertIsNotNone(partial_loss_func)
        self.assertTrue(callable(partial_loss_func))
        
        # Verify: partial function has bound loss_mask and model
        # Calling partial function should use bound loss_mask and model
        test_output = torch.randn(10, 8, 1000)
        with patch('flagscale.train.train_gpt.get_args', return_value=self.mock_args):
            with patch('flagscale.train.train_gpt.get_rerun_state_machine') as mock_rerun:
                mock_rerun.return_value.validate_result = Mock()
                result = partial_loss_func(test_output)
                
                # Verify returned correct structure
                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 3)
                loss, num_tokens, metrics = result
                self.assertIsInstance(loss, torch.Tensor)
                self.assertIsInstance(num_tokens, torch.Tensor)
                self.assertIsInstance(metrics, dict)
                self.assertIn('lm loss', metrics)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_timers')
    @patch('flagscale.train.train_gpt.get_batch')
    @patch('flagscale.train.train_gpt.stimer')
    def test_model_forward_positional_args(self, mock_stimer, mock_get_batch, mock_get_timers, mock_get_args):
        """
        Test case 5: Verify positional argument order for model forward
        
        Key verification: Positional arguments should be (tokens, position_ids, attention_mask)
        """
        # Setup mocks
        mock_get_args.return_value = self.mock_args
        mock_get_timers.return_value = self.mock_timers
        mock_get_batch.return_value = (
            self.tokens, self.labels, self.loss_mask, 
            self.attention_mask, self.position_ids
        )
        mock_stimer.__enter__ = Mock()
        mock_stimer.__exit__ = Mock(return_value=False)
        
        data_iterator = Mock()
        
        # Execute
        forward_step(data_iterator, self.mock_model)
        
        # Verify: Positional arguments are in correct order
        call_args = self.mock_model.call_args[0]
        self.assertEqual(len(call_args), 3, "Should have 3 positional arguments")
        torch.testing.assert_close(call_args[0], self.tokens)
        torch.testing.assert_close(call_args[1], self.position_ids)
        torch.testing.assert_close(call_args[2], self.attention_mask)


if __name__ == "__main__":
    unittest.main()
