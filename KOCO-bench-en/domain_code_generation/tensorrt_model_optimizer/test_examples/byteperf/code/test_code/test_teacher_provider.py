#!/usr/bin/env python3
# Copyright 2024 ByteMLPerf. All rights reserved.

"""
Test _teacher_provider function
"""

import torch
import torch.nn as nn
import unittest
from unittest.mock import Mock, MagicMock, patch
from argparse import Namespace
import sys
import os

# Mock transformer_engine before any imports
sys.modules['transformer_engine'] = MagicMock()
sys.modules['transformer_engine.pytorch'] = MagicMock()

# Add the parent directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'byte_train_perf', 'Megatron-LM'))

# Import ground truth implementation
from megatron.inference.gpt.model_provider import _teacher_provider


class TestTeacherProvider(unittest.TestCase):
    """Test_teacher_providerfunction"""
    
    def setUp(self):
        """Set up test environment"""
        torch.manual_seed(42)
        
        # Createmock args
        self.mock_args = Mock()
        self.mock_args.finetune = False
        self.mock_args.export_kd_teacher_load = "/fake/path/to/teacher"
        self.mock_args.hidden_size = 768
        self.mock_args.num_layers = 12
        self.mock_args.num_attention_heads = 12
        self.mock_args.ffn_hidden_size = 3072
        self.mock_args.max_position_embeddings = 2048
        self.mock_args.padded_vocab_size = 50257
        self.mock_args.tensor_model_parallel_size = 1
        self.mock_args.pipeline_model_parallel_size = 1
        self.mock_args.sequence_parallel = False
        self.mock_args.use_cpu_initialization = False
        self.mock_args.params_dtype = torch.float32
        self.mock_args.perform_initialization = True
    
    @patch('megatron.inference.gpt.model_provider.load_modelopt_checkpoint')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.get_args')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    def test_teacher_creation_with_config(self, mock_model_class, mock_get_args, 
                                         mock_config_from_args, mock_add_hooks, 
                                         mock_print, mock_load_ckpt):
        """
        Testcase1: Create teacher model with given configuration
        
        Input:
        - config: Namespace object
        - model_kwargs: Model parameters
        
        Expected behavior:
        1. Call core_transformer_config_from_args to convert configuration
        2. Set non_homogeneous_layers=True
        3. Create MCoreGPTModel
        4. Add load convert hooks
        5. Load checkpoint
        """
        mock_get_args.return_value = self.mock_args
        
        # Mock TransformerConfig
        mock_transformer_config = Mock()
        mock_transformer_config.hidden_size = 768
        mock_transformer_config.num_layers = 12
        mock_config_from_args.return_value = mock_transformer_config
        
        # Mock model instance
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # CreateInput
        config = Namespace(
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12
        )
        model_kwargs = {
            'pre_process': True,
            'post_process': True
        }
        
        # Execute
        result = _teacher_provider(config, model_kwargs)
        
        # Verify: core_transformer_config_from_args is called
        mock_config_from_args.assert_called_once_with(config)
        
        # Verify: non_homogeneous_layers is set to True
        self.assertTrue(mock_transformer_config.non_homogeneous_layers)
        
        # Verify: MCoreGPTModel is created with correct parameters
        mock_model_class.assert_called_once()
        call_kwargs = mock_model_class.call_args[1]
        self.assertEqual(call_kwargs['config'], mock_transformer_config)
        self.assertEqual(call_kwargs['pre_process'], True)
        self.assertEqual(call_kwargs['post_process'], True)
        
        # Verify: add_load_convert_hooks is called
        mock_add_hooks.assert_called_once_with(mock_model)
        
        # Verify: print_rank_0 is called
        mock_print.assert_called()
        
        # Verify: load_modelopt_checkpoint is called
        mock_load_ckpt.assert_called_once()
        load_args = mock_load_ckpt.call_args
        self.assertEqual(load_args[0][0][0], mock_model)  # First parameter is [teacher]
        self.assertEqual(load_args[1]['load_arg'], 'export_kd_teacher_load')
        
        # Verify: Teacher model is returned
        self.assertEqual(result, mock_model)
    
    @patch('megatron.inference.gpt.model_provider.load_modelopt_checkpoint')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.get_args')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    def test_finetune_flag_temporarily_set(self, mock_model_class, mock_get_args,
                                          mock_config_from_args, mock_add_hooks,
                                          mock_print, mock_load_ckpt):
        """
        Testcase2: finetune flag temporarily set to True
        
        Input: args.finetune = False
        Expected behavior:
        1. Temporarily set finetune=True when loading checkpoint
        2. Restore original value after loading
        """
        self.mock_args.finetune = False
        original_finetune = self.mock_args.finetune
        mock_get_args.return_value = self.mock_args
        
        mock_transformer_config = Mock()
        mock_config_from_args.return_value = mock_transformer_config
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # Track finetune changes
        finetune_during_load = []
        
        def track_finetune(*args, **kwargs):
            finetune_during_load.append(self.mock_args.finetune)
        
        mock_load_ckpt.side_effect = track_finetune
        
        config = Namespace(hidden_size=768)
        model_kwargs = {}
        
        # Execute
        result = _teacher_provider(config, model_kwargs)
        
        # Verify: finetune is set to True during loading
        self.assertTrue(len(finetune_during_load) > 0)
        self.assertTrue(finetune_during_load[0])  # Should be True during loading
        
        # Verify: Original value is restored after loading
        self.assertEqual(self.mock_args.finetune, original_finetune)
    
    @patch('megatron.inference.gpt.model_provider.load_modelopt_checkpoint')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.get_args')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    def test_non_homogeneous_layers_always_true(self, mock_model_class, mock_get_args,
                                                mock_config_from_args, mock_add_hooks,
                                                mock_print, mock_load_ckpt):
        """
        Testcase3: non_homogeneous_layers always set to True
        
        Verify non_homogeneous_layers attribute of configuration is correctly set
        """
        mock_get_args.return_value = self.mock_args
        
        mock_transformer_config = Mock()
        mock_transformer_config.non_homogeneous_layers = False  # Initially False
        mock_config_from_args.return_value = mock_transformer_config
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        config = Namespace(hidden_size=768)
        model_kwargs = {}
        
        # Execute
        result = _teacher_provider(config, model_kwargs)
        
        # Verify: non_homogeneous_layers is set to True
        self.assertTrue(mock_transformer_config.non_homogeneous_layers)
    
    @patch('megatron.inference.gpt.model_provider.load_modelopt_checkpoint')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.get_args')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    def test_model_kwargs_passed_correctly(self, mock_model_class, mock_get_args,
                                          mock_config_from_args, mock_add_hooks,
                                          mock_print, mock_load_ckpt):
        """
        Testcase4: model_kwargs correctly passed to MCoreGPTModel
        
        Test different model_kwargs combinations
        """
        mock_get_args.return_value = self.mock_args
        
        mock_transformer_config = Mock()
        mock_config_from_args.return_value = mock_transformer_config
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # Test different kwargs combinations
        test_cases = [
            {'pre_process': True, 'post_process': True},
            {'pre_process': False, 'post_process': True},
            {'pre_process': True, 'post_process': False, 'parallel_output': False},
        ]
        
        for model_kwargs in test_cases:
            with self.subTest(model_kwargs=model_kwargs):
                mock_model_class.reset_mock()
                
                config = Namespace(hidden_size=768)
                
                result = _teacher_provider(config, model_kwargs)
                
                # Verify kwargs are correctly passed
                call_kwargs = mock_model_class.call_args[1]
                for key, value in model_kwargs.items():
                    self.assertEqual(call_kwargs[key], value)
    
    @patch('megatron.inference.gpt.model_provider.load_modelopt_checkpoint')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.get_args')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    def test_print_loading_message(self, mock_model_class, mock_get_args,
                                   mock_config_from_args, mock_add_hooks,
                                   mock_print, mock_load_ckpt):
        """
        Testcase5: Print message for loading teacher checkpoint
        
        Verify print_rank_0 is called to output loading information
        """
        mock_get_args.return_value = self.mock_args
        
        mock_transformer_config = Mock()
        mock_config_from_args.return_value = mock_transformer_config
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        config = Namespace(hidden_size=768)
        model_kwargs = {}
        
        # Execute
        result = _teacher_provider(config, model_kwargs)
        
        # Verify print_rank_0 is called
        mock_print.assert_called()
        
        # Verify message content
        call_args = [str(call[0][0]) for call in mock_print.call_args_list]
        self.assertTrue(any('teacher' in arg.lower() and 'checkpoint' in arg.lower() 
                           for arg in call_args))
    
    @patch('megatron.inference.gpt.model_provider.load_modelopt_checkpoint')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.get_args')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    def test_hooks_added_before_loading(self, mock_model_class, mock_get_args,
                                       mock_config_from_args, mock_add_hooks,
                                       mock_print, mock_load_ckpt):
        """
        Testcase6: hooks added before loading checkpoint
        
        Verify _add_load_convert_hooks is called before load_modelopt_checkpoint
        """
        mock_get_args.return_value = self.mock_args
        
        mock_transformer_config = Mock()
        mock_config_from_args.return_value = mock_transformer_config
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # Track call order
        call_order = []
        
        def track_add_hooks(model):
            call_order.append('add_hooks')
        
        def track_load_ckpt(*args, **kwargs):
            call_order.append('load_ckpt')
        
        mock_add_hooks.side_effect = track_add_hooks
        mock_load_ckpt.side_effect = track_load_ckpt
        
        config = Namespace(hidden_size=768)
        model_kwargs = {}
        
        # Execute
        result = _teacher_provider(config, model_kwargs)
        
        # Verify order: add_hooks -> load_ckpt
        self.assertEqual(call_order, ['add_hooks', 'load_ckpt'])


class TestTeacherProviderEdgeCases(unittest.TestCase):
    """TestEdge cases"""
    
    def setUp(self):
        torch.manual_seed(42)
        
        self.mock_args = Mock()
        self.mock_args.finetune = False
        self.mock_args.export_kd_teacher_load = "/fake/path/to/teacher"
    
    @patch('megatron.inference.gpt.model_provider.load_modelopt_checkpoint')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.get_args')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    def test_finetune_already_true(self, mock_model_class, mock_get_args,
                                   mock_config_from_args, mock_add_hooks,
                                   mock_print, mock_load_ckpt):
        """
        Testcase7: finetune already True case
        
        Verify original value is correctly restored
        """
        self.mock_args.finetune = True  # Already True
        mock_get_args.return_value = self.mock_args
        
        mock_transformer_config = Mock()
        mock_config_from_args.return_value = mock_transformer_config
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        config = Namespace(hidden_size=768)
        model_kwargs = {}
        
        # Execute
        result = _teacher_provider(config, model_kwargs)
        
        # Verify: finetune is still True after loading
        self.assertTrue(self.mock_args.finetune)
    
    @patch('megatron.inference.gpt.model_provider.load_modelopt_checkpoint')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.get_args')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    def test_empty_model_kwargs(self, mock_model_class, mock_get_args,
                                mock_config_from_args, mock_add_hooks,
                                mock_print, mock_load_ckpt):
        """
        Testcase8: Empty model_kwargs
        
        Verify empty kwargs can be handled correctly
        """
        mock_get_args.return_value = self.mock_args
        
        mock_transformer_config = Mock()
        mock_config_from_args.return_value = mock_transformer_config
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        config = Namespace(hidden_size=768)
        model_kwargs = {}  # Empty kwargs
        
        # Execute
        result = _teacher_provider(config, model_kwargs)
        
        # Verify: Function executes normally, returns model
        self.assertEqual(result, mock_model)


if __name__ == "__main__":
    unittest.main()
