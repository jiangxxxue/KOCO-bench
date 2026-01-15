#!/usr/bin/env python3
# Copyright 2024 ByteMLPerf. All rights reserved.

"""
Test model_provider function
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
from megatron.inference.gpt.model_provider import model_provider


class TestModelProvider(unittest.TestCase):
    """Testmodel_providerfunction"""
    
    def setUp(self):
        """Set up test environment"""
        torch.manual_seed(42)
        
        # Createmock args
        self.mock_args = Mock()
        self.mock_args.use_legacy_models = False
        self.mock_args.spec = None
        self.mock_args.num_experts = 0
        self.mock_args.moe_grouped_gemm = False
        self.mock_args.export_te_mcore_model = False
        self.mock_args.padded_vocab_size = 50257
        self.mock_args.max_position_embeddings = 2048
        self.mock_args.fp16_lm_cross_entropy = False
        self.mock_args.untie_embeddings_and_output_weights = False
        self.mock_args.position_embedding_type = "learned_absolute"
        self.mock_args.rotary_percent = 1.0
        self.mock_args.rotary_base = 10000
        self.mock_args.use_rope_scaling = None
        self.mock_args.load = None
        self.mock_args.export_kd_teacher_load = None
        self.mock_args.export_kd_finalize = False
        self.mock_args.manual_gc = False
        self.mock_args.export_kd_cfg = None
    
    @patch('megatron.inference.gpt.model_provider.get_tensor_model_parallel_rank')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.load_modelopt_state')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    @patch('megatron.inference.gpt.model_provider.get_gpt_layer_modelopt_spec')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider.get_args')
    def test_basic_model_creation(self, mock_get_args, mock_print, mock_config_from_args,
                                  mock_get_spec, mock_model_class, mock_load_state,
                                  mock_add_hooks, mock_get_rank):
        """
        Testcase1: Basic model creation
        
        Input: Default parameters
        Expected behavior:
        1. Create TransformerConfig
        2. Set non_homogeneous_layers=True
        3. Get layer spec
        4. Create MCoreGPTModel
        5. Add hooks
        """
        mock_get_args.return_value = self.mock_args
        mock_get_rank.return_value = 0
        
        # Mock config
        mock_config = Mock()
        mock_config_from_args.return_value = mock_config
        
        # Mock spec
        mock_spec = Mock()
        mock_get_spec.return_value = mock_spec
        
        # Mock model
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        mock_load_state.return_value = {}
        
        # Execute
        result = model_provider(pre_process=True, post_process=True, parallel_output=True)
        
        # Verify: config is created
        mock_config_from_args.assert_called_once_with(self.mock_args)
        
        # Verify: non_homogeneous_layers is set to True
        self.assertTrue(mock_config.non_homogeneous_layers)
        
        # Verify: get_gpt_layer_modelopt_spec is called
        mock_get_spec.assert_called_once()
        call_kwargs = mock_get_spec.call_args[1]
        self.assertEqual(call_kwargs['num_experts'], 0)
        self.assertEqual(call_kwargs['moe_grouped_gemm'], False)
        self.assertEqual(call_kwargs['qk_layernorm'], False)
        
        # Verify: MCoreGPTModel is created
        mock_model_class.assert_called_once()
        model_call_kwargs = mock_model_class.call_args[1]
        self.assertEqual(model_call_kwargs['config'], mock_config)
        self.assertEqual(model_call_kwargs['pre_process'], True)
        self.assertEqual(model_call_kwargs['post_process'], True)
        self.assertEqual(model_call_kwargs['parallel_output'], True)
        
        # Verify: hooks are added
        mock_add_hooks.assert_called_once_with(mock_model)
        
        # Verify: model is returned
        self.assertEqual(result, mock_model)
    
    @patch('megatron.inference.gpt.model_provider.get_tensor_model_parallel_rank')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.load_modelopt_state')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    @patch('megatron.inference.gpt.model_provider.get_gpt_layer_modelopt_spec')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider.get_args')
    def test_legacy_models_raises_error(self, mock_get_args, mock_print, mock_config_from_args,
                                       mock_get_spec, mock_model_class, mock_load_state,
                                       mock_add_hooks, mock_get_rank):
        """
        Testcase2: When use_legacy_models=True, raise error
        
        Input: args.use_legacy_models = True
        Expected: Raise ValueError
        """
        self.mock_args.use_legacy_models = True
        mock_get_args.return_value = self.mock_args
        mock_get_rank.return_value = 0
        
        mock_config = Mock()
        mock_config_from_args.return_value = mock_config
        
        # Execute and verify exception is raised
        with self.assertRaises(ValueError) as context:
            model_provider()
        
        # VerifyError message
        self.assertIn("ModelOpt", str(context.exception))
        self.assertIn("MCore", str(context.exception))
    
    @patch('megatron.inference.gpt.model_provider.get_tensor_model_parallel_rank')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.mto.restore_from_modelopt_state')
    @patch('megatron.inference.gpt.model_provider.load_modelopt_state')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    @patch('megatron.inference.gpt.model_provider.get_gpt_layer_modelopt_spec')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider.get_args')
    def test_restore_from_modelopt_state(self, mock_get_args, mock_print, mock_config_from_args,
                                        mock_get_spec, mock_model_class, mock_load_state,
                                        mock_restore, mock_add_hooks, mock_get_rank):
        """
        Testcase3: Restore model when modelopt_state exists
        
        Input: args.load exists, load_modelopt_state returns non-empty dictionary
        Expected: Call mto.restore_from_modelopt_state
        """
        self.mock_args.load = "/path/to/checkpoint"
        mock_get_args.return_value = self.mock_args
        mock_get_rank.return_value = 0
        
        mock_config = Mock()
        mock_config_from_args.return_value = mock_config
        
        mock_spec = Mock()
        mock_get_spec.return_value = mock_spec
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # Return non-empty state
        mock_modelopt_state = {"state": "data"}
        mock_load_state.return_value = mock_modelopt_state
        
        restored_model = Mock()
        mock_restore.return_value = restored_model
        
        # Execute
        result = model_provider()
        
        # Verify: load_modelopt_state is called
        mock_load_state.assert_called_once()
        
        # Verify: restore_from_modelopt_state is called
        mock_restore.assert_called_once_with(mock_model, mock_modelopt_state)
        
        # Verify: hooks are added to restored model
        mock_add_hooks.assert_called_once_with(restored_model)
    
    @patch('megatron.inference.gpt.model_provider.get_tensor_model_parallel_rank')
    @patch('megatron.inference.gpt.model_provider.distillation.adjust_distillation_model_for_mcore')
    @patch('megatron.inference.gpt.model_provider.distillation.load_distillation_config')
    @patch('megatron.inference.gpt.model_provider._load_teacher_model_config')
    @patch('megatron.inference.gpt.model_provider.mtd.convert')
    @patch('megatron.inference.gpt.model_provider.mto.conversion.get_mode')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.load_modelopt_state')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    @patch('megatron.inference.gpt.model_provider.get_gpt_layer_modelopt_spec')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider.get_args')
    def test_distillation_mode_enabled(self, mock_get_args, mock_print, mock_config_from_args,
                                      mock_get_spec, mock_model_class, mock_load_state,
                                      mock_add_hooks, mock_get_mode, mock_convert,
                                      mock_load_teacher_cfg, mock_load_distill_cfg,
                                      mock_adjust, mock_get_rank):
        """
        Testcase4: Distillation mode enabled
        
        Input: args.export_kd_teacher_load exists
        Expected behavior:
        1. Load teacher configuration
        2. Load distillation configuration
        3. Convert model to DistillationModel
        4. Call adjust_distillation_model_for_mcore
        """
        self.mock_args.export_kd_teacher_load = "/path/to/teacher"
        mock_get_args.return_value = self.mock_args
        mock_get_rank.return_value = 0
        
        mock_config = Mock()
        mock_config_from_args.return_value = mock_config
        
        mock_spec = Mock()
        mock_get_spec.return_value = mock_spec
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        mock_load_state.return_value = {}
        
        # Mock distillation components
        mock_teacher_config = Mock()
        mock_load_teacher_cfg.return_value = mock_teacher_config
        
        mock_distill_cfg = {
            "criterion": Mock(),
            "loss_balancer": Mock()
        }
        mock_load_distill_cfg.return_value = mock_distill_cfg
        
        # Mock model mode as NOT kd_loss
        mock_get_mode.return_value = "normal"
        
        # Mock converted distillation model
        mock_distill_model = Mock(spec=['__class__'])
        mock_distill_model.__class__.__name__ = 'DistillationModel'
        mock_convert.return_value = mock_distill_model
        
        # Execute
        result = model_provider()
        
        # Verify: Load teacher configuration
        mock_load_teacher_cfg.assert_called_once_with(self.mock_args.export_kd_teacher_load)
        
        # Verify: Load distillation configuration
        mock_load_distill_cfg.assert_called_once()
        
        # Verify: mtd.convert is called
        mock_convert.assert_called_once()
        
        # Verify: adjust_distillation_model_for_mcore is called
        mock_adjust.assert_called_once_with(mock_distill_model, mock_distill_cfg)
    
    @patch('megatron.inference.gpt.model_provider.get_tensor_model_parallel_rank')
    @patch('megatron.inference.gpt.model_provider.mtd.export')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.load_modelopt_state')
    @patch('megatron.inference.gpt.model_provider.mtd.DistillationModel')
    @patch('megatron.inference.gpt.model_provider.get_gpt_layer_modelopt_spec')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider.get_args')
    def test_distillation_finalize_export(self, mock_get_args, mock_print, mock_config_from_args,
                                         mock_get_spec, mock_distill_model_class, mock_load_state,
                                         mock_add_hooks, mock_export, mock_get_rank):
        """
        Testcase5: Distillation finalize mode, export student model
        
        Input:
        - args.export_kd_finalize = True
        - model is DistillationModel
        
        Expected: Call mtd.export to export student model
        """
        self.mock_args.export_kd_finalize = True
        mock_get_args.return_value = self.mock_args
        mock_get_rank.return_value = 0
        
        mock_config = Mock()
        mock_config_from_args.return_value = mock_config
        
        mock_spec = Mock()
        mock_get_spec.return_value = mock_spec
        
        # Create a DistillationModel instance
        from unittest.mock import MagicMock
        import modelopt.torch.distill as mtd
        
        mock_model = MagicMock(spec=mtd.DistillationModel)
        # Let isinstance check pass
        mock_model.__class__ = mtd.DistillationModel
        
        # Directly patch MCoreGPTModel, but let it return DistillationModel
        with patch('megatron.inference.gpt.model_provider.MCoreGPTModel', return_value=mock_model):
            mock_load_state.return_value = {}
            
            exported_model = Mock()
            mock_export.return_value = exported_model
            
            # Execute
            result = model_provider()
            
            # Verify: mtd.export is called
            mock_export.assert_called_once_with(mock_model)
            
            # Verify: Exported model is returned
            self.assertEqual(result, exported_model)
    
    @patch('megatron.inference.gpt.model_provider.get_tensor_model_parallel_rank')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.load_modelopt_state')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    @patch('megatron.inference.gpt.model_provider.get_gpt_layer_modelopt_spec')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider.get_args')
    def test_manual_gc_with_distillation_raises_error(self, mock_get_args, mock_print, 
                                                      mock_config_from_args, mock_get_spec,
                                                      mock_model_class, mock_load_state,
                                                      mock_add_hooks, mock_get_rank):
        """
        Testcase6: manual_gc incompatible with distillation
        
        Input:
        - args.export_kd_teacher_load exists
        - args.manual_gc = True
        
        Expected: Raise AssertionError
        """
        self.mock_args.export_kd_teacher_load = "/path/to/teacher"
        self.mock_args.manual_gc = True
        mock_get_args.return_value = self.mock_args
        mock_get_rank.return_value = 0
        
        mock_config = Mock()
        mock_config_from_args.return_value = mock_config
        
        mock_spec = Mock()
        mock_get_spec.return_value = mock_spec
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        mock_load_state.return_value = {}
        
        # Execute and verify exception is raised
        with self.assertRaises(AssertionError) as context:
            model_provider()
        
        # VerifyError message
        self.assertIn("manual-gc", str(context.exception))
    
    @patch('megatron.inference.gpt.model_provider.get_tensor_model_parallel_rank')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.load_modelopt_state')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    @patch('megatron.inference.gpt.model_provider.get_gpt_layer_modelopt_spec')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider.get_args')
    def test_model_kwargs_construction(self, mock_get_args, mock_print, mock_config_from_args,
                                      mock_get_spec, mock_model_class, mock_load_state,
                                      mock_add_hooks, mock_get_rank):
        """
        Testcase7: model_kwargs correctly constructed
        
        Verify kwargs passed to MCoreGPTModel contain all required fields
        """
        mock_get_args.return_value = self.mock_args
        mock_get_rank.return_value = 0
        
        mock_config = Mock()
        mock_config_from_args.return_value = mock_config
        
        mock_spec = Mock()
        mock_get_spec.return_value = mock_spec
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        mock_load_state.return_value = {}
        
        # Execute
        result = model_provider(pre_process=False, post_process=True, parallel_output=False)
        
        # Verifymodel_kwargs
        call_kwargs = mock_model_class.call_args[1]
        
        # Verify required fields
        self.assertIn('transformer_layer_spec', call_kwargs)
        self.assertIn('vocab_size', call_kwargs)
        self.assertIn('max_sequence_length', call_kwargs)
        self.assertIn('pre_process', call_kwargs)
        self.assertIn('post_process', call_kwargs)
        self.assertIn('parallel_output', call_kwargs)
        self.assertIn('share_embeddings_and_output_weights', call_kwargs)
        self.assertIn('position_embedding_type', call_kwargs)
        self.assertIn('rotary_percent', call_kwargs)
        self.assertIn('rotary_base', call_kwargs)
        
        # Verify values
        self.assertEqual(call_kwargs['vocab_size'], 50257)
        self.assertEqual(call_kwargs['max_sequence_length'], 2048)
        self.assertEqual(call_kwargs['pre_process'], False)
        self.assertEqual(call_kwargs['post_process'], True)
        self.assertEqual(call_kwargs['parallel_output'], False)


class TestModelProviderEdgeCases(unittest.TestCase):
    """TestEdge cases"""
    
    def setUp(self):
        torch.manual_seed(42)
        
        self.mock_args = Mock()
        self.mock_args.use_legacy_models = False
        self.mock_args.spec = None
        self.mock_args.num_experts = 0
        self.mock_args.moe_grouped_gemm = False
        self.mock_args.export_te_mcore_model = False
        self.mock_args.padded_vocab_size = 50257
        self.mock_args.max_position_embeddings = 2048
        self.mock_args.fp16_lm_cross_entropy = False
        self.mock_args.untie_embeddings_and_output_weights = False
        self.mock_args.position_embedding_type = "learned_absolute"
        self.mock_args.rotary_percent = 1.0
        self.mock_args.rotary_base = 10000
        self.mock_args.use_rope_scaling = None
        self.mock_args.load = None
        self.mock_args.export_kd_teacher_load = None
        self.mock_args.export_kd_finalize = False
        self.mock_args.manual_gc = False
        self.mock_args.export_kd_cfg = None
    
    @patch('megatron.inference.gpt.model_provider.get_tensor_model_parallel_rank')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.load_modelopt_state')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    @patch('megatron.inference.gpt.model_provider.get_gpt_layer_modelopt_spec')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider.get_args')
    def test_non_zero_tp_rank(self, mock_get_args, mock_print, mock_config_from_args,
                              mock_get_spec, mock_model_class, mock_load_state,
                              mock_add_hooks, mock_get_rank):
        """
        Testcase8: Non-zero TP rank does not print model
        
        Input: get_tensor_model_parallel_rank() > 0
        Expected: Do not print model structure
        """
        mock_get_args.return_value = self.mock_args
        mock_get_rank.return_value = 1  # Non-zero rank
        
        mock_config = Mock()
        mock_config_from_args.return_value = mock_config
        
        mock_spec = Mock()
        mock_get_spec.return_value = mock_spec
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        mock_load_state.return_value = {}
        
        # Execute
        result = model_provider()
        
        # Verify: Model is still created
        self.assertEqual(result, mock_model)


if __name__ == "__main__":
    unittest.main()
