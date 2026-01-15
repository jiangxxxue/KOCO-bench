#!/usr/bin/env python3
# Copyright 2024 FlagScale. All rights reserved.

"""
Test model_provider function
"""

import torch
import unittest
import sys
import os
from unittest.mock import Mock, MagicMock, patch

# Add the parent directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Setup minimal mocks before importing
from test_code._mock_megatron import MinimalMock
MinimalMock.setup_megatron_mocks()

# Import after mocks are set up
try:
    from flagscale.train.train_gpt import model_provider
    IMPORT_SUCCESS = True
except Exception as e:
    print(f"Warning: Unable to import implementation code: {e}")
    IMPORT_SUCCESS = False


class TestModelProvider(unittest.TestCase):
    """Test model_provider function"""
    
    def setUp(self):
        """Setup test environment"""
        torch.manual_seed(42)
        
        # Mock args
        self.mock_args = Mock()
        self.mock_args.use_legacy_models = False
        self.mock_args.transformer_impl = "local"
        self.mock_args.record_memory_history = False
        self.mock_args.yaml_cfg = None
        self.mock_args.num_experts = None
        self.mock_args.heterogeneous_layers_config_path = None
        self.mock_args.spec = None
        self.mock_args.normalization = "LayerNorm"
        self.mock_args.qk_l2_norm = False
        self.mock_args.moe_grouped_gemm = False
        self.mock_args.qk_layernorm = False
        self.mock_args.multi_latent_attention = False
        self.mock_args.moe_use_legacy_grouped_gemm = False
        self.mock_args.mtp_num_layers = None
        self.mock_args.padded_vocab_size = 1000
        self.mock_args.max_position_embeddings = 512
        self.mock_args.fp16_lm_cross_entropy = False
        self.mock_args.untie_embeddings_and_output_weights = False
        self.mock_args.position_embedding_type = "learned_absolute"
        self.mock_args.rotary_percent = 1.0
        self.mock_args.rotary_base = 10000
        self.mock_args.use_rope_scaling = False
        
        # Mock config
        self.mock_config = Mock()
        self.mock_config.hidden_size = 128
        self.mock_config.num_layers = 2
        self.mock_config.num_attention_heads = 4
        self.mock_config.use_kitchen = False
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_parallel_context')
    @patch('flagscale.train.train_gpt.core_transformer_config_from_args')
    @patch('flagscale.train.train_gpt.core_transformer_config_from_yaml')
    @patch('flagscale.train.train_gpt.get_gpt_layer_local_spec')
    @patch('flagscale.train.train_gpt.GPTModel')
    @patch('flagscale.train.train_gpt.has_nvidia_modelopt', False)
    def test_calls_model_provider_modelopt_when_enabled(self, mock_gpt_model, mock_get_spec, 
                                                         mock_config_yaml, mock_config_args,
                                                         mock_para_ctx, mock_get_args):
        """
        Test case 1: Call model_provider_modelopt when ModelOpt is enabled
        
        Key verification: When has_nvidia_modelopt=True and modelopt_args_enabled=True, should call model_provider_modelopt
        """
        # Setup mocks
        mock_get_args.return_value = self.mock_args
        mock_para_ctx.return_value = None
        
        with patch('flagscale.train.train_gpt.has_nvidia_modelopt', True):
            with patch('flagscale.train.train_gpt.modelopt_args_enabled', return_value=True):
                with patch('flagscale.train.train_gpt.model_provider_modelopt') as mock_modelopt_provider:
                    mock_modelopt_provider.return_value = Mock()
                    
                    # Execute
                    result = model_provider(pre_process=True, post_process=True)
                    
                    # Verify: Called model_provider_modelopt
                    mock_modelopt_provider.assert_called_once_with(True, True)
                    self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_parallel_context')
    @patch('flagscale.train.train_gpt.core_transformer_config_from_args')
    @patch('flagscale.train.train_gpt.get_gpt_layer_local_spec')
    @patch('flagscale.train.train_gpt.GPTModel')
    @patch('flagscale.train.train_gpt.has_nvidia_modelopt', False)
    def test_handles_pre_post_process_parameters(self, mock_gpt_model, mock_get_spec,
                                                  mock_config_args, mock_para_ctx, mock_get_args):
        """
        Test case 2: Correctly handle pre_process and post_process parameters
        
        Key verification: Parameters should be passed to model constructor
        """
        # Setup mocks
        mock_get_args.return_value = self.mock_args
        mock_para_ctx.return_value = None
        mock_config_args.return_value = self.mock_config
        mock_get_spec.return_value = Mock()
        mock_gpt_model.return_value = Mock()
        
        # Execute
        result = model_provider(pre_process=True, post_process=False)
        
        # Verify: GPTModel was called with pre_process and post_process parameters
        mock_gpt_model.assert_called_once()
        call_kwargs = mock_gpt_model.call_args[1]
        self.assertEqual(call_kwargs['pre_process'], True)
        self.assertEqual(call_kwargs['post_process'], False)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_parallel_context')
    @patch('flagscale.train.train_gpt.core_transformer_config_from_args')
    @patch('flagscale.train.train_gpt.megatron')
    @patch('flagscale.train.train_gpt.has_nvidia_modelopt', False)
    def test_creates_legacy_model_when_enabled(self, mock_megatron, mock_config_args,
                                                 mock_para_ctx, mock_get_args):
        """
        Test case 3: Create legacy model when use_legacy_models=True
        
        Key verification: Should call megatron.legacy.model.GPTModel
        """
        # Setup mocks
        self.mock_args.use_legacy_models = True
        mock_get_args.return_value = self.mock_args
        mock_para_ctx.return_value = None
        mock_config_args.return_value = self.mock_config
        
        mock_legacy_model = Mock()
        mock_megatron.legacy.model.GPTModel = Mock(return_value=mock_legacy_model)
        
        # Execute
        result = model_provider(pre_process=True, post_process=True)
        
        # Verify: Called legacy GPTModel
        mock_megatron.legacy.model.GPTModel.assert_called_once()
        call_kwargs = mock_megatron.legacy.model.GPTModel.call_args[1]
        self.assertEqual(call_kwargs['pre_process'], True)
        self.assertEqual(call_kwargs['post_process'], True)
        self.assertEqual(call_kwargs['parallel_output'], True)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_parallel_context')
    @patch('flagscale.train.train_gpt.core_transformer_config_from_args')
    @patch('flagscale.train.train_gpt.get_gpt_layer_local_spec')
    @patch('flagscale.train.train_gpt.GPTModel')
    @patch('flagscale.train.train_gpt.has_nvidia_modelopt', False)
    def test_creates_core_model_when_not_legacy(self, mock_gpt_model, mock_get_spec,
                                                  mock_config_args, mock_para_ctx, mock_get_args):
        """
        Test case 4: Create core model when use_legacy_models=False
        
        Key verification: Should call GPTModel (core model)
        """
        # Setup mocks
        mock_get_args.return_value = self.mock_args
        mock_para_ctx.return_value = None
        mock_config_args.return_value = self.mock_config
        mock_get_spec.return_value = Mock()
        mock_gpt_model.return_value = Mock()
        
        # Execute
        result = model_provider(pre_process=True, post_process=True)
        
        # Verify: Called GPTModel (core model)
        mock_gpt_model.assert_called_once()
        call_kwargs = mock_gpt_model.call_args[1]
        self.assertEqual(call_kwargs['pre_process'], True)
        self.assertEqual(call_kwargs['post_process'], True)
        self.assertEqual(call_kwargs['config'], self.mock_config)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_parallel_context')
    @patch('flagscale.train.train_gpt.core_transformer_config_from_args')
    @patch('flagscale.train.train_gpt.get_gpt_layer_local_spec')
    @patch('flagscale.train.train_gpt.GPTModel')
    @patch('flagscale.train.train_gpt.has_nvidia_modelopt', False)
    def test_calls_core_transformer_config_from_args(self, mock_gpt_model, mock_get_spec,
                                                       mock_config_args, mock_para_ctx, mock_get_args):
        """
        Test case 5: Call core_transformer_config_from_args to create config
        
        Key verification: Should call core_transformer_config_from_args
        """
        # Setup mocks
        mock_get_args.return_value = self.mock_args
        mock_para_ctx.return_value = None
        mock_config_args.return_value = self.mock_config
        mock_get_spec.return_value = Mock()
        mock_gpt_model.return_value = Mock()
        
        # Execute
        model_provider(pre_process=True, post_process=True)
        
        # Verify: Called core_transformer_config_from_args
        mock_config_args.assert_called_once_with(self.mock_args)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('flagscale.train.train_gpt.get_args')
    @patch('flagscale.train.train_gpt.get_parallel_context')
    @patch('flagscale.train.train_gpt.core_transformer_config_from_yaml')
    @patch('flagscale.train.train_gpt.get_gpt_layer_local_spec')
    @patch('flagscale.train.train_gpt.GPTModel')
    @patch('flagscale.train.train_gpt.has_nvidia_modelopt', False)
    def test_uses_yaml_config_when_provided(self, mock_gpt_model, mock_get_spec,
                                             mock_config_yaml, mock_para_ctx, mock_get_args):
        """
        Test case 6: Use yaml config when yaml_cfg is not None
        
        Key verification: Should call core_transformer_config_from_yaml
        """
        # Setup mocks
        self.mock_args.yaml_cfg = "config.yaml"
        mock_get_args.return_value = self.mock_args
        mock_para_ctx.return_value = None
        mock_config_yaml.return_value = self.mock_config
        mock_get_spec.return_value = Mock()
        mock_gpt_model.return_value = Mock()
        
        # Execute
        model_provider(pre_process=True, post_process=True)
        
        # Verify: Called core_transformer_config_from_yaml
        mock_config_yaml.assert_called_once_with(self.mock_args, "language_model")


if __name__ == "__main__":
    unittest.main()
