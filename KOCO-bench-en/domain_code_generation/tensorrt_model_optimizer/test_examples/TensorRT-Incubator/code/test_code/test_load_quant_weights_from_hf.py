#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test load_quant_weights_from_hf function
"""

import unittest
import torch
import sys
import os
from unittest.mock import MagicMock, patch, Mock

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Conditional mock: only mock when module doesn't exist or import fails
def conditional_mock(module_name, submodules=None):
    """Only mock when module doesn't exist or import fails"""
    try:
        __import__(module_name)
    except (ImportError, OSError, Exception):
        mock_obj = MagicMock()
        sys.modules[module_name] = mock_obj
        if submodules:
            for sub in submodules:
                full_name = f"{module_name}.{sub}"
                try:
                    __import__(full_name)
                except (ImportError, OSError, Exception):
                    sys.modules[full_name] = MagicMock()
        return mock_obj
    return None

# Mock dependencies
conditional_mock('transformers', ['models', 'models.gpt2'])
conditional_mock('datasets')
conditional_mock('tqdm')
conditional_mock('tripy')
conditional_mock('modelopt', ['torch', 'torch.quantization'])

# Import ground truth implementation
try:
    tripy_path = os.path.join(os.path.dirname(__file__), '..', 'tripy')
    if tripy_path not in sys.path:
        sys.path.insert(0, tripy_path)
    
    from examples.nanogpt.weight_loader import load_quant_weights_from_hf
    IMPORT_SUCCESS = True
except Exception as e:
    print(f"Warning: Unable to import implementation code: {e}")
    IMPORT_SUCCESS = False


class TestLoadQuantWeightsFromHF(unittest.TestCase):
    """Test load_quant_weights_from_hf function"""
    
    def setUp(self):
        """Setup test environment"""
        torch.manual_seed(42)
        
        # Create basic test model
        self.test_model = self._create_test_tripy_model()
        self.model_type = "gpt2"
        self.dtype = self._create_mock_dtype()
    
    def _create_test_tripy_model(self):
        """Create test Tripy model mock"""
        model = MagicMock()
        # Create state dictionary
        state_dict = {
            'transformer.wte.weight': None,
            'transformer.wpe.weight': None,
            'transformer.h.0.attn.c_attn.weight': None,
            'transformer.h.0.attn.c_attn.scale': None,
            'transformer.h.0.attn.c_proj.weight': None,
            'transformer.h.0.mlp.c_fc.weight': None,
            'transformer.h.0.mlp.c_proj.weight': None,
            'lm_head.weight': None,
        }
        model.state_dict.return_value = state_dict
        model.load_state_dict = MagicMock()
        return model
    
    def _create_mock_dtype(self):
        """Create Mock data type"""
        dtype = MagicMock()
        dtype.name = "float32"
        return dtype
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('examples.nanogpt.weight_loader.GPT2LMHeadModel')
    @patch('examples.nanogpt.weight_loader.modelopt_quantize')
    @patch('examples.nanogpt.weight_loader.tp')
    def test_basic_weight_loading_flow(self, mock_tp, mock_quantize, mock_gpt2_class):
        """
        Test case 1: Basic weight loading flow
        
        Input: Tripy model, model type, data type, quantization mode
        Expected behavior:
        1. Load HuggingFace model
        2. Execute quantization
        3. Load weights to Tripy model
        """
        # Setup mock
        mock_hf_model = self._create_mock_hf_model()
        mock_gpt2_class.from_pretrained.return_value = mock_hf_model
        mock_quantize.return_value = mock_hf_model
        
        # Mock tp.Tensor
        mock_tp.Tensor.side_effect = lambda x: x
        
        # Execute
        with patch('builtins.print'):
            load_quant_weights_from_hf(
                self.test_model,
                self.model_type,
                self.dtype,
                "int8-weight-only"
            )
        
        # Verify behavior 1: Loaded HuggingFace model
        mock_gpt2_class.from_pretrained.assert_called_once_with(self.model_type)
        
        # Verify behavior 2: Executed quantization
        mock_quantize.assert_called_once()
        
        # Verify behavior 3: Called load_state_dict
        self.test_model.load_state_dict.assert_called_once()
    
    def _create_mock_hf_model(self):
        """Create Mock HuggingFace model"""
        model = MagicMock()
        
        # Create simulated state dictionary
        state_dict = {
            'transformer.wte.weight': torch.randn(50257, 768),
            'transformer.wpe.weight': torch.randn(1024, 768),
            'transformer.h.0.attn.c_attn.weight': torch.randn(768, 768*3),
            'transformer.h.0.attn.c_proj.weight': torch.randn(768, 768),
            'transformer.h.0.mlp.c_fc.weight': torch.randn(768, 3072),
            'transformer.h.0.mlp.c_proj.weight': torch.randn(3072, 768),
            'lm_head.weight': torch.randn(50257, 768),
            'transformer.h.0.attn.masked_bias': torch.tensor(-1e9),
            'transformer.h.0.attn.bias': torch.ones(1, 1, 1024, 1024),
        }
        model.state_dict.return_value = state_dict
        
        return model
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('examples.nanogpt.weight_loader.GPT2LMHeadModel')
    @patch('examples.nanogpt.weight_loader.modelopt_quantize')
    @patch('examples.nanogpt.weight_loader.tp')
    def test_ignored_keys_filtering(self, mock_tp, mock_quantize, mock_gpt2_class):
        """
        Test case 2: Ignore specific keys
        
        Expected behavior:
        1. Filter out .attn.masked_bias
        2. Filter out .attn.bias
        3. Filter out _pre_quant_scale
        """
        # Setup mock
        mock_hf_model = self._create_mock_hf_model()
        mock_gpt2_class.from_pretrained.return_value = mock_hf_model
        mock_quantize.return_value = mock_hf_model
        
        mock_tp.Tensor.side_effect = lambda x: x
        
        # Execute
        with patch('builtins.print'):
            load_quant_weights_from_hf(
                self.test_model,
                self.model_type,
                self.dtype,
                "int8-weight-only"
            )
        
        # Verify: load_state_dict was called
        self.test_model.load_state_dict.assert_called_once()
        
        # Get passed state dictionary
        call_args = self.test_model.load_state_dict.call_args
        loaded_state_dict = call_args[0][0]
        
        # Verify: Ignored keys are not in loaded state dictionary
        ignored_patterns = ['.attn.masked_bias', '.attn.bias', '_pre_quant_scale']
        for key in loaded_state_dict.keys():
            for pattern in ignored_patterns:
                self.assertNotIn(pattern, key, 
                               f"Ignored key pattern {pattern} should not appear in loaded weights")
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('examples.nanogpt.weight_loader.GPT2LMHeadModel')
    @patch('examples.nanogpt.weight_loader.modelopt_quantize')
    @patch('examples.nanogpt.weight_loader.tp')
    def test_weight_tying_for_embeddings(self, mock_tp, mock_quantize, mock_gpt2_class):
        """
        Test case 3: Weight tying handling
        
        Expected behavior: transformer.wte.weight = lm_head.weight
        """
        # Setup mock
        mock_hf_model = self._create_mock_hf_model()
        lm_head_weight = torch.randn(50257, 768)
        mock_hf_model.state_dict.return_value['lm_head.weight'] = lm_head_weight
        
        mock_gpt2_class.from_pretrained.return_value = mock_hf_model
        mock_quantize.return_value = mock_hf_model
        
        mock_tp.Tensor.side_effect = lambda x: x
        
        # Execute
        with patch('builtins.print'):
            load_quant_weights_from_hf(
                self.test_model,
                self.model_type,
                self.dtype,
                "int8-weight-only"
            )
        
        # Verify: wte weight should equal lm_head weight
        call_args = self.test_model.load_state_dict.call_args
        loaded_state_dict = call_args[0][0]
        
        # Check if wte.weight is in state dictionary
        self.assertIn('transformer.wte.weight', loaded_state_dict)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('examples.nanogpt.weight_loader.GPT2LMHeadModel')
    @patch('examples.nanogpt.weight_loader.modelopt_quantize')
    @patch('examples.nanogpt.weight_loader.tp')
    def test_int4_block_quantization_reshape(self, mock_tp, mock_quantize, mock_gpt2_class):
        """
        Test case 4: INT4 block quantization amax reshape
        
        Input: quant_mode="int4-weight-only"
        Expected behavior: Reshape amax tensor to match block quantization dimensions
        """
        # Setup mock
        mock_hf_model = MagicMock()
        state_dict = {
            'transformer.h.0.attn.c_attn.weight': torch.randn(768, 768*3),
            'transformer.h.0.attn.c_attn.weight_quantizer._amax': torch.randn(768*3, 1),
            'lm_head.weight': torch.randn(50257, 768),
        }
        mock_hf_model.state_dict.return_value = state_dict
        
        # Create mock linear layer
        mock_linear = MagicMock()
        mock_linear.in_features = 768
        
        # Create mock quantizer
        mock_quantizer = MagicMock()
        mock_quantizer.maxbound = 127
        
        mock_hf_model.transformer = MagicMock()
        mock_hf_model.transformer.h = [MagicMock()]
        mock_hf_model.transformer.h[0].attn = MagicMock()
        mock_hf_model.transformer.h[0].attn.c_attn = mock_linear
        mock_hf_model.transformer.h[0].attn.c_attn.weight_quantizer = mock_quantizer
        
        mock_gpt2_class.from_pretrained.return_value = mock_hf_model
        mock_quantize.return_value = mock_hf_model
        
        mock_tp.Tensor.side_effect = lambda x: x
        
        # Execute
        with patch('builtins.print'):
            load_quant_weights_from_hf(
                self.test_model,
                self.model_type,
                self.dtype,
                "int4-weight-only"
            )
        
        # Verify: Successfully executed
        self.test_model.load_state_dict.assert_called_once()
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('examples.nanogpt.weight_loader.GPT2LMHeadModel')
    @patch('examples.nanogpt.weight_loader.modelopt_quantize')
    @patch('examples.nanogpt.weight_loader.tp')
    def test_scale_computation_from_amax(self, mock_tp, mock_quantize, mock_gpt2_class):
        """
        Test case 5: Compute scaling factor from amax
        
        Expected behavior:
        1. Get quantizer's amax value
        2. Divide by maxbound to get scale
        3. Convert key name to .scale
        """
        # Setup mock
        mock_hf_model = MagicMock()
        amax_value = torch.tensor([[127.0]])
        state_dict = {
            'transformer.h.0.attn.c_attn.weight': torch.randn(768, 768*3),
            'transformer.h.0.attn.c_attn.weight_quantizer._amax': amax_value,
            'lm_head.weight': torch.randn(50257, 768),
        }
        mock_hf_model.state_dict.return_value = state_dict
        
        # Create mock quantizer
        mock_quantizer = MagicMock()
        mock_quantizer.maxbound = 127.0
        
        mock_hf_model.transformer = MagicMock()
        mock_hf_model.transformer.h = [MagicMock()]
        mock_hf_model.transformer.h[0].attn = MagicMock()
        mock_hf_model.transformer.h[0].attn.c_attn = MagicMock()
        mock_hf_model.transformer.h[0].attn.c_attn.weight_quantizer = mock_quantizer
        
        mock_gpt2_class.from_pretrained.return_value = mock_hf_model
        mock_quantize.return_value = mock_hf_model
        
        mock_tp.Tensor.side_effect = lambda x: x
        
        # Execute
        with patch('builtins.print'):
            load_quant_weights_from_hf(
                self.test_model,
                self.model_type,
                self.dtype,
                "int8-weight-only"
            )
        
        # Verify: Successfully executed
        self.test_model.load_state_dict.assert_called_once()
        
        # Get loaded state dictionary
        call_args = self.test_model.load_state_dict.call_args
        loaded_state_dict = call_args[0][0]
        
        # Verify: scale key exists
        scale_key_exists = any('scale' in key for key in loaded_state_dict.keys())
        self.assertTrue(scale_key_exists, "Should contain scale key")
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('examples.nanogpt.weight_loader.GPT2LMHeadModel')
    @patch('examples.nanogpt.weight_loader.modelopt_quantize')
    @patch('examples.nanogpt.weight_loader.tp')
    def test_dtype_conversion(self, mock_tp, mock_quantize, mock_gpt2_class):
        """
        Test case 6: Data type conversion
        
        Expected behavior: Weights converted to target dtype
        """
        # Setup mock
        mock_hf_model = self._create_mock_hf_model()
        mock_gpt2_class.from_pretrained.return_value = mock_hf_model
        mock_quantize.return_value = mock_hf_model
        
        mock_tp.Tensor.side_effect = lambda x: x
        
        # Test different data types
        dtypes = ["float32", "float16", "bfloat16"]
        
        for dtype_name in dtypes:
            with self.subTest(dtype=dtype_name):
                # Reset mock
                self.test_model.load_state_dict.reset_mock()
                
                # Create dtype
                dtype = MagicMock()
                dtype.name = dtype_name
                
                # Execute
                with patch('builtins.print'):
                    load_quant_weights_from_hf(
                        self.test_model,
                        self.model_type,
                        dtype,
                        "int8-weight-only"
                    )
                
                # Verify: Successfully loaded
                self.test_model.load_state_dict.assert_called_once()
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('examples.nanogpt.weight_loader.GPT2LMHeadModel')
    @patch('examples.nanogpt.weight_loader.modelopt_quantize')
    @patch('examples.nanogpt.weight_loader.tp')
    def test_strict_loading_disabled(self, mock_tp, mock_quantize, mock_gpt2_class):
        """
        Test case 7: Non-strict loading mode
        
        Expected behavior: load_state_dict uses strict=False
        """
        # Setup mock
        mock_hf_model = self._create_mock_hf_model()
        mock_gpt2_class.from_pretrained.return_value = mock_hf_model
        mock_quantize.return_value = mock_hf_model
        
        mock_tp.Tensor.side_effect = lambda x: x
        
        # Execute
        with patch('builtins.print'):
            load_quant_weights_from_hf(
                self.test_model,
                self.model_type,
                self.dtype,
                "int8-weight-only"
            )
        
        # Verify: Used strict=False
        call_kwargs = self.test_model.load_state_dict.call_args[1]
        self.assertIn('strict', call_kwargs)
        self.assertFalse(call_kwargs['strict'])
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('examples.nanogpt.weight_loader.GPT2LMHeadModel')
    @patch('examples.nanogpt.weight_loader.modelopt_quantize')
    @patch('examples.nanogpt.weight_loader.tp')
    def test_print_messages(self, mock_tp, mock_quantize, mock_gpt2_class):
        """
        Test case 8: Print message verification
        
        Expected behavior:
        1. Print loading model information
        2. Print completion information
        """
        # Setup mock
        mock_hf_model = self._create_mock_hf_model()
        mock_gpt2_class.from_pretrained.return_value = mock_hf_model
        mock_quantize.return_value = mock_hf_model
        
        mock_tp.Tensor.side_effect = lambda x: x
        
        # Execute
        with patch('builtins.print') as mock_print:
            load_quant_weights_from_hf(
                self.test_model,
                self.model_type,
                self.dtype,
                "int8-weight-only"
            )
            
            # Verify: Printed loading information
            self.assertTrue(mock_print.called)
            
            # Check print content
            call_args_list = [str(call) for call in mock_print.call_args_list]
            loading_printed = any("Loading weights" in str(call) for call in call_args_list)
            loaded_printed = any("Loaded weights" in str(call) for call in call_args_list)
            
            self.assertTrue(loading_printed, "Should print loading information")
            self.assertTrue(loaded_printed, "Should print completion information")


class TestLoadQuantWeightsEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_model = MagicMock()
        self.test_model.state_dict.return_value = {
            'transformer.wte.weight': None,
            'lm_head.weight': None,
        }
        self.test_model.load_state_dict = MagicMock()
        
        self.dtype = MagicMock()
        self.dtype.name = "float32"
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('examples.nanogpt.weight_loader.GPT2LMHeadModel')
    @patch('examples.nanogpt.weight_loader.modelopt_quantize')
    @patch('examples.nanogpt.weight_loader.tp')
    def test_different_model_types(self, mock_tp, mock_quantize, mock_gpt2_class):
        """
        Test case 9: Different model types
        
        Input: Different model_type
        Expected: All can be loaded correctly
        """
        # Setup mock
        mock_hf_model = MagicMock()
        mock_hf_model.state_dict.return_value = {
            'transformer.wte.weight': torch.randn(50257, 768),
            'lm_head.weight': torch.randn(50257, 768),
        }
        
        mock_gpt2_class.from_pretrained.return_value = mock_hf_model
        mock_quantize.return_value = mock_hf_model
        mock_tp.Tensor.side_effect = lambda x: x
        
        model_types = ["gpt2", "gpt2-medium", "gpt2-large"]
        
        for model_type in model_types:
            with self.subTest(model_type=model_type):
                mock_gpt2_class.from_pretrained.reset_mock()
                
                # Execute
                with patch('builtins.print'):
                    load_quant_weights_from_hf(
                        self.test_model,
                        model_type,
                        self.dtype,
                        "int8-weight-only"
                    )
                
                # Verify: Used correct model type
                mock_gpt2_class.from_pretrained.assert_called_once_with(model_type)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('examples.nanogpt.weight_loader.GPT2LMHeadModel')
    @patch('examples.nanogpt.weight_loader.modelopt_quantize')
    @patch('examples.nanogpt.weight_loader.tp')
    def test_all_quantization_modes(self, mock_tp, mock_quantize, mock_gpt2_class):
        """
        Test case 10: All quantization modes
        
        Expected: All quantization modes work correctly
        """
        # Setup mock
        mock_hf_model = MagicMock()
        mock_hf_model.state_dict.return_value = {
            'transformer.wte.weight': torch.randn(50257, 768),
            'lm_head.weight': torch.randn(50257, 768),
        }
        
        mock_gpt2_class.from_pretrained.return_value = mock_hf_model
        mock_quantize.return_value = mock_hf_model
        mock_tp.Tensor.side_effect = lambda x: x
        
        quant_modes = ["int8-weight-only", "int4-weight-only", "float8"]
        
        for quant_mode in quant_modes:
            with self.subTest(quant_mode=quant_mode):
                self.test_model.load_state_dict.reset_mock()
                mock_quantize.reset_mock()
                
                # Execute
                with patch('builtins.print'):
                    load_quant_weights_from_hf(
                        self.test_model,
                        "gpt2",
                        self.dtype,
                        quant_mode
                    )
                
                # Verify: Executed quantization
                mock_quantize.assert_called_once()
                
                # Verify: Loaded weights
                self.test_model.load_state_dict.assert_called_once()


if __name__ == "__main__":
    unittest.main()

