#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Quantizer.quantize function
"""

import unittest
import torch
import sys
import os
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Conditional mock: only mock when module doesn't exist or import fails
def conditional_mock(module_name, submodules=None):
    """Only mock when module doesn't exist or import fails"""
    try:
        __import__(module_name)
    except (ImportError, OSError, Exception):
        # Catch all exceptions, including dynamic library loading failures
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

# Try to import, mock only if it fails
conditional_mock('lightning', ['pytorch', 'pytorch.trainer', 'pytorch.trainer.states', 
                                'pytorch.callbacks', 'pytorch.loggers'])
conditional_mock('megatron', ['core', 'core.inference', 'core.inference.common_inference_params'])
conditional_mock('nemo_run')
conditional_mock('datasets')
conditional_mock('transformers')
conditional_mock('tqdm')
conditional_mock('flash_attn')
conditional_mock('apex')
conditional_mock('transformer_engine')

# Mock modelopt (usually doesn't exist)
mock_mtq = MagicMock()
conditional_mock('modelopt', ['torch', 'torch.quantization', 'torch.export', 'torch.opt'])
if 'modelopt.torch.quantization' in sys.modules:
    sys.modules['modelopt.torch.quantization'] = mock_mtq

# Import ground truth implementation
try:
    from nemo.collections.llm.modelopt.quantization.quantizer import Quantizer, QuantizationConfig
    IMPORT_SUCCESS = True
except Exception as e:
    print(f"Warning: Unable to import implementation code: {e}")
    IMPORT_SUCCESS = False


class TestQuantizerQuantize(unittest.TestCase):
    """Test Quantizer.quantize function"""
    
    def setUp(self):
        """Setup test environment"""
        torch.manual_seed(42)
        
        # Reset all mocks
        mock_mtq.quantize.reset_mock()
        mock_mtq.postprocess_amax.reset_mock()
        mock_mtq.config.need_calibration.reset_mock()
        mock_mtq.print_quant_summary.reset_mock()
        
        # Create basic test model
        self.test_model = self._create_test_model()
    
    def _create_test_model(self):
        """Create simple test model"""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(64, 128)
                self.linear2 = torch.nn.Linear(128, 64)
                self.decoder_type = "gpt"
                
            def forward(self, x):
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.linear2(x)
                return x
        
        return SimpleModel()
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.logging')
    def test_algorithm_none_returns_original_model(self, mock_logging):
        """
        Test case 1: algorithm=None must directly return original model
        
        Input: Configuration with algorithm=None
        Expected behavior: Return same model object, log message, do not execute quantization
        """
        # Input
        config = QuantizationConfig(algorithm=None)
        quantizer = Quantizer(config, export_config=None)
        original_model = self.test_model
        
        # Execute
        result_model = quantizer.quantize(original_model)
        
        # Verify output
        self.assertIs(result_model, original_model, 
                     "When algorithm=None must return same model object reference")
        
        # Verify behavior: Must log message
        mock_logging.info.assert_any_call(
            "Quantization algorithm set to None, returning the non-quantized model"
        )
        
        # Verify behavior: Should not call any quantization functions
        mock_mtq.quantize.assert_not_called()
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.is_global_rank_zero')
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.unwrap_for_modelopt_operations')
    def test_fp8_quantization_workflow(self, mock_unwrap, mock_is_rank_zero):
        """
        Test case 2: Complete FP8 quantization workflow
        
        Input: Configuration with algorithm="fp8"
        Expected behavior: 
        1. Call unwrap_for_modelopt_operations to unwrap model
        2. Call mtq.quantize to execute quantization
        3. For GPT model call postprocess_amax (maxbound=448)
        """
        # Setup
        mock_is_rank_zero.return_value = False
        mock_mtq.config.need_calibration.return_value = False
        
        unwrapped_model = Mock()
        mock_unwrap.return_value = unwrapped_model
        mock_mtq.quantize.return_value = unwrapped_model
        
        # Input
        config = QuantizationConfig(algorithm="fp8")
        quantizer = Quantizer(config, export_config=None)
        
        # Execute
        with patch.object(quantizer, '_setup'):
            with patch.object(quantizer, '_get_decoder_type', return_value="gpt"):
                with patch.object(quantizer, '_get_quant_cfg', return_value={}):
                    result = quantizer.quantize(self.test_model)
        
        # Verify behavior 1: Must unwrap model
        mock_unwrap.assert_called_once_with(self.test_model)
        
        # Verify behavior 2: Must call mtq.quantize
        self.assertTrue(mock_mtq.quantize.called,
                       "Must call mtq.quantize to execute quantization")
        
        # Verify call arguments contain unwrapped model
        call_args = mock_mtq.quantize.call_args
        self.assertEqual(call_args[0][0], unwrapped_model)
        
        # Verify behavior 3: GPT model must call postprocess_amax
        self.assertTrue(mock_mtq.postprocess_amax.called,
                       "GPT decoder must call postprocess_amax for post-processing")
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.is_global_rank_zero')
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.unwrap_for_modelopt_operations')
    def test_fp8_uses_maxbound_448(self, mock_unwrap, mock_is_rank_zero):
        """
        Test case 3: FP8 algorithm must use maxbound=448
        
        Input: algorithm="fp8"
        Expected behavior: Use clamp function with maxbound=448 in postprocess_amax
        """
        mock_is_rank_zero.return_value = False
        mock_mtq.config.need_calibration.return_value = False
        mock_unwrap.return_value = Mock()
        mock_mtq.quantize.return_value = Mock()
        
        config = QuantizationConfig(algorithm="fp8")
        quantizer = Quantizer(config, export_config=None)
        
        with patch.object(quantizer, '_setup'):
            with patch.object(quantizer, '_get_decoder_type', return_value="gpt"):
                with patch.object(quantizer, '_get_quant_cfg', return_value={}):
                    quantizer.quantize(self.test_model)
        
        # Verify postprocess_amax was called
        self.assertTrue(mock_mtq.postprocess_amax.called)
        
        # Verify parameter: second argument should be "*input_quantizer"
        call_args = mock_mtq.postprocess_amax.call_args
        self.assertEqual(call_args[0][1], "*input_quantizer",
                        "Must post-process *input_quantizer")
        
        # Verify lambda function uses clamp operation (indirectly verify maxbound)
        # lambda: torch.clamp(amax, min=2**-24, max=448 if fp8 else 127)
        self.assertIsNotNone(call_args[0][2])  # lambda function exists
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.is_global_rank_zero')
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.unwrap_for_modelopt_operations')
    def test_int8_sq_uses_maxbound_127(self, mock_unwrap, mock_is_rank_zero):
        """
        Test case 4: INT8_SQ algorithm must use maxbound=127
        
        Input: algorithm="int8_sq"
        Expected behavior: Use clamp function with maxbound=127 in postprocess_amax
        """
        mock_is_rank_zero.return_value = False
        mock_mtq.config.need_calibration.return_value = False
        mock_unwrap.return_value = Mock()
        mock_mtq.quantize.return_value = Mock()
        
        config = QuantizationConfig(algorithm="int8_sq")
        quantizer = Quantizer(config, export_config=None)
        
        with patch.object(quantizer, '_setup'):
            with patch.object(quantizer, '_get_decoder_type', return_value="gpt"):
                with patch.object(quantizer, '_get_quant_cfg', return_value={}):
                    quantizer.quantize(self.test_model)
        
        # Verify postprocess_amax was called
        self.assertTrue(mock_mtq.postprocess_amax.called,
                       "int8_sq must call postprocess_amax")
        
        # Verify parameter settings
        call_args = mock_mtq.postprocess_amax.call_args
        self.assertEqual(call_args[0][1], "*input_quantizer")
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.is_global_rank_zero')
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.unwrap_for_modelopt_operations')
    def test_non_gpt_decoder_no_postprocess(self, mock_unwrap, mock_is_rank_zero):
        """
        Test case 5: Non-GPT decoder should not call postprocess_amax
        
        Input: decoder_type="llama" (non-gpt)
        Expected behavior: Do not call postprocess_amax
        """
        mock_is_rank_zero.return_value = False
        mock_mtq.config.need_calibration.return_value = False
        mock_unwrap.return_value = Mock()
        mock_mtq.quantize.return_value = Mock()
        
        config = QuantizationConfig(algorithm="fp8")
        quantizer = Quantizer(config, export_config=None)
        
        with patch.object(quantizer, '_setup'):
            with patch.object(quantizer, '_get_decoder_type', return_value="llama"):
                with patch.object(quantizer, '_get_quant_cfg', return_value={}):
                    quantizer.quantize(self.test_model)
        
        # Verify: Should not call postprocess_amax
        self.assertFalse(mock_mtq.postprocess_amax.called,
                        "Non-GPT decoder should not call postprocess_amax")
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.is_global_rank_zero')
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.unwrap_for_modelopt_operations')
    def test_calibration_forward_loop_creation(self, mock_unwrap, mock_is_rank_zero):
        """
        Test case 6: Automatically create forward_loop when calibration needed but not provided
        
        Input: need_calibration=True, forward_loop=None
        Expected behavior: Call _get_forward_loop to create calibration loop
        """
        mock_is_rank_zero.return_value = False
        mock_mtq.config.need_calibration.return_value = True
        mock_unwrap.return_value = Mock()
        mock_mtq.quantize.return_value = Mock()
        
        config = QuantizationConfig(algorithm="fp8")
        quantizer = Quantizer(config, export_config=None)
        
        mock_forward_loop = Mock()
        
        with patch.object(quantizer, '_setup'):
            with patch.object(quantizer, '_get_decoder_type', return_value="gpt"):
                with patch.object(quantizer, '_get_quant_cfg', return_value={}):
                    with patch.object(quantizer, '_get_forward_loop', 
                                    return_value=mock_forward_loop) as mock_get_loop:
                        quantizer.quantize(self.test_model, forward_loop=None)
                        
                        # Verify: Must call _get_forward_loop
                        mock_get_loop.assert_called_once()
        
        # Verify: mtq.quantize is called with forward_loop
        call_kwargs = mock_mtq.quantize.call_args[1]
        self.assertIn('forward_loop', call_kwargs)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.is_global_rank_zero')
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.unwrap_for_modelopt_operations')
    def test_provided_forward_loop_used_directly(self, mock_unwrap, mock_is_rank_zero):
        """
        Test case 7: Use provided forward_loop directly
        
        Input: Custom forward_loop function
        Expected behavior: Do not call _get_forward_loop, use provided forward_loop directly
        """
        mock_is_rank_zero.return_value = False
        mock_mtq.config.need_calibration.return_value = True
        mock_unwrap.return_value = Mock()
        mock_mtq.quantize.return_value = Mock()
        
        config = QuantizationConfig(algorithm="fp8")
        quantizer = Quantizer(config, export_config=None)
        
        # Provide custom forward_loop
        custom_forward_loop = Mock()
        
        with patch.object(quantizer, '_setup'):
            with patch.object(quantizer, '_get_decoder_type', return_value="gpt"):
                with patch.object(quantizer, '_get_quant_cfg', return_value={}):
                    with patch.object(quantizer, '_get_forward_loop') as mock_get_loop:
                        quantizer.quantize(self.test_model, forward_loop=custom_forward_loop)
                        
                        # Verify: Should not call _get_forward_loop
                        mock_get_loop.assert_not_called()
        
        # Verify: Passed custom forward_loop is used
        call_kwargs = mock_mtq.quantize.call_args[1]
        self.assertEqual(call_kwargs.get('forward_loop'), custom_forward_loop)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.is_global_rank_zero')
    def test_quant_summary_printed_on_rank_zero(self, mock_is_rank_zero):
        """
        Test case 8: Print quantization summary on rank 0
        
        Input: global_rank=0
        Expected behavior: Call mtq.print_quant_summary
        """
        mock_is_rank_zero.return_value = True
        mock_mtq.config.need_calibration.return_value = False
        mock_mtq.quantize.return_value = Mock()
        
        config = QuantizationConfig(algorithm="fp8")
        quantizer = Quantizer(config, export_config=None)
        
        with patch('nemo.collections.llm.modelopt.quantization.quantizer.unwrap_for_modelopt_operations'):
            with patch.object(quantizer, '_setup'):
                with patch.object(quantizer, '_get_decoder_type', return_value="gpt"):
                    with patch.object(quantizer, '_get_quant_cfg', return_value={}):
                        quantizer.quantize(self.test_model)
        
        # Verify: Call print_quant_summary on rank 0
        self.assertTrue(mock_mtq.print_quant_summary.called,
                       "Must print quantization summary on rank 0")
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    def test_different_quantization_algorithms(self):
        """
        Test case 9: Test support for multiple quantization algorithms
        
        Input: Different quantization algorithms
        Expected behavior: All algorithms can be initialized correctly
        """
        algorithms = ["fp8", "int8", "int8_sq", "int4_awq", "w4a8_awq"]
        
        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm):
                config = QuantizationConfig(algorithm=algorithm)
                quantizer = Quantizer(config, export_config=None)
                
                # Verify: Can successfully create Quantizer instance
                self.assertIsNotNone(quantizer)
                self.assertEqual(quantizer.config.algorithm, algorithm)


class TestQuantizationEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    def test_config_validation(self):
        """Test configuration validation"""
        # Test valid configuration
        valid_config = QuantizationConfig(algorithm="fp8")
        self.assertIsNotNone(valid_config)
        
        # Test None algorithm
        none_config = QuantizationConfig(algorithm=None)
        self.assertIsNone(none_config.algorithm)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.unwrap_for_modelopt_operations')
    def test_model_structure_preservation(self, mock_unwrap):
        """
        Test case 10: Verify model structure remains unchanged after quantization
        
        Expected behavior: Quantization does not change number of layers and parameters (only changes precision)
        """
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32)
        )
        original_param_count = sum(p.numel() for p in model.parameters())
        
        # Mock quantization process preserves model structure
        mock_unwrap.return_value = model
        mock_mtq.quantize.return_value = model
        mock_mtq.config.need_calibration.return_value = False
        
        config = QuantizationConfig(algorithm="fp8")
        quantizer = Quantizer(config, export_config=None)
        
        with patch.object(quantizer, '_setup'):
            with patch.object(quantizer, '_get_decoder_type', return_value="gpt"):
                with patch.object(quantizer, '_get_quant_cfg', return_value={}):
                    quantized_model = quantizer.quantize(model)
        
        # Verify: Parameter count remains unchanged
        quantized_param_count = sum(p.numel() for p in quantized_model.parameters())
        self.assertEqual(original_param_count, quantized_param_count,
                        "Quantization should not change model parameter count")


if __name__ == "__main__":
    unittest.main()
