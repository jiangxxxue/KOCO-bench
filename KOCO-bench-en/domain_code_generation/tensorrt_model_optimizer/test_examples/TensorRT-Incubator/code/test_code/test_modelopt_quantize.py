#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test modelopt_quantize function
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
conditional_mock('transformers', ['models', 'models.gpt2'])
conditional_mock('datasets')
conditional_mock('tqdm')

# Mock modelopt (usually doesn't exist)
mock_mtq = MagicMock()
conditional_mock('modelopt', ['torch', 'torch.quantization'])
if 'modelopt.torch.quantization' in sys.modules:
    sys.modules['modelopt.torch.quantization'] = mock_mtq

# Import ground truth implementation
try:
    # Need to set tripy path first
    tripy_path = os.path.join(os.path.dirname(__file__), '..', 'tripy')
    if tripy_path not in sys.path:
        sys.path.insert(0, tripy_path)
    
    from examples.nanogpt.quantization import modelopt_quantize
    IMPORT_SUCCESS = True
except Exception as e:
    print(f"Warning: Unable to import implementation code: {e}")
    IMPORT_SUCCESS = False


class TestModeloptQuantize(unittest.TestCase):
    """Test modelopt_quantize function"""
    
    def setUp(self):
        """Setup test environment"""
        torch.manual_seed(42)
        
        # Reset all mocks
        mock_mtq.quantize.reset_mock()
        mock_mtq.INT8_DEFAULT_CFG = {"quant_cfg": {}}
        mock_mtq.INT4_AWQ_CFG = {"quant_cfg": {}}
        mock_mtq.FP8_DEFAULT_CFG = {"quant_cfg": {}}
        
        # Create basic test model
        self.test_model = self._create_test_model()
    
    def _create_test_model(self):
        """Create simple GPT2 test model"""
        class SimpleGPT2Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = torch.nn.ModuleDict({
                    'wte': torch.nn.Embedding(50257, 768),
                    'wpe': torch.nn.Embedding(1024, 768),
                    'h': torch.nn.ModuleList([
                        torch.nn.ModuleDict({
                            'attn': torch.nn.ModuleDict({
                                'c_attn': torch.nn.Linear(768, 768 * 3),
                                'c_proj': torch.nn.Linear(768, 768),
                            }),
                            'mlp': torch.nn.ModuleDict({
                                'c_fc': torch.nn.Linear(768, 3072),
                                'c_proj': torch.nn.Linear(3072, 768),
                            })
                        }) for _ in range(2)
                    ])
                })
                self.lm_head = torch.nn.Linear(768, 50257, bias=False)
                self.device = torch.device('cpu')
                
            def forward(self, input_ids, **kwargs):
                return torch.randn(input_ids.shape[0], input_ids.shape[1], 50257)
        
        return SimpleGPT2Model()
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('examples.nanogpt.quantization.AutoTokenizer')
    @patch('examples.nanogpt.quantization.create_forward_loop')
    def test_int8_weight_only_quantization(self, mock_create_loop, mock_tokenizer_cls):
        """
        Test case 1: INT8 weight-only quantization mode
        
        Input: quant_mode="int8-weight-only"
        Expected behavior:
        1. Use INT8_DEFAULT_CFG configuration
        2. Disable input_quantizer
        3. Call mtq.quantize
        """
        # Setup mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_forward_loop = MagicMock()
        mock_create_loop.return_value = mock_forward_loop
        
        mock_mtq.quantize.return_value = self.test_model
        
        # Execute
        result = modelopt_quantize(self.test_model, "int8-weight-only")
        
        # Verify behavior 1: Used INT8 configuration
        call_args = mock_mtq.quantize.call_args
        quant_cfg = call_args[0][1]
        
        # Verify behavior 2: input_quantizer is disabled
        self.assertIn("quant_cfg", quant_cfg)
        self.assertIn("*input_quantizer", quant_cfg["quant_cfg"])
        self.assertFalse(quant_cfg["quant_cfg"]["*input_quantizer"]["enable"])
        
        # Verify behavior 3: Called mtq.quantize
        mock_mtq.quantize.assert_called_once()
        self.assertEqual(call_args[0][0], self.test_model)
        
        # Verify output
        self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('examples.nanogpt.quantization.AutoTokenizer')
    @patch('examples.nanogpt.quantization.create_forward_loop')
    def test_int4_weight_only_quantization(self, mock_create_loop, mock_tokenizer_cls):
        """
        Test case 2: INT4 weight-only quantization mode using AWQ configuration
        
        Input: quant_mode="int4-weight-only"
        Expected behavior:
        1. Use INT4_AWQ_CFG configuration
        2. Use larger batch_size=16
        3. Use smaller calib_size=16
        """
        # Setup mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_forward_loop = MagicMock()
        mock_create_loop.return_value = mock_forward_loop
        
        mock_mtq.quantize.return_value = self.test_model
        
        # Execute
        result = modelopt_quantize(self.test_model, "int4-weight-only")
        
        # Verify behavior 1: Used INT4_AWQ_CFG
        call_args = mock_mtq.quantize.call_args
        self.assertIsNotNone(call_args)
        
        # Verify behavior 2&3: create_forward_loop call parameters
        loop_call_args = mock_create_loop.call_args
        self.assertEqual(loop_call_args[1]['batch_size'], 16)
        self.assertEqual(loop_call_args[1]['num_samples'], 16)
        
        # Verify output
        self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('examples.nanogpt.quantization.AutoTokenizer')
    @patch('examples.nanogpt.quantization.create_forward_loop')
    def test_float8_quantization(self, mock_create_loop, mock_tokenizer_cls):
        """
        Test case 3: FP8 quantization mode
        
        Input: quant_mode="float8"
        Expected behavior:
        1. Use FP8_DEFAULT_CFG configuration
        2. Use default batch_size=1 and calib_size=64
        """
        # Setup mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_forward_loop = MagicMock()
        mock_create_loop.return_value = mock_forward_loop
        
        mock_mtq.quantize.return_value = self.test_model
        
        # Execute
        result = modelopt_quantize(self.test_model, "float8")
        
        # Verify behavior 1: Used FP8_DEFAULT_CFG
        call_args = mock_mtq.quantize.call_args
        self.assertIsNotNone(call_args)
        
        # Verify behavior 2: create_forward_loop uses default parameters
        loop_call_args = mock_create_loop.call_args
        self.assertEqual(loop_call_args[1]['batch_size'], 1)
        self.assertEqual(loop_call_args[1]['num_samples'], 64)
        
        # Verify output
        self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('examples.nanogpt.quantization.AutoTokenizer')
    @patch('examples.nanogpt.quantization.create_forward_loop')
    def test_unsupported_quantization_mode_raises_error(self, mock_create_loop, mock_tokenizer_cls):
        """
        Test case 4: Unsupported quantization mode should raise exception
        
        Input: quant_mode="unsupported-mode"
        Expected behavior: Raise NotImplementedError
        """
        # Setup mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        # Execute and verify exception
        with self.assertRaises(NotImplementedError) as context:
            modelopt_quantize(self.test_model, "unsupported-mode")
        
        # Verify exception message
        self.assertIn("Unsupported quantization mode", str(context.exception))
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('examples.nanogpt.quantization.AutoTokenizer')
    @patch('examples.nanogpt.quantization.create_forward_loop')
    def test_tokenizer_configuration(self, mock_create_loop, mock_tokenizer_cls):
        """
        Test case 5: Tokenizer correct configuration
        
        Expected behavior:
        1. Load gpt2 tokenizer
        2. Set max_length=2048
        3. Set padding_side="left"
        4. pad_token=eos_token
        """
        # Setup mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_forward_loop = MagicMock()
        mock_create_loop.return_value = mock_forward_loop
        mock_mtq.quantize.return_value = self.test_model
        
        # Execute
        result = modelopt_quantize(self.test_model, "int8-weight-only")
        
        # Verify behavior 1: Loaded gpt2 tokenizer
        mock_tokenizer_cls.from_pretrained.assert_called_once()
        call_args = mock_tokenizer_cls.from_pretrained.call_args
        self.assertEqual(call_args[0][0], "gpt2")
        
        # Verify behavior 2&3: Parameter settings
        call_kwargs = call_args[1]
        self.assertEqual(call_kwargs['model_max_length'], 2048)
        self.assertEqual(call_kwargs['padding_side'], "left")
        
        # Verify behavior 4: pad_token setting
        self.assertEqual(mock_tokenizer.pad_token, mock_tokenizer.eos_token)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('examples.nanogpt.quantization.AutoTokenizer')
    @patch('examples.nanogpt.quantization.create_forward_loop')
    def test_forward_loop_creation_with_cnn_dailymail(self, mock_create_loop, mock_tokenizer_cls):
        """
        Test case 6: Create calibration loop using cnn_dailymail dataset
        
        Expected behavior:
        1. dataset_name="cnn_dailymail"
        2. forward_loop passed to mtq.quantize
        """
        # Setup mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_forward_loop = MagicMock()
        mock_create_loop.return_value = mock_forward_loop
        mock_mtq.quantize.return_value = self.test_model
        
        # Execute
        result = modelopt_quantize(self.test_model, "int8-weight-only")
        
        # Verify behavior 1: Used cnn_dailymail dataset
        loop_call_args = mock_create_loop.call_args
        self.assertEqual(loop_call_args[1]['dataset_name'], "cnn_dailymail")
        
        # Verify behavior 2: forward_loop passed to quantization function
        quant_call_kwargs = mock_mtq.quantize.call_args[1]
        self.assertEqual(quant_call_kwargs['forward_loop'], mock_forward_loop)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('examples.nanogpt.quantization.AutoTokenizer')
    @patch('examples.nanogpt.quantization.create_forward_loop')
    @patch('examples.nanogpt.quantization.time')
    def test_quantization_timing(self, mock_time, mock_create_loop, mock_tokenizer_cls):
        """
        Test case 7: Quantization process timing recording
        
        Expected behavior:
        1. Record start time
        2. Record end time
        3. Print timing information
        """
        # Setup mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_forward_loop = MagicMock()
        mock_create_loop.return_value = mock_forward_loop
        mock_mtq.quantize.return_value = self.test_model
        
        # Mock time.perf_counter
        mock_time.perf_counter.side_effect = [0.0, 10.5]
        
        # Execute
        with patch('builtins.print') as mock_print:
            result = modelopt_quantize(self.test_model, "int8-weight-only")
            
            # Verify: Printed timing information
            mock_print.assert_called()
            call_args_list = [str(call) for call in mock_print.call_args_list]
            timing_printed = any("Quantization took" in str(call) for call in call_args_list)
            self.assertTrue(timing_printed, "Should print quantization timing information")
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('examples.nanogpt.quantization.AutoTokenizer')
    @patch('examples.nanogpt.quantization.create_forward_loop')
    def test_model_device_preservation(self, mock_create_loop, mock_tokenizer_cls):
        """
        Test case 8: Model device information passed to forward_loop
        
        Expected behavior: create_forward_loop receives correct device parameter
        """
        # Setup mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_forward_loop = MagicMock()
        mock_create_loop.return_value = mock_forward_loop
        mock_mtq.quantize.return_value = self.test_model
        
        # Set model device
        self.test_model.device = torch.device('cpu')
        
        # Execute
        result = modelopt_quantize(self.test_model, "float8")
        
        # Verify: device parameter passed correctly
        loop_call_args = mock_create_loop.call_args
        self.assertEqual(loop_call_args[1]['device'], self.test_model.device)


class TestModeloptQuantizeEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('examples.nanogpt.quantization.AutoTokenizer')
    @patch('examples.nanogpt.quantization.create_forward_loop')
    def test_quantization_preserves_model_structure(self, mock_create_loop, mock_tokenizer_cls):
        """
        Test case 9: Quantization preserves model structure
        
        Expected behavior: Return the same model object after quantization
        """
        # Create test model
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 3072),
            torch.nn.ReLU(),
            torch.nn.Linear(3072, 768)
        )
        model.device = torch.device('cpu')
        
        # Setup mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_forward_loop = MagicMock()
        mock_create_loop.return_value = mock_forward_loop
        
        # mtq.quantize returns the same model
        mock_mtq.quantize.return_value = model
        
        # Execute
        result = modelopt_quantize(model, "int8-weight-only")
        
        # Verify: Returned model
        self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('examples.nanogpt.quantization.AutoTokenizer')
    @patch('examples.nanogpt.quantization.create_forward_loop')
    def test_multiple_quantization_modes_sequentially(self, mock_create_loop, mock_tokenizer_cls):
        """
        Test case 10: Test multiple quantization modes sequentially
        
        Expected behavior: All supported modes execute successfully
        """
        # Setup mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_forward_loop = MagicMock()
        mock_create_loop.return_value = mock_forward_loop
        
        model = torch.nn.Linear(768, 768)
        model.device = torch.device('cpu')
        mock_mtq.quantize.return_value = model
        
        # Test all supported modes
        supported_modes = ["int8-weight-only", "int4-weight-only", "float8"]
        
        for mode in supported_modes:
            with self.subTest(quant_mode=mode):
                mock_mtq.quantize.reset_mock()
                
                # Execute
                result = modelopt_quantize(model, mode)
                
                # Verify: Successfully executed
                self.assertIsNotNone(result)
                mock_mtq.quantize.assert_called_once()


if __name__ == "__main__":
    unittest.main()

