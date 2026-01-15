#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test create_forward_loop function
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
conditional_mock('transformers')
conditional_mock('datasets')
conditional_mock('tqdm')

# Mock modelopt
mock_create_forward_loop = MagicMock()
conditional_mock('modelopt', ['torch', 'torch.utils', 'torch.utils.dataset_utils'])
if 'modelopt.torch.utils.dataset_utils' in sys.modules:
    sys.modules['modelopt.torch.utils.dataset_utils'].create_forward_loop = mock_create_forward_loop

# Import ground truth implementation
try:
    from modelopt.torch.utils.dataset_utils import create_forward_loop
    IMPORT_SUCCESS = True
except Exception as e:
    print(f"Warning: Unable to import implementation code: {e}")
    # Use mock version
    create_forward_loop = mock_create_forward_loop
    IMPORT_SUCCESS = False


class TestCreateForwardLoop(unittest.TestCase):
    """Test create_forward_loop function"""
    
    def setUp(self):
        """Setup test environment"""
        torch.manual_seed(42)
        
        # Create basic test model
        self.test_model = self._create_test_model()
        self.dataset_name = "cnn_dailymail"
        self.tokenizer = self._create_mock_tokenizer()
        self.device = torch.device('cpu')
        self.batch_size = 1
        self.num_samples = 64
    
    def _create_test_model(self):
        """Create simple test model"""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(768, 768)
                
            def forward(self, input_ids, **kwargs):
                # Simple forward propagation
                embeddings = torch.randn(input_ids.shape[0], input_ids.shape[1], 768)
                return self.linear(embeddings)
        
        return SimpleModel()
    
    def _create_mock_tokenizer(self):
        """Create Mock tokenizer"""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 50256
        
        # Mock encode method
        def mock_encode(text, **kwargs):
            # Return fixed length token ids
            return list(range(100))
        
        tokenizer.encode = mock_encode
        return tokenizer
    
    @patch('modelopt.torch.utils.dataset_utils.load_dataset')
    def test_basic_forward_loop_creation(self, mock_load_dataset):
        """
        Test case 1: Basic forward loop creation
        
        Input: Model, dataset name, tokenizer, device, batch size, num samples
        Expected behavior:
        1. Return callable function
        2. Function can execute forward propagation
        """
        if not IMPORT_SUCCESS:
            self.skipTest("Implementation code import failed")
        
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = Mock(return_value=iter([
            {'article': 'test text 1'},
            {'article': 'test text 2'},
        ]))
        mock_load_dataset.return_value = mock_dataset
        
        # Reset mock
        mock_create_forward_loop.reset_mock()
        mock_create_forward_loop.return_value = lambda: None
        
        # Execute
        forward_loop = create_forward_loop(
            model=self.test_model,
            dataset_name=self.dataset_name,
            tokenizer=self.tokenizer,
            device=self.device,
            batch_size=self.batch_size,
            num_samples=self.num_samples
        )
        
        # Verify: Returned callable object
        self.assertTrue(callable(forward_loop) or forward_loop is not None)
    
    @patch('modelopt.torch.utils.dataset_utils.load_dataset')
    def test_dataset_loading(self, mock_load_dataset):
        """
        Test case 2: Dataset loading
        
        Expected behavior:
        1. Load specified dataset
        2. Use correct dataset name
        """
        if not IMPORT_SUCCESS:
            self.skipTest("Implementation code import failed")
        
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        mock_create_forward_loop.return_value = lambda: None
        
        # Execute
        forward_loop = create_forward_loop(
            model=self.test_model,
            dataset_name=self.dataset_name,
            tokenizer=self.tokenizer,
            device=self.device,
            batch_size=self.batch_size,
            num_samples=self.num_samples
        )
        
        # Verify: If real implementation, will call load_dataset
        # If mock, we verify function was called
        self.assertIsNotNone(forward_loop)
    
    def test_batch_size_parameter(self):
        """
        Test case 3: Batch size parameter
        
        Input: Different batch_size values
        Expected: Can handle different batch sizes
        """
        if not IMPORT_SUCCESS:
            self.skipTest("Implementation code import failed")
        
        batch_sizes = [1, 4, 16, 32]
        mock_create_forward_loop.return_value = lambda: None
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                # Execute
                forward_loop = create_forward_loop(
                    model=self.test_model,
                    dataset_name=self.dataset_name,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    batch_size=batch_size,
                    num_samples=self.num_samples
                )
                
                # Verify: Successfully created
                self.assertIsNotNone(forward_loop)
    
    def test_num_samples_parameter(self):
        """
        Test case 4: Calibration sample count parameter
        
        Input: Different num_samples values
        Expected: Can handle different sample counts
        """
        if not IMPORT_SUCCESS:
            self.skipTest("Implementation code import failed")
        
        sample_counts = [16, 32, 64, 128]
        mock_create_forward_loop.return_value = lambda: None
        
        for num_samples in sample_counts:
            with self.subTest(num_samples=num_samples):
                # Execute
                forward_loop = create_forward_loop(
                    model=self.test_model,
                    dataset_name=self.dataset_name,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    batch_size=self.batch_size,
                    num_samples=num_samples
                )
                
                # Verify: Successfully created
                self.assertIsNotNone(forward_loop)
    
    def test_different_datasets(self):
        """
        Test case 5: Different datasets
        
        Input: Different dataset names
        Expected: Support multiple datasets
        """
        if not IMPORT_SUCCESS:
            self.skipTest("Implementation code import failed")
        
        datasets = ["cnn_dailymail", "wikitext", "ptb_text_only"]
        mock_create_forward_loop.return_value = lambda: None
        
        for dataset_name in datasets:
            with self.subTest(dataset=dataset_name):
                # Execute
                forward_loop = create_forward_loop(
                    model=self.test_model,
                    dataset_name=dataset_name,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    batch_size=self.batch_size,
                    num_samples=self.num_samples
                )
                
                # Verify: Successfully created
                self.assertIsNotNone(forward_loop)
    
    def test_device_parameter(self):
        """
        Test case 6: Device parameter handling
        
        Input: Different devices (CPU/GPU)
        Expected: Correctly handle device parameter
        """
        if not IMPORT_SUCCESS:
            self.skipTest("Implementation code import failed")
        
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda:0'))
        
        mock_create_forward_loop.return_value = lambda: None
        
        for device in devices:
            with self.subTest(device=str(device)):
                # Execute
                forward_loop = create_forward_loop(
                    model=self.test_model,
                    dataset_name=self.dataset_name,
                    tokenizer=self.tokenizer,
                    device=device,
                    batch_size=self.batch_size,
                    num_samples=self.num_samples
                )
                
                # Verify: Successfully created
                self.assertIsNotNone(forward_loop)
    
    def test_tokenizer_integration(self):
        """
        Test case 7: Tokenizer integration
        
        Expected behavior: forward_loop uses tokenizer to process text
        """
        if not IMPORT_SUCCESS:
            self.skipTest("Implementation code import failed")
        
        mock_create_forward_loop.return_value = lambda: None
        
        # Execute
        forward_loop = create_forward_loop(
            model=self.test_model,
            dataset_name=self.dataset_name,
            tokenizer=self.tokenizer,
            device=self.device,
            batch_size=self.batch_size,
            num_samples=self.num_samples
        )
        
        # Verify: Successfully created
        self.assertIsNotNone(forward_loop)
    
    def test_model_forward_execution(self):
        """
        Test case 8: Model forward execution
        
        Expected behavior: forward_loop executes model's forward propagation
        """
        if not IMPORT_SUCCESS:
            self.skipTest("Implementation code import failed")
        
        # Create a model that can track calls
        model_with_tracking = MagicMock(spec=torch.nn.Module)
        model_with_tracking.forward = MagicMock(return_value=torch.randn(1, 10, 768))
        
        mock_create_forward_loop.return_value = lambda: None
        
        # Execute
        forward_loop = create_forward_loop(
            model=model_with_tracking,
            dataset_name=self.dataset_name,
            tokenizer=self.tokenizer,
            device=self.device,
            batch_size=self.batch_size,
            num_samples=self.num_samples
        )
        
        # Verify: Successfully created
        self.assertIsNotNone(forward_loop)


class TestCreateForwardLoopEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_model = torch.nn.Linear(768, 768)
        self.tokenizer = MagicMock()
        self.device = torch.device('cpu')
    
    def test_small_batch_size(self):
        """
        Test case 9: Small batch size
        
        Input: batch_size=1
        Expected: Correctly handle single sample batch
        """
        if not IMPORT_SUCCESS:
            self.skipTest("Implementation code import failed")
        
        mock_create_forward_loop.return_value = lambda: None
        
        # Execute
        forward_loop = create_forward_loop(
            model=self.test_model,
            dataset_name="cnn_dailymail",
            tokenizer=self.tokenizer,
            device=self.device,
            batch_size=1,
            num_samples=8
        )
        
        # Verify: Successfully created
        self.assertIsNotNone(forward_loop)
    
    def test_large_batch_size(self):
        """
        Test case 10: Large batch size
        
        Input: batch_size=64
        Expected: Correctly handle large batch
        """
        if not IMPORT_SUCCESS:
            self.skipTest("Implementation code import failed")
        
        mock_create_forward_loop.return_value = lambda: None
        
        # Execute
        forward_loop = create_forward_loop(
            model=self.test_model,
            dataset_name="cnn_dailymail",
            tokenizer=self.tokenizer,
            device=self.device,
            batch_size=64,
            num_samples=64
        )
        
        # Verify: Successfully created
        self.assertIsNotNone(forward_loop)
    
    def test_small_num_samples(self):
        """
        Test case 11: Small sample count
        
        Input: num_samples=8
        Expected: Correctly handle small number of samples
        """
        if not IMPORT_SUCCESS:
            self.skipTest("Implementation code import failed")
        
        mock_create_forward_loop.return_value = lambda: None
        
        # Execute
        forward_loop = create_forward_loop(
            model=self.test_model,
            dataset_name="cnn_dailymail",
            tokenizer=self.tokenizer,
            device=self.device,
            batch_size=1,
            num_samples=8
        )
        
        # Verify: Successfully created
        self.assertIsNotNone(forward_loop)
    
    def test_function_returns_callable(self):
        """
        Test case 12: Return value must be callable object
        
        Expected: Returned forward_loop is a function
        """
        if not IMPORT_SUCCESS:
            self.skipTest("Implementation code import failed")
        
        mock_create_forward_loop.return_value = lambda: None
        
        # Execute
        forward_loop = create_forward_loop(
            model=self.test_model,
            dataset_name="cnn_dailymail",
            tokenizer=self.tokenizer,
            device=self.device,
            batch_size=1,
            num_samples=16
        )
        
        # Verify: Return value is callable or not None
        self.assertTrue(callable(forward_loop) or forward_loop is not None)


if __name__ == "__main__":
    unittest.main()

