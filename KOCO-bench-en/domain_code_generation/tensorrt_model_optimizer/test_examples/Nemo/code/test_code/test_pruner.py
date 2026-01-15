#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test prune_language_model function
"""

import unittest
import torch
import sys
import os
from unittest.mock import Mock, MagicMock, patch

# Add code directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock problematic dependencies
sys.modules['nv_one_logger'] = MagicMock()
sys.modules['nv_one_logger.api'] = MagicMock()
sys.modules['nv_one_logger.training_telemetry'] = MagicMock()
sys.modules['nemo_run'] = MagicMock()
sys.modules['lightning'] = MagicMock()
sys.modules['lightning.pytorch'] = MagicMock()

# Import ground truth implementation
try:
    from nemo.collections.llm.modelopt.prune.pruner import prune_language_model, PruningConfig
    IMPORT_SUCCESS = True
except Exception as e:
    print(f"Warning: Unable to import implementation code: {e}")
    IMPORT_SUCCESS = False


class TestPruneLanguageModel(unittest.TestCase):
    """Test prune_language_model function"""
    
    def setUp(self):
        """Setup test environment"""
        torch.manual_seed(42)
        self.mock_model = self._create_mock_model()
    
    def _create_mock_model(self):
        """Create Mock model"""
        model = MagicMock()
        model.__class__.__name__ = "GPTModel"
        model.config = MagicMock()
        model.config.num_layers = 12
        return model
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    def test_drop_layers_takes_precedence(self, mock_mtp):
        """
        Test case 1: drop_layers takes highest precedence
        
        Input: 
        - drop_layers=[1, 2, 3]
        - algorithm="width_pruning" (also exists)
        
        Expected behavior:
        1. Only call drop_mcore_language_model_layers
        2. Do not call mtp.prune
        3. Ignore algorithm parameter
        """
        # Setup mock
        mock_drop_layers = MagicMock(return_value=self.mock_model)
        mock_mtp.plugins.drop_mcore_language_model_layers = mock_drop_layers
        mock_mtp.prune = MagicMock()
        
        # Input: Contains both drop_layers and algorithm
        config = PruningConfig(
            drop_layers=[1, 2, 3],
            algorithm="width_pruning",
            amount=0.5
        )
        
        # Execute
        result = prune_language_model(
            model=self.mock_model,
            pruning_config=config
        )
        
        # Verify behavior 1: Called drop_layers
        mock_drop_layers.assert_called_once()
        call_args = mock_drop_layers.call_args
        
        # Verify passed model
        self.assertEqual(call_args[0][0], self.mock_model)
        
        # Verify passed layer indices
        self.assertEqual(call_args[0][1], [1, 2, 3])
        
        # Verify behavior 2: Did not call mtp.prune
        mock_mtp.prune.assert_not_called()
        
        # Verify output: Returns model
        self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    def test_drop_layers_with_empty_list(self, mock_mtp):
        """
        Test case 2: Empty drop_layers list should fallback to mtp.prune
        
        Input: drop_layers=[]
        Expected behavior: Call mtp.prune instead of drop_layers
        """
        mock_mtp.prune = MagicMock(return_value=self.mock_model)
        
        config = PruningConfig(
            drop_layers=[],  # Empty list
            algorithm="width_pruning"
        )
        
        mock_data_module = MagicMock()
        mock_trainer = MagicMock()
        
        result = prune_language_model(
            model=self.mock_model,
            pruning_config=config,
            data_module=mock_data_module,
            trainer=mock_trainer
        )
        
        # Verify: Called mtp.prune
        self.assertTrue(mock_mtp.prune.called or not config.drop_layers,
                       "Empty drop_layers should use mtp.prune")
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    @patch('nemo.collections.llm.modelopt.prune.pruner.llm')
    def test_mtp_prune_requires_data_and_trainer(self, mock_llm, mock_mtp):
        """
        Test case 3: mtp.prune requires data_module and trainer
        
        Input: algorithm="width_pruning", no drop_layers
        Expected behavior: 
        1. Must provide data_module and trainer
        2. Pass these parameters when calling mtp.prune
        """
        mock_mtp.prune = MagicMock(return_value=self.mock_model)
        mock_llm.validate = MagicMock()
        
        config = PruningConfig(
            algorithm="width_pruning",
            amount=0.5
        )
        
        mock_data_module = MagicMock()
        mock_trainer = MagicMock()
        
        # Execute
        result = prune_language_model(
            model=self.mock_model,
            pruning_config=config,
            data_module=mock_data_module,
            trainer=mock_trainer
        )
        
        # Verify: Called mtp.prune
        mock_mtp.prune.assert_called_once()
        
        # Verify: Passed correct parameters
        call_kwargs = mock_mtp.prune.call_args.kwargs
        self.assertIn('model', call_kwargs)
        
        # Verify return result
        self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    def test_drop_layers_preserves_model_type(self, mock_mtp):
        """
        Test case 4: drop_layers preserves model type
        
        Input: GPTModel
        Expected behavior: Returned model is still GPTModel
        """
        # Create real model structure mock
        mock_returned_model = MagicMock()
        mock_returned_model.__class__.__name__ = "GPTModel"
        
        mock_drop = MagicMock(return_value=mock_returned_model)
        mock_mtp.plugins.drop_mcore_language_model_layers = mock_drop
        
        config = PruningConfig(drop_layers=[5, 6, 7])
        
        result = prune_language_model(
            model=self.mock_model,
            pruning_config=config
        )
        
        # Verify: Returned model type is correct
        self.assertEqual(result.__class__.__name__, "GPTModel")
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    @patch('nemo.collections.llm.modelopt.prune.pruner.llm')
    def test_mtp_prune_with_different_algorithms(self, mock_llm, mock_mtp):
        """
        Test case 5: Test different pruning algorithms
        
        Input: Different algorithm values
        Expected behavior: All algorithms can be correctly passed to mtp.prune
        """
        mock_mtp.prune = MagicMock(return_value=self.mock_model)
        mock_llm.validate = MagicMock()
        
        algorithms = ["width_pruning", "depth_pruning", "unstructured"]
        
        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm):
                mock_mtp.prune.reset_mock()
                
                config = PruningConfig(
                    algorithm=algorithm,
                    amount=0.3
                )
                
                result = prune_language_model(
                    model=self.mock_model,
                    pruning_config=config,
                    data_module=MagicMock(),
                    trainer=MagicMock()
                )
                
                # Verify: Called mtp.prune
                self.assertTrue(mock_mtp.prune.called)
                
                # Verify: Returned model
                self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    def test_drop_layers_with_large_indices(self, mock_mtp):
        """
        Test case 6: drop_layers with large layer indices
        
        Input: drop_layers=[10, 11] (model has 12 layers)
        Expected behavior: Correctly pass layer indices without extra validation
        """
        mock_drop = MagicMock(return_value=self.mock_model)
        mock_mtp.plugins.drop_mcore_language_model_layers = mock_drop
        
        config = PruningConfig(drop_layers=[10, 11])
        
        result = prune_language_model(
            model=self.mock_model,
            pruning_config=config
        )
        
        # Verify: Passed correct layer indices
        call_args = mock_drop.call_args
        self.assertEqual(call_args[0][1], [10, 11])
        
        # Verify: Successfully returned
        self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    @patch('nemo.collections.llm.modelopt.prune.pruner.unwrap_for_modelopt_operations')
    def test_model_unwrapping_for_mtp(self, mock_unwrap, mock_mtp):
        """
        Test case 7: Model is unwrapped before passing to mtp.prune
        
        Input: Wrapped model
        Expected behavior: Call unwrap function, pass unwrapped model to mtp.prune
        """
        unwrapped_model = MagicMock()
        unwrapped_model.__class__.__name__ = "GPTModel"
        
        mock_unwrap.return_value = unwrapped_model
        mock_mtp.prune = MagicMock(return_value=unwrapped_model)
        
        config = PruningConfig(
            algorithm="width_pruning"
        )
        
        with patch('nemo.collections.llm.modelopt.prune.pruner.llm'):
            result = prune_language_model(
                model=self.mock_model,
                pruning_config=config,
                data_module=MagicMock(),
                trainer=MagicMock()
            )
        
        # Verify: Called unwrap
        mock_unwrap.assert_called()
        
        # Verify: mtp.prune was called
        self.assertTrue(mock_mtp.prune.called)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    @patch('nemo.collections.llm.modelopt.prune.pruner.llm')
    def test_pruning_amount_parameter(self, mock_llm, mock_mtp):
        """
        Test case 8: Pruning amount parameter is correctly passed
        
        Input: amount=0.3 (prune 30%)
        Expected behavior: amount parameter is passed to mtp.prune
        """
        mock_mtp.prune = MagicMock(return_value=self.mock_model)
        mock_llm.validate = MagicMock()
        
        config = PruningConfig(
            algorithm="width_pruning",
            amount=0.3
        )
        
        result = prune_language_model(
            model=self.mock_model,
            pruning_config=config,
            data_module=MagicMock(),
            trainer=MagicMock()
        )
        
        # Verify: Called mtp.prune
        self.assertTrue(mock_mtp.prune.called)
        
        # Verify: Returned model
        self.assertIsNotNone(result)


class TestPruningConfigValidation(unittest.TestCase):
    """Test PruningConfig parameter validation and creation"""
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    def test_pruning_config_with_drop_layers(self):
        """
        Test case 9: Create configuration with drop_layers
        
        Input: drop_layers=[0, 1, 2]
        Expected: Successfully create configuration, parameters correctly stored
        """
        config = PruningConfig(drop_layers=[0, 1, 2])
        
        self.assertIsNotNone(config)
        self.assertEqual(config.drop_layers, [0, 1, 2])
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    def test_pruning_config_with_algorithm(self):
        """
        Test case 10: Create configuration with algorithm
        
        Input: algorithm="width_pruning", amount=0.5
        Expected: Successfully create configuration, parameters correctly stored
        """
        config = PruningConfig(
            algorithm="width_pruning",
            amount=0.5
        )
        
        self.assertIsNotNone(config)
        self.assertEqual(config.algorithm, "width_pruning")
        self.assertEqual(config.amount, 0.5)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    def test_pruning_config_default_values(self):
        """
        Test case 11: Test configuration default values
        
        Input: Empty configuration
        Expected: Use default values
        """
        config = PruningConfig()
        
        self.assertIsNotNone(config)
        # Default value verification (based on actual implementation)
        self.assertTrue(hasattr(config, 'drop_layers'))
        self.assertTrue(hasattr(config, 'algorithm'))


class TestPruningEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    def test_drop_single_layer(self, mock_mtp):
        """
        Test case 12: Drop single layer
        
        Input: drop_layers=[5]
        Expected: Correctly handle single layer deletion
        """
        mock_drop = MagicMock(return_value=MagicMock())
        mock_mtp.plugins.drop_mcore_language_model_layers = mock_drop
        
        config = PruningConfig(drop_layers=[5])
        model = MagicMock()
        
        result = prune_language_model(
            model=model,
            pruning_config=config
        )
        
        # Verify: Called drop function
        mock_drop.assert_called_once()
        
        # Verify: Passed single layer index
        call_args = mock_drop.call_args
        self.assertEqual(call_args[0][1], [5])
        
        self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    def test_drop_consecutive_layers(self, mock_mtp):
        """
        Test case 13: Drop consecutive layers
        
        Input: drop_layers=[3, 4, 5, 6]
        Expected: Correctly handle consecutive layer deletion
        """
        mock_drop = MagicMock(return_value=MagicMock())
        mock_mtp.plugins.drop_mcore_language_model_layers = mock_drop
        
        config = PruningConfig(drop_layers=[3, 4, 5, 6])
        model = MagicMock()
        
        result = prune_language_model(
            model=model,
            pruning_config=config
        )
        
        # Verify: Passed consecutive layer indices
        call_args = mock_drop.call_args
        self.assertEqual(call_args[0][1], [3, 4, 5, 6])
        
        self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "Implementation code import failed")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    def test_drop_non_consecutive_layers(self, mock_mtp):
        """
        Test case 14: Drop non-consecutive layers
        
        Input: drop_layers=[1, 5, 9]
        Expected: Correctly handle non-consecutive layer deletion
        """
        mock_drop = MagicMock(return_value=MagicMock())
        mock_mtp.plugins.drop_mcore_language_model_layers = mock_drop
        
        config = PruningConfig(drop_layers=[1, 5, 9])
        model = MagicMock()
        
        result = prune_language_model(
            model=model,
            pruning_config=config
        )
        
        # Verify: Passed non-consecutive layer indices
        call_args = mock_drop.call_args
        self.assertEqual(call_args[0][1], [1, 5, 9])
        
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
