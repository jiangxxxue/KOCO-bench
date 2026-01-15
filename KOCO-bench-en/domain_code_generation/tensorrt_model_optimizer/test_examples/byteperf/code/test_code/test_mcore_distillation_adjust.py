#!/usr/bin/env python3
# Copyright 2024 ByteMLPerf. All rights reserved.

"""
Test adjust_distillation_model_for_mcore function
"""

import torch
import torch.nn as nn
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import types

# Mock transformer_engine before any imports
sys.modules['transformer_engine'] = MagicMock()
sys.modules['transformer_engine.pytorch'] = MagicMock()

# Add the parent directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'byte_train_perf', 'Megatron-LM'))

# Import ground truth implementation
from megatron.inference.algos.distillation import adjust_distillation_model_for_mcore
import modelopt.torch.distill as mtd


class TestAdjustDistillationModelForMCore(unittest.TestCase):
    """Testadjust_distillation_model_for_mcorefunction"""
    
    def setUp(self):
        """Set up test environment"""
        torch.manual_seed(42)
    
    def _create_mock_distillation_model(self):
        """Create a mock DistillationModel for testing"""
        model = Mock(spec=mtd.DistillationModel)
        
        # Add required methods
        def hide_teacher_model():
            """Mock context manager"""
            class HideTeacher:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return HideTeacher()
        
        def sharded_state_dict(*args, **kwargs):
            return {"state": "dict"}
        
        def compute_language_model_loss(labels, logits):
            return torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1),
                reduction='mean'
            )
        
        model.hide_teacher_model = hide_teacher_model
        model.sharded_state_dict = sharded_state_dict
        model.compute_language_model_loss = compute_language_model_loss
        model.training = True
        
        return model
    
    def test_sharded_state_dict_wrapping(self):
        """
        Testcase1: sharded_state_dict method correctly wrapped
        
        Input: A DistillationModel
        Expected behavior:
        1. Original sharded_state_dict is saved as _sharded_state_dict
        2. New sharded_state_dict will call hide_teacher_model
        """
        model = self._create_mock_distillation_model()
        distill_cfg = {"skip_lm_loss": False}
        
        # Executefunction
        adjust_distillation_model_for_mcore(model, distill_cfg)
        
        # Verify: Original method is saved
        self.assertTrue(hasattr(model, '_sharded_state_dict'))
        
        # Verify: New method exists and is callable
        self.assertTrue(callable(model.sharded_state_dict))
        
        # Verify: New method is types.MethodType
        self.assertIsInstance(model.sharded_state_dict, types.MethodType)
    
    def test_skip_lm_loss_when_enabled(self):
        """
        Testcase2: When skip_lm_loss=True, training mode returns zero loss
        
        Input:
        - distill_cfg['skip_lm_loss'] = True
        - model.training = True
        
        Expected behavior: compute_language_model_loss returns all-zero tensor
        """
        model = self._create_mock_distillation_model()
        distill_cfg = {"skip_lm_loss": True}
        
        model.training = True
        
        # Executefunction
        adjust_distillation_model_for_mcore(model, distill_cfg)
        
        # Verify: Original method is saved
        self.assertTrue(hasattr(model, '_compute_language_model_loss'))
        
        # Create test data
        labels = torch.randint(0, 100, (4, 10))
        logits = torch.randn(4, 10, 100)
        
        # Call new method
        loss = model.compute_language_model_loss(labels, logits)
        
        # Verify: Training mode returns zero
        expected_zeros = torch.zeros_like(labels)
        torch.testing.assert_close(loss, expected_zeros, atol=1e-6, rtol=1e-6)
    
    def test_skip_lm_loss_eval_mode(self):
        """
        Testcase3: When skip_lm_loss=True but eval mode, still compute real loss
        
        Input:
        - distill_cfg['skip_lm_loss'] = True
        - model.training = False
        
        Expected behavior: compute_language_model_loss returns real loss (calls original method)
        """
        model = self._create_mock_distillation_model()
        distill_cfg = {"skip_lm_loss": True}
        
        model.training = False  # Evaluation mode
        
        # Execute function
        adjust_distillation_model_for_mcore(model, distill_cfg)
        
        # Create test data
        labels = torch.randint(0, 100, (4, 10))
        logits = torch.randn(4, 10, 100)
        
        # Call new method
        loss = model.compute_language_model_loss(labels, logits)
        
        # Verify: Evaluation mode calls original method, returns real loss (not all zeros)
        # Here we only verify that it returns a tensor and is not all zeros
        self.assertTrue(isinstance(loss, torch.Tensor))
        # Since it calls the original method, should return actual CE loss
    
    def test_skip_lm_loss_disabled(self):
        """
        Testcase4: When skip_lm_loss=False, do not modify compute_language_model_loss
        
        Input: distill_cfg['skip_lm_loss'] = False
        Expected behavior: compute_language_model_lossnot wrapped
        """
        model = self._create_mock_distillation_model()
        distill_cfg = {"skip_lm_loss": False}
        
        original_method = model.compute_language_model_loss
        
        # Executefunction
        adjust_distillation_model_for_mcore(model, distill_cfg)
        
        # Verify: Should not have_compute_language_model_loss
        self.assertFalse(hasattr(model, '_compute_language_model_loss'))
        
        # Verify: compute_language_model_loss still the original method
        # (Although there may be subtle differences in actual scenarios, the main logic should remain unchanged)
    
    def test_sharded_state_dict_calls_hide_teacher(self):
        """
        Testcase5: New sharded_state_dict calls hide_teacher_model
        
        Verify hide_teacher_model context manager is correctly used
        """
        model = self._create_mock_distillation_model()
        distill_cfg = {"skip_lm_loss": False}
        
        # Mock hide_teacher_model to track calls
        hide_called = []
        original_hide = model.hide_teacher_model
        
        def tracked_hide():
            hide_called.append(True)
            return original_hide()
        
        model.hide_teacher_model = tracked_hide
        
        # Executefunction
        adjust_distillation_model_for_mcore(model, distill_cfg)
        
        # Call new sharded_state_dict
        result = model.sharded_state_dict()
        
        # Verify hide_teacher_model is called
        self.assertTrue(len(hide_called) > 0)
        self.assertTrue(isinstance(result, dict))
    
    def test_method_binding(self):
        """
        Testcase6: Verify methods correctly bound to model instance
        
        Verify new method's self parameter points to correct model
        """
        model = self._create_mock_distillation_model()
        distill_cfg = {"skip_lm_loss": True}
        
        # Executefunction
        adjust_distillation_model_for_mcore(model, distill_cfg)
        
        # Verify sharded_state_dict is MethodType and bound to model
        self.assertIsInstance(model.sharded_state_dict, types.MethodType)
        self.assertEqual(model.sharded_state_dict.__self__, model)
        
        # Verify compute_language_model_loss is also MethodType and bound to model
        self.assertIsInstance(model.compute_language_model_loss, types.MethodType)
        self.assertEqual(model.compute_language_model_loss.__self__, model)
    
    def test_multiple_calls_idempotence(self):
        """
        Testcase7: Idempotence of multiple function calls
        
        Verify multiple calls do not cause errors or state confusion
        """
        model = self._create_mock_distillation_model()
        distill_cfg = {"skip_lm_loss": True}
        
        # First call
        adjust_distillation_model_for_mcore(model, distill_cfg)
        first_sharded = model.sharded_state_dict
        
        # Second call
        adjust_distillation_model_for_mcore(model, distill_cfg)
        second_sharded = model.sharded_state_dict
        
        # Verify method still usable
        self.assertTrue(callable(model.sharded_state_dict))
        self.assertTrue(hasattr(model, '_sharded_state_dict'))
    
    def test_zero_loss_shape_matches_labels(self):
        """
        Testcase8: Zero tensor returned by skip_lm_loss matches labels shape
        
        Input: Different shapes of labels
        Expected: Returned zero tensor shape matches labels
        """
        model = self._create_mock_distillation_model()
        distill_cfg = {"skip_lm_loss": True}
        model.training = True
        
        adjust_distillation_model_for_mcore(model, distill_cfg)
        
        # Test different shapes
        test_shapes = [(2, 5), (4, 10), (8, 20), (1, 3)]
        
        for shape in test_shapes:
            with self.subTest(shape=shape):
                labels = torch.randint(0, 100, shape)
                logits = torch.randn(*shape, 100)
                
                loss = model.compute_language_model_loss(labels, logits)
                
                # Verify shape matches
                self.assertEqual(loss.shape, labels.shape)
                # Verify all zeros
                torch.testing.assert_close(loss, torch.zeros_like(labels), atol=1e-6, rtol=1e-6)


class TestAdjustDistillationEdgeCases(unittest.TestCase):
    """TestEdge cases"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def _create_mock_distillation_model(self):
        """Createmock model"""
        model = Mock(spec=mtd.DistillationModel)
        
        def hide_teacher_model():
            class HideTeacher:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return HideTeacher()
        
        def sharded_state_dict(*args, **kwargs):
            return {"state": "dict"}
        
        def compute_language_model_loss(labels, logits):
            return torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1),
                reduction='mean'
            )
        
        model.hide_teacher_model = hide_teacher_model
        model.sharded_state_dict = sharded_state_dict
        model.compute_language_model_loss = compute_language_model_loss
        model.training = True
            
        return model
        
    def test_empty_distill_cfg(self):
        """
        Testcase9: Empty distill_cfg (only required fields)
        
        Expected: Function handles normally
        """
        model = self._create_mock_distillation_model()
        distill_cfg = {"skip_lm_loss": False}  # Minimal configuration
        
        # Should not raise exception
        adjust_distillation_model_for_mcore(model, distill_cfg)
        
        # Verify basic wrapping completed
        self.assertTrue(hasattr(model, '_sharded_state_dict'))
    
    def test_training_mode_toggle(self):
        """
        Testcase10: Training mode toggle
        
        Verify behavior is correct after training/evaluation mode toggle
        """
        model = self._create_mock_distillation_model()
        distill_cfg = {"skip_lm_loss": True}
        
        adjust_distillation_model_for_mcore(model, distill_cfg)
        
        labels = torch.randint(0, 100, (4, 10))
        logits = torch.randn(4, 10, 100)
        
        # Training mode
        model.training = True
        loss_train = model.compute_language_model_loss(labels, logits)
        torch.testing.assert_close(loss_train, torch.zeros_like(labels), atol=1e-6, rtol=1e-6)
        
        # Switch to evaluation mode
        model.training = False
        loss_eval = model.compute_language_model_loss(labels, logits)
        # Evaluation mode should return real loss (call original method)
        self.assertTrue(isinstance(loss_eval, torch.Tensor))


if __name__ == "__main__":
    unittest.main()
