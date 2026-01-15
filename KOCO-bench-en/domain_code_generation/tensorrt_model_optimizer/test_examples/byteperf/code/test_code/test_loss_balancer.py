#!/usr/bin/env python3
# Copyright 2024 ByteMLPerf. All rights reserved.

"""
Test LogitsAndIntermediatesLossBalancer.forward function
"""

import torch
import torch.nn as nn
import unittest
from unittest.mock import Mock, patch

# Import the actual implementation
import sys
import os
# Use insert(0) instead of append to avoid being overridden by installed megatron package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'byte_train_perf', 'Megatron-LM'))

# Ensure correct import of actual implementation
from megatron.inference.algos.distillation import LogitsAndIntermediatesLossBalancer
import modelopt.torch.distill as mtd
from modelopt.torch.distill import DistillationLossBalancer


class TestLogitsAndIntermediatesLossBalancer(unittest.TestCase):
    """TestLogitsAndIntermediatesLossBalancer.forwardfunction"""
    
    def setUp(self):
        """Set up test environment"""
        torch.manual_seed(42)
    
    def test_init_default_params(self):
        """
        Testcase1: Default parameter initialization
        
        Input: No parameters
        Expected behavior: kd_loss_scale=1.0, skip_original_loss=False
        """
        balancer = LogitsAndIntermediatesLossBalancer()
        
        self.assertEqual(balancer._kd_loss_scale, 1.0)
        self.assertEqual(balancer._skip_original_loss, False)
    
    def test_init_custom_params(self):
        """
        Testcase2: Custom parameter initialization
        
        Input: kd_loss_scale=2.0, skip_original_loss=True
        Expected behavior: Parameters correctly set
        """
        balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=2.0, skip_original_loss=True)
        
        self.assertEqual(balancer._kd_loss_scale, 2.0)
        self.assertEqual(balancer._skip_original_loss, True)
    
    def test_forward_with_intermediate_loss(self):
        """
        Testcase3: Forward pass with intermediate loss
        
        Input:
        - student_loss: 2.0
        - LogitsKLLoss: 1.0
        - intermediate_loss: 0.5
        - kd_loss_scale: 1.0
        - skip_original_loss: False
        
        Expected behavior: 
        1. Remove student_loss
        2. Compute dynamic scaling
        3. Return balanced total loss
        """
        balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=1.0, skip_original_loss=False)
        
        loss_dict = {
            "student_loss": torch.tensor(2.0),
            "LogitsKLLoss": torch.tensor(1.0),
            "intermediate_loss": torch.tensor(0.5)
        }
        
        result = balancer.forward(loss_dict)
        
        # VerifyReturntensor
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.dim(), 0)
        
        # Verify specific values:
        # logits_loss = 1.0, intermediate_loss = 0.5
        # dynamic_scale = 1.0 / 0.5 = 2.0
        # intermediate_loss_scaled = 0.5 * 2.0 = 1.0
        # kd_loss_scale = 0.5 (original * 0.5)
        # kd_loss = (1.0 + 1.0) * 0.5 = 1.0
        # dynamic_scale2 = 2.0 / 1.0 = 2.0
        # total_loss = 2.0 + 1.0 * 2.0 = 4.0
        expected_loss = torch.tensor(4.0)
        torch.testing.assert_close(result, expected_loss, atol=1e-4, rtol=1e-4)
    
    def test_forward_without_intermediate_loss(self):
        """
        Testcase4: Forward pass without intermediate loss
        
        Input:
        - student_loss: 2.0
        - LogitsKLLoss: 1.0
        - kd_loss_scale: 1.0
        - skip_original_loss: False
        
        Expected behavior: Only use logits loss, no intermediate layer scaling
        """
        balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=1.0, skip_original_loss=False)
        
        loss_dict = {
            "student_loss": torch.tensor(2.0),
            "LogitsKLLoss": torch.tensor(1.0)
        }
        
        result = balancer.forward(loss_dict)
        
        # Verify: intermediate_loss = 0, no scaling
        # kd_loss = 1.0 * 1.0 = 1.0
        # dynamic_scale = 2.0 / 1.0 = 2.0
        # total_loss = 2.0 + 1.0 * 2.0 = 4.0
        expected_loss = torch.tensor(4.0)
        torch.testing.assert_close(result, expected_loss, atol=1e-4, rtol=1e-4)
    
    def test_forward_skip_original_loss(self):
        """
        Testcase5: Skip original loss
        
        Input:
        - student_loss: 2.0
        - LogitsKLLoss: 1.0
        - intermediate_loss: 0.5
        - skip_original_loss: True
        
        Expected behavior: Only return distillation loss, excluding original student_loss
        """
        balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=1.0, skip_original_loss=True)
        
        loss_dict = {
            "student_loss": torch.tensor(2.0),
            "LogitsKLLoss": torch.tensor(1.0),
            "intermediate_loss": torch.tensor(0.5)
        }
        
        result = balancer.forward(loss_dict)
        
        # Verify：skip_original_loss=True
        # logits_loss = 1.0, intermediate_loss = 0.5
        # dynamic_scale = 1.0 / 0.5 = 2.0
        # intermediate_loss_scaled = 0.5 * 2.0 = 1.0
        # total_loss = 1.0 + 1.0 = 2.0 (excluding original_loss)
        # But needs to multiply by kd_loss_scale (0.5 after halving)
        # Actual: logits_loss + intermediate_loss_scaled = 1.0 + 1.0 = 2.0
        expected_loss = torch.tensor(2.0)
        torch.testing.assert_close(result, expected_loss, atol=1e-4, rtol=1e-4)
    
    def test_forward_dynamic_scaling(self):
        """
        Testcase6: Dynamic scaling computation
        
        Input:
        - student_loss: 3.0
        - LogitsKLLoss: 2.0
        - intermediate_loss: 1.0
        - kd_loss_scale: 2.0
        
        Expected behavior: Verify correctness of dynamic scaling
        """
        balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=2.0, skip_original_loss=False)
        
        loss_dict = {
            "student_loss": torch.tensor(3.0),
            "LogitsKLLoss": torch.tensor(2.0),
            "intermediate_loss": torch.tensor(1.0)
        }
        
        result = balancer.forward(loss_dict)
        
        # Verify dynamic scaling:
        # dynamic_scale = 2.0 / 1.0 = 2.0
        # intermediate_loss_scaled = 1.0 * 2.0 = 2.0
        # kd_loss_scale = 2.0 * 0.5 = 1.0
        # kd_loss = (2.0 + 2.0) * 1.0 = 4.0
        # dynamic_scale2 = 3.0 / 4.0 = 0.75
        # total_loss = 3.0 + 4.0 * 0.75 = 6.0
        expected_loss = torch.tensor(6.0)
        torch.testing.assert_close(result, expected_loss, atol=1e-4, rtol=1e-4)
    
    def test_forward_specific_values(self):
        """
        Testcase7: Specific value verification
        
        Verify each step of computation using specific values
        """
        balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=1.5, skip_original_loss=False)
        
        loss_dict = {
            "student_loss": torch.tensor(5.0),
            "LogitsKLLoss": torch.tensor(3.0),
            "intermediate_loss": torch.tensor(2.0)
        }
        
        result = balancer.forward(loss_dict)
        
        # Manual calculation:
        # logits_loss = 3.0
        # intermediate_loss = 2.0
        # dynamic_scale = 3.0 / 2.0 = 1.5
        # intermediate_loss_scaled = 2.0 * 1.5 = 3.0
        # kd_loss_scale = 1.5 * 0.5 = 0.75
        # kd_loss = (3.0 + 3.0) * 0.75 = 4.5
        # dynamic_scale2 = 5.0 / 4.5 = 1.1111...
        # total_loss = 5.0 + 4.5 * 1.1111... = 10.0
        expected_loss = torch.tensor(10.0)
        torch.testing.assert_close(result, expected_loss, atol=1e-4, rtol=1e-4)
    
    def test_intermediate_loss_zero(self):
        """
        Testcase8: Intermediate loss is zero
        
        Input: intermediate_loss = 0.0
        Expected behavior: No dynamic scaling, use original kd_loss_scale
        """
        balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=1.0, skip_original_loss=False)
        
        loss_dict = {
            "student_loss": torch.tensor(2.0),
            "LogitsKLLoss": torch.tensor(1.0),
            "intermediate_loss": torch.tensor(0.0)
        }
        
        result = balancer.forward(loss_dict)
        
        # intermediate_loss = 0, no scaling
        # kd_loss = 1.0 * 1.0 = 1.0
        # dynamic_scale = 2.0 / 1.0 = 2.0
        # total_loss = 2.0 + 1.0 * 2.0 = 4.0
        expected_loss = torch.tensor(4.0)
        torch.testing.assert_close(result, expected_loss, atol=1e-4, rtol=1e-4)
    
    def test_loss_dict_modification(self):
        """
        Testcase9: Verifyloss_dict is correctly modified
        
        Expected behavior: student_loss removed from dictionary
        """
        balancer = LogitsAndIntermediatesLossBalancer()
        
        loss_dict = {
            "student_loss": torch.tensor(2.0),
            "LogitsKLLoss": torch.tensor(1.0),
            "intermediate_loss": torch.tensor(0.5)
        }
        
        # Save original keys
        original_keys = set(loss_dict.keys())
        
        result = balancer.forward(loss_dict)
        
        # Verify student_loss is removed
        self.assertNotIn("student_loss", loss_dict)
        
        # Verify other keys still exist
        self.assertIn("LogitsKLLoss", loss_dict)
        self.assertIn("intermediate_loss", loss_dict)
    
    def test_gradient_flow(self):
        """
        Testcase10: VerifyGradient flow
        
        Expected behavior: Output tensor requires gradient and can backpropagate
        """
        balancer = LogitsAndIntermediatesLossBalancer()
        
        # Create tensors requiring gradient
        loss_dict = {
            "student_loss": torch.tensor(2.0, requires_grad=True),
            "LogitsKLLoss": torch.tensor(1.0, requires_grad=True),
            "intermediate_loss": torch.tensor(0.5, requires_grad=True)
        }
        
        result = balancer.forward(loss_dict)
        
        # Verify output requires gradient
        self.assertTrue(result.requires_grad)
        
        # Verify backpropagation is possible
        result.backward()
        
        # Verify gradients are computed
        self.assertTrue(loss_dict["LogitsKLLoss"].grad is not None)
    
    def test_multiple_intermediate_losses(self):
        """
        Testcase11: Multiple intermediate losses
        
        Input: Multiple intermediate loss terms
        Expected behavior: Correctly aggregate all non-LogitsKL losses
        """
        balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=1.0, skip_original_loss=False)
        
        loss_dict = {
            "student_loss": torch.tensor(3.0),
            "LogitsKLLoss": torch.tensor(2.0),
            "intermediate_loss_1": torch.tensor(0.5),
            "intermediate_loss_2": torch.tensor(0.3)
        }
        
        result = balancer.forward(loss_dict)
        
        # intermediate_loss = 0.5 + 0.3 = 0.8
        # dynamic_scale = 2.0 / 0.8 = 2.5
        # intermediate_loss_scaled = 0.8 * 2.5 = 2.0
        # kd_loss_scale = 1.0 * 0.5 = 0.5
        # kd_loss = (2.0 + 2.0) * 0.5 = 2.0
        # dynamic_scale2 = 3.0 / 2.0 = 1.5
        # total_loss = 3.0 + 2.0 * 1.5 = 6.0
        expected_loss = torch.tensor(6.0)
        torch.testing.assert_close(result, expected_loss, atol=1e-4, rtol=1e-4)
    
    def test_different_kd_loss_scales(self):
        """
        Testcase12: Different kd_loss_scale values
        
        Test impact of different scaling factors
        """
        scales = [0.5, 1.0, 1.5, 2.0]
        
        for scale in scales:
            with self.subTest(kd_loss_scale=scale):
                balancer = LogitsAndIntermediatesLossBalancer(
                    kd_loss_scale=scale, 
                    skip_original_loss=False
                )
                
                loss_dict = {
                    "student_loss": torch.tensor(4.0),
                    "LogitsKLLoss": torch.tensor(2.0),
                    "intermediate_loss": torch.tensor(1.0)
                }
                
                result = balancer.forward(loss_dict)
                
                # Verify valid tensor returned
                self.assertTrue(isinstance(result, torch.Tensor))
                self.assertFalse(torch.isnan(result))
                self.assertFalse(torch.isinf(result))


class TestLossBalancerEdgeCases(unittest.TestCase):
    """TestEdge cases"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_only_logits_loss(self):
        """
        Testcase13: Only logits loss, no intermediate loss
        
        Input: Only student_loss and LogitsKLLoss
        Expected behavior: Normal processing, intermediate_loss=0
        """
        balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=1.0, skip_original_loss=False)
        
        loss_dict = {
            "student_loss": torch.tensor(3.0),
            "LogitsKLLoss": torch.tensor(2.0)
        }
        
        result = balancer.forward(loss_dict)
        
        # intermediate_loss = 0
        # kd_loss = 2.0 * 1.0 = 2.0
        # dynamic_scale = 3.0 / 2.0 = 1.5
        # total_loss = 3.0 + 2.0 * 1.5 = 6.0
        expected_loss = torch.tensor(6.0)
        torch.testing.assert_close(result, expected_loss, atol=1e-4, rtol=1e-4)
    
    def test_small_values(self):
        """
        Testcase14: Very small loss values
        
        Input: Loss values close to zero
        Expected behavior: Numerical stability, no nan/inf
        """
        balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=1.0, skip_original_loss=False)
        
        loss_dict = {
            "student_loss": torch.tensor(0.001),
            "LogitsKLLoss": torch.tensor(0.001),
            "intermediate_loss": torch.tensor(0.0001)
        }
        
        result = balancer.forward(loss_dict)
        
        # Verify numerical stability
        self.assertFalse(torch.isnan(result))
        self.assertFalse(torch.isinf(result))
        self.assertTrue(result.item() > 0)
    
    def test_large_values(self):
        """
        Testcase15: Very large loss values
        
        Input: Large loss values
        Expected behavior: Numerical stability, correct handling
        """
        balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=1.0, skip_original_loss=False)
        
        loss_dict = {
            "student_loss": torch.tensor(100.0),
            "LogitsKLLoss": torch.tensor(50.0),
            "intermediate_loss": torch.tensor(25.0)
        }
        
        result = balancer.forward(loss_dict)
        
        # Verify numerical stability
        self.assertFalse(torch.isnan(result))
        self.assertFalse(torch.isinf(result))
        self.assertTrue(result.item() > 0)


if __name__ == "__main__":
    unittest.main()
