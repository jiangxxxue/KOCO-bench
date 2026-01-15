#!/usr/bin/env python3
# Copyright 2024 ByteMLPerf. All rights reserved.

"""
测试 LogitsAndIntermediatesLossBalancer.forward 函数
"""

import torch
import torch.nn as nn
import unittest
from unittest.mock import Mock, patch

# Import the actual implementation
import sys
import os
# 使用 insert(0) 而不是 append，避免被已安装的 megatron 包覆盖
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'byte_train_perf', 'Megatron-LM'))

# 确保能正确导入真实实现
from megatron.inference.algos.distillation import LogitsAndIntermediatesLossBalancer
import modelopt.torch.distill as mtd
from modelopt.torch.distill import DistillationLossBalancer


class TestLogitsAndIntermediatesLossBalancer(unittest.TestCase):
    """测试LogitsAndIntermediatesLossBalancer.forward函数"""
    
    def setUp(self):
        """设置测试环境"""
        torch.manual_seed(42)
    
    def test_init_default_params(self):
        """
        测试样例1: 默认参数初始化
        
        输入: 无参数
        预期行为: kd_loss_scale=1.0, skip_original_loss=False
        """
        balancer = LogitsAndIntermediatesLossBalancer()
        
        self.assertEqual(balancer._kd_loss_scale, 1.0)
        self.assertEqual(balancer._skip_original_loss, False)
    
    def test_init_custom_params(self):
        """
        测试样例2: 自定义参数初始化
        
        输入: kd_loss_scale=2.0, skip_original_loss=True
        预期行为: 参数正确设置
        """
        balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=2.0, skip_original_loss=True)
        
        self.assertEqual(balancer._kd_loss_scale, 2.0)
        self.assertEqual(balancer._skip_original_loss, True)
    
    def test_forward_with_intermediate_loss(self):
        """
        测试样例3: 带中间层损失的前向传播
        
        输入:
        - student_loss: 2.0
        - LogitsKLLoss: 1.0
        - intermediate_loss: 0.5
        - kd_loss_scale: 1.0
        - skip_original_loss: False
        
        预期行为: 
        1. 移除student_loss
        2. 计算动态缩放
        3. 返回平衡后的总损失
        """
        balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=1.0, skip_original_loss=False)
        
        loss_dict = {
            "student_loss": torch.tensor(2.0),
            "LogitsKLLoss": torch.tensor(1.0),
            "intermediate_loss": torch.tensor(0.5)
        }
        
        result = balancer.forward(loss_dict)
        
        # 验证返回tensor
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.dim(), 0)
        
        # 验证具体数值：
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
        测试样例4: 无中间层损失的前向传播
        
        输入:
        - student_loss: 2.0
        - LogitsKLLoss: 1.0
        - kd_loss_scale: 1.0
        - skip_original_loss: False
        
        预期行为: 只使用logits损失，不进行中间层缩放
        """
        balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=1.0, skip_original_loss=False)
        
        loss_dict = {
            "student_loss": torch.tensor(2.0),
            "LogitsKLLoss": torch.tensor(1.0)
        }
        
        result = balancer.forward(loss_dict)
        
        # 验证：intermediate_loss = 0，不进行缩放
        # kd_loss = 1.0 * 1.0 = 1.0
        # dynamic_scale = 2.0 / 1.0 = 2.0
        # total_loss = 2.0 + 1.0 * 2.0 = 4.0
        expected_loss = torch.tensor(4.0)
        torch.testing.assert_close(result, expected_loss, atol=1e-4, rtol=1e-4)
    
    def test_forward_skip_original_loss(self):
        """
        测试样例5: 跳过原始损失
        
        输入:
        - student_loss: 2.0
        - LogitsKLLoss: 1.0
        - intermediate_loss: 0.5
        - skip_original_loss: True
        
        预期行为: 只返回蒸馏损失，不包含原始student_loss
        """
        balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=1.0, skip_original_loss=True)
        
        loss_dict = {
            "student_loss": torch.tensor(2.0),
            "LogitsKLLoss": torch.tensor(1.0),
            "intermediate_loss": torch.tensor(0.5)
        }
        
        result = balancer.forward(loss_dict)
        
        # 验证：skip_original_loss=True
        # logits_loss = 1.0, intermediate_loss = 0.5
        # dynamic_scale = 1.0 / 0.5 = 2.0
        # intermediate_loss_scaled = 0.5 * 2.0 = 1.0
        # total_loss = 1.0 + 1.0 = 2.0 (不包含original_loss)
        # 但需要乘以 kd_loss_scale (0.5 after halving)
        # 实际: logits_loss + intermediate_loss_scaled = 1.0 + 1.0 = 2.0
        expected_loss = torch.tensor(2.0)
        torch.testing.assert_close(result, expected_loss, atol=1e-4, rtol=1e-4)
    
    def test_forward_dynamic_scaling(self):
        """
        测试样例6: 动态缩放计算
        
        输入:
        - student_loss: 3.0
        - LogitsKLLoss: 2.0
        - intermediate_loss: 1.0
        - kd_loss_scale: 2.0
        
        预期行为: 验证动态缩放的正确性
        """
        balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=2.0, skip_original_loss=False)
        
        loss_dict = {
            "student_loss": torch.tensor(3.0),
            "LogitsKLLoss": torch.tensor(2.0),
            "intermediate_loss": torch.tensor(1.0)
        }
        
        result = balancer.forward(loss_dict)
        
        # 验证动态缩放：
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
        测试样例7: 具体数值验证
        
        使用具体数值验证计算的每个步骤
        """
        balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=1.5, skip_original_loss=False)
        
        loss_dict = {
            "student_loss": torch.tensor(5.0),
            "LogitsKLLoss": torch.tensor(3.0),
            "intermediate_loss": torch.tensor(2.0)
        }
        
        result = balancer.forward(loss_dict)
        
        # 手动计算:
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
        测试样例8: 中间层损失为零
        
        输入: intermediate_loss = 0.0
        预期行为: 不进行动态缩放，使用原始kd_loss_scale
        """
        balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=1.0, skip_original_loss=False)
        
        loss_dict = {
            "student_loss": torch.tensor(2.0),
            "LogitsKLLoss": torch.tensor(1.0),
            "intermediate_loss": torch.tensor(0.0)
        }
        
        result = balancer.forward(loss_dict)
        
        # intermediate_loss = 0，不缩放
        # kd_loss = 1.0 * 1.0 = 1.0
        # dynamic_scale = 2.0 / 1.0 = 2.0
        # total_loss = 2.0 + 1.0 * 2.0 = 4.0
        expected_loss = torch.tensor(4.0)
        torch.testing.assert_close(result, expected_loss, atol=1e-4, rtol=1e-4)
    
    def test_loss_dict_modification(self):
        """
        测试样例9: 验证loss_dict被正确修改
        
        预期行为: student_loss从字典中移除
        """
        balancer = LogitsAndIntermediatesLossBalancer()
        
        loss_dict = {
            "student_loss": torch.tensor(2.0),
            "LogitsKLLoss": torch.tensor(1.0),
            "intermediate_loss": torch.tensor(0.5)
        }
        
        # 保存原始键
        original_keys = set(loss_dict.keys())
        
        result = balancer.forward(loss_dict)
        
        # 验证student_loss被移除
        self.assertNotIn("student_loss", loss_dict)
        
        # 验证其他键仍然存在
        self.assertIn("LogitsKLLoss", loss_dict)
        self.assertIn("intermediate_loss", loss_dict)
    
    def test_gradient_flow(self):
        """
        测试样例10: 验证梯度流动
        
        预期行为: 输出tensor需要梯度且可以反向传播
        """
        balancer = LogitsAndIntermediatesLossBalancer()
        
        # 创建需要梯度的tensor
        loss_dict = {
            "student_loss": torch.tensor(2.0, requires_grad=True),
            "LogitsKLLoss": torch.tensor(1.0, requires_grad=True),
            "intermediate_loss": torch.tensor(0.5, requires_grad=True)
        }
        
        result = balancer.forward(loss_dict)
        
        # 验证输出需要梯度
        self.assertTrue(result.requires_grad)
        
        # 验证可以反向传播
        result.backward()
        
        # 验证梯度已计算
        self.assertTrue(loss_dict["LogitsKLLoss"].grad is not None)
    
    def test_multiple_intermediate_losses(self):
        """
        测试样例11: 多个中间层损失
        
        输入: 多个中间层损失项
        预期行为: 正确聚合所有非LogitsKL的损失
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
        测试样例12: 不同的kd_loss_scale值
        
        测试不同缩放因子的影响
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
                
                # 验证返回有效tensor
                self.assertTrue(isinstance(result, torch.Tensor))
                self.assertFalse(torch.isnan(result))
                self.assertFalse(torch.isinf(result))


class TestLossBalancerEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_only_logits_loss(self):
        """
        测试样例13: 只有logits损失，无中间层损失
        
        输入: 只有student_loss和LogitsKLLoss
        预期行为: 正常处理，intermediate_loss=0
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
        测试样例14: 很小的损失值
        
        输入: 接近零的损失值
        预期行为: 数值稳定，无nan/inf
        """
        balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=1.0, skip_original_loss=False)
        
        loss_dict = {
            "student_loss": torch.tensor(0.001),
            "LogitsKLLoss": torch.tensor(0.001),
            "intermediate_loss": torch.tensor(0.0001)
        }
        
        result = balancer.forward(loss_dict)
        
        # 验证数值稳定
        self.assertFalse(torch.isnan(result))
        self.assertFalse(torch.isinf(result))
        self.assertTrue(result.item() > 0)
    
    def test_large_values(self):
        """
        测试样例15: 很大的损失值
        
        输入: 较大的损失值
        预期行为: 数值稳定，正确处理
        """
        balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=1.0, skip_original_loss=False)
        
        loss_dict = {
            "student_loss": torch.tensor(100.0),
            "LogitsKLLoss": torch.tensor(50.0),
            "intermediate_loss": torch.tensor(25.0)
        }
        
        result = balancer.forward(loss_dict)
        
        # 验证数值稳定
        self.assertFalse(torch.isnan(result))
        self.assertFalse(torch.isinf(result))
        self.assertTrue(result.item() > 0)


if __name__ == "__main__":
    unittest.main()
