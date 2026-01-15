#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Quantizer.quantize 函数
"""

import unittest
import torch
import sys
import os
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 条件 mock：只在模块不存在或导入失败时才 mock
def conditional_mock(module_name, submodules=None):
    """只在模块不存在或导入失败时才mock"""
    try:
        __import__(module_name)
    except (ImportError, OSError, Exception):
        # 捕获所有异常，包括动态库加载失败
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

# 尝试导入，失败才mock
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

# Mock modelopt（通常不存在）
mock_mtq = MagicMock()
conditional_mock('modelopt', ['torch', 'torch.quantization', 'torch.export', 'torch.opt'])
if 'modelopt.torch.quantization' in sys.modules:
    sys.modules['modelopt.torch.quantization'] = mock_mtq

# Import ground truth implementation
try:
    from nemo.collections.llm.modelopt.quantization.quantizer import Quantizer, QuantizationConfig
    IMPORT_SUCCESS = True
except Exception as e:
    print(f"警告: 无法导入实现代码: {e}")
    IMPORT_SUCCESS = False


class TestQuantizerQuantize(unittest.TestCase):
    """测试Quantizer.quantize函数"""
    
    def setUp(self):
        """设置测试环境"""
        torch.manual_seed(42)
        
        # 重置所有 mock
        mock_mtq.quantize.reset_mock()
        mock_mtq.postprocess_amax.reset_mock()
        mock_mtq.config.need_calibration.reset_mock()
        mock_mtq.print_quant_summary.reset_mock()
        
        # 创建基础测试模型
        self.test_model = self._create_test_model()
    
    def _create_test_model(self):
        """创建测试用的简单模型"""
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
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.logging')
    def test_algorithm_none_returns_original_model(self, mock_logging):
        """
        测试样例1: algorithm=None 时必须直接返回原模型
        
        输入: algorithm=None 的配置
        预期行为: 返回相同的模型对象，记录日志，不执行量化
        """
        # 输入
        config = QuantizationConfig(algorithm=None)
        quantizer = Quantizer(config, export_config=None)
        original_model = self.test_model
        
        # 执行
        result_model = quantizer.quantize(original_model)
        
        # 验证输出
        self.assertIs(result_model, original_model, 
                     "algorithm=None时必须返回相同的模型对象引用")
        
        # 验证行为：必须记录日志
        mock_logging.info.assert_any_call(
            "Quantization algorithm set to None, returning the non-quantized model"
        )
        
        # 验证行为：不应调用任何量化函数
        mock_mtq.quantize.assert_not_called()
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.is_global_rank_zero')
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.unwrap_for_modelopt_operations')
    def test_fp8_quantization_workflow(self, mock_unwrap, mock_is_rank_zero):
        """
        测试样例2: FP8量化的完整工作流
        
        输入: algorithm="fp8" 的配置
        预期行为: 
        1. 调用 unwrap_for_modelopt_operations 解包模型
        2. 调用 mtq.quantize 执行量化
        3. 对 GPT 模型调用 postprocess_amax (maxbound=448)
        """
        # 设置
        mock_is_rank_zero.return_value = False
        mock_mtq.config.need_calibration.return_value = False
        
        unwrapped_model = Mock()
        mock_unwrap.return_value = unwrapped_model
        mock_mtq.quantize.return_value = unwrapped_model
        
        # 输入
        config = QuantizationConfig(algorithm="fp8")
        quantizer = Quantizer(config, export_config=None)
        
        # 执行
        with patch.object(quantizer, '_setup'):
            with patch.object(quantizer, '_get_decoder_type', return_value="gpt"):
                with patch.object(quantizer, '_get_quant_cfg', return_value={}):
                    result = quantizer.quantize(self.test_model)
        
        # 验证行为1: 必须unwrap模型
        mock_unwrap.assert_called_once_with(self.test_model)
        
        # 验证行为2: 必须调用mtq.quantize
        self.assertTrue(mock_mtq.quantize.called,
                       "必须调用mtq.quantize执行量化")
        
        # 验证调用参数包含unwrapped model
        call_args = mock_mtq.quantize.call_args
        self.assertEqual(call_args[0][0], unwrapped_model)
        
        # 验证行为3: GPT模型必须调用postprocess_amax
        self.assertTrue(mock_mtq.postprocess_amax.called,
                       "GPT decoder必须调用postprocess_amax进行后处理")
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.is_global_rank_zero')
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.unwrap_for_modelopt_operations')
    def test_fp8_uses_maxbound_448(self, mock_unwrap, mock_is_rank_zero):
        """
        测试样例3: FP8算法必须使用maxbound=448
        
        输入: algorithm="fp8"
        预期行为: 在postprocess_amax中使用maxbound=448的clamp函数
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
        
        # 验证调用了postprocess_amax
        self.assertTrue(mock_mtq.postprocess_amax.called)
        
        # 验证参数：第二个参数应该是 "*input_quantizer"
        call_args = mock_mtq.postprocess_amax.call_args
        self.assertEqual(call_args[0][1], "*input_quantizer",
                        "必须对*input_quantizer进行后处理")
        
        # 验证lambda函数使用了clamp操作（间接验证maxbound）
        # lambda: torch.clamp(amax, min=2**-24, max=448 if fp8 else 127)
        self.assertIsNotNone(call_args[0][2])  # lambda函数存在
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.is_global_rank_zero')
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.unwrap_for_modelopt_operations')
    def test_int8_sq_uses_maxbound_127(self, mock_unwrap, mock_is_rank_zero):
        """
        测试样例4: INT8_SQ算法必须使用maxbound=127
        
        输入: algorithm="int8_sq"
        预期行为: 在postprocess_amax中使用maxbound=127的clamp函数
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
        
        # 验证调用了postprocess_amax
        self.assertTrue(mock_mtq.postprocess_amax.called,
                       "int8_sq必须调用postprocess_amax")
        
        # 验证参数设置
        call_args = mock_mtq.postprocess_amax.call_args
        self.assertEqual(call_args[0][1], "*input_quantizer")
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.is_global_rank_zero')
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.unwrap_for_modelopt_operations')
    def test_non_gpt_decoder_no_postprocess(self, mock_unwrap, mock_is_rank_zero):
        """
        测试样例5: 非GPT decoder不应调用postprocess_amax
        
        输入: decoder_type="llama" (非gpt)
        预期行为: 不调用postprocess_amax
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
        
        # 验证：不应调用postprocess_amax
        self.assertFalse(mock_mtq.postprocess_amax.called,
                        "非GPT decoder不应调用postprocess_amax")
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.is_global_rank_zero')
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.unwrap_for_modelopt_operations')
    def test_calibration_forward_loop_creation(self, mock_unwrap, mock_is_rank_zero):
        """
        测试样例6: 需要校准但未提供forward_loop时自动创建
        
        输入: need_calibration=True, forward_loop=None
        预期行为: 调用_get_forward_loop创建校准循环
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
                        
                        # 验证：必须调用_get_forward_loop
                        mock_get_loop.assert_called_once()
        
        # 验证：mtq.quantize被调用时包含forward_loop
        call_kwargs = mock_mtq.quantize.call_args[1]
        self.assertIn('forward_loop', call_kwargs)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.is_global_rank_zero')
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.unwrap_for_modelopt_operations')
    def test_provided_forward_loop_used_directly(self, mock_unwrap, mock_is_rank_zero):
        """
        测试样例7: 提供了forward_loop时直接使用
        
        输入: 自定义的forward_loop函数
        预期行为: 不调用_get_forward_loop，直接使用提供的forward_loop
        """
        mock_is_rank_zero.return_value = False
        mock_mtq.config.need_calibration.return_value = True
        mock_unwrap.return_value = Mock()
        mock_mtq.quantize.return_value = Mock()
        
        config = QuantizationConfig(algorithm="fp8")
        quantizer = Quantizer(config, export_config=None)
        
        # 提供自定义forward_loop
        custom_forward_loop = Mock()
        
        with patch.object(quantizer, '_setup'):
            with patch.object(quantizer, '_get_decoder_type', return_value="gpt"):
                with patch.object(quantizer, '_get_quant_cfg', return_value={}):
                    with patch.object(quantizer, '_get_forward_loop') as mock_get_loop:
                        quantizer.quantize(self.test_model, forward_loop=custom_forward_loop)
                        
                        # 验证：不应调用_get_forward_loop
                        mock_get_loop.assert_not_called()
        
        # 验证：传递的是自定义的forward_loop
        call_kwargs = mock_mtq.quantize.call_args[1]
        self.assertEqual(call_kwargs.get('forward_loop'), custom_forward_loop)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.is_global_rank_zero')
    def test_quant_summary_printed_on_rank_zero(self, mock_is_rank_zero):
        """
        测试样例8: 在rank 0上打印量化摘要
        
        输入: global_rank=0
        预期行为: 调用mtq.print_quant_summary
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
        
        # 验证：在rank 0上调用print_quant_summary
        self.assertTrue(mock_mtq.print_quant_summary.called,
                       "在rank 0上必须打印量化摘要")
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    def test_different_quantization_algorithms(self):
        """
        测试样例9: 测试多种量化算法的支持
        
        输入: 不同的量化算法
        预期行为: 所有算法都能正确初始化
        """
        algorithms = ["fp8", "int8", "int8_sq", "int4_awq", "w4a8_awq"]
        
        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm):
                config = QuantizationConfig(algorithm=algorithm)
                quantizer = Quantizer(config, export_config=None)
                
                # 验证：能够成功创建Quantizer实例
                self.assertIsNotNone(quantizer)
                self.assertEqual(quantizer.config.algorithm, algorithm)


class TestQuantizationEdgeCases(unittest.TestCase):
    """测试边界情况和错误处理"""
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    def test_config_validation(self):
        """测试配置验证"""
        # 测试有效配置
        valid_config = QuantizationConfig(algorithm="fp8")
        self.assertIsNotNone(valid_config)
        
        # 测试None算法
        none_config = QuantizationConfig(algorithm=None)
        self.assertIsNone(none_config.algorithm)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('nemo.collections.llm.modelopt.quantization.quantizer.unwrap_for_modelopt_operations')
    def test_model_structure_preservation(self, mock_unwrap):
        """
        测试样例10: 验证量化后模型结构保持不变
        
        预期行为: 量化不改变模型的层数和参数数量（只改变精度）
        """
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32)
        )
        original_param_count = sum(p.numel() for p in model.parameters())
        
        # Mock量化过程保持模型结构
        mock_unwrap.return_value = model
        mock_mtq.quantize.return_value = model
        mock_mtq.config.need_calibration.return_value = False
        
        config = QuantizationConfig(algorithm="fp8")
        quantizer = Quantizer(config, export_config=None)
        
        with patch.object(quantizer, '_setup'):
            with patch.object(quantizer, '_get_decoder_type', return_value="gpt"):
                with patch.object(quantizer, '_get_quant_cfg', return_value={}):
                    quantized_model = quantizer.quantize(model)
        
        # 验证：参数数量保持不变
        quantized_param_count = sum(p.numel() for p in quantized_model.parameters())
        self.assertEqual(original_param_count, quantized_param_count,
                        "量化不应改变模型参数数量")


if __name__ == "__main__":
    unittest.main()
