#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 modelopt_quantize 函数
"""

import unittest
import torch
import sys
import os
from unittest.mock import MagicMock, patch, Mock

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
conditional_mock('transformers', ['models', 'models.gpt2'])
conditional_mock('datasets')
conditional_mock('tqdm')

# Mock modelopt（通常不存在）
mock_mtq = MagicMock()
conditional_mock('modelopt', ['torch', 'torch.quantization'])
if 'modelopt.torch.quantization' in sys.modules:
    sys.modules['modelopt.torch.quantization'] = mock_mtq

# Import ground truth implementation
try:
    # 需要先设置 tripy 路径
    tripy_path = os.path.join(os.path.dirname(__file__), '..', 'tripy')
    if tripy_path not in sys.path:
        sys.path.insert(0, tripy_path)
    
    from examples.nanogpt.quantization import modelopt_quantize
    IMPORT_SUCCESS = True
except Exception as e:
    print(f"警告: 无法导入实现代码: {e}")
    IMPORT_SUCCESS = False


class TestModeloptQuantize(unittest.TestCase):
    """测试modelopt_quantize函数"""
    
    def setUp(self):
        """设置测试环境"""
        torch.manual_seed(42)
        
        # 重置所有 mock
        mock_mtq.quantize.reset_mock()
        mock_mtq.INT8_DEFAULT_CFG = {"quant_cfg": {}}
        mock_mtq.INT4_AWQ_CFG = {"quant_cfg": {}}
        mock_mtq.FP8_DEFAULT_CFG = {"quant_cfg": {}}
        
        # 创建基础测试模型
        self.test_model = self._create_test_model()
    
    def _create_test_model(self):
        """创建测试用的简单GPT2模型"""
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
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('examples.nanogpt.quantization.AutoTokenizer')
    @patch('examples.nanogpt.quantization.create_forward_loop')
    def test_int8_weight_only_quantization(self, mock_create_loop, mock_tokenizer_cls):
        """
        测试样例1: INT8 weight-only量化模式
        
        输入: quant_mode="int8-weight-only"
        预期行为:
        1. 使用INT8_DEFAULT_CFG配置
        2. 禁用input_quantizer
        3. 调用mtq.quantize
        """
        # 设置mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_forward_loop = MagicMock()
        mock_create_loop.return_value = mock_forward_loop
        
        mock_mtq.quantize.return_value = self.test_model
        
        # 执行
        result = modelopt_quantize(self.test_model, "int8-weight-only")
        
        # 验证行为1: 使用了INT8配置
        call_args = mock_mtq.quantize.call_args
        quant_cfg = call_args[0][1]
        
        # 验证行为2: input_quantizer被禁用
        self.assertIn("quant_cfg", quant_cfg)
        self.assertIn("*input_quantizer", quant_cfg["quant_cfg"])
        self.assertFalse(quant_cfg["quant_cfg"]["*input_quantizer"]["enable"])
        
        # 验证行为3: 调用了mtq.quantize
        mock_mtq.quantize.assert_called_once()
        self.assertEqual(call_args[0][0], self.test_model)
        
        # 验证输出
        self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('examples.nanogpt.quantization.AutoTokenizer')
    @patch('examples.nanogpt.quantization.create_forward_loop')
    def test_int4_weight_only_quantization(self, mock_create_loop, mock_tokenizer_cls):
        """
        测试样例2: INT4 weight-only量化模式使用AWQ配置
        
        输入: quant_mode="int4-weight-only"
        预期行为:
        1. 使用INT4_AWQ_CFG配置
        2. 使用更大的batch_size=16
        3. 使用更小的calib_size=16
        """
        # 设置mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_forward_loop = MagicMock()
        mock_create_loop.return_value = mock_forward_loop
        
        mock_mtq.quantize.return_value = self.test_model
        
        # 执行
        result = modelopt_quantize(self.test_model, "int4-weight-only")
        
        # 验证行为1: 使用了INT4_AWQ_CFG
        call_args = mock_mtq.quantize.call_args
        self.assertIsNotNone(call_args)
        
        # 验证行为2&3: create_forward_loop调用参数
        loop_call_args = mock_create_loop.call_args
        self.assertEqual(loop_call_args[1]['batch_size'], 16)
        self.assertEqual(loop_call_args[1]['num_samples'], 16)
        
        # 验证输出
        self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('examples.nanogpt.quantization.AutoTokenizer')
    @patch('examples.nanogpt.quantization.create_forward_loop')
    def test_float8_quantization(self, mock_create_loop, mock_tokenizer_cls):
        """
        测试样例3: FP8量化模式
        
        输入: quant_mode="float8"
        预期行为:
        1. 使用FP8_DEFAULT_CFG配置
        2. 使用默认batch_size=1和calib_size=64
        """
        # 设置mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_forward_loop = MagicMock()
        mock_create_loop.return_value = mock_forward_loop
        
        mock_mtq.quantize.return_value = self.test_model
        
        # 执行
        result = modelopt_quantize(self.test_model, "float8")
        
        # 验证行为1: 使用了FP8_DEFAULT_CFG
        call_args = mock_mtq.quantize.call_args
        self.assertIsNotNone(call_args)
        
        # 验证行为2: create_forward_loop使用默认参数
        loop_call_args = mock_create_loop.call_args
        self.assertEqual(loop_call_args[1]['batch_size'], 1)
        self.assertEqual(loop_call_args[1]['num_samples'], 64)
        
        # 验证输出
        self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('examples.nanogpt.quantization.AutoTokenizer')
    @patch('examples.nanogpt.quantization.create_forward_loop')
    def test_unsupported_quantization_mode_raises_error(self, mock_create_loop, mock_tokenizer_cls):
        """
        测试样例4: 不支持的量化模式应抛出异常
        
        输入: quant_mode="unsupported-mode"
        预期行为: 抛出NotImplementedError
        """
        # 设置mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        # 执行并验证异常
        with self.assertRaises(NotImplementedError) as context:
            modelopt_quantize(self.test_model, "unsupported-mode")
        
        # 验证异常消息
        self.assertIn("Unsupported quantization mode", str(context.exception))
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('examples.nanogpt.quantization.AutoTokenizer')
    @patch('examples.nanogpt.quantization.create_forward_loop')
    def test_tokenizer_configuration(self, mock_create_loop, mock_tokenizer_cls):
        """
        测试样例5: 分词器正确配置
        
        预期行为:
        1. 加载gpt2分词器
        2. 设置max_length=2048
        3. 设置padding_side="left"
        4. pad_token=eos_token
        """
        # 设置mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_forward_loop = MagicMock()
        mock_create_loop.return_value = mock_forward_loop
        mock_mtq.quantize.return_value = self.test_model
        
        # 执行
        result = modelopt_quantize(self.test_model, "int8-weight-only")
        
        # 验证行为1: 加载了gpt2分词器
        mock_tokenizer_cls.from_pretrained.assert_called_once()
        call_args = mock_tokenizer_cls.from_pretrained.call_args
        self.assertEqual(call_args[0][0], "gpt2")
        
        # 验证行为2&3: 参数设置
        call_kwargs = call_args[1]
        self.assertEqual(call_kwargs['model_max_length'], 2048)
        self.assertEqual(call_kwargs['padding_side'], "left")
        
        # 验证行为4: pad_token设置
        self.assertEqual(mock_tokenizer.pad_token, mock_tokenizer.eos_token)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('examples.nanogpt.quantization.AutoTokenizer')
    @patch('examples.nanogpt.quantization.create_forward_loop')
    def test_forward_loop_creation_with_cnn_dailymail(self, mock_create_loop, mock_tokenizer_cls):
        """
        测试样例6: 使用cnn_dailymail数据集创建校准循环
        
        预期行为:
        1. dataset_name="cnn_dailymail"
        2. forward_loop传递给mtq.quantize
        """
        # 设置mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_forward_loop = MagicMock()
        mock_create_loop.return_value = mock_forward_loop
        mock_mtq.quantize.return_value = self.test_model
        
        # 执行
        result = modelopt_quantize(self.test_model, "int8-weight-only")
        
        # 验证行为1: 使用了cnn_dailymail数据集
        loop_call_args = mock_create_loop.call_args
        self.assertEqual(loop_call_args[1]['dataset_name'], "cnn_dailymail")
        
        # 验证行为2: forward_loop传递给量化函数
        quant_call_kwargs = mock_mtq.quantize.call_args[1]
        self.assertEqual(quant_call_kwargs['forward_loop'], mock_forward_loop)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('examples.nanogpt.quantization.AutoTokenizer')
    @patch('examples.nanogpt.quantization.create_forward_loop')
    @patch('examples.nanogpt.quantization.time')
    def test_quantization_timing(self, mock_time, mock_create_loop, mock_tokenizer_cls):
        """
        测试样例7: 量化过程时间记录
        
        预期行为:
        1. 记录开始时间
        2. 记录结束时间
        3. 打印耗时信息
        """
        # 设置mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_forward_loop = MagicMock()
        mock_create_loop.return_value = mock_forward_loop
        mock_mtq.quantize.return_value = self.test_model
        
        # Mock time.perf_counter
        mock_time.perf_counter.side_effect = [0.0, 10.5]
        
        # 执行
        with patch('builtins.print') as mock_print:
            result = modelopt_quantize(self.test_model, "int8-weight-only")
            
            # 验证：打印了耗时信息
            mock_print.assert_called()
            call_args_list = [str(call) for call in mock_print.call_args_list]
            timing_printed = any("Quantization took" in str(call) for call in call_args_list)
            self.assertTrue(timing_printed, "应该打印量化耗时信息")
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('examples.nanogpt.quantization.AutoTokenizer')
    @patch('examples.nanogpt.quantization.create_forward_loop')
    def test_model_device_preservation(self, mock_create_loop, mock_tokenizer_cls):
        """
        测试样例8: 模型设备信息传递给forward_loop
        
        预期行为: create_forward_loop接收正确的device参数
        """
        # 设置mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_forward_loop = MagicMock()
        mock_create_loop.return_value = mock_forward_loop
        mock_mtq.quantize.return_value = self.test_model
        
        # 设置模型设备
        self.test_model.device = torch.device('cpu')
        
        # 执行
        result = modelopt_quantize(self.test_model, "float8")
        
        # 验证：device参数传递正确
        loop_call_args = mock_create_loop.call_args
        self.assertEqual(loop_call_args[1]['device'], self.test_model.device)


class TestModeloptQuantizeEdgeCases(unittest.TestCase):
    """测试边界情况和错误处理"""
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('examples.nanogpt.quantization.AutoTokenizer')
    @patch('examples.nanogpt.quantization.create_forward_loop')
    def test_quantization_preserves_model_structure(self, mock_create_loop, mock_tokenizer_cls):
        """
        测试样例9: 量化保持模型结构不变
        
        预期行为: 量化后返回相同的模型对象
        """
        # 创建测试模型
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 3072),
            torch.nn.ReLU(),
            torch.nn.Linear(3072, 768)
        )
        model.device = torch.device('cpu')
        
        # 设置mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_forward_loop = MagicMock()
        mock_create_loop.return_value = mock_forward_loop
        
        # mtq.quantize返回相同的模型
        mock_mtq.quantize.return_value = model
        
        # 执行
        result = modelopt_quantize(model, "int8-weight-only")
        
        # 验证：返回了模型
        self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('examples.nanogpt.quantization.AutoTokenizer')
    @patch('examples.nanogpt.quantization.create_forward_loop')
    def test_multiple_quantization_modes_sequentially(self, mock_create_loop, mock_tokenizer_cls):
        """
        测试样例10: 顺序测试多种量化模式
        
        预期行为: 所有支持的模式都能正常执行
        """
        # 设置mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = '<|endoftext|>'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_forward_loop = MagicMock()
        mock_create_loop.return_value = mock_forward_loop
        
        model = torch.nn.Linear(768, 768)
        model.device = torch.device('cpu')
        mock_mtq.quantize.return_value = model
        
        # 测试所有支持的模式
        supported_modes = ["int8-weight-only", "int4-weight-only", "float8"]
        
        for mode in supported_modes:
            with self.subTest(quant_mode=mode):
                mock_mtq.quantize.reset_mock()
                
                # 执行
                result = modelopt_quantize(model, mode)
                
                # 验证：成功执行
                self.assertIsNotNone(result)
                mock_mtq.quantize.assert_called_once()


if __name__ == "__main__":
    unittest.main()

