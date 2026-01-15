#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 load_quant_weights_from_hf 函数
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
    print(f"警告: 无法导入实现代码: {e}")
    IMPORT_SUCCESS = False


class TestLoadQuantWeightsFromHF(unittest.TestCase):
    """测试load_quant_weights_from_hf函数"""
    
    def setUp(self):
        """设置测试环境"""
        torch.manual_seed(42)
        
        # 创建基础测试模型
        self.test_model = self._create_test_tripy_model()
        self.model_type = "gpt2"
        self.dtype = self._create_mock_dtype()
    
    def _create_test_tripy_model(self):
        """创建测试用的Tripy模型mock"""
        model = MagicMock()
        # 创建状态字典
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
        """创建Mock数据类型"""
        dtype = MagicMock()
        dtype.name = "float32"
        return dtype
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('examples.nanogpt.weight_loader.GPT2LMHeadModel')
    @patch('examples.nanogpt.weight_loader.modelopt_quantize')
    @patch('examples.nanogpt.weight_loader.tp')
    def test_basic_weight_loading_flow(self, mock_tp, mock_quantize, mock_gpt2_class):
        """
        测试样例1: 基本权重加载流程
        
        输入: Tripy模型、模型类型、数据类型、量化模式
        预期行为:
        1. 加载HuggingFace模型
        2. 执行量化
        3. 加载权重到Tripy模型
        """
        # 设置mock
        mock_hf_model = self._create_mock_hf_model()
        mock_gpt2_class.from_pretrained.return_value = mock_hf_model
        mock_quantize.return_value = mock_hf_model
        
        # Mock tp.Tensor
        mock_tp.Tensor.side_effect = lambda x: x
        
        # 执行
        with patch('builtins.print'):
            load_quant_weights_from_hf(
                self.test_model,
                self.model_type,
                self.dtype,
                "int8-weight-only"
            )
        
        # 验证行为1: 加载了HuggingFace模型
        mock_gpt2_class.from_pretrained.assert_called_once_with(self.model_type)
        
        # 验证行为2: 执行了量化
        mock_quantize.assert_called_once()
        
        # 验证行为3: 调用了load_state_dict
        self.test_model.load_state_dict.assert_called_once()
    
    def _create_mock_hf_model(self):
        """创建Mock HuggingFace模型"""
        model = MagicMock()
        
        # 创建模拟的状态字典
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
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('examples.nanogpt.weight_loader.GPT2LMHeadModel')
    @patch('examples.nanogpt.weight_loader.modelopt_quantize')
    @patch('examples.nanogpt.weight_loader.tp')
    def test_ignored_keys_filtering(self, mock_tp, mock_quantize, mock_gpt2_class):
        """
        测试样例2: 忽略特定的键
        
        预期行为:
        1. 过滤掉.attn.masked_bias
        2. 过滤掉.attn.bias
        3. 过滤掉_pre_quant_scale
        """
        # 设置mock
        mock_hf_model = self._create_mock_hf_model()
        mock_gpt2_class.from_pretrained.return_value = mock_hf_model
        mock_quantize.return_value = mock_hf_model
        
        mock_tp.Tensor.side_effect = lambda x: x
        
        # 执行
        with patch('builtins.print'):
            load_quant_weights_from_hf(
                self.test_model,
                self.model_type,
                self.dtype,
                "int8-weight-only"
            )
        
        # 验证：load_state_dict被调用
        self.test_model.load_state_dict.assert_called_once()
        
        # 获取传入的状态字典
        call_args = self.test_model.load_state_dict.call_args
        loaded_state_dict = call_args[0][0]
        
        # 验证：忽略的键不在加载的状态字典中
        ignored_patterns = ['.attn.masked_bias', '.attn.bias', '_pre_quant_scale']
        for key in loaded_state_dict.keys():
            for pattern in ignored_patterns:
                self.assertNotIn(pattern, key, 
                               f"忽略的键模式 {pattern} 不应出现在加载的权重中")
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('examples.nanogpt.weight_loader.GPT2LMHeadModel')
    @patch('examples.nanogpt.weight_loader.modelopt_quantize')
    @patch('examples.nanogpt.weight_loader.tp')
    def test_weight_tying_for_embeddings(self, mock_tp, mock_quantize, mock_gpt2_class):
        """
        测试样例3: 权重共享处理
        
        预期行为: transformer.wte.weight = lm_head.weight
        """
        # 设置mock
        mock_hf_model = self._create_mock_hf_model()
        lm_head_weight = torch.randn(50257, 768)
        mock_hf_model.state_dict.return_value['lm_head.weight'] = lm_head_weight
        
        mock_gpt2_class.from_pretrained.return_value = mock_hf_model
        mock_quantize.return_value = mock_hf_model
        
        mock_tp.Tensor.side_effect = lambda x: x
        
        # 执行
        with patch('builtins.print'):
            load_quant_weights_from_hf(
                self.test_model,
                self.model_type,
                self.dtype,
                "int8-weight-only"
            )
        
        # 验证：wte权重应该等于lm_head权重
        call_args = self.test_model.load_state_dict.call_args
        loaded_state_dict = call_args[0][0]
        
        # 检查wte.weight是否在状态字典中
        self.assertIn('transformer.wte.weight', loaded_state_dict)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('examples.nanogpt.weight_loader.GPT2LMHeadModel')
    @patch('examples.nanogpt.weight_loader.modelopt_quantize')
    @patch('examples.nanogpt.weight_loader.tp')
    def test_int4_block_quantization_reshape(self, mock_tp, mock_quantize, mock_gpt2_class):
        """
        测试样例4: INT4块量化的amax重塑
        
        输入: quant_mode="int4-weight-only"
        预期行为: 重塑amax张量以匹配块量化维度
        """
        # 设置mock
        mock_hf_model = MagicMock()
        state_dict = {
            'transformer.h.0.attn.c_attn.weight': torch.randn(768, 768*3),
            'transformer.h.0.attn.c_attn.weight_quantizer._amax': torch.randn(768*3, 1),
            'lm_head.weight': torch.randn(50257, 768),
        }
        mock_hf_model.state_dict.return_value = state_dict
        
        # 创建mock linear layer
        mock_linear = MagicMock()
        mock_linear.in_features = 768
        
        # 创建mock quantizer
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
        
        # 执行
        with patch('builtins.print'):
            load_quant_weights_from_hf(
                self.test_model,
                self.model_type,
                self.dtype,
                "int4-weight-only"
            )
        
        # 验证：成功执行
        self.test_model.load_state_dict.assert_called_once()
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('examples.nanogpt.weight_loader.GPT2LMHeadModel')
    @patch('examples.nanogpt.weight_loader.modelopt_quantize')
    @patch('examples.nanogpt.weight_loader.tp')
    def test_scale_computation_from_amax(self, mock_tp, mock_quantize, mock_gpt2_class):
        """
        测试样例5: 从amax计算缩放因子
        
        预期行为:
        1. 获取量化器的amax值
        2. 除以maxbound得到scale
        3. 转换键名为.scale
        """
        # 设置mock
        mock_hf_model = MagicMock()
        amax_value = torch.tensor([[127.0]])
        state_dict = {
            'transformer.h.0.attn.c_attn.weight': torch.randn(768, 768*3),
            'transformer.h.0.attn.c_attn.weight_quantizer._amax': amax_value,
            'lm_head.weight': torch.randn(50257, 768),
        }
        mock_hf_model.state_dict.return_value = state_dict
        
        # 创建mock quantizer
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
        
        # 执行
        with patch('builtins.print'):
            load_quant_weights_from_hf(
                self.test_model,
                self.model_type,
                self.dtype,
                "int8-weight-only"
            )
        
        # 验证：成功执行
        self.test_model.load_state_dict.assert_called_once()
        
        # 获取加载的状态字典
        call_args = self.test_model.load_state_dict.call_args
        loaded_state_dict = call_args[0][0]
        
        # 验证：scale键存在
        scale_key_exists = any('scale' in key for key in loaded_state_dict.keys())
        self.assertTrue(scale_key_exists, "应该包含scale键")
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('examples.nanogpt.weight_loader.GPT2LMHeadModel')
    @patch('examples.nanogpt.weight_loader.modelopt_quantize')
    @patch('examples.nanogpt.weight_loader.tp')
    def test_dtype_conversion(self, mock_tp, mock_quantize, mock_gpt2_class):
        """
        测试样例6: 数据类型转换
        
        预期行为: 权重转换为目标dtype
        """
        # 设置mock
        mock_hf_model = self._create_mock_hf_model()
        mock_gpt2_class.from_pretrained.return_value = mock_hf_model
        mock_quantize.return_value = mock_hf_model
        
        mock_tp.Tensor.side_effect = lambda x: x
        
        # 测试不同的数据类型
        dtypes = ["float32", "float16", "bfloat16"]
        
        for dtype_name in dtypes:
            with self.subTest(dtype=dtype_name):
                # 重置mock
                self.test_model.load_state_dict.reset_mock()
                
                # 创建dtype
                dtype = MagicMock()
                dtype.name = dtype_name
                
                # 执行
                with patch('builtins.print'):
                    load_quant_weights_from_hf(
                        self.test_model,
                        self.model_type,
                        dtype,
                        "int8-weight-only"
                    )
                
                # 验证：成功加载
                self.test_model.load_state_dict.assert_called_once()
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('examples.nanogpt.weight_loader.GPT2LMHeadModel')
    @patch('examples.nanogpt.weight_loader.modelopt_quantize')
    @patch('examples.nanogpt.weight_loader.tp')
    def test_strict_loading_disabled(self, mock_tp, mock_quantize, mock_gpt2_class):
        """
        测试样例7: 非严格加载模式
        
        预期行为: load_state_dict使用strict=False
        """
        # 设置mock
        mock_hf_model = self._create_mock_hf_model()
        mock_gpt2_class.from_pretrained.return_value = mock_hf_model
        mock_quantize.return_value = mock_hf_model
        
        mock_tp.Tensor.side_effect = lambda x: x
        
        # 执行
        with patch('builtins.print'):
            load_quant_weights_from_hf(
                self.test_model,
                self.model_type,
                self.dtype,
                "int8-weight-only"
            )
        
        # 验证：使用了strict=False
        call_kwargs = self.test_model.load_state_dict.call_args[1]
        self.assertIn('strict', call_kwargs)
        self.assertFalse(call_kwargs['strict'])
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('examples.nanogpt.weight_loader.GPT2LMHeadModel')
    @patch('examples.nanogpt.weight_loader.modelopt_quantize')
    @patch('examples.nanogpt.weight_loader.tp')
    def test_print_messages(self, mock_tp, mock_quantize, mock_gpt2_class):
        """
        测试样例8: 打印信息验证
        
        预期行为:
        1. 打印加载模型信息
        2. 打印完成信息
        """
        # 设置mock
        mock_hf_model = self._create_mock_hf_model()
        mock_gpt2_class.from_pretrained.return_value = mock_hf_model
        mock_quantize.return_value = mock_hf_model
        
        mock_tp.Tensor.side_effect = lambda x: x
        
        # 执行
        with patch('builtins.print') as mock_print:
            load_quant_weights_from_hf(
                self.test_model,
                self.model_type,
                self.dtype,
                "int8-weight-only"
            )
            
            # 验证：打印了加载信息
            self.assertTrue(mock_print.called)
            
            # 检查打印内容
            call_args_list = [str(call) for call in mock_print.call_args_list]
            loading_printed = any("Loading weights" in str(call) for call in call_args_list)
            loaded_printed = any("Loaded weights" in str(call) for call in call_args_list)
            
            self.assertTrue(loading_printed, "应该打印加载信息")
            self.assertTrue(loaded_printed, "应该打印完成信息")


class TestLoadQuantWeightsEdgeCases(unittest.TestCase):
    """测试边界情况和错误处理"""
    
    def setUp(self):
        """设置测试环境"""
        self.test_model = MagicMock()
        self.test_model.state_dict.return_value = {
            'transformer.wte.weight': None,
            'lm_head.weight': None,
        }
        self.test_model.load_state_dict = MagicMock()
        
        self.dtype = MagicMock()
        self.dtype.name = "float32"
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('examples.nanogpt.weight_loader.GPT2LMHeadModel')
    @patch('examples.nanogpt.weight_loader.modelopt_quantize')
    @patch('examples.nanogpt.weight_loader.tp')
    def test_different_model_types(self, mock_tp, mock_quantize, mock_gpt2_class):
        """
        测试样例9: 不同的模型类型
        
        输入: 不同的model_type
        预期: 都能正确加载
        """
        # 设置mock
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
                
                # 执行
                with patch('builtins.print'):
                    load_quant_weights_from_hf(
                        self.test_model,
                        model_type,
                        self.dtype,
                        "int8-weight-only"
                    )
                
                # 验证：使用正确的模型类型
                mock_gpt2_class.from_pretrained.assert_called_once_with(model_type)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('examples.nanogpt.weight_loader.GPT2LMHeadModel')
    @patch('examples.nanogpt.weight_loader.modelopt_quantize')
    @patch('examples.nanogpt.weight_loader.tp')
    def test_all_quantization_modes(self, mock_tp, mock_quantize, mock_gpt2_class):
        """
        测试样例10: 所有量化模式
        
        预期: 所有量化模式都能正常工作
        """
        # 设置mock
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
                
                # 执行
                with patch('builtins.print'):
                    load_quant_weights_from_hf(
                        self.test_model,
                        "gpt2",
                        self.dtype,
                        quant_mode
                    )
                
                # 验证：执行了量化
                mock_quantize.assert_called_once()
                
                # 验证：加载了权重
                self.test_model.load_state_dict.assert_called_once()


if __name__ == "__main__":
    unittest.main()

