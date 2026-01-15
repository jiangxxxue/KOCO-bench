#!/usr/bin/env python3
# Copyright 2024 ByteMLPerf. All rights reserved.

"""
测试 _teacher_provider 函数
"""

import torch
import torch.nn as nn
import unittest
from unittest.mock import Mock, MagicMock, patch
from argparse import Namespace
import sys
import os

# Mock transformer_engine before any imports
sys.modules['transformer_engine'] = MagicMock()
sys.modules['transformer_engine.pytorch'] = MagicMock()

# Add the parent directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'byte_train_perf', 'Megatron-LM'))

# Import ground truth implementation
from megatron.inference.gpt.model_provider import _teacher_provider


class TestTeacherProvider(unittest.TestCase):
    """测试_teacher_provider函数"""
    
    def setUp(self):
        """设置测试环境"""
        torch.manual_seed(42)
        
        # 创建mock args
        self.mock_args = Mock()
        self.mock_args.finetune = False
        self.mock_args.export_kd_teacher_load = "/fake/path/to/teacher"
        self.mock_args.hidden_size = 768
        self.mock_args.num_layers = 12
        self.mock_args.num_attention_heads = 12
        self.mock_args.ffn_hidden_size = 3072
        self.mock_args.max_position_embeddings = 2048
        self.mock_args.padded_vocab_size = 50257
        self.mock_args.tensor_model_parallel_size = 1
        self.mock_args.pipeline_model_parallel_size = 1
        self.mock_args.sequence_parallel = False
        self.mock_args.use_cpu_initialization = False
        self.mock_args.params_dtype = torch.float32
        self.mock_args.perform_initialization = True
    
    @patch('megatron.inference.gpt.model_provider.load_modelopt_checkpoint')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.get_args')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    def test_teacher_creation_with_config(self, mock_model_class, mock_get_args, 
                                         mock_config_from_args, mock_add_hooks, 
                                         mock_print, mock_load_ckpt):
        """
        测试样例1: 使用给定配置创建教师模型
        
        输入:
        - config: Namespace对象
        - model_kwargs: 模型参数
        
        预期行为:
        1. 调用core_transformer_config_from_args转换配置
        2. 设置non_homogeneous_layers=True
        3. 创建MCoreGPTModel
        4. 添加load convert hooks
        5. 加载checkpoint
        """
        mock_get_args.return_value = self.mock_args
        
        # Mock TransformerConfig
        mock_transformer_config = Mock()
        mock_transformer_config.hidden_size = 768
        mock_transformer_config.num_layers = 12
        mock_config_from_args.return_value = mock_transformer_config
        
        # Mock model instance
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # 创建输入
        config = Namespace(
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12
        )
        model_kwargs = {
            'pre_process': True,
            'post_process': True
        }
        
        # 执行
        result = _teacher_provider(config, model_kwargs)
        
        # 验证: core_transformer_config_from_args被调用
        mock_config_from_args.assert_called_once_with(config)
        
        # 验证: non_homogeneous_layers被设置为True
        self.assertTrue(mock_transformer_config.non_homogeneous_layers)
        
        # 验证: MCoreGPTModel被创建，传入正确参数
        mock_model_class.assert_called_once()
        call_kwargs = mock_model_class.call_args[1]
        self.assertEqual(call_kwargs['config'], mock_transformer_config)
        self.assertEqual(call_kwargs['pre_process'], True)
        self.assertEqual(call_kwargs['post_process'], True)
        
        # 验证: add_load_convert_hooks被调用
        mock_add_hooks.assert_called_once_with(mock_model)
        
        # 验证: print_rank_0被调用
        mock_print.assert_called()
        
        # 验证: load_modelopt_checkpoint被调用
        mock_load_ckpt.assert_called_once()
        load_args = mock_load_ckpt.call_args
        self.assertEqual(load_args[0][0][0], mock_model)  # 第一个参数是[teacher]
        self.assertEqual(load_args[1]['load_arg'], 'export_kd_teacher_load')
        
        # 验证: 返回教师模型
        self.assertEqual(result, mock_model)
    
    @patch('megatron.inference.gpt.model_provider.load_modelopt_checkpoint')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.get_args')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    def test_finetune_flag_temporarily_set(self, mock_model_class, mock_get_args,
                                          mock_config_from_args, mock_add_hooks,
                                          mock_print, mock_load_ckpt):
        """
        测试样例2: finetune标志临时设置为True
        
        输入: args.finetune = False
        预期行为:
        1. 在加载checkpoint时临时设置finetune=True
        2. 加载后恢复原始值
        """
        self.mock_args.finetune = False
        original_finetune = self.mock_args.finetune
        mock_get_args.return_value = self.mock_args
        
        mock_transformer_config = Mock()
        mock_config_from_args.return_value = mock_transformer_config
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # 追踪finetune的变化
        finetune_during_load = []
        
        def track_finetune(*args, **kwargs):
            finetune_during_load.append(self.mock_args.finetune)
        
        mock_load_ckpt.side_effect = track_finetune
        
        config = Namespace(hidden_size=768)
        model_kwargs = {}
        
        # 执行
        result = _teacher_provider(config, model_kwargs)
        
        # 验证: 加载时finetune被设置为True
        self.assertTrue(len(finetune_during_load) > 0)
        self.assertTrue(finetune_during_load[0])  # 加载时应该是True
        
        # 验证: 加载后恢复原始值
        self.assertEqual(self.mock_args.finetune, original_finetune)
    
    @patch('megatron.inference.gpt.model_provider.load_modelopt_checkpoint')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.get_args')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    def test_non_homogeneous_layers_always_true(self, mock_model_class, mock_get_args,
                                                mock_config_from_args, mock_add_hooks,
                                                mock_print, mock_load_ckpt):
        """
        测试样例3: non_homogeneous_layers总是设置为True
        
        验证配置的non_homogeneous_layers属性被正确设置
        """
        mock_get_args.return_value = self.mock_args
        
        mock_transformer_config = Mock()
        mock_transformer_config.non_homogeneous_layers = False  # 初始为False
        mock_config_from_args.return_value = mock_transformer_config
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        config = Namespace(hidden_size=768)
        model_kwargs = {}
        
        # 执行
        result = _teacher_provider(config, model_kwargs)
        
        # 验证: non_homogeneous_layers被设置为True
        self.assertTrue(mock_transformer_config.non_homogeneous_layers)
    
    @patch('megatron.inference.gpt.model_provider.load_modelopt_checkpoint')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.get_args')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    def test_model_kwargs_passed_correctly(self, mock_model_class, mock_get_args,
                                          mock_config_from_args, mock_add_hooks,
                                          mock_print, mock_load_ckpt):
        """
        测试样例4: model_kwargs正确传递给MCoreGPTModel
        
        测试不同的model_kwargs组合
        """
        mock_get_args.return_value = self.mock_args
        
        mock_transformer_config = Mock()
        mock_config_from_args.return_value = mock_transformer_config
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # 测试不同的kwargs组合
        test_cases = [
            {'pre_process': True, 'post_process': True},
            {'pre_process': False, 'post_process': True},
            {'pre_process': True, 'post_process': False, 'parallel_output': False},
        ]
        
        for model_kwargs in test_cases:
            with self.subTest(model_kwargs=model_kwargs):
                mock_model_class.reset_mock()
                
                config = Namespace(hidden_size=768)
                
                result = _teacher_provider(config, model_kwargs)
                
                # 验证kwargs被正确传递
                call_kwargs = mock_model_class.call_args[1]
                for key, value in model_kwargs.items():
                    self.assertEqual(call_kwargs[key], value)
    
    @patch('megatron.inference.gpt.model_provider.load_modelopt_checkpoint')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.get_args')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    def test_print_loading_message(self, mock_model_class, mock_get_args,
                                   mock_config_from_args, mock_add_hooks,
                                   mock_print, mock_load_ckpt):
        """
        测试样例5: 打印加载教师checkpoint的消息
        
        验证print_rank_0被调用，输出加载信息
        """
        mock_get_args.return_value = self.mock_args
        
        mock_transformer_config = Mock()
        mock_config_from_args.return_value = mock_transformer_config
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        config = Namespace(hidden_size=768)
        model_kwargs = {}
        
        # 执行
        result = _teacher_provider(config, model_kwargs)
        
        # 验证print_rank_0被调用
        mock_print.assert_called()
        
        # 验证消息内容
        call_args = [str(call[0][0]) for call in mock_print.call_args_list]
        self.assertTrue(any('teacher' in arg.lower() and 'checkpoint' in arg.lower() 
                           for arg in call_args))
    
    @patch('megatron.inference.gpt.model_provider.load_modelopt_checkpoint')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.get_args')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    def test_hooks_added_before_loading(self, mock_model_class, mock_get_args,
                                       mock_config_from_args, mock_add_hooks,
                                       mock_print, mock_load_ckpt):
        """
        测试样例6: hooks在加载checkpoint之前添加
        
        验证_add_load_convert_hooks在load_modelopt_checkpoint之前被调用
        """
        mock_get_args.return_value = self.mock_args
        
        mock_transformer_config = Mock()
        mock_config_from_args.return_value = mock_transformer_config
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # 追踪调用顺序
        call_order = []
        
        def track_add_hooks(model):
            call_order.append('add_hooks')
        
        def track_load_ckpt(*args, **kwargs):
            call_order.append('load_ckpt')
        
        mock_add_hooks.side_effect = track_add_hooks
        mock_load_ckpt.side_effect = track_load_ckpt
        
        config = Namespace(hidden_size=768)
        model_kwargs = {}
        
        # 执行
        result = _teacher_provider(config, model_kwargs)
        
        # 验证顺序: add_hooks -> load_ckpt
        self.assertEqual(call_order, ['add_hooks', 'load_ckpt'])


class TestTeacherProviderEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def setUp(self):
        torch.manual_seed(42)
        
        self.mock_args = Mock()
        self.mock_args.finetune = False
        self.mock_args.export_kd_teacher_load = "/fake/path/to/teacher"
    
    @patch('megatron.inference.gpt.model_provider.load_modelopt_checkpoint')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.get_args')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    def test_finetune_already_true(self, mock_model_class, mock_get_args,
                                   mock_config_from_args, mock_add_hooks,
                                   mock_print, mock_load_ckpt):
        """
        测试样例7: finetune已经是True的情况
        
        验证原始值正确恢复
        """
        self.mock_args.finetune = True  # 已经是True
        mock_get_args.return_value = self.mock_args
        
        mock_transformer_config = Mock()
        mock_config_from_args.return_value = mock_transformer_config
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        config = Namespace(hidden_size=768)
        model_kwargs = {}
        
        # 执行
        result = _teacher_provider(config, model_kwargs)
        
        # 验证: 加载后finetune仍然是True
        self.assertTrue(self.mock_args.finetune)
    
    @patch('megatron.inference.gpt.model_provider.load_modelopt_checkpoint')
    @patch('megatron.inference.gpt.model_provider.print_rank_0')
    @patch('megatron.inference.gpt.model_provider._add_load_convert_hooks')
    @patch('megatron.inference.gpt.model_provider.core_transformer_config_from_args')
    @patch('megatron.inference.gpt.model_provider.get_args')
    @patch('megatron.inference.gpt.model_provider.MCoreGPTModel')
    def test_empty_model_kwargs(self, mock_model_class, mock_get_args,
                                mock_config_from_args, mock_add_hooks,
                                mock_print, mock_load_ckpt):
        """
        测试样例8: 空的model_kwargs
        
        验证空kwargs也能正确处理
        """
        mock_get_args.return_value = self.mock_args
        
        mock_transformer_config = Mock()
        mock_config_from_args.return_value = mock_transformer_config
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        config = Namespace(hidden_size=768)
        model_kwargs = {}  # 空kwargs
        
        # 执行
        result = _teacher_provider(config, model_kwargs)
        
        # 验证: 函数正常执行，返回模型
        self.assertEqual(result, mock_model)


if __name__ == "__main__":
    unittest.main()
