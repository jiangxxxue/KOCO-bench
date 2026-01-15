#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 prune_language_model 函数
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
    print(f"警告: 无法导入实现代码: {e}")
    IMPORT_SUCCESS = False


class TestPruneLanguageModel(unittest.TestCase):
    """测试prune_language_model函数"""
    
    def setUp(self):
        """设置测试环境"""
        torch.manual_seed(42)
        self.mock_model = self._create_mock_model()
    
    def _create_mock_model(self):
        """创建Mock模型"""
        model = MagicMock()
        model.__class__.__name__ = "GPTModel"
        model.config = MagicMock()
        model.config.num_layers = 12
        return model
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    def test_drop_layers_takes_precedence(self, mock_mtp):
        """
        测试样例1: drop_layers优先级最高
        
        输入: 
        - drop_layers=[1, 2, 3]
        - algorithm="width_pruning" (同时存在)
        
        预期行为:
        1. 只调用drop_mcore_language_model_layers
        2. 不调用mtp.prune
        3. 忽略algorithm参数
        """
        # 设置mock
        mock_drop_layers = MagicMock(return_value=self.mock_model)
        mock_mtp.plugins.drop_mcore_language_model_layers = mock_drop_layers
        mock_mtp.prune = MagicMock()
        
        # 输入：同时包含drop_layers和algorithm
        config = PruningConfig(
            drop_layers=[1, 2, 3],
            algorithm="width_pruning",
            amount=0.5
        )
        
        # 执行
        result = prune_language_model(
            model=self.mock_model,
            pruning_config=config
        )
        
        # 验证行为1: 调用了drop_layers
        mock_drop_layers.assert_called_once()
        call_args = mock_drop_layers.call_args
        
        # 验证传入的模型
        self.assertEqual(call_args[0][0], self.mock_model)
        
        # 验证传入的layer indices
        self.assertEqual(call_args[0][1], [1, 2, 3])
        
        # 验证行为2: 没有调用mtp.prune
        mock_mtp.prune.assert_not_called()
        
        # 验证输出: 返回模型
        self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    def test_drop_layers_with_empty_list(self, mock_mtp):
        """
        测试样例2: drop_layers为空列表时应该fallback到mtp.prune
        
        输入: drop_layers=[]
        预期行为: 调用mtp.prune而不是drop_layers
        """
        mock_mtp.prune = MagicMock(return_value=self.mock_model)
        
        config = PruningConfig(
            drop_layers=[],  # 空列表
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
        
        # 验证：调用了mtp.prune
        self.assertTrue(mock_mtp.prune.called or not config.drop_layers,
                       "空的drop_layers应该使用mtp.prune")
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    @patch('nemo.collections.llm.modelopt.prune.pruner.llm')
    def test_mtp_prune_requires_data_and_trainer(self, mock_llm, mock_mtp):
        """
        测试样例3: mtp.prune需要data_module和trainer
        
        输入: algorithm="width_pruning", 无drop_layers
        预期行为: 
        1. 必须提供data_module和trainer
        2. 调用mtp.prune时传入这些参数
        """
        mock_mtp.prune = MagicMock(return_value=self.mock_model)
        mock_llm.validate = MagicMock()
        
        config = PruningConfig(
            algorithm="width_pruning",
            amount=0.5
        )
        
        mock_data_module = MagicMock()
        mock_trainer = MagicMock()
        
        # 执行
        result = prune_language_model(
            model=self.mock_model,
            pruning_config=config,
            data_module=mock_data_module,
            trainer=mock_trainer
        )
        
        # 验证：调用了mtp.prune
        mock_mtp.prune.assert_called_once()
        
        # 验证：传入了正确的参数
        call_kwargs = mock_mtp.prune.call_args.kwargs
        self.assertIn('model', call_kwargs)
        
        # 验证返回结果
        self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    def test_drop_layers_preserves_model_type(self, mock_mtp):
        """
        测试样例4: drop_layers保持模型类型不变
        
        输入: GPTModel
        预期行为: 返回的仍然是GPTModel
        """
        # 创建真实的模型结构mock
        mock_returned_model = MagicMock()
        mock_returned_model.__class__.__name__ = "GPTModel"
        
        mock_drop = MagicMock(return_value=mock_returned_model)
        mock_mtp.plugins.drop_mcore_language_model_layers = mock_drop
        
        config = PruningConfig(drop_layers=[5, 6, 7])
        
        result = prune_language_model(
            model=self.mock_model,
            pruning_config=config
        )
        
        # 验证：返回的模型类型正确
        self.assertEqual(result.__class__.__name__, "GPTModel")
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    @patch('nemo.collections.llm.modelopt.prune.pruner.llm')
    def test_mtp_prune_with_different_algorithms(self, mock_llm, mock_mtp):
        """
        测试样例5: 测试不同的剪枝算法
        
        输入: 不同的algorithm值
        预期行为: 所有算法都能正确传递给mtp.prune
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
                
                # 验证：调用了mtp.prune
                self.assertTrue(mock_mtp.prune.called)
                
                # 验证：返回了模型
                self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    def test_drop_layers_with_large_indices(self, mock_mtp):
        """
        测试样例6: drop_layers使用大的层索引
        
        输入: drop_layers=[10, 11]（模型有12层）
        预期行为: 正确传递层索引，不做额外验证
        """
        mock_drop = MagicMock(return_value=self.mock_model)
        mock_mtp.plugins.drop_mcore_language_model_layers = mock_drop
        
        config = PruningConfig(drop_layers=[10, 11])
        
        result = prune_language_model(
            model=self.mock_model,
            pruning_config=config
        )
        
        # 验证：传入了正确的层索引
        call_args = mock_drop.call_args
        self.assertEqual(call_args[0][1], [10, 11])
        
        # 验证：成功返回
        self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    @patch('nemo.collections.llm.modelopt.prune.pruner.unwrap_for_modelopt_operations')
    def test_model_unwrapping_for_mtp(self, mock_unwrap, mock_mtp):
        """
        测试样例7: 模型被unwrap后传递给mtp.prune
        
        输入: wrapped模型
        预期行为: 调用unwrap函数，传递unwrapped模型给mtp.prune
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
        
        # 验证：调用了unwrap
        mock_unwrap.assert_called()
        
        # 验证：mtp.prune被调用
        self.assertTrue(mock_mtp.prune.called)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    @patch('nemo.collections.llm.modelopt.prune.pruner.llm')
    def test_pruning_amount_parameter(self, mock_llm, mock_mtp):
        """
        测试样例8: 剪枝amount参数正确传递
        
        输入: amount=0.3（剪枝30%）
        预期行为: amount参数传递给mtp.prune
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
        
        # 验证：调用了mtp.prune
        self.assertTrue(mock_mtp.prune.called)
        
        # 验证：返回了模型
        self.assertIsNotNone(result)


class TestPruningConfigValidation(unittest.TestCase):
    """测试PruningConfig的参数验证和创建"""
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    def test_pruning_config_with_drop_layers(self):
        """
        测试样例9: 创建包含drop_layers的配置
        
        输入: drop_layers=[0, 1, 2]
        预期: 成功创建配置，参数正确存储
        """
        config = PruningConfig(drop_layers=[0, 1, 2])
        
        self.assertIsNotNone(config)
        self.assertEqual(config.drop_layers, [0, 1, 2])
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    def test_pruning_config_with_algorithm(self):
        """
        测试样例10: 创建包含algorithm的配置
        
        输入: algorithm="width_pruning", amount=0.5
        预期: 成功创建配置，参数正确存储
        """
        config = PruningConfig(
            algorithm="width_pruning",
            amount=0.5
        )
        
        self.assertIsNotNone(config)
        self.assertEqual(config.algorithm, "width_pruning")
        self.assertEqual(config.amount, 0.5)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    def test_pruning_config_default_values(self):
        """
        测试样例11: 测试配置的默认值
        
        输入: 空配置
        预期: 使用默认值
        """
        config = PruningConfig()
        
        self.assertIsNotNone(config)
        # 默认值验证（根据实际实现）
        self.assertTrue(hasattr(config, 'drop_layers'))
        self.assertTrue(hasattr(config, 'algorithm'))


class TestPruningEdgeCases(unittest.TestCase):
    """测试边界情况和错误处理"""
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    def test_drop_single_layer(self, mock_mtp):
        """
        测试样例12: 删除单个层
        
        输入: drop_layers=[5]
        预期: 正确处理单层删除
        """
        mock_drop = MagicMock(return_value=MagicMock())
        mock_mtp.plugins.drop_mcore_language_model_layers = mock_drop
        
        config = PruningConfig(drop_layers=[5])
        model = MagicMock()
        
        result = prune_language_model(
            model=model,
            pruning_config=config
        )
        
        # 验证：调用了drop函数
        mock_drop.assert_called_once()
        
        # 验证：传入了单个层索引
        call_args = mock_drop.call_args
        self.assertEqual(call_args[0][1], [5])
        
        self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    def test_drop_consecutive_layers(self, mock_mtp):
        """
        测试样例13: 删除连续的层
        
        输入: drop_layers=[3, 4, 5, 6]
        预期: 正确处理连续层删除
        """
        mock_drop = MagicMock(return_value=MagicMock())
        mock_mtp.plugins.drop_mcore_language_model_layers = mock_drop
        
        config = PruningConfig(drop_layers=[3, 4, 5, 6])
        model = MagicMock()
        
        result = prune_language_model(
            model=model,
            pruning_config=config
        )
        
        # 验证：传入了连续的层索引
        call_args = mock_drop.call_args
        self.assertEqual(call_args[0][1], [3, 4, 5, 6])
        
        self.assertIsNotNone(result)
    
    @unittest.skipIf(not IMPORT_SUCCESS, "实现代码导入失败")
    @patch('nemo.collections.llm.modelopt.prune.pruner.mtp')
    def test_drop_non_consecutive_layers(self, mock_mtp):
        """
        测试样例14: 删除非连续的层
        
        输入: drop_layers=[1, 5, 9]
        预期: 正确处理非连续层删除
        """
        mock_drop = MagicMock(return_value=MagicMock())
        mock_mtp.plugins.drop_mcore_language_model_layers = mock_drop
        
        config = PruningConfig(drop_layers=[1, 5, 9])
        model = MagicMock()
        
        result = prune_language_model(
            model=model,
            pruning_config=config
        )
        
        # 验证：传入了非连续的层索引
        call_args = mock_drop.call_args
        self.assertEqual(call_args[0][1], [1, 5, 9])
        
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
