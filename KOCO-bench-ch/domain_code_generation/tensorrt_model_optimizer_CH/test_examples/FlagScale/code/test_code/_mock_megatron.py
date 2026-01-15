"""
最小化的megatron mock，只提供必需的接口，保留核心测试逻辑
"""
from unittest.mock import Mock, MagicMock, patch


class MinimalMock:
    """提供最小化的mock，不影响核心测试逻辑"""
    
    @staticmethod
    def setup_megatron_mocks():
        """设置必需的megatron mocks"""
        import sys
        from types import ModuleType
        
        # 先尝试真实导入，失败才mock
        # apex: 尝试真实导入，失败才mock
        try:
            import apex
            import apex.transformer
        except (ImportError, ModuleNotFoundError):
            apex_mock = ModuleType('apex')
            apex_transformer_mock = ModuleType('apex.transformer')
            sys.modules['apex'] = apex_mock
            sys.modules['apex.transformer'] = apex_transformer_mock
        
        # transformer_engine: 尝试真实导入，失败才mock
        try:
            import transformer_engine
            import transformer_engine.pytorch
            # 如果导入成功，检查是否有必需的属性，如果没有则补充
            if not hasattr(transformer_engine.pytorch, 'Linear'):
                transformer_engine.pytorch.Linear = MagicMock(name='Linear')
            if not hasattr(transformer_engine.pytorch, 'LayerNorm'):
                transformer_engine.pytorch.LayerNorm = MagicMock(name='LayerNorm')
            if not hasattr(transformer_engine.pytorch, 'LayerNormLinear'):
                transformer_engine.pytorch.LayerNormLinear = MagicMock(name='LayerNormLinear')
            if not hasattr(transformer_engine.pytorch, 'make_graphed_callables'):
                transformer_engine.pytorch.make_graphed_callables = MagicMock(name='make_graphed_callables')
        except (ImportError, ModuleNotFoundError, AttributeError, AssertionError):
            # 只有在导入失败时才mock version和创建mock模块
            # Mock importlib.metadata.version to handle transformer-engine version check
            import importlib.metadata
            original_version = importlib.metadata.version
            
            def mock_version(package_name):
                if package_name == 'transformer-engine':
                    return '1.0.0'  # 返回一个假版本号
                return original_version(package_name)
            
            importlib.metadata.version = mock_version
            # 创建完整的transformer_engine mock树（使用真正的模块类型）
            te_mock = ModuleType('transformer_engine')
            te_pytorch_mock = ModuleType('transformer_engine.pytorch')
            te_float8_mock = ModuleType('transformer_engine.pytorch.float8_tensor')
            te_cpp_mock = ModuleType('transformer_engine.pytorch.cpp_extensions')
            te_common_mock = ModuleType('transformer_engine.common')
            te_common_recipe_mock = ModuleType('transformer_engine.common.recipe')
            
            # 添加必需的属性和类
            te_float8_mock.Float8Tensor = MagicMock(name='Float8Tensor')
            te_cpp_mock.cast_to_fp8 = MagicMock(name='cast_to_fp8')
            te_cpp_mock.cast_from_fp8 = MagicMock(name='cast_from_fp8')
            
            # 添加 Linear 类（常用层）
            te_pytorch_mock.Linear = MagicMock(name='Linear')
            te_pytorch_mock.LayerNorm = MagicMock(name='LayerNorm')
            te_pytorch_mock.LayerNormLinear = MagicMock(name='LayerNormLinear')
            te_pytorch_mock.make_graphed_callables = MagicMock(name='make_graphed_callables')
            
            # 添加 common.recipe 相关
            te_common_recipe_mock.Format = MagicMock()
            te_common_recipe_mock.Format.E4M3 = 'E4M3'
            te_common_recipe_mock.Format.HYBRID = 'HYBRID'
            te_common_mock.recipe = te_common_recipe_mock
            
            # 设置模块关系
            te_mock.pytorch = te_pytorch_mock
            te_mock.common = te_common_mock
            te_pytorch_mock.float8_tensor = te_float8_mock
            te_pytorch_mock.cpp_extensions = te_cpp_mock
            
            # 注册到sys.modules
            sys.modules['transformer_engine'] = te_mock
            sys.modules['transformer_engine.pytorch'] = te_pytorch_mock
            sys.modules['transformer_engine.pytorch.float8_tensor'] = te_float8_mock
            sys.modules['transformer_engine.pytorch.cpp_extensions'] = te_cpp_mock
            sys.modules['transformer_engine.common'] = te_common_mock
            sys.modules['transformer_engine.common.recipe'] = te_common_recipe_mock
        
        # 如果megatron.training不存在，提供最小stub
        try:
            import megatron.training
        except (ImportError, ModuleNotFoundError):
            # 创建megatron.training的stub
            from types import ModuleType
            
            # 创建training模块
            training = ModuleType('training')
            training.get_args = Mock(name='get_args')
            training.get_timers = Mock(name='get_timers')
            training.get_tokenizer = Mock(name='get_tokenizer')
            training.print_rank_0 = Mock(name='print_rank_0')
            training.pretrain = Mock(name='pretrain')
            training.inprocess_restart = Mock(name='inprocess_restart')
            
            # 创建arguments模块
            arguments = ModuleType('arguments')
            arguments.core_transformer_config_from_args = Mock(name='core_transformer_config_from_args')
            
            # 创建yaml_arguments模块
            yaml_arguments = ModuleType('yaml_arguments')
            yaml_arguments.core_transformer_config_from_yaml = Mock(name='core_transformer_config_from_yaml')
            
            # 创建utils模块
            utils = ModuleType('utils')
            utils.get_batch_on_this_cp_rank = Mock(name='get_batch_on_this_cp_rank')
            utils.get_batch_on_this_tp_rank = Mock(name='get_batch_on_this_tp_rank')
            utils.get_blend_and_blend_per_split = Mock(name='get_blend_and_blend_per_split')
            
            training.arguments = arguments
            training.yaml_arguments = yaml_arguments
            training.utils = utils
            
            sys.modules['megatron.training'] = training
            sys.modules['megatron.training.arguments'] = arguments
            sys.modules['megatron.training.yaml_arguments'] = yaml_arguments
            sys.modules['megatron.training.utils'] = utils
        
        # Mock post_training (ModelOpt相关，可选)
        try:
            import megatron.post_training
        except (ImportError, ModuleNotFoundError):
            post_training = ModuleType('post_training')
            post_training.arguments = ModuleType('arguments')
            post_training.arguments.add_modelopt_args = Mock()
            post_training.arguments.modelopt_args_enabled = Mock(return_value=False)
            post_training.loss_func = ModuleType('loss_func')
            post_training.loss_func.loss_func = Mock()
            post_training.model_provider = ModuleType('model_provider')
            post_training.model_provider.model_provider = Mock()
            
            sys.modules['megatron.post_training'] = post_training
            sys.modules['megatron.post_training.arguments'] = post_training.arguments
            sys.modules['megatron.post_training.loss_func'] = post_training.loss_func
            sys.modules['megatron.post_training.model_provider'] = post_training.model_provider
        
        # Mock rerun_state_machine
        try:
            from megatron.core.rerun_state_machine import get_rerun_state_machine
        except (ImportError, AttributeError):
            if 'megatron.core.rerun_state_machine' not in sys.modules:
                rerun = ModuleType('rerun_state_machine')
                rerun.get_rerun_state_machine = Mock(name='get_rerun_state_machine')
                sys.modules['megatron.core.rerun_state_machine'] = rerun

