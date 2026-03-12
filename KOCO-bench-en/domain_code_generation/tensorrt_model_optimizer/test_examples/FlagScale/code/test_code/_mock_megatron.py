"""
Minimal megatron mock, providing only necessary interfaces while preserving core test logic
"""
from unittest.mock import Mock, MagicMock, patch
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec


class _MockLoader(Loader):
    """Loader that creates a MagicMock module."""
    def create_module(self, spec):
        return MagicMock()
    def exec_module(self, module):
        pass


class _AutoMockFinder(MetaPathFinder):
    """Import hook that auto-mocks any submodule under known prefixes."""
    _PREFIXES = (
        'megatron.training.',
        'megatron.post_training.',
        'megatron.core.',
        'megatron.legacy.',
    )

    def find_module(self, fullname, path=None):
        for prefix in self._PREFIXES:
            if fullname.startswith(prefix):
                return self
        return None

    def find_spec(self, fullname, path, target=None):
        if self.find_module(fullname, path) is not None:
            return ModuleSpec(fullname, _MockLoader(), is_package=True)
        return None


class MinimalMock:
    """Provides minimal mock without affecting core test logic"""

    @staticmethod
    def setup_megatron_mocks():
        """Setup necessary megatron mocks"""
        import sys
        from types import ModuleType

        # Install auto-mock finder for megatron submodules
        if not any(isinstance(f, _AutoMockFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, _AutoMockFinder())
        
        # Try real import first, mock only if it fails
        # apex: Try real import first, mock only if it fails
        try:
            import apex
            import apex.transformer
        except (ImportError, ModuleNotFoundError):
            apex_mock = ModuleType('apex')
            apex_transformer_mock = ModuleType('apex.transformer')
            sys.modules['apex'] = apex_mock
            sys.modules['apex.transformer'] = apex_transformer_mock
        
        # transformer_engine: Try real import first, mock only if it fails
        try:
            import transformer_engine
            import transformer_engine.pytorch
            # If import succeeds, check for required attributes and add them if missing
            if not hasattr(transformer_engine.pytorch, 'Linear'):
                transformer_engine.pytorch.Linear = MagicMock(name='Linear')
            if not hasattr(transformer_engine.pytorch, 'LayerNorm'):
                transformer_engine.pytorch.LayerNorm = MagicMock(name='LayerNorm')
            if not hasattr(transformer_engine.pytorch, 'LayerNormLinear'):
                transformer_engine.pytorch.LayerNormLinear = MagicMock(name='LayerNormLinear')
            if not hasattr(transformer_engine.pytorch, 'make_graphed_callables'):
                transformer_engine.pytorch.make_graphed_callables = MagicMock(name='make_graphed_callables')
        except (ImportError, ModuleNotFoundError, AttributeError, AssertionError):
            # Only mock version and create mock modules when import fails
            # Mock importlib.metadata.version to handle transformer-engine version check
            import importlib.metadata
            original_version = importlib.metadata.version
            
            def mock_version(package_name):
                if package_name == 'transformer-engine':
                    return '1.0.0'  # Return a fake version number
                return original_version(package_name)
            
            importlib.metadata.version = mock_version
            # Create complete transformer_engine mock tree (using real module types)
            te_mock = ModuleType('transformer_engine')
            te_pytorch_mock = ModuleType('transformer_engine.pytorch')
            te_float8_mock = ModuleType('transformer_engine.pytorch.float8_tensor')
            te_cpp_mock = ModuleType('transformer_engine.pytorch.cpp_extensions')
            te_common_mock = ModuleType('transformer_engine.common')
            te_common_recipe_mock = ModuleType('transformer_engine.common.recipe')
            
            te_mock.__version__ = '1.0.0'

            # Add required attributes and classes
            te_float8_mock.Float8Tensor = MagicMock(name='Float8Tensor')
            te_cpp_mock.cast_to_fp8 = MagicMock(name='cast_to_fp8')
            te_cpp_mock.cast_from_fp8 = MagicMock(name='cast_from_fp8')

            # Add Linear class (commonly used layer)
            te_pytorch_mock.Linear = MagicMock(name='Linear')
            te_pytorch_mock.LayerNorm = MagicMock(name='LayerNorm')
            te_pytorch_mock.LayerNormLinear = MagicMock(name='LayerNormLinear')
            te_pytorch_mock.DotProductAttention = MagicMock(name='DotProductAttention')
            te_pytorch_mock.make_graphed_callables = MagicMock(name='make_graphed_callables')

            te_pytorch_distributed = ModuleType('transformer_engine.pytorch.distributed')
            te_pytorch_distributed.CudaRNGStatesTracker = MagicMock(name='CudaRNGStatesTracker')
            te_pytorch_mock.distributed = te_pytorch_distributed
            
            # Add common.recipe related
            te_common_recipe_mock.Format = MagicMock()
            te_common_recipe_mock.Format.E4M3 = 'E4M3'
            te_common_recipe_mock.Format.HYBRID = 'HYBRID'
            te_common_mock.recipe = te_common_recipe_mock
            
            # Set module relationships
            te_mock.pytorch = te_pytorch_mock
            te_mock.common = te_common_mock
            te_pytorch_mock.float8_tensor = te_float8_mock
            te_pytorch_mock.cpp_extensions = te_cpp_mock
            
            # Register to sys.modules
            sys.modules['transformer_engine'] = te_mock
            sys.modules['transformer_engine.pytorch'] = te_pytorch_mock
            sys.modules['transformer_engine.pytorch.float8_tensor'] = te_float8_mock
            sys.modules['transformer_engine.pytorch.cpp_extensions'] = te_cpp_mock
            sys.modules['transformer_engine.pytorch.distributed'] = te_pytorch_distributed
            sys.modules['transformer_engine.common'] = te_common_mock
            sys.modules['transformer_engine.common.recipe'] = te_common_recipe_mock
        
        # Mock the full megatron package tree so that
        # `from megatron.core import parallel_state` etc. succeed.
        # We use MagicMock for all submodules so any attribute access works.
        try:
            import megatron.core
        except (ImportError, ModuleNotFoundError):
            _megatron_submodules = [
                'megatron',
                'megatron.core',
                'megatron.core.parallel_state',
                'megatron.core.mpu',
                'megatron.core.enums',
                'megatron.core.utils',
                'megatron.core.fp8_utils',
                'megatron.core.num_microbatches_calculator',
                'megatron.core.ModelParallelConfig',
                'megatron.core.msc_utils',
                'megatron.core.datasets',
                'megatron.core.datasets.blended_megatron_dataset_builder',
                'megatron.core.datasets.gpt_dataset',
                'megatron.core.datasets.indexed_dataset',
                'megatron.core.datasets.utils',
                'megatron.core.models',
                'megatron.core.models.gpt',
                'megatron.core.models.gpt.gpt_layer_specs',
                'megatron.core.models.gpt.heterogeneous',
                'megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs',
                'megatron.core.transformer',
                'megatron.core.transformer.spec_utils',
                'megatron.core.transformer.module',
                'megatron.core.transformer.mlp',
                'megatron.core.transformer.multi_token_prediction',
                'megatron.core.transformer.moe',
                'megatron.core.transformer.moe.upcycling_utils',
                'megatron.core.transformer.moe.moe_utils',
                'megatron.core.distributed',
                'megatron.core.distributed.custom_fsdp',
                'megatron.core.optimizer',
                'megatron.core.optimizer_param_scheduler',
                'megatron.core.pipeline_parallel',
                'megatron.core.pipeline_parallel.schedules',
                'megatron.core.pipeline_parallel.p2p_communication',
                'megatron.core.tensor_parallel',
                'megatron.core.tensor_parallel.mappings',
                'megatron.core.dist_checkpointing',
                'megatron.core.dist_checkpointing.mapping',
                'megatron.core.rerun_state_machine',
                'megatron.core.extensions',
                'megatron.core.extensions.transformer_engine',
                'megatron.legacy',
                'megatron.legacy.model',
                'megatron.legacy.data',
                'megatron.legacy.data.data_samplers',
            ]
            for mod_name in _megatron_submodules:
                if mod_name not in sys.modules:
                    sys.modules[mod_name] = MagicMock()

            # Make GPTDatasetConfig a real class so @dataclass inheritance works
            # (MagicMock doesn't support __mro__ which @dataclass needs)
            gpt_dataset_mod = sys.modules['megatron.core.datasets.gpt_dataset']
            gpt_dataset_mod.GPTDatasetConfig = type('GPTDatasetConfig', (), {})

        # Register megatron.training and megatron.post_training as proper
        # packages (ModuleType with __path__) so the _AutoMockFinder can
        # handle all submodule imports dynamically.
        for parent in ['megatron.training', 'megatron.post_training']:
            try:
                __import__(parent)
            except (ImportError, ModuleNotFoundError):
                if parent not in sys.modules:
                    mod = ModuleType(parent)
                    mod.__path__ = []
                    mod.__package__ = parent
                    mod.__getattr__ = lambda name: MagicMock()
                    sys.modules[parent] = mod

