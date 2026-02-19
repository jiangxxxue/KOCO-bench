"""
Minimal megatron mock for byteperf tests, providing only necessary interfaces
while preserving core test logic.

Mocks transitive dependencies (regex, modelopt, transformer_engine) so that
the actual byteperf Megatron-LM code can be imported without hard failures.
"""
from unittest.mock import Mock, MagicMock


class MinimalMock:
    """Provides minimal mock without affecting core test logic"""

    @staticmethod
    def setup_megatron_mocks():
        """Setup necessary mocks for byteperf's megatron dependencies"""
        import sys
        from types import ModuleType

        # regex: needed by megatron.training.tokenizer.gpt2_tokenization
        try:
            import regex
        except (ImportError, ModuleNotFoundError):
            regex_mock = MagicMock()
            sys.modules['regex'] = regex_mock

        # modelopt: needed by megatron.inference.algos.distillation
        try:
            import modelopt
            import modelopt.torch.distill
        except (ImportError, ModuleNotFoundError):
            modelopt_mock = ModuleType('modelopt')
            modelopt_torch = ModuleType('modelopt.torch')
            modelopt_torch_distill = ModuleType('modelopt.torch.distill')
            modelopt_torch_opt = ModuleType('modelopt.torch.opt')
            modelopt_torch_opt_conversion = ModuleType('modelopt.torch.opt.conversion')
            modelopt_torch_quantization = ModuleType('modelopt.torch.quantization')

            # Add required classes/functions to modelopt.torch.distill
            modelopt_torch_distill.DistillationModel = MagicMock(name='DistillationModel')
            modelopt_torch_distill.DistillationLossBalancer = MagicMock(name='DistillationLossBalancer')
            modelopt_torch_distill.convert = MagicMock(name='convert')
            modelopt_torch_distill.export = MagicMock(name='export')

            # Add required classes/functions to modelopt.torch.opt
            modelopt_torch_opt.restore_from_modelopt_state = MagicMock(name='restore_from_modelopt_state')
            modelopt_torch_opt.conversion = modelopt_torch_opt_conversion
            modelopt_torch_opt_conversion.get_mode = MagicMock(name='get_mode')

            # Set module relationships
            modelopt_mock.torch = modelopt_torch
            modelopt_torch.distill = modelopt_torch_distill
            modelopt_torch.opt = modelopt_torch_opt
            modelopt_torch.quantization = modelopt_torch_quantization

            # Register all modules
            sys.modules['modelopt'] = modelopt_mock
            sys.modules['modelopt.torch'] = modelopt_torch
            sys.modules['modelopt.torch.distill'] = modelopt_torch_distill
            sys.modules['modelopt.torch.opt'] = modelopt_torch_opt
            sys.modules['modelopt.torch.opt.conversion'] = modelopt_torch_opt_conversion
            sys.modules['modelopt.torch.quantization'] = modelopt_torch_quantization

        # transformer_engine: needed by various megatron modules
        try:
            import transformer_engine
            import transformer_engine.pytorch
        except (ImportError, ModuleNotFoundError, AttributeError, AssertionError):
            te_mock = ModuleType('transformer_engine')
            te_pytorch_mock = ModuleType('transformer_engine.pytorch')

            te_pytorch_mock.Linear = MagicMock(name='Linear')
            te_pytorch_mock.LayerNorm = MagicMock(name='LayerNorm')
            te_pytorch_mock.LayerNormLinear = MagicMock(name='LayerNormLinear')
            te_pytorch_mock.make_graphed_callables = MagicMock(name='make_graphed_callables')

            te_mock.pytorch = te_pytorch_mock

            sys.modules['transformer_engine'] = te_mock
            sys.modules['transformer_engine.pytorch'] = te_pytorch_mock
