"""
Shared mock setup for Nemo test files.

Instead of trying to resolve the full Nemo transitive import chain,
this module:
1. Mocks ALL external dependencies (megatron, lightning, etc.) with
   a MetaPathFinder.
2. Mocks the nemo package tree itself with the same finder.
3. Provides import_nemo_module() to load a specific .py file directly,
   bypassing __init__.py chains.

Usage:
    from _mock_nemo_deps import setup_nemo_mocks, import_nemo_module
    setup_nemo_mocks()
    utils = import_nemo_module('nemo.collections.llm.modelopt.distill.utils')
    adjust_distillation_model_for_mcore = utils.adjust_distillation_model_for_mcore
"""
import sys
import os
import abc as _abc
import importlib.util
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec
from types import ModuleType
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Mock class infrastructure (supports inheritance, issubclass, generics)
# ---------------------------------------------------------------------------

class _MockMeta(_abc.ABCMeta):
    """Metaclass compatible with ABCMeta."""
    def __getattr__(cls, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        return _make_mock_class(f'{cls.__name__}.{name}')

    def __instancecheck__(cls, instance):
        return True

    def __subclasscheck__(cls, subclass):
        return True


def _make_mock_class(name):
    return _MockMeta(name, (), {
        '__init__': lambda self, *a, **kw: None,
        '__call__': lambda self, *a, **kw: MagicMock(),
        '__getattr__': lambda self, n: MagicMock(),
        '__class_getitem__': classmethod(lambda cls, *a: cls),
    })


class _MockModule:
    """Mock module whose attributes are classes safe for inheritance."""

    def __init__(self, name='mock'):
        self.__name__ = name
        self.__path__ = []
        self.__package__ = name
        self.__spec__ = ModuleSpec(name, None, is_package=True)
        self.__version__ = '0.0.0'
        self.__file__ = f'<mock {name}>'
        self.__all__ = []
        self._cache = {}

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        if name not in self._cache:
            self._cache[name] = _make_mock_class(f'{self.__name__}.{name}')
        return self._cache[name]

    def __repr__(self):
        return f'<_MockModule {self.__name__!r}>'


class _MockLoader(Loader):
    def create_module(self, spec):
        return _MockModule(spec.name)

    def exec_module(self, module):
        pass


# ---------------------------------------------------------------------------
# Import hook
# ---------------------------------------------------------------------------

# Prefixes whose submodules (and top-level) should be auto-mocked.
_MOCK_PREFIXES = (
    'lightning.', 'pytorch_lightning.', 'megatron.', 'modelopt.',
    'nemo_run.', 'nv_one_logger.', 'fiddle.', 'wrapt.',
    'accelerate.', 'flash_attn.', 'apex.', 'transformer_engine.',
    'hydra.', 'omegaconf.', 'lhotse.', 'sentencepiece.',
    'webdataset.', 'braceexpand.', 'tensorstore.', 'zarr.',
    'datasets.', 'tqdm.', 'einops.',
    'nemo.',  # mock nemo itself to avoid __init__.py chain
)
# Extra top-level packages not in _MOCK_PREFIXES that need mocking
_EXTRA_TOPLEVEL = {'datasets', 'tqdm', 'einops'}
_MOCK_TOPLEVEL = {p[:-1] for p in _MOCK_PREFIXES} | _EXTRA_TOPLEVEL


class _PrefixFinder(MetaPathFinder):
    """Intercepts imports for known prefixes and returns mock modules."""

    def find_spec(self, fullname, path, target=None):
        # Check prefixed submodules
        for prefix in _MOCK_PREFIXES:
            if fullname.startswith(prefix):
                return ModuleSpec(fullname, _MockLoader(), is_package=True)
        # Check exact top-level matches
        if fullname in _MOCK_TOPLEVEL:
            return ModuleSpec(fullname, _MockLoader(), is_package=True)
        return None


# ---------------------------------------------------------------------------
# Direct file import (bypasses __init__.py chain)
# ---------------------------------------------------------------------------

def import_nemo_module(dotted_name):
    """Import a specific nemo module file, bypassing __init__.py chains.

    Args:
        dotted_name: e.g. 'nemo.collections.llm.modelopt.distill.utils'

    Returns:
        The loaded module object.
    """
    # Find the code directory (parent of test_code)
    code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Convert dotted name to file path
    parts = dotted_name.split('.')
    file_path = os.path.join(code_dir, *parts) + '.py'
    if not os.path.isfile(file_path):
        # Try as package __init__
        file_path = os.path.join(code_dir, *parts, '__init__.py')

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Cannot find file for {dotted_name}: tried {file_path}")

    # Register parent packages as mock modules if not already present
    for i in range(1, len(parts)):
        parent = '.'.join(parts[:i])
        if parent not in sys.modules:
            sys.modules[parent] = _MockModule(parent)

    # Load the actual file
    spec = importlib.util.spec_from_file_location(dotted_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_nemo_mocks():
    """Install auto-mock import hook for external dependencies."""

    # Install prefix-based finder FIRST so it intercepts before real finders.
    if not any(isinstance(f, _PrefixFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _PrefixFinder())

    # Pre-register top-level mock packages.
    for mod_name in _MOCK_TOPLEVEL:
        if mod_name not in sys.modules:
            mock_mod = ModuleType(mod_name)
            mock_mod.__path__ = []
            mock_mod.__package__ = mod_name
            mock_mod.__spec__ = ModuleSpec(mod_name, None, is_package=True)
            mock_mod.__version__ = '0.0.0'
            mock_mod.__getattr__ = lambda name, _m=mod_name: MagicMock()
            sys.modules[mod_name] = mock_mod

    # Patch importlib.metadata.version for mocked packages.
    import importlib.metadata as _meta
    if not getattr(_meta.version, '_nemo_patched', False):
        _orig = _meta.version

        def _patched(name):
            try:
                return _orig(name)
            except _meta.PackageNotFoundError:
                return '0.0.0'

        _patched._nemo_patched = True
        _meta.version = _patched
