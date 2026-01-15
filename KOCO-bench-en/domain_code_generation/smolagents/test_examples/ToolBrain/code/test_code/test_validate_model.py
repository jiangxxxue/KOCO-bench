import sys
import os
from pathlib import Path

# Ensure actual code under this instance can be imported
THIS_DIR = Path(__file__).resolve().parent
CODE_ROOT = THIS_DIR.parent  # .../ToolBrain/code
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
os.chdir(str(CODE_ROOT))

import pytest  # type: ignore
from toolbrain.prompt import validate_model  # type: ignore


def test_validate_model_none_raises_runtime_error():
    with pytest.raises(RuntimeError):
        validate_model(None)


def test_validate_model_missing_generate_raises_not_implemented():
    class NoGen:
        pass

    with pytest.raises(NotImplementedError):
        validate_model(NoGen())


def test_validate_model_generate_without_positional_args_raises_type_error():
    class BadGenSig:
        def generate(self, *, messages=None):
            return "ok"

    with pytest.raises(TypeError):
        validate_model(BadGenSig())


def test_validate_model_with_valid_generate_passes():
    class GoodModel:
        def generate(self, messages, *args, **kwargs):
            return "ok"

    # Should not raise exception
    validate_model(GoodModel())