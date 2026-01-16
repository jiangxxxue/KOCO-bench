"""
Minimal stub of `peft` to satisfy optional dependency checks during tests.

This package exists only for local testing. It provides:
- LoraConfig: a lightweight container for LoRA parameters
- get_peft_model(model, config): no-op that returns the passed model

Having this as a real package (directory with __init__.py) ensures that
`importlib.util.find_spec("peft")` returns a non-None spec so that
Transformers' optional dependency probing does not raise:
ValueError: peft.__spec__ is None
"""

from __future__ import annotations
from typing import Any, Iterable, Optional


class LoraConfig:
    """
    Minimal stub matching attribute names accessed in ToolBrain code.
    All parameters are optional and stored as attributes.
    """
    def __init__(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        target_modules: Optional[Iterable[str]] = None,
        lora_dropout: float = 0.0,
        bias: str = "none",
        task_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.r = r
        self.lora_alpha = lora_alpha
        self.target_modules = list(target_modules) if target_modules is not None else []
        self.lora_dropout = lora_dropout
        self.bias = bias
        self.task_type = task_type
        # Store any extra fields for forward compatibility
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        return (
            f"LoraConfig(r={self.r}, lora_alpha={self.lora_alpha}, "
            f"target_modules={self.target_modules}, lora_dropout={self.lora_dropout}, "
            f"bias={self.bias!r}, task_type={self.task_type!r})"
        )


def get_peft_model(model: Any, config: LoraConfig, *args: Any, **kwargs: Any) -> Any:
    """
    No-op stub: return the model unchanged.
    Exists so code paths calling `peft.get_peft_model(...)` don't fail during tests.
    """
    return model


__all__ = ["LoraConfig", "get_peft_model"]