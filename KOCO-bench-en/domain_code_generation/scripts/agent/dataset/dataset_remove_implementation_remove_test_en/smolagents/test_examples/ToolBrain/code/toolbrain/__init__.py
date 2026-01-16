"""
ToolBrain - A framework for training LLM-powered agents to use tools effectively.

This __init__ keeps imports lazy to avoid importing heavy optional deps (e.g. peft)
during package import. Submodules like `toolbrain.prompt` can be imported without
triggering adapter/model backends.
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

__version__ = "0.1.0"

# Public API
__all__ = [
    # === PRIMARY API ===
    "Brain",

    # === FACTORY FUNCTIONS ===
    "create_agent",
    "create_optimized_model",

    # === CONFIGURATION ===
    "get_config",
    "GRPOConfig",
    "DPOConfig",
    "SupervisedConfig",
    "BaseConfig",

    # === REWARD FUNCTIONS ===
    "reward_exact_match",
    "reward_tool_execution_success",
    "reward_step_efficiency",
    "reward_behavior_uses_search_first",
    "reward_safety_no_os_system",
    "reward_combined",
    "reward_llm_judge_via_ranking",
    "RewardFunctionWrapper",
    "create_reward_function",

    # === ADVANCED/LEGACY ===
    "Trace",
    "BaseAgentAdapter",
    "SmolAgentAdapter",
    "LangChainAdapter",
    "create_huggingface_chat_model",
]

# Lazy export map: name -> (module, attribute)
_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Factory
    "create_agent": (".factory", "create_agent"),
    "create_optimized_model": (".factory", "create_optimized_model"),

    # Core
    "Brain": (".brain", "Brain"),

    # Config
    "get_config": (".config", "get_config"),
    "GRPOConfig": (".config", "GRPOConfig"),
    "DPOConfig": (".config", "DPOConfig"),
    "SupervisedConfig": (".config", "SupervisedConfig"),
    "BaseConfig": (".config", "BaseConfig"),

    # Rewards
    "reward_exact_match": (".rewards", "reward_exact_match"),
    "reward_tool_execution_success": (".rewards", "reward_tool_execution_success"),
    "reward_step_efficiency": (".rewards", "reward_step_efficiency"),
    "reward_behavior_uses_search_first": (".rewards", "reward_behavior_uses_search_first"),
    "reward_safety_no_os_system": (".rewards", "reward_safety_no_os_system"),
    "reward_combined": (".rewards", "reward_combined"),
    "reward_llm_judge_via_ranking": (".rewards", "reward_llm_judge_via_ranking"),
    "RewardFunctionWrapper": (".rewards", "RewardFunctionWrapper"),
    "create_reward_function": (".rewards", "create_reward_function"),

    # Types
    "Trace": (".core_types", "Trace"),

    # Adapters (may pull optional deps; keep lazy)
    "BaseAgentAdapter": (".adapters.base_adapter", "BaseAgentAdapter"),
    "SmolAgentAdapter": (".adapters.smolagent.smolagent_adapter", "SmolAgentAdapter"),
    "LangChainAdapter": (".adapters.langchain.langchain_adapter", "LangChainAdapter"),
    "create_huggingface_chat_model": (".adapters.langchain.langchain_adapter", "create_huggingface_chat_model"),
}


def __getattr__(name: str):
    """PEP 562 lazy attribute loader for public API."""
    try:
        module_path, attr = _EXPORTS[name]
    except KeyError as e:
        raise AttributeError(f"module 'toolbrain' has no attribute {name!r}") from e
    mod = importlib.import_module(module_path, __name__)
    try:
        return getattr(mod, attr)
    except AttributeError as e:
        raise AttributeError(f"failed to load attribute {attr!r} from {module_path}") from e


def __dir__():
    return sorted(list(globals().keys()) + __all__)
