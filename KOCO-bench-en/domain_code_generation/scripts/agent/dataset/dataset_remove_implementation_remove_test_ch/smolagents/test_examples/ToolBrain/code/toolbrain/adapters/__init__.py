"""
ToolBrain adapters module.

This module provides adapters for different agent frameworks,
organized into separate subdirectories for better code organization.
"""

from .base_adapter import BaseAgentAdapter
from .smolagent import SmolAgentAdapter

# 可选导入：LangChain 相关依赖缺失时不阻断整个包的导入
try:
    from .langchain import LangChainAdapter, create_huggingface_chat_model
except Exception:
    LangChainAdapter = None

    def create_huggingface_chat_model(*args, **kwargs):
        raise ImportError(
            "LangChain dependencies not found. Please install 'langchain langgraph langchain-huggingface' to use LangChainAdapter."
        )

__all__ = [
    "BaseAgentAdapter",
    "SmolAgentAdapter",
    "LangChainAdapter",
    "create_huggingface_chat_model",  # Helper function
]