#!/usr/bin/env python
# -*- coding: utf-8 -*-
# src/agents/__init__.py

"""
DeepSearchAgent <main package>
Provides agent execution, management, and UI components

NOTE: Some components like Redis and SSE have been temporarily disabled for
      reconstruction
"""

# lazy import to avoid heavy dependencies during package initialization
def __getattr__(name):
    if name == "AgentRuntime":
        from .runtime import AgentRuntime
        return AgentRuntime
    if name == "ConsoleFormatter":
        from .ui_common.console_formatter import ConsoleFormatter
        return ConsoleFormatter
    if name == "RunResult":
        from .run_result import RunResult
        return RunResult
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = [
    "AgentRuntime",
    # "agent_observer",  # ensure observer is initialized first
    # "AgentStepCallback",  # removed temporarily, class has been refactored
    "ConsoleFormatter",
    "RunResult"  # v1.19.0 feature
]
