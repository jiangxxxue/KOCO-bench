"""OpenHands SDK agent wrapper.

Provides a thin interface over the OpenHands SDK to run an agent in a
given workspace directory.  All SDK-specific imports and configuration
are isolated here so that the rest of the codebase stays decoupled.
"""

try:
    from pydantic import SecretStr
    from openhands.sdk import (
        LLM, Agent, Conversation, Tool, ConversationExecutionStatus,
    )
    from openhands.sdk.conversation.exceptions import ConversationRunError
    from openhands.tools import register_default_tools
    register_default_tools()  # registers terminal, file_editor, etc. into SDK registry
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    # Re-export a sentinel so callers can reference the name without importing
    # the real class.
    ConversationExecutionStatus = None  # type: ignore[assignment,misc]
    ConversationRunError = Exception    # type: ignore[assignment,misc]


def _resolve_llm_model(model: str, base_url: str) -> str:
    """Add ``openrouter/`` prefix when the base URL is OpenRouter.

    litellm uses the model-name prefix for provider routing.  Without the
    ``openrouter/`` prefix, ``deepseek/…`` gets routed directly to the
    DeepSeek API, ignoring the custom base URL.
    """
    if "openrouter.ai" in base_url and not model.startswith("openrouter/"):
        return f"openrouter/{model}"
    return model


def run_sdk_agent(prompt, workspace, model, api_key, base_url,
                  max_iterations=50, corpus_dirs=None):
    """Run the OpenHands SDK agent and return (events, status).

    Creates an ephemeral LLM → Agent → Conversation pipeline, sends
    ``prompt``, and blocks until the agent finishes.

    Parameters:
        corpus_dirs: List of directories to index for knowledge search.
            When non-empty, the ``knowledge_search`` hybrid-search tool is
            registered and added to the agent's tool list.

    Returns:
        (events, status) where *events* is a list of SDK event objects and
        *status* is a :class:`ConversationExecutionStatus` enum value.

    Raises:
        RuntimeError: If the SDK is not installed.
        ConversationRunError: If the agent encounters a fatal error.
    """
    if not SDK_AVAILABLE:
        raise RuntimeError(
            "openhands-sdk not installed. "
            "Install with: uv pip install openhands-sdk openhands-tools --python 3.12"
        )

    llm_model = _resolve_llm_model(model, base_url)

    llm = LLM(
        model=llm_model,
        api_key=SecretStr(api_key),
        base_url=base_url,
        temperature=0.0,
        max_output_tokens=65536,
    )

    tools_list = [
        Tool(name="terminal"),
        Tool(name="file_editor"),
    ]
    if corpus_dirs:
        import tools.knowledge_search  # noqa: F401 — triggers register_tool()
        tools_list.append(Tool(name="knowledge_search", params={"corpus_dirs": corpus_dirs}))

    agent = Agent(
        llm=llm,
        tools=tools_list,
        include_default_tools=["FinishTool", "ThinkTool"],
    )

    conversation = Conversation(
        agent=agent,
        workspace=workspace,
        max_iteration_per_run=max_iterations,
        visualizer=None,
    )

    try:
        conversation.send_message(prompt)
        conversation.run()
    finally:
        events = list(conversation.state.events)
        status = conversation.state.execution_status
        conversation.close()

    return events, status
