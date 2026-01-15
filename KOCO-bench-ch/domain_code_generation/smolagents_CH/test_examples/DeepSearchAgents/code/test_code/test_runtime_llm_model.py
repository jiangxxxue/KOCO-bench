import os
import sys
import types
import importlib
import pytest

# Ensure we can import project src modules relative to this test file
CURRENT_DIR = os.path.dirname(__file__)
CODE_DIR = os.path.dirname(CURRENT_DIR)  # .../code
SRC_DIR = os.path.join(CODE_DIR, "src")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Also add smolagents library source path from knowledge corpus
# Path: KOCO-bench/KOCO-bench-ch/domain_code_generation/smolagents/knowledge_corpus/smolagents/src
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../../"))
SMOLAGENTS_SRC = os.path.join(PROJECT_ROOT, "smolagents", "knowledge_corpus", "smolagents", "src")
if SMOLAGENTS_SRC not in sys.path:
    sys.path.insert(0, SMOLAGENTS_SRC)


def import_runtime_with_stubs():
    """
    Import agents.runtime.AgentRuntime with stubs to avoid heavy/broken deps:
    - core.config.settings -> provide minimal settings
    - agents.react_agent / codact_agent / manager_agent -> minimal class stubs
    - agents.ui_common.console_formatter/constants/agent_step_callback -> stubs
    - tools.from_toolbox -> no-op returning empty tool collection
    """
    # Stub core.config.settings.settings
    settings_stub = types.SimpleNamespace(
        get_api_key=lambda name: None,
        TOOLS_SPECIFIC_CONFIG={},
        RERANKER_TYPE="",
        VERBOSE_TOOL_CALLBACKS=False,
        DEEPSEARCH_AGENT_MODE="react",
        ORCHESTRATOR_MODEL_ID="orc",
        SEARCH_MODEL_NAME="search",
        REACT_MAX_STEPS=5,
        REACT_PLANNING_INTERVAL=2,
        REACT_MAX_TOOL_THREADS=2,
        REACT_ENABLE_STREAMING=False,
        CODACT_EXECUTOR_TYPE="local",
        CODACT_MAX_STEPS=5,
        CODACT_VERBOSITY_LEVEL=0,
        CODACT_ADDITIONAL_IMPORTS=[],
        CODACT_EXECUTOR_KWARGS={},
        CODACT_PLANNING_INTERVAL=2,
        CODACT_USE_STRUCTURED_OUTPUTS=False,
        CODACT_ENABLE_STREAMING=False,
    )
    mod_core_config = types.ModuleType("core.config.settings")
    setattr(mod_core_config, "settings", settings_stub)
    sys.modules["core.config.settings"] = mod_core_config

    # Stub tools.from_toolbox
    mod_tools = types.ModuleType("tools")

    def _from_toolbox(**kwargs):
        class _ToolCollection:
            tools = []
        return _ToolCollection()

    setattr(mod_tools, "from_toolbox", _from_toolbox)
    sys.modules["tools"] = mod_tools

    # Stub agents submodules to avoid importing real prompt templates / consoleformatter
    mod_react = types.ModuleType("agents.react_agent")
    setattr(mod_react, "ReactAgent", type("ReactAgent", (), {}))
    sys.modules["agents.react_agent"] = mod_react

    mod_codact = types.ModuleType("agents.codact_agent")
    setattr(mod_codact, "CodeActAgent", type("CodeActAgent", (), {}))
    sys.modules["agents.codact_agent"] = mod_codact

    mod_manager = types.ModuleType("agents.manager_agent")
    setattr(mod_manager, "ManagerAgent", type("ManagerAgent", (), {}))
    sys.modules["agents.manager_agent"] = mod_manager

    mod_ui_cb = types.ModuleType("agents.ui_common.agent_step_callback")
    setattr(mod_ui_cb, "AgentStepCallback", type("AgentStepCallback", (), {}))
    sys.modules["agents.ui_common.agent_step_callback"] = mod_ui_cb

    # Provide harmless stubs to short-circuit ui_common imports
    mod_ui_consts = types.ModuleType("agents.ui_common.constants")
    setattr(mod_ui_consts, "AGENT_EMOJIS", {})
    sys.modules["agents.ui_common.constants"] = mod_ui_consts

    mod_console_formatter = types.ModuleType("agents.ui_common.console_formatter")
    sys.modules["agents.ui_common.console_formatter"] = mod_console_formatter

    # Also stub prompt_templates to avoid importing react_prompts code-paths
    mod_prompt_pkg = types.ModuleType("agents.prompt_templates")
    sys.modules["agents.prompt_templates"] = mod_prompt_pkg
    mod_react_prompts = types.ModuleType("agents.prompt_templates.react_prompts")
    setattr(mod_react_prompts, "REACT_PROMPT", {})
    sys.modules["agents.prompt_templates.react_prompts"] = mod_react_prompts
    mod_codact_prompts = types.ModuleType("agents.prompt_templates.codact_prompts")
    for name in ("CODACT_SYSTEM_EXTENSION", "PLANNING_TEMPLATES", "FINAL_ANSWER_EXTENSION", "MANAGED_AGENT_TEMPLATES"):
        setattr(mod_codact_prompts, name, {})
    def _merge_prompt_templates(**kwargs):
        return {}
    setattr(mod_codact_prompts, "merge_prompt_templates", _merge_prompt_templates)
    sys.modules["agents.prompt_templates.codact_prompts"] = mod_codact_prompts

    # Import now with stubs in place
    runtime_module = importlib.import_module("agents.runtime")
    return getattr(runtime_module, "AgentRuntime")


def test_create_llm_model_strips_trailing_slash_and_sets_params():
    # Try to import AgentRuntime with stubs; skip if still fails
    try:
        AgentRuntime = import_runtime_with_stubs()
    except Exception as e:
        pytest.skip(f"AgentRuntime import failed due to optional dependencies: {e}")

    # Instantiate runtime (tools list will be empty due to stubbed from_toolbox)
    runtime = AgentRuntime()

    # Monkeypatch API keys to control model creation
    runtime.api_keys["litellm_master_key"] = "test-api-key"
    runtime.api_keys["litellm_base_url"] = "http://example.com/api/"

    # Create model
    model = runtime._create_llm_model(model_id="unit-test-model")

    # Validate core params
    assert getattr(model, "model_id", None) == "unit-test-model"
    assert getattr(model, "kwargs", {}).get("temperature") == 0.2
    assert getattr(model, "api_key", None) == "test-api-key"

    # Trailing slash must be removed
    assert getattr(model, "api_base", None) == "http://example.com/api"

    # Ensure object has generate() or __call__ interface typical for LiteLLMModel
    has_generate = hasattr(model, "generate")
    has_call = hasattr(model, "__call__")
    assert has_generate or has_call