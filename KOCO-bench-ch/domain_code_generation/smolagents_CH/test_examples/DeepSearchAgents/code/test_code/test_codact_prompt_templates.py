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
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../../"))
SMOLAGENTS_SRC = os.path.join(PROJECT_ROOT, "smolagents", "knowledge_corpus", "smolagents", "src")
if SMOLAGENTS_SRC not in sys.path:
    sys.path.insert(0, SMOLAGENTS_SRC)


def _stub_settings(codact_planning_interval=4):
    """
    Provide a stub for src.core.config.settings.settings used by merge_prompt_templates.
    """
    settings_stub = types.SimpleNamespace(
        CODACT_PLANNING_INTERVAL=codact_planning_interval
    )
    mod_core_config = types.ModuleType("src.core.config.settings")
    setattr(mod_core_config, "settings", settings_stub)
    sys.modules["src.core.config.settings"] = mod_core_config


def _prep_constants_stub():
    """
    Ensure AGENT_EMOJIS exists to expand emojis in the system prompt.
    """
    mod_constants = types.ModuleType("src.agents.ui_common.constants")
    setattr(
        mod_constants,
        "AGENT_EMOJIS",
        {
            "thinking": "💭",
            "planning": "🧠",
            "replanning": "🔁",
            "action": "⚙️",
            "final": "✅",
            "error": "💥",
        },
    )
    sys.modules["src.agents.ui_common.constants"] = mod_constants


class DummySelf:
    """
    Minimal stub to call CodeActAgent._create_prompt_templates without full initialization.
    """
    def __init__(self, use_structured_outputs_internally: bool):
        self.use_structured_outputs_internally = use_structured_outputs_internally


def _import_codact_agent():
    """
    Import agents.codact_agent, ensuring its dependencies resolve via stubs.
    """
    # Stubs needed by codact_prompts merge function
    _stub_settings(codact_planning_interval=7)
    _prep_constants_stub()

    # Stub out src.tools package to avoid importing heavy search engines (xai_sdk, etc.)
    # Create a dummy package 'src.tools' and submodule 'src.tools.toolbox'
    mod_src_tools = types.ModuleType("src.tools")
    sys.modules["src.tools"] = mod_src_tools
    mod_toolbox = types.ModuleType("src.tools.toolbox")
    setattr(mod_toolbox, "TOOL_ICONS", {
    "search_links": "🔍",
    "read_url": "📄",
    "chunk_text": "✂️",
    "rerank_texts": "🏆",
    "embed_texts": "🧩",
    "wolfram": "🧮",
    "final_answer": "✅",
    "github_repo_qa": "🐙",
    "xcom_deep_qa": "🐦",
    "search_fast": "⚡🔍",
})
    sys.modules["src.tools.toolbox"] = mod_toolbox

    # Import codact_agent
    from agents.codact_agent import CodeActAgent
    return CodeActAgent


def test_create_prompt_templates_regular_mode_has_required_keys():
    CodeActAgent = _import_codact_agent()
    dummy = DummySelf(use_structured_outputs_internally=False)

    templates = CodeActAgent._create_prompt_templates(dummy)
    assert isinstance(templates, dict)

    # Check top-level keys expected by PromptTemplates ctor
    assert "system_prompt" in templates
    assert "planning" in templates
    assert "final_answer" in templates
    assert "managed_agent" in templates

    # System prompt includes emojis and planning interval expansion
    system_prompt = templates["system_prompt"]
    assert "{{ CURRENT_TIME }}" not in system_prompt  # placeholder should be replaced; actual timestamp line remains
    assert "💭" in system_prompt  # thinking emoji
    assert "🧠" in system_prompt  # planning emoji
    assert "⚙️" in system_prompt  # action emoji
    assert "✅" in system_prompt  # final emoji
    assert "Every 7 steps" in system_prompt  # no literal here, but ensure interval was inserted
    # The codact_prompts merges planning_interval by replacing {{planning_interval}}
    assert "7" in system_prompt  # our stubbed interval present as text

    # Planning templates present with required subkeys (regular mode uses full templates)
    planning = templates["planning"]
    assert "initial_plan" in planning
    assert "update_plan_pre_messages" in planning
    assert "update_plan_post_messages" in planning
    assert "Facts survey" in planning["initial_plan"]  # from PLANNING_TEMPLATES


def test_create_prompt_templates_structured_mode_simplifies_planning():
    CodeActAgent = _import_codact_agent()
    dummy = DummySelf(use_structured_outputs_internally=True)

    templates = CodeActAgent._create_prompt_templates(dummy)
    planning = templates["planning"]

    # Structured outputs => simplified planning strings per codact_prompts
    assert planning["initial_plan"].startswith("Think step by step")
    assert "Facts survey" not in planning["initial_plan"]
    assert "Review the task and history" in planning["update_plan_pre_messages"]
    assert "Based on the above" in planning["update_plan_post_messages"]


def test_final_answer_extension_included():
    CodeActAgent = _import_codact_agent()
    dummy = DummySelf(use_structured_outputs_internally=False)

    templates = CodeActAgent._create_prompt_templates(dummy)
    final_answer = templates["final_answer"]
    assert "pre_messages" in final_answer and "post_messages" in final_answer
    # Check formatting requirements hint exists
    assert "Your answer should be well-structured" in final_answer["post_messages"]


def test_managed_agent_templates_included():
    CodeActAgent = _import_codact_agent()
    dummy = DummySelf(use_structured_outputs_internally=False)

    templates = CodeActAgent._create_prompt_templates(dummy)
    managed = templates["managed_agent"]
    assert "task" in managed and "report" in managed
    assert "You're a helpful code execution agent named" in managed["task"]