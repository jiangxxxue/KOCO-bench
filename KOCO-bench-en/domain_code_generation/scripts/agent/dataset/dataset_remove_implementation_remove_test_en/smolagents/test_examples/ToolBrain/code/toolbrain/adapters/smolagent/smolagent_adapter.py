"""
SmolAgent adapter for ToolBrain.

This module provides the adapter implementation for smolagents CodeAgent.
"""

try: 
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except (ImportError, NotImplementedError):
    # ImportError: unsloth not installed
    # NotImplementedError: unsupported GPU (e.g., no NVIDIA/Intel GPU)
    FastLanguageModel = None
    UNSLOTH_AVAILABLE = False

from smolagents.models import get_clean_message_list, tool_role_conversions
from typing import List, Any, Tuple
from peft import get_peft_model, LoraConfig
from smolagents import CodeAgent, TransformersModel
import torch
import re
import logging

from ..base_adapter import BaseAgentAdapter
from ...core_types import Trace, Turn, ParsedCompletion, ChatSegment
from ...models import UnslothModel


class SmolAgentAdapter(BaseAgentAdapter):
    """Adapter for smolagents CodeAgent using a local TransformersModel."""

    def __init__(self, agent: CodeAgent, config):
        """
        Initialize the SmolAgentAdapter.

        Args:
            agent: A smolagents CodeAgent instance configured with a TransformersModel.
            config: general config
        """
        if not isinstance(agent, CodeAgent):
            raise TypeError(f"Expected CodeAgent instance, got {type(agent)}")
        # 允许“等效”的本地模型：具备 tokenizer.apply_chat_template 与 model 属性
        model = getattr(agent, "model", None)
        has_equivalent_local_model = (
            model is not None
            and hasattr(model, "tokenizer")
            and callable(getattr(model.tokenizer, "apply_chat_template", None))
            and hasattr(model, "model")
        )
        if not (isinstance(model, TransformersModel) or has_equivalent_local_model):
            raise TypeError(
                "Training is only supported for agents using a local smolagents.TransformersModel "
                "or an equivalent local model exposing 'tokenizer.apply_chat_template' and 'model' attributes."
            )

        self.agent = agent
        self.config = config
        self._set_lora_finetuning()

    def get_trainable_model(self) -> TransformersModel:
        """Returns the agent's underlying TransformersModel."""
        return self.agent.model

    def get_tools(self) -> List[str]:
        """Returns the list of tool names available in the agent."""
        return [tool for tool in self.agent.tools]
    
    def get_callable_tools(self) -> List[Any]:
        """
        Returns the list of actual tool callables available in the agent.
        
        Returns:
            List of tool callables from self.agent.tools.
        """
        return list(self.agent.tools.values())

    def run(self, query: str) -> Tuple[Trace, Any, Any]:
        """
        Executes the agent and then extracts a structured, high-fidelity trace
        from the agent's memory.

        Returns:
            tuple: (structured_trace, rl_input, raw_memory_steps)
                - structured_trace: Trace (List[Turn]) - processed trace for standard use
                - rl_input: Any - input prepared for RL training
                - raw_memory_steps: List[Any] - raw agent memory steps for advanced analysis
        """
        # TODO: Implement run method
        raise NotImplementedError("SmolAgentAdapter.run method needs to be implemented")

    def _extract_trace_from_memory(self) -> Trace:
        """
        Parses the agent's internal memory into our standardized Trace format.
        This version leverages pre-parsed fields from ActionStep where possible
        and parses the rest from the raw model_output.

        Also generates formatted conversation text using smolagents utilities
        for consistent formatting with training data.
        """
        # TODO: Implement _extract_trace_from_memory method
        raise NotImplementedError("_extract_trace_from_memory method needs to be implemented")

    def _build_input_for_rl_from_memory(self) -> Any:
        """
        Parses the agent's internal memory, build input for RL learning.
        Be robust to different message shapes (dicts, ChatMessage, etc.).
        """
        def _normalize_simple_messages(msgs: list) -> list:
            """Fallback normalization to [{'role': str, 'content': str}]"""
            norm: list[dict] = []
            for m in msgs or []:
                if isinstance(m, dict):
                    role = str(m.get("role", "user"))
                    content = m.get("content", "")
                else:
                    role = str(getattr(m, "role", "user"))
                    content = getattr(m, "content", "")
                # content may be list[{'type':'text','text':...}] or list[str]
                if isinstance(content, list):
                    pieces = []
                    for c in content:
                        if isinstance(c, dict) and "text" in c:
                            pieces.append(str(c["text"]))
                        else:
                            pieces.append(str(c))
                    content = " ".join(pieces)
                else:
                    content = str(content)
                norm.append({"role": role, "content": content})
            return norm

        # First pass: render full conversation text
        raw_messages_1 = self.agent.write_memory_to_messages()
        try:
            cleaned_messages_1 = get_clean_message_list(
                raw_messages_1,
                role_conversions=tool_role_conversions,
                flatten_messages_as_text=True,
            )
        except Exception:
            cleaned_messages_1 = _normalize_simple_messages(raw_messages_1)

        out_text = self.agent.model.tokenizer.apply_chat_template(
            cleaned_messages_1, add_generation_prompt=True, tokenize=False
        )

        # Second pass: build segments using assistant-only labeling
        raw_messages_2 = self.agent.write_memory_to_messages()
        try:
            cleaned_messages_2 = get_clean_message_list(
                raw_messages_2,
                flatten_messages_as_text=True,
            )
        except Exception:
            cleaned_messages_2 = _normalize_simple_messages(raw_messages_2)

        segments = self._segment_text_with_assistant(out_text, cleaned_messages_2)
        return segments

    def _segment_text_with_assistant(
        self, full_text: str, messages: list
    ) -> list[dict]:
        """
        Split full_text into contiguous segments, labeling only assistant messages.
        Ensures the sum of all segments equals the original text.

        Args:
            full_text: str, rendered chat text
            messages: list of {'role': str, 'content': str}

        Returns:
            List of segments [{'role': 'assistant' or 'other', 'start': int, 'end': int, 'text': str}]
        """
        # TODO: Implement _segment_text_with_assistant method
        raise NotImplementedError("_segment_text_with_assistant method needs to be implemented")

    def _parse_missing_parts(self, model_output: str) -> dict:
        """
        A helper to parse thought and a cleaned final_answer from the raw model output.
        """
        # TODO: Implement _parse_missing_parts method
        raise NotImplementedError("_parse_missing_parts method needs to be implemented")

    def _set_lora_finetuning(self):
        lora_config = self.config.get("lora_config", None)
        if lora_config:
            # Convert dict to LoraConfig object if needed
            if isinstance(lora_config, dict):
                lora_config = LoraConfig(**lora_config)
            
            is_unsloth_model = (
                UNSLOTH_AVAILABLE
                and hasattr(self.agent.model, "model")
                and isinstance(self.agent.model, UnslothModel)
            )

            if is_unsloth_model:
                unsloth_params = {
                    "use_gradient_checkpointing": self.config.get("gradient_checkpointing", "unsloth"),
                    "max_seq_length": self.config.get("max_seq_length", 4096),
                    "use_rslora": self.config.get("use_rslora", True),
                    "use_dora": self.config.get("use_dora", False),
                }
                
                lora_params = {
                    "r": lora_config.r,
                    "lora_alpha": lora_config.lora_alpha,
                    "target_modules": lora_config.target_modules,
                    "lora_dropout": lora_config.lora_dropout,
                    "bias": lora_config.bias,
                    "task_type": lora_config.task_type,
                }
                
                # Allow additional unsloth-specific overrides
                unsloth_overrides = self.config.get("unsloth_config_overrides", {})
                unsloth_params.update(unsloth_overrides)
                
                # Allow LoRA parameter overrides
                lora_overrides = self.config.get("lora_config_overrides", {})
                lora_params.update(lora_overrides)
                
                # Combine for FastLanguageModel.get_peft_model()
                final_config = {**unsloth_params, **lora_params}
                
                self.agent.model.model = FastLanguageModel.get_peft_model(
                    self.agent.model.model,
                    **final_config
                )
            else:
                # Base LoRA parameters
                base_lora_params = {
                    "r": lora_config.r,
                    "lora_alpha": lora_config.lora_alpha,
                    "target_modules": lora_config.target_modules,
                    "lora_dropout": lora_config.lora_dropout,
                    "bias": lora_config.bias,
                    "task_type": lora_config.task_type,
                }
                # Apply LoRA overrides
                lora_overrides = self.config.get("lora_config_overrides", {})
                base_lora_params.update(lora_overrides)
                
                enhanced_lora_config = LoraConfig(**base_lora_params)
                
                # Apply enhanced config
                hf_model = get_peft_model(self.agent.model.model, enhanced_lora_config)
                self.agent.model.model = hf_model
                
            total_params = sum(p.numel() for p in self.agent.model.model.parameters())
            trainable_params = sum(
                p.numel()
                for p in self.agent.model.model.parameters()
                if p.requires_grad
            )
            percentage = (
                100 * trainable_params / total_params if total_params > 0 else 0
            )
            logging.info(
                f"📊 LoRA applied: {trainable_params:,} / {total_params:,} params trainable ({percentage:.2f}%)"
            )