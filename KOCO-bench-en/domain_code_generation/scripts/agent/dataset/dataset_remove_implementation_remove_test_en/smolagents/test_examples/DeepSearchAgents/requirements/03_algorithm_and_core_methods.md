# DeepSearchAgents Core Function Descriptions (11 Functions)

# FUNCTION: MultiModelRouter.generate_stream

## Function Overview
Selects the appropriate model based on message content and generates responses in a streaming manner, following smolagents v1.19.0's "external aggregation" pattern.

## Function Signature
def generate_stream(self, messages: List[Dict[str, Any]], **kwargs) -> Generator[ChatMessageStreamDelta, None, None]

## Input Parameters
- `messages`: List of conversation messages
- `**kwargs`: Additional parameters passed to the underlying model

## Detailed Description
Determines the currently used model through internal model selection logic: uses the orchestrator model for planning or final answers, and the search model for other cases. Uses an external streaming wrapper to aggregate the streaming output from the underlying model, maintaining stable token statistics and content concatenation. Returns error deltas during exception handling to ensure stability for upstream consumers.

## Output
- Generator: Progressively outputs ChatMessageStreamDelta increments

## Function Implementation
code/src/agents/base_agent.py:line 117-152

## Test Code
code/test_code/test_multi_model_router.py

---

# FUNCTION: BaseAgent.run

## Function Overview
Unified agent run entry point, supporting both streaming and non-streaming modes, compatible with smolagents' run and memory management, and returns structured run results in non-streaming mode.

## Function Signature
def run(self, user_input: str, stream: bool = False, images: List = None, reset: bool = True, additional_args: Dict[str, Any] = None, return_result: bool = False, **kwargs) -> Union[str, Generator, RunResult]

## Input Parameters
- `user_input`: User query
- `stream`: Whether to enable streaming output
- `images`: Image inputs (interface compatibility)
- `reset`: Whether to reset agent memory
- `additional_args`: Additional parameters
- `return_result`: Whether to return structured RunResult

## Detailed Description
In streaming mode, reduces redundant logging and passes parameters based on capability detection. In non-streaming mode, supports JSON structure extraction from final answer tool returns. Collects execution time, model ID, token usage information and generates RunResult. Returns structured error information in case of exceptions (when return_result is True).

## Output
- String/Generator/RunResult (depending on mode and return_result)

## Function Implementation
code/src/agents/base_agent.py:line 313-469

## Test Code
code/test_code/test_base_agent_run.py

---

# FUNCTION: AgentStepCallback._extract_token_stats

## Function Overview
Based on smolagents' TokenUsage interface, extracts and aggregates token usage statistics from memory steps.

## Function Signature
def _extract_token_stats(self, memory_step: MemoryStep, step_data: Dict[str, Any]) -> None

## Input Parameters
- `memory_step`: Memory step object
- `step_data`: Step structured data dictionary (output enhancement)

## Detailed Description
Prioritizes reading input/output token counts from the memory step's token usage information (can be object or dictionary format). Accumulates global statistics and attaches them to step data for UI display. Synchronously builds token statistics snapshots for each step, providing a foundation for subsequent overall statistics.

## Output
- Through step_data field enhancement (contains current step and cumulative token information)

## Function Implementation
code/src/agents/ui_common/agent_step_callback.py:line 192-279

## Test Code
code/test_code/test_agent_step_callback_token_stats.py

---

# FUNCTION: StreamAggregator.aggregate_stream

## Function Overview
Aggregates streaming content by increments, estimates tokens, and forwards delta objects as-is, conforming to smolagents v1.19.0's "external aggregation" design.

## Function Signature
def aggregate_stream(self, stream_generator: Generator[ChatMessageStreamDelta, None, None], track_tokens: bool = True) -> Generator[ChatMessageStreamDelta, None, None]

## Input Parameters
- `stream_generator`: Generator producing ChatMessageStreamDelta
- `track_tokens`: Whether to track token counts

## Detailed Description
Concatenates text content from each delta into a complete response. Optionally performs simple token estimation (counting by space-separated words). Produces error deltas during exceptions to ensure stable streaming consumption.

## Output
- Generator: Forwards original deltas

## Function Implementation
code/src/agents/stream_aggregator.py:line 32-61

## Test Code
code/test_code/test_stream_aggregator.py

---

# FUNCTION: ModelStreamWrapper.generate_stream

## Function Overview
Wraps the underlying model's streaming capability with a unified interface and implements compatibility and aggregation using an aggregator.

## Function Signature
def generate_stream(self, messages: list, **kwargs) -> Generator[ChatMessageStreamDelta, None, None]

## Input Parameters
- `messages`: Message list
- `**kwargs`: Additional model parameters

## Detailed Description
Performs fallback handling (non-streaming) for models without streaming generation capability. Aggregates and forwards output for models with streaming capability. Provides an encapsulation point for subsequent retrieval of aggregated content and token statistics.

## Output
- Generator: Forwards aggregated deltas

## Function Implementation
code/src/agents/stream_aggregator.py:line 114-145

## Test Code
code/test_code/test_stream_aggregator.py

---

# FUNCTION: EnhancedFinalAnswerTool.forward

## Function Overview
Uniformly formats final_answer, ensuring consistency of Markdown content and source lists, compatible with output from both agent modes.

## Function Signature
def forward(self, answer: Any) -> Any

## Input Parameters
- `answer`: Final answer (string or structured object)

## Detailed Description
If input is a JSON string or dictionary, standardizes it into a unified structure containing title, content, and sources. Supplements content with Markdown blocks for source information, unifying source display. Returns fallback structure during exceptions to avoid UI rendering failures.

## Output
- Standardized JSON object (containing title, content, sources, and format identifier)

## Function Implementation
code/src/tools/final_answer.py:line 45-86

## Test Code
code/test_code/test_final_answer_tool.py

---

# FUNCTION: CodeActAgent._create_prompt_templates

## Function Overview
Extends system prompt, planning, and final answer templates according to configuration while maintaining smolagents' prompt structure.

## Function Signature
def _create_prompt_templates(self) -> dict

## Input Parameters
- No explicit external input (reads configuration and base templates)

## Detailed Description
Prioritizes loading structured templates, falls back to base templates when unavailable. Injects current time, planning interval, and extended content (system prompt/planning/final answer/managed agents). In structured output mode, uses simplified planning templates to improve robustness.

## Output
- Dictionary of prompt template collections (for PromptTemplates construction)

## Function Implementation
code/src/agents/codact_agent.py:line 197-249

## Test Code
code/test_code/test_codact_prompt_templates.py

---

# FUNCTION: AgentRuntime._create_llm_model

## Function Overview
Creates LiteLLMModel instances with a unified interface, supporting environment variables/base URL and other configurations, with necessary robustness prompts.

## Function Signature
def _create_llm_model(self, model_id: str) -> Union[LiteLLMModel]

## Input Parameters
- `model_id`: Model ID

## Detailed Description
Cleans trailing slashes from base URL to avoid double-slash issues. Provides warnings when API keys are missing while allowing the process to continue. Uniformly sets temperature and key parameters, compatible with upper-layer calls.

## Output
- LiteLLMModel instance

## Function Implementation
code/src/agents/runtime.py:line 223-259

## Test Code
code/test_code/test_runtime_llm_model.py

---

# FUNCTION: process_planning_step

## Function Overview
Transforms smolagents' PlanningStep events into web frontend-friendly message sequences, including header/content/footer and separators.

## Function Signature
def process_planning_step(step_log: PlanningStep, step_number: int, session_id: Optional[str] = None, skip_model_outputs: bool = False, is_streaming: bool = False, planning_interval: Optional[int] = None) -> Generator[DSAgentRunMessage, None, None]

## Input Parameters
- `step_log`: Planning step log
- `step_number`: Step number
- `session_id`: Session ID
- `skip_model_outputs`: Whether to skip model outputs (streaming scenario)
- `is_streaming`: Whether in streaming mode
- `planning_interval`: Planning interval

## Detailed Description
Determines initial/updated planning type and generates header metadata. Cleans content (code block parsing) and attaches token and timing metadata. Generates footer and separator for frontend component layout. Provides initial streaming message context in streaming mode.

## Output
- DSAgentRunMessage generator (ASSISTANT role)

## Function Implementation
code/src/api/v2/web_ui.py:line 186-192

## Test Code
code/test_code/test_web_ui_processing.py

---

# FUNCTION: process_action_step

## Function Overview
Splits ActionStep events into thought fragments, tool badges, code/chat component messages, terminal logs, errors, and footers to meet Web UI's componentized presentation.

## Function Signature
def process_action_step(step_log: ActionStep, session_id: Optional[str] = None, skip_model_outputs: bool = False) -> Generator[DSAgentRunMessage, None, None]

## Input Parameters
- `step_log`: Action step log
- `session_id`: Session ID
- `skip_model_outputs`: Whether to skip model outputs (streaming scenario)

## Detailed Description
Generates header message (step title). Processes thought content and generates metadata for preview and complete fragments. Generates badge messages for each tool call; code interpreter generates code component messages, other tools go to chat component. Extracts execution logs and determines errors, attaching timing information. Generates footer and separator messages to complete structure.

## Output
- DSAgentRunMessage generator (ASSISTANT role)

## Function Implementation
code/src/api/v2/web_ui.py:line 395-740

## Test Code
code/test_code/test_web_ui_processing.py

---

# FUNCTION: process_final_answer_step

## Function Overview
Processes FinalAnswerStep events, supporting direct injection of structured JSON content as metadata to avoid raw JSON rendering and improve frontend display consistency.

## Function Signature
def process_final_answer_step(step_log: FinalAnswerStep, session_id: Optional[str] = None, step_number: Optional[int] = None) -> Generator[DSAgentRunMessage, None, None]

## Input Parameters
- `step_log`: Final answer step log
- `session_id`: Session ID
- `step_number`: Step number

## Detailed Description
Cleans final answer prefixes in text content and attempts JSON parsing. When parsing succeeds, writes title, content, and sources to metadata with empty content output; when parsing fails, falls back to original text format. Compatible with multiple types of final answers including images/audio/dictionaries.

## Output
- DSAgentRunMessage generator (ASSISTANT role)

## Function Implementation
code/src/api/v2/web_ui.py:line 743-894

## Test Code
code/test_code/test_web_ui_processing.py

---
