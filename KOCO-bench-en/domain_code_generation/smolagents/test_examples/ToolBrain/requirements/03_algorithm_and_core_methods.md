# ToolBrain Core Algorithm Function Descriptions (13 Functions)

# FUNCTION: factory.create_agent

## Function Overview
Builds a standard code agent execution engine based on smolagents, assembling optimized models with tool collections and ensuring chat template availability.

## Function Signature
def create_agent(model_id: str, tools: List[Any], *, use_unsloth: Optional[bool] = None, max_seq_length: int = 4096, max_new_tokens: int = 512, load_in_4bit: bool = True, max_steps: int = 10, **model_kwargs) -> CodeAgent

## Input Parameters
- `model_id`: (str) HuggingFace model ID
- `tools`: (List[Any]) Tool functions for code agent use
- Other parameters: Model capacity and optimization options, see signature for details

## Detailed Description
Creates a complete code agent instance, first building an optimized model that supports both standard Transformers models and efficient Unsloth models. Checks and ensures the model's chat template is available, automatically filling in a generic template if missing. Assembles the tool collection with the model into a code agent, returning a ready-to-use agent instance. This function supports various model optimization options, including 4-bit quantization, sequence length limits, and maximum generation token configuration.

## Output
- Fully assembled code agent instance

## Function Implementation
code/toolbrain/factory.py:line 87-168

## Test Code
code/test_code/test_factory_create_agent.py

---

# FUNCTION: SmolAgentAdapter.run

## Function Overview
Executes a single inference run of the smolagents code agent and extracts high-fidelity traces, reinforcement learning inputs, and raw memory step copies from agent memory.

## Function Signature
def run(query: str) -> Tuple[Trace, Any, Any]

## Input Parameters
- `query`: (str) User query

## Detailed Description
Executes a single agent inference process, wrapping agent execution in inference mode for improved efficiency. Extracts standardized trace data from the agent's memory steps, containing the complete interaction process. Constructs input format suitable for reinforcement learning, facilitating subsequent training and analysis. Simultaneously preserves copies of raw memory steps, supporting advanced analysis and debugging. The entire process ensures trace data integrity and consistency, providing high-quality data for training.

## Output
- Tuple containing trace, reinforcement learning input, and raw memory steps

## Function Implementation
code/toolbrain/adapters/smolagent/smolagent_adapter.py:line 77-188

## Test Code
code/test_code/test_smolagent_adapter_parse_and_segment.py

---

# FUNCTION: SmolAgentAdapter._extract_trace_from_memory

## Function Overview
Parses smolagents memory steps into a standardized turn list (trace) and attempts to generate formatted conversations.

## Function Signature
def _extract_trace_from_memory() -> Trace

## Input Parameters
- None (uses internal agent memory)

## Detailed Description
Extracts and parses interaction steps from agent memory, building standardized trace data structures. Utilizes framework-provided message processing functionality to generate formatted conversation representations. Performs detailed parsing of each action step, extracting model outputs, observations, action outputs, and code actions. When detecting a final answer action, extracts answer content and cleans tool code. Uses auxiliary parsing methods to handle missing thought processes and final answer fragments, ultimately combining them into complete turn data.

## Output
- Standardized trace (turn list)

## Function Implementation
code/toolbrain/adapters/smolagent/smolagent_adapter.py:line 190-316

## Test Code
code/test_code/test_smolagent_adapter_parse_and_segment.py

---

# FUNCTION: SmolAgentAdapter._parse_missing_parts

## Function Overview
Parses thought processes and final answer paragraphs from raw model output text, extracting structured fields.

## Function Signature
def _parse_missing_parts(model_output: str) -> dict

## Input Parameters
- `model_output`: (str) Raw model output text

## Detailed Description
Analyzes raw model output text, identifying and extracting thought process and final answer sections. Uses regular expressions to match specific paragraph markers, locating thought content and final answers. For answers containing numbers, extracts the first number as the primary answer; for non-numeric answers, preserves complete text content. This method ensures extraction of key information from unstructured output, providing structured data for trace construction.

## Output
- Dictionary containing thought and final answer

## Function Implementation
code/toolbrain/adapters/smolagent/smolagent_adapter.py:line 424-454

## Test Code
code/test_code/test_smolagent_adapter_parse_and_segment.py

---

# FUNCTION: SmolAgentAdapter._segment_text_with_assistant

## Function Overview
Segments complete conversation text into fragments by messages, annotating assistant role fragments and marking others as other roles.

## Function Signature
def _segment_text_with_assistant(full_text: str, messages: list) -> list[dict]

## Input Parameters
- `full_text`: (str) Rendered complete conversation text
- `messages`: (list[dict]) Message list with roles and content

## Detailed Description
Processes complete conversation text, segmenting it into multiple fragments based on the message list. Iterates through the message list, locating assistant role positions in the complete text, annotating each fragment with the corresponding role type. Non-assistant role fragments are uniformly annotated as other types. After processing, validates that all fragments concatenated match the original complete text, ensuring data integrity. This method provides precise role annotation for training data.

## Output
- Fragment list containing roles and text

## Function Implementation
code/toolbrain/adapters/smolagent/smolagent_adapter.py:line 375-422

## Test Code
code/test_code/test_smolagent_adapter_parse_and_segment.py

---

# FUNCTION: Brain._get_adapter_for_agent

## Function Overview
Automatically selects the appropriate adapter instance based on agent type, supporting seamless integration of multiple agent frameworks.

## Function Signature
def _get_adapter_for_agent(agent_instance: Any, trainable_model: Optional[Any]) -> BaseAgentAdapter

## Input Parameters
- `agent_instance`: (Any) Passed agent instance
- `trainable_model`: (Any|None) Trainable model handle

## Detailed Description
Implements intelligent adapter selection mechanism, automatically matching the most suitable adapter based on the passed agent instance type. For LangChain framework's compiled state graphs, returns the corresponding LangChain adapter instance. For smolagents framework's code agents, returns the dedicated SmolAgent adapter instance. When encountering unsupported agent types, raises a type error exception. This mechanism ensures agents from different frameworks can seamlessly integrate into unified training and inference workflows.

## Output
- Base agent adapter subclass instance

## Function Implementation
code/toolbrain/brain.py:line 270-284

## Test Code
code/test_code/test_brain_and_adapter.py

---

# FUNCTION: Brain.is_agent_supported

## Function Overview
Determines whether the passed agent is a type supported by ToolBrain, currently primarily supporting smolagents code agents.

## Function Signature
def is_agent_supported(agent: Any) -> bool

## Input Parameters
- `agent`: (Any) Object being detected

## Detailed Description
Implements agent type detection mechanism, verifying whether the passed object is a supported agent type. Performs type identification based on smolagents framework's code agent, checking object class inheritance relationships and interface implementations. Returns a boolean value indicating whether the agent is supported by the current system. This function provides users with quick compatibility checks, avoiding runtime errors from using unsupported agent types.

## Output
- Boolean value indicating whether the agent is supported

## Function Implementation
code/toolbrain/brain.py:line 465-487

## Test Code
code/test_code/test_brain_and_adapter.py

---

# FUNCTION: Brain.generate_training_examples

## Function Overview
Uses large language models combined with tool cards to generate training samples, with optional self-ranking functionality.

## Function Signature
def generate_training_examples(task_description: str | None = None, num_examples: int = 5, min_tool_calls: int = 1, max_words: int = 80, guidance_example: str | None = None, external_model: Any = None, external_tools: List[str] = None, self_rank: bool = False) -> List[str]

## Input Parameters
- `task_description`: (str|None) Task description
- `num_examples`: (int) Number of samples
- `min_tool_calls`: (int) Minimum number of different tools each sample must use
- `max_words`: (int) Maximum word count limit
- `guidance_example`: (str|None) Few-shot guidance
- `external_model`: (Any|None) External model
- `external_tools`: (List[str]|None) External tool list
- `self_rank`: (bool) Whether to use model ranking for samples

## Detailed Description
Generates high-quality training samples, first performing strict validation of language models and tools to ensure compatibility and availability. Generates standardized tool card descriptions based on tool collections, constructing complete prompts containing task descriptions, constraint conditions, and example guidance. Uses language models to generate training samples meeting requirements, supporting self-ranking functionality to evaluate and rank sample quality. The entire process ensures generated samples meet specific requirements for tool usage training, including necessary parameter values and minimum tool call counts.

## Output
- List of generated training samples

## Function Implementation
code/toolbrain/brain.py:line 488-560

## Test Code
code/test_code/test_brain_and_adapter.py

---

# FUNCTION: prompt._extract_tool_meta

## Function Overview
Extracts unified tool metadata from smolagents tool instances, including name, description, parameters, return values, and examples.

## Function Signature
def _extract_tool_meta(tool: Tool) -> Dict[str, Any]

## Input Parameters
- `tool`: (Tool) smolagents tool object

## Detailed Description
Extracts standardized metadata information from tool objects, first validating tool type to ensure compatibility. Prioritizes extracting parameter definitions from tool specifications, including function parameters and property parameters; when specifications are unavailable, falls back to input definitions and supports string dictionary parsing. Parses return value descriptions and usage examples from tool docstrings. Integrates all extracted information into a unified metadata dictionary, providing standardized data for tool card generation and training prompt construction.

## Output
- Unified metadata dictionary

## Function Implementation
code/toolbrain/prompt.py:line 9-125

## Test Code
code/test_code/test_extract_tool_meta.py

---

# FUNCTION: prompt.tools_to_card

## Function Overview
Batch renders cards for multiple tools, including headers, numbering, and separators, generating complete tool description documentation.

## Function Signature
def tools_to_card(tools: Sequence[Any]) -> str

## Input Parameters
- `tools`: (Sequence[Tool]) Tool sequence

## Detailed Description
Generates standardized card description documentation for tool collections, first adding header information containing tool count. Iterates through the tool sequence, calling individual card rendering functions for each tool to generate detailed tool descriptions. Assigns unique numbers to each tool for easy reference and identification. Adds separators between tool cards to improve document readability and structure. Finally returns complete tool card paragraphs, directly usable for training prompts and documentation generation.

## Output
- Batch tool card paragraphs

## Function Implementation
code/toolbrain/prompt.py:line 197-224

## Test Code
code/test_code/test_tools_to_card.py

---

# FUNCTION: prompt.validate_model

## Function Overview
Validates whether externally passed large language model objects meet minimum usable interface constraints.

## Function Signature
def validate_model(model: Any) -> None

## Input Parameters
- `model`: (Any) Model instance needed for inference/training

## Detailed Description
Performs strict interface validation on model objects, ensuring they meet system's basic requirements. Checks whether the model object is empty; empty objects will raise a runtime error. Verifies whether the model contains a callable generation method; missing this method will raise a not implemented error. Uses reflection mechanism to check generation method parameter signature, ensuring at least one positional parameter is included. Any validation failure will raise corresponding exceptions, providing users with clear error information.

## Output
- None (exception indicates failure)

## Function Implementation
code/toolbrain/prompt.py:line 228-261

## Test Code
code/test_code/test_validate_model.py

---

# FUNCTION: prompt.validate_tools

## Function Overview
Performs strict validation on tool collections, ensuring consistency with smolagents tool constraints.

## Function Signature
def validate_tools(tools: Sequence[Tool]) -> None

## Input Parameters
- `tools`: (Sequence[Tool]) Tool sequence

## Detailed Description
Performs comprehensive validation on tool collections, first checking input type, rejecting string, byte types, and non-sequence objects. Empty sequences will raise a value error exception. Performs detailed checks on each tool item, ensuring it is a smolagents tool type, callable object, and has a valid name. Checks tool name uniqueness; duplicate names ignoring case will raise a value error exception. The entire validation process ensures tool collections meet system's strict constraint requirements.

## Output
- None (exception indicates failure)

## Function Implementation
code/toolbrain/prompt.py:line 264-296

## Test Code
code/test_code/test_validate_tools.py

---

# FUNCTION: prompt.build_prompt_to_generate_training_examples

## Function Overview
Assembles prompt text for generating training samples, merging task descriptions, tool cards, and constraint explanations.

## Function Signature
def build_prompt_to_generate_training_examples(tools_description: str, task_description: str | None = None, min_tool_calls: int = 1, max_words: int = 80, guidance_example: str | None = None) -> str

## Input Parameters
- `tools_description`: (str) Tool description section
- `task_description`: (str|None) Task brief
- `min_tool_calls`: (int) Minimum number of different tools each sample must involve
- `max_words`: (int) Maximum allowed word count for generated queries
- `guidance_example`: (str|None) Few-shot style guidance

## Detailed Description
Constructs complete training sample generation prompts, including guidance sections with role definitions and goal explanations. Adds task descriptions and few-shot guidance examples as needed, providing specific generation guidance. Explicitly specifies generation constraint conditions, including output queries only, containing necessary parameter values, using at least the specified number of tools, returning results starting with standard answer markers. Applies minimum value limits to maximum word count, ensuring validity of generated content. Finally returns structured prompt text, directly usable for language model training sample generation.

## Output
- Training example generation prompt text

## Function Implementation
code/toolbrain/prompt.py:line 299-353

## Test Code
code/test_code/test_build_prompt_to_generate_training_examples.py

---
