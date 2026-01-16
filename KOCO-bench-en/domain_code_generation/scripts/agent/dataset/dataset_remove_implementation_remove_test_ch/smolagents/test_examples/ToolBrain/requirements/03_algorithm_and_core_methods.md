# ToolBrain算法核心函数描述共13个

# FUNCTION: factory.create_agent

## 功能概述
构建基于 smolagents 的标准代码代理执行引擎，装配优化模型与工具集合，并确保对话模板可用。

## 函数签名
def create_agent(model_id: str, tools: List[Any], *, use_unsloth: Optional[bool] = None, max_seq_length: int = 4096, max_new_tokens: int = 512, load_in_4bit: bool = True, max_steps: int = 10, **model_kwargs) -> CodeAgent

## 输入参数
- `model_id`: (str) HuggingFace 模型ID
- `tools`: (List[Any]) 供代码代理使用的工具函数
- 其余参数：模型容量与优化选项，详见签名

## 详细描述
创建一个完整的代码代理实例，首先构建优化模型，支持标准Transformers模型和高效Unsloth模型。检查并确保模型的对话模板可用，如果缺少则自动填充通用模板。将工具集合与模型装配到代码代理中，返回一个可直接使用的代理实例。该函数支持多种模型优化选项，包括4位量化、序列长度限制和最大生成令牌数等配置。

## 输出
- 已装配完成的代码代理实例

## 函数实现
code/toolbrain/factory.py:line 87-168

## 测试代码
code/test_code/test_factory_create_agent.py

---

# FUNCTION: SmolAgentAdapter.run

## 功能概述
执行 smolagents 代码代理一次推理，并从代理记忆中抽取高保真轨迹、强化学习输入与原始记忆步骤副本。

## 函数签名
def run(query: str) -> Tuple[Trace, Any, Any]

## 输入参数
- `query`: (str) 用户查询

## 详细描述
执行单次代理推理过程，使用推理模式包装代理执行以提高效率。从代理的记忆步骤中提取标准化轨迹数据，包含完整的交互过程。构建适用于强化学习的输入格式，便于后续训练和分析。同时保留原始记忆步骤的副本，支持高级分析和调试。整个过程确保轨迹数据的完整性和一致性，为训练提供高质量数据。

## 输出
- 包含轨迹、强化学习输入和原始记忆步骤的元组

## 函数实现
code/toolbrain/adapters/smolagent/smolagent_adapter.py:line 77-188

## 测试代码
code/test_code/test_smolagent_adapter_extract_trace.py

---

# FUNCTION: SmolAgentAdapter._extract_trace_from_memory

## 功能概述
将 smolagents 的记忆步骤解析为标准化轮次列表（轨迹），并尝试生成格式化对话。

## 函数签名
def _extract_trace_from_memory() -> Trace

## 输入参数
- 无（使用内部代理记忆）

## 详细描述
从代理的记忆中提取并解析交互步骤，构建标准化的轨迹数据结构。利用框架提供的消息处理功能，生成格式化的对话表示。对每个动作步骤进行详细解析，提取模型输出、观察结果、动作输出和代码动作。当检测到最终答案动作时，提取答案内容并清理工具代码。使用辅助解析方法处理缺失的思考过程和最终答案片段，最终组合成完整的轮次数据。

## 输出
- 标准化轨迹（轮次列表）

## 函数实现
code/toolbrain/adapters/smolagent/smolagent_adapter.py:line 190-316

## 测试代码
code/test_code/test_smolagent_adapter_extract_trace.py

---

# FUNCTION: SmolAgentAdapter._parse_missing_parts

## 功能概述
从原始模型输出文本中解析思考过程与最终答案段落，提取结构化字段。

## 函数签名
def _parse_missing_parts(model_output: str) -> dict

## 输入参数
- `model_output`: (str) 模型原始输出文本

## 详细描述
分析模型输出的原始文本，识别并提取思考过程和最终答案部分。使用正则表达式匹配特定的段落标记，定位思考内容和最终答案。对于包含数字的答案，提取首个数字作为主要答案；对于非数字答案，保留完整的文本内容。该方法确保从非结构化输出中提取关键信息，为轨迹构建提供结构化数据。

## 输出
- 包含思考和最终答案的字典

## 函数实现
code/toolbrain/adapters/smolagent/smolagent_adapter.py:line 424-454

## 测试代码
code/test_code/test_smolagent_adapter_parse_missing_parts.py

---

# FUNCTION: SmolAgentAdapter._segment_text_with_assistant

## 功能概述
将完整对话文本按消息切分成片段，标注助手角色片段，其余标注为其他角色。

## 函数签名
def _segment_text_with_assistant(full_text: str, messages: list) -> list[dict]

## 输入参数
- `full_text`: (str) 渲染后的完整对话文本
- `messages`: (list[dict]) 角色与内容的消息列表

## 详细描述
处理完整的对话文本，根据消息列表将其切分为多个片段。遍历消息列表，定位助手角色在完整文本中的位置，为每个片段标注相应的角色类型。非助手角色的片段统一标注为其他类型。在处理完成后，验证所有片段拼接后的文本与原始完整文本的一致性，确保数据完整性。该方法为训练数据提供精确的角色标注。

## 输出
- 包含角色和文本的片段列表

## 函数实现
code/toolbrain/adapters/smolagent/smolagent_adapter.py:line 375-422

## 测试代码
code/test_code/test_smolagent_adapter_segment.py

---

# FUNCTION: Brain._get_adapter_for_agent

## 功能概述
根据代理类型自动选择合适的适配器实例，支持多种代理框架的无缝集成。

## 函数签名
def _get_adapter_for_agent(agent_instance: Any, trainable_model: Optional[Any]) -> BaseAgentAdapter

## 输入参数
- `agent_instance`: (Any) 传入的智能体实例
- `trainable_model`: (Any|None) 可训练模型句柄

## 详细描述
实现智能适配器选择机制，根据传入的代理实例类型自动匹配最适合的适配器。对于LangChain框架的编译状态图，返回对应的LangChain适配器实例。对于smolagents框架的代码代理，返回专用的SmolAgent适配器实例。当遇到不支持的代理类型时，抛出类型错误异常。该机制确保不同框架的代理都能无缝集成到统一的训练和推理流程中。

## 输出
- 基础代理适配器子类实例

## 函数实现
code/toolbrain/brain.py:line 270-284

## 测试代码
code/test_code/test_brain_get_adapter_for_agent.py

---

# FUNCTION: Brain.is_agent_supported

## 功能概述
判断传入代理是否为ToolBrain支持的类型，当前主要支持smolagents代码代理。

## 函数签名
def is_agent_supported(agent: Any) -> bool

## 输入参数
- `agent`: (Any) 被检测对象

## 详细描述
实现代理类型检测机制，验证传入的对象是否为支持的代理类型。基于smolagents框架的代码代理进行类型识别，检查对象的类继承关系和接口实现。返回布尔值指示该代理是否受当前系统支持。该函数为用户提供快速的兼容性检查，避免使用不支持的代理类型导致的运行时错误。

## 输出
- 布尔值，表示是否支持该代理

## 函数实现
code/toolbrain/brain.py:line 465-487

## 测试代码
code/test_code/test_brain_is_agent_supported.py

---

# FUNCTION: Brain.generate_training_examples

## 功能概述
使用大语言模型结合工具卡片生成训练样本，可选择自评排序功能。

## 函数签名
def generate_training_examples(task_description: str | None = None, num_examples: int = 5, min_tool_calls: int = 1, max_words: int = 80, guidance_example: str | None = None, external_model: Any = None, external_tools: List[str] = None, self_rank: bool = False) -> List[str]

## 输入参数
- `task_description`: (str|None) 任务描述
- `num_examples`: (int) 样本数量
- `min_tool_calls`: (int) 每个样本至少使用的不同工具数
- `max_words`: (int) 最大词数上限
- `guidance_example`: (str|None) 少样本引导
- `external_model`: (Any|None) 外部模型
- `external_tools`: (List[str]|None) 外部工具列表
- `self_rank`: (bool) 是否使用模型排序样本

## 详细描述
生成高质量的训练样本，首先对语言模型和工具进行严格校验，确保兼容性和可用性。基于工具集合生成标准化的工具卡片描述，构建包含任务描述、约束条件和示例引导的完整提示。使用语言模型生成符合要求的训练样本，支持自评排序功能对样本质量进行评估和排序。整个过程确保生成的样本符合工具使用训练的特定要求，包含必要的参数值和最少工具调用次数。

## 输出
- 生成的训练样本列表

## 函数实现
code/toolbrain/brain.py:line 488-560

## 测试代码
code/test_code/test_brain_generate_training_examples.py

---

# FUNCTION: prompt._extract_tool_meta

## 功能概述
从smolagents工具实例中抽取统一的工具元数据，包括名称、描述、参数、返回值和示例等。

## 函数签名
def _extract_tool_meta(tool: Tool) -> Dict[str, Any]

## 输入参数
- `tool`: (Tool) smolagents工具对象

## 详细描述
从工具对象中提取标准化的元数据信息，首先验证工具类型确保兼容性。优先从工具规范中提取参数定义，包括函数参数和属性参数；当规范不可用时，回退到输入定义并支持字符串字典解析。从工具的文档字符串中解析返回值描述和使用示例。整合所有提取的信息为统一的元数据字典，为工具卡片生成和训练提示构建提供标准化数据。

## 输出
- 统一元数据字典

## 函数实现
code/toolbrain/prompt.py:line 9-125

## 测试代码
code/test_code/test_extract_tool_meta.py

---

# FUNCTION: prompt.tools_to_card

## 功能概述
批量渲染多个工具的卡片，包含页眉、编号和分隔线，生成完整的工具描述文档。

## 函数签名
def tools_to_card(tools: Sequence[Any]) -> str

## 输入参数
- `tools`: (Sequence[Tool]) 工具序列

## 详细描述
为工具集合生成标准化的卡片描述文档，首先添加包含工具数量的页眉信息。遍历工具序列，为每个工具调用单独的卡片渲染函数，生成详细的工具描述。为每个工具分配唯一编号，便于引用和识别。在工具卡片之间添加分隔线，提高文档的可读性和结构化程度。最终返回完整的工具卡片段落，可直接用于训练提示和文档生成。

## 输出
- 批量工具卡片段落

## 函数实现
code/toolbrain/prompt.py:line 197-224

## 测试代码
code/test_code/test_tools_to_card.py

---

# FUNCTION: prompt.validate_model

## 功能概述
校验外部传入的大语言模型对象是否满足最小可用接口约束。

## 函数签名
def validate_model(model: Any) -> None

## 输入参数
- `model`: (Any) 需要用于推理/训练的模型实例

## 详细描述
对模型对象进行严格的接口验证，确保其满足系统的基本要求。检查模型对象是否为空，空对象将引发运行时错误。验证模型是否包含可调用的生成方法，缺少该方法将引发未实现错误。使用反射机制检查生成方法的参数签名，确保至少包含一个位置参数。任何验证失败都会抛出相应的异常，为用户提供明确的错误信息。

## 输出
- 无（异常即为失败）

## 函数实现
code/toolbrain/prompt.py:line 228-261

## 测试代码
code/test_code/test_validate_model.py

---

# FUNCTION: prompt.validate_tools

## 功能概述
对工具集合进行严格校验，确保与smolagents工具约束一致。

## 函数签名
def validate_tools(tools: Sequence[Tool]) -> None

## 输入参数
- `tools`: (Sequence[Tool]) 工具序列

## 详细描述
对工具集合进行全面验证，首先检查输入类型，拒绝字符串、字节类型和非序列对象。空序列将引发值错误异常。对每个工具项进行详细检查，确保其为smolagents工具类型、可调用对象且具有合法名称。检查工具名称的唯一性，忽略大小写的重复名称将引发值错误异常。整个验证过程确保工具集合符合系统的严格约束要求。

## 输出
- 无（异常即为失败）

## 函数实现
code/toolbrain/prompt.py:line 264-296

## 测试代码
code/test_code/test_validate_tools.py

---

# FUNCTION: prompt.build_prompt_to_generate_training_examples

## 功能概述
拼装生成训练样本的提示文本，合并任务描述、工具卡片与约束说明。

## 函数签名
def build_prompt_to_generate_training_examples(tools_description: str, task_description: str | None = None, min_tool_calls: int = 1, max_words: int = 80, guidance_example: str | None = None) -> str

## 输入参数
- `tools_description`: (str) 工具说明段
- `task_description`: (str|None) 任务简述
- `min_tool_calls`: (int) 每个样本必须涉及的最少不同工具数
- `max_words`: (int) 生成的查询允许的最多词数
- `guidance_example`: (str|None) 少样本风格引导

## 详细描述
构建完整的训练样本生成提示，包含角色定义和目标说明的引导部分。根据需要添加任务描述和少样本引导示例，提供具体的生成指导。明确指定生成约束条件，包括仅输出查询、包含必要参数值、至少使用指定数量的工具、返回以标准答案标记开头的结果。对最大词数进行最小值限制，确保生成内容的有效性。最终返回结构化的提示文本，可直接用于语言模型生成训练样本。

## 输出
- 训练示例生成提示文本

## 函数实现
code/toolbrain/prompt.py:line 299-353

## 测试代码
code/test_code/test_build_prompt_to_generate_training_examples.py
