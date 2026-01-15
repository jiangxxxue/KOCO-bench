# DeepSearchAgents核心函数描述共11个

# FUNCTION:MultiModelRouter.generate_stream

## 功能概述
基于消息内容选择合适模型并以流式方式生成响应，遵循smolagents v1.19.0的"模型外部聚合"模式。

## 函数签名
def generate_stream(self, messages: List[Dict[str, Any]], **kwargs) -> Generator[ChatMessageStreamDelta, None, None]

## 输入参数
- `messages`: 对话消息列表
- `**kwargs`: 传递给底层模型的附加参数

## 详细描述
通过内部的模型选择逻辑确定当前使用的模型，规划或最终答案时使用编排器模型，其他情况使用搜索模型。使用外部的流式封装器对底层模型的流式输出进行聚合，维持稳定的令牌统计与内容串联。异常处理时返回错误增量，保证上游消费方稳定。

## 输出
- 生成器：逐步输出ChatMessageStreamDelta增量

## 函数实现
code/src/agents/base_agent.py:line 117-152

## 测试代码
code/test_code/test_multi_model_router.py

---

# FUNCTION:BaseAgent.run

## 功能概述
统一的智能体运行入口，支持流式与非流式模式，兼容smolagents的运行与内存管理，并在非流式模式返回结构化运行结果。

## 函数签名
def run(self, user_input: str, stream: bool = False, images: List = None, reset: bool = True, additional_args: Dict[str, Any] = None, return_result: bool = False, **kwargs) -> Union[str, Generator, RunResult]

## 输入参数
- `user_input`: 用户查询
- `stream`: 是否启用流式输出
- `images`: 图像输入（兼容接口）
- `reset`: 是否重置智能体内存
- `additional_args`: 额外参数
- `return_result`: 是否返回结构化RunResult

## 详细描述
流式模式下降低冗余日志并按能力探测传参。非流式模式支持最终答案工具返回的JSON结构抽取。统计执行时间、模型ID、令牌使用信息并生成RunResult。异常情况下返回结构化错误信息（当return_result为True）。

## 输出
- 字符串/生成器/RunResult（取决于模式与return_result）

## 函数实现
code/src/agents/base_agent.py:line 313-469

## 测试代码
code/test_code/test_base_agent_run.py

---

# FUNCTION:AgentStepCallback._extract_token_stats

## 功能概述
基于smolagents的TokenUsage接口，从记忆步骤中提取并汇总令牌使用统计。

## 函数签名
def _extract_token_stats(self, memory_step: MemoryStep, step_data: Dict[str, Any]) -> None

## 输入参数
- `memory_step`: 记忆步骤对象
- `step_data`: 步骤结构化数据字典（输出增强）

## 详细描述
优先从记忆步骤的令牌使用信息（可以是对象或字典格式）读取输入/输出令牌数。累加全局统计并附加到步骤数据中用于UI展示。同步构建每步的令牌统计快照，为后续总体统计提供基础。

## 输出
- 通过step_data的字段增强（包含当前步与累计令牌信息）

## 函数实现
code/src/agents/ui_common/agent_step_callback.py:line 192-279

## 测试代码
code/test_code/test_agent_step_callback_token_stats.py

---

# FUNCTION:StreamAggregator.aggregate_stream

## 功能概述
按增量聚合流式内容，估算令牌，并原样转发增量对象，符合smolagents v1.19.0"模型外聚合"的设计。

## 函数签名
def aggregate_stream(self, stream_generator: Generator[ChatMessageStreamDelta, None, None], track_tokens: bool = True) -> Generator[ChatMessageStreamDelta, None, None]

## 输入参数
- `stream_generator`: 产生ChatMessageStreamDelta的生成器
- `track_tokens`: 是否跟踪令牌计数

## 详细描述
将各增量的文本内容拼接为完整响应。可选进行简单的令牌估算（按空格分词计数）。异常时产生错误增量保障流式消费稳定。

## 输出
- 生成器：转发原始增量

## 函数实现
code/src/agents/stream_aggregator.py:line 32-61

## 测试代码
code/test_code/test_stream_aggregator.py

---

# FUNCTION:ModelStreamWrapper.generate_stream

## 功能概述
以统一接口封装底层模型的流式能力，并使用聚合器实现兼容与聚合。

## 函数签名
def generate_stream(self, messages: list, **kwargs) -> Generator[ChatMessageStreamDelta, None, None]

## 输入参数
- `messages`: 消息列表
- `**kwargs`: 模型附加参数

## 详细描述
对不具备流式生成能力的模型进行回退处理（非流式）。对具备流式能力的模型输出进行聚合并转发。为后续获取汇总内容与令牌统计提供封装点。

## 输出
- 生成器：转发聚合后的增量

## 函数实现
code/src/agents/stream_aggregator.py:line 114-145

## 测试代码
code/test_code/test_stream_aggregator.py

---

# FUNCTION:EnhancedFinalAnswerTool.forward

## 功能概述
对final_answer进行统一格式化，保证Markdown内容与来源列表一致，兼容两个智能体模式的输出。

## 函数签名
def forward(self, answer: Any) -> Any

## 输入参数
- `answer`: 最终答案（字符串或结构化对象）

## 详细描述
若输入为JSON字符串或字典，标准化为包含标题、内容、来源的统一结构。对内容补充来源信息的Markdown区块，统一来源展示。异常时返回回退结构避免UI渲染失败。

## 输出
- 标准化JSON对象（含标题、内容、来源与格式标识）

## 函数实现
code/src/tools/final_answer.py:line 45-86

## 测试代码
code/test_code/test_final_answer_tool.py

---

# FUNCTION:CodeActAgent._create_prompt_templates

## 功能概述
在保持smolagents提示结构的前提下，按配置扩展系统提示、规划与最终答案模板。

## 函数签名
def _create_prompt_templates(self) -> dict

## 输入参数
- 无显式外部输入（读取配置与基础模板）

## 详细描述
优先加载结构化模板，不可用时回退至基础模板。注入当前时间、规划间隔与扩展内容（系统提示/规划/最终答案/托管智能体）。在结构化输出模式下，使用简化规划模板提升鲁棒性。

## 输出
- 字典形式的提示模板集合（用于PromptTemplates构造）

## 函数实现
code/src/agents/codact_agent.py:line 197-249

## 测试代码
code/test_code/test_codact_prompt_templates.py

---

# FUNCTION:AgentRuntime._create_llm_model

## 功能概述
按统一接口创建LiteLLMModel实例，支持环境变量/基础URL等配置，并进行必要的健壮性提示。

## 函数签名
def _create_llm_model(self, model_id: str) -> Union[LiteLLMModel]

## 输入参数
- `model_id`: 模型ID

## 详细描述
清理基础URL尾斜杠，避免双斜杠问题。缺失API密钥时给出警告，保持流程继续。统一设置温度与关键参数，兼容上层调用。

## 输出
- LiteLLMModel实例

## 函数实现
code/src/agents/runtime.py:line 223-259

## 测试代码
code/test_code/test_runtime_llm_model.py

---

# FUNCTION:process_planning_step

## 功能概述
将smolagents的PlanningStep事件转化为Web前端友好的消息序列，包含头/内容/脚注与分隔。

## 函数签名
def process_planning_step(step_log: PlanningStep, step_number: int, session_id: Optional[str] = None, skip_model_outputs: bool = False, is_streaming: bool = False, planning_interval: Optional[int] = None) -> Generator[DSAgentRunMessage, None, None]

## 输入参数
- `step_log`: 规划步骤日志
- `step_number`: 步骤编号
- `session_id`: 会话ID
- `skip_model_outputs`: 是否跳过模型输出（流式场景）
- `is_streaming`: 是否处于流式模式
- `planning_interval`: 规划间隔

## 详细描述
首次/更新规划类型判别与头部元数据生成。内容清理（代码块解析）并附加令牌与耗时元数据。生成脚注与分隔用于前端组件布局。流式模式下提供初始流消息上下文。

## 输出
- DSAgentRunMessage生成器（ASSISTANT角色）

## 函数实现
code/src/api/v2/web_ui.py:line 186-192

## 测试代码
code/test_code/test_web_ui_processing.py

---

# FUNCTION:process_action_step

## 功能概述
将ActionStep事件拆分为思考片段、工具徽章、代码/聊天组件消息、终端日志、错误与脚注等消息，满足Web UI的组件化呈现。

## 函数签名
def process_action_step(step_log: ActionStep, session_id: Optional[str] = None, skip_model_outputs: bool = False) -> Generator[DSAgentRunMessage, None, None]

## 输入参数
- `step_log`: 动作步骤日志
- `session_id`: 会话ID
- `skip_model_outputs`: 是否跳过模型输出（流式场景）

## 详细描述
生成头部消息（步骤标题）。处理思考内容并生成预览与完整片段的元数据。逐个工具调用生成徽章消息；代码解释器生成代码组件消息，其他工具进聊天组件。提取执行日志并判断错误，附加耗时信息。生成脚注与分隔消息完成结构。

## 输出
- DSAgentRunMessage生成器（ASSISTANT角色）

## 函数实现
code/src/api/v2/web_ui.py:line 395-740

## 测试代码
code/test_code/test_web_ui_processing.py

---

# FUNCTION:process_final_answer_step

## 功能概述
处理FinalAnswerStep事件，支持结构化JSON内容的直接注入为元数据，避免原始JSON渲染，提升前端展示一致性。

## 函数签名
def process_final_answer_step(step_log: FinalAnswerStep, session_id: Optional[str] = None, step_number: Optional[int] = None) -> Generator[DSAgentRunMessage, None, None]

## 输入参数
- `step_log`: 最终答案步骤日志
- `session_id`: 会话ID
- `step_number`: 步骤编号

## 详细描述
对文本内容中的最终答案前缀进行清理与JSON解析尝试。解析成功时，将标题、内容与来源写入元数据并空内容输出；解析失败时回退到原始文本格式。兼容图像/音频/字典等多类型最终答案。

## 输出
- DSAgentRunMessage生成器（ASSISTANT角色）

## 函数实现
code/src/api/v2/web_ui.py:line 743-894

## 测试代码
code/test_code/test_web_ui_processing.py

---
