# SmolCC算法核心函数描述共17个

# FUNCTION: execute_tool_call

## 功能概述
执行工具调用并显示丰富的可视化反馈

## 函数签名
def execute_tool_call(tool_name: str, tool_arguments: Dict[str, Any]) -> Any

## 输入参数
- `tool_name`: (str) 要执行的工具名称
- `tool_arguments`: (Dict[str, Any]) 传递给工具的参数字典

## 详细描述
执行工具调用并记录执行时间。首先检查是否为最终答案或最终输出工具调用，然后显示工具调用信息。通过获取工具方法获取工具实例，执行工具并记录执行时间。结果处理支持多种输出格式：将普通结果转换为工具输出对象，根据内容类型自动选择适当的输出格式。对于非最终答案的工具调用，显示格式化输出和执行时间统计。

## 输出
- 工具执行结果，支持多种输出格式

## 函数实现
code/smolcc/agent.py:line 99-148

## 测试代码
code/test_code/test_agent_execute_tool_call.py

---

# FUNCTION: format_messages_for_llm

## 功能概述
装备LLM消息并显示思考状态

## 函数签名
def format_messages_for_llm(prompt: str) -> List[Dict[str, str]]

## 输入参数
- `prompt`: (str) 用户提示

## 详细描述
在准备LLM消息时显示思考状态旋转器。使用Rich的进度组件创建临时进度条，在LLM格式化操作期间提供视觉反馈。调用父类的格式化消息方法获取格式化后的消息列表，确保消息格式完全兼容。该方法在LLM处理时间较长时提供用户友好的等待指示。

## 输出
- 格式化后的LLM消息列表

## 函数实现
code/smolcc/agent.py:line 183-204

## 测试代码
code/test_code/test_agent_llm_run_log.py

---

# FUNCTION: get_system_prompt

## 功能概述
生成动态系统提示

## 函数签名
def get_system_prompt(cwd: Optional[str] = None) -> str

## 输入参数
- `cwd`: (str) 当前工作目录

## 详细描述
从系统消息模板文件读取基础系统提示，然后动态填充占位符：当前日期、工作目录、平台信息、模型名称。集成Git仓库状态检测：检查当前目录是否为Git仓库，如果是则获取Git状态信息（当前分支、主分支、状态、最近提交）。生成目录结构信息，忽略常见开发目录和文件模式。最终将动态生成的系统提示返回，确保提示包含最新的环境上下文信息。

## 输出
- 动态生成的系统提示字符串

## 函数实现
code/smolcc/system_prompt.py:line 127-165

## 测试代码
code/test_code/test_system_prompt_utils.py

---

# FUNCTION: convert_to_tool_output

## 功能概述
将普通结果转换为适当的工具输出对象

## 函数签名
def convert_to_tool_output(result: Any) -> ToolOutput

## 输入参数
- `result`: (Any) 要转换的结果

## 详细描述
将各种类型的工具执行结果转换为统一的工具输出对象。首先检查结果是否已经是工具输出实例，如果是则直接返回。对于字符串结果，自动检测是否为代码内容：检查是否包含换行符且行首有缩进或Python关键字。如果是代码则返回代码输出，否则返回文本输出。对于列表结果，检测是否为表格数据或文件信息列表，分别返回表格输出或文件列表输出。默认情况下返回文本输出作为后备。

## 输出
- 工具输出对象，支持多种输出格式

## 函数实现
code/smolcc/tool_output.py:line 250-279

## 测试代码
code/test_code/test_tool_output_convert.py

---

# FUNCTION: create_agent

## 功能概述
创建配置完整的工具智能体实例

## 函数签名
def create_agent(cwd: Optional[str] = None, log_file: Optional[str] = "tool_agent.log") -> ToolAgent

## 输入参数
- `cwd`: (str) 当前工作目录
- `log_file`: (str) 日志文件路径

## 详细描述
创建配置完整的工具智能体实例。首先加载环境变量，设置工作目录。调用获取系统提示方法生成动态系统提示。创建轻量语言模型实例，配置Claude模型和API密钥。导入并实例化所有可用工具：Bash工具、编辑工具、全局工具、列表工具、查看工具、用户输入工具、替换工具。使用初始化机制创建工具智能体，传入工具列表、模型配置和日志设置。返回配置完成的智能体实例，可以直接用于执行任务。

## 输出
- 配置完成的工具智能体实例

## 函数实现
code/smolcc/agent.py:line 227-295

## 测试代码
code/test_code/test_create_agent.py


---

# FUNCTION: ToolAgent.run

## 功能概述
运行智能体主循环

## 函数签名
def run(user_input: str, stream: bool = False) -> str

## 输入参数
- `user_input`: (str) 用户输入查询
- `stream`: (bool) 是否流式输出

## 详细描述
运行智能体主循环，添加了Rich控制台输出和错误处理。调用父类的运行方法执行智能体，捕获执行过程中的异常。在运行完成后，添加空白行分隔符，将结果作为助手消息显示在控制台中。

## 输出
- 智能体的响应结果

## 函数实现
code/smolcc/agent.py:line 150-181

## 测试代码
code/test_code/test_agent_llm_run_log.py

---

# FUNCTION: ToolAgent.get_tool

## 功能概述
根据名称获取工具实例

## 函数签名
def get_tool(tool_name: str) -> Tool

## 输入参数
- `tool_name`: (str) 工具名称

## 详细描述
从智能体的工具字典中查找指定名称的工具实例。如果工具不存在，抛出值错误异常。所有工具调用都必须通过此方法获取工具实例。

## 输出
- 工具实例

## 函数实现
code/smolcc/agent.py:line 206-224

## 测试代码
code/test_code/test_agent_get_tool.py

---

# FUNCTION: RichConsoleLogger.log_error

## 功能概述
错误日志记录方法

## 函数签名
def log_error(error_message: str) -> None

## 输入参数
- `error_message`: (str) 错误消息字符串

## 详细描述
使用Rich控制台突出显示错误信息。同时将错误信息写入日志文件，确保错误跟踪的完整性。

## 输出
- 无

## 函数实现
code/smolcc/agent.py:line 51-57

## 测试代码
code/test_code/test_agent_llm_run_log.py

---

# FUNCTION: ToolOutput.display

## 功能概述
工具输出显示方法

## 函数签名
def display(console: Console) -> None

## 输入参数
- `console`: (Console) Rich控制台对象

## 详细描述
使用Rich控制台显示工具调用信息。显示工具名称和参数，应用黄色样式突出显示。

## 输出
- 无

## 函数实现
code/smolcc/tool_output.py:line 75-77

## 测试代码
code/test_code/test_tool_output_convert.py

---

# FUNCTION: TextOutput.display

## 功能概述
文本输出显示方法

## 函数签名
def display(console: Console) -> None

## 输入参数
- `console`: (Console) Rich控制台对象

## 详细描述
使用Rich控制台显示文本内容。处理多行文本的截断显示，应用灰色样式。

## 输出
- 无

## 函数实现
code/smolcc/tool_output.py:line 83-97

## 测试代码
code/test_code/test_tool_output_convert.py

---

# FUNCTION: CodeOutput.display

## 功能概述
代码输出显示方法

## 函数签名
def display(console: Console) -> None

## 输入参数
- `console`: (Console) Rich控制台对象

## 详细描述
使用Rich控制台显示代码内容。显示代码预览和摘要，对于长代码块提供行号显示。应用语法高亮和主题设置，增强代码可读性。

## 输出
- 无

## 函数实现
code/smolcc/tool_output.py:line 100-148

## 测试代码
code/test_code/test_tool_output_convert.py

---

# FUNCTION: TableOutput.display

## 功能概述
表格数据输出显示方法

## 函数签名
def display(console: Console) -> None

## 输入参数
- `console`: (Console) Rich控制台对象

## 详细描述
使用Rich控制台显示表格数据。创建格式化表格，添加列标题和行数据。

## 输出
- 无

## 函数实现
code/smolcc/tool_output.py:line 151-186

## 测试代码
code/test_code/test_tool_output_convert.py

---

# FUNCTION: FileListOutput.display

## 功能概述
文件列表输出显示方法

## 函数签名
def display(console: Console) -> None

## 输入参数
- `console`: (Console) Rich控制台对象

## 详细描述
使用Rich控制台显示文件列表信息。创建格式化表格，显示文件名、类型和大小。限制显示文件数量以避免信息过载。

## 输出
- 无

## 函数实现
code/smolcc/tool_output.py:line 188-246

## 测试代码
code/test_code/test_tool_output_convert.py

---

# FUNCTION: AssistantOutput.display

## 功能概述
助手消息显示方法

## 函数签名
def display(console: Console) -> None

## 输入参数
- `console`: (Console) Rich控制台对象

## 详细描述
使用Rich控制台显示助手消息。应用白色样式突出显示助手响应。
## 输出
- 无

## 函数实现
code/smolcc/tool_output.py:line 244-246

## 测试代码
code/test_code/test_tool_output_convert.py

---

# FUNCTION: get_directory_structure

## 功能概述
生成目录结构信息

## 函数签名
def get_directory_structure(start_path: str, ignore_patterns: Optional[List[str]] = None) -> str

## 输入参数
- `start_path`: (str) 起始目录路径
- `ignore_patterns`: (List[str]) 忽略模式列表

## 详细描述
递归遍历目录树生成格式化结构。过滤隐藏文件和开发目录，限制文件数量避免信息过载。该方法提供动态的目录上下文信息，支持Git仓库状态检测。

## 输出
- 格式化的目录结构字符串

## 函数实现
code/smolcc/system_prompt.py:line 9-110

## 测试代码
code/test_code/test_system_prompt_utils.py

---

# FUNCTION: is_git_repo

## 功能概述
检查目录是否为Git仓库

## 函数签名
def is_git_repo(path: str) -> bool

## 输入参数
- `path`: (str) 目录路径

## 详细描述
使用Git命令检查目录是否为Git仓库。处理命令执行异常，返回检测结果。该方法提供Git仓库状态信息。

## 输出
- 布尔值，表示是否为Git仓库

## 函数实现
code/smolcc/system_prompt.py:line 113-124

## 测试代码
code/test_code/test_system_prompt_utils.py

---

# FUNCTION: get_git_status

## 功能概述
获取Git仓库状态信息

## 函数签名
def get_git_status(cwd: str) -> str

## 输入参数
- `cwd`: (str) 当前工作目录

## 详细描述
执行多个Git命令获取仓库状态信息。包括当前分支、主分支、文件状态和最近提交。该方法提供详细的Git上下文信息。

## 输出
- 格式化的Git状态字符串

## 函数实现
code/smolcc/system_prompt.py:line 168-209

## 测试代码
code/test_code/test_system_prompt_utils.py

---