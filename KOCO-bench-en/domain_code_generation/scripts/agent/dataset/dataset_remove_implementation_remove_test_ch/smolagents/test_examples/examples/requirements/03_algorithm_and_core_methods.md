# SmolAgents Examples核心算法与方法描述共11个

# FUNCTION: multi_model_agent_execution

## 功能概述
使用多种模型后端执行智能体任务，展示SmolAgents的模型兼容性和灵活性。

## 函数签名
def multi_model_agent_execution(chosen_inference: str) -> tuple

## 输入参数
- `chosen_inference`: (str) 推理引擎类型选择，支持["inference_client", "transformers", "ollama", "litellm", "openai"]。

## 详细描述
根据传入的推理引擎类型参数，动态选择并实例化一个模型。然后创建工具调用智能体和代码智能体，并将实例化的模型和天气工具传递给它们。使用固定的查询来运行这两个智能体，并返回它们的执行结果。

## 输出
- `tool_calling_result`: (any) 工具调用智能体的执行结果。
- `code_agent_result`: (any) 代码智能体的执行结果。

## 函数实现
code/agent_from_any_llm.py:line 27-64

## 测试代码
code/test_code/test_multi_model_agent_execution.py

---

# FUNCTION: multi_llm_load_balancing

## 功能概述
实现多LLM负载均衡和故障转移，提高智能体系统的可靠性和性能。

## 函数签名
def multi_llm_load_balancing(model_list: list, routing_strategy: str = "simple-shuffle") -> any

## 输入参数
- `model_list`: (list) 模型配置列表，包含多个模型实例的配置信息。
- `routing_strategy`: (str) 路由策略，默认为 "simple-shuffle"。

## 详细描述
接收一个模型列表和一个路由策略，并使用这些参数来实例化一个负载均衡模型。然后创建一个代码智能体，该智能体使用这个负载均衡模型和一个网络搜索工具。使用固定的查询运行该智能体并返回结果，从而展示了如何配置和使用具有负载均衡和故障转移能力的智能体。

## 输出
- `full_result`: (any) 代码智能体的执行结果。

## 函数实现
code/multi_llm_agent.py:line 27-40

## 测试代码
code/test_code/test_multi_llm_load_balancing.py

---

# FUNCTION: multi_tool_integration

## 功能概述
集成多个外部API工具，扩展智能体的信息获取和处理能力。

## 函数签名
def multi_tool_integration(query: str) -> any

## 输入参数
- `query`: (str) 用户输入的查询字符串。

## 详细描述
首先实例化一个推理客户端模型。然后将一组预定义的、通过工具装饰器创建的工具传递给一个新创建的代码智能体。使用传入的查询参数运行该智能体，并返回结果。

## 输出
- `result`: (any) 代码智能体的执行结果。

## 函数实现
code/multiple_tools.py:line 228-246

## 测试代码
code/test_code/test_multi_tool_integration.py

---

# FUNCTION: rag_with_lexical_search

## 功能概述
实现基于词汇搜索的检索增强生成(RAG)系统，提高智能体对特定领域知识的回答准确性。

## 函数签名
def rag_with_lexical_search(query: str) -> any

## 输入参数
- `query`: (str) 用户输入的查询字符串。

## 详细描述
实现了完整的词汇搜索RAG流程。首先通过数据集库加载知识库，然后使用递归字符文本分割器对文档进行分块处理。接着定义了一个名为检索工具的内部类，该类使用BM25检索器来执行词汇搜索。最后实例化检索工具和一个代码智能体，并使用传入的查询运行智能体，返回最终的答案。

## 输出
- `agent_output`: (any) 代码智能体的执行结果。

## 函数实现
code/rag.py:line 12-69

## 测试代码
code/test_code/test_rag_with_lexical_search.py

---

# FUNCTION: text_to_sql_conversion

## 功能概述
将自然语言查询转换为SQL语句并执行，实现自然语言数据库交互。

## 函数签名
def text_to_sql_conversion(query: str) -> any

## 输入参数
- `query`: (str) 用户的自然语言查询字符串。

## 详细描述
首先创建一个代码智能体，并为其配备一个名为SQL引擎的工具，该工具能够对预先设置好的内存中SQLite数据库执行SQL查询。然后使用传入的查询运行该智能体，智能体将自然语言转换为SQL，调用SQL引擎工具执行查询，并返回最终结果。

## 输出
- `result`: (any) 代码智能体的执行结果。

## 函数实现
code/text_to_sql.py:line 75-85

## 测试代码
code/test_code/test_text_to_sql_conversion.py

---

# FUNCTION: structured_output_with_mcp

## 功能概述
使用模型控制协议(MCP)实现结构化输出，确保工具返回格式化的数据。

## 函数签名
def structured_output_with_mcp(query: str) -> any

## 输入参数
- `query`: (str) 用户输入的查询字符串。

## 详细描述
展示了如何通过MCP集成带有结构化输出的工具。首先定义并启动一个内联的MCP服务器，该服务器提供一个返回Pydantic模型的天气信息工具。然后通过MCP客户端连接到该服务器，并将结构化输出设置为真。获取到的工具被传递给一个代码智能体，该智能体最后使用传入的查询运行并返回结果。

## 输出
- `result`: (any) 代码智能体的执行结果。

## 函数实现
code/structured_output_tool.py:line 55-75

## 测试代码
code/test_code/test_structured_output_with_mcp.py

---

# FUNCTION: multi_agent_coordination

## 功能概述
协调多个智能体协同工作，实现复杂任务的分解和执行。

## 函数签名
def multi_agent_coordination(task: str) -> any

## 输入参数
- `task`: (str) 需要执行的任务描述。

## 详细描述
展示了如何设置一个层级多智能体系统。首先创建一个配备了搜索工具的工具调用智能体作为"工作者"智能体。然后创建另一个代码智能体作为"管理者"智能体，并将"工作者"智能体通过管理智能体参数传递给它。最后，管理者智能体接收并执行传入的任务，在需要时可以委派任务给工作者智能体。

## 输出
- `run_result`: (any) 管理者代码智能体的执行结果，包含令牌使用和性能统计信息。

## 函数实现
code/inspect_multiagent_run.py:line 9-35

## 测试代码
code/test_code/test_multi_agent_coordination.py

---

# FUNCTION: rag_with_vector_database

## 功能概述
使用向量数据库实现语义搜索的RAG系统，提供更精准的文档检索。

## 函数签名
def rag_with_vector_database(query: str) -> any

## 输入参数
- `query`: (str) 用户输入的查询字符串。

## 详细描述
实现了完整的向量搜索RAG流程。加载数据集，使用HuggingFace嵌入和Chroma向量数据库对文档进行处理、嵌入和存储。然后定义了一个检索工具内部类，该类使用向量存储的相似性搜索来执行语义搜索。最后实例化该工具和一个代码智能体，并使用传入的查询运行智能体，返回最终答案。

## 输出
- `agent_output`: (any) 代码智能体的执行结果。

## 函数实现
code/rag_using_chromadb.py:line 21-95

## 测试代码
code/test_code/test_rag_with_vector_database.py

---

# FUNCTION: async_agent_execution

## 功能概述
在异步Web框架中执行同步智能体，实现非阻塞的智能体服务。

## 函数签名
async def run_agent_in_thread(task: str) -> any

## 输入参数
- `task`: (str) 需要执行的任务字符串。

## 详细描述
此功能由在线程中运行智能体函数实现，它在一个Starlette Web应用的上下文中被调用。该函数接收一个任务字符串，然后调用获取智能体函数来获取一个同步的代码智能体实例。关键步骤是使用异步线程运行同步在后台线程中运行同步的智能体运行方法，从而避免阻塞Starlette的事件循环。这使得同步的智能体可以在异步框架中高效使用。

## 输出
- `result`: (any) 代码智能体的执行结果。

## 函数实现
code/async_agent/main.py:line 26-30

## 测试代码
code/test_code/test_async_agent_execution.py

---

# FUNCTION: plan_customization

## 功能概述
允许用户审查和修改智能体生成的执行计划，提供人机协作的执行控制。

## 函数签名
def plan_customization() -> any

## 输入参数
- 无。该函数通过输入直接与用户交互。

## 详细描述
通过设置步骤回调来实现计划的定制。定义了一个计划后中断回调函数，该函数在规划步骤之后触发。在回调中，它向用户显示生成的计划，并提供批准、修改或取消的选项。根据用户的输入，它可能会继续执行、使用修改后的计划继续执行，或者调用智能体中断方法来停止执行。该函数还演示了如何在取消后使用重置参数为假来恢复智能体的执行，以保留其记忆。

## 输出
- `result` / `resumed_result`: (any) 代码智能体的执行结果，可能是在初次运行后或在恢复后返回。

## 函数实现
code/plan_customization/plan_customization.py:line 100-167

## 测试代码
code/test_code/test_plan_customization.py

---

# FUNCTION: mcp_server_integration

## 功能概述
集成远程MCP服务器提供的工具，扩展智能体的功能范围。

## 函数签名
async def chat(request: Request) -> JSONResponse

## 输入参数
- `request`: (starlette.requests.Request) 包含用户消息的HTTP请求。

## 详细描述
此功能由服务器主文件中的Starlette应用实现。在应用启动时，它会创建一个MCP客户端来连接到一个远程的MCP服务器。然后它调用MCP客户端获取工具方法来发现并获取所有可用的远程工具。这些工具与一个代码智能体一起被实例化。该应用提供了一个聊天端点，它接收用户的消息，在后台线程中运行智能体运行，并将结果返回给前端的聊天界面。

## 输出
- `JSONResponse`: (starlette.responses.JSONResponse) 包含智能体回复的JSON响应。

## 函数实现
code/server/main.py:line 197-204

## 测试代码
code/test_code/test_mcp_server_integration.py

---
