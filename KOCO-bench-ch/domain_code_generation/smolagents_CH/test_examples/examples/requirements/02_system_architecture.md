# SmolAgents高级示例系统架构

## 1. SmolAgents模块文件结构

### 1.1 文件组织
```
examples/
├── agent_from_any_llm.py          # 多模型集成示例
├── gradio_ui.py                  # Gradio用户界面
├── inspect_multiagent_run.py       # 多智能体运行监控
├── multi_llm_agent.py             # 多模型负载均衡
├── multiple_tools.py              # 多功能工具集成
├── rag_using_chromadb.py         # ChromaDB向量检索
├── rag.py                        # BM25文本检索
├── sandboxed_execution.py         # 安全沙箱执行
├── structured_output_tool.py       # 结构化输出工具
├── text_to_sql.py                # SQL数据库交互
├── async_agent/                  # 异步智能体应用
│   ├── main.py                   # Starlette异步集成
│   └── README.md                 # 异步应用说明
├── open_deep_research/            # 深度研究系统
│   ├── app.py                    # Gradio界面
│   ├── run.py                    # 研究智能体实现
│   └── scripts/                  # 研究工具集
├── plan_customization/            # 计划定制系统
│   ├── plan_customization.py      # 人机协作计划
│   └── README.md                 # 计划定制说明
├── server/                       # Web服务器
│   ├── main.py                   # MCP协议服务器
│   └── README.md                 # 服务器说明
└── smolagents_benchmark/          # 性能基准测试
    └── run.py                    # 基准测试实现
```

### 1.2 模块职责划分
- agent_from_any_llm.py: 展示多种模型后端的集成方式
- multiple_tools.py: 演示多功能工具的集成与使用
- rag_using_chromadb.py/rag.py: 两种不同的检索增强生成实现
- sandboxed_execution.py: 多种安全执行环境的配置与使用
- async_agent/: 异步框架中的智能体集成
- open_deep_research/: 复杂研究任务的自动化实现
- plan_customization/: 人机协作的计划定制与执行
- server/: 基于MCP协议的Web服务器实现
- smolagents_benchmark/: 智能体性能评估框架

## 2. SmolAgents模块设计

### 2.1 多模型集成架构 (agent_from_any_llm.py)

核心类: `InferenceClientModel`, `TransformersModel`, `LiteLLMModel`, `OpenAIModel`, `ToolCallingAgent`, `CodeAgent`

主要功能:
- 统一接口: 抽象不同模型后端的差异
- 动态选择: 运行时切换不同模型类型
- 配置化: 通过参数控制模型行为

关键方法:
- `agent.run()`: 执行智能体任务
- `get_weather()`: 自定义工具示例

### 2.2 多模型负载均衡 (multi_llm_agent.py)

核心类: `LiteLLMRouterModel`, `CodeAgent`

主要功能:
- 请求分发: 按配置策略分发请求到不同模型
- 故障转移: 自动切换到可用模型
- 性能优化: 根据负载情况动态调整

关键方法:
- `LiteLLMRouterModel()`: 初始化负载均衡模型
- `agent.run()`: 执行负载均衡的智能体任务

### 2.3 多功能工具系统 (multiple_tools.py)

核心工具函数:
- `get_weather()`: 天气查询工具
- `convert_currency()`: 货币转换工具
- `get_news_headlines()`: 新闻获取工具
- `get_joke()`: 笑话生成工具
- `get_time_in_timezone()`: 时间查询工具
- `get_random_fact()`: 随机事实工具
- `search_wikipedia()`: 维基百科搜索工具

工具特点:
- 标准接口: 统一的输入输出格式
- 错误处理: 完善的异常处理机制
- API集成: 与外部服务无缝对接

### 2.4 检索增强生成 (rag_using_chromadb.py, rag.py)

ChromaDB向量检索:
核心类: `RetrieverTool`, `Chroma`, `HuggingFaceEmbeddings`, `CodeAgent`

关键方法:
- `RetrieverTool.forward()`: 执行向量相似度搜索
- `vector_store.similarity_search()`: 检索相似文档
- `agent.run()`: 执行RAG任务

BM25文本检索:
核心类: `RetrieverTool`, `BM25Retriever`, `CodeAgent`

关键方法:
- `RetrieverTool.forward()`: 执行BM25文本检索
- `retriever.invoke()`: 调用BM25检索器
- `agent.run()`: 执行检索任务

### 2.5 安全沙箱执行 (sandboxed_execution.py)

核心类: `CodeAgent`, `WebSearchTool`

执行环境:
- Docker: `executor_type="docker"`
- E2B: `executor_type="e2b"`
- Modal: `executor_type="modal"`
- WebAssembly: `executor_type="wasm"`

关键方法:
- `CodeAgent.__enter__()`: 初始化执行环境
- `CodeAgent.run()`: 在指定环境中执行任务
- `CodeAgent.__exit__()`: 清理执行环境

### 2.6 异步智能体集成 (async_agent/main.py)

核心类: `CodeAgent`, `InferenceClientModel`, `Starlette`, `Route`

关键方法:
- `get_agent()`: 创建智能体实例
- `run_agent_in_thread()`: 在线程中运行同步智能体
- `run_agent_endpoint()`: 处理异步HTTP请求
- `anyio.to_thread.run_sync()`: 将同步操作转为异步

### 2.7 深度研究系统 (open_deep_research/run.py)

核心类: `CodeAgent`, `ToolCallingAgent`, `GoogleSearchTool`, `VisitWebpageTool`, `TextInspectorTool`

关键方法:
- `create_agent()`: 创建研究智能体系统
- `TextInspectorTool()`: 文本内容检查工具
- `SimpleTextBrowser()`: 网页浏览工具
- `agent.run()`: 执行深度研究任务

### 2.8 计划定制系统 (plan_customization/plan_customization.py)

核心类: `CodeAgent`, `DuckDuckGoSearchTool`, `PlanningStep`

关键方法:
- `interrupt_after_plan()`: 计划创建后的中断回调
- `display_plan()`: 格式化显示计划内容
- `get_user_choice()`: 获取用户选择
- `get_modified_plan()`: 获取用户修改的计划
- `agent.run()`: 执行定制化计划任务

### 2.9 Web服务器实现 (server/main.py)

核心类: `CodeAgent`, `InferenceClientModel`, `MCPClient`, `Starlette`

关键方法:
- `MCPClient.get_tools()`: 获取MCP工具集
- `homepage()`: 提供Web界面
- `chat()`: 处理聊天请求
- `to_thread.run_sync()`: 异步执行智能体任务
- `mcp_client.disconnect()`: 断开MCP连接

### 2.10 性能基准测试 (smolagents_benchmark/run.py)

核心类: `CodeAgent`, `ToolCallingAgent`, `GoogleSearchTool`, `VisitWebpageTool`, `PythonInterpreterTool`

关键方法:
- `parse_arguments()`: 解析命令行参数
- `load_eval_dataset()`: 加载评估数据集
- `answer_single_question()`: 回答单个问题
- `answer_questions()`: 批量回答问题
- `append_answer()`: 保存答案结果

## 3. SmolAgents数据流与控制

### 3.1 模型间通信

数据传递:
- 参数传递: 通过函数参数传递任务需求
- 结果返回: 标准化输出格式便于后续处理
- 上下文保持: 维护多轮对话历史

### 3.2 执行控制机制

控制参数:
- max_steps: 控制智能体执行步数
- executor_type: 指定代码执行环境
- planning_interval: 控制计划生成频率
- verbosity_level: 控制输出详细程度

### 3.3 工具扩展机制

扩展方式:
- 自定义工具: 继承Tool类实现特定功能
- MCP协议: 通过Model Control Protocol集成外部工具
- API集成: 与外部服务API无缝对接

SmolAgents高级示例通过这种模块化、可扩展的架构设计，实现了从简单工具集成到复杂多智能体系统的完整解决方案。