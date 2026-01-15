# SmolAgents Advanced Examples System Architecture

## 1. SmolAgents Module File Structure

### 1.1 File Organization
```
examples/
├── agent_from_any_llm.py          # Multi-model integration example
├── gradio_ui.py                  # Gradio user interface
├── inspect_multiagent_run.py       # Multi-agent run monitoring
├── multi_llm_agent.py             # Multi-model load balancing
├── multiple_tools.py              # Multi-functional tool integration
├── rag_using_chromadb.py         # ChromaDB vector retrieval
├── rag.py                        # BM25 text retrieval
├── sandboxed_execution.py         # Secure sandbox execution
├── structured_output_tool.py       # Structured output tool
├── text_to_sql.py                # SQL database interaction
├── async_agent/                  # Async agent applications
│   ├── main.py                   # Starlette async integration
│   └── README.md                 # Async application documentation
├── open_deep_research/            # Deep research system
│   ├── app.py                    # Gradio interface
│   ├── run.py                    # Research agent implementation
│   └── scripts/                  # Research toolset
├── plan_customization/            # Plan customization system
│   ├── plan_customization.py      # Human-in-the-loop planning
│   └── README.md                 # Plan customization documentation
├── server/                       # Web server
│   ├── main.py                   # MCP protocol server
│   └── README.md                 # Server documentation
└── smolagents_benchmark/          # Performance benchmarking
    └── run.py                    # Benchmark implementation
```

### 1.2 Module Responsibilities
- agent_from_any_llm.py: Demonstrates integration methods for various model backends
- multiple_tools.py: Demonstrates integration and usage of multi-functional tools
- rag_using_chromadb.py/rag.py: Two different retrieval-augmented generation implementations
- sandboxed_execution.py: Configuration and usage of multiple secure execution environments
- async_agent/: Agent integration in async frameworks
- open_deep_research/: Automation implementation for complex research tasks
- plan_customization/: Human-in-the-loop plan customization and execution
- server/: Web server implementation based on MCP protocol
- smolagents_benchmark/: Agent performance evaluation framework

## 2. SmolAgents Module Design

### 2.1 Multi-Model Integration Architecture (agent_from_any_llm.py)

Core classes: `InferenceClientModel`, `TransformersModel`, `LiteLLMModel`, `OpenAIModel`, `ToolCallingAgent`, `CodeAgent`

Main functionality:
- Unified interface: Abstracts differences between model backends
- Dynamic selection: Runtime switching between different model types
- Configuration-based: Controls model behavior through parameters

Key methods:
- `agent.run()`: Execute agent tasks
- `get_weather()`: Custom tool example

### 2.2 Multi-Model Load Balancing (multi_llm_agent.py)

Core classes: `LiteLLMRouterModel`, `CodeAgent`

Main functionality:
- Request distribution: Distributes requests to different models based on configured strategy
- Failover: Automatically switches to available models
- Performance optimization: Dynamically adjusts based on load conditions

Key methods:
- `LiteLLMRouterModel()`: Initialize load balancing model
- `agent.run()`: Execute load-balanced agent tasks

### 2.3 Multi-Functional Tool System (multiple_tools.py)

Core tool functions:
- `get_weather()`: Weather query tool
- `convert_currency()`: Currency conversion tool
- `get_news_headlines()`: News retrieval tool
- `get_joke()`: Joke generation tool
- `get_time_in_timezone()`: Time query tool
- `get_random_fact()`: Random fact tool
- `search_wikipedia()`: Wikipedia search tool

Tool features:
- Standard interface: Unified input/output format
- Error handling: Comprehensive exception handling mechanism
- API integration: Seamless integration with external services

### 2.4 Retrieval-Augmented Generation (rag_using_chromadb.py, rag.py)

ChromaDB Vector Retrieval:
Core classes: `RetrieverTool`, `Chroma`, `HuggingFaceEmbeddings`, `CodeAgent`

Key methods:
- `RetrieverTool.forward()`: Execute vector similarity search
- `vector_store.similarity_search()`: Retrieve similar documents
- `agent.run()`: Execute RAG tasks

BM25 Text Retrieval:
Core classes: `RetrieverTool`, `BM25Retriever`, `CodeAgent`

Key methods:
- `RetrieverTool.forward()`: Execute BM25 text retrieval
- `retriever.invoke()`: Invoke BM25 retriever
- `agent.run()`: Execute retrieval tasks

### 2.5 Secure Sandbox Execution (sandboxed_execution.py)

Core classes: `CodeAgent`, `WebSearchTool`

Execution environments:
- Docker: `executor_type="docker"`
- E2B: `executor_type="e2b"`
- Modal: `executor_type="modal"`
- WebAssembly: `executor_type="wasm"`

Key methods:
- `CodeAgent.__enter__()`: Initialize execution environment
- `CodeAgent.run()`: Execute tasks in specified environment
- `CodeAgent.__exit__()`: Clean up execution environment

### 2.6 Async Agent Integration (async_agent/main.py)

Core classes: `CodeAgent`, `InferenceClientModel`, `Starlette`, `Route`

Key methods:
- `get_agent()`: Create agent instance
- `run_agent_in_thread()`: Run synchronous agent in thread
- `run_agent_endpoint()`: Handle async HTTP requests
- `anyio.to_thread.run_sync()`: Convert synchronous operations to async

### 2.7 Deep Research System (open_deep_research/run.py)

Core classes: `CodeAgent`, `ToolCallingAgent`, `GoogleSearchTool`, `VisitWebpageTool`, `TextInspectorTool`

Key methods:
- `create_agent()`: Create research agent system
- `TextInspectorTool()`: Text content inspection tool
- `SimpleTextBrowser()`: Web browsing tool
- `agent.run()`: Execute deep research tasks

### 2.8 Plan Customization System (plan_customization/plan_customization.py)

Core classes: `CodeAgent`, `DuckDuckGoSearchTool`, `PlanningStep`

Key methods:
- `interrupt_after_plan()`: Interrupt callback after plan creation
- `display_plan()`: Format and display plan content
- `get_user_choice()`: Get user choice
- `get_modified_plan()`: Get user-modified plan
- `agent.run()`: Execute customized plan tasks

### 2.9 Web Server Implementation (server/main.py)

Core classes: `CodeAgent`, `InferenceClientModel`, `MCPClient`, `Starlette`

Key methods:
- `MCPClient.get_tools()`: Get MCP toolset
- `homepage()`: Provide web interface
- `chat()`: Handle chat requests
- `to_thread.run_sync()`: Async agent task execution
- `mcp_client.disconnect()`: Disconnect MCP connection

### 2.10 Performance Benchmarking (smolagents_benchmark/run.py)

Core classes: `CodeAgent`, `ToolCallingAgent`, `GoogleSearchTool`, `VisitWebpageTool`, `PythonInterpreterTool`

Key methods:
- `parse_arguments()`: Parse command-line arguments
- `load_eval_dataset()`: Load evaluation dataset
- `answer_single_question()`: Answer single question
- `answer_questions()`: Batch answer questions
- `append_answer()`: Save answer results

## 3. SmolAgents Data Flow and Control

### 3.1 Inter-Model Communication

Data transfer:
- Parameter passing: Pass task requirements through function parameters
- Result return: Standardized output format for subsequent processing
- Context maintenance: Maintain multi-turn conversation history

### 3.2 Execution Control Mechanism

Control parameters:
- max_steps: Control agent execution steps
- executor_type: Specify code execution environment
- planning_interval: Control planning generation frequency
- verbosity_level: Control output verbosity

### 3.3 Tool Extension Mechanism

Extension methods:
- Custom tools: Inherit Tool class to implement specific functionality
- MCP protocol: Integrate external tools through Model Context Protocol
- API integration: Seamless integration with external service APIs

SmolAgents advanced examples achieve complete solutions from simple tool integration to complex multi-agent systems through this modular, extensible architecture design.
