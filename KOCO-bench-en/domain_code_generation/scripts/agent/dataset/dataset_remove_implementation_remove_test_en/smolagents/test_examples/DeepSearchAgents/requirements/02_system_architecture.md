# DeepSearchAgents System Architecture and Module Design

## 1. DeepSearchAgents Module File Structure

### 1.1 File Organization
```
DeepSearchAgents/
├── src/
│   ├── agents/                    # Agent implementation modules
│   │   ├── base_agent.py         # Base agent class
│   │   ├── react_agent.py        # ReAct agent implementation
│   │   ├── codact_agent.py       # CodeAct agent implementation
│   │   ├── manager_agent.py      # Manager agent implementation
│   │   ├── runtime.py            # Agent runtime management
│   │   ├── run_result.py         # Run result encapsulation
│   │   ├── stream_aggregator.py  # Streaming output aggregator
│   │   ├── prompt_templates/     # Prompt templates
│   │   ├── servers/             # Server implementations
│   │   └── ui_common/          # UI common components
│   ├── tools/                     # Tool collection modules
│   │   ├── search.py             # Search tools
│   │   ├── search_fast.py        # Fast search tools
│   │   ├── search_helpers.py     # Search helper tools
│   │   ├── readurl.py            # URL reading tools
│   │   ├── chunk.py             # Text chunking tools
│   │   ├── embed.py             # Text embedding tools
│   │   ├── rerank.py            # Text reranking tools
│   │   ├── wolfram.py           # WolframAlpha tools
│   │   ├── xcom_qa.py           # X.com Q&A tools
│   │   ├── github_qa.py         # GitHub Q&A tools
│   │   ├── final_answer.py       # Final answer tool
│   │   └── toolbox.py           # Toolbox management
│   ├── core/                      # Core functionality modules
│   │   ├── config/              # Configuration management
│   │   ├── search_engines/       # Search engines
│   │   ├── scraping/            # Web scraping
│   │   ├── ranking/             # Content ranking
│   │   ├── chunk/               # Text chunking
│   │   ├── academic_tookit/     # Academic toolkit
│   │   ├── github_toolkit/      # GitHub toolkit
│   │   └── xcom_toolkit/       # X.com toolkit
│   ├── api/                       # API interface modules
│   │   ├── v1/                  # API v1 version
│   │   └── v2/                  # API v2 version
│   ├── cli.py                     # Command-line interface
│   └── main.py                    # Main program entry point
├── frontend/                      # Frontend interface
├── tests/                        # Test cases
└── docs/                         # Documentation
```

### 1.2 Module Responsibilities
- agents/: Agent implementations extending SmolAgents, including ReAct, CodeAct, and manager agents
- tools/: Tool collection extending SmolAgents tool system for search, processing, and analysis
- core/: Core functionality including configuration management, search engines, web scraping, and content processing
- api/: RESTful API and web interface support, including v1 and v2 versions
- cli.py: Command-line interface supporting interactive and batch modes
- main.py: FastAPI main program entry point

## 2. DeepSearchAgents Module Design

### 2.1 Agent Base Module (agents/base_agent.py)

Core classes: `BaseAgent`, `MultiModelRouter`

Main functionality:
- Multi-model routing: Dynamically selects search model or orchestrator model based on prompt content
- Base functionality: Provides shared functionality for all agent types
- State management: Extends SmolAgents' state management mechanism
- Streaming output: Supports streaming output and real-time feedback
- Resource management: Provides context manager for resource cleanup

### 2.2 ReAct Agent Module (agents/react_agent.py)

Core classes: `DebuggingToolCallingAgent`, `ReactAgent`

Main functionality:
- Empty answer detection: Detects and handles empty final_answer calls
- Retry mechanism: Provides detailed error information and retry guidance
- Debugging support: Captures model output for debugging analysis
- Parallel tool execution: Supports concurrent multiple tool calls
- Planning interval: Periodic planning steps

### 2.3 CodeAct Agent Module (agents/codact_agent.py)

Core class: `CodeActAgent`

Main functionality:
- Code security validation: Checks for dangerous code patterns and blocks execution
- Import management: Controls the list of allowed Python modules for import
- Prompt template extension: Adds deep search-specific prompt templates
- Structured output: Supports JSON format for internal communication
- Executor configuration: Supports local and remote code executors

### 2.4 Manager Agent Module (agents/manager_agent.py)

Core class: `ManagerAgent`

Main functionality:
- Agent orchestration: Coordinates workflows of multiple sub-agents
- Task decomposition: Breaks down complex tasks into subtasks and assigns them
- Result aggregation: Integrates outputs from multiple agents
- Delegation depth control: Prevents infinite recursion in delegation mechanism
- Dynamic team creation: Supports research team and custom team configurations

### 2.5 Agent Runtime Module (agents/runtime.py)

Core class: `AgentRuntime`

Main functionality:
- Agent factory: Creates and manages different types of agent instances
- Tool initialization: Uniformly initializes and configures all tools
- API key management: Validates and manages API keys for various services
- Session management: Tracks and manages active agent sessions
- Model creation: Uniformly creates and configures LLM model instances

### 2.6 Search Tool Module (tools/search.py)

Core class: `SearchLinksTool`

Main functionality:
- Multi-source search: Integrates Google (Serper), X.com (xAI), and hybrid search
- Intelligent source selection: Automatically selects the most appropriate search source based on query content
- Advanced filtering: Supports domain filtering, time range, safe search, etc.
- Result normalization: Converts results from different search sources to a unified format
- Error handling: Comprehensive error handling and retry mechanism

### 2.7 Configuration Management Module (core/config/settings.py)

Core class: `Settings`

Main functionality:
- Layered configuration: Supports TOML files, environment variables, and default values
- Type validation: Uses Pydantic for configuration type validation
- Agent configuration: Separately configures ReAct, CodeAct, and manager agents
- Tool configuration: Configures specific parameters for various tools
- Model configuration: Manages orchestrator model and search model configurations

## 3. DeepSearchAgents Data Protocols and Module Interactions

### 3.1 DeepSearchAgents Data Protocols

Data protocol structure: DeepSearchAgents uses standardized data protocols to ensure consistency in data transfer between modules.

RunResult field specification:
```python
run_result = {
    "final_answer": str,              # Final answer content
    "execution_time": float,          # Execution time (seconds)
    "token_usage": TokenUsage,        # Token usage statistics
    "agent_type": str,               # Agent type
    "model_info": dict,              # Model information
    "steps": List[dict],            # Execution step list
    "error": str                     # Error information (if any)
}
```

Agent state specification:
```python
agent_state = {
    "visited_urls": set,              # Set of visited URLs
    "search_queries": list,            # Search query history
    "key_findings": dict,            # Key findings index
    "search_depth": dict,             # Current search depth
    "reranking_history": list,        # Reranking history
    "content_quality": dict           # Content quality assessment
}
```

### 3.2 DeepSearchAgents Workflow

1. Initialization phase:
   - Load configuration files and environment variables
   - Validate API keys and service availability
   - Initialize tool collection and search engines
   - Create multi-model router

2. Agent creation phase:
   - Select agent type based on configuration
   - Create orchestrator model and search model
   - Initialize agent state and tools
   - Register step callbacks and monitoring

3. Query processing phase:
   - Receive and parse user query
   - Select initial model based on query content
   - Execute agent main loop
   - Perform periodic planning steps

4. Result generation phase:
   - Collect all execution steps and results
   - Generate structured final answer
   - Calculate execution statistics and token usage
   - Clean up resources and release connections

### 3.3 DeepSearchAgents Data Flow

- Input: User query × Configuration parameters
- Routing: Multi-model router selects model based on content
- Execution: Agent executes main loop, calling tools
- Aggregation: Collects tool execution results and intermediate states
- Output: Structured answer containing content and sources

### 3.4 DeepSearchAgents Module Interactions

- Agent-Tool interaction: Through SmolAgents' standard tool interface
- Agent-Model interaction: Through multi-model router unified interface
- Agent-Agent interaction: Through manager's delegation mechanism
- Runtime-Configuration interaction: Through Pydantic settings' type-safe interface
- API-Agent interaction: Through runtime's factory methods and session management

## 4. DeepSearchAgents Extension Mechanisms

### 4.1 Agent Extension

Extension methods:
- Inherit BaseAgent: Implement new agent types
- Implement create_agent: Define agent creation logic
- Register to runtime: Register via AgentRuntime.register_agent
- Configuration support: Add corresponding configuration parameters

### 4.2 Tool Extension

Supported tools:
- Inherit Tool class: Follow SmolAgents tool interface
- Add to toolbox: Register new tools via toolbox.py
- Configuration parameters: Support tool-specific configuration parameters
- API key management: Uniformly manage API keys through settings

### 4.3 Search Engine Extension

Extension methods:
- Implement base class interface: Inherit from search_engines/base.py
- Register to search tool: Integrate in SearchLinksTool
- Result format normalization: Convert to unified result format
- Configuration parameters: Support search engine-specific configuration
