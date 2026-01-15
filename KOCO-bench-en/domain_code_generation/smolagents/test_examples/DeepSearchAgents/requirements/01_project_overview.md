# DeepSearchAgents Deep Search Agent System

## 1. System Overview

DeepSearchAgents is a deep search agent system built on the SmolAgents framework, implementing advanced information retrieval, content analysis, and knowledge discovery by extending SmolAgents' core functionality. The system supports both ReAct and CodeAct agent modes, and provides a manager agent for hierarchical orchestration, enabling complex deep search tasks through multi-model routing and tool integration.

## 2. Workflow

```
Algorithm: DeepSearchAgents Pipeline
Input: User query Q, Agent type A, Tools T
Parameters: max_steps=25, planning_interval=5, model_router=MultiModelRouter
Output: Search results with sources and analysis

1: Initialize agent runtime and tools
2: Select agent type based on configuration
3: Create model router with orchestrator and search models
4: Configure agent with tools and initial state
5: 
6: if A == "react" then
7:    agent ← DebuggingToolCallingAgent(model_router, T, max_steps)
8:    // ReAct mode: Reasoning-Acting loop
9:    for step = 1 to max_steps do
10:       // Planning phase (every planning_interval steps)
11:       if step % planning_interval == 0 then
12:          planning ← model_router.orchestrator_model(context)
13:          update_agent_state(planning)
14:       end if
15:       
16:       // Tool calling phase
17:       action ← model_router.search_model(context)
18:       result ← execute_tool(action, T)
19:       update_memory(action, result)
20:    end for
21: 
22: elif A == "codact" then
23:    agent ← CodeActAgent(model_router, T, max_steps)
24:    // CodeAct mode: Code execution-Analysis loop
25:    for step = 1 to max_steps do
26:       // Generate code execution plan
27:       code ← model_router.search_model(context)
28:       result ← execute_code(code, authorized_imports)
29:       update_memory(code, result)
30:       
31:       // Planning phase (every planning_interval steps)
32:       if step % planning_interval == 0 then
33:          planning ← model_router.orchestrator_model(context)
34:          update_agent_state(planning)
35:       end if
36:    end for
37: 
38: elif A == "manager" then
39:    agent ← ManagerAgent(model_router, managed_agents)
40:    // Manager mode: Agent orchestration
41:    for step = 1 to max_steps do
42:       // Analyze task and select appropriate sub-agent
43:       task_analysis ← model_router.orchestrator_model(context)
44:       selected_agent ← select_agent(task_analysis, managed_agents)
45:       result ← selected_agent.execute(subtask)
46:       update_memory(task_analysis, result)
47:    end for
48: end if
49: 
50: // Generate final answer
51: final_answer ← generate_structured_output(memory, sources)
52: return final_answer
```

Key function descriptions in the workflow:
- `DebuggingToolCallingAgent`: Extends SmolAgents' ToolCallingAgent, adding empty answer detection and retry logic
- `CodeActAgent`: Extends SmolAgents' CodeAgent, adding secure code validation and deep search functionality
- `ManagerAgent`: Extends CodeAgent to implement hierarchical agent orchestration
- `MultiModelRouter`: Routes to different models (search model or orchestrator model) based on prompt content
- `execute_tool`: Executes tool operations such as search, read, and rerank
- `execute_code`: Executes Python code in a secure sandbox
- `generate_structured_output`: Generates structured answers with title, content, and sources

## 3. Application Scenarios

### Web Search and Information Retrieval
- Input: User query
- Output: Structured answer with source links
- Search methods: Google search (Serper), X.com search (xAI), hybrid search
- Data processing: Content scraping, text chunking, embedding and reranking

### Academic Research and Literature Review
- Input: Academic research topic
- Output: Comprehensive research report with literature citations
- Data sources: arXiv, bioRxiv, medRxiv, Google Scholar
- Processing methods: Paper retrieval, metadata merging, content parsing

### Code Generation and Data Analysis
- Input: Data processing or analysis task
- Output: Executable Python code and analysis results
- Execution environment: Secure local Python executor
- Supported libraries: pandas, numpy, requests, matplotlib, etc.

### Multi-Agent Collaborative Tasks
- Input: Complex multi-step task
- Output: Comprehensive results from specialized agent collaboration
- Agent specialization: Web search specialist, data analysis specialist
- Coordination method: Manager agent dynamically assigns subtasks
