# SmolAgents Examples Core Algorithm and Method Descriptions (11 Functions)

# FUNCTION: multi_model_agent_execution

## Function Overview
Executes agent tasks using multiple model backends, demonstrating SmolAgents' model compatibility and flexibility.

## Function Signature
def multi_model_agent_execution(chosen_inference: str) -> tuple

## Input Parameters
- `chosen_inference`: (str) Inference engine type selection, supports ["inference_client", "transformers", "ollama", "litellm", "openai"].

## Detailed Description
Based on the passed inference engine type parameter, dynamically selects and instantiates a model. Then creates a tool calling agent and a code agent, passing the instantiated model and weather tool to them. Uses a fixed query to run both agents and returns their execution results.

## Output
- `tool_calling_result`: (any) Execution result of the tool calling agent.
- `code_agent_result`: (any) Execution result of the code agent.

## Function Implementation
code/agent_from_any_llm.py:line 27-64

## Test Code
code/test_code/test_multi_model_agent_execution.py

---

# FUNCTION: multi_llm_load_balancing

## Function Overview
Implements multi-LLM load balancing and failover, improving agent system reliability and performance.

## Function Signature
def multi_llm_load_balancing(model_list: list, routing_strategy: str = "simple-shuffle") -> any

## Input Parameters
- `model_list`: (list) Model configuration list containing configuration information for multiple model instances.
- `routing_strategy`: (str) Routing strategy, defaults to "simple-shuffle".

## Detailed Description
Receives a model list and a routing strategy, using these parameters to instantiate a load balancing model. Then creates a code agent that uses this load balancing model and a web search tool. Runs the agent with a fixed query and returns the result, demonstrating how to configure and use an agent with load balancing and failover capabilities.

## Output
- `full_result`: (any) Execution result of the code agent.

## Function Implementation
code/multi_llm_agent.py:line 27-40

## Test Code
code/test_code/test_multi_llm_load_balancing.py

---

# FUNCTION: multi_tool_integration

## Function Overview
Integrates multiple external API tools, extending the agent's information retrieval and processing capabilities.

## Function Signature
def multi_tool_integration(query: str) -> any

## Input Parameters
- `query`: (str) User input query string.

## Detailed Description
First instantiates an inference client model. Then passes a set of predefined tools created through tool decorators to a newly created code agent. Runs the agent with the passed query parameter and returns the result.

## Output
- `result`: (any) Execution result of the code agent.

## Function Implementation
code/multiple_tools.py:line 228-246

## Test Code
code/test_code/test_multi_tool_integration.py

---

# FUNCTION: rag_with_lexical_search

## Function Overview
Implements a retrieval-augmented generation (RAG) system based on lexical search, improving agent answer accuracy for domain-specific knowledge.

## Function Signature
def rag_with_lexical_search(query: str) -> any

## Input Parameters
- `query`: (str) User input query string.

## Detailed Description
Implements a complete lexical search RAG workflow. First loads the knowledge base through the datasets library, then uses a recursive character text splitter to chunk documents. Next defines an internal class called retriever tool that uses a BM25 retriever to perform lexical search. Finally instantiates the retriever tool and a code agent, runs the agent with the passed query, and returns the final answer.

## Output
- `agent_output`: (any) Execution result of the code agent.

## Function Implementation
code/rag.py:line 12-69

## Test Code
code/test_code/test_rag_with_lexical_search.py

---

# FUNCTION: text_to_sql_conversion

## Function Overview
Converts natural language queries to SQL statements and executes them, implementing natural language database interaction.

## Function Signature
def text_to_sql_conversion(query: str) -> any

## Input Parameters
- `query`: (str) User's natural language query string.

## Detailed Description
First creates a code agent and equips it with a tool called SQL engine that can execute SQL queries on a pre-configured in-memory SQLite database. Then runs the agent with the passed query; the agent converts natural language to SQL, calls the SQL engine tool to execute the query, and returns the final result.

## Output
- `result`: (any) Execution result of the code agent.

## Function Implementation
code/text_to_sql.py:line 75-85

## Test Code
code/test_code/test_text_to_sql_conversion.py

---

# FUNCTION: structured_output_with_mcp

## Function Overview
Uses Model Context Protocol (MCP) to implement structured output, ensuring tools return formatted data.

## Function Signature
def structured_output_with_mcp(query: str) -> any

## Input Parameters
- `query`: (str) User input query string.

## Detailed Description
Demonstrates how to integrate tools with structured output through MCP. First defines and starts an inline MCP server that provides a weather information tool returning a Pydantic model. Then connects to the server through an MCP client and sets structured output to true. The obtained tools are passed to a code agent, which finally runs with the passed query and returns the result.

## Output
- `result`: (any) Execution result of the code agent.

## Function Implementation
code/structured_output_tool.py:line 55-75

## Test Code
code/test_code/test_structured_output_with_mcp.py

---

# FUNCTION: multi_agent_coordination

## Function Overview
Coordinates multiple agents working together, implementing decomposition and execution of complex tasks.

## Function Signature
def multi_agent_coordination(task: str) -> any

## Input Parameters
- `task`: (str) Task description to be executed.

## Detailed Description
Demonstrates how to set up a hierarchical multi-agent system. First creates a tool calling agent equipped with search tools as a "worker" agent. Then creates another code agent as a "manager" agent, passing the "worker" agent to it through the managed agents parameter. Finally, the manager agent receives and executes the passed task, delegating tasks to worker agents when needed.

## Output
- `run_result`: (any) Execution result of the manager code agent, including token usage and performance statistics.

## Function Implementation
code/inspect_multiagent_run.py:line 9-35

## Test Code
code/test_code/test_multi_agent_coordination.py

---

# FUNCTION: rag_with_vector_database

## Function Overview
Implements a RAG system using vector databases for semantic search, providing more precise document retrieval.

## Function Signature
def rag_with_vector_database(query: str) -> any

## Input Parameters
- `query`: (str) User input query string.

## Detailed Description
Implements a complete vector search RAG workflow. Loads datasets, processes, embeds, and stores documents using HuggingFace embeddings and Chroma vector database. Then defines an internal retriever tool class that uses vector store similarity search to perform semantic search. Finally instantiates the tool and a code agent, runs the agent with the passed query, and returns the final answer.

## Output
- `agent_output`: (any) Execution result of the code agent.

## Function Implementation
code/rag_using_chromadb.py:line 21-95

## Test Code
code/test_code/test_rag_with_vector_database.py

---

# FUNCTION: async_agent_execution

## Function Overview
Executes synchronous agents in async web frameworks, implementing non-blocking agent services.

## Function Signature
async def run_agent_in_thread(task: str) -> any

## Input Parameters
- `task`: (str) Task string to be executed.

## Detailed Description
This functionality is implemented by the run agent in thread function, which is called in the context of a Starlette web application. The function receives a task string, then calls the get agent function to obtain a synchronous code agent instance. The key step is using anyio.to_thread.run_sync to run the synchronous agent run method in a background thread, avoiding blocking Starlette's event loop. This enables efficient use of synchronous agents in async frameworks.

## Output
- `result`: (any) Execution result of the code agent.

## Function Implementation
code/async_agent/main.py:line 26-30

## Test Code
code/test_code/test_async_agent_execution.py

---

# FUNCTION: plan_customization

## Function Overview
Allows users to review and modify agent-generated execution plans, providing human-in-the-loop execution control.

## Function Signature
def plan_customization() -> any

## Input Parameters
- None. This function interacts directly with users through input.

## Detailed Description
Implements plan customization by setting step callbacks. Defines an interrupt after plan callback function that triggers after planning steps. In the callback, it displays the generated plan to the user and provides options to approve, modify, or cancel. Based on user input, it may continue execution, continue with a modified plan, or invoke the agent interrupt method to stop execution. The function also demonstrates how to resume agent execution after cancellation using reset parameter set to false to preserve its memory.

## Output
- `result` / `resumed_result`: (any) Execution result of the code agent, possibly returned after initial run or after resumption.

## Function Implementation
code/plan_customization/plan_customization.py:line 100-167

## Test Code
code/test_code/test_plan_customization.py

---

# FUNCTION: mcp_server_integration

## Function Overview
Integrates tools provided by remote MCP servers, extending the agent's functional scope.

## Function Signature
async def chat(request: Request) -> JSONResponse

## Input Parameters
- `request`: (starlette.requests.Request) HTTP request containing user message.

## Detailed Description
This functionality is implemented by the Starlette application in the server main file. At application startup, it creates an MCP client to connect to a remote MCP server. Then it calls the MCP client get tools method to discover and obtain all available remote tools. These tools are instantiated together with a code agent. The application provides a chat endpoint that receives user messages, runs the agent in a background thread, and returns results to the frontend chat interface.

## Output
- `JSONResponse`: (starlette.responses.JSONResponse) JSON response containing agent reply.

## Function Implementation
code/server/main.py:line 197-204

## Test Code
code/test_code/test_mcp_server_integration.py

---
