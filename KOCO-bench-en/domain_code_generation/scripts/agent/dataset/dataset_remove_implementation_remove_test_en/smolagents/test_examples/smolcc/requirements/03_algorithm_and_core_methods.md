# SmolCC Core Algorithm Function Descriptions (17 Functions)

# FUNCTION: execute_tool_call

## Function Overview
Executes tool calls and displays rich visual feedback

## Function Signature
def execute_tool_call(tool_name: str, tool_arguments: Dict[str, Any]) -> Any

## Input Parameters
- `tool_name`: (str) Name of the tool to execute
- `tool_arguments`: (Dict[str, Any]) Parameter dictionary passed to the tool

## Detailed Description
Executes tool calls and records execution time. First checks if it's a final answer or final output tool call, then displays tool call information. Obtains tool instance through get tool method, executes the tool, and records execution time. Result processing supports multiple output formats: converts plain results to tool output objects, automatically selects appropriate output format based on content type. For non-final answer tool calls, displays formatted output and execution time statistics.

## Output
- Tool execution result, supporting multiple output formats

## Function Implementation
code/smolcc/agent.py:line 99-148

## Test Code
code/test_code/test_agent_execute_tool_call.py

---

# FUNCTION: format_messages_for_llm

## Function Overview
Prepares LLM messages and displays thinking status

## Function Signature
def format_messages_for_llm(prompt: str) -> List[Dict[str, str]]

## Input Parameters
- `prompt`: (str) User prompt

## Detailed Description
Displays thinking status spinner while preparing LLM messages. Creates temporary progress bar using Rich's progress component, providing visual feedback during LLM formatting operations. Calls parent class's format messages method to obtain formatted message list, ensuring message format is fully compatible. This method provides user-friendly waiting indication when LLM processing time is long.

## Output
- Formatted LLM message list

## Function Implementation
code/smolcc/agent.py:line 183-204

## Test Code
code/test_code/test_agent_llm_run_log.py

---

# FUNCTION: get_system_prompt

## Function Overview
Generates dynamic system prompt

## Function Signature
def get_system_prompt(cwd: Optional[str] = None) -> str

## Input Parameters
- `cwd`: (str) Current working directory

## Detailed Description
Reads base system prompt from system message template file, then dynamically fills placeholders: current date, working directory, platform information, model name. Integrates Git repository status detection: checks if current directory is a Git repository, if so obtains Git status information (current branch, main branch, status, recent commits). Generates directory structure information, ignoring common development directories and file patterns. Finally returns dynamically generated system prompt, ensuring prompt contains latest environment context information.

## Output
- Dynamically generated system prompt string

## Function Implementation
code/smolcc/system_prompt.py:line 127-165

## Test Code
code/test_code/test_system_prompt_utils.py

---

# FUNCTION: convert_to_tool_output

## Function Overview
Converts plain results to appropriate tool output objects

## Function Signature
def convert_to_tool_output(result: Any) -> ToolOutput

## Input Parameters
- `result`: (Any) Result to convert

## Detailed Description
Converts various types of tool execution results to unified tool output objects. First checks if result is already a tool output instance, if so returns directly. For string results, automatically detects if it's code content: checks if it contains newlines and has indentation at line start or Python keywords. If it's code returns code output, otherwise returns text output. For list results, detects if it's table data or file information list, returning table output or file list output respectively. By default returns text output as fallback.

## Output
- Tool output object, supporting multiple output formats

## Function Implementation
code/smolcc/tool_output.py:line 250-279

## Test Code
code/test_code/test_tool_output_convert.py

---

# FUNCTION: create_agent

## Function Overview
Creates fully configured tool agent instance

## Function Signature
def create_agent(cwd: Optional[str] = None, log_file: Optional[str] = "tool_agent.log") -> ToolAgent

## Input Parameters
- `cwd`: (str) Current working directory
- `log_file`: (str) Log file path

## Detailed Description
Creates fully configured tool agent instance. First loads environment variables, sets working directory. Calls get system prompt method to generate dynamic system prompt. Creates LiteLLM model instance, configures Claude model and API key. Imports and instantiates all available tools: Bash tool, edit tool, glob tool, list tool, view tool, user input tool, replace tool. Uses initialization mechanism to create tool agent, passing tool list, model configuration, and logging settings. Returns configured agent instance, ready for task execution.

## Output
- Configured tool agent instance

## Function Implementation
code/smolcc/agent.py:line 227-295

## Test Code
code/test_code/test_create_agent.py

---

# FUNCTION: ToolAgent.run

## Function Overview
Runs agent main loop

## Function Signature
def run(user_input: str, stream: bool = False) -> str

## Input Parameters
- `user_input`: (str) User input query
- `stream`: (bool) Whether to stream output

## Detailed Description
Runs agent main loop, adding Rich console output and error handling. Calls parent class's run method to execute agent, capturing exceptions during execution. After run completion, adds blank line separator, displays result as assistant message in console.

## Output
- Agent's response result

## Function Implementation
code/smolcc/agent.py:line 150-181

## Test Code
code/test_code/test_agent_llm_run_log.py

---

# FUNCTION: ToolAgent.get_tool

## Function Overview
Gets tool instance by name

## Function Signature
def get_tool(tool_name: str) -> Tool

## Input Parameters
- `tool_name`: (str) Tool name

## Detailed Description
Looks up tool instance with specified name from agent's tool dictionary. If tool doesn't exist, raises value error exception. All tool calls must obtain tool instances through this method.

## Output
- Tool instance

## Function Implementation
code/smolcc/agent.py:line 206-224

## Test Code
code/test_code/test_agent_get_tool.py

---

# FUNCTION: RichConsoleLogger.log_error

## Function Overview
Error logging method

## Function Signature
def log_error(error_message: str) -> None

## Input Parameters
- `error_message`: (str) Error message string

## Detailed Description
Highlights error information using Rich console. Also writes error information to log file, ensuring completeness of error tracking.

## Output
- None

## Function Implementation
code/smolcc/agent.py:line 51-57

## Test Code
code/test_code/test_agent_llm_run_log.py

---

# FUNCTION: ToolOutput.display

## Function Overview
Tool output display method

## Function Signature
def display(console: Console) -> None

## Input Parameters
- `console`: (Console) Rich console object

## Detailed Description
Displays tool call information using Rich console. Shows tool name and parameters, applying yellow style for highlighting.

## Output
- None

## Function Implementation
code/smolcc/tool_output.py:line 75-77

## Test Code
code/test_code/test_tool_output_convert.py

---

# FUNCTION: TextOutput.display

## Function Overview
Text output display method

## Function Signature
def display(console: Console) -> None

## Input Parameters
- `console`: (Console) Rich console object

## Detailed Description
Displays text content using Rich console. Handles truncated display of multi-line text, applying gray style.

## Output
- None

## Function Implementation
code/smolcc/tool_output.py:line 83-97

## Test Code
code/test_code/test_tool_output_convert.py

---

# FUNCTION: CodeOutput.display

## Function Overview
Code output display method

## Function Signature
def display(console: Console) -> None

## Input Parameters
- `console`: (Console) Rich console object

## Detailed Description
Displays code content using Rich console. Shows code preview and summary, provides line number display for long code blocks. Applies syntax highlighting and theme settings, enhancing code readability.

## Output
- None

## Function Implementation
code/smolcc/tool_output.py:line 100-148

## Test Code
code/test_code/test_tool_output_convert.py

---

# FUNCTION: TableOutput.display

## Function Overview
Table data output display method

## Function Signature
def display(console: Console) -> None

## Input Parameters
- `console`: (Console) Rich console object

## Detailed Description
Displays table data using Rich console. Creates formatted table, adds column headers and row data.

## Output
- None

## Function Implementation
code/smolcc/tool_output.py:line 151-186

## Test Code
code/test_code/test_tool_output_convert.py

---

# FUNCTION: FileListOutput.display

## Function Overview
File list output display method

## Function Signature
def display(console: Console) -> None

## Input Parameters
- `console`: (Console) Rich console object

## Detailed Description
Displays file list information using Rich console. Creates formatted table, shows file name, type, and size. Limits number of displayed files to avoid information overload.

## Output
- None

## Function Implementation
code/smolcc/tool_output.py:line 188-246

## Test Code
code/test_code/test_tool_output_convert.py

---

# FUNCTION: AssistantOutput.display

## Function Overview
Assistant message display method

## Function Signature
def display(console: Console) -> None

## Input Parameters
- `console`: (Console) Rich console object

## Detailed Description
Displays assistant message using Rich console. Applies white style to highlight assistant response.

## Output
- None

## Function Implementation
code/smolcc/tool_output.py:line 244-246

## Test Code
code/test_code/test_tool_output_convert.py

---

# FUNCTION: get_directory_structure

## Function Overview
Generates directory structure information

## Function Signature
def get_directory_structure(start_path: str, ignore_patterns: Optional[List[str]] = None) -> str

## Input Parameters
- `start_path`: (str) Starting directory path
- `ignore_patterns`: (List[str]) Ignore pattern list

## Detailed Description
Recursively traverses directory tree to generate formatted structure. Filters hidden files and development directories, limits file count to avoid information overload. This method provides dynamic directory context information, supports Git repository status detection.

## Output
- Formatted directory structure string

## Function Implementation
code/smolcc/system_prompt.py:line 9-110

## Test Code
code/test_code/test_system_prompt_utils.py

---

# FUNCTION: is_git_repo

## Function Overview
Checks if directory is a Git repository

## Function Signature
def is_git_repo(path: str) -> bool

## Input Parameters
- `path`: (str) Directory path

## Detailed Description
Uses Git command to check if directory is a Git repository. Handles command execution exceptions, returns detection result. This method provides Git repository status information.

## Output
- Boolean value indicating whether it's a Git repository

## Function Implementation
code/smolcc/system_prompt.py:line 113-124

## Test Code
code/test_code/test_system_prompt_utils.py

---

# FUNCTION: get_git_status

## Function Overview
Gets Git repository status information

## Function Signature
def get_git_status(cwd: str) -> str

## Input Parameters
- `cwd`: (str) Current working directory

## Detailed Description
Executes multiple Git commands to obtain repository status information. Includes current branch, main branch, file status, and recent commits. This method provides detailed Git context information.

## Output
- Formatted Git status string

## Function Implementation
code/smolcc/system_prompt.py:line 168-209

## Test Code
code/test_code/test_system_prompt_utils.py

---
