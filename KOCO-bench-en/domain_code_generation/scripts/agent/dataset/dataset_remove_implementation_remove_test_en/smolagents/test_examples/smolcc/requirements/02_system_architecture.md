# SmolCC System Architecture and Module Design

## 1. SmolCC Module File Structure

### 1.1 File Organization
```
smolcc/
├── smolcc.py                    # SmolCC main program entry
├── agent.py                     # ToolAgent core implementation
├── system_prompt.py             # Dynamic system prompt generator
├── tool_output.py               # Tool output formatting system
├── utils.py                     # Utility functions
├── system_message.txt           # Base system prompt template
└── tools/
    ├── __init__.py              # Tool package initialization
    ├── bash_tool.py             # Bash command execution tool
    ├── edit_tool.py             # File editing tool
    ├── glob_tool.py             # Wildcard file finding tool
    ├── grep_tool.py             # Regular expression search tool
    ├── ls_tool.py               # Directory listing tool
    ├── replace_tool.py          # Content replacement tool
    ├── user_input_tool.py       # User input tool
    └── view_tool.py             # File viewing tool
```

### 1.2 Module Responsibilities
- smolcc.py: SmolCC program startup logic and main loop
- agent.py: ToolAgent class implementation, inheriting from smolagents' ToolCallingAgent
- system_prompt.py: Dynamically generates context-aware system prompts
- tool_output.py: Tool execution result formatting output system
- tools/: Specific implementations of various code operation tools

## 2. SmolCC Module Design

### 2.1 Main Program Entry (smolcc.py)

Core functions:
- `main()`: Command-line argument parsing and program startup entry
- `run_interactive_mode(agent)`: Interactive mode execution logic

Main functionality:
- Command-line argument handling (--interactive, --cwd, --no-log, --log-file)
- Working directory setup and logging configuration
- Interactive session management
- Rich console interface display

### 2.2 ToolAgent Core Class (agent.py)

Core class: `ToolAgent` (inherits from smolagents.ToolCallingAgent)

Main methods:
- `__init__(*args, **kwargs)`: Initialize toolset and system prompt
- `execute_tool_call(tool_name, tool_arguments)`: Execute tool call and display results
- `run(user_input, stream=False)`: Run Agent main loop
- `format_messages_for_llm(prompt)`: Prepare LLM messages and display thinking status
- `get_tool(tool_name)`: Get tool instance by name

Core components:
- `RichConsoleLogger`: Custom logger integrating Rich console output
- `create_agent(cwd, log_file)`: Factory function creating configured ToolAgent instance

### 2.3 Dynamic Prompt System (system_prompt.py)

Core functions:
- `get_system_prompt(cwd=None)`: Generate dynamic system prompt
- `get_directory_structure(start_path, ignore_patterns)`: Generate directory structure
- `is_git_repo(path)`: Check if directory is a Git repository
- `get_git_status(cwd)`: Get Git status information

Main functionality:
- Dynamically fills placeholders in system prompt template
- Automatically detects current working directory structure
- Integrates Git repository status information
- Ignores common development directories and files

### 2.4 Tool Output System (tool_output.py)

Core classes:
- `ToolOutput`: Tool output base class
- `ToolCallOutput`: Tool call output display
- `TextOutput`: Plain text output
- `CodeOutput`: Code output (with syntax highlighting)
- `TableOutput`: Table data output
- `FileListOutput`: File list output
- `AssistantOutput`: Assistant message output

Main methods:
- `display(console)`: Display formatted output in console
- `convert_to_tool_output(result)`: Convert plain results to ToolOutput objects

### 2.5 Tool Ecosystem

#### BashTool (bash_tool.py)
Core class: `BashTool`
Main methods:
- `forward(command, timeout)`: Execute Bash command
- `_execute_command_with_timeout(command, timeout_sec)`: Command execution with timeout
- `_is_banned_command(command)`: Check for banned commands
- `_is_code_command(command)`: Determine if command is a code command

Feature highlights:
- Persistent shell sessions
- Command whitelist security mechanism
- Automatic syntax highlighting detection
- Output truncation and formatting

#### ViewTool (view_tool.py)
Core class: `ViewTool`
Main methods:
- `forward(file_path, offset, limit)`: Read file content
- `_is_code_file(file_path)`: Determine if file is a code file
- `_get_language_for_file(file_path)`: Get syntax highlighting language

Feature highlights:
- Supports multiple file encodings
- Automatic syntax highlighting
- Paginated display for large files
- Special handling for image files

#### EditTool (edit_tool.py)
Core class: `FileEditTool`
Main methods:
- `forward(file_path, old_string, new_string)`: Edit file content
- `_get_snippet(new_content, old_string, new_string)`: Get edit snippet
- `_add_line_numbers(content, start_line)`: Add line numbers

Feature highlights:
- Precise text replacement
- Uniqueness validation
- Edit result preview
- New file creation support

#### GrepTool (grep_tool.py)
Core class: `GrepTool`
Main methods:
- `forward(pattern, include, path)`: Regular expression search
- `_find_files(base_path, include)`: Find matching files
- `_get_file_matches(file_path, regex)`: Get file match results
- `_is_binary_file(file_path)`: Check for binary files

Feature highlights:
- Supports complex regular expressions
- Multi-file type filtering
- Context line display
- Performance-optimized large file handling

#### LsTool (ls_tool.py)
Core class: `LSTool`
Main methods:
- `forward(path, ignore)`: List directory contents
- `_get_directory_contents(path, ignore_patterns)`: Get directory information
- `_should_skip(path, ignore_patterns)`: Determine if path should be skipped

Feature highlights:
- Categorized display of directories and files
- Ignore pattern support
- File size and modification time display
- Hidden file filtering

## 3. SmolCC Data Flow and Interactions

### 3.1 User Interaction Flow
1. **Initialization**: Load toolset, generate dynamic system prompt
2. **Input Parsing**: Receive user query, analyze intent
3. **Tool Selection**: Select appropriate tools based on query content
4. **Parameter Extraction**: Extract tool parameters from query
5. **Execute Operations**: Call tools to execute specific operations
6. **Result Formatting**: Format output using ToolOutput
7. **State Update**: Update session history and system state

### 3.2 Tool Calling Protocol
```python
# Tool call example
tool_call = {
    "name": "view_tool",
    "arguments": {
        "filepath": "example.py",
        "start_line": 1,
        "end_line": 10
    }
}

# Execution result
tool_result = {
    "success": True,
    "output": ToolOutput object,
    "error": None
}
```

## 4. SmolCC Overall Workflow

### 4.1 System Startup Flow
1. **Environment Initialization**: Load environment variables, set working directory
2. **Tool Registration**: Scan and register all available tools
3. **Prompt Generation**: Dynamically generate system prompt containing current environment
4. **Model Configuration**: Configure LLM model and API keys
5. **Session Start**: Enter interactive or single-query mode

### 4.2 Query Processing Flow
```
User Input → Intent Analysis → Tool Selection → Parameter Extraction → Tool Execution → Result Formatting → Output Display
```

### 4.3 Tool Collaboration Workflow
- **File Operation Chain**: ViewTool → EditTool → Verify Results
- **Search Operation Chain**: LsTool → GrepTool → ViewTool
- **Batch Operation Chain**: GlobTool → Multiple EditTool calls
- **Command Execution Chain**: BashTool → System command execution

### 4.4 State Management Flow
- **Session State**: Maintain user conversation history and tool call records
- **Environment State**: Track current working directory and Git repository status
- **Tool State**: Manage persistent tool instances (e.g., BashTool's shell session)

### 4.5 Error Handling Flow
- **Tool Errors**: Capture tool execution exceptions, return friendly error messages
- **Parameter Errors**: Validate input parameters, provide detailed error descriptions
- **Permission Errors**: Check file system permissions, suggest solutions
- **Network Errors**: Handle API call failures, provide retry mechanism

### 4.6 Performance Optimization Flow
- **Tool Caching**: Reuse tool instances to avoid repeated initialization
- **Output Optimization**: Intelligently truncate large file outputs
- **Parallel Processing**: Support parallel execution of multiple tools
- **Resource Management**: Monitor system resource usage, prevent resource exhaustion
