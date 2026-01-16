# SmolCC Intelligent Code Assistant

## 1. Project Overview

SmolCC (Small Code Companion) is an intelligent code assistant system built on the smolagents framework. The system provides developers with intelligent code editing, file management, and system operation capabilities by integrating various code operation tools. SmolCC adopts the ReAct (Reasoning and Acting) framework, combining dynamic system prompts with a rich tool ecosystem to achieve efficient code understanding and operation capabilities.

## 2. Core Features

### Code Editing and Viewing
- File content viewing with syntax highlighting
- Code editing and saving operations
- Regular expression-based code search
- Multi-file content replacement and pattern replacement

### File System Management
- Directory structure browsing and file list viewing
- Wildcard pattern file finding
- File operations and directory navigation

### System Command Execution
- Bash command execution and interactive shell sessions
- Environment awareness and state management
- Secure command execution mechanism

## 3. Technical Features

### Based on smolagents Framework
- Inherits ToolCallingAgent to implement tool calling capabilities
- Integrates ReAct reasoning framework
- Supports multi-tool collaboration

### Dynamic Prompt System
- Context-aware system prompt generation
- Automatic Git repository status detection
- Real-time tool documentation integration

### Rich Tool Ecosystem
- Code editing tools: ViewTool, EditTool, ReplaceTool
- File management tools: LsTool, GlobTool
- Code search tools: GrepTool
- System command tools: BashTool
