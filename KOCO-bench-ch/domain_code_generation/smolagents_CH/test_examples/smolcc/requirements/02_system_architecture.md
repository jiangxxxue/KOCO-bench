# SmolCC系统架构与模块设计

## 1. SmolCC模块文件结构

### 1.1 文件组织
```
smolcc/
├── smolcc.py                    # SmolCC主程序入口
├── agent.py                     # ToolAgent核心实现
├── system_prompt.py             # 动态系统提示生成器
├── tool_output.py               # 工具输出格式化系统
├── utils.py                     # 工具函数
├── system_message.txt           # 基础系统提示模板
└── tools/
    ├── __init__.py              # 工具包初始化
    ├── bash_tool.py             # Bash命令执行工具
    ├── edit_tool.py             # 文件编辑工具
    ├── glob_tool.py             # 通配符文件查找工具
    ├── grep_tool.py             # 正则表达式搜索工具
    ├── ls_tool.py               # 目录列表工具
    ├── replace_tool.py          # 内容替换工具
    ├── user_input_tool.py       # 用户输入工具
    └── view_tool.py             # 文件查看工具
```

### 1.2 模块职责划分
- smolcc.py: SmolCC程序的启动逻辑和主循环
- agent.py: ToolAgent类实现，继承自smolagents的ToolCallingAgent
- system_prompt.py: 动态生成上下文感知的系统提示
- tool_output.py: 工具执行结果的格式化输出系统
- tools/: 各种代码操作工具的具体实现

## 2. SmolCC模块设计

### 2.1 主程序入口 (smolcc.py)

核心函数:
- `main()`: 命令行参数解析和程序启动入口
- `run_interactive_mode(agent)`: 交互式模式运行逻辑

主要功能:
- 命令行参数处理 (--interactive, --cwd, --no-log, --log-file)
- 工作目录设置和日志配置
- 交互式会话管理
- Rich控制台界面显示

### 2.2 ToolAgent核心类 (agent.py)

核心类: `ToolAgent` (继承自smolagents.ToolCallingAgent)

主要方法:
- `__init__(*args, **kwargs)`: 初始化工具集和系统提示
- `execute_tool_call(tool_name, tool_arguments)`: 执行工具调用并显示结果
- `run(user_input, stream=False)`: 运行Agent主循环
- `format_messages_for_llm(prompt)`: 准备LLM消息并显示思考状态
- `get_tool(tool_name)`: 根据名称获取工具实例

核心组件:
- `RichConsoleLogger`: 自定义日志器，集成Rich控制台输出
- `create_agent(cwd, log_file)`: 工厂函数，创建配置好的ToolAgent实例

### 2.3 动态提示系统 (system_prompt.py)

核心函数:
- `get_system_prompt(cwd=None)`: 生成动态系统提示
- `get_directory_structure(start_path, ignore_patterns)`: 生成目录结构
- `is_git_repo(path)`: 检查是否为Git仓库
- `get_git_status(cwd)`: 获取Git状态信息

主要功能:
- 动态填充系统提示模板中的占位符
- 自动检测当前工作目录结构
- 集成Git仓库状态信息
- 忽略常见开发目录和文件

### 2.4 工具输出系统 (tool_output.py)

核心类:
- `ToolOutput`: 工具输出基类
- `ToolCallOutput`: 工具调用输出显示
- `TextOutput`: 普通文本输出
- `CodeOutput`: 代码输出（带语法高亮）
- `TableOutput`: 表格数据输出
- `FileListOutput`: 文件列表输出
- `AssistantOutput`: 助手消息输出

主要方法:
- `display(console)`: 在控制台中显示格式化输出
- `convert_to_tool_output(result)`: 将普通结果转换为ToolOutput对象

### 2.5 工具生态系统

#### BashTool (bash_tool.py)
核心类: `BashTool`
主要方法:
- `forward(command, timeout)`: 执行Bash命令
- `_execute_command_with_timeout(command, timeout_sec)`: 带超时的命令执行
- `_is_banned_command(command)`: 检查禁止的命令
- `_is_code_command(command)`: 判断是否为代码命令

功能特点:
- 持久化Shell会话
- 命令白名单安全机制
- 自动语法高亮检测
- 输出截断和格式化

#### ViewTool (view_tool.py)
核心类: `ViewTool`
主要方法:
- `forward(file_path, offset, limit)`: 读取文件内容
- `_is_code_file(file_path)`: 判断是否为代码文件
- `_get_language_for_file(file_path)`: 获取语法高亮语言

功能特点:
- 支持多种文件编码
- 自动语法高亮
- 大文件分页显示
- 图像文件特殊处理

#### EditTool (edit_tool.py)
核心类: `FileEditTool`
主要方法:
- `forward(file_path, old_string, new_string)`: 编辑文件内容
- `_get_snippet(new_content, old_string, new_string)`: 获取编辑片段
- `_add_line_numbers(content, start_line)`: 添加行号

功能特点:
- 精确文本替换
- 唯一性验证
- 编辑结果预览
- 新文件创建支持

#### GrepTool (grep_tool.py)
核心类: `GrepTool`
主要方法:
- `forward(pattern, include, path)`: 正则表达式搜索
- `_find_files(base_path, include)`: 查找匹配文件
- `_get_file_matches(file_path, regex)`: 获取文件匹配结果
- `_is_binary_file(file_path)`: 检查二进制文件

功能特点:
- 支持复杂正则表达式
- 多文件类型过滤
- 上下文行显示
- 性能优化的大文件处理

#### LsTool (ls_tool.py)
核心类: `LSTool`
主要方法:
- `forward(path, ignore)`: 列出目录内容
- `_get_directory_contents(path, ignore_patterns)`: 获取目录信息
- `_should_skip(path, ignore_patterns)`: 判断是否跳过路径

功能特点:
- 目录和文件分类显示
- 忽略模式支持
- 文件大小和修改时间显示
- 隐藏文件过滤

## 3. SmolCC数据流与交互

### 3.1 用户交互流程
1. **初始化**: 加载工具集，生成动态系统提示
2. **输入解析**: 接收用户查询，分析意图
3. **工具选择**: 基于查询内容选择合适工具
4. **参数提取**: 从查询中提取工具参数
5. **执行操作**: 调用工具执行具体操作
6. **结果格式化**: 使用ToolOutput格式化输出
7. **状态更新**: 更新会话历史和系统状态

### 3.2 工具调用协议
```python
# 工具调用示例
tool_call = {
    "name": "view_tool",
    "arguments": {
        "filepath": "example.py",
        "start_line": 1,
        "end_line": 10
    }
}

# 执行结果
tool_result = {
    "success": True,
    "output": ToolOutput对象,
    "error": None
}
```

## 4. SmolCC整体工作流程

### 4.1 系统启动流程
1. **环境初始化**: 加载环境变量，设置工作目录
2. **工具注册**: 扫描并注册所有可用工具
3. **提示生成**: 动态生成包含当前环境的系统提示
4. **模型配置**: 配置LLM模型和API密钥
5. **会话启动**: 进入交互式或单次查询模式

### 4.2 查询处理流程
```
用户输入 → 意图分析 → 工具选择 → 参数提取 → 工具执行 → 结果格式化 → 输出显示
```

### 4.3 工具协同工作流程
- **文件操作链**: ViewTool → EditTool → 验证结果
- **搜索操作链**: LsTool → GrepTool → ViewTool
- **批量操作链**: GlobTool → 多文件EditTool调用
- **命令执行链**: BashTool → 系统命令执行

### 4.4 状态管理流程
- **会话状态**: 维护用户对话历史和工具调用记录
- **环境状态**: 跟踪当前工作目录和Git仓库状态
- **工具状态**: 管理持久化工具实例（如BashTool的Shell会话）

### 4.5 错误处理流程
- **工具错误**: 捕获工具执行异常，返回友好错误信息
- **参数错误**: 验证输入参数，提供详细错误说明
- **权限错误**: 检查文件系统权限，提示解决方案
- **网络错误**: 处理API调用失败，提供重试机制

### 4.6 性能优化流程
- **工具缓存**: 复用工具实例避免重复初始化
- **输出优化**: 智能截断大文件输出
- **并行处理**: 支持多工具并行执行
- **资源管理**: 监控系统资源使用，防止资源耗尽