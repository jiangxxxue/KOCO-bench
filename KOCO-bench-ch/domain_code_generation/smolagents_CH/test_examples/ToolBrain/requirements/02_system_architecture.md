# ToolBrain系统架构与模块设计

## 1. ToolBrain模块文件结构

### 1.1 文件组织
```
code/
└── toolbrain/
    ├── brain.py                     # 训练/执行大脑（策略调度、轨迹组织）
    ├── factory.py                   # 工厂与装配（构建适配器/模型/策略）
    ├── config.py                    # 运行/训练配置
    ├── core_types.py                # 核心类型（Trace/Turn/ChatSegment 等）
    ├── models.py                    # 模型封装（高效训练扩展）
    ├── prompt.py                    # 工具卡片/校验/示例生成提示
    ├── retriever.py                 # 工具/知识检索
    ├── rewards.py                   # 奖励函数与组合
    ├── adapters/
    │   ├── __init__.py
    │   ├── base_adapter.py          # 统一的适配器抽象
    │   ├── smolagent/               # 基于原生代理的适配器
    │   │   └── smolagent_adapter.py
    │   └── langchain/               # 基于LangChain的适配器
    │       ├── __init__.py
    │       ├── hf_tool_wrapper.py   # 聊天模型的工具调用包装
    │       └── langchain_adapter.py # 适配器与封装
    └── learning/
        ├── __init__.py
        ├── policy.py                # 策略接口/通用逻辑
        ├── dpo/                     # DPO算法实现
        │   ├── __init__.py
        │   ├── algo.py
        │   └── utils.py
        ├── grpo/                    # GRPO算法实现
        │   ├── __init__.py
        │   ├── algo.py
        │   ├── losses.py
        │   └── utils.py
        └── supervised/              # 监督式学习
            ├── __init__.py
            └── algo.py
```

### 1.2 模块职责划分
- brain.py: 训练/执行大脑，负责策略调度和轨迹组织
- factory.py: 工厂与装配，构建适配器、模型和策略
- config.py: 运行/训练配置管理
- core_types.py: 核心类型定义，包括轨迹、轮次和聊天分段
- models.py: 模型封装，提供高效训练扩展
- prompt.py: 工具卡片渲染、校验和示例生成提示
- retriever.py: 工具和知识检索功能
- rewards.py: 奖励函数与组合
- adapters/: 适配器层，提供跨框架兼容性
- learning/: 学习策略实现，包括多种算法

## 2. ToolBrain模块设计

### 2.1 训练/执行大脑 (brain.py)

核心类: `Brain`

主要方法:
- `__init__(config)`: 初始化大脑配置
- `run(query)`: 执行查询并返回结果
- `train(data)`: 训练模型
- `organize_trace()`: 组织轨迹数据
- `generate_training_examples()`: 生成训练样本
- `is_agent_supported()`: 检查代理支持情况
- `_get_adapter_for_agent()`: 获取代理适配器

主要功能:
- 策略调度和执行
- 轨迹数据组织和管理
- 训练流程控制
- 模型推理协调
- 训练样本生成
- 代理类型检测和适配器选择

### 2.2 工厂与装配 (factory.py)

核心函数:
- `create_agent(model_id, tools, **kwargs)`: 创建代理实例
- `create_optimized_model(model_id, **kwargs)`: 创建优化模型
- `create_adapter(config)`: 创建适配器实例
- `create_model(config)`: 创建模型实例
- `create_strategy(config)`: 创建学习策略
- `assemble_components(config)`: 装配所有组件

主要功能:
- 组件创建和初始化
- 配置解析和应用
- 依赖注入和管理
- 组件装配和集成
- 代理实例创建和配置
- 模型优化和加载

### 2.3 核心类型 (core_types.py)

核心类:
- `Trace`: 轨迹数据结构
- `Turn`: 轮次数据结构
- `ChatSegment`: 聊天分段结构
- `ParsedCompletion`: 解析完成结构

主要方法:
- `to_dict()`: 转换为字典格式
- `from_dict()`: 从字典创建实例
- `validate()`: 验证数据完整性
- `merge()`: 合并多个实例

功能特点:
- 数据结构标准化
- 序列化和反序列化支持
- 数据完整性验证
- 多实例合并能力

### 2.4 模型封装 (models.py)

核心类: `UnslothModel`

主要方法:
- `__init__(config)`: 初始化模型配置
- `load_model()`: 加载模型
- `generate()`: 生成响应
- `train()`: 训练模型

功能特点:
- 高效训练支持
- 低内存占用
- 快速推理能力
- 兼容多种模型格式
- 量化模型支持
- 自适应批处理

### 2.5 工具提示与校验 (prompt.py)

核心函数:
- `_extract_tool_meta(tool)`: 从工具对象抽取元数据
- `tool_to_card(tool)`: 将工具转换为卡片格式
- `tools_to_card(tools)`: 批量转换工具为卡片
- `validate_tools(tools)`: 验证工具集合
- `validate_model(model)`: 验证模型接口
- `build_prompt_to_generate_training_examples()`: 构建训练示例生成提示

主要功能:
- 工具元数据提取和标准化
- 工具卡片渲染
- 工具集合验证
- 模型接口验证
- 训练提示构建
- 多工具批量处理

### 2.6 适配器层

#### 基础适配器 (adapters/base_adapter.py)

核心类: `BaseAgentAdapter`

主要方法:
- `__init__(config)`: 初始化适配器
- `run(query)`: 运行代理
- `get_trainable_model()`: 获取可训练模型
- `validate_config()`: 验证配置

功能特点:
- 统一接口定义
- 抽象基类实现
- 配置验证机制
- 模型访问接口

#### SmolAgent适配器 (adapters/smolagent/smolagent_adapter.py)

核心类: `SmolAgentAdapter`

主要方法:
- `__init__(agent)`: 初始化适配器
- `run(query)`: 运行代理并提取轨迹
- `_extract_trace_from_memory()`: 从记忆中提取轨迹
- `_parse_missing_parts()`: 解析缺失部分
- `_segment_text_with_assistant()`: 分段文本并标注角色
- `_build_input_for_rl_from_memory()`: 构建强化学习输入

功能特点:
- smolagents框架集成
- 高保真轨迹提取
- 记忆步骤解析
- 强化学习输入构建
- 文本分段和角色标注

#### LangChain适配器 (adapters/langchain/langchain_adapter.py)

核心类: `LangChainAdapter`

主要方法:
- `__init__(config)`: 初始化适配器
- `create_huggingface_chat_model()`: 创建聊天模型
- `run(query)`: 运行代理
- `get_trainable_model()`: 获取可训练模型
- `_set_lora_finetuning()`: 设置低秩适应微调

功能特点:
- LangChain框架集成
- 聊天模型创建
- 低秩适应支持
- 轨迹重建能力

#### 工具调用包装 (adapters/langchain/hf_tool_wrapper.py)

核心类: `HuggingFaceToolWrapper`

主要方法:
- `__init__(model, tools)`: 初始化包装器
- `invoke_with_tools(query)`: 使用工具调用
- `parse_tool_call()`: 解析工具调用
- `execute_tool()`: 执行工具

功能特点:
- 工具调用封装
- JSON格式支持
- 流式处理能力
- 错误处理机制

### 2.7 学习策略

#### 策略接口 (learning/policy.py)

核心类: `Policy`

主要方法:
- `__init__(config)`: 初始化策略
- `compute_loss()`: 计算损失
- `update()`: 更新参数
- `evaluate()`: 评估性能

功能特点:
- 统一策略接口
- 损失计算抽象
- 参数更新机制
- 性能评估框架

#### DPO算法 (learning/dpo/algo.py)

核心类: `DPOPolicy`

主要方法:
- `__init__(config)`: 初始化DPO策略
- `compute_loss()`: 计算DPO损失
- `update()`: 更新模型参数
- `sample()`: 采样响应

功能特点:
- 直接偏好优化
- 对比学习机制
- 高效参数更新
- 稳定训练过程

#### GRPO算法 (learning/grpo/algo.py)

核心类: `GRPOPolicy`

主要方法:
- `__init__(config)`: 初始化GRPO策略
- `compute_loss()`: 计算GRPO损失
- `update()`: 更新策略参数
- `sample()`: 采样动作

功能特点:
- 群体相对策略优化
- 多样本对比学习
- 高效梯度计算
- 稳定收敛特性

#### 监督学习 (learning/supervised/algo.py)

核心类: `SupervisedPolicy`

主要方法:
- `__init__(config)`: 初始化监督策略
- `compute_loss()`: 计算监督损失
- `update()`: 更新模型参数
- `predict()`: 预测输出

功能特点:
- 标准监督学习
- 标签匹配优化
- 梯度反向传播
- 快速收敛能力

## 3. ToolBrain数据流与交互

### 3.1 执行/训练数据流
1. **初始化**: 加载配置，初始化组件
2. **查询处理**: 接收用户查询，解析意图
3. **工具选择**: 根据查询选择合适工具
4. **执行操作**: 调用工具执行具体任务
5. **轨迹记录**: 记录执行过程和结果
6. **结果返回**: 格式化并返回结果
7. **训练更新**: 根据反馈更新模型

### 3.2 工具调用协议
```python
# 工具调用示例
tool_call = {
    "name": "example_tool",
    "arguments": {
        "param1": "value1",
        "param2": "value2"
    }
}

# 执行结果
tool_result = {
    "success": True,
    "output": "工具执行结果",
    "error": None
}
```

## 4. ToolBrain整体工作流程

### 4.1 系统启动流程
1. **环境初始化**: 加载环境变量和配置
2. **组件创建**: 创建适配器、模型和策略
3. **工具注册**: 注册和验证可用工具
4. **模型加载**: 加载预训练模型
5. **系统就绪**: 进入就绪状态，等待查询

### 4.2 查询处理流程
```
用户输入 → 意图解析 → 工具选择 → 参数提取 → 工具执行 → 结果格式化 → 输出返回
```

### 4.3 训练流程
- **数据准备**: 收集和预处理训练数据
- **批次构建**: 构建训练批次
- **前向传播**: 计算模型输出
- **损失计算**: 计算训练损失
- **反向传播**: 计算梯度并更新参数
- **模型评估**: 评估模型性能

### 4.4 状态管理流程
- **会话状态**: 维护用户对话历史
- **模型状态**: 跟踪模型参数和配置
- **工具状态**: 管理工具实例和状态
- **训练状态**: 监控训练进度和指标

### 4.5 错误处理流程
- **工具错误**: 捕获工具执行异常
- **模型错误**: 处理模型推理失败
- **配置错误**: 验证配置参数
- **系统错误**: 处理系统级异常

### 4.6 性能优化流程
- **模型优化**: 使用高效模型架构
- **内存管理**: 优化内存使用
- **并行处理**: 支持多线程/多进程
- **缓存机制**: 缓存常用计算结果