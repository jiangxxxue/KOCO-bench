# DeepSearchAgents深度搜索智能体系统

## 1. 系统概述

DeepSearchAgents是基于SmolAgents框架构建的深度搜索智能体系统，通过扩展SmolAgents的核心功能实现高级信息检索、内容分析和知识发现。系统支持ReAct和CodeAct两种智能体模式，并提供管理器智能体进行层次化编排，通过多模型路由和工具集成实现复杂的深度搜索任务。

## 2. 工作流程

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
8:    // ReAct模式：推理-行动循环
9:    for step = 1 to max_steps do
10:       // 规划阶段（每planning_interval步）
11:       if step % planning_interval == 0 then
12:          planning ← model_router.orchestrator_model(context)
13:          update_agent_state(planning)
14:       end if
15:       
16:       // 工具调用阶段
17:       action ← model_router.search_model(context)
18:       result ← execute_tool(action, T)
19:       update_memory(action, result)
20:    end for
21: 
22: elif A == "codact" then
23:    agent ← CodeActAgent(model_router, T, max_steps)
24:    // CodeAct模式：代码执行-分析循环
25:    for step = 1 to max_steps do
26:       // 生成代码执行计划
27:       code ← model_router.search_model(context)
28:       result ← execute_code(code, authorized_imports)
29:       update_memory(code, result)
30:       
31:       // 规划阶段（每planning_interval步）
32:       if step % planning_interval == 0 then
33:          planning ← model_router.orchestrator_model(context)
34:          update_agent_state(planning)
35:       end if
36:    end for
37: 
38: elif A == "manager" then
39:    agent ← ManagerAgent(model_router, managed_agents)
40:    // 管理器模式：智能体编排
41:    for step = 1 to max_steps do
42:       // 分析任务并选择合适的子智能体
43:       task_analysis ← model_router.orchestrator_model(context)
44:       selected_agent ← select_agent(task_analysis, managed_agents)
45:       result ← selected_agent.execute(subtask)
46:       update_memory(task_analysis, result)
47:    end for
48: end if
49: 
50: // 生成最终答案
51: final_answer ← generate_structured_output(memory, sources)
52: return final_answer
```

工作流程中的关键函数说明：
- `DebuggingToolCallingAgent`: 扩展SmolAgents的ToolCallingAgent，添加空答案检测和重试逻辑
- `CodeActAgent`: 扩展SmolAgents的CodeAgent，增加安全代码验证和深度搜索功能
- `ManagerAgent`: 扩展CodeAgent实现层次化智能体编排
- `MultiModelRouter`: 根据提示内容路由到不同模型（搜索模型或编排器模型）
- `execute_tool`: 执行搜索、读取、重排序等工具操作
- `execute_code`: 在安全沙箱中执行Python代码
- `generate_structured_output`: 生成包含标题、内容和来源的结构化答案

## 3. 应用场景

### 网络搜索与信息检索
- 输入: 用户查询问题
- 输出: 包含来源链接的结构化答案
- 搜索方式: Google搜索(Serper)、X.com搜索(xAI)、混合搜索
- 数据处理: 内容抓取、文本分块、嵌入和重排序

### 学术研究与文献调研
- 输入: 学术研究主题
- 输出: 综合性研究报告和文献引用
- 数据来源: arXiv、bioRxiv、medRxiv、Google Scholar
- 处理方式: 论文检索、元数据合并、内容解析

### 代码生成与数据分析
- 输入: 数据处理或分析任务
- 输出: 可执行的Python代码和分析结果
- 执行环境: 安全的本地Python执行器
- 支持库: pandas, numpy, requests, matplotlib等

### 多智能体协作任务
- 输入: 复杂的多步骤任务
- 输出: 专业智能体协作完成的综合结果
- 智能体分工: Web搜索专家、数据分析专家
- 协调方式: 管理器智能体动态分配子任务