# SmolAgents高级示例系统

## 1. 项目概述

本示例展示在SmolAgents框架基础上构建的高级应用系统，实现了从多模型集成到复杂执行环境的完整应用场景。

## 2. 实现特点

### 多模型集成架构
- 支持InferenceClientModel、LiteLLMModel、TransformersModel等多种后端
- 通过LiteLLMRouterModel实现多模型负载均衡
- 动态切换不同模型提供商和执行环境

### 高级工具系统
- 多功能工具集：天气查询、货币转换、新闻获取、时间查询等
- RAG集成：基于ChromaDB和BM25的检索增强生成
- SQL交互：直接与数据库交互的智能查询工具
- 结构化输出：通过MCP协议实现结构化数据输出

### 安全执行环境
- 沙箱执行：支持Docker、E2B、Modal、WebAssembly等多种执行环境
- 隔离运行：确保代码执行的安全性和可控性

### 异步与服务器集成
- 异步支持：在Starlette等异步框架中集成智能体
- Web服务：构建基于HTTP的智能体服务
- MCP协议：通过Model Control Protocol扩展智能体能力

### 高级应用场景
- 深度研究：OpenAI Deep Research的开源实现
- 计划定制：人机协作的计划定制与执行
- 性能评估：全面的智能体基准测试框架

## 3. 应用场景

- 多模型智能体系统
- 检索增强生成(RAG)应用
- 安全代码执行环境
- 异步Web服务集成
- 复杂研究任务自动化