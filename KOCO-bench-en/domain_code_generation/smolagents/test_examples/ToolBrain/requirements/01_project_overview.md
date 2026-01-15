# ToolBrain Intelligent Tool Learning Framework

## 1. Project Overview

ToolBrain is a lightweight framework for tool usage capability learning, built around the smolagents tool ecosystem. It provides unified tool description and validation, cross-framework adapter execution, and trainable data flows for various learning strategies. The system integrates multiple learning algorithms and adapter patterns, offering developers intelligent tool usage capability training and inference functionality. ToolBrain adopts a modular design, combining dynamic tool card generation with a rich adapter ecosystem to achieve efficient tool usage capability learning and application.

## 2. Core Features

### Tool Description and Prompt Construction
- Extracts unified metadata and parameter schemas from tool objects, outputting standardized tool cards
- Training sample generation with prompt assembly and minimum tool call constraints
- Batch tool card rendering and training example generation prompt construction

### Tool and Model Validation
- Strict type, callability, and duplicate name validation for tool lists
- Callable generation interface signature checking for external models
- Tool collection validation and model interface checking

### Cross-Framework Adaptation and Execution Tracking
- Unified adapter interface, compatible with native agents and LangChain agents
- Supports chat model and tool calling wrapping, producing trainable traces
- Base adapter abstraction and tool calling wrapping

### Learning Strategies and Training Optimization
- Multiple learning algorithm support, including DPO, GRPO, and supervised learning
- Low-rank adaptation fine-tuning, reducing memory overhead
- Efficient model training and inference capabilities

## 3. Technical Features

### Based on smolagents Tool Ecosystem
- Strongly coupled with tools and models, directly consuming and validating tool definitions
- Ensures tool constraint consistency between training and inference phases through unified tool cards
- Supports multiple tool types and model formats

### Trainable Data Flow with Minimal Integration Changes
- Produces high-fidelity traces through adapters, including thoughts, tool code, tool outputs, and final answers
- Non-invasive integration with various agents, preserving native calling paradigms
- Supports multiple training data formats and trace recording methods

### Lightweight and Efficient Training Optimization Path
- Supports low-rank adaptation refinement, reducing memory overhead
- Prompt construction and tool constraints directly usable for data synthesis and policy learning
- Multiple learning algorithm support, including direct preference optimization and group relative policy optimization

### Rich Adapter Ecosystem
- Base adapter abstraction, providing unified interface definition
- Native agent adapter, compatible with smolagents framework
- LangChain adapter, supporting LangChain ecosystem integration
