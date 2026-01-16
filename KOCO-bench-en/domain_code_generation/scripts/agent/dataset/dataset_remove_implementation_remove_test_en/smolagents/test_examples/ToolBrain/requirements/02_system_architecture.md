# ToolBrain System Architecture and Module Design

## 1. ToolBrain Module File Structure

### 1.1 File Organization
```
code/
└── toolbrain/
    ├── brain.py                     # Training/execution brain (policy scheduling, trace organization)
    ├── factory.py                   # Factory and assembly (build adapters/models/policies)
    ├── config.py                    # Runtime/training configuration
    ├── core_types.py                # Core types (Trace/Turn/ChatSegment, etc.)
    ├── models.py                    # Model encapsulation (efficient training extensions)
    ├── prompt.py                    # Tool cards/validation/example generation prompts
    ├── retriever.py                 # Tool/knowledge retrieval
    ├── rewards.py                   # Reward functions and composition
    ├── adapters/
    │   ├── __init__.py
    │   ├── base_adapter.py          # Unified adapter abstraction
    │   ├── smolagent/               # Native agent-based adapters
    │   │   └── smolagent_adapter.py
    │   └── langchain/               # LangChain-based adapters
    │       ├── __init__.py
    │       ├── hf_tool_wrapper.py   # Tool calling wrapper for chat models
    │       └── langchain_adapter.py # Adapter and wrapper
    └── learning/
        ├── __init__.py
        ├── policy.py                # Policy interface/common logic
        ├── dpo/                     # DPO algorithm implementation
        │   ├── __init__.py
        │   ├── algo.py
        │   └── utils.py
        ├── grpo/                    # GRPO algorithm implementation
        │   ├── __init__.py
        │   ├── algo.py
        │   ├── losses.py
        │   └── utils.py
        └── supervised/              # Supervised learning
            ├── __init__.py
            └── algo.py
```

### 1.2 Module Responsibilities
- brain.py: Training/execution brain, responsible for policy scheduling and trace organization
- factory.py: Factory and assembly, building adapters, models, and policies
- config.py: Runtime/training configuration management
- core_types.py: Core type definitions, including traces, turns, and chat segments
- models.py: Model encapsulation, providing efficient training extensions
- prompt.py: Tool card rendering, validation, and example generation prompts
- retriever.py: Tool and knowledge retrieval functionality
- rewards.py: Reward functions and composition
- adapters/: Adapter layer, providing cross-framework compatibility
- learning/: Learning strategy implementations, including multiple algorithms

## 2. ToolBrain Module Design

### 2.1 Training/Execution Brain (brain.py)

Core class: `Brain`

Main methods:
- `__init__(config)`: Initialize brain configuration
- `run(query)`: Execute query and return results
- `train(data)`: Train model
- `organize_trace()`: Organize trace data
- `generate_training_examples()`: Generate training samples
- `is_agent_supported()`: Check agent support status
- `_get_adapter_for_agent()`: Get agent adapter

Main functionality:
- Policy scheduling and execution
- Trace data organization and management
- Training process control
- Model inference coordination
- Training sample generation
- Agent type detection and adapter selection

### 2.2 Factory and Assembly (factory.py)

Core functions:
- `create_agent(model_id, tools, **kwargs)`: Create agent instance
- `create_optimized_model(model_id, **kwargs)`: Create optimized model
- `create_adapter(config)`: Create adapter instance
- `create_model(config)`: Create model instance
- `create_strategy(config)`: Create learning strategy
- `assemble_components(config)`: Assemble all components

Main functionality:
- Component creation and initialization
- Configuration parsing and application
- Dependency injection and management
- Component assembly and integration
- Agent instance creation and configuration
- Model optimization and loading

### 2.3 Core Types (core_types.py)

Core classes:
- `Trace`: Trace data structure
- `Turn`: Turn data structure
- `ChatSegment`: Chat segment structure
- `ParsedCompletion`: Parsed completion structure

Main methods:
- `to_dict()`: Convert to dictionary format
- `from_dict()`: Create instance from dictionary
- `validate()`: Validate data integrity
- `merge()`: Merge multiple instances

Feature highlights:
- Data structure standardization
- Serialization and deserialization support
- Data integrity validation
- Multi-instance merging capability

### 2.4 Model Encapsulation (models.py)

Core class: `UnslothModel`

Main methods:
- `__init__(config)`: Initialize model configuration
- `load_model()`: Load model
- `generate()`: Generate response
- `train()`: Train model

Feature highlights:
- Efficient training support
- Low memory footprint
- Fast inference capability
- Compatible with multiple model formats
- Quantized model support
- Adaptive batching

### 2.5 Tool Prompts and Validation (prompt.py)

Core functions:
- `_extract_tool_meta(tool)`: Extract metadata from tool object
- `tool_to_card(tool)`: Convert tool to card format
- `tools_to_card(tools)`: Batch convert tools to cards
- `validate_tools(tools)`: Validate tool collection
- `validate_model(model)`: Validate model interface
- `build_prompt_to_generate_training_examples()`: Build training example generation prompt

Main functionality:
- Tool metadata extraction and standardization
- Tool card rendering
- Tool collection validation
- Model interface validation
- Training prompt construction
- Multi-tool batch processing

### 2.6 Adapter Layer

#### Base Adapter (adapters/base_adapter.py)

Core class: `BaseAgentAdapter`

Main methods:
- `__init__(config)`: Initialize adapter
- `run(query)`: Run agent
- `get_trainable_model()`: Get trainable model
- `validate_config()`: Validate configuration

Feature highlights:
- Unified interface definition
- Abstract base class implementation
- Configuration validation mechanism
- Model access interface

#### SmolAgent Adapter (adapters/smolagent/smolagent_adapter.py)

Core class: `SmolAgentAdapter`

Main methods:
- `__init__(agent)`: Initialize adapter
- `run(query)`: Run agent and extract trace
- `_extract_trace_from_memory()`: Extract trace from memory
- `_parse_missing_parts()`: Parse missing parts
- `_segment_text_with_assistant()`: Segment text and annotate roles
- `_build_input_for_rl_from_memory()`: Build reinforcement learning input

Feature highlights:
- smolagents framework integration
- High-fidelity trace extraction
- Memory step parsing
- Reinforcement learning input construction
- Text segmentation and role annotation

#### LangChain Adapter (adapters/langchain/langchain_adapter.py)

Core class: `LangChainAdapter`

Main methods:
- `__init__(config)`: Initialize adapter
- `create_huggingface_chat_model()`: Create chat model
- `run(query)`: Run agent
- `get_trainable_model()`: Get trainable model
- `_set_lora_finetuning()`: Set low-rank adaptation fine-tuning

Feature highlights:
- LangChain framework integration
- Chat model creation
- Low-rank adaptation support
- Trace reconstruction capability

#### Tool Calling Wrapper (adapters/langchain/hf_tool_wrapper.py)

Core class: `HuggingFaceToolWrapper`

Main methods:
- `__init__(model, tools)`: Initialize wrapper
- `invoke_with_tools(query)`: Invoke with tools
- `parse_tool_call()`: Parse tool call
- `execute_tool()`: Execute tool

Feature highlights:
- Tool calling encapsulation
- JSON format support
- Streaming processing capability
- Error handling mechanism

### 2.7 Learning Strategies

#### Policy Interface (learning/policy.py)

Core class: `Policy`

Main methods:
- `__init__(config)`: Initialize policy
- `compute_loss()`: Compute loss
- `update()`: Update parameters
- `evaluate()`: Evaluate performance

Feature highlights:
- Unified policy interface
- Loss computation abstraction
- Parameter update mechanism
- Performance evaluation framework

#### DPO Algorithm (learning/dpo/algo.py)

Core class: `DPOPolicy`

Main methods:
- `__init__(config)`: Initialize DPO policy
- `compute_loss()`: Compute DPO loss
- `update()`: Update model parameters
- `sample()`: Sample responses

Feature highlights:
- Direct preference optimization
- Contrastive learning mechanism
- Efficient parameter updates
- Stable training process

#### GRPO Algorithm (learning/grpo/algo.py)

Core class: `GRPOPolicy`

Main methods:
- `__init__(config)`: Initialize GRPO policy
- `compute_loss()`: Compute GRPO loss
- `update()`: Update policy parameters
- `sample()`: Sample actions

Feature highlights:
- Group relative policy optimization
- Multi-sample contrastive learning
- Efficient gradient computation
- Stable convergence characteristics

#### Supervised Learning (learning/supervised/algo.py)

Core class: `SupervisedPolicy`

Main methods:
- `__init__(config)`: Initialize supervised policy
- `compute_loss()`: Compute supervised loss
- `update()`: Update model parameters
- `predict()`: Predict output

Feature highlights:
- Standard supervised learning
- Label matching optimization
- Gradient backpropagation
- Fast convergence capability

## 3. ToolBrain Data Flow and Interactions

### 3.1 Execution/Training Data Flow
1. **Initialization**: Load configuration, initialize components
2. **Query Processing**: Receive user query, parse intent
3. **Tool Selection**: Select appropriate tools based on query
4. **Execute Operations**: Call tools to execute specific tasks
5. **Trace Recording**: Record execution process and results
6. **Return Results**: Format and return results
7. **Training Update**: Update model based on feedback

### 3.2 Tool Calling Protocol
```python
# Tool call example
tool_call = {
    "name": "example_tool",
    "arguments": {
        "param1": "value1",
        "param2": "value2"
    }
}

# Execution result
tool_result = {
    "success": True,
    "output": "Tool execution result",
    "error": None
}
```

## 4. ToolBrain Overall Workflow

### 4.1 System Startup Flow
1. **Environment Initialization**: Load environment variables and configuration
2. **Component Creation**: Create adapters, models, and policies
3. **Tool Registration**: Register and validate available tools
4. **Model Loading**: Load pre-trained models
5. **System Ready**: Enter ready state, waiting for queries

### 4.2 Query Processing Flow
```
User Input → Intent Parsing → Tool Selection → Parameter Extraction → Tool Execution → Result Formatting → Output Return
```

### 4.3 Training Flow
- **Data Preparation**: Collect and preprocess training data
- **Batch Construction**: Build training batches
- **Forward Pass**: Compute model outputs
- **Loss Computation**: Calculate training loss
- **Backward Pass**: Compute gradients and update parameters
- **Model Evaluation**: Evaluate model performance

### 4.4 State Management Flow
- **Session State**: Maintain user conversation history
- **Model State**: Track model parameters and configuration
- **Tool State**: Manage tool instances and state
- **Training State**: Monitor training progress and metrics

### 4.5 Error Handling Flow
- **Tool Errors**: Capture tool execution exceptions
- **Model Errors**: Handle model inference failures
- **Configuration Errors**: Validate configuration parameters
- **System Errors**: Handle system-level exceptions

### 4.6 Performance Optimization Flow
- **Model Optimization**: Use efficient model architectures
- **Memory Management**: Optimize memory usage
- **Parallel Processing**: Support multi-threading/multi-processing
- **Caching Mechanism**: Cache frequently used computation results
