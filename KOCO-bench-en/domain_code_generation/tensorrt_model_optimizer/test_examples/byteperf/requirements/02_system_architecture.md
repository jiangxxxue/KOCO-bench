# ByteMLPerf Training Performance Testing System Architecture

## 1. Module File Structure

### 1.1 File Organization
```
byte_train_perf/
└── Megatron-LM/                  # Training framework based on Megatron-LM
    ├── megatron/                 # Core training framework
    │   ├── inference/            # Inference-related modules
    │   │   ├── algos/           # Algorithm implementations
    │   │   │   └── distillation.py  # Knowledge distillation algorithm
    │   │   ├── arguments.py     # Parameter configuration
    │   │   ├── checkpointing.py  # Checkpoint management
    │   │   └── gpt/             # GPT model related
    │   │       └── model_provider.py  # Model provider
    │   └── training/            # Training-related modules
    │       └── checkpointing.py  # Training checkpoints
    └── examples/                # Examples and tools
        └── export/              # Model export
            └── ptq_and_trtllm_export/  # PTQ and TensorRT-LLM export
                ├── text_generation_ptq.py  # PTQ quantization implementation
                └── README.md     # Usage instructions
```

### 1.2 Module Responsibilities
- `megatron/inference/algos/distillation.py`: Knowledge distillation algorithm implementation
- `megatron/inference/checkpointing.py`: Inference checkpoint management
- `megatron/training/checkpointing.py`: Training checkpoint management
- `examples/export/ptq_and_trtllm_export/`: Model quantization and export tools

## 2. Core Module Design

### 2.1 Knowledge Distillation Module (distillation.py)

Core Class: `LogitsAndIntermediatesLossBalancer`

Main Functions:
- Dynamic Loss Balancing: Automatically adjust loss weights during knowledge distillation
- Configuration Management: Support combined configuration of logits distillation and intermediate layer distillation
- MCore Adaptation: Handle configuration requirements specific to Megatron-Core architecture

Core Methods:
- `forward()`: Dynamically balance distillation loss and original loss
- `load_distillation_config()`: Load knowledge distillation configuration file
- `adjust_distillation_model_for_mcore()`: Adjust distillation model to adapt to MCore architecture

### 2.2 Model Export Module (ptq_and_trtllm_export/)

Core Function: Implement post-training quantization and TensorRT-LLM export

Main Components:
- `text_generation_ptq.py`: PTQ quantization implementation
- Support for multiple quantization strategies including INT8, FP8, INT4
- TensorRT-LLM format export

Key Features:
- Multiple quantization configuration support
- Custom calibration data support
- Hardware-specific optimization

### 2.3 Checkpoint Management Module

Core Function: Manage model state saving and loading during training and inference

Main Components:
- `megatron/inference/checkpointing.py`: Inference checkpoint management
- `megatron/training/checkpointing.py`: Training checkpoint management

Key Features:
- Distributed checkpoint support
- ModelOpt state saving
- Resume training from checkpoint support

### 2.4 Parameter Configuration Module

Core Function: Manage system parameters and model configuration

Main Components:
- `megatron/inference/arguments.py`: Inference parameter configuration
- `megatron/inference/gpt/model_provider.py`: GPT model configuration

Key Features:
- Command-line argument parsing
- Model configuration management
- Hardware configuration adaptation

## 3. Module Interaction and Data Flow

### 3.1 Knowledge Distillation Workflow
1. Configuration Loading: Load distillation configuration file, set teacher/student model parameters
2. Model Initialization: Create distillation model, configure loss functions and optimizers
3. Distillation Training: Execute knowledge distillation training, dynamically adjust loss weights
4. Model Export: Save optimized model, support subsequent quantization

### 3.2 Quantization Export Workflow
1. Model Loading: Load pre-trained model, prepare calibration dataset
2. Quantization Execution: Execute post-training quantization, generate quantized model
3. TensorRT-LLM Export: Convert to TensorRT-LLM format, optimize inference performance

### 3.3 Checkpoint Management Process
1. State Saving: Save model state, optimizer state, training progress
2. Distributed Synchronization: Synchronize checkpoint data in distributed environment
3. State Loading: Restore training state from checkpoint
4. Resume Training: Continue training or perform inference
