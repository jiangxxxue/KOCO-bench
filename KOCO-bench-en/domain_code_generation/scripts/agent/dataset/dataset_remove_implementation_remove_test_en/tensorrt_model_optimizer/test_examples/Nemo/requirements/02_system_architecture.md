# NeMo System Architecture and Module Design

## 1. NeMo ModelOpt Module File Structure

### 1.1 File Organization
```
nemo/collections/llm/modelopt/
├── __init__.py                    # ModelOpt module entry point
├── distill/                      # Knowledge distillation module
│   ├── __init__.py
│   ├── model.py                  # Distillation model implementation
│   ├── loss.py                   # Distillation loss functions
│   └── utils.py                  # Distillation utility functions
├── quantization/                 # Quantization module
│   ├── __init__.py
│   ├── quantizer.py              # Quantizer implementation
│   ├── quant_cfg_choices.py      # Quantization configuration choices
│   └── utils.py                 # Quantization utility functions
├── prune/                        # Pruning module
│   ├── __init__.py
│   └── pruner.py                # Pruner implementation
├── recipes/                      # Optimization recipe module
│   ├── __init__.py
│   ├── distillation_recipe.py   # Distillation recipe
│   └── prune_recipe.py          # Pruning recipe
├── speculative/                  # Speculative decoding module
│   ├── __init__.py
│   └── model_transform.py       # Model transformation
└── model_utils.py               # Model utility functions
```

### 1.2 Module Responsibility Division
- `distill/`: Responsible for knowledge distillation-related model construction, loss calculation, and training process management
- `quantization/`: Responsible for model quantization, including configuration selection, quantizer implementation, and export functionality
- `prune/`: Responsible for model pruning, including structured pruning and importance evaluation
- `recipes/`: Provides high-level optimization recipes, encapsulating common optimization workflows
- `speculative/`: Responsible for speculative decoding-related model transformation and optimization
- `model_utils.py`: Provides common model loading, saving, and state management functionality

## 2. NeMo ModelOpt Module Design

### 2.1 Quantization Module (`quantization/`)
The core functionality of this module is to implement model quantization. The `Quantizer` class is responsible for the entire quantization workflow, including model loading, calibration data preparation, quantization configuration generation, model quantization, and export. It supports multiple quantization algorithms (FP8, INT8, INT4, etc.), achieving efficient model compression through ModelOpt's quantization tools. The module also provides a quantization configuration selector that automatically selects the optimal quantization strategy based on model type and hardware characteristics.

### 2.2 Distillation Module (`distill/`)
This module implements knowledge distillation functionality. The `DistillationGPTModel` class inherits from GPTModel and integrates ModelOpt's distillation framework. It converts standard models to distillation models through `mtd.convert()`, supporting teacher-student architecture. The `_DistillationLossReduction` class is responsible for calculating distillation loss, including logits loss and intermediate layer loss. The module also provides distillation configuration loading and model adjustment functionality, ensuring compatibility with MCore architecture.

### 2.3 Pruning Module (`prune/`)
This module implements structured pruning functionality. The `prune_language_model()` function is the core interface, supporting width pruning (FFN hidden layers, attention heads, query groups) and depth pruning (layer reduction). It executes pruning operations through ModelOpt's pruning tool `mtp.prune()`, using importance metrics for layer selection. The module also provides pruning configuration validation and model saving functionality.

### 2.4 Recipe Module (`recipes/`)
This module provides high-level optimization recipes, encapsulating common optimization workflows. Distillation recipes and pruning recipes provide simplified interfaces, allowing users to execute complex optimization operations by providing only basic configurations. The recipe module is responsible for parameter validation, workflow orchestration, and result output.

## 3. NeMo ModelOpt Module Interaction and Data Flow

The NeMo ModelOpt workflow begins with users calling the corresponding optimization interface. The quantization workflow is executed through the `Quantizer` class, including model loading, calibration, quantization, and export. The distillation workflow is implemented through the `DistillationGPTModel` class, converting standard models to distillation models and calculating distillation loss through custom loss functions. The pruning workflow is executed through the `prune_language_model()` function, using ModelOpt's pruning tools for structured pruning. All modules perform underlying optimization operations through the ModelOpt framework, ensuring compatibility with the TensorRT ecosystem. Finally, optimized models can be exported in multiple formats, supporting different deployment environments.
