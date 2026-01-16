# NeMo 系统架构与模块设计

## 1. NeMo ModelOpt 模块文件结构

### 1.1 文件组织
```
nemo/collections/llm/modelopt/
├── __init__.py                    # ModelOpt 模块入口
├── distill/                      # 知识蒸馏模块
│   ├── __init__.py
│   ├── model.py                  # 蒸馏模型实现
│   ├── loss.py                   # 蒸馏损失函数
│   └── utils.py                  # 蒸馏工具函数
├── quantization/                 # 量化模块
│   ├── __init__.py
│   ├── quantizer.py              # 量化器实现
│   ├── quant_cfg_choices.py      # 量化配置选择
│   └── utils.py                 # 量化工具函数
├── prune/                        # 剪枝模块
│   ├── __init__.py
│   └── pruner.py                # 剪枝器实现
├── recipes/                      # 优化配方模块
│   ├── __init__.py
│   ├── distillation_recipe.py   # 蒸馏配方
│   └── prune_recipe.py          # 剪枝配方
├── speculative/                  # 推测解码模块
│   ├── __init__.py
│   └── model_transform.py       # 模型转换
└── model_utils.py               # 模型工具函数
```

### 1.2 模块职责划分
- `distill/`: 负责知识蒸馏相关的模型构建、损失计算和训练流程管理
- `quantization/`: 负责模型量化，包括配置选择、量化器实现和导出功能
- `prune/`: 负责模型剪枝，包括结构化剪枝和重要性评估
- `recipes/`: 提供高级的优化配方，封装常用的优化流程
- `speculative/`: 负责推测解码相关的模型转换和优化
- `model_utils.py`: 提供通用的模型加载、保存和状态管理功能

## 2. NeMo ModelOpt 模块设计

### 2.1 量化模块 (`quantization/`)
该模块的核心功能是实现模型量化。`Quantizer` 类负责整个量化流程，包括模型加载、校准数据准备、量化配置生成、模型量化和导出。它支持多种量化算法（FP8、INT8、INT4等），通过 ModelOpt 的量化工具实现高效的模型压缩。模块还提供量化配置选择器，根据模型类型和硬件特性自动选择最优的量化策略。

### 2.2 蒸馏模块 (`distill/`)
该模块实现知识蒸馏功能。`DistillationGPTModel` 类继承自 GPTModel，集成了 ModelOpt 的蒸馏框架。它通过 `mtd.convert()` 将标准模型转换为蒸馏模型，支持教师-学生架构。`_DistillationLossReduction` 类负责计算蒸馏损失，包括 logits 损失和中间层损失。模块还提供蒸馏配置加载和模型调整功能，确保与 MCore 架构的兼容性。

### 2.3 剪枝模块 (`prune/`)
该模块实现结构化剪枝功能。`prune_language_model()` 函数是核心接口，支持宽度剪枝（FFN隐藏层、注意力头、查询组）和深度剪枝（层数减少）。它通过 ModelOpt 的剪枝工具 `mtp.prune()` 执行剪枝操作，使用重要性指标进行层选择。模块还提供剪枝配置验证和模型保存功能。

### 2.4 配方模块 (`recipes/`)
该模块提供高级的优化配方，封装常用的优化流程。蒸馏配方和剪枝配方提供简化的接口，用户只需提供基本配置即可执行复杂的优化操作。配方模块负责参数验证、流程编排和结果输出。

## 3. NeMo ModelOpt 模块交互与数据流

NeMo ModelOpt 的工作流程始于用户调用相应的优化接口。量化流程通过 `Quantizer` 类执行，包括模型加载、校准、量化和导出。蒸馏流程通过 `DistillationGPTModel` 类实现，将标准模型转换为蒸馏模型，通过自定义的损失函数计算蒸馏损失。剪枝流程通过 `prune_language_model()` 函数执行，使用 ModelOpt 的剪枝工具进行结构化剪枝。所有模块都通过 ModelOpt 框架进行底层优化操作，确保与 TensorRT 生态系统的兼容性。最终，优化后的模型可以导出为多种格式，支持不同的部署环境。

