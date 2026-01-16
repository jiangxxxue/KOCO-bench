# NeMo Project Overview

## 1. Overall Overview

NeMo is a framework for building and training large-scale language models, multimodal models, and speech models, integrated with TensorRT Model Optimizer for model optimization. Its core capability lies in providing a complete model training, inference, and optimization pipeline, supporting model optimization techniques such as quantization, pruning, and distillation. Through integration with the ModelOpt framework, NeMo can achieve efficient model compression and acceleration, particularly suitable for model deployment in production environments.

## 2. Core Features

### 2.1 Model Quantization
Supports multiple quantization algorithms, including FP8, INT8, INT4 precisions, achieving model compression and acceleration through ModelOpt's quantization tools.

### 2.2 Model Pruning
Provides structured pruning functionality, supporting width pruning (hidden layer dimensions, attention head count) and depth pruning (layer reduction), with layer selection based on importance metrics.

### 2.3 Knowledge Distillation
Implements teacher-student model knowledge transfer, supporting logits distillation and intermediate layer distillation, optimizing the training process through ModelOpt's distillation framework.

### 2.4 Model Export
Supports multiple export formats, including TensorRT-LLM, HuggingFace, and NeMo native formats, facilitating use in different deployment environments.

### 2.5 Inference Optimization
Achieves efficient inference acceleration through ModelOpt integration, supporting multiple hardware platforms and deployment scenarios.
