# TensorRT-Incubator 项目概述

## 1. 总体概述

TensorRT-Incubator 是一个包含与 TensorRT 相关的实验性项目的仓库，主要包含 Tripy 和 MLIR-TensorRT 两个核心子项目。该项目集成了 TensorRT Model Optimizer (ModelOpt) 进行模型量化优化，提供了 Pythonic 的 TensorRT 前端接口和 MLIR 方言支持。其核心能力在于提供可调试、高性能的深度学习推理编译框架，支持量化、模型转换和优化流程，特别适用于生产环境中的模型部署和优化。

## 2. 核心功能

### 2.1 Tripy - Pythonic TensorRT 前端
Tripy 是一个可调试的、Pythonic 的 TensorRT 前端，提供了直观的 API 接口，遵循生态系统约定。支持即时模式（eager mode）进行交互式调试，提供友好的错误信息和文档。通过集成 ModelOpt 框架，Tripy 能够实现高效的模型量化优化。

### 2.2 MLIR-TensorRT - MLIR 方言支持
提供 MLIR 方言来精确建模 TensorRT 操作符集，包括静态验证、类型推断、优化和转换功能。支持从 StableHLO 到 TensorRT 方言的转换，提供完整的编译器基础设施，支持动态形状和 Python 接口。

### 2.3 模型量化 (Quantization)
通过集成 ModelOpt 框架，支持多种量化算法，包括 INT8、INT4、FP8 等精度。提供校准功能来确定缩放因子，支持权重量化和激活量化。量化后的模型可以无缝集成到 Tripy 的推理流程中。

### 2.4 模型编译与优化
利用 TensorRT 的优化能力实现高性能推理。支持模型编译、形状推断、操作融合等优化技术。提供 MLIR 中间表示，支持跨平台部署和优化。

### 2.5 示例与文档
提供丰富的示例代码和文档，包括 nanoGPT 量化示例、用户指南和 API 文档。支持 Jupyter Notebook 格式的教程，便于学习和使用。

