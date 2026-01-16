# TensorRT-Incubator Project Overview

## 1. Overall Overview

TensorRT-Incubator is a repository containing experimental projects related to TensorRT, primarily consisting of two core sub-projects: Tripy and MLIR-TensorRT. The project integrates TensorRT Model Optimizer (ModelOpt) for model quantization optimization, providing a Pythonic TensorRT frontend interface and MLIR dialect support. Its core capability lies in providing a debuggable, high-performance deep learning inference compilation framework that supports quantization, model conversion, and optimization workflows, particularly suitable for model deployment and optimization in production environments.

## 2. Core Features

### 2.1 Tripy - Pythonic TensorRT Frontend
Tripy is a debuggable, Pythonic TensorRT frontend that provides an intuitive API interface following ecosystem conventions. It supports eager mode for interactive debugging and provides friendly error messages and documentation. Through integration with the ModelOpt framework, Tripy enables efficient model quantization optimization.

### 2.2 MLIR-TensorRT - MLIR Dialect Support
Provides MLIR dialect to precisely model the TensorRT operator set, including static verification, type inference, optimization, and conversion capabilities. Supports conversion from StableHLO to TensorRT dialect, provides complete compiler infrastructure, and supports dynamic shapes and Python interface.

### 2.3 Model Quantization
Through integration with the ModelOpt framework, supports multiple quantization algorithms including INT8, INT4, FP8 precisions. Provides calibration functionality to determine scaling factors, supports weight quantization and activation quantization. Quantized models can be seamlessly integrated into Tripy's inference workflow.

### 2.4 Model Compilation and Optimization
Leverages TensorRT's optimization capabilities to achieve high-performance inference. Supports model compilation, shape inference, operation fusion, and other optimization techniques. Provides MLIR intermediate representation, supports cross-platform deployment and optimization.

### 2.5 Examples and Documentation
Provides rich example code and documentation, including nanoGPT quantization examples, user guides, and API documentation. Supports Jupyter Notebook format tutorials for easy learning and usage.

