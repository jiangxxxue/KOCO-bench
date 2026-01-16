# TensorRT-Incubator System Architecture and Module Design

## 1. TensorRT-Incubator Project File Structure

### 1.1 Project Organization
```
TensorRT-Incubator/
├── code/
│   ├── tripy/                      # Tripy Python frontend project
│   │   ├── nvtripy/                # Core Python package
│   │   │   ├── __init__.py         # Package entry point
│   │   │   ├── frontend/           # Frontend module
│   │   │   │   ├── module/         # Module definitions
│   │   │   │   ├── ops/            # Operator definitions
│   │   │   │   └── tensor.py      # Tensor implementation
│   │   │   ├── backend/            # Backend module
│   │   │   │   ├── api/           # API interface
│   │   │   │   └── mlir/          # MLIR backend
│   │   │   ├── export.py          # Export functionality
│   │   │   └── config.py          # Configuration management
│   │   ├── examples/               # Example code
│   │   │   └── nanogpt/           # nanoGPT example
│   │   │       ├── quantization.py # ModelOpt quantization example
│   │   │       └── weight_loader.py # Weight loader
│   │   ├── docs/                   # Documentation directory
│   │   │   └── pre0_user_guides/  # User guides
│   │   │       └── 02-quantization.md # Quantization guide
│   │   ├── tests/                  # Test code
│   │   └── pyproject.toml         # Project configuration
│   └── mlir-tensorrt/              # MLIR-TensorRT project
│       ├── tensorrt/               # TensorRT dialect
│       ├── compiler/               # Compiler implementation
│       ├── executor/               # Executor
│       └── integrations/           # Integration modules
└── requirements/                   # Requirements documentation directory
```

### 1.2 Module Responsibilities
- `tripy/`: Provides Pythonic TensorRT frontend interface, supports model definition, compilation, and inference
- `tripy/nvtripy/frontend/`: Responsible for frontend API definition, including modules, operators, and tensors
- `tripy/nvtripy/backend/`: Responsible for backend implementation, including MLIR conversion and TensorRT API calls
- `tripy/examples/nanogpt/`: Provides ModelOpt quantization integration examples
- `mlir-tensorrt/`: Provides MLIR dialect and compiler infrastructure

## 2. Tripy ModelOpt Integration Module Design

### 2.1 Quantization Module (`examples/nanogpt/quantization.py`)
The core functionality of this module is to integrate ModelOpt for model quantization. The `modelopt_quantize()` function handles the entire quantization workflow, including model loading, quantization configuration selection, calibration data preparation, and model quantization. It supports multiple quantization modes (INT8, INT4, FP8) and achieves efficient model compression through ModelOpt's `mtq.quantize()` function. The module also provides calibration loop creation functionality, using `create_forward_loop()` to generate calibration data from datasets.

### 2.2 Weight Loading Module (`examples/nanogpt/weight_loader.py`)
This module implements loading and conversion of quantized weights. The `load_quant_weights_from_hf()` function is responsible for loading quantized weights and scaling factors from HuggingFace models. It obtains the quantized model by calling `modelopt_quantize()`, then extracts the quantizer's `_amax` values and converts them to scaling factors. The module also handles weight transposition and format conversion to ensure compatibility with Tripy's module interface.

### 2.3 Documentation Module (`docs/pre0_user_guides/02-quantization.md`)
This module provides quantization usage guide, explaining how to use ModelOpt for post-training quantization. The documentation details the calibration process, quantization configuration settings, scaling factor extraction, and methods for loading into Tripy modules. It provides complete code examples and best practices.

## 3. TensorRT-Incubator ModelOpt Module Interaction and Data Flow

The TensorRT-Incubator ModelOpt integration workflow begins with the user calling the `modelopt_quantize()` function. The quantization process first loads the HuggingFace model, then selects the appropriate configuration based on the quantization mode (INT8_DEFAULT_CFG, INT4_AWQ_CFG, or FP8_DEFAULT_CFG). It uses `create_forward_loop()` to create a calibration loop, generating calibration samples from datasets (such as cnn_dailymail). ModelOpt's `mtq.quantize()` function is called to perform quantization, replacing linear layers with `QuantLinear` modules containing calibration information. After quantization is complete, `load_quant_weights_from_hf()` extracts quantized weights and scaling factors, converts them to Tripy format, and loads them into the Tripy model. Finally, the quantized model can be converted to a TensorRT engine through Tripy's compilation interface to achieve high-performance inference.

