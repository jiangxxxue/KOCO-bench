# TensorRT-Incubator Core Algorithms and Methods

# FUNCTION: modelopt_quantize

## Function Overview
Quantize and calibrate HuggingFace models using ModelOpt, supporting multiple quantization algorithms

## Function Signature
def modelopt_quantize(model_hf, quant_mode):

## Input Parameters
- `model_hf`: HuggingFace model instance (e.g., GPT2LMHeadModel)
- `quant_mode`: str quantization mode, supports "int8-weight-only", "int4-weight-only", "float8"

## Detailed Description
Converts HuggingFace models from floating-point representation to low-precision numerical representation to reduce computational and storage overhead. Selects appropriate quantization configuration strategy based on quantization mode, supporting multiple modes including integer weight quantization and floating-point quantization. For weight quantization modes, disables input quantizers to reduce precision loss. Creates calibration loop, generates calibration samples using real datasets, and analyzes model activation distribution on actual data through forward propagation. Executes quantization operation, replacing linear layers in the model with quantized modules containing calibration information. The quantization process determines appropriate scaling parameters to avoid numerical underflow or precision loss.

## Output
Quantized HuggingFace model instance containing quantized modules and calibration information

## Function Implementation
code/tripy/examples/nanogpt/quantization.py:line 26-67

## Test Code
code/test_code/test_modelopt_quantize.py

---

# FUNCTION: load_quant_weights_from_hf

## Function Overview
Load quantized weights and scaling factors from quantized HuggingFace model to Tripy model

## Function Signature
def load_quant_weights_from_hf(model, model_type, dtype, quant_mode):

## Input Parameters
- `model`: Tripy model instance
- `model_type`: str HuggingFace model type (e.g., "gpt2")
- `dtype`: Tripy data type
- `quant_mode`: str quantization mode

## Detailed Description
Loads quantized weights and scaling factors from HuggingFace pretrained models. First quantizes the HuggingFace model to obtain a quantized model containing quantized modules. Iterates through the quantized model's state dictionary to extract the quantizer's dynamic range maximum values. For block quantization modes, needs to reshape dynamic range tensors to match quantization dimensions. Converts dynamic range maximum values to scaling factors by dividing by the maximum value representable by the quantized data type. Converts scaling factors to target framework's key name format. Handles weight transposition and format conversion to ensure compatibility with target framework's module interface. Loads weights and scaling factors into target model's state dictionary.

## Output
Tripy model loaded with quantized weights and scaling factors

## Function Implementation
code/tripy/examples/nanogpt/weight_loader.py:line 61-119

## Test Code
code/test_code/test_load_quant_weights_from_hf.py

---

# FUNCTION: create_forward_loop

## Function Overview
Create forward loop function for model calibration, generating calibration samples from dataset

## Function Signature
def create_forward_loop(model, dataset_name, tokenizer, device, batch_size, num_samples):

## Input Parameters
- `model`: Model instance
- `dataset_name`: str dataset name (e.g., "cnn_dailymail")
- `tokenizer`: AutoTokenizer tokenizer instance
- `device`: Device (CPU or GPU)
- `batch_size`: int batch size
- `num_samples`: int number of calibration samples

## Detailed Description
Creates forward loop function for model quantization calibration. Loads data samples from specified dataset and encodes text using tokenizer. Pads and truncates inputs according to model's sequence length requirements. Generates specified number of calibration samples organized into batches. Forward loop function iterates through these batches, performs forward propagation on the model, and collects activation value statistics. These statistics are used to determine quantization scaling factors, ensuring quantized model maintains reasonable accuracy. Different quantization modes may require different numbers of calibration samples to accommodate different precision requirements.

## Output
Forward loop function for ModelOpt quantization calibration

## Function Implementation
code/tripy/examples/nanogpt/quantization.py:line 55-62

## Test Code
code/test_code/test_create_forward_loop.py


---

# FUNCTION: convert_to_scale

## Function Overview
Convert ModelOpt quantizer's amax values to scaling factors

## Function Signature
def convert_to_scale(amax, maxbound):

## Input Parameters
- `amax`: torch.Tensor maximum absolute value of dynamic range
- `maxbound`: float maximum value representable by quantized data type (e.g., 127 for int8)

## Detailed Description
Converts dynamic range maximum values in quantizer to scaling factors. Scaling factors are used to scale numerical values during quantization and dequantization processes, ensuring quantized values can correctly represent the range of original floating-point numbers. Calculates scaling factor by dividing dynamic range maximum value by the maximum value representable by the quantized data type. Uses corresponding maximum value ranges for different precision quantizations. Converted scaling factors are converted to floating-point type with extra dimensions removed. These scaling factors are loaded into target framework's modules for quantization and dequantization operations during inference.

## Output
Scaling factor tensor for quantization and dequantization

## Function Implementation
code/tripy/examples/nanogpt/weight_loader.py:line 67-68

## Test Code
code/test_code/test_convert_to_scale.py

---

# FUNCTION: get_submodule

## Function Overview
Get submodule from model hierarchy, supporting ModuleList and ModuleDict

## Function Signature
def get_submodule(module, attr_name):

## Input Parameters
- `module`: torch.nn.Module model instance
- `attr_name`: str attribute name path (e.g., "transformer.h.0.attn.c_attn")

## Detailed Description
Gets specified submodule from model's hierarchy. Supports accessing nested modules through dot-separated attribute paths. Handles list and dictionary type module containers, correctly parsing indices and key names. Iterates through each part of attribute path, selecting appropriate access method based on module type: for list-type module containers, uses integer index access; for dictionary-type module containers, uses key name access; for regular modules, uses attribute access. This function is used to extract quantizer objects from quantized models to access quantization parameters.

## Output
Submodule instance

## Function Implementation
code/tripy/examples/nanogpt/weight_loader.py:line 70-79

## Test Code
code/test_code/test_get_submodule.py

