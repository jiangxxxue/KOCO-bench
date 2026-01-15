# TensorRT-Incubator 核心算法与方法

# FUNCTION:modelopt_quantize

## 功能概述
使用 ModelOpt 对 HuggingFace 模型进行量化并校准，支持多种量化算法

## 函数签名
def modelopt_quantize(model_hf, quant_mode):

## 输入参数
- `model_hf`: HuggingFace 模型实例（如 GPT2LMHeadModel）
- `quant_mode`: str 量化模式，支持 "int8-weight-only"、"int4-weight-only"、"float8"

## 详细描述
将 HuggingFace 模型从浮点数表示转换为低精度数值表示以减少计算和存储开销。根据量化模式选择相应的量化配置策略，支持整数权重量化和浮点量化等多种模式。对于权重量化模式，禁用输入量化器以减少精度损失。创建校准循环，使用真实数据集生成校准样本，通过前向传播分析模型在实际数据上的激活分布。执行量化操作，将模型中的线性层替换为包含校准信息的量化模块。量化过程会确定合适的缩放参数以避免数值下溢或精度损失。

## 输出
量化后的 HuggingFace 模型实例，包含量化模块和校准信息

## 函数实现
code/tripy/examples/nanogpt/quantization.py:line 26-67

## 测试代码
code/test_code/test_modelopt_quantize.py

---

# FUNCTION:load_quant_weights_from_hf

## 功能概述
从量化后的 HuggingFace 模型加载量化权重和缩放因子到 Tripy 模型

## 函数签名
def load_quant_weights_from_hf(model, model_type, dtype, quant_mode):

## 输入参数
- `model`: Tripy 模型实例
- `model_type`: str HuggingFace 模型类型（如 "gpt2"）
- `dtype`: Tripy 数据类型
- `quant_mode`: str 量化模式

## 详细描述
从 HuggingFace 预训练模型加载量化权重和缩放因子。首先对 HuggingFace 模型进行量化，获取包含量化模块的量化模型。遍历量化模型的状态字典，提取量化器的动态范围最大值。对于块量化模式，需要重塑动态范围张量以匹配量化维度。将动态范围最大值转换为缩放因子，通过除以量化数据类型可表示的最大值来计算。将缩放因子转换为目标框架的键名格式。处理权重的转置和格式转换，确保与目标框架的模块接口兼容。加载权重和缩放因子到目标模型的状态字典中。

## 输出
加载了量化权重和缩放因子的 Tripy 模型

## 函数实现
code/tripy/examples/nanogpt/weight_loader.py:line 61-119

## 测试代码
code/test_code/test_load_quant_weights_from_hf.py

---

# FUNCTION:create_forward_loop

## 功能概述
创建用于模型校准的前向循环函数，从数据集生成校准样本

## 函数签名
def create_forward_loop(model, dataset_name, tokenizer, device, batch_size, num_samples):

## 输入参数
- `model`: 模型实例
- `dataset_name`: str 数据集名称（如 "cnn_dailymail"）
- `tokenizer`: AutoTokenizer 分词器实例
- `device`: 设备（CPU 或 GPU）
- `batch_size`: int 批次大小
- `num_samples`: int 校准样本数量

## 详细描述
创建用于模型量化校准的前向循环函数。从指定的数据集加载数据样本，使用分词器对文本进行编码。根据模型的序列长度要求对输入进行填充和截断。生成指定数量的校准样本，组织成批次。前向循环函数会遍历这些批次，对模型执行前向传播，收集激活值的统计信息。这些统计信息用于确定量化缩放因子，确保量化后的模型能够保持合理的精度。对于不同的量化模式，可能需要不同数量的校准样本以适应不同的精度要求。

## 输出
前向循环函数，用于 ModelOpt 量化校准

## 函数实现
code/tripy/examples/nanogpt/quantization.py:line 55-62

## 测试代码
code/test_code/test_create_forward_loop.py


---

# FUNCTION:convert_to_scale

## 功能概述
将 ModelOpt 量化器的 amax 值转换为缩放因子

## 函数签名
def convert_to_scale(amax, maxbound):

## 输入参数
- `amax`: torch.Tensor 动态范围的最大绝对值
- `maxbound`: float 量化数据类型可表示的最大值（如 int8 为 127）

## 详细描述
将量化器中的动态范围最大值转换为缩放因子。缩放因子用于在量化和反量化过程中缩放数值，确保量化后的值能够正确表示原始浮点数的范围。通过将动态范围最大值除以量化数据类型可表示的最大值来计算缩放因子。对于不同精度的量化，使用相应的最大值范围。转换后的缩放因子会转换为浮点数类型，并去除多余的维度。这些缩放因子会被加载到目标框架的模块中，用于推理时的量化和反量化操作。

## 输出
缩放因子张量，用于量化和反量化

## 函数实现
code/tripy/examples/nanogpt/weight_loader.py:line 67-68

## 测试代码
code/test_code/test_convert_to_scale.py

---

# FUNCTION:get_submodule

## 功能概述
从模型层次结构中获取子模块，支持 ModuleList 和 ModuleDict

## 函数签名
def get_submodule(module, attr_name):

## 输入参数
- `module`: torch.nn.Module 模型实例
- `attr_name`: str 属性名称路径（如 "transformer.h.0.attn.c_attn"）

## 详细描述
从模型的层次结构中获取指定的子模块。支持通过点号分隔的属性路径访问嵌套模块。处理列表和字典类型的模块容器，能够正确解析索引和键名。遍历属性路径的每个部分，根据模块类型选择相应的访问方式：对于列表类型的模块容器，使用整数索引访问；对于字典类型的模块容器，使用键名访问；对于普通模块，使用属性访问。该函数用于从量化模型中提取量化器对象，以便访问量化参数。

## 输出
子模块实例

## 函数实现
code/tripy/examples/nanogpt/weight_loader.py:line 70-79

## 测试代码
code/test_code/test_get_submodule.py

