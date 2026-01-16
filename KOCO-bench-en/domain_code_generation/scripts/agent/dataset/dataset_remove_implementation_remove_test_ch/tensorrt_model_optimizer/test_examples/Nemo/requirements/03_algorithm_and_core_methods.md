# NeMo 核心算法与方法

# FUNCTION:Quantizer.quantize

## 功能概述
对模型进行量化并校准，支持多种量化算法

## 函数签名
def quantize(self, model: "MegatronParallel", forward_loop=None):

## 输入参数
- `model`: MegatronParallel 模型实例
- `forward_loop`: 可选的校准前向循环函数

## 详细描述
将模型从浮点数表示转换为低精度数值表示以减少计算和存储开销。根据选择的量化算法确定数值精度和量化方式。通过校准过程分析模型在实际数据上的激活分布以确定合适的缩放参数。对量化参数进行后处理以避免数值下溢或精度损失。生成量化前后的对比信息用于验证量化效果。

## 输出
量化后的模型实例

## 函数实现
code/nemo/collections/llm/modelopt/quantization/quantizer.py:line 293-336

## 测试代码
code/test_code/test_quantizer.py

---

# FUNCTION:prune_language_model

## 功能概述
对GPT/Mamba模型进行结构化剪枝

## 函数签名
def prune_language_model(model: llm.GPTModel, pruning_config: PruningConfig, data_module: pl.LightningDataModule | None = None, trainer: nl.Trainer | None = None) -> llm.GPTModel:

## 输入参数
- `model`: llm.GPTModel 模型实例
- `pruning_config`: PruningConfig 剪枝配置
- `data_module`: pl.LightningDataModule 数据模块（非drop_layers时必需）
- `trainer`: nl.Trainer 训练器（非drop_layers时必需）

## 详细描述
通过移除模型中的冗余结构来减小模型规模。支持直接删除整层或基于重要性评估进行结构化剪枝。在基于重要性的剪枝中通过在真实数据上评估各部分的贡献度来确定剪枝目标。根据配置的剪枝比例对注意力头、中间层维度等结构进行裁剪。保持模型的功能完整性使其在缩小后仍能正常工作。

## 输出
剪枝后的GPTModel实例

## 函数实现
code/nemo/collections/llm/modelopt/prune/pruner.py:line 84-125

## 测试代码
code/test_code/test_pruner.py

---

# FUNCTION:adjust_distillation_model_for_mcore

## 功能概述
调整蒸馏模型以适配Megatron-Core架构

## 函数签名
def adjust_distillation_model_for_mcore(model: DistillationModel, distill_cfg: DistillationConfig):

## 输入参数
- `model`: DistillationModel 蒸馏模型实例
- `distill_cfg`: DistillationConfig 蒸馏配置

## 详细描述
将蒸馏模型适配到分布式并行训练架构。调整模型状态的保存和加载机制以正确处理学生和教师模型在并行切分下的参数分布。修改损失计算逻辑以支持跳过学生模型的原始训练损失只保留蒸馏损失。确保教师模型不参与梯度计算和参数更新。使蒸馏训练能够在多设备并行环境中正常运行。

## 输出
调整后的蒸馏模型，已适配MCore架构

## 函数实现
code/nemo/collections/llm/modelopt/distill/utils.py:line 168-194

## 测试代码
code/test_code/test_distillation_utils.py

---

# FUNCTION:teacher_provider

## 功能概述
教师模型提供者函数，用于蒸馏时加载和初始化教师模型

## 函数签名
def teacher_provider(config: llm.GPTConfig, ckpt_path: str, tokenizer: "TokenizerSpec", trainer: nl.Trainer) -> "MCoreGPTModel":

## 输入参数
- `config`: llm.GPTConfig 教师模型配置
- `ckpt_path`: str 教师模型检查点路径
- `tokenizer`: TokenizerSpec 分词器
- `trainer`: nl.Trainer 训练器对象

## 详细描述
创建并初始化用于蒸馏训练的教师模型实例。根据配置构建教师模型结构，从保存的检查点加载已训练的模型参数。在分布式并行环境中正确处理参数的切分和加载，确保每个设备加载对应的参数分片。处理检查点格式的兼容性以支持不同保存方式的权重文件。加载完成后释放临时占用的内存资源。

## 输出
加载了权重的教师模型实例

## 函数实现
code/nemo/collections/llm/modelopt/distill/utils.py:line 137-165

## 测试代码
code/test_code/test_teacher_provider.py

---

# FUNCTION:get_tensor_shapes_adjust_fn_for_distillation

## 功能概述
返回用于蒸馏过程中调整张量形状的函数，支持流水线并行

## 函数签名
def get_tensor_shapes_adjust_fn_for_distillation(model: Union[torch.nn.Module, List[torch.nn.Module]], seq_length: int, micro_batch_size: int, decoder_seq_length: Optional[int] = None, forward_only: bool = False) -> Union[Callable, None]:

## 输入参数
- `model`: torch.nn.Module 或模型列表
- `seq_length`: int 序列长度
- `micro_batch_size`: int 微批次大小
- `decoder_seq_length`: Optional[int] 解码器序列长度
- `forward_only`: bool 是否仅前向传播

## 详细描述
在流水线并行的蒸馏训练中协调数据传输的张量维度。当模型被切分到多个流水线阶段时，需要在阶段间传递学生模型和教师模型的中间结果。根据教师模型和学生模型的配置计算各自输出的张量维度，调整跨阶段传输的数据形状以容纳两个模型的输出。确保在流水线的接收和发送操作中使用正确的张量尺寸避免维度不匹配错误。仅在需要时应用调整以避免不必要的开销。

## 输出
张量形状调整函数或None

## 函数实现
code/nemo/collections/llm/modelopt/distill/utils.py:line 230-292

## 测试代码
code/test_code/test_get_tensor_shapes_adjust_fn.py

