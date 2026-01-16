# ByteMLPerf 核心算法与方法

# FUNCTION:LogitsAndIntermediatesLossBalancer.forward

## 功能概述
动态平衡蒸馏损失和原始损失的损失平衡器

## 函数签名
def forward(self, loss_dict: Dict[str, Tensor]) -> Tensor:

## 输入参数
- `loss_dict`: 所有标量损失的字典，包含student_loss、LogitsKLLoss等键值对

## 详细描述
在知识蒸馏训练中协调多个损失项的相对权重。分离学生模型的原始学习损失与来自教师指导的蒸馏损失，根据两类损失的数值关系动态调整它们的比例以实现平衡训练。当存在中间层蒸馏损失时按比例缩放使其与输出层蒸馏损失保持协调。根据训练策略决定是否保留原始学习损失。

## 输出
聚合后的总标量损失张量

## 函数实现
code/byte_train_perf/Megatron-LM/megatron/inference/algos/distillation.py:line 333-363

## 测试代码
code/test_code/test_loss_balancer.py

---

# FUNCTION:adjust_distillation_model_for_mcore

## 功能概述
调整蒸馏模型以适配Megatron-Core架构

## 函数签名
def adjust_distillation_model_for_mcore(model: mtd.DistillationModel, distill_cfg: Dict[str, Any]):

## 输入参数
- `model`: 蒸馏模型 (mtd.DistillationModel类型)
- `distill_cfg`: 蒸馏配置字典，包含层名称、损失函数等配置

## 详细描述
将蒸馏模型适配到分布式并行训练架构。隐藏教师模型使其不参与梯度计算和参数更新。调整模型层的命名映射以匹配并行切分后的层结构。确保蒸馏损失计算能够正确处理跨设备的并行数据流和梯度传播。

## 输出
调整后的蒸馏模型，已适配MCore架构

## 函数实现
code/byte_train_perf/Megatron-LM/megatron/inference/algos/distillation.py:line 435-455

## 测试代码
code/test_code/test_mcore_distillation_adjust.py

---

# FUNCTION:_teacher_provider

## 功能概述
教师模型工厂函数，用于蒸馏时创建和加载教师模型

## 函数签名
def _teacher_provider(config: Namespace, model_kwargs: Dict[str, Any]) -> MCoreGPTModel:

## 输入参数
- `config`: Namespace 教师模型配置
- `model_kwargs`: Dict[str, Any] 模型构建参数字典

## 详细描述
创建并初始化用于蒸馏训练的教师模型实例。根据配置信息构建模型结构，从已训练的检查点加载教师模型的参数权重。处理不同模型格式之间的参数名称差异以确保正确加载。支持非同构层结构以适应不同的模型架构配置。

## 输出
加载了检查点的教师模型实例

## 函数实现
code/byte_train_perf/Megatron-LM/megatron/inference/gpt/model_provider.py:line 98-119

## 测试代码
code/test_code/test_teacher_provider.py

---

# FUNCTION:model_provider

## 功能概述
构建GPT模型的主函数，支持ModelOpt蒸馏模式

## 函数签名
def model_provider(pre_process=True, post_process=True, parallel_output=True) -> MCoreGPTModel:

## 输入参数
- `pre_process`: bool 是否计算嵌入
- `post_process`: bool 是否计算输出logits/loss
- `parallel_output`: bool 是否对输出logits进行allgather

## 详细描述
构建语言模型实例并根据训练模式进行配置。创建模型结构时支持非同构层以适应优化需求。当存在已保存的模型状态时恢复模型参数。处理不同格式模型参数的兼容性转换。在蒸馏训练模式下集成教师模型，配置蒸馏策略包括损失计算方式和损失平衡机制。在蒸馏完成后支持导出纯净的学生模型，或在继续训练时调整蒸馏模型以适配分布式并行架构。

## 输出
MCoreGPTModel实例（可能被包装为DistillationModel）

## 函数实现
code/byte_train_perf/Megatron-LM/megatron/inference/gpt/model_provider.py:line 122-222

## 测试代码
code/test_code/test_model_provider.py

---

# FUNCTION:loss_func

## 功能概述
损失函数，支持知识蒸馏损失计算

## 函数签名
def loss_func(loss_mask: torch.Tensor, model: GPTModel, output_tensor: torch.Tensor):

## 输入参数
- `loss_mask`: torch.Tensor 损失掩码张量
- `model`: GPTModel 模型对象
- `output_tensor`: torch.Tensor 模型输出张量

## 详细描述
计算训练过程中的损失值并汇总统计信息。解析模型输出获取预测结果，应用掩码过滤无效位置的损失贡献。在标准训练模式下计算语言模型的预测损失。在蒸馏训练模式下额外计算教师模型指导的蒸馏损失，同时保留原始预测损失用于对比分析。跨分布式设备聚合损失值以获得全局统计用于监控训练进展。

## 输出
损失值张量和报告字典

## 函数实现
code/byte_train_perf/Megatron-LM/megatron/inference/gpt/loss_func.py:line 60-90

## 测试代码
code/test_code/test_loss_func.py

