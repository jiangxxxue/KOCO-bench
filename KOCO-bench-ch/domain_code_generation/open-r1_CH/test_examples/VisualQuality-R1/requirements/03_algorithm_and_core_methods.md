# VisualQuality-R1 算法核心函数描述

# FUNCTION: fidelity_reward

## 功能概述
计算单个样本对之间的排序保真度奖励，衡量模型预测的排序关系与真实质量标签的一致性。

## 函数签名
def fidelity_reward(pred1, pred2, var1, var2, gt, device):

## 输入参数
- `pred1`: 第一个样本的预测评分（标量或张量）
- `pred2`: 第二个样本的预测评分均值（标量或张量）
- `var1`: 第一个样本的预测方差（标量或张量）
- `var2`: 第二个样本的预测方差（标量或张量）
- `gt`: 真实排序关系（0.0 表示 pred2 优于 pred1，1.0 表示 pred1 优于 pred2，0.5 表示相等）
- `device`: 计算设备

## 详细描述
1. 计算标准化差异：Δ = (pred1 - pred2) / √(var1 + var2 + ε)，其中 ε = 1e-6 防止除零
2. 通过标准正态分布的累积分布函数计算排序概率：p = Φ(Δ)
3. 计算保真度奖励：reward = √(p × gt + ε) + √((1-p) × (1-gt) + ε)
4. 该奖励函数在预测排序与真实排序一致时达到最大值 2.0，完全不一致时接近 √(2ε)

## 输出
- `reward`: 保真度奖励值（标量或张量）

## 函数实现
code/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py:line 128-139

## 测试代码
code/test_code/test_fidelity_reward.py

---

# FUNCTION: accuracy_reward

## 功能概述
批量计算所有样本的保真度奖励，通过成对比较学习图像质量的相对排序关系。

## 函数签名
def accuracy_reward(completions, solution, **kwargs):

## 输入参数
- `completions`: 生成的响应列表，每个响应包含 `[{"role": "assistant", "content": "..."}]` 格式
- `solution`: 真实 MOS 评分列表（包含 `<answer>` 标签）
- `kwargs`: 额外参数
  - `device`: 计算设备
  - `num_generations`: 每个样本的生成数量 G
  - `image_path`: 图像路径（用于调试日志）
  - `problem`: 问题文本（用于调试日志）

## 详细描述
1. 提取真实评分：从 `solution` 中解析 `<answer>` 标签内的 MOS 值，重塑为 `[batch_size, num_generations]`
2. 提取预测评分：从 `completions` 中解析 `<answer>` 标签内的评分，使用正则表达式提取第一个数字
3. 计算统计量：对每个样本的 G 个预测计算均值 μ 和方差 σ²
4. 成对比较：对批次内每对样本 (i, k)：
   - 确定真实排序关系：
     - 若 MOS_i > MOS_k，则 gt = 1.0
     - 若 MOS_i < MOS_k，则 gt = 0.0
     - 若 MOS_i = MOS_k，则 gt = 0.5
   - 对样本 i 的每个生成 j，调用 `fidelity_reward(pred_ij, μ_k, σ²_i, σ²_k, gt, device)`
   - 累加所有其他样本的保真度奖励并求平均
5. 异常处理：若解析失败，使用随机评分 `random.uniform(1, 5)`
6. 调试日志：若 `DEBUG_MODE=true`，记录奖励值、图像路径、问题和响应内容

## 输出
- `rewards`: 保真度奖励列表，长度为 `batch_size × num_generations`

## 函数实现
code/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py:line 142-225

## 测试代码
code/test_code/test_accuracy_reward.py

---

# FUNCTION: format_reward

## 功能概述
检查生成的响应是否符合指定的格式要求（包含 `<think>` 和 `<answer>` 标签）。

## 函数签名
def format_reward(completions, **kwargs):

## 输入参数
- `completions`: 生成的响应列表，每个响应包含 `[{"role": "assistant", "content": "..."}]` 格式
- `kwargs`: 额外参数（用于调试日志）

## 详细描述
1. 定义格式模式：`pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"`
2. 提取响应内容：从 `completions` 中提取 `content` 字段
3. 格式匹配：使用正则表达式 `re.fullmatch` 检查每个响应是否完全匹配格式
4. 计算奖励：匹配成功返回 1.0，否则返回 0.0
5. 调试日志：若 `DEBUG_MODE=true`，记录每个响应的内容和格式检查结果

## 输出
- `rewards`: 格式奖励列表，每个元素为 1.0 或 0.0

## 函数实现
code/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py:line 228-243

## 测试代码
code/test_code/test_format_reward.py

---

# FUNCTION: VLMGRPOTrainer._generate_and_score_completions

## 功能概述
生成多个响应并计算奖励，为 GRPO 训练提供必要的数据。

## 函数签名
def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]], model) -> dict[str, Union[torch.Tensor, Any]]:

## 输入参数
- `inputs`: 输入数据字典，包含 `prompt`、`image` 或 `image_path` 等字段
- `model`: 策略模型

## 详细描述
1. 准备输入：
   - 提取提示文本和图像（从路径加载或直接使用）
   - 确保图像最小尺寸为 28×28 像素（必要时调整大小）
   - 使用 VLM Module 将图像和文本转换为模型输入张量
2. 生成响应：
   - 使用 `model.generate` 生成 `max_completion_length` 个 token
   - 采样参数：`do_sample=True`, `temperature=1.0`
   - 分离提示和生成的完成部分
3. 掩码处理：
   - 检测 EOS token 位置
   - 创建 `completion_mask`，将 EOS 后的 token 掩码为 0
4. 计算 log 概率：
   - 当 `num_iterations > 1` 时，计算 `old_per_token_logps`（用于 PPO 比率）
   - 若 `beta > 0`，计算参考模型的 `ref_per_token_logps`（用于 KL 惩罚）
5. 解码响应：使用 tokenizer 将 token ID 解码为文本
6. 计算奖励：
   - 遍历所有奖励函数（保真度奖励、格式奖励等）
   - 若奖励函数是预训练模型，使用模型推理计算奖励
   - 若奖励函数是自定义函数，直接调用并传递必要参数
   - 跨进程收集奖励并求和
7. 计算优势：
   - 按组（相同提示的 G 个响应）计算奖励的均值和标准差
   - 归一化：advantages = (rewards - mean) / (std + 1e-4)
   - 仅保留当前进程的优势值
8. 记录指标：完成长度、各奖励函数的平均值、总奖励、奖励标准差

## 输出
- 字典，包含：
  - `prompt_ids`: 提示的 token ID
  - `prompt_mask`: 提示的注意力掩码
  - `completion_ids`: 生成的完成部分的 token ID
  - `completion_mask`: 完成部分的注意力掩码
  - `old_per_token_logps`: 旧策略的 token 级 log 概率
  - `ref_per_token_logps`: 参考模型的 token 级 log 概率
  - `advantages`: 归一化的优势值
  - `multimodal_inputs`: 多模态输入（如 `pixel_values`）

## 函数实现
code/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py:line 631-826

## 测试代码
code/test_code/test_generate_and_score_completions.py

---

# FUNCTION: VLMGRPOTrainer.compute_loss

## 功能概述
计算 GRPO 训练的损失函数，包括 PPO 裁剪损失和 KL 散度惩罚。

## 函数签名
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

## 输入参数
- `model`: 策略模型
- `inputs`: 输入数据（可能是原始数据或缓存的生成结果）
- `return_outputs`: 是否返回输出（GRPO 不支持，必须为 False）
- `num_items_in_batch`: 批次中的样本数量（未使用）

## 详细描述
1. 检查是否需要生成新响应：
   - 若 `global_step % num_iterations == 0`，调用 `_generate_and_score_completions` 生成新响应
   - 否则，使用缓存的响应（支持多次迭代更新）
2. 准备输入：
   - 拼接提示和完成部分：`input_ids = [prompt_ids, completion_ids]`
   - 拼接注意力掩码：`attention_mask = [prompt_mask, completion_mask]`
3. 计算当前策略的 log 概率：
   - 调用 `_get_per_token_logps` 获取 token 级 log 概率
   - 去除提示部分，仅保留完成部分的 log 概率
4. 计算策略比率：
   - `ratio = exp(per_token_logps - old_per_token_logps)`
   - 裁剪比率：`clipped_ratio = clip(ratio, 1-ε_low, 1+ε_high)`
5. 计算 PPO 损失：
   - `loss1 = ratio × advantages`
   - `loss2 = clipped_ratio × advantages`
   - `per_token_loss = -min(loss1, loss2)`
6. 添加 KL 惩罚（若 `beta > 0`）：
   - `per_token_kl = exp(ref_logps - cur_logps) - (ref_logps - cur_logps) - 1`
   - `per_token_loss += beta × per_token_kl`
   - 记录平均 KL 散度
7. 计算最终损失：
   - 对每个样本，计算完成部分的平均损失（考虑掩码）
   - 对批次求平均
8. 记录裁剪比率：计算被裁剪的 token 比例

## 输出
- `loss`: 标量损失值

## 函数实现
code/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py:line 828-886

## 测试代码
code/test_code/test_compute_loss.py

---

# FUNCTION: RepeatRandomSampler.__iter__

## 功能概述
实现自定义采样器，确保每个 GPU 的批次可以来自不同数据集，并支持按图像前缀分组采样。

## 函数签名
def __iter__(self):

## 输入参数
- 无（使用类的成员变量）

## 详细描述
1. 初始化：
   - 复制数据集索引（按数据集名称和前缀分组）
   - 创建数据集名称列表（用于随机选择）
2. 填充 GPU 批次：
   - 对每个 GPU，随机选择一个数据集
   - 若数据集支持前缀分组（如 KADID-10K）：
     - 随机选择一个前缀
     - 从该前缀的索引中随机抽取 `per_gpu_batch_size` 个样本
     - 更新剩余索引，若前缀耗尽则移除
   - 若数据集不支持前缀分组：
     - 直接从数据集索引中随机抽取 `per_gpu_batch_size` 个样本
     - 更新剩余索引，若数据集耗尽则移除
3. 重复采样：
   - 对每个批次，重复 `repeat_count` 次（对应 `num_iterations`）
   - 对每个索引，重复 `mini_repeat_count` 次（对应 `num_generations`）
4. 终止条件：当所有数据集的索引都耗尽时停止

## 输出
- 生成器，逐个产生样本索引

## 函数实现
code/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py:line 130-232

## 测试代码
code/test_code/test_repeat_random_sampler.py
