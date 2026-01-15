# DiffuCoder 算法核心函数描述

# FUNCTION: DiffuGRPOTrainer.forward_process

## 功能概述
生成三个版本的掩码序列，用于 Coupled Sampling 方案。通过配对掩码确保每个 token 在至少一次前向传播中被计算，降低方差并提高训练效率。

## 函数签名
def forward_process(self, batch, prompt_index, mask_id, seed=None, accumulate=False)

## 输入参数
- `batch`: 输入序列张量，形状 `[batch_size, seq_len]`
- `prompt_index`: 布尔张量，标记哪些位置是提示 token，形状 `[seq_len]`
- `mask_id`: 掩码 token 的 ID（通常是 `tokenizer.mask_token_id`）
- `seed`: 随机种子，用于控制掩码模式的可重复性
- `accumulate`: 是否在梯度累积步骤间共享随机矩阵

## 详细描述
函数为扩散模型训练生成三个掩码序列。第一个版本将所有生成部分完全掩码，第二个版本按随机概率掩码生成token，第三个版本掩码第二个版本中未被掩码的生成token。

掩码概率从区间[0.2,0.8]内随机采样。函数需要生成与输入序列同形状的随机矩阵用于决定每个 token 是否被掩码。

在梯度累积场景下，多个累积步骤共享同一随机矩阵以保持掩码模式一致。支持通过随机种子控制掩码的可重复性。

## 输出
- `noisy_batch`: 三个版本的掩码序列列表，每个形状 `[batch_size, seq_len]`
- `weights`: 权重列表 `[1, 1/mask_ratio, 1/(1-mask_ratio)]` 
- `completion_mask`: 布尔张量，标记 Version 2 中哪些 token 被掩码，形状 `[batch_size, seq_len]`

## 函数实现
code/src/open_r1/coupled_grpo.py:line 254-279

## 测试代码
code/test_code/test_forward_process.py

---

# FUNCTION: selective_log_softmax

## 功能概述
高效计算加权对数概率，避免内存溢出。对于每个序列，计算三个版本（原始、掩码、反向掩码）的对数概率，并根据掩码状态和权重进行加权平均。

## 函数签名
def selective_log_softmax(logits, index, weights=None, mask=None)

## 输入参数
- `logits`: 模型输出的 logits，形状 `[num_iterations * 3 * batch_size, seq_len, vocab_size]`
- `index`: 目标 token ID，形状 `[num_iterations * batch_size, seq_len]`
- `weights`: 权重张量，形状 `[num_iterations * 3]`，用于加权不同版本的对数概率
- `mask`: 掩码指示器，形状 `[num_iterations * batch_size, seq_len]`，标记哪些 token 被掩码

## 详细描述

函数从三个版本的logits中计算加权对数概率。输入的logits张量按迭代分块组织，每个迭代块包含三个连续批次，分别对应所有序列的三个掩码版本。函数逐个处理序列，提取三个版本的logits，对词汇表维度应用log softmax，然后用gather操作根据目标token索引提取对应位置的对数概率。对每个token位置，如果在第二个版本中被遮蔽，使用第二个版本的对数概率乘以第二个权重；如果未被遮蔽，使用第三个版本的对数概率乘以第三个权重。将加权结果与第一个版本的对数概率相加后除以2，得到最终对数概率。将所有序列的对数概率堆叠成张量后返回。

## 输出
- `per_token_logps`: token 级对数概率，形状 `[num_iterations * batch_size, seq_len]`

## 函数实现
code/src/open_r1/coupled_grpo.py:line 59-131

## 测试代码
code/test_code/test_selective_log_softmax.py

---

# FUNCTION: DiffuGRPOTrainer._get_per_token_logps

## 功能概述
计算 token 级对数概率，使用 Coupled Sampling 方案。对于每个迭代，生成三个版本的掩码序列，前向传播计算 logits，然后使用 `selective_log_softmax` 计算加权对数概率。

## 函数签名
def _get_per_token_logps(self, model, input_ids, logits_to_keep, mask_seeds)

## 输入参数
- `model`: 策略模型或参考模型
- `input_ids`: 输入序列，形状 `[num_iterations, batch_size, seq_len]`
- `logits_to_keep`: 需要计算 logits 的 token 数量（通常是完成部分的长度）
- `mask_seeds`: 掩码种子列表，长度为 `num_iterations`

## 详细描述

函数根据序列长度和completion长度确定prompt部分的位置。对每次迭代，使用对应的mask种子并调用相关函数，生成三个版本的掩码序列，同时获得用于加权的系数和标记遮蔽位置的掩码。

将掩码序列输入模型获取logits，但只保留序列末尾completion部分的输出。从原始输入和遮蔽掩码中也提取对应的completion部分，作为目标token和遮蔽指示器，结合logits和权重系数，调用相关函数计算每个token位置的加权对数概率。

计算结果被重组为三维张量，维度顺序调整为批次、迭代、completion长度，使得每个样本在不同迭代下的对数概率被组织在一起。函数在返回前释放中间计算占用的内存。

## 输出
- `per_token_logps`: token 级对数概率，形状 `[batch_size, num_iterations, logits_to_keep]`

## 函数实现
code/src/open_r1/coupled_grpo.py:line 290-362

## 测试代码
code/test_code/test_get_per_token_logps.py

---

# FUNCTION: DiffuGRPOTrainer._generate_and_score_completions

## 功能概述
生成代码完成并计算奖励。使用扩散生成方法生成 k 个响应，然后调用奖励函数计算每个响应的奖励分数，最后计算 leave-one-out 优势。

## 函数签名
def _generate_and_score_completions(self, inputs)

## 输入参数
- `inputs`: 输入数据字典，包含 `prompt` 等字段

## 详细描述
函数接收包含prompt的输入字典，对prompt应用聊天模版、编码和可选的长度截断，然后使用扩散生成方法产生completion。

扩散生成过程采用迭代去噪机制，通过指定的扩散步数、温度参数和熵算法逐步从遮蔽状态恢复出完整的代码序列。生成过程分批进行。

函数根据配置决定遮蔽种子的生成方式。在随机遮蔽模式下，为每次迭代生成不同的随机种子；在固定模式下，所有迭代使用相同种子。

生成的completion被解码为文本后，如果是对话格式，构建对话消息，然后传递给多个奖励函数进行评估。每个奖励函数返回一个分数，函数将这些分数按预设权重加权求和得到最终奖励。奖励计算完成后，函数采用leave-one-out方法计算优势估计。可选地，优势可以用组内标准差进行归一化。

函数记录多项指标，包括completion长度、各奖励函数的分数、总体奖励均值和标准差，以及零标准差样本的比例。最后返回包含prompt和completion的token ID、掩码、对数概率、优势估计和遮蔽种子的字典。

## 输出
- 字典，包含以下字段：
  - `prompt_ids`: 提示 token ID
  - `prompt_mask`: 提示掩码
  - `completion_ids`: 完成 token ID
  - `completion_mask`: 完成掩码
  - `old_per_token_logps`: 策略模型的对数概率（如果 `num_iterations > 1`）
  - `ref_per_token_logps`: 参考模型的对数概率（如果 `beta > 0`）
  - `advantages`: 优势估计
  - `mask_seeds`: 掩码种子列表

## 函数实现
code/src/open_r1/coupled_grpo.py:line 383-614

## 测试代码
code/test_code/test_generate_and_score_completions.py

---

# FUNCTION: code_reward

## 功能概述
评估生成的代码并计算测试用例通过率。提取代码块，构建评估脚本，通过代码执行提供者执行脚本，解析输出中的通过率作为奖励。

## 函数签名
def code_reward(completions, num_parallel=2, provider_type="e2b", enforce_same_language=False, **kwargs)

## 输入参数
- `completions`: 模型生成的完成列表
- `num_parallel`: 并行代码执行数量
- `provider_type`: 代码执行提供者类型（`e2b`、`local`、`morph`）
- `enforce_same_language`: 是否强制所有问题使用相同语言
- `**kwargs`: 额外参数，包含 `test_cases` 等字段

## 详细描述
函数首先调用相关函数检查每个completion的代码格式。对于格式不正确的completion直接分配零分，对于格式正确的则提取代码内容并结合测试用例构建评估脚本。评估脚本是独立的Python程序，通过子进程逐个运行测试用例，计算通过率并输出。

构建好的脚本提交给代码执行提供者并行执行，执行完成后从输出中解析测试通过率作为奖励分数。函数返回与输入长度相同的奖励列表，格式错误或执行失败的位置为零。

## 输出
- `rewards`: 奖励列表，每个元素是测试用例通过率（范围 [0, 1]）

## 函数实现
code/src/open_r1/rewards.py:line 443-525

## 测试代码
code/test_code/test_code_reward.py

---

# FUNCTION: get_code_format_reward

## 功能概述
检查代码格式是否正确，包括代码块标记和语法检查。返回一个闭包函数，用于计算格式奖励。

## 函数签名
def get_code_format_reward(language="python")

## 输入参数
- `language`: 编程语言（默认 `python`）

## 详细描述
外层函数 get_code_format_reward 接收语言参数，配置格式模式。格式模式要求字符串以 ```python\n 开始代码块，以 ```\n 结束，代码块前后可以有其他内容但不能包含额外的代码块标记内层函数。

内层函数code_format_reward检查每一个completion是否匹配格式模式，如果匹配则提取代码块内容。

对于Python代码，函数进一步使用抽象语法树解析器检查语法正确性。函数返回三级奖励：格式不正确为0.0，格式正确但语法错误为0.5，格式和语法都正确为1.0。对于其他语言，目前只检查格式，格式正确即为满分。

## 输出
- `code_format_reward`: 闭包函数，接受 `completions` 参数，返回格式奖励列表

## 函数实现
code/src/open_r1/rewards.py:line 529-580

## 测试代码
code/test_code/test_code_format_reward.py
