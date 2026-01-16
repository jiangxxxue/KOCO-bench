# CANOE 算法核心函数描述

# FUNCTION: accuracy_reward

## 功能概述
验证模型生成的答案是否正确，支持符号验证和字符串精确匹配两种方式。首先尝试符号验证（使用latex2sympy和math_verify），如果失败则回退到字符串匹配。

## 函数签名
def accuracy_reward(completions, solution, **kwargs):


## 输入参数
- `completions`: 模型生成结果列表，每个元素为包含`{"role": "assistant", "content": "..."}` 的字典列表
- `solution`: 正确答案列表，每个元素为字符串，可能包含`<answer>`标签
- `**kwargs`: 额外参数（未使用）

## 详细描述

该奖励函数采用了两层验证策略来判断生成的数学答案是否正确。首先尝试符号验证，将模型生成的答案和标准答案都解析为符号表达式，然后通过符号计算验证两者的数学等价性，若验证成功则直接奖励 1.0。如果符号验证失败，则从标准答案和生成答案中提取被 <answer> 标签包裹的内容（去除前后空白字符）进行字符串精确比较，如果没有找到标签则使用完整的字符串。如果完全相同则奖励1.0。在 DEBUG_MODE 环境变量为 "true" 时，将每次的验证信息写入日志文件，记录时间戳、奖励值、生成内容和正确答案。日志文件名为`accuracy_reward.log`。

## 输出
- `rewards`: 浮点数列表，每个元素为0.0或1.0

## 函数实现
code/train/src/open_r1/grpo.py:line 59-122

## 测试代码
code/test_code/test_accuracy_reward.py

---

# FUNCTION: influence_reward

## 功能概述
评估长答案（long_answer）对最终答案的影响力。通过将长答案替换原始context后重新生成，检查新生成的答案是否仍然正确。

## 函数签名
def influence_reward(completions, solution, completions_long_answer, **kwargs):


## 输入参数
- `completions`: 模型生成结果列表，每个元素为包含`{"role": "assistant", "content": "..."}` 的字典列表
- `solution`: 正确答案列表
- `completions_long_answer`: 基于长答案重新生成的结果列表，每个元素可能为None
- `**kwargs`: 额外参数（未使用）

## 详细描述

该函数接收额外的长答案生成结果作为输入，如果长答案为None则奖励为0.0并跳过后续验证，否则从中提取content字段。接着从长答案和标准答案的 <answer> 标签中提取答案部分（去除前后空白字符）进行字符串精确比较，如果完全相同则奖励1.0。在 DEBUG_MODE 环境变量为 "true" 时，将每次的验证信息写入日志文件，记录时间戳、奖励值、生成内容和正确答案。日志文件名为`influence_accuracy_reward.log`。


## 输出
- `rewards`: 浮点数列表，每个元素为0.0或1.0

## 函数实现
code/train/src/open_r1/grpo.py:line 124-168

## 测试代码
code/test_code/test_influence_reward.py

---

# FUNCTION: format_reward

## 功能概述
检查生成内容是否符合预定义的格式要求，即是否包含完整的`<think>...</think><long_answer>...</long_answer><answer>...</answer>`结构。

## 函数签名
def format_reward(completions, **kwargs):


## 输入参数
- `completions`: 模型生成结果列表，每个元素为包含`{"role": "assistant", "content": "..."}` 的字典列表
- `**kwargs`: 额外参数（未使用）

## 详细描述

检查生成内容是否符合预定义的格式要求，即是否包含完整的`<think>...</think><long_answer>...</long_answer><answer>...</answer>`结构，匹配成功返回1.0，失败返回0.0。

## 输出
- `rewards`: 浮点数列表，每个元素为0.0或1.0

## 函数实现
code/train/src/open_r1/grpo.py:line 171-176

## 测试代码
code/test_code/test_format_reward.py

---

# FUNCTION: len_reward

## 功能概述
约束长答案（long_answer）的长度在合理范围内，确保其长度在原始context长度的20%-80%之间。

## 函数签名
def len_reward(completions, **kwargs):


## 输入参数
- `completions`: 模型生成结果列表，每个元素为包含`{"role": "assistant", "content": "..."}` 的字典列表
- `**kwargs`: 必须包含`problem`字段，即原始输入（包含context）

## 详细描述
检查长答案<long_answer>的长度是否在对应<context>长度的20%-80%之间。如果是则奖励1.0，如果长度不符合该区间或无法提取<context>或<long_answer>，奖励为0.0。

## 输出
- `rewards`: 浮点数列表，每个元素为0.0或1.0

## 函数实现
code/train/src/open_r1/grpo.py:line 178-215

## 测试代码
code/test_code/test_len_reward.py

