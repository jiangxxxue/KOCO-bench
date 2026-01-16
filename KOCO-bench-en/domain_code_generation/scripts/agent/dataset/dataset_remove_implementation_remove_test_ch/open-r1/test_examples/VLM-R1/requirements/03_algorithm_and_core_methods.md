# VLM-R1 算法核心函数描述

# FUNCTION: default_accuracy_reward

## 功能概述
默认的准确性奖励函数，支持多种验证方式：符号验证、数值匹配、多选题匹配、模糊匹配。首先尝试符号验证，失败后根据答案类型选择合适的验证方法。

## 函数签名
def default_accuracy_reward(content, sol, **kwargs):

## 输入参数
- `content`: 模型生成的完整响应文本，可能包含 `<think>` 和 `<answer>` 标签
- `sol`: 标准答案文本，可能包含 `<answer>` 标签
- `**kwargs`: 额外参数（未使用）

## 详细描述
1. **提取答案内容**：
   - 从 `sol` 中提取 `<answer>` 标签内的内容作为标准答案
   - 从 `content` 中提取最后一个 `<answer>` 标签内的内容作为学生答案
   - 如果没有标签，使用整个字符串（去除前后空白）

2. **符号验证**（第一优先级）：
   - 使用 `parse()` 函数解析学生答案和标准答案
   - 使用 `verify()` 函数进行符号验证
   - 如果验证成功（返回值>0），奖励为1.0

3. **类型判断与匹配**（符号验证失败时）：
   - **数值答案**：检查标准答案是否包含数字
     - 尝试使用 `numeric_reward()` 进行精确数值匹配
     - 如果失败，使用 Levenshtein 距离进行模糊匹配
   - **多选题**：使用 `extract_choice()` 提取选项字母
     - 提取标准答案和学生答案中的选项（A/B/C/D等）
     - 进行精确匹配，相同则奖励1.0，否则0.0
   - **文本答案**：使用 Levenshtein 距离计算相似度
     - 计算清洗后文本的相似度比率（0-1之间）

4. **异常处理**：
   - 所有验证方法失败时，保持奖励为0.0

## 输出
- `reward`: 浮点数，范围0.0-1.0

## 函数实现
code/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py:line 790-825

## 测试代码
code/test_code/test_default_accuracy_reward.py

---

# FUNCTION: iou_reward

## 功能概述
计算预测边界框与真实边界框之间的IoU（Intersection over Union），用于指代表达理解（REC）任务的准确性评估。支持边界框坐标的自动缩放调整。

## 函数签名
def iou_reward(completions, solution, **kwargs):

## 输入参数
- `completions`: 模型生成结果列表，每个元素为包含 `{"role": "assistant", "content": "..."}` 的字典列表
- `solution`: 标准答案列表，每个元素为包含边界框坐标的JSON字符串
- `**kwargs`: 必须包含以下字段：
  - `image_grid_thw`: 图像网格尺寸信息（用于坐标缩放）
  - `image_path`: 图像路径列表（用于获取原始图像尺寸）

## 详细描述
1. **IoU计算函数**：
   - 计算两个边界框的交集区域
   - 计算两个边界框的并集区域
   - IoU = 交集面积 / 并集面积

2. **边界框坐标缩放**：
   - 从 `image_grid_thw` 获取模型输入尺寸：
     - `input_height = image_grid_thw[1] × 14`
     - `input_width = image_grid_thw[2] × 14`
   - 从图像文件获取原始尺寸：`image_width, image_height`
   - 缩放公式：
     - `bbox[0] = bbox[0] / input_width × image_width`
     - `bbox[1] = bbox[1] / input_height × image_height`
     - `bbox[2] = bbox[2] / input_width × image_width`
     - `bbox[3] = bbox[3] / input_height × image_height`

3. **答案提取与验证**：
   - 使用正则表达式 `r'<answer>(.*?)</answer>'` 提取答案内容
   - 从标准答案中解析JSON格式的边界框坐标
   - 从模型输出中提取边界框坐标 `[x1, y1, x2, y2]`
   - 对预测边界框进行坐标缩放
   - 计算缩放后的预测框与真实框的IoU作为奖励

4. **调试日志**：
   - 当 `DEBUG_MODE="true"` 时，记录奖励值、图像路径、问题、生成内容和标准答案
   - 日志文件路径由 `LOG_PATH` 环境变量指定

## 输出
- `rewards`: 浮点数列表，每个元素为0.0-1.0之间的IoU值

## 函数实现
code/src/open-r1-multimodal/src/open_r1/vlm_modules/qwen_module.py:line 110-180

## 测试代码
code/test_code/test_iou_reward.py

---

# FUNCTION: detection_score

## 功能概述
计算目标检测任务的综合得分，考虑位置准确性、标签准确性和完整性（漏检/误检）。使用加权求和方式融合多个评估维度。

## 函数签名
def detection_score(content, sol, iou_threshold=0.5, alpha=0.7, beta=0.0, gamma=0.3):

## 输入参数
- `content`: 模型生成的响应文本，包含JSON格式的预测边界框列表
- `sol`: 标准答案文本，包含JSON格式的真实边界框列表
- `iou_threshold`: IoU阈值，默认0.5，用于判断预测框与真实框是否匹配
- `alpha`: 位置准确性权重，默认0.7
- `beta`: 标签准确性权重，默认0.0
- `gamma`: 完整性权重，默认0.3

## 详细描述
1. **边界框解析**：
   - 使用正则表达式 `r'```json(.*?)```'` 提取JSON代码块
   - 解析预测边界框列表：`[{"bbox_2d": [x1, y1, x2, y2], "label": "类别名"}, ...]`
   - 解析真实边界框列表（格式相同）

2. **边缘情况处理**：
   - 如果真实框为空且预测框为空，返回1.0（完美匹配）
   - 如果真实框为空但预测框不为空，返回0.0（误检）
   - 如果预测框为空但真实框不为空，返回0.0（漏检）

3. **IoU矩阵计算**：
   - 计算所有预测框与所有真实框之间的IoU
   - 构建IoU矩阵：`iou_matrix[pred_idx][gt_idx]`

4. **贪婪匹配算法**：
   - 初始化未匹配的预测框和真实框列表
   - 循环查找最大IoU值
   - 如果最大IoU ≥ `iou_threshold`：
     - 检查标签是否匹配
     - 如果标签匹配，记录IoU值和标签正确标志
     - 如果标签不匹配，记录IoU=0和标签错误标志
     - 从未匹配列表中移除该对
   - 如果最大IoU < `iou_threshold`，停止匹配

5. **得分计算**：
   - **位置准确性得分**：`position_score = sum(匹配对的IoU) / len(真实框)`
   - **标签准确性得分**：`label_score = sum(标签正确的匹配对数) / len(真实框)`
   - **完整性得分**：
     - 漏检率 = `len(未匹配真实框) / len(真实框)`
     - 误检率 = `len(未匹配预测框) / len(预测框)`
     - 完整性得分 = `1.0 - (漏检率 + 误检率) / 2`
   - **综合得分**：
     - `final_score = (alpha × position_score + beta × label_score + gamma × completeness_score) / (alpha + beta + gamma)`

## 输出
- `score`: 浮点数，范围0.0-1.0，表示目标检测的综合得分

## 函数实现
code/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py:line 450-560

## 测试代码
code/test_code/test_detection_score.py

---

# FUNCTION: cosine_reward

## 功能概述
基于余弦函数的长度奖励，鼓励模型生成简洁的回答。对正确答案，长度越短奖励越高；对错误答案，长度越短惩罚越大。

## 函数签名
def cosine_reward(content, tokenizer, acc_reward, **kwargs):

## 输入参数
- `content`: 模型生成的响应文本
- `tokenizer`: 分词器，用于计算文本长度
- `acc_reward`: 准确性奖励值，用于判断答案是否正确
- `**kwargs`: 额外参数（未使用）

## 详细描述
1. **配置参数**：
   - `min_len_value_wrong = 0.0`：错误答案的最小长度奖励
   - `max_len_value_wrong = -0.5`：错误答案的最大长度惩罚
   - `min_len_value_correct = 1.0`：正确答案的最小长度奖励
   - `max_len_value_correct = 0.5`：正确答案的最大长度奖励
   - `cosine_max_len = 1024`：余弦函数的周期长度

2. **长度计算**：
   - 使用分词器编码文本：`gen_len = len(tokenizer.encode(content))`

3. **正确性判断**：
   - 如果 `acc_reward ≥ 0.7`，认为答案正确
   - 正确答案：使用 `min_value = max_len_value_correct`, `max_value = min_len_value_correct`
   - 错误答案：使用 `min_value = min_len_value_wrong`, `max_value = max_len_value_wrong`

4. **余弦奖励计算**：
   - 公式：`reward = max_value - (max_value - min_value) × (1 - cos(gen_len × π / cosine_max_len)) / 2`
   - 对正确答案：
     - 长度=0时，reward=1.0（最高奖励）
     - 长度=1024时，reward=0.5（最低奖励）
   - 对错误答案：
     - 长度=0时，reward=0.0（无惩罚）
     - 长度=1024时，reward=-0.5（最大惩罚）

## 输出
- `reward`: 浮点数，正确答案范围0.5-1.0，错误答案范围-0.5-0.0

## 函数实现
code/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py:line 562-580

## 测试代码
code/test_code/test_cosine_reward.py

---

# FUNCTION: repetition_reward

## 功能概述
检测生成内容中的重复模式，对重复内容进行惩罚。支持JSON格式数据（检测重复边界框）和纯文本数据（检测重复短语）。

## 函数签名
def repetition_reward(content, **kwargs):

## 输入参数
- `content`: 模型生成的响应文本
- `**kwargs`: 额外参数（未使用）

## 详细描述
1. **配置参数**：
   - `max_penalty = -1.0`：最大惩罚值

2. **JSON数据处理**：
   - 尝试提取 ` ```json...``` ` 代码块
   - 如果提取失败，尝试提取 ` ```...``` ` 代码块
   - 如果仍失败，尝试匹配包含 `bbox_2d` 和 `label` 的JSON数组
   - 解析JSON数据，如果失败则使用 `json_repair` 修复
   - 对JSON数据，设置 `ngram_size = 1`
   - 将每个对象的 `bbox_2d` 和 `label` 组合成字符串：`f"{bbox_2d}_{label}"`
   - 使用n-gram统计检测重复项

3. **纯文本处理**：
   - 如果无法解析为JSON，按纯文本处理
   - 设置 `ngram_size = 6`
   - 将文本转换为小写并分词
   - 使用6-gram统计检测重复短语

4. **n-gram统计**：
   - 定义 `zipngram()` 函数：生成所有n-gram
   - 统计唯一n-gram数量和总n-gram数量
   - 计算重复率：`scaling = 1 - unique_ngrams / total_ngrams`
   - 计算惩罚：`reward = scaling × max_penalty`
   - 重复率越高，惩罚越大（接近-1.0）

5. **边缘情况**：
   - 如果内容为空，返回0.0
   - 如果文本长度小于n-gram大小，返回0.0
   - 如果总n-gram数为0，返回0.0

## 输出
- `reward`: 浮点数，范围-1.0-0.0，重复率越高惩罚越大

## 函数实现
code/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py:line 582-650

## 测试代码
code/test_code/test_repetition_reward.py

---

# FUNCTION: format_reward

## 功能概述
检查生成内容是否符合预定义的格式要求，即是否包含完整的 `<think>...</think><answer>...</answer>` 结构。

## 函数签名
def format_reward(completions, **kwargs):

## 输入参数
- `completions`: 模型生成结果列表，每个元素为包含 `{"role": "assistant", "content": "..."}` 的字典列表
- `**kwargs`: 额外参数（未使用）

## 详细描述
1. **定义格式模式**：
   - 正则表达式：`r"<think>.*?</think>\s*<answer>.*?</answer>"`
   - 要求两个标签按顺序出现，标签之间可以有空白字符
   - 使用 `re.DOTALL` 标志支持跨行匹配

2. **格式匹配**：
   - 提取每个completion的content字段
   - 使用 `re.fullmatch()` 进行完整匹配
   - 匹配成功返回1.0，失败返回0.0

3. **调试日志**：
   - 当 `DEBUG_MODE="true"` 时，记录格式验证信息
   - 日志文件名：`{LOG_PATH}_format.txt`
   - 记录每个生成内容和格式匹配结果

## 输出
- `rewards`: 浮点数列表，每个元素为0.0或1.0

## 函数实现
code/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py:line 880-895

## 测试代码
code/test_code/test_format_reward.py

---

# FUNCTION: format_reward_rec

## 功能概述
REC任务专用的格式奖励函数，检查输出是否包含 `<think>` 标签、`<answer>` 标签以及JSON格式的边界框坐标。

## 函数签名
def format_reward_rec(completions, **kwargs):

## 输入参数
- `completions`: 模型生成结果列表，每个元素为包含 `{"role": "assistant", "content": "..."}` 的字典列表
- `**kwargs`: 额外参数（未使用）

## 详细描述
1. **定义格式模式**：
   - 正则表达式：`r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"`
   - 要求包含：
     - `<think>` 标签和内容
     - `<answer>` 标签
     - 花括号 `{}`（JSON对象）
     - 方括号 `[]` 内包含4个数字（边界框坐标）
   - 使用 `re.DOTALL` 标志支持跨行匹配

2. **格式匹配**：
   - 提取每个completion的content字段
   - 使用 `re.search()` 检查是否包含该模式
   - 匹配成功返回1.0，失败返回0.0

3. **调试日志**：
   - 当 `DEBUG_MODE="true"` 时，记录格式验证信息
   - 日志文件名：`{LOG_PATH}_format.txt`

## 输出
- `rewards`: 浮点数列表，每个元素为0.0或1.0

## 函数实现
code/src/open-r1-multimodal/src/open_r1/vlm_modules/qwen_module.py:line 95-108

## 测试代码
code/test_code/test_format_reward_rec.py
