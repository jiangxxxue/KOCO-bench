# VisualQuality-R1 测试代码

本目录包含针对 VisualQuality-R1 项目核心算法函数的测试代码。这些测试用于验证 LLM 生成的代码是否符合要求。

## 测试文件说明

### 1. test_fidelity_reward.py
测试 `fidelity_reward` 函数 - 计算单个样本对之间的排序保真度奖励

**测试内容：**
- 基本保真度奖励计算
- 预测排序正确/错误的情况
- 质量相等的情况 (gt = 0.5)
- 方差对奖励的影响
- 张量输入处理
- 边缘情况（极小方差、零方差）
- 不同真实排序关系 (gt = 0.0, 0.5, 1.0)
- 对称性验证

**关键测试点：**
- 奖励值范围应在 [0, 2.0] 之间
- 预测正确时奖励接近 2.0
- 预测错误时奖励较低
- 所有输出应该是有限的（finite）

### 2. test_accuracy_reward.py
测试 `accuracy_reward` 函数 - 批量计算所有样本的保真度奖励

**测试内容：**
- 基本准确度奖励计算
- 预测正确的情况
- 不同生成结果的方差影响
- 不同格式的答案解析
- 成对比较逻辑
- 错误处理（无法解析的答案）
- 单个样本的情况

**关键测试点：**
- 能够从 `<answer>` 标签中提取评分
- 能够处理不同格式的答案文本
- 成对比较所有样本
- 计算每个样本的均值和方差
- 使用 fidelity_reward 计算排序一致性

### 3. test_format_reward.py
测试 `format_reward` 函数 - 检查生成的响应是否符合指定的格式要求

**测试内容：**
- 正确格式的响应（包含 `<think>` 和 `<answer>` 标签）
- 缺少 think 标签
- 缺少 answer 标签
- 标签顺序错误
- 标签前后有额外内容
- 多行内容处理
- 空标签
- 嵌套标签
- 特殊字符和中文处理
- 批量处理
- 不同空白字符的处理

**关键测试点：**
- 正确格式返回 1.0
- 错误格式返回 0.0
- 使用正则表达式 `fullmatch` 进行严格匹配
- 允许 `<think>` 和 `<answer>` 之间有空白字符

### 4. test_generate_and_score_completions.py
测试 `VLMGRPOTrainer._generate_and_score_completions` 函数 - 生成多个响应并计算奖励

**测试内容：**
- 输入数据准备逻辑
- 图像加载和调整大小（最小 28×28 像素）
- 完成生成的模拟
- EOS token 掩码逻辑
- 奖励计算的模拟
- 优势计算
- 多模态输入提取
- 输出数据结构

**关键测试点：**
- 图像尺寸小于 28×28 时需要调整
- EOS token 后的内容应该被掩码
- 奖励来自多个奖励函数的总和
- 优势通过分组归一化计算
- 输出包含 prompt_ids, completion_ids, advantages 等

### 5. test_compute_loss.py
测试 `VLMGRPOTrainer.compute_loss` 函数 - 计算 GRPO 训练的损失函数

**测试内容：**
- 策略比率计算
- 比率裁剪逻辑
- PPO 损失计算
- KL 散度惩罚计算
- 带掩码的损失计算
- 最终损失聚合
- 裁剪比率指标
- 优势对损失的影响

**关键测试点：**
- 策略比率 = exp(log_prob - old_log_prob)
- 比率裁剪到 [1-ε_low, 1+ε_high]
- PPO 损失 = -min(ratio × advantage, clipped_ratio × advantage)
- KL 惩罚 = β × KL(ref || current)
- 损失考虑 completion_mask

### 6. test_repeat_random_sampler.py
测试 `RepeatRandomSampler.__iter__` 函数 - 自定义采样器实现

**测试内容：**
- 基本采样功能
- mini_repeat_count 参数
- repeat_count 参数
- 多 GPU 采样
- 基于前缀的采样（KADID-10K 数据集）
- 混合数据集采样
- 采样器长度计算
- 使用种子的确定性采样

**关键测试点：**
- 每个 GPU 的批次可以来自不同数据集
- 支持按图像前缀分组采样
- 每个索引重复 mini_repeat_count 次
- 每个批次重复 repeat_count 次
- 使用相同种子产生相同的采样序列

## 运行测试

### 运行所有测试（带详细统计）
推荐使用此方式，会生成详细的统计报告：
```bash
cd open-r1/test_examples/VisualQuality-R1/code/test_code
python3 run_tests_with_stats.py
```

此脚本会：
- 运行所有测试文件
- 显示每个测试的通过/失败状态和执行时间
- 生成详细的统计报告，包括：
  - 总体统计（通过率、失败数、错误数等）
  - 按测试文件分组的统计
  - 最慢的10个测试
  - 失败测试的详细错误信息
- 将报告保存到 `test_report.txt` 文件

### 运行所有测试（简单版本）
```bash
cd open-r1/test_examples/VisualQuality-R1/code/test_code
./run_all_tests.sh
```

### 运行单个测试
```bash
cd open-r1/test_examples/VisualQuality-R1/code/test_code
python3 test_fidelity_reward.py
python3 test_accuracy_reward.py
python3 test_format_reward.py
python3 test_generate_and_score_completions.py
python3 test_compute_loss.py
python3 test_repeat_random_sampler.py
```

### 运行特定测试用例
```bash
python3 test_fidelity_reward.py TestFidelityReward.test_fidelity_reward_basic
```

### 查看测试报告
运行 `run_tests_with_stats.py` 后，会生成 `test_report.txt` 文件，包含完整的测试结果和统计信息：
```bash
cat test_report.txt
```

## 测试设计原则

1. **直接导入测试**：测试代码直接 import 原始代码中的函数，而不是复制一份
2. **独立性**：每个测试用例相互独立，可以单独运行
3. **覆盖性**：测试覆盖正常情况、边缘情况和错误情况
4. **可验证性**：测试结果明确，易于判断通过或失败
5. **参考 PURE**：测试代码风格参考 verl/test_examples/PURE/code/test_code

## 注意事项

1. 测试代码假设源代码位于 `../src/open-r1-multimodal/src/` 目录
2. 某些测试使用模拟（mock）对象来避免复杂的依赖
3. 测试主要关注函数的逻辑正确性，而不是性能
4. 对于涉及随机性的测试，使用固定的随机种子确保可重复性

## 测试结果解释

- **PASS**：测试通过，函数实现符合要求
- **FAIL**：测试失败，函数实现存在问题

测试失败时，会显示具体的错误信息，包括：
- 期望值 vs 实际值
- 失败的断言
- 错误堆栈跟踪

## 与 LLM 代码生成的集成

这些测试代码的主要用途是验证 LLM 生成的代码：

1. 在训练数据构建时，我们会删除 03_algorithm_and_core_methods.md 中描述的函数实现
2. 让 LLM 学习 open-r1 框架和 VisualQuality-R1 的基础代码
3. 让 LLM 生成这些核心函数的实现
4. 使用这些测试代码验证 LLM 生成的代码是否正确

测试通过意味着 LLM 成功理解了需求并生成了正确的实现。
