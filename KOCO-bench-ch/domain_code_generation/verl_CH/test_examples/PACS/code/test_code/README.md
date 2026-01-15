# PACS 测试代码说明

本目录包含PACS项目核心函数的测试代码，用于验证LLM生成的代码是否符合要求。

## 测试文件

### 1. test_compute_reward.py
测试 `compute_reward` 函数 - 策略改进奖励计算

**测试内容：**
- 方法1：对数概率差的总和计算
- 方法2：归一化的对数概率总和计算
- beta参数的影响
- response_mask的应用
- 正负策略改进情况
- 边缘情况和批处理

### 2. test_compute_weight.py
测试 `compute_weight` 函数 - 样本权重计算

**测试内容：**
- question模式：平衡和不平衡样本的权重计算
- only_positive模式：只保留正样本
- only_negative模式：只保留负样本
- 多组样本的权重计算
- 边缘情况和批处理一致性

### 3. test_compute_pacs_loss.py
测试 `compute_pacs_loss` 函数 - PACS核心损失函数

**测试内容：**
- RLOO优势估计
- GRPO优势估计
- Naive优势估计
- 样本权重的应用
- 不同reward_method的影响
- response_mask的处理
- prompt验证
- 端到端工作流程

## 运行测试

### 运行所有测试
```bash
cd verl/test_examples/PACS/code/test_code
bash run_pacs_tests.sh
```

### 运行单个测试文件
```bash
python test_compute_reward.py -v
python test_compute_weight.py -v
python test_compute_pacs_loss.py -v
```

### 运行特定测试用例
```bash
python test_compute_reward.py TestComputeReward.test_compute_reward_method_1_basic -v
```

## 测试设计原则

1. **直接导入原函数**：测试代码直接从 `pacs.pacs_core_algos` 导入函数，而不是复制代码
2. **全面覆盖**：测试覆盖基本功能、边缘情况、参数变化、批处理等多个方面
3. **精确验证**：使用手动计算的期望值验证函数输出的正确性
4. **独立性**：每个测试用例相互独立，可以单独运行

## 测试用途

这些测试代码用于：
1. 验证LLM生成的函数实现是否正确
2. 确保函数在各种输入情况下都能正常工作
3. 检查函数的输出是否符合文档描述的行为

## 注意事项

- 测试代码假设原函数已经正确实现在 `src/pacs/pacs_core_algos.py` 中
- 在评测LLM生成的代码时，需要先将LLM生成的函数替换到原文件中
- 测试通过表示LLM生成的代码符合要求
