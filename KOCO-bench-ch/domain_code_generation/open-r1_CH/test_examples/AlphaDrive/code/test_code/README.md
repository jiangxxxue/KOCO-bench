# AlphaDrive 测试代码

本目录包含 AlphaDrive 项目核心函数的测试代码，用于验证 LLM 生成的代码是否符合要求。

## 测试文件说明

### 1. test_plan_speed_reward.py
测试速度规划奖励函数 `plan_speed_reward`，验证：
- F1 分数计算（precision, recall）
- 复杂度因子计算
- 多样性因子计算
- 奖励融合逻辑

### 2. test_plan_path_reward.py
测试路径规划奖励函数 `plan_path_reward`，验证：
- F1 分数计算（precision, recall）
- 复杂度因子计算
- 多样性因子计算
- 奖励融合逻辑

### 3. test_plan_format_reward.py
测试格式验证奖励函数 `plan_format_reward`，验证：
- 正确格式识别：`<think>...</think><answer>...</answer>`
- 错误格式检测
- 边缘情况处理

### 4. test_get_per_token_logps.py
测试逐 token 对数概率计算函数 `_get_per_token_logps`，验证：
- logits 和 input_ids 的对齐
- log_softmax 计算
- gather 操作的正确性
- 内存高效的逐行处理

### 5. test_compute_loss.py
测试 GRPO 损失计算函数 `compute_loss`，验证：
- 多模态输入处理
- prompt 和 completion 分离
- EOS 掩码创建
- KL 散度计算
- 奖励聚合
- GRPO 优势估计
- 策略梯度损失计算
- 掩码平均损失

## 运行测试

### 运行所有测试
```bash
cd test_code
./run_tests.sh
```

### 运行单个测试
```bash
python test_plan_speed_reward.py
python test_plan_path_reward.py
python test_plan_format_reward.py
python test_get_per_token_logps.py
python test_compute_loss.py
```

### 运行特定测试用例
```bash
python test_plan_speed_reward.py TestPlanSpeedReward.test_plan_speed_reward_perfect_match
```

## 测试设计原则

1. **直接导入原函数**：测试代码直接 import 原代码中的函数，而不是复制一份
2. **标准答案验证**：基于文档中的算法描述，构造准确的测试用例
3. **覆盖多种场景**：
   - 基本功能测试
   - 边缘情况测试
   - 集成测试
   - 错误处理测试

## 测试用例结构

每个测试文件包含：
- **基础测试类**：测试核心功能
- **集成测试类**：测试与其他组件的集成
- **边缘情况测试类**（如适用）：测试特殊情况

## 注意事项

- 测试代码需要能够成功 import 原代码中的函数
- 测试用例基于 `03_algorithm_and_core_methods.md` 中的函数描述编写
- 测试的目的是验证 LLM 生成的代码是否正确实现了算法逻辑
