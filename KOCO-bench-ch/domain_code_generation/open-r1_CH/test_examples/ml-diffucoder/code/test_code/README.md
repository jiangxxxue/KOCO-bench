# ml-diffucoder 测试代码

本目录包含针对 ml-diffucoder 项目核心算法函数的测试代码。这些测试用于验证 LLM 生成的代码实现是否符合要求。

## 测试文件说明

### 1. test_forward_process.py
测试 `DiffuGRPOTrainer.forward_process` 函数
- **功能**: 生成三个版本的掩码序列，用于 Coupled Sampling 方案
- **测试内容**:
  - 基本输出结构验证
  - Version 1: 掩码所有完成token
  - Version 2: 按mask_ratio随机掩码
  - Version 3: 反向掩码（与Version 2互补）
  - 权重计算正确性
  - 随机种子可重复性
  - 边缘情况处理

### 2. test_selective_log_softmax.py
测试 `selective_log_softmax` 函数
- **功能**: 高效计算加权对数概率，避免内存溢出
- **测试内容**:
  - 基本输出形状验证
  - 不使用权重时的行为
  - 加权平均的正确性
  - 多次迭代处理
  - mask效果验证
  - 内存效率测试
  - 数值稳定性

### 3. test_get_per_token_logps.py
测试 `DiffuGRPOTrainer._get_per_token_logps` 函数
- **功能**: 计算token级对数概率，使用Coupled Sampling方案
- **测试内容**:
  - 输入验证（维度、mask_seeds长度）
  - 输出形状正确性
  - prompt_index创建
  - 多次迭代处理
  - 批次拼接
  - 完成部分提取
  - 权重传播
  - 输出维度排列
  - 数据类型转换

### 4. test_generate_and_score_completions.py
测试 `DiffuGRPOTrainer._generate_and_score_completions` 函数
- **功能**: 生成代码完成并计算奖励
- **测试内容**:
  - 基本输出结构
  - 提示处理
  - EOS token后的掩码
  - mask seeds生成（随机/固定）
  - 优势计算（leave-one-out baseline）
  - 多个奖励函数的加权
  - 指标记录

### 5. test_code_reward.py
测试 `code_reward` 函数
- **功能**: 评估生成的代码并计算测试用例通过率
- **测试内容**:
  - 基本输出结构
  - 格式检查
  - 代码提取
  - 测试用例执行
  - 部分通过情况
  - 不同执行提供者类型
  - 并行执行
  - 评估脚本模板
  - 边缘情况

### 6. test_code_format_reward.py
测试 `get_code_format_reward` 函数
- **功能**: 检查代码格式和语法
- **测试内容**:
  - 格式正确的代码（返回1.0）
  - 缺少代码块标记（返回0.0）
  - 语法错误（返回0.5）
  - 额外文本处理
  - 多个代码块
  - 空代码块
  - 复杂代码
  - 各种语法错误
  - 不同编程语言支持

## 运行测试

### 运行所有测试
```bash
cd open-r1/test_examples/ml-diffucoder/code/test_code
./run_all_tests.sh
```

### 运行单个测试文件
```bash
python3 test_forward_process.py
python3 test_selective_log_softmax.py
python3 test_get_per_token_logps.py
python3 test_generate_and_score_completions.py
python3 test_code_reward.py
python3 test_code_format_reward.py
```

### 运行特定测试类或方法
```bash
# 运行特定测试类
python3 test_forward_process.py TestForwardProcess

# 运行特定测试方法
python3 test_forward_process.py TestForwardProcess.test_forward_process_basic_output_structure
```

## 测试设计原则

1. **直接导入测试**: 测试代码直接从源代码中导入函数，而不是复制实现
2. **全面覆盖**: 包括正常情况、边缘情况和错误情况
3. **独立性**: 每个测试独立运行，不依赖其他测试
4. **可重复性**: 使用固定的随机种子确保结果可重复
5. **清晰的断言**: 每个断言都有明确的错误消息

## 测试数据

测试使用模拟数据和mock对象，不依赖外部资源：
- 使用 `torch.manual_seed()` 确保可重复性
- 使用 `unittest.mock` 模拟复杂依赖
- 使用小规模数据确保测试快速运行

## 预期结果

所有测试应该通过。如果测试失败，说明：
1. LLM生成的代码实现有误
2. 代码不符合规范要求
3. 存在边缘情况未处理

## 注意事项

1. 测试代码假设源代码位于 `../src/open_r1/` 目录
2. 需要安装必要的依赖：`torch`, `transformers`, `trl` 等
3. 某些测试使用mock对象模拟复杂的依赖，实际运行时可能需要调整
4. 测试主要关注函数的输入输出行为，而不是内部实现细节

## 扩展测试

如果需要添加新的测试：
1. 在相应的测试文件中添加新的测试方法
2. 遵循现有的命名约定：`test_<function_name>_<test_aspect>`
3. 添加清晰的文档字符串说明测试目的
4. 更新 `run_all_tests.sh` 如果添加了新的测试文件
