# FlagScale 测试代码说明

## 概述

本目录包含 FlagScale 核心函数的单元测试代码，基于源代码实现编写，验证函数行为与源代码一致。

## 测试文件

### 1. test_model_provider.py
测试 `model_provider` 函数（flagscale/train/train_gpt.py:line 69-191）

**测试用例（7个）：**
1. 标准模型创建流程
2. Legacy 模型创建
3. YAML 配置加载
4. 从 parallel_context 获取配置
5. Transformer Engine 规格选择
6. ModelOpt 模型提供者
7. 模型参数传递完整性

### 2. test_loss_func.py
测试 `loss_func` 函数（flagscale/train/train_gpt.py:line 216-275）

**测试用例（7个）：**
1. 基本损失计算
2. view(-1) 操作
3. num_tokens 计算
4. reporting_loss 结构
5. NaN 检查启用
6. Spiky loss 检查
7. ModelOpt loss_func

### 3. test_forward_step.py
测试 `forward_step` 函数（flagscale/train/train_gpt.py:line 278-304）

**测试用例（6个）：**
1. 调用 get_batch
2. Core 模型 forward 调用
3. Legacy 模型 forward 调用
4. 返回值结构
5. partial loss_func 的使用
6. stimer 上下文管理器

## 环境要求

测试代码需要在包含必要依赖的 Python 环境中运行。

### 方法1：使用 modelopt 环境（推荐）

modelopt 环境已经包含了所需的依赖（megatron-core, transformer-engine等）：

```bash
cd /path/to/FlagScale/code/test_code

# 使用配置脚本（会自动激活 modelopt 环境并设置 PYTHONPATH）
source setup_test_env_modelopt.sh

# 运行测试
python test_model_provider.py
python test_loss_func.py
python test_forward_step.py
```

### 方法2：使用 exper 环境

如果要在 exper 环境中运行，需要安装额外的依赖：

```bash
conda activate exper

# 安装 transformer-engine（需要 CUDA 支持）
pip install transformer-engine[pytorch]

# 或者如果没有 CUDA，可以卸载 transformer-engine
# megatron-core 会在运行时跳过 transformer-engine 相关功能
pip uninstall transformer-engine transformer-engine_cu12

cd /path/to/FlagScale/code/test_code
source setup_test_env.sh
python test_model_provider.py
```

### 方法3：完整安装 FlagScale

```bash
cd /path/to/FlagScale/code

# 安装 FlagScale 和 Megatron-LM 后端
PYTHONPATH=./:$PYTHONPATH pip install . --no-build-isolation --verbose \
    --config-settings=device="gpu" \
    --config-settings=backend="Megatron-LM"

# 运行测试
cd test_code
python test_model_provider.py
```

## 测试设计原则

1. **基于源代码**：每个测试用例都基于源代码的实际实现逻辑编写
2. **验证行为一致性**：测试验证函数的行为与源代码描述一致
3. **Mock 外部依赖**：使用 unittest.mock 模拟外部依赖，专注测试目标函数
4. **清晰的输出**：每个测试通过时会打印成功信息，便于跟踪测试进度

## 测试输出示例

```
✓ 成功导入 model_provider 模块

======================================================================
model_provider 测试
======================================================================
✓ 测试 1 通过: 标准模型创建流程
✓ 测试 2 通过: Legacy 模型创建
✓ 测试 3 通过: YAML 配置加载
✓ 测试 4 通过: 从 parallel_context 获取配置
✓ 测试 5 通过: Transformer Engine 规格选择
✓ 测试 6 通过: ModelOpt 模型提供者
✓ 测试 7 通过: 模型参数传递完整性
.......
----------------------------------------------------------------------
Ran 7 tests in 0.123s

OK
```

## 注意事项

1. **导入失败处理**：如果导入失败，测试会跳过（skip），不会报错
2. **环境依赖**：测试需要完整的 FlagScale 环境，包括特定版本的 Megatron-LM
3. **GPU 不是必需的**：测试使用 Mock，不需要实际的 GPU 硬件
4. **Megatron-LM 版本**：FlagScale 使用的是定制版本的 Megatron-LM（位于 `third_party/Megatron-LM`），与 PyPI 上的 `megatron-core` 不兼容

## 当前环境限制

由于 FlagScale 依赖特定的 Megatron-LM 版本（需要通过完整安装获取），在没有完整安装 FlagScale 的情况下，测试代码会因为缺少 `megatron.training` 模块而无法导入。

### 解决方案

**推荐：完整安装 FlagScale**

```bash
cd /path/to/FlagScale/code

# 1. 初始化子模块（获取 Megatron-LM）
git submodule update --init --recursive

# 2. 安装 FlagScale
PYTHONPATH=./:$PYTHONPATH pip install . --no-build-isolation --verbose \
    --config-settings=device="gpu" \
    --config-settings=backend="Megatron-LM"

# 3. 运行测试
cd test_code
python test_model_provider.py
python test_loss_func.py
python test_forward_step.py
```

如果导入仍然失败，测试会自动跳过，不会影响测试执行。

## 文档参考

测试代码对应的函数文档：
- `requirements/03_algorithm_and_core_methods.md`

源代码位置：
- `flagscale/train/train_gpt.py`

