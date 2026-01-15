# ByteMLPerf 训练性能测试系统架构

## 1. 模块文件结构

### 1.1 文件组织
```
byte_train_perf/
└── Megatron-LM/                  # 基于Megatron-LM的训练框架
    ├── megatron/                 # 核心训练框架
    │   ├── inference/            # 推理相关模块
    │   │   ├── algos/           # 算法实现
    │   │   │   └── distillation.py  # 知识蒸馏算法
    │   │   ├── arguments.py     # 参数配置
    │   │   ├── checkpointing.py  # 检查点管理
    │   │   └── gpt/             # GPT模型相关
    │   │       └── model_provider.py  # 模型提供者
    │   └── training/            # 训练相关模块
    │       └── checkpointing.py  # 训练检查点
    └── examples/                # 示例和工具
        └── export/              # 模型导出
            └── ptq_and_trtllm_export/  # PTQ和TensorRT-LLM导出
                ├── text_generation_ptq.py  # PTQ量化实现
                └── README.md     # 使用说明
```

### 1.2 模块职责划分
- `megatron/inference/algos/distillation.py`: 知识蒸馏算法实现
- `megatron/inference/checkpointing.py`: 推理检查点管理
- `megatron/training/checkpointing.py`: 训练检查点管理
- `examples/export/ptq_and_trtllm_export/`: 模型量化和导出工具

## 2. 核心模块设计

### 2.1 知识蒸馏模块 (distillation.py)

核心类: `LogitsAndIntermediatesLossBalancer`

主要功能:
- 动态损失平衡: 自动调整知识蒸馏过程中的损失权重
- 配置管理: 支持logits蒸馏和中间层蒸馏的组合配置
- MCore适配: 处理Megatron-Core架构特有的配置需求

核心方法:
- `forward()`: 动态平衡蒸馏损失和原始损失
- `load_distillation_config()`: 加载知识蒸馏配置文件
- `adjust_distillation_model_for_mcore()`: 调整蒸馏模型以适配MCore架构

### 2.2 模型导出模块 (ptq_and_trtllm_export/)

核心功能: 实现后训练量化和TensorRT-LLM导出

主要组件:
- `text_generation_ptq.py`: PTQ量化实现
- 支持INT8、FP8、INT4等多种量化策略
- TensorRT-LLM格式导出

关键特性:
- 多种量化配置支持
- 自定义校准数据支持
- 硬件特定优化

### 2.3 检查点管理模块

核心功能: 管理训练和推理过程中的模型状态保存与加载

主要组件:
- `megatron/inference/checkpointing.py`: 推理检查点管理
- `megatron/training/checkpointing.py`: 训练检查点管理

关键特性:
- 分布式检查点支持
- ModelOpt状态保存
- 断点续训支持

### 2.4 参数配置模块

核心功能: 管理系统参数和模型配置

主要组件:
- `megatron/inference/arguments.py`: 推理参数配置
- `megatron/inference/gpt/model_provider.py`: GPT模型配置

关键特性:
- 命令行参数解析
- 模型配置管理
- 硬件配置适配

## 3. 模块交互与数据流

### 3.1 知识蒸馏工作流
1. 配置加载: 加载蒸馏配置文件，设置教师/学生模型参数
2. 模型初始化: 创建蒸馏模型，配置损失函数和优化器
3. 蒸馏训练: 执行知识蒸馏训练，动态调整损失权重
4. 模型导出: 保存优化后的模型，支持后续量化

### 3.2 量化导出工作流
1. 模型加载: 加载预训练模型，准备校准数据集
2. 量化执行: 执行后训练量化，生成量化模型
3. TensorRT-LLM导出: 转换为TensorRT-LLM格式，优化推理性能

### 3.3 检查点管理流程
1. 状态保存: 保存模型状态、优化器状态、训练进度
2. 分布式同步: 在分布式环境中同步检查点数据
3. 状态加载: 从检查点恢复训练状态
4. 恢复训练: 继续训练或进行推理
