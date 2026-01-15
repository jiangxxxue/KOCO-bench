# VisualQuality-R1 系统架构与模块设计

## 1. 代码结构与主要组件

### 1.1 目录组织（VisualQuality-R1 核心）
```
code/src/open-r1-multimodal/
├── src/open_r1/
│   ├── grpo_jsonl.py               # 训练入口，数据加载与奖励函数定义
│   ├── trainer/
│   │   ├── grpo_trainer.py         # VLMGRPOTrainer 核心训练逻辑
│   │   ├── grpo_config.py          # GRPO 训练配置参数
│   │   └── vllm_grpo_trainer.py    # vLLM 加速版本（可选）
│   ├── vlm_modules/
│   │   ├── vlm_module.py           # VLM 基础模块接口
│   │   ├── qwen_module.py          # Qwen2.5-VL 模型适配
│   │   └── internvl_module.py      # InternVL 模型适配（可选）
│   ├── utils/
│   │   ├── evaluation.py           # 评估工具
│   │   ├── math.py                 # 数学计算工具
│   │   └── callbacks.py            # 训练回调
│   └── qwen2_5vl_monkey_patch.py   # Qwen2.5-VL Flash Attention 优化
├── run_scripts/
│   └── KADID-10K/
│       ├── one_node_run_kadid.sh   # 单节点训练脚本
│       └── multi_run_kadid.sh      # 多节点训练脚本
├── configs/
│   ├── qwen2vl_sft_config.yaml     # 模型配置
│   └── zero3.json                  # DeepSpeed ZeRO-3 配置
└── datasets/
    ├── make_data.py                # 数据集预处理脚本
    └── KADID-10K/                  # 示例数据集
```

### 1.2 角色与数据流
- 角色
  - Policy Model (π_θ)：基于 Qwen2.5-VL-7B-Instruct 的策略模型，使用 GRPO 更新；支持 LoRA 微调和视觉模块冻结。
  - Reference Model (π_ref)：参考模型，用于计算 KL 散度；当使用 LoRA 时，可通过禁用 adapter 获得。
  - VLM Module：多模态模型适配层，处理图像和文本输入，生成结构化响应。
  - Reward Functions：奖励函数集合，包括保真度奖励（accuracy_reward）和格式奖励（format_reward）。

- 关键数据
  - prompts：包含图像和文本提示的多模态输入。
  - responses：生成的结构化响应，包含 `<think>推理过程</think>` 和 `<answer>评分</answer>`。
  - predictions：从响应中提取的质量评分（1-5 之间的浮点数）。
  - fidelity_rewards：基于成对排序计算的保真度奖励。
  - format_rewards：检查响应格式是否符合要求的奖励。

## 2. 训练管线与配置要点

### 2.1 训练入口与模式切换
- 入口：`python -m open_r1.grpo_jsonl`（通过 torchrun 启动分布式训练）
- 关键配置（`run_scripts/KADID-10K/one_node_run_kadid.sh`）
  - 模型路径：`--model_name_or_path` 指定 Qwen2.5-VL-7B-Instruct 路径
  - 数据路径：`--data_file_paths` 和 `--image_folders` 指定训练数据
  - 生成数量：`--num_generations` 设置每个图像生成的响应数量（默认 6）
  - 视觉模块：`--freeze_vision_modules` 控制是否冻结视觉编码器
  - 问题模板：`--question_template scoring` 指定使用评分任务模板

### 2.2 奖励函数设计
- 保真度奖励（Fidelity Reward）
  - 位置：`grpo_jsonl.py::fidelity_reward` 和 `grpo_jsonl.py::accuracy_reward`
  - 计算流程：
    1. 提取每个响应的预测评分（从 `<answer>` 标签中解析）
    2. 计算每个样本的 G 个预测的均值 μ 和方差 σ²
    3. 对批次内每对样本 (i, k)，计算排序概率：p = Φ((pred_i - μ_k) / √(σ²_i + σ²_k))
    4. 根据真实 MOS 确定真实排序关系 gt（1.0 表示 i 优于 k，0.0 表示 k 优于 i，0.5 表示相等）
    5. 计算保真度：r = √(p·gt) + √((1-p)·(1-gt))
    6. 对所有其他样本求平均得到最终奖励

- 格式奖励（Format Reward）
  - 位置：`grpo_jsonl.py::format_reward`
  - 检查响应是否符合 `<think>...</think><answer>...</answer>` 格式
  - 符合格式返回 1.0，否则返回 0.0

### 2.3 GRPO 训练循环
- 配置项：`GRPOConfig`（`trainer/grpo_config.py`）
  - `num_generations`：每个提示生成的响应数量（必须能被全局批次大小整除）
  - `num_iterations`：每个批次的更新迭代次数（μ）
  - `beta`：KL 散度系数（控制与参考模型的偏离程度）
  - `epsilon` / `epsilon_high`：PPO 裁剪范围
  - `max_completion_length`：生成响应的最大长度

- 实现位置：`VLMGRPOTrainer`（`trainer/grpo_trainer.py`）
  - 生成与评分：`_generate_and_score_completions` 方法
    - 使用 vLLM 或 HuggingFace 生成器生成 G 个响应
    - 调用奖励函数计算每个响应的奖励
    - 计算分组优势（group-wise advantage）
  - 损失计算：`compute_loss` 方法
    - 计算策略比率 ratio = exp(log_π_θ - log_π_old)
    - 应用 PPO 裁剪：L_clip = min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)
    - 添加 KL 惩罚：L = -L_clip + β·KL
  - 采样器：`RepeatRandomSampler`
    - 确保每个 GPU 的批次可以来自不同数据集
    - 支持按图像前缀分组（用于 KADID-10K 等数据集）
    - 重复采样以支持多次迭代

### 2.4 数据预处理
- 数据格式：JSONL 格式，每行包含：
  - `id`：样本 ID
  - `dataset_name`：数据集名称
  - `image`：图像文件名（单张或多张）
  - `conversations`：对话格式的问题和答案
    - `from: "human"`：包含任务描述和问题
    - `from: "gpt"`：包含 MOS 评分（归一化到 1-5）

- 归一化：`make_data.py::normalize`
  - 将原始 MOS 分数归一化到 [1, 5] 区间
  - 公式：normalized_mos = (mos - min_mos) / (max_mos - min_mos) × 4 + 1

- 数据分割：按参考图像分割训练/验证/测试集（6:2:2）

### 2.5 多模态处理
- VLM Module 接口：`VLMBaseModule`（`vlm_modules/vlm_module.py`）
  - `prepare_prompt`：将数据转换为模型输入格式
  - `prepare_model_inputs`：处理图像和文本，生成模型输入张量
  - `get_vision_modules_keywords`：返回视觉模块的关键字（用于冻结或 LoRA）
  - `get_custom_multimodal_keywords`：返回多模态输入的关键字（如 `pixel_values`, `image_grid_thw`）

- Qwen2.5-VL 适配：`Qwen2VLModule`（`vlm_modules/qwen_module.py`）
  - 支持动态分辨率和多图像输入
  - 配置 `max_pixels` 和 `min_pixels` 控制图像分辨率
  - 使用 Flash Attention 2 加速训练

### 2.6 优化技术
- LoRA 微调：自动识别所有线性层（排除视觉模块）作为 target_modules
- 视觉模块冻结：通过 `freeze_vision_modules=True` 冻结视觉编码器，减少显存占用
- 梯度检查点：通过 `gradient_checkpointing=True` 启用，降低显存峰值
- DeepSpeed ZeRO-3：支持大模型训练，配置文件 `local_scripts/zero3.json`
- Flash Attention 2：通过 monkey patch 优化 Qwen2.5-VL 的注意力计算
