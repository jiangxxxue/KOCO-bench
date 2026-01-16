# VLM-R1 系统架构与模块设计

## 1. 代码结构与主要组件

### 1.1 目录组织（VLM-R1核心）
```
code/
├── src/
│   ├── open-r1-multimodal/
│   │   └── src/open_r1/
│   │       ├── grpo_jsonl.py           # GRPO训练主入口（支持JSONL数据格式）
│   │       ├── grpo_rec.py             # REC任务专用训练脚本
│   │       ├── grpo.py                 # 通用GRPO训练脚本
│   │       ├── configs.py              # 训练配置类（GRPOConfig、SFTConfig）
│   │       ├── sft.py                  # 监督微调脚本
│   │       ├── generate.py             # 推理生成脚本
│   │       ├── evaluate.py             # 评估工具
│   │       ├── qwen2_5vl_monkey_patch.py  # Qwen2.5-VL模型补丁
│   │       ├── trainer/
│   │       │   ├── grpo_trainer.py     # VLM GRPO Trainer实现
│   │       │   └── grpo_config.py      # GRPO配置
│   │       ├── vlm_modules/            # 视觉-语言模型模块
│   │       │   ├── vlm_module.py       # VLM基类（抽象接口）
│   │       │   ├── qwen_module.py      # Qwen2-VL/Qwen2.5-VL模块
│   │       │   ├── internvl_module.py  # InternVL模块
│   │       │   └── glm_module.py       # GLM-V模块
│   │       └── utils/
│   │           ├── callbacks.py        # 训练回调函数
│   │           ├── evaluation.py       # 评估工具
│   │           ├── hub.py              # HuggingFace Hub集成
│   │           └── pycocotools/        # COCO评估工具
│   └── eval/                           # 评估脚本
│       ├── test_rec_r1.py              # REC任务评估（R1模型）
│       ├── test_rec_baseline.py        # REC任务评估（SFT基线）
│       ├── test_rec_r1_internvl.py     # REC任务评估（InternVL）
│       └── test_od_r1.py               # OVD任务评估
├── run_scripts/                        # 训练脚本
│   ├── run_grpo_rec.sh                 # REC任务全参数微调
│   ├── run_grpo_rec_lora.sh            # REC任务LoRA微调
│   ├── run_grpo_rec_internvl.sh        # InternVL REC训练
│   ├── run_grpo_rec_more_params.sh     # 更多参数配置示例
│   ├── run_grpo_gui.sh                 # GUI缺陷检测（多图像输入）
│   ├── run_grpo_gui_defect_detection.sh # GUI缺陷检测专用
│   ├── multinode_training_demo.sh      # 多节点训练示例
│   └── multinode_training_args.yaml    # 多节点训练配置
├── ascend_inference/                   # 华为昇腾推理适配
│   ├── 910B/                           # Atlas 800T A2适配
│   │   ├── vllm_ascend/                # vLLM-Ascend框架
│   │   └── xllm/                       # xLLM框架
│   └── 300IDuo/                        # Atlas 300I Duo适配
├── assets/                             # 文档资源
│   └── add_new_model.md                # 添加新模型指南
└── README.md                           # 项目文档
```

### 1.2 核心模块与数据流

#### 1.2.1 VLM模块抽象层
VLM-R1 通过 `VLMBaseModule` 抽象基类实现对不同视觉-语言模型的统一接口：

- **模型初始化**：
  - `get_model_class()`：返回模型类（如 Qwen2VLForConditionalGeneration）
  - `get_processing_class()`：返回处理器类（如 AutoProcessor）
  - `post_model_init()`：模型初始化后的自定义处理

- **模型配置**：
  - `get_vision_modules_keywords()`：返回视觉模块关键字（用于冻结）
  - `get_custom_multimodal_keywords()`：返回多模态特定参数
  - `get_custom_processing_keywords()`：返回处理器配置参数

- **输入处理**：
  - `prepare_prompt()`：准备提示文本
  - `prepare_model_inputs()`：处理多模态输入（图像+文本）

- **任务特定方法**：
  - `get_question_template()`：根据任务类型返回问题模板
  - `select_reward_func()`：根据任务类型选择奖励函数

#### 1.2.2 训练模块
- **VLMGRPOTrainer**：扩展自TRL的GRPOTrainer，专门针对视觉-语言模型优化
  - 支持多模态输入处理（图像路径延迟加载）
  - 支持视觉模块冻结（freeze_vision_modules）
  - 支持LoRA微调（通过peft_config）
  - 支持vLLM加速推理（use_vllm=True）
  - 支持DeepSpeed ZeRO-2/ZeRO-3并行训练

- **奖励函数模块**：
  - `accuracy_reward`：根据任务类型选择不同的准确性验证方法
    - `default_accuracy_reward`：符号验证 + 模糊匹配
    - `mcq_reward`：多选题选项提取与匹配
    - `yes_no_reward`：是非题匹配
    - `llm_reward`：使用LLM评估答案相似度
    - `map_reward`：计算目标检测mAP
    - `od_reward`：目标检测准确性（mAP或mAP@50）
    - `odLength_reward`：带长度惩罚的目标检测奖励
    - `math_reward`：数学问题符号验证
    - `numeric_reward`：数值精确匹配
    - `all_match_reward`：文本精确匹配
  - `format_reward`：检查输出格式（`<think><answer>`结构）
  - `cosine_rewards`：基于余弦函数的长度奖励
  - `repetition_rewards`：重复内容惩罚

#### 1.2.3 数据格式
- **输入数据（JSONL格式）**：
  ```json
  {
    "id": 1,
    "image": "path/to/image.jpg",  // 或 ["path1.jpg", "path2.jpg"] 用于多图像
    "conversations": [
      {"from": "human", "value": "<image>问题描述"},
      {"from": "gpt", "value": "标准答案"}
    ],
    "accu_reward_method": "default"  // 可选，指定准确性奖励方法
  }
  ```

- **系统提示词**（默认）：
  ```
  A conversation between User and Assistant. The user asks a question, and the 
  Assistant solves it. The assistant first thinks about the reasoning process in 
  the mind and then provides the user with the answer. The reasoning process and 
  answer are enclosed within <think> </think> and <answer> </answer> tags, 
  respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>
  ```

- **任务特定提示词模板**：
  - REC任务：`"{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."`
  - OVD任务：包含详细的推理过程要求
  - 其他任务：根据需求自定义

- **生成输出**：
  ```
  <think>推理步骤和分析过程...</think>
  <answer>最终答案（可能是文本、数字、JSON格式的边界框等）</answer>
  ```

## 2. 训练管线与配置要点

### 2.1 训练入口与模式
- 入口：`torchrun --nproc_per_node=X src/open_r1/grpo_jsonl.py --config ...`
- 关键配置参数：
  ```yaml
  # 模型配置
  model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
  torch_dtype: bfloat16
  attn_implementation: flash_attention_2
  freeze_vision_modules: false  # 是否冻结视觉模块
  
  # 数据配置
  data_file_paths: "path1.jsonl:path2.jsonl"  # 多个数据文件用":"分隔
  image_folders: "path1/:path2/"              # 对应的图像文件夹
  reward_method: "default:mcq"                # 每个数据集的奖励方法
  task_type: "rec"                            # 任务类型
  is_reward_customized_from_vlm_module: false # 是否使用VLM模块自定义奖励
  
  # GRPO配置
  use_vllm: true                              # 使用vLLM加速
  vllm_gpu_memory_utilization: 0.7            # GPU内存利用率
  num_generations: 7                          # 每个prompt生成7个响应
  
  # 训练超参数
  learning_rate: 1.0e-06
  num_train_epochs: 2
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  max_prompt_length: 1024
  max_completion_length: 1024
  
  # 奖励函数
  reward_funcs: [accuracy, format, length, repetition]
  
  # 视觉处理参数（Qwen2-VL）
  max_pixels: 12845056
  min_pixels: 3136
  
  # 视觉处理参数（InternVL）
  max_anyres_num: 12
  
  # DeepSpeed配置
  deepspeed: path/to/zero3.json
  ```

### 2.2 多任务奖励机制

#### 2.2.1 REC任务（指代表达理解）
- **准确性奖励**：计算预测边界框与真实边界框的IoU
  - 提取模型输出中的边界框坐标 `[x1, y1, x2, y2]`
  - 根据图像实际尺寸调整边界框坐标（resize操作）
  - 计算IoU值作为奖励（范围0-1）
- **格式奖励**：检查输出是否包含 `<think>...<answer>...[x1, y1, x2, y2]...</answer>` 格式

#### 2.2.2 OVD任务（开放词汇目标检测）
- **准确性奖励**：计算mAP或mAP@50
  - 解析JSON格式的边界框列表
  - 使用COCO评估工具计算mAP
  - 支持带长度惩罚的变体（odLength_reward）
- **加权求和奖励**（weighted_sum）：
  - 位置准确性（alpha=0.7）：平均IoU
  - 标签准确性（beta=0.0）：类别匹配率
  - 完整性（gamma=0.3）：考虑漏检和误检
  - 最终得分 = (alpha × 位置得分 + beta × 标签得分 + gamma × 完整性得分) / (alpha + beta + gamma)

#### 2.2.3 Math任务（多模态数学推理）
- **准确性奖励**：符号验证
  - 使用 `math_verify` 库进行符号解析和验证
  - 支持数值精确匹配作为后备方案

#### 2.2.4 通用奖励
- **长度奖励**（cosine_reward）：
  - 基于余弦函数的长度控制
  - 正确答案：长度越短奖励越高（1.0 → 0.5）
  - 错误答案：长度越短惩罚越大（0.0 → -0.5）
  - 公式：`reward = max_value - (max_value - min_value) × (1 - cos(len × π / max_len)) / 2`

- **重复惩罚**（repetition_reward）：
  - 使用n-gram统计检测重复内容
  - 对JSON数据：n-gram_size=1（检测重复的边界框）
  - 对文本数据：n-gram_size=6（检测重复的短语）
  - 惩罚公式：`reward = (1 - unique_ngrams / total_ngrams) × max_penalty`
  - max_penalty = -1.0

### 2.3 模块化扩展机制

#### 2.3.1 添加新模型
1. 在 `vlm_modules/` 下创建新模块类，继承 `VLMBaseModule`
2. 实现所有抽象方法（模型加载、输入处理等）
3. 在 `get_vlm_module()` 函数中注册新模型
4. 可选：实现任务特定的奖励函数

#### 2.3.2 添加新任务
1. 在 `get_question_template()` 中添加任务特定的提示模板
2. 实现任务特定的奖励函数
3. 在 `reward_funcs_registry` 中注册奖励函数
4. 在训练脚本中指定 `task_type` 和 `reward_method`

### 2.4 高效训练策略

#### 2.4.1 视觉模块冻结
- 设置 `freeze_vision_modules=true`
- 仅训练语言模型部分，大幅减少显存占用和训练时间
- 适用于视觉特征已经足够好的场景

#### 2.4.2 LoRA微调
- 通过 `peft_config` 配置LoRA参数
- 仅训练少量适配器参数
- 显著降低显存需求，支持更大的batch size

#### 2.4.3 多节点训练
- 使用 `torchrun` 启动多节点训练
- 配置 `--nnodes`, `--node_rank`, `--master_addr`, `--master_port`
- 支持跨节点的梯度同步和数据并行

#### 2.4.4 vLLM加速
- 设置 `use_vllm=true`
- 使用vLLM引擎加速推理生成阶段
- 显著提升生成吞吐量，缩短训练时间
