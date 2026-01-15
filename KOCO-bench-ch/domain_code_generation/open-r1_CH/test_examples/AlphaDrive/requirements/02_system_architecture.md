# AlphaDrive 系统架构与模块设计

## 1. 代码结构与主要组件

### 1.1 目录组织
```
AlphaDrive/
├── src/
│   ├── r1-v/                           # 基于 open-r1 的训练框架
│   │   ├── src/open_r1/
│   │   │   ├── grpo.py                 # GRPO 训练入口
│   │   │   ├── sft.py                  # 监督微调入口
│   │   │   └── trainer/
│   │   │       └── grpo_trainer.py     # GRPO 训练器核心实现
│   │   └── configs/                    # 训练配置文件
│   ├── distill_r1/                     # R1 推理数据生成工具
│   │   ├── query_r1.py                 # 查询 R1 生成推理轨迹
│   │   ├── filter_r1.py                # 过滤有效推理轨迹
│   │   ├── prompt.py                   # R1 系统提示词
│   │   └── create_hf_dataset.py        # 创建 HuggingFace 数据集
│   ├── eval/                           # 评估工具
│   └── scripts/                        # 训练脚本
├── data_tools/
│   ├── prompt_hub.py                   # 提示词模板库
│   └── example.json                    # 数据样例
├── eval_tools/
│   ├── qwen2vl_plan_cmd_eval_grpo.py   # GRPO 模型评估脚本
│   └── qwen2vl_plan_cmd_eval_sft.py    # SFT 模型评估脚本
└── train_tools/
    ├── run_train_grpo.sh               # GRPO 训练启动脚本
    └── run_train_sft.sh                # SFT 训练启动脚本
```

### 1.2 核心模块
- 训练模块
  - `Qwen2VLGRPOTrainer`：扩展自 TRL 的 GRPOTrainer，支持视觉-语言模型的多模态输入
  - 支持 FSDP、DeepSpeed ZeRO-2/ZeRO-3 等分布式训练策略
  - 集成 vLLM 后端加速生成过程
  
- 奖励模块
  - `plan_format_reward`：格式验证奖励函数
  - `plan_speed_reward`：速度规划奖励函数
  - `plan_path_reward`：路径规划奖励函数
  
- 数据处理模块
  - 多模态数据加载：处理图像 + 文本的组合输入
  - 提示词构建：根据场景信息动态生成提示
  - 响应解析：提取 `<think>` 和 `<answer>` 标签内容

## 2. 训练管线与配置要点

### 2.1 训练入口与模式
- SFT 入口：`src/r1-v/src/open_r1/sft.py`
  - 在蒸馏的推理数据上进行监督微调
  - 学习生成结构化的推理过程和决策
  
- GRPO 入口：`src/r1-v/src/open_r1/grpo.py`
  - 通过强化学习优化决策质量
  - 支持多奖励函数融合

### 2.2 多模态输入处理
AlphaDrive 处理三类输入信息：
1. 视觉输入：驾驶场景图像
   - 使用 Qwen2-VL 的图像处理器
   - 支持动态分辨率（通过 `max_pixels` 和 `min_pixels` 控制）
   
2. 文本输入：结构化提示
   - 当前速度：`{speed}m/s`
   - 导航指令：如 "turn left", "go straight" 等
   - 任务描述：要求模型输出推理过程和决策
   
3. 输出格式：
   ```
   <think>
   [推理过程：分析场景、考虑约束、推导决策]
   </think>
   <answer>
   [速度决策], [路径决策]
   </answer>
   ```

### 2.3 奖励函数设计

#### 2.3.1 格式奖励（Format Reward）
- 功能：验证输出是否符合预期格式
- 实现：使用正则表达式匹配 `<think>.*?</think>\s*<answer>.*?</answer>` 模式
- 奖励值：符合格式返回 1.0，否则返回 0.0

#### 2.3.2 速度规划奖励（Speed Reward）
- 决策空间：{KEEP, ACCELERATE, DECELERATE, STOP}
- 复杂度权重：
  - ACCELERATE: 0.9
  - DECELERATE: 1.0
  - STOP: 1.0
  - KEEP: 0.8
- 计算公式：
  ```
  R_speed = F1_score × complexity_factor + diversity_factor
  
  其中：
  F1_score = 2 × (precision × recall) / (precision + recall)
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  complexity_factor = Σ(weight_i) / |predicted_actions|
  diversity_factor = ±0.4（根据批次内是否首次出现）
  ```

#### 2.3.3 路径规划奖励（Path Reward）
- 决策空间：{STRAIGHT, LEFT_TURN, RIGHT_TURN, LEFT_CHANGE, RIGHT_CHANGE}
- 复杂度权重：
  - LEFT_TURN: 1.0
  - RIGHT_TURN: 1.0
  - LEFT_CHANGE: 1.0
  - RIGHT_CHANGE: 1.0
  - STRAIGHT: 0.8
- 计算公式：与速度奖励相同的结构

### 2.4 GRPO 训练流程

#### 2.4.1 生成阶段
1. 构建多模态输入：
   - 使用 `AutoProcessor` 处理图像和文本
   - 设置 `padding_side="left"` 以支持批量生成
   
2. 生成多个响应：
   - 每个 prompt 生成 G 个响应（默认 G=2）
   - 使用采样策略（temperature=1.0）增加多样性

#### 2.4.2 奖励计算阶段
1. 解析生成的响应，提取决策内容
2. 并行计算三个奖励函数
3. 求和得到总奖励：`R_total = R_format + R_speed + R_path`

#### 2.4.3 优势估计阶段
1. 按 prompt 分组计算统计量：
   ```
   mean_R = mean(R_total[group])
   std_R = std(R_total[group])
   ```
   
2. 标准化优势：
   ```
   advantages = (R_total - mean_R) / (std_R + 1e-4)
   ```

#### 2.4.4 策略更新阶段
1. 计算对数概率：
   - 策略模型：`logp_θ = log P_θ(response | prompt)`
   - 参考模型：`logp_ref = log P_ref(response | prompt)`
   
2. 计算 KL 散度：
   ```
   KL = exp(logp_ref - logp_θ) - (logp_ref - logp_θ) - 1
   ```
   
3. 计算损失：
   ```
   loss = -mean(exp(logp_θ - logp_θ.detach()) × advantages - β × KL)
   ```
   
4. 梯度更新：使用 Adam 优化器更新策略模型参数

### 2.5 关键配置参数
- 模型配置：
  - `model_name_or_path`: 基础 VLM 模型（如 Qwen2-VL-2B-Instruct）
  - `attn_implementation`: flash_attention_2（加速注意力计算）
  - `max_pixels`: 401408（图像最大像素数）
  
- 训练配置：
  - `per_device_train_batch_size`: 1（每设备批量大小）
  - `gradient_accumulation_steps`: 2（梯度累积步数）
  - `num_generations`: 2（每个 prompt 生成的响应数）
  - `max_prompt_length`: 1024（最大提示长度）
  - `max_completion_length`: 自动（最大生成长度）
  
- 奖励配置：
  - `reward_funcs`: ["plan_speed_reward", "plan_path_reward", "plan_format_reward"]
  - `beta`: KL 惩罚系数（控制与参考模型的偏离程度）

## 3. 评估系统

### 3.1 评估指标
1. 整体规划准确率：
   ```
   Accuracy = (速度正确 AND 路径正确的样本数) / 总样本数
   ```

2. 分类 F1 分数：
   - 针对每个决策类别（9 个类别）计算 F1 分数
   - 综合评估精确率和召回率

3. 细粒度准确率：
   - 统计每种速度-路径组合的准确率
   - 共 4×5=20 种组合

### 3.2 评估流程
1. 加载训练好的模型和评估数据集
2. 对每个样本：
   - 构建多模态输入
   - 生成决策（采样 2 个响应，取第一个）
   - 解析 `<answer>` 标签内容
   - 与 ground truth 比较
3. 统计各项指标并保存结果

## 4. 数据生成流程

### 4.1 推理轨迹生成
1. 构建场景描述：结合图像、速度、导航信息
2. 查询 DeepSeek-R1：
   - 使用 R1 系统提示词
   - 生成包含 `<think>` 推理过程的响应
3. 解析和存储生成的轨迹

### 4.2 数据过滤
1. 验证答案正确性：
   - 提取 `<answer>` 标签内容
   - 与 ground truth 比较
2. 过滤格式错误或答案错误的样本
3. 保留高质量的推理轨迹

### 4.3 数据集构建
1. 整理过滤后的数据
2. 转换为 HuggingFace 数据集格式
3. 上传到 Hub 供训练使用
