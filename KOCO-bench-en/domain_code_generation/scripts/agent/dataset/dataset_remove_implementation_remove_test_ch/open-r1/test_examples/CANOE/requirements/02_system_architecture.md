# CANOE 系统架构与模块设计

## 1. 代码结构与主要组件

### 1.1 目录组织（CANOE核心）
```
code/train/
├── src/open_r1/
│   ├── grpo.py                      # GRPO训练入口与Dual-GRPO实现
│   ├── configs.py                   # 训练配置类（GRPOConfig扩展）
│   ├── sft.py                       # 监督微调脚本
│   ├── generate.py                  # 数据生成脚本
│   └── utils/
│       ├── callbacks.py             # 训练回调函数
│       ├── evaluation.py            # 评估工具
│       └── hub.py                   # HuggingFace Hub集成
├── TRL/                             # 修改后的TRL库
│   └── trl/trainer/
│       ├── grpo_trainer.py          # GRPO Trainer核心实现（支持Dual-GRPO）
│       └── grpo_config.py           # GRPO配置
├── recipes/                         # 训练配置文件
│   ├── LLama/
│   │   └── LLama-Instruct/
│   │       └── llama3_8b_2epoch_10k.yaml
│   ├── Qwen/
│   │   └── Qwen2.5-Instruct/
│   │       ├── qwen_7b_2epoch_10k.yaml
│   │       └── qwen_14b_2epoch_10k.yaml
│   └── accelerate_configs/          # DeepSpeed/DDP配置
│       ├── zero2.yaml
│       └── zero3.yaml
├── train_data/                      # 训练数据目录
│   └── final_10k.jsonl             # 合成的训练数据
└── scripts/
    ├── generate_reasoning.py        # 推理生成脚本
    └── run_benchmarks.py            # 基准测试脚本

code/eval/                           # 评估代码
├── ConFiQA_FiQA/                   # 反事实和事实QA评估
├── CNQ/                            # 反事实多选题评估
├── FaithEval/                      # 反事实QA评估
├── FollowRAG/                      # RAG生成评估
├── CLAPNQ/                         # 长文本QA评估
└── XSum_WiKiLarge/                 # 摘要和简化评估
```

### 1.2 核心模块与数据流

#### 1.2.1 训练模块
- **GRPO Trainer**：基于TRL的GRPOTrainer，扩展支持Dual-GRPO
  - 第一阶段：生成短文本答案（包含<think>、<long_answer>、<answer>标签）
  - 第二阶段：提取<long_answer>替换context，重新生成以评估影响力
  - 支持vLLM加速推理（use_vllm=True）
  - 支持DeepSpeed ZeRO-2/ZeRO-3并行训练

- **奖励函数模块**：
  - `accuracy_reward`：验证答案正确性（符号验证 + 字符串匹配）
  - `format_reward`：检查输出格式（<think><long_answer><answer>结构）
  - `influence_reward`：评估长答案对最终答案的影响
  - `length_reward`：约束长答案长度在合理范围

#### 1.2.2 数据格式
- **输入数据**：
  ```json
  {
    "problem": "<context>上下文内容</context>\n\n问题描述",
    "solution": "<answer>正确答案</answer>"
  }
  ```

- **系统提示词**（SYSTEM_PROMPT）：
  ```
  A conversation between User and Assistant. The user gives an instruction 
  that consists of two parts: a passage and the actual instruction, 
  separated by two newline characters.
  
  The passage is provided within <context> and </context> tags. 
  The Assistant need to refer to the given passage and complete the instruction.
  
  The response must be structured and include the following three sections:
  - Reasoning Process: <think>推理过程</think>
  - Long Answer: <long_answer>详细答案</long_answer>
  - Short Answer: <answer>简洁答案</answer>
  ```

- **生成输出**：
  ```
  <think>推理步骤...</think>
  <long_answer>完整的、语法和语义完整的句子...</long_answer>
  <answer>简洁的直接答案</answer>
  ```

## 2. 训练管线与配置要点

### 2.1 训练入口与模式
- 入口：`accelerate launch src/open_r1/grpo.py --config recipes/xxx.yaml`
- 关键配置参数（`recipes/xxx.yaml`）：
  ```yaml
  # 模型配置
  model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
  torch_dtype: bfloat16
  attn_implementation: flash_attention_2
  
  # GRPO配置
  use_vllm: true                      # 使用vLLM加速
  vllm_gpu_memory_utilization: 0.7    # GPU内存利用率
  num_generations: 7                  # 每个prompt生成7个响应
  do_long_answer: true                # 启用Dual-GRPO（长答案生成）
  
  # 训练超参数
  learning_rate: 1.0e-06
  num_train_epochs: 2
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  max_prompt_length: 1024
  max_completion_length: 1024
  
  # 奖励函数
  reward_funcs: [accuracy, format, influence, length]
  ```

### 2.2 Dual-GRPO实现细节
- **第一阶段生成**（grpo.py）：
  - 使用标准GRPO生成n个短文本响应
  - 计算accuracy_reward和format_reward
  
- **第二阶段生成**（grpo_trainer.py中的扩展）：
  - 当`do_long_answer=True`时触发
  - 从第一阶段生成中提取`<long_answer>`内容
  - 使用正则表达式替换原始context：
    ```python
    pattern = re.compile(r'<context>.*?</context>', flags=re.DOTALL)
    new_context = pattern.sub(f'<context>{long_answer}</context>', original_context)
    ```
  - 基于新context重新生成（temperature相同，n=1）
  - 计算influence_reward和length_reward

- **奖励融合**：
  - 所有奖励函数返回0.0或1.0的二值奖励
  - 总奖励 = accuracy + format + influence + length（范围0-4）
  - GRPO使用总奖励计算组内相对优势



