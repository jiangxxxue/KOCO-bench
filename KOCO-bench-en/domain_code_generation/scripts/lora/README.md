# LoRA 微调脚本

基于原版全量微调代码 (`finetuning.py`) 的 LoRA 参数高效微调实现。

## 📋 文件说明

- `finetuning_lora.py` - LoRA 微调主程序
- `run_finetuning_lora.sh` - LoRA 微调运行脚本
- `inference_lora.py` - LoRA 推理示例脚本
- `README.md` - 本文档

## 🚀 快速开始

### 1. 准备数据集

可以复用父目录的数据集构建器：

```bash
cd ..
python finetune_dataset_builder.py \
  --source-dir /path/to/verl \
  --output-file ./data/verl/verl_training_dataset.jsonl
```

### 2. 配置训练参数

编辑 `run_finetuning_lora.sh` 中的参数：

```bash
# 基础路径
MODEL_PATH="/path/to/Qwen2.5-Coder-7B-Instruct"
DATA_PATH="../data/verl/verl_training_dataset.jsonl"
OUTPUT_DIR="../models/qwen2.5-coder-7b-verl-lora"

# LoRA 参数
LORA_R=16              # LoRA rank，建议 8-64
LORA_ALPHA=32          # 通常为 2*r
LORA_DROPOUT=0.05      # dropout 概率
TARGET_MODULES="q_proj,v_proj,k_proj,o_proj"  # 目标模块
```

### 3. 开始训练

```bash
cd lora
bash run_finetuning_lora.sh
```

## 🎯 LoRA 参数说明

### 核心参数

- **lora_r** (rank): LoRA 矩阵的秩
  - 更大的 r = 更强的表达能力，但参数量增加
  - 推荐值：8-64
  - 默认：16

- **lora_alpha**: LoRA 缩放参数
  - 控制 LoRA 层的影响力
  - 通常设置为 2*r
  - 默认：32

- **lora_dropout**: Dropout 概率
  - 防止过拟合
  - 推荐值：0.05-0.1
  - 默认：0.05

- **target_modules**: 应用 LoRA 的模块
  - Qwen2.5 推荐：`q_proj,v_proj,k_proj,o_proj`
  - 也可以添加：`gate_proj,up_proj,down_proj` (FFN 层)
  - 更多模块 = 更强表达能力，但参数量增加

### 高级参数

- **use_rslora**: Rank-Stabilized LoRA
  - 改进的 LoRA 变体，训练更稳定
  - 默认：false

- **use_dora**: DoRA (Weight-Decomposed Low-Rank Adaptation)
  - 将权重分解为幅度和方向两部分
  - 通常效果更好，但略慢
  - 默认：false

## 💡 训练参数调优建议

### 相比全量微调的差异

| 参数 | 全量微调 | LoRA 微调 | 说明 |
|------|---------|----------|------|
| 学习率 | 5e-6 | 1e-4 ~ 3e-4 | LoRA 可用更大学习率 |
| Batch Size | 1-2 | 4-8 | LoRA 显存占用少 |
| 训练轮数 | 2-3 | 3-5 | LoRA 收敛快 |
| 可训练参数 | 100% | ~0.1-1% | 大幅减少 |

### 不同任务的 LoRA 配置

**代码补全 / 续写**（当前场景）
```bash
LORA_R=16
LORA_ALPHA=32
LEARNING_RATE=1e-4
TARGET_MODULES="q_proj,v_proj,k_proj,o_proj"
```

**复杂代码生成**
```bash
LORA_R=32
LORA_ALPHA=64
LEARNING_RATE=1e-4
TARGET_MODULES="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"
```

**特定领域适配**
```bash
LORA_R=8
LORA_ALPHA=16
LEARNING_RATE=2e-4
TARGET_MODULES="q_proj,v_proj"
```

## 📊 训练监控

训练过程会输出：
- 可训练参数统计（通常 < 1%）
- 训练/验证损失
- 训练速度（samples/s, tokens/s）

输出示例：
```
trainable params: 8,388,608 || all params: 7,615,684,608 || trainable%: 0.11
```

## 🔍 推理使用

### 方式 1: 加载 LoRA Adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "/path/to/Qwen2.5-Coder-7B-Instruct",
    device_map="auto",
    torch_dtype="auto"
)

# 加载 LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "../models/qwen2.5-coder-7b-verl-lora"
)

tokenizer = AutoTokenizer.from_pretrained(
    "../models/qwen2.5-coder-7b-verl-lora"
)

# 推理
prompt = "def fibonacci(n):"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 方式 2: 合并后使用

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# 加载并合并
base_model = AutoModelForCausalLM.from_pretrained("...")
model = PeftModel.from_pretrained(base_model, "lora_output")
merged_model = model.merge_and_unload()

# 保存合并后的模型（可直接用 transformers 加载）
merged_model.save_pretrained("./merged_model")
```

## 📦 依赖安装

```bash
pip install transformers>=4.36.0
pip install peft>=0.7.0
pip install datasets
pip install accelerate
pip install bitsandbytes  # 可选，支持量化训练
```

## ⚙️ 显存占用参考

**Qwen2.5-Coder-7B 模型**

| 配置 | 显存占用 | 说明 |
|------|---------|------|
| 全量微调 (bf16) | ~28GB | 需要 A100 40GB |
| LoRA r=16 (bf16) | ~16GB | RTX 4090 24GB 可训 |
| LoRA r=8 (bf16) | ~14GB | RTX 3090 24GB 可训 |
| LoRA + 8bit 量化 | ~10GB | RTX 3080 Ti 可训 |

## 🔧 常见问题

### Q: 训练损失不下降？
A: 尝试增大学习率（1e-4 → 2e-4）或增加 lora_r

### Q: 过拟合？
A: 增加 lora_dropout 或减少训练轮数

### Q: 效果不如全量微调？
A: 增加 lora_r 和 target_modules，或使用 DoRA

### Q: 显存不足？
A: 减小 batch_size 或使用 8bit 量化（添加 `--load_in_8bit true`）

## 📚 相关资源

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [Qwen2.5 模型卡](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)

