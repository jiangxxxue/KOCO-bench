# LoRA 推理服务器使用指南

## 📋 概述

LoRA 推理服务器支持加载**基础模型 + LoRA adapter**，提供代码生成服务。

### ⚠️ 重要说明

**LoRA 模型加载方式：**
- LoRA 训练后生成的是 **adapter 权重**，不是完整模型
- 推理时需要：**基础模型** + **LoRA adapter** 一起加载
- 使用 PEFT 库实现在线加载

**两种使用方案：**
1. **方案 A（推荐）**: 使用 LoRA 服务器 - 在线加载 base model + adapter
2. **方案 B**: 先合并权重 - 使用 `merge_lora.py` 合并后当普通模型用

---

## 🚀 方案 A: 使用 LoRA 推理服务器

### 步骤 1: 启动 LoRA 推理服务器

```bash
# 使用默认配置启动（需要设置环境变量）
BASE_MODEL_PATH=/home/user/models/Qwen2.5-Coder-7B-Instruct \
LORA_ADAPTER_PATH=../models/qwen2.5-coder-7b-verl-lora \
bash scripts/lora/start_inference_server_lora.sh

# 或者指定端口
BASE_MODEL_PATH=/home/user/models/Qwen2.5-Coder-7B-Instruct \
LORA_ADAPTER_PATH=../models/qwen2.5-coder-7b-verl-lora \
SERVER_PORT=8001 \
bash scripts/lora/start_inference_server_lora.sh
```

**启动参数（环境变量）**：
- `BASE_MODEL_PATH`: **基础模型路径**（必须，例如 Qwen2.5-Coder-7B-Instruct）
- `LORA_ADAPTER_PATH`: **LoRA adapter 路径**（必须，训练输出的目录）
- `SERVER_PORT`: 服务器端口（默认: `8001`，避免和 SFT 服务器冲突）
- `SERVER_HOST`: 服务器地址（默认: `0.0.0.0`）
- `MAX_CONTEXT_LEN`: 最大上下文长度（默认: `4096`）
- `TORCH_DTYPE`: 数据类型（默认: `bfloat16`）
- `CUDA_VISIBLE_DEVICES`: GPU 设备（默认: `0`）

**注意事项**：
- 首次启动需要加载基础模型和 adapter，大约需要 2-3 分钟
- 确保安装了 `peft` 库: `pip install peft`
- 日志保存在 `../logs/inference_server_lora.log`
- PID 保存在 `../logs/inference_server_lora.pid`

### 步骤 2: 检查服务器状态

```bash
# 方法 1: 使用 curl 检查健康状态
curl http://localhost:8001/health

# 方法 2: 查看日志
tail -f ../logs/inference_server_lora.log

# 方法 3: 检查进程
cat ../logs/inference_server_lora.pid
ps aux | grep inference_server_lora
```

### 步骤 3: 批量生成代码

```bash
# 使用 LoRA 服务器批量生成
FRAMEWORK=verl \
MODEL_NAME=qwen2.5-coder-7b-verl-lora \
bash scripts/lora/run_batch_code_generation_with_lora_server.sh

# 生成多个补全
FRAMEWORK=verl \
MODEL_NAME=qwen2.5-coder-7b-verl-lora \
NUM_COMPLETIONS=4 \
TEMPERATURE=0.8 \
bash scripts/lora/run_batch_code_generation_with_lora_server.sh
```

### 步骤 4: 停止服务器

```bash
# 正常停止服务器
bash scripts/lora/stop_inference_server_lora.sh

# 或者直接 kill 进程
kill $(cat ../logs/inference_server_lora.pid)
```

---

## 🔀 方案 B: 合并权重后使用

如果不想使用服务器架构，可以先合并 LoRA 权重到基础模型：

```bash
# 合并 LoRA adapter 到基础模型
python merge_lora.py \
    --base_model /path/to/base/model \
    --lora_adapter ../models/qwen2.5-coder-7b-verl-lora \
    --output_dir ../models/qwen2.5-coder-7b-verl-merged

# 然后像普通模型一样使用
# 可以用 inference 目录下的服务器或 apicall 方式
```

---

## 📖 详细使用示例

### 示例 1: 完整的 LoRA 训练和推理流程

```bash
# 1. 训练 LoRA adapter
bash scripts/lora/run_finetuning_lora.sh

# 2. 启动 LoRA 推理服务器
BASE_MODEL_PATH=/home/user/models/Qwen2.5-Coder-7B-Instruct \
LORA_ADAPTER_PATH=../models/qwen2.5-coder-7b-verl-lora \
bash scripts/lora/start_inference_server_lora.sh

# 3. 等待服务器启动（查看日志）
tail -f ../logs/inference_server_lora.log

# 4. 批量生成代码（在另一个终端）
FRAMEWORK=verl \
MODEL_NAME=qwen2.5-coder-7b-verl-lora \
bash scripts/lora/run_batch_code_generation_with_lora_server.sh

# 5. 停止服务器
bash scripts/lora/stop_inference_server_lora.sh
```

### 示例 2: 手动调用客户端

```bash
# 使用客户端脚本直接调用
python inference_client_lora.py \
    --server_url http://localhost:8001 \
    --input_file ../data/verl/algorithm_methods_data_ARES.jsonl \
    --model_name qwen2.5-coder-7b-verl-lora \
    --num_completions 1 \
    --max_tokens 2048
```

---

## 🔧 API 调用示例

### 健康检查

```bash
curl http://localhost:8001/health
```

返回示例：
```json
{
  "status": "healthy",
  "model": "/path/to/base/model + ../models/qwen2.5-coder-7b-verl-lora",
  "base_model": "/path/to/base/model",
  "lora_adapter": "../models/qwen2.5-coder-7b-verl-lora",
  "device": "cuda:0"
}
```

### 代码生成

```bash
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["def fibonacci(n):\n    "],
    "num_completions": 1,
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.95
  }'
```

---

## ❓ 常见问题

### Q1: LoRA 和 SFT 有什么区别？

- **SFT (Supervised Fine-Tuning)**: 全参数微调，生成完整模型
- **LoRA**: 参数高效微调，只训练少量 adapter 权重
  - 优点：显存占用少，训练快，适合多任务
  - 缺点：推理时需要额外加载 adapter

### Q2: LoRA adapter 在哪里？

LoRA 训练后，adapter 权重保存在输出目录中：
```
../models/qwen2.5-coder-7b-verl-lora/
├── adapter_config.json
├── adapter_model.safetensors  # LoRA 权重
└── ...
```

### Q3: 为什么要用服务器架构？

- 基础模型加载一次，多次复用
- 避免每次推理都重新加载模型
- 大幅提升批量生成效率

### Q4: LoRA 服务器和普通服务器能同时运行吗？

可以！两者使用不同的端口：
- SFT/普通服务器: 端口 `8000`
- LoRA 服务器: 端口 `8001`（默认）

### Q5: 如何选择使用哪种方案？

**使用 LoRA 服务器（方案 A）**适合：
- 需要快速切换不同的 adapter
- 多个 LoRA 模型复用同一个基础模型
- 想节省磁盘空间（不需要多个完整模型）

**合并权重（方案 B）**适合：
- 只有一个 LoRA 模型长期使用
- 不想依赖 PEFT 库
- 希望推理速度更快（合并后少一次权重加载）

---

## 🆘 故障排查

### 问题 1: 服务器启动失败

```bash
# 查看日志
tail -50 ../logs/inference_server_lora.log

# 常见原因：
# 1. 基础模型路径错误
# 2. LoRA adapter 路径错误
# 3. 缺少 peft 库: pip install peft
# 4. 显存不足
```

### 问题 2: 生成失败

```bash
# 检查服务器状态
curl http://localhost:8001/health

# 查看客户端日志
cat /tmp/gen_lora_*.log

# 重启服务器
bash scripts/lora/stop_inference_server_lora.sh
bash scripts/lora/start_inference_server_lora.sh
```

---

## 📚 相关文档

- [LoRA 原始论文](https://arxiv.org/abs/2106.09685)
- [PEFT 库文档](https://github.com/huggingface/peft)
- [Qwen2.5 模型文档](https://github.com/QwenLM/Qwen2.5)

