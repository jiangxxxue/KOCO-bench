# 推理服务器使用说明

## 📋 概述

这是一个基于服务器-客户端架构的代码生成系统，解决了原来每次都要重新加载模型的问题。

### 优势
- ✅ **模型只加载一次**：服务器启动后一直运行，不需要重复加载模型
- ✅ **提高效率**：批量生成时大幅提升速度
- ✅ **资源复用**：多个任务可以共享同一个模型服务
- ✅ **便于管理**：独立的启动/停止脚本，易于控制

### 系统组成

1. **inference_server.py** - 推理服务器（后台运行，加载模型并提供 API）
2. **inference_client.py** - 推理客户端（请求服务器生成代码）
3. **start_inference_server.sh** - 启动服务器脚本
4. **stop_inference_server.sh** - 停止服务器脚本
5. **run_batch_code_generation_with_server.sh** - 新的批量生成脚本（使用服务器）

---

## 🚀 快速开始

### 步骤 1: 启动推理服务器

```bash
# 使用默认配置启动
bash scripts/inference/start_inference_server.sh

# 或者指定模型路径
MODEL_PATH=../models/your-model bash scripts/inference/start_inference_server.sh

# 或者指定端口
SERVER_PORT=8001 MODEL_PATH=../models/your-model bash scripts/inference/start_inference_server.sh
```

**启动参数（环境变量）**：
- `MODEL_PATH`: 模型路径（默认: `../models/qwen2.5-coder-7b-modelopt-sft`）
- `SERVER_PORT`: 服务器端口（默认: `8000`）
- `SERVER_HOST`: 服务器地址（默认: `0.0.0.0`）
- `MAX_CONTEXT_LEN`: 最大上下文长度（默认: `4096`）
- `CUDA_VISIBLE_DEVICES`: GPU 设备（默认: `0,1,2,3`）

**注意事项**：
- 首次启动需要加载模型，大约需要 1-2 分钟
- 启动后会自动进行健康检查
- 日志保存在 `../logs/inference_server.log`
- PID 保存在 `../logs/inference_server.pid`

### 步骤 2: 检查服务器状态

```bash
# 方法 1: 使用 curl 检查健康状态
curl http://localhost:8000/health

# 方法 2: 查看日志
tail -f ../logs/inference_server.log

# 方法 3: 检查进程
cat ../logs/inference_server.pid
ps aux | grep inference_server
```

### 步骤 3: 批量生成代码

```bash
# 使用默认配置
bash scripts/run_batch_code_generation_with_server.sh

# 或者指定框架和参数
FRAMEWORK=verl \
NUM_COMPLETIONS=4 \
TEMPERATURE=0.2 \
bash scripts/run_batch_code_generation_with_server.sh
```

**生成参数（环境变量）**：
- `FRAMEWORK`: 框架名称（默认: `verl`）
- `MODEL_NAME`: 模型名称（默认: `qwen2.5-coder-7b-verl-ntp`）
- `SERVER_URL`: 服务器地址（默认: `http://localhost:8000`）
- `NUM_COMPLETIONS`: 每个样本生成数量（默认: `1`）
- `MAX_TOKENS`: 最大生成 tokens（默认: `2048`）
- `TEMPERATURE`: 采样温度（默认: `0.7`）
- `TOP_P`: Top-p 采样（默认: `0.95`）
- `BATCH_SIZE`: 批处理大小（默认: `1`）
- `SKIP_EXISTING`: 是否跳过已存在的文件（默认: `false`）

### 步骤 4: 停止服务器

```bash
# 正常停止服务器
bash scripts/inference/stop_inference_server.sh

# 或者直接 kill 进程
kill $(cat ../logs/inference_server.pid)
```

---

## 📖 详细使用示例

### 示例 1: 基本使用流程

```bash
# 1. 启动服务器
bash scripts/start_inference_server.sh

# 等待提示 "推理服务器准备就绪！"

# 2. 运行批量生成
bash scripts/run_batch_code_generation_with_server.sh

# 3. 查看结果
ls -lh ../data/verl/qwen2.5-coder-7b-verl-ntp/*_output.jsonl

# 4. 完成后停止服务器
bash scripts/stop_inference_server.sh
```

### 示例 2: 使用自定义模型

```bash
# 1. 启动服务器（使用自定义模型）
MODEL_PATH=../models/qwen2.5-coder-7b-modelopt-sft \
bash scripts/start_inference_server.sh

# 2. 运行批量生成（匹配模型名称）
MODEL_NAME=qwen2.5-coder-7b-modelopt-sft \
bash scripts/run_batch_code_generation_with_server.sh
```

### 示例 3: 生成多个补全用于评估

```bash
# 1. 确保服务器已启动
curl http://localhost:8000/health

# 2. 生成每个样本 4 个补全，使用较低温度
NUM_COMPLETIONS=4 \
TEMPERATURE=0.2 \
MAX_TOKENS=2048 \
bash scripts/run_batch_code_generation_with_server.sh
```

### 示例 4: 使用批处理加速

```bash
# 每次请求处理 4 个样本（如果数据较多可以加速）
BATCH_SIZE=4 \
bash scripts/run_batch_code_generation_with_server.sh
```

### 示例 5: 单独使用客户端

```bash
# 对单个文件生成代码
python scripts/inference_client.py \
    --server_url http://localhost:8000 \
    --input_file ./../data/verl/algorithm_methods_data_prime.jsonl \
    --output_file ./output/prime_output.jsonl \
    --num_completions 2 \
    --temperature 0.2
```

---

## 🔧 高级用法

### 使用不同的端口

```bash
# 启动服务器在 8001 端口
SERVER_PORT=8001 bash scripts/start_inference_server.sh

# 客户端连接到 8001 端口
SERVER_URL=http://localhost:8001 \
bash scripts/run_batch_code_generation_with_server.sh
```

### 多个服务器同时运行

```bash
# 服务器 1: 端口 8000，使用 GPU 0,1
CUDA_VISIBLE_DEVICES=0,1 \
SERVER_PORT=8000 \
MODEL_PATH=../models/model1 \
bash scripts/start_inference_server.sh

# 服务器 2: 端口 8001，使用 GPU 2,3
CUDA_VISIBLE_DEVICES=2,3 \
SERVER_PORT=8001 \
MODEL_PATH=../models/model2 \
bash scripts/start_inference_server.sh

# 使用不同的服务器生成
SERVER_URL=http://localhost:8000 bash scripts/run_batch_code_generation_with_server.sh
SERVER_URL=http://localhost:8001 bash scripts/run_batch_code_generation_with_server.sh
```

### 远程服务器

```bash
# 在服务器 A 启动推理服务
SERVER_HOST=0.0.0.0 SERVER_PORT=8000 bash scripts/start_inference_server.sh

# 在服务器 B 请求生成
SERVER_URL=http://server-a-ip:8000 \
bash scripts/run_batch_code_generation_with_server.sh
```

---

## 🐛 故障排查

### 问题 1: 服务器启动失败

**检查日志**：
```bash
tail -50 ./logs/inference_server.log
```

**常见原因**：
- GPU 内存不足：减少 GPU 数量或使用更小的模型
- 模型路径错误：检查 `MODEL_PATH` 是否正确
- 端口被占用：更换端口或停止占用端口的进程

### 问题 2: 客户端连接失败

**检查服务器状态**：
```bash
# 检查服务器是否运行
curl http://localhost:8000/health

# 检查进程
ps aux | grep inference_server
```

**常见原因**：
- 服务器未启动：先运行 `start_inference_server.sh`
- 端口不匹配：确保 `SERVER_URL` 和 `SERVER_PORT` 一致
- 防火墙阻止：检查防火墙设置

### 问题 3: 生成速度慢

**优化建议**：
- 增加 `BATCH_SIZE`（例如设置为 4 或 8）
- 使用多个 GPU（通过 `CUDA_VISIBLE_DEVICES`）
- 减少 `MAX_TOKENS`
- 降低 `NUM_COMPLETIONS`

### 问题 4: 服务器内存溢出

**解决方案**：
- 减少 GPU 数量
- 降低 `MAX_CONTEXT_LEN`
- 使用量化模型
- 减小 `BATCH_SIZE`

---

## 📊 性能对比

### 原方案 vs 新方案

| 项目 | 原方案 | 新方案（服务器） | 提升 |
|------|--------|-----------------|------|
| 模型加载次数 | N 次（每个文件） | 1 次 | N 倍 |
| 总耗时（10 个文件） | ~30 分钟 | ~5 分钟 | 6 倍 |
| GPU 利用率 | 低（加载时间长） | 高 | - |
| 资源复用 | 否 | 是 | - |

---

## 🔄 与原脚本对比

### 原脚本（run_batch_code_generation_improved.sh）
- ❌ 每个文件都重新加载模型
- ❌ 效率低下
- ✅ 使用简单（单个脚本）

### 新脚本（run_batch_code_generation_with_server.sh）
- ✅ 模型只加载一次
- ✅ 效率高
- ✅ 支持多任务共享
- ⚠️ 需要先启动服务器

**建议**：
- 单次生成少量文件：可以使用原脚本
- 批量生成多个文件：**强烈推荐使用新脚本**
- 频繁生成代码：**强烈推荐使用新脚本**

---

## 📝 完整工作流示例

```bash
# ========================================
# 完整的代码生成和评估流程
# ========================================

# 1. 启动推理服务器
echo "启动推理服务器..."
MODEL_PATH=../models/qwen2.5-coder-7b-modelopt-sft \
bash scripts/start_inference_server.sh

# 2. 等待服务器就绪（检查健康状态）
echo "检查服务器状态..."
curl http://localhost:8000/health

# 3. 批量生成代码（生成 4 个补全用于评估）
echo "批量生成代码..."
FRAMEWORK=verl \
MODEL_NAME=qwen2.5-coder-7b-modelopt-sft \
NUM_COMPLETIONS=4 \
TEMPERATURE=0.2 \
MAX_TOKENS=2048 \
bash scripts/run_batch_code_generation_with_server.sh

# 4. 运行执行评估
echo "运行执行评估..."
bash scripts/run_batch_execution_evaluation_pure.sh

# 5. 完成后停止服务器
echo "停止服务器..."
bash scripts/stop_inference_server.sh

echo "完成！"
```

---

## 🆘 需要帮助？

如果遇到问题：

1. **查看日志**：`tail -f ./logs/inference_server.log`
2. **检查服务器状态**：`curl http://localhost:8000/health`
3. **查看进程**：`ps aux | grep inference_server`
4. **清理重启**：
   ```bash
   bash scripts/stop_inference_server.sh
   bash scripts/start_inference_server.sh
   ```

---

## 📚 API 文档

服务器启动后，可以访问交互式 API 文档：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 主要 API 端点

#### GET /health
健康检查

```bash
curl http://localhost:8000/health
```

响应：
```json
{
  "status": "healthy",
  "model": "qwen2.5-coder-7b-modelopt-sft",
  "device": "cuda:0"
}
```

#### POST /generate
生成代码补全

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["def hello():\n    "],
    "num_completions": 2,
    "max_tokens": 512,
    "temperature": 0.2,
    "top_p": 0.95
  }'
```

响应：
```json
{
  "completions": [
    ["print('Hello, World!')", "return 'Hello'"]
  ],
  "model": "qwen2.5-coder-7b-modelopt-sft",
  "status": "success"
}
```

