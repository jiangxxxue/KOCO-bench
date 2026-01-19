# LoRA Inference Server Commands

## Start Server

```bash
BASE_MODEL_PATH=/path/to/base/model \
LORA_ADAPTER_PATH=../models/your_framework-lora \
bash scripts/lora/start_inference_server_lora.sh
```

```bash
BASE_MODEL_PATH=/path/to/base/model \
LORA_ADAPTER_PATH=../models/your_framework-lora \
SERVER_PORT=8001 \
bash scripts/lora/start_inference_server_lora.sh
```

## Check Server Status

```bash
curl http://localhost:8001/health
```

```bash
tail -f ../logs/inference_server_lora.log
```

```bash
cat ../logs/inference_server_lora.pid
ps aux | grep inference_server_lora
```

## Batch Code Generation

```bash
FRAMEWORK=your_framework \
MODEL_NAME=your_framework-lora \
bash scripts/lora/run_batch_code_generation_with_lora_server.sh
```

```bash
FRAMEWORK=your_framework \
MODEL_NAME=your_framework-lora \
NUM_COMPLETIONS=4 \
TEMPERATURE=0.8 \
bash scripts/lora/run_batch_code_generation_with_lora_server.sh
```

## Stop Server

```bash
bash scripts/lora/stop_inference_server_lora.sh
```

```bash
kill $(cat ../logs/inference_server_lora.pid)
```