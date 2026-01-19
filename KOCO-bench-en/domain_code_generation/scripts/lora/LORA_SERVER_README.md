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

## Client Usage

```bash
python inference_client_lora.py \
    --server_url http://localhost:8001 \
    --input_file ../data/your_framework/algorithm_methods_data_example.jsonl \
    --model_name your_framework-lora \
    --num_completions 1 \
    --max_tokens 2048
```

## API Calls

```bash
curl http://localhost:8001/health
```

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

## Troubleshooting

```bash
tail -50 ../logs/inference_server_lora.log
```

```bash
cat /tmp/gen_lora_*.log
```

```bash
bash scripts/lora/stop_inference_server_lora.sh
bash scripts/lora/start_inference_server_lora.sh
```
