# Inference Server Commands

## Start Server

```bash
bash scripts/inference/start_inference_server.sh
```

```bash
MODEL_PATH=../models/your_model bash scripts/inference/start_inference_server.sh
```

```bash
SERVER_PORT=8001 MODEL_PATH=../models/your_model bash scripts/inference/start_inference_server.sh
```

## Check Server Status

```bash
curl http://localhost:8000/health
```

```bash
tail -f ../logs/inference_server.log
```

```bash
cat ../logs/inference_server.pid
ps aux | grep inference_server
```

## Batch Code Generation

```bash
bash scripts/inference/run_batch_code_generation_with_server.sh
```

```bash
FRAMEWORK=your_framework \
NUM_COMPLETIONS=4 \
TEMPERATURE=0.2 \
bash scripts/inference/run_batch_code_generation_with_server.sh
```

## Stop Server

```bash
bash scripts/inference/stop_inference_server.sh
```

```bash
kill $(cat ../logs/inference_server.pid)
```

## Client Usage

```bash
python scripts/inference/inference_client.py \
    --server_url http://localhost:8000 \
    --input_file ../data/your_framework/algorithm_methods_data_example.jsonl \
    --output_file ./output/example_output.jsonl \
    --num_completions 2 \
    --temperature 0.2
```

## API Calls

```bash
curl http://localhost:8000/health
```

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
